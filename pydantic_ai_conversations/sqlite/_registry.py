"""`ConversationRegistry` on SQLite: the port implementation, in SQL.

Everything a SQL backend owes the protocol lives here -- global `conversation_id` primary key
with an indexed `namespace_key` column, `INSERT ... ON CONFLICT DO NOTHING` + re-select for the
atomic id claim, `SET message_count = message_count + ?` for the stats increment, and purge as a
permanent in-table tombstone (`purged = 1`, turns deleted, digest columns cleared) so the id stays
claimed against every namespace. Timestamps are Unix-epoch REALs: ISO strings order wrong across
microsecond precision.

Connection handling, thread dispatch, and the `sqlite3.Error` -> `RegistryUnavailable` mapping are
`_driver.py`'s; each method here hands the driver one unit of work that sees one connection.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic_ai_conversations.errors import ConversationNotFound
from pydantic_ai_conversations.registry import ConversationMeta, RunPhase, TurnError, TurnRecord
from pydantic_ai_conversations.sqlite._driver import CursorRoutedConnection, RegistryDriver

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    namespace_key TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_message_at REAL,
    message_count INTEGER NOT NULL DEFAULT 0,
    title TEXT,
    deleted_at REAL,
    attrs TEXT NOT NULL DEFAULT '{}',
    purged INTEGER NOT NULL DEFAULT 0
);
-- `list`'s exact shape: leftmost namespace_key, then its sort key, over live rows only.
-- SQLite walks it in order and stops at LIMIT instead of sorting the namespace.
CREATE INDEX IF NOT EXISTS ix_conversations_activity ON conversations(
    namespace_key, COALESCE(last_message_at, created_at) DESC, conversation_id DESC
) WHERE deleted_at IS NULL AND purged = 0;
-- Superseded by the index above, which carries namespace_key as its leftmost column.
DROP INDEX IF EXISTS ix_conversations_namespace;
CREATE TABLE IF NOT EXISTS turns (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    author_user_id TEXT NOT NULL,
    at REAL NOT NULL,
    phase TEXT NOT NULL DEFAULT 'started',
    previous_run_id TEXT,
    error_code TEXT,
    error_message TEXT,
    error_transient INTEGER
);
CREATE INDEX IF NOT EXISTS ix_turns_conversation ON turns(conversation_id);
"""


def _migrate(connection: sqlite3.Connection) -> None:
    """Add the columns a file created by an earlier version lacks: `CREATE TABLE IF NOT EXISTS`
    leaves an existing table as it is, and SQLite has no `ADD COLUMN IF NOT EXISTS`."""
    columns = {row[1] for row in connection.execute('PRAGMA table_info(turns)')}
    for column, kind in (
        ('previous_run_id', 'TEXT'),
        ('error_code', 'TEXT'),
        ('error_message', 'TEXT'),
        ('error_transient', 'INTEGER'),
    ):
        if column not in columns:
            connection.execute(f'ALTER TABLE turns ADD COLUMN {column} {kind}')


_LIVE = 'deleted_at IS NULL AND purged = 0'
_SORT_AT = 'COALESCE(last_message_at, created_at)'
"""The activity timestamp `list` orders by; `ix_conversations_activity` indexes this expression."""
_META_COLUMNS = (
    'conversation_id, namespace_key, created_by, created_at, last_message_at, message_count, title, deleted_at, attrs'
)


def _attrs_predicate(attrs: Mapping[str, str] | None) -> tuple[str, list[str]]:
    """`list`'s attrs filter as SQL: one correlated EXISTS per pair, ANDed.

    `json_each` takes the key and the value as bound parameters, so no JSON path is built from
    caller input -- a key holding a dot or a quote filters correctly instead of reshaping the
    query. An absent or empty mapping filters nothing, as the protocol requires.
    """
    if not attrs:
        return '', []
    clause = ''.join(
        ' AND EXISTS (SELECT 1 FROM json_each(conversations.attrs) WHERE key = ? AND value = ?)' for _ in attrs
    )
    return clause, [item for pair in attrs.items() for item in pair]


def _now() -> float:
    return datetime.now(UTC).timestamp()


def _to_datetime(value: float | None) -> datetime | None:
    return None if value is None else datetime.fromtimestamp(value, tz=UTC)


def _meta(row: sqlite3.Row) -> ConversationMeta:
    return ConversationMeta(
        namespace_key=row['namespace_key'],
        conversation_id=row['conversation_id'],
        created_by=row['created_by'],
        created_at=datetime.fromtimestamp(row['created_at'], tz=UTC),
        last_message_at=_to_datetime(row['last_message_at']),
        message_count=row['message_count'],
        title=row['title'],
        deleted_at=_to_datetime(row['deleted_at']),
        attrs=json.loads(row['attrs']),
    )


def _require_live(connection: CursorRoutedConnection, namespace_key: str, conversation_id: str) -> None:
    row = connection.execute(
        f'SELECT 1 FROM conversations WHERE conversation_id = ? AND namespace_key = ? AND {_LIVE}',
        (conversation_id, namespace_key),
    ).fetchone()
    if row is None:
        raise ConversationNotFound(conversation_id)


class SqliteConversationRegistry:
    """`ConversationRegistry` on a SQLite file.

    Methods lazy-connect, so `open()` is optional; after `close()` every method raises
    `RegistryUnavailable` until `open()` reconnects. SQL runs on worker threads over one autocommit
    connection in WAL mode (see `_driver.py`), so the event loop never blocks on the file and
    `SqliteConversationStepStore` or the harness's own `SqliteStepStore` may share it: one file
    carries a deployment's runs and snapshots alongside its conversations and turns.

    `list` filters, pages, and limits in the query, over a partial index that already carries the
    activity order (`ix_conversations_activity`), so a page costs what the page holds rather than
    what the namespace holds.
    """

    def __init__(self, database: str | Path) -> None:
        self._driver = RegistryDriver(database, schema=_SCHEMA, migrate=_migrate)

    async def open(self) -> None:
        """Connect and create the schema, so a bad path fails at startup rather than on the first
        request; clears a previous `close()`."""
        await self._driver.open()

        def work(connection: CursorRoutedConnection) -> None:
            connection.execute('SELECT 1')

        await self._driver.run(work)

    async def close(self) -> None:
        await self._driver.close()

    async def ping(self) -> Mapping[str, Any]:
        def work(connection: CursorRoutedConnection) -> Mapping[str, Any]:
            # A round trip that also proves the table is there; no count, the report is public.
            connection.execute('SELECT 1 FROM conversations LIMIT 1').fetchone()
            return {'ok': True}

        return await self._driver.run(work)

    async def register_conversation(
        self, namespace_key: str, conversation_id: str, *, created_by: str, attrs: Mapping[str, str] | None = None
    ) -> ConversationMeta:
        created_at, attrs_json = _now(), json.dumps(dict(attrs or {}))

        def work(connection: CursorRoutedConnection) -> ConversationMeta:
            connection.execute(
                'INSERT INTO conversations (conversation_id, namespace_key, created_by, created_at, attrs) '
                'VALUES (?, ?, ?, ?, ?) ON CONFLICT (conversation_id) DO NOTHING',
                (conversation_id, namespace_key, created_by, created_at, attrs_json),
            )
            # Purge never deletes the row (it tombstones it), so the claim above guarantees one exists.
            row = connection.execute(
                f'SELECT {_META_COLUMNS}, purged FROM conversations WHERE conversation_id = ?', (conversation_id,)
            ).fetchone()
            if row['purged'] or row['namespace_key'] != namespace_key or row['deleted_at'] is not None:
                raise ConversationNotFound(conversation_id)
            return _meta(row)

        return await self._driver.run(work)

    async def get(self, namespace_key: str, conversation_id: str) -> ConversationMeta | None:
        def work(connection: CursorRoutedConnection) -> ConversationMeta | None:
            row = connection.execute(
                f'SELECT {_META_COLUMNS} FROM conversations '
                f'WHERE conversation_id = ? AND namespace_key = ? AND {_LIVE}',
                (conversation_id, namespace_key),
            ).fetchone()
            return None if row is None else _meta(row)

        return await self._driver.run(work)

    async def list(
        self,
        namespace_key: str,
        *,
        attrs: Mapping[str, str] | None = None,
        page_after: str | None = None,
        limit: int = 100,
    ) -> Sequence[ConversationMeta]:
        if limit <= 0:
            raise ValueError(f'limit must be positive, got {limit}')

        attrs_sql, attrs_params = _attrs_predicate(attrs)

        def work(connection: CursorRoutedConnection) -> Sequence[ConversationMeta]:
            keyset_sql, keyset_params = '', []
            if page_after is not None:
                # The cursor resolves inside the same filtered set, so one that is unknown,
                # soft-deleted, or excluded by `attrs` ends the list rather than raising.
                cursor = connection.execute(
                    f'SELECT {_SORT_AT} AS sort_at FROM conversations '
                    f'WHERE conversation_id = ? AND namespace_key = ? AND {_LIVE}{attrs_sql}',
                    (page_after, namespace_key, *attrs_params),
                ).fetchone()
                if cursor is None:
                    return []
                keyset_sql = f' AND ({_SORT_AT}, conversation_id) < (?, ?)'
                keyset_params = [cursor['sort_at'], page_after]
            rows = connection.execute(
                f'SELECT {_META_COLUMNS} FROM conversations WHERE namespace_key = ? AND {_LIVE}'
                f'{attrs_sql}{keyset_sql} ORDER BY {_SORT_AT} DESC, conversation_id DESC LIMIT ?',
                (namespace_key, *attrs_params, *keyset_params, limit),
            ).fetchall()
            return [_meta(row) for row in rows]

        return await self._driver.run(work)

    async def set_title(self, namespace_key: str, conversation_id: str, title: str) -> None:
        def work(connection: CursorRoutedConnection) -> None:
            cursor = connection.execute(
                f'UPDATE conversations SET title = ? WHERE conversation_id = ? AND namespace_key = ? AND {_LIVE}',
                (title, conversation_id, namespace_key),
            )
            if cursor.rowcount == 0:
                raise ConversationNotFound(conversation_id)

        await self._driver.run(work)

    async def update_message_stats(
        self, namespace_key: str, conversation_id: str, *, last_message_at: datetime, message_count_delta: int
    ) -> None:
        def work(connection: CursorRoutedConnection) -> None:
            cursor = connection.execute(
                'UPDATE conversations SET last_message_at = ?, message_count = message_count + ? '
                f'WHERE conversation_id = ? AND namespace_key = ? AND {_LIVE}',
                (last_message_at.timestamp(), message_count_delta, conversation_id, namespace_key),
            )
            if cursor.rowcount == 0:
                raise ConversationNotFound(conversation_id)

        await self._driver.run(work)

    async def soft_delete(self, namespace_key: str, conversation_id: str) -> ConversationMeta:
        deleted_at = _now()

        def work(connection: CursorRoutedConnection) -> ConversationMeta:
            cursor = connection.execute(
                f'UPDATE conversations SET deleted_at = ? WHERE conversation_id = ? AND namespace_key = ? AND {_LIVE}',
                (deleted_at, conversation_id, namespace_key),
            )
            if cursor.rowcount == 0:
                raise ConversationNotFound(conversation_id)
            row = connection.execute(
                f'SELECT {_META_COLUMNS} FROM conversations WHERE conversation_id = ? AND namespace_key = ?',
                (conversation_id, namespace_key),
            ).fetchone()
            return _meta(row)

        return await self._driver.run(work)

    async def start_turn(
        self,
        namespace_key: str,
        conversation_id: str,
        *,
        run_id: str,
        author_user_id: str,
        previous_run_id: str | None = None,
    ) -> None:
        at = _now()

        def work(connection: CursorRoutedConnection) -> None:
            _require_live(connection, namespace_key, conversation_id)
            connection.execute(
                'INSERT INTO turns (conversation_id, run_id, author_user_id, at, previous_run_id) '
                'VALUES (?, ?, ?, ?, ?)',
                (conversation_id, run_id, author_user_id, at, previous_run_id),
            )

        await self._driver.run(work)

    async def set_turn_phase(
        self, namespace_key: str, conversation_id: str, *, run_id: str, phase: RunPhase, error: TurnError | None = None
    ) -> None:
        def work(connection: CursorRoutedConnection) -> None:
            _require_live(connection, namespace_key, conversation_id)
            # An unknown run_id updates zero rows: the contract's silent no-op.
            connection.execute(
                'UPDATE turns SET phase = ?, error_code = ?, error_message = ?, error_transient = ? '
                'WHERE conversation_id = ? AND run_id = ?',
                (
                    phase,
                    None if error is None else error.code,
                    None if error is None else error.message,
                    None if error is None else error.transient,
                    conversation_id,
                    run_id,
                ),
            )

        await self._driver.run(work)

    async def list_turns(self, namespace_key: str, conversation_id: str) -> Sequence[TurnRecord]:
        def work(connection: CursorRoutedConnection) -> Sequence[TurnRecord]:
            _require_live(connection, namespace_key, conversation_id)
            rows = connection.execute(
                'SELECT run_id, author_user_id, at, phase, previous_run_id, error_code, error_message, '
                'error_transient FROM turns WHERE conversation_id = ? ORDER BY seq',
                (conversation_id,),
            ).fetchall()
            return [
                TurnRecord(
                    run_id=row['run_id'],
                    author_user_id=row['author_user_id'],
                    at=datetime.fromtimestamp(row['at'], tz=UTC),
                    phase=cast('RunPhase', row['phase']),
                    previous_run_id=row['previous_run_id'],
                    error=None
                    if row['error_code'] is None
                    else TurnError(row['error_code'], row['error_message'], bool(row['error_transient'])),
                )
                for row in rows
            ]

        return await self._driver.run(work)

    async def purge(self, namespace_key: str, conversation_id: str) -> bool:
        purged_at = _now()

        def work(connection: CursorRoutedConnection) -> bool:
            row = connection.execute(
                'SELECT namespace_key FROM conversations WHERE conversation_id = ?', (conversation_id,)
            ).fetchone()
            if row is None or row['namespace_key'] != namespace_key:
                return False
            # The tombstone keeps namespace_key, so a repeated purge by the owner still answers True.
            connection.execute(
                'UPDATE conversations SET purged = 1, title = NULL, attrs = ?, deleted_at = COALESCE(deleted_at, ?) '
                'WHERE conversation_id = ?',
                ('{}', purged_at, conversation_id),
            )
            connection.execute('DELETE FROM turns WHERE conversation_id = ?', (conversation_id,))
            return True

        return await self._driver.run(work)
