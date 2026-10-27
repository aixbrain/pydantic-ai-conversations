"""`TurnFeedbackStore` on SQLite, in the same file as the registry when the app passes the same path."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_ai_conversations.feedback.store import TurnFeedbackRecord
from pydantic_ai_conversations.sqlite._driver import STEP_STORE_TRANSIENT_ERRORS, CursorRoutedConnection, SqliteDriver

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    kind TEXT NOT NULL,
    namespace_key TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    author_user_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (kind, namespace_key, conversation_id, run_id, author_user_id)
);
CREATE INDEX IF NOT EXISTS ix_feedback_conversation ON feedback(conversation_id, kind);
"""

_COLUMNS = 'kind, namespace_key, conversation_id, run_id, author_user_id, created_at, updated_at, data'


def _now() -> float:
    return datetime.now(UTC).timestamp()


def _record(row: Any) -> TurnFeedbackRecord:
    return TurnFeedbackRecord(
        kind=row['kind'],
        namespace_key=row['namespace_key'],
        conversation_id=row['conversation_id'],
        run_id=row['run_id'],
        author_user_id=row['author_user_id'],
        created_at=datetime.fromtimestamp(row['created_at'], UTC),
        updated_at=datetime.fromtimestamp(row['updated_at'], UTC),
        data=json.loads(row['data']),
    )


class SqliteTurnFeedbackStore:
    """`TurnFeedbackStore` over the kit's driver: one held WAL connection, each call one unit of work.

    Raises `sqlite3` errors as they are; `transient_errors` names the ones the routes turn into a
    503, the same set the kit's step store declares.
    """

    transient_errors = STEP_STORE_TRANSIENT_ERRORS

    def __init__(self, database: str | Path) -> None:
        self._driver = SqliteDriver(database, schema=_SCHEMA)

    async def open(self) -> None:
        def work(connection: CursorRoutedConnection) -> None:
            connection.execute('SELECT 1 FROM feedback LIMIT 1')

        await self._driver.run(work)

    async def close(self) -> None:
        await self._driver.close()

    async def upsert(
        self,
        *,
        kind: str,
        namespace_key: str,
        conversation_id: str,
        run_id: str,
        author_user_id: str,
        data: Mapping[str, Any],
    ) -> TurnFeedbackRecord:
        now, payload = _now(), json.dumps(dict(data))

        def work(connection: CursorRoutedConnection) -> TurnFeedbackRecord:
            connection.execute(
                f'INSERT INTO feedback ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?) '
                'ON CONFLICT (kind, namespace_key, conversation_id, run_id, author_user_id) DO UPDATE SET '
                'updated_at = excluded.updated_at, data = excluded.data',
                (kind, namespace_key, conversation_id, run_id, author_user_id, now, now, payload),
            )
            row = connection.execute(
                f'SELECT {_COLUMNS} FROM feedback WHERE kind = ? AND namespace_key = ? AND conversation_id = ? '
                'AND run_id = ? AND author_user_id = ?',
                (kind, namespace_key, conversation_id, run_id, author_user_id),
            ).fetchone()
            return _record(row)

        return await self._driver.run(work)

    async def delete(
        self, *, kind: str, namespace_key: str, conversation_id: str, run_id: str, author_user_id: str
    ) -> bool:
        def work(connection: CursorRoutedConnection) -> bool:
            cursor = connection.execute(
                'DELETE FROM feedback WHERE kind = ? AND run_id = ? AND author_user_id = ? '
                'AND namespace_key = ? AND conversation_id = ?',
                (kind, run_id, author_user_id, namespace_key, conversation_id),
            )
            return cursor.rowcount > 0

        return await self._driver.run(work)

    async def by_run_ids(
        self,
        *,
        kind: str,
        namespace_key: str,
        conversation_id: str,
        run_ids: Sequence[str],
        author_user_id: str | None = None,
    ) -> Sequence[TurnFeedbackRecord]:
        if not run_ids:
            return []
        placeholders = ', '.join('?' for _ in run_ids)
        author_clause = '' if author_user_id is None else ' AND author_user_id = ?'
        params: list[Any] = [kind, namespace_key, conversation_id, *run_ids]
        if author_user_id is not None:
            params.append(author_user_id)

        def work(connection: CursorRoutedConnection) -> list[TurnFeedbackRecord]:
            rows = connection.execute(
                f'SELECT {_COLUMNS} FROM feedback WHERE kind = ? AND namespace_key = ? AND conversation_id = ? '
                f'AND run_id IN ({placeholders}){author_clause}',
                params,
            ).fetchall()
            return [_record(row) for row in rows]

        return await self._driver.run(work)

    async def list(self, *, kind: str, namespace_key: str, conversation_id: str) -> Sequence[TurnFeedbackRecord]:
        def work(connection: CursorRoutedConnection) -> list[TurnFeedbackRecord]:
            rows = connection.execute(
                f'SELECT {_COLUMNS} FROM feedback WHERE kind = ? AND namespace_key = ? AND conversation_id = ? '
                'ORDER BY created_at, author_user_id',
                (kind, namespace_key, conversation_id),
            ).fetchall()
            return [_record(row) for row in rows]

        return await self._driver.run(work)

    async def delete_conversation(self, *, namespace_key: str, conversation_id: str) -> int:
        def work(connection: CursorRoutedConnection) -> int:
            cursor = connection.execute(
                'DELETE FROM feedback WHERE namespace_key = ? AND conversation_id = ?', (namespace_key, conversation_id)
            )
            return cursor.rowcount

        return await self._driver.run(work)

    async def ping(self) -> Mapping[str, Any]:
        await self.open()
        return {'ok': True}
