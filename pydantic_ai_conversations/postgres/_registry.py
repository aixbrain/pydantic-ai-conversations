"""`ConversationRegistry` on PostgreSQL: the port implementation, in SQL.

Everything a SQL backend owes the protocol lives here -- a global `conversation_id` primary key,
`INSERT ... ON CONFLICT DO NOTHING` + re-select for the atomic id claim,
`SET message_count = message_count + %s` for the stats increment, and purge as a permanent
in-table tombstone so the id stays claimed against every namespace.

Timestamps are `TIMESTAMPTZ` and psycopg returns them timezone-aware with microseconds intact, so
unlike the Mongo kit this backend needs no normalization on read. The ordering expression is
`COALESCE(last_message_at, created_at)`, never a bare column: under `DESC` Postgres sorts NULLs
first, which would float never-used conversations to the top of a chat list.

Pool ownership, the startup schema probe and the `psycopg.Error` -> `RegistryUnavailable` mapping
are `_driver.py`'s.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import psycopg
from psycopg import AsyncConnection, sql
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from pydantic_ai_conversations.errors import ConversationNotFound, RegistryUnavailable
from pydantic_ai_conversations.postgres._driver import PostgresDriver, maps_backend_errors
from pydantic_ai_conversations.registry import ConversationMeta, RunPhase, TurnError, TurnRecord

_LIVE = sql.SQL('deleted_at IS NULL AND NOT purged')
"""The live-entry predicate every read and mutation carries, alongside the namespace."""

_SORT_AT = sql.SQL('COALESCE(last_message_at, created_at)')
"""The activity timestamp `list` orders by; `ix_conversations_activity` indexes this expression.
Never a bare `last_message_at`: under `DESC` Postgres sorts NULLs first."""

_META_COLUMNS = sql.SQL(
    'conversation_id, namespace_key, created_by, created_at, last_message_at, message_count, title, deleted_at, attrs'
)


def _meta(row: DictRow) -> ConversationMeta:
    return ConversationMeta(
        namespace_key=row['namespace_key'],
        conversation_id=row['conversation_id'],
        created_by=row['created_by'],
        created_at=row['created_at'],
        last_message_at=row['last_message_at'],
        message_count=row['message_count'],
        title=row['title'],
        deleted_at=row['deleted_at'],
        attrs=row['attrs'],
    )


class PostgresConversationRegistry:
    """`ConversationRegistry` on a PostgreSQL database.

    Methods lazy-connect, so `open()` is optional; calling it in the feature's lifespan reports a
    missing schema or an unreachable server at startup rather than on the first request. After
    `close()` every method raises `RegistryUnavailable` until `open()` reopens.

    `create_schema` is False by default: the DDL ships as `schema.sql` for your migration tool to
    apply. `schema=` names the Postgres schema, so the tables need not live in `public`.
    """

    def __init__(
        self,
        *,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = None,
        dsn: str | None = None,
        schema: str = 'public',
        create_schema: bool = False,
    ) -> None:
        self._driver = PostgresDriver(pool=pool, dsn=dsn, schema=schema, create_schema=create_schema)

    async def open(self) -> None:
        await self._driver.open()

    async def close(self) -> None:
        await self._driver.close()

    @maps_backend_errors
    async def ping(self) -> Mapping[str, Any]:
        """Proves the connection and the table through the catalog, without reading rows. The
        report is public, so no count."""
        async with self._driver.connection() as conn:
            cur = await conn.execute(
                'SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace '
                'WHERE n.nspname = %s AND c.relname = %s',
                (self._driver.schema, 'conversations'),
            )
            row = await cur.fetchone()
        if row is None:
            # `ready()`'s schema probe runs once, at `open()` -- not on every call. If `conversations`
            # is dropped or its schema renamed afterward, this catalog query returns no row, so name
            # what's missing rather than answering healthy over a table that is gone.
            raise RegistryUnavailable(
                f'schema {self._driver.schema!r} is missing conversations -- '
                'it existed when this registry opened but is gone now'
            )
        return {'ok': True}

    @maps_backend_errors
    async def register_conversation(
        self, namespace_key: str, conversation_id: str, *, created_by: str, attrs: Mapping[str, str] | None = None
    ) -> ConversationMeta:
        table = self._driver.table('conversations')
        async with self._driver.connection() as conn:
            await conn.execute(
                sql.SQL(
                    'INSERT INTO {table} (conversation_id, namespace_key, created_by, created_at, attrs) '
                    'VALUES (%s, %s, %s, now(), %s) ON CONFLICT (conversation_id) DO NOTHING'
                ).format(table=table),
                # `attrs` is NOT NULL DEFAULT '{}'; naming the column means passing a value, since
                # the default only applies when the column is omitted.
                (conversation_id, namespace_key, created_by, Jsonb(dict(attrs or {}))),
            )
            # Purge tombstones the row rather than deleting it, so under this backend's own
            # operations the `ON CONFLICT DO NOTHING` above always leaves one behind. `row is None`
            # below covers the row being gone for a reason outside this code -- the table truncated,
            # or rows deleted by something other than the application.
            cur = await conn.execute(
                sql.SQL('SELECT {cols}, purged FROM {table} WHERE conversation_id = %s').format(
                    cols=_META_COLUMNS, table=table
                ),
                (conversation_id,),
            )
            row = await cur.fetchone()
        if row is None or row['purged'] or row['namespace_key'] != namespace_key or row['deleted_at'] is not None:
            raise ConversationNotFound(conversation_id)
        return _meta(row)

    @maps_backend_errors
    async def get(self, namespace_key: str, conversation_id: str) -> ConversationMeta | None:
        async with self._driver.connection() as conn:
            cur = await conn.execute(
                sql.SQL(
                    'SELECT {cols} FROM {table} WHERE conversation_id = %s AND namespace_key = %s AND {live}'
                ).format(cols=_META_COLUMNS, table=self._driver.table('conversations'), live=_LIVE),
                (conversation_id, namespace_key),
            )
            row = await cur.fetchone()
        return None if row is None else _meta(row)

    async def _require_live(self, conn: psycopg.AsyncConnection[DictRow], namespace_key: str, cid: str) -> None:
        cur = await conn.execute(
            sql.SQL('SELECT 1 FROM {table} WHERE conversation_id = %s AND namespace_key = %s AND {live}').format(
                table=self._driver.table('conversations'), live=_LIVE
            ),
            (cid, namespace_key),
        )
        if await cur.fetchone() is None:
            raise ConversationNotFound(cid)

    @maps_backend_errors
    async def set_title(self, namespace_key: str, conversation_id: str, title: str) -> None:
        async with self._driver.connection() as conn:
            cur = await conn.execute(
                sql.SQL(
                    'UPDATE {table} SET title = %s WHERE conversation_id = %s AND namespace_key = %s AND {live}'
                ).format(table=self._driver.table('conversations'), live=_LIVE),
                (title, conversation_id, namespace_key),
            )
            rowcount = cur.rowcount
        if rowcount == 0:
            raise ConversationNotFound(conversation_id)

    @maps_backend_errors
    async def update_message_stats(
        self, namespace_key: str, conversation_id: str, *, last_message_at: datetime, message_count_delta: int
    ) -> None:
        async with self._driver.connection() as conn:
            cur = await conn.execute(
                sql.SQL(
                    'UPDATE {table} SET last_message_at = %s, message_count = message_count + %s '
                    'WHERE conversation_id = %s AND namespace_key = %s AND {live}'
                ).format(table=self._driver.table('conversations'), live=_LIVE),
                (last_message_at, message_count_delta, conversation_id, namespace_key),
            )
            rowcount = cur.rowcount
        if rowcount == 0:
            raise ConversationNotFound(conversation_id)

    @maps_backend_errors
    async def soft_delete(self, namespace_key: str, conversation_id: str) -> ConversationMeta:
        async with self._driver.connection() as conn:
            cur = await conn.execute(
                sql.SQL(
                    'UPDATE {table} SET deleted_at = now() '
                    'WHERE conversation_id = %s AND namespace_key = %s AND {live} RETURNING {cols}'
                ).format(table=self._driver.table('conversations'), live=_LIVE, cols=_META_COLUMNS),
                (conversation_id, namespace_key),
            )
            row = await cur.fetchone()
        if row is None:
            raise ConversationNotFound(conversation_id)
        return _meta(row)

    @maps_backend_errors
    async def start_turn(
        self,
        namespace_key: str,
        conversation_id: str,
        *,
        run_id: str,
        author_user_id: str,
        previous_run_id: str | None = None,
    ) -> None:
        async with self._driver.connection() as conn:
            await self._require_live(conn, namespace_key, conversation_id)
            await conn.execute(
                sql.SQL(
                    'INSERT INTO {table} (conversation_id, run_id, author_user_id, at, previous_run_id) '
                    'VALUES (%s, %s, %s, now(), %s)'
                ).format(table=self._driver.table('turns')),
                (conversation_id, run_id, author_user_id, previous_run_id),
            )

    @maps_backend_errors
    async def set_turn_phase(
        self, namespace_key: str, conversation_id: str, *, run_id: str, phase: RunPhase, error: TurnError | None = None
    ) -> None:
        async with self._driver.connection() as conn:
            await self._require_live(conn, namespace_key, conversation_id)
            # An unknown run_id updates zero rows: the contract's silent no-op.
            await conn.execute(
                sql.SQL(
                    'UPDATE {table} SET phase = %s, error_code = %s, error_message = %s, error_transient = %s '
                    'WHERE conversation_id = %s AND run_id = %s'
                ).format(table=self._driver.table('turns')),
                (
                    phase,
                    None if error is None else error.code,
                    None if error is None else error.message,
                    None if error is None else error.transient,
                    conversation_id,
                    run_id,
                ),
            )

    @maps_backend_errors
    async def list_turns(self, namespace_key: str, conversation_id: str) -> Sequence[TurnRecord]:
        async with self._driver.connection() as conn:
            await self._require_live(conn, namespace_key, conversation_id)
            cur = await conn.execute(
                sql.SQL(
                    'SELECT run_id, author_user_id, at, phase, previous_run_id, error_code, error_message, '
                    'error_transient FROM {table} WHERE conversation_id = %s ORDER BY seq'
                ).format(table=self._driver.table('turns')),
                (conversation_id,),
            )
            rows = await cur.fetchall()
        return [
            TurnRecord(
                run_id=r['run_id'],
                author_user_id=r['author_user_id'],
                at=r['at'],
                phase=r['phase'],
                previous_run_id=r['previous_run_id'],
                error=None
                if r['error_code'] is None
                else TurnError(r['error_code'], r['error_message'], r['error_transient']),
            )
            for r in rows
        ]

    @maps_backend_errors
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
        table = self._driver.table('conversations')
        # JSONB containment binds the whole object as one value, so a key holding a dot filters on
        # itself rather than being read as a path. GIN-indexed; empty attrs filter nothing.
        attrs_sql = sql.SQL(' AND attrs @> %s') if attrs else sql.SQL('')
        attrs_params: list[Any] = [Jsonb(dict(attrs))] if attrs else []
        async with self._driver.connection() as conn:
            keyset_sql, keyset_params = sql.SQL(''), []
            if page_after is not None:
                # Resolved inside the same filtered set, so an unknown, deleted, or attrs-excluded
                # cursor ends the list rather than raising.
                cur = await conn.execute(
                    sql.SQL(
                        'SELECT {sort} AS activity FROM {table} '
                        'WHERE conversation_id = %s AND namespace_key = %s AND {live}{attrs}'
                    ).format(sort=_SORT_AT, table=table, live=_LIVE, attrs=attrs_sql),
                    (page_after, namespace_key, *attrs_params),
                )
                cursor_row = await cur.fetchone()
                if cursor_row is None:
                    return []
                keyset_sql = sql.SQL(' AND ({sort}, conversation_id) < (%s, %s)').format(sort=_SORT_AT)
                keyset_params = [cursor_row['activity'], page_after]
            cur = await conn.execute(
                sql.SQL(
                    'SELECT {cols} FROM {table} WHERE namespace_key = %s AND {live}{attrs}{keyset} '
                    'ORDER BY {sort} DESC, conversation_id DESC LIMIT %s'
                ).format(
                    cols=_META_COLUMNS, table=table, live=_LIVE, attrs=attrs_sql, keyset=keyset_sql, sort=_SORT_AT
                ),
                (namespace_key, *attrs_params, *keyset_params, limit),
            )
            rows = await cur.fetchall()
        return [_meta(row) for row in rows]

    @maps_backend_errors
    async def purge(self, namespace_key: str, conversation_id: str) -> bool:
        table = self._driver.table('conversations')
        async with self._driver.connection() as conn:
            # One statement: tombstone the row only if this namespace owns it, and report whether
            # it did. `deleted_at` keeps an earlier soft-delete's timestamp.
            cur = await conn.execute(
                sql.SQL(
                    'UPDATE {table} SET purged = TRUE, title = NULL, attrs = %s, '
                    'deleted_at = COALESCE(deleted_at, now()) '
                    'WHERE conversation_id = %s AND namespace_key = %s RETURNING conversation_id'
                ).format(table=table),
                (Jsonb({}), conversation_id, namespace_key),
            )
            owned = await cur.fetchone() is not None
            if owned:
                await conn.execute(
                    sql.SQL('DELETE FROM {table} WHERE conversation_id = %s').format(table=self._driver.table('turns')),
                    (conversation_id,),
                )
        return owned
