"""`TurnFeedbackStore` on PostgreSQL: one `feedback` table with a JSONB payload."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from psycopg import AsyncConnection, sql
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from pydantic_ai_conversations.feedback.store import TurnFeedbackRecord
from pydantic_ai_conversations.postgres._schema import create_feedback_schema

_COLUMNS = sql.SQL('kind, namespace_key, conversation_id, run_id, author_user_id, created_at, updated_at, data')


def _record(row: DictRow) -> TurnFeedbackRecord:
    return TurnFeedbackRecord(
        kind=row['kind'],
        namespace_key=row['namespace_key'],
        conversation_id=row['conversation_id'],
        run_id=row['run_id'],
        author_user_id=row['author_user_id'],
        created_at=row['created_at'],
        updated_at=row['updated_at'],
        data=row['data'],
    )


class PostgresTurnFeedbackStore:
    """`TurnFeedbackStore` on a psycopg pool, the registry's when the app passes the same one.

    `create_schema` is False by default, as for the registry: the DDL ships as `feedback.sql` for
    the app's migration tool. `open()` probes the table and names the file when it is missing.
    With `dsn=` the store owns its pool and, since a closed psycopg pool cannot be reopened,
    makes a new one on `open()` after `close()`; with `pool=` the app's pool stays open.
    """

    transient_errors = (psycopg.OperationalError,)

    def __init__(
        self,
        *,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = None,
        dsn: str | None = None,
        schema: str = 'public',
        create_schema: bool = False,
    ) -> None:
        if (pool is None) == (dsn is None):
            raise ValueError('provide exactly one of `pool=` or `dsn=`')
        self._own_pool = pool is None
        self._dsn = dsn or ''
        self._pool: AsyncConnectionPool[AsyncConnection[DictRow]] = pool if pool is not None else self._new_pool()
        self._schema = schema
        self._create_schema = create_schema
        self._opened = False
        self._closed = False
        # Serializes `open()`: concurrent first use would race `AsyncConnectionPool.open(wait=True)`.
        self._open_lock = asyncio.Lock()

    def _new_pool(self) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
        return AsyncConnectionPool(self._dsn, open=False, kwargs={'row_factory': dict_row})

    def _table(self) -> sql.Composed:
        return sql.SQL('{}.{}').format(sql.Identifier(self._schema), sql.Identifier('feedback'))

    async def open(self) -> None:
        async with self._open_lock:
            if self._opened:
                return
            if self._closed and self._own_pool:
                self._pool = self._new_pool()
            self._closed = False
            await self._pool.open(wait=True, timeout=self._pool.timeout)
            if self._create_schema:
                await create_feedback_schema(self._pool, self._schema)
            async with self._pool.connection() as conn:
                cur = await conn.execute('SELECT to_regclass(%s) AS oid', (f'{self._schema}.feedback',))
                row = await cur.fetchone()
            if row is None or row['oid'] is None:
                raise RuntimeError(
                    f'schema {self._schema!r} is missing feedback -- apply the DDL shipped as '
                    'pydantic_ai_conversations/postgres/feedback.sql, or construct the store with create_schema=True'
                )
            self._opened = True

    async def close(self) -> None:
        self._opened = False
        self._closed = True
        if self._own_pool:
            await self._pool.close()

    @asynccontextmanager
    async def _connection(self) -> AsyncGenerator[AsyncConnection[DictRow]]:
        if not self._opened:
            await self.open()
        async with self._pool.connection() as conn:
            previous = conn.row_factory
            conn.row_factory = dict_row
            try:
                yield conn
                await conn.commit()
            finally:
                conn.row_factory = previous

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
        async with self._connection() as conn:
            cur = await conn.execute(
                sql.SQL(
                    'INSERT INTO {table} ({cols}) VALUES (%s, %s, %s, %s, %s, now(), now(), %s) '
                    'ON CONFLICT (kind, namespace_key, conversation_id, run_id, author_user_id) DO UPDATE SET '
                    'updated_at = now(), data = excluded.data RETURNING {cols}'
                ).format(table=self._table(), cols=_COLUMNS),
                (kind, namespace_key, conversation_id, run_id, author_user_id, Jsonb(dict(data))),
            )
            row = await cur.fetchone()
        if row is None:  # pragma: no cover - RETURNING on an upsert always yields the row
            raise RuntimeError('upsert returned no row')
        return _record(row)

    async def delete(
        self, *, kind: str, namespace_key: str, conversation_id: str, run_id: str, author_user_id: str
    ) -> bool:
        async with self._connection() as conn:
            cur = await conn.execute(
                sql.SQL(
                    'DELETE FROM {table} WHERE kind = %s AND run_id = %s AND author_user_id = %s '
                    'AND namespace_key = %s AND conversation_id = %s'
                ).format(table=self._table()),
                (kind, run_id, author_user_id, namespace_key, conversation_id),
            )
            return cur.rowcount > 0

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
        author = sql.SQL('') if author_user_id is None else sql.SQL(' AND author_user_id = %s')
        params: list[Any] = [kind, namespace_key, conversation_id, list(run_ids)]
        if author_user_id is not None:
            params.append(author_user_id)
        async with self._connection() as conn:
            cur = await conn.execute(
                sql.SQL(
                    'SELECT {cols} FROM {table} WHERE kind = %s AND namespace_key = %s AND conversation_id = %s '
                    'AND run_id = ANY(%s){author}'
                ).format(cols=_COLUMNS, table=self._table(), author=author),
                params,
            )
            return [_record(row) for row in await cur.fetchall()]

    async def list(self, *, kind: str, namespace_key: str, conversation_id: str) -> Sequence[TurnFeedbackRecord]:
        async with self._connection() as conn:
            cur = await conn.execute(
                sql.SQL(
                    'SELECT {cols} FROM {table} WHERE kind = %s AND namespace_key = %s AND conversation_id = %s '
                    'ORDER BY created_at, author_user_id'
                ).format(cols=_COLUMNS, table=self._table()),
                (kind, namespace_key, conversation_id),
            )
            return [_record(row) for row in await cur.fetchall()]

    async def delete_conversation(self, *, namespace_key: str, conversation_id: str) -> int:
        async with self._connection() as conn:
            cur = await conn.execute(
                sql.SQL('DELETE FROM {table} WHERE namespace_key = %s AND conversation_id = %s').format(
                    table=self._table()
                ),
                (namespace_key, conversation_id),
            )
            return cur.rowcount

    async def ping(self) -> Mapping[str, Any]:
        async with self._connection() as conn:
            await conn.execute('SELECT 1')
        return {'ok': True}
