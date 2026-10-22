"""Driver plumbing for the PostgreSQL backend: pool ownership, the schema probe, error mapping.

Separate from the port implementation so `_registry.py` carries SQL and protocol semantics only.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager, suppress
from types import CoroutineType
from typing import Any, ParamSpec, TypeVar

import asyncpg
import psycopg
from psycopg import AsyncConnection, IsolationLevel, sql
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from pydantic_ai_conversations.errors import RegistryUnavailable
from pydantic_ai_conversations.postgres._schema import TABLES, create_schema

_P = ParamSpec('_P')
_R = TypeVar('_R')

STEP_STORE_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    asyncpg.PostgresConnectionError,
    asyncpg.OperatorInterventionError,
    asyncpg.InsufficientResourcesError,
    asyncpg.TransactionRollbackError,
    OSError,
)
"""What the asyncpg step store raises when the server is unreachable or restarting, out of
connections or resources, or rolled a transaction back on a deadlock or serialization conflict:
the SQLSTATE classes 08, 57, 53 and 40. A unique violation is left out; the harness store turns a
run id already stored into `ValueError`."""


def maps_backend_errors(
    method: Callable[_P, CoroutineType[Any, Any, _R]],
) -> Callable[_P, CoroutineType[Any, Any, _R]]:
    """`psycopg.Error` -> `RegistryUnavailable`, per the `errors.py` contract; everything else
    (`ConversationNotFound`, a caller's `ValueError`) passes through untranslated.

    The annotation is `types.CoroutineType`, not `collections.abc.Coroutine`: pyright models an
    `async def`'s return as the concrete type, and the looser one fails the assignment when the
    decorated class is checked against the `ConversationRegistry` protocol.
    """

    @functools.wraps(method)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return await method(*args, **kwargs)
        except psycopg.Error as exc:
            raise RegistryUnavailable(str(exc)) from exc

    return wrapper


class PostgresDriver:
    """Holds the pool, owns it or borrows it, and checks the schema is there before serving.

    Whoever opens the pool closes it: with `dsn=` the driver opened it and `close()` closes it;
    with `pool=` the app shares it with its own queries, and `close()` leaves it open. Either way
    `close()` puts the driver in its outage state, and `open()` lifts it. `ready()` opens the pool
    and probes the schema on first use, so an explicit `open()` only moves those errors to
    startup. `create_schema` defaults to False because Postgres deployments apply schema in a step
    before the app starts; `open()` then names missing tables rather than letting the first
    request fail on `relation does not exist`.
    """

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
        self._dsn = dsn
        self._schema = schema
        self._create_schema = create_schema
        self._own_pool = pool is None
        self._pool = pool if pool is not None else self._new_pool()
        self._closed = False
        self._opened = False
        # Guards `open()`: `AsyncConnectionPool.open(wait=True)` is documented safe to call again
        # on an already-open pool, but NOT safe to call concurrently before the first call's own
        # `wait()` resolves -- a second in-flight `wait()` hits `assert not self._pool_full_event`
        # inside psycopg_pool. Concurrent first-use (every caller sees `_opened=False` and races
        # into `open()` together) makes this routine, not theoretical -- `ready()` triggers exactly
        # that shape whenever nobody calls `open()` explicitly before the first concurrent burst.
        self._open_lock = asyncio.Lock()

    def _new_pool(self) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
        assert self._dsn is not None
        return AsyncConnectionPool(self._dsn, open=False, kwargs={'row_factory': dict_row})

    def table(self, name: str) -> sql.Composed:
        """`schema.name`, both quoted as identifiers rather than interpolated."""
        return sql.SQL('{}.{}').format(sql.Identifier(self._schema), sql.Identifier(name))

    @property
    def schema(self) -> str:
        """The plain schema name, for callers that need to match it against catalog text (e.g.
        `pg_namespace.nspname`) rather than embed it as a quoted `sql.Identifier`."""
        return self._schema

    async def ready(self) -> None:
        """Every operation's guard: refuse while closed, and open lazily on first use."""
        if self._closed:
            raise RegistryUnavailable('registry is closed')
        if not self._opened:
            await self.open()

    @asynccontextmanager
    async def _checkout(self) -> AsyncGenerator[AsyncConnection[DictRow]]:
        """Check out a connection with `dict_row` rows and `READ COMMITTED`, restoring the pool's
        own settings before the connection goes back.

        A `pool=` pool is the app's and may default to any isolation level. Under `REPEATABLE READ`
        or `SERIALIZABLE` the re-select after `register_conversation`'s `ON CONFLICT DO NOTHING`
        reads a snapshot fixed at its first statement and can miss a concurrent winner's row. psycopg
        refuses to change the isolation level inside a transaction, so this commits or rolls back
        before restoring it. It sits below `ready()` because `open()` calls it through
        `_require_tables`.
        """
        async with self._pool.connection() as conn:
            previous_row_factory = conn.row_factory
            previous_isolation_level = conn.isolation_level
            conn.row_factory = dict_row
            await conn.set_isolation_level(IsolationLevel.READ_COMMITTED)
            try:
                yield conn
            except BaseException:
                if not conn.closed:
                    with suppress(psycopg.Error):
                        await conn.rollback()
                raise
            else:
                await conn.commit()
            finally:
                conn.row_factory = previous_row_factory
                # A commit that failed leaves the connection inside its transaction, where psycopg
                # refuses to change the isolation level, and a lost connection refuses everything.
                # Neither may replace the error on its way out; the pool resets or discards the
                # connection when it comes back.
                if not conn.closed:
                    with suppress(psycopg.Error):
                        await conn.set_isolation_level(previous_isolation_level)

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[AsyncConnection[DictRow]]:
        """Yield a checked-out connection (see `_checkout`), opening the pool first if this is the
        first use (see `ready`)."""
        await self.ready()
        async with self._checkout() as conn:
            yield conn

    @maps_backend_errors
    async def open(self) -> None:
        async with self._open_lock:
            if self._opened:
                # A concurrent caller already finished opening while we waited for the lock --
                # nothing left to do. Without this check every racer would re-run the pool open,
                # the schema DDL, and the table probe in turn, harmlessly but needlessly.
                return
            if self._closed and self._own_pool:
                # A pool we opened, we can reopen; a closed psycopg pool is not reusable.
                self._pool = self._new_pool()
            self._closed = False
            # `AsyncConnectionPool.open()` hardcodes its own 30s default for `timeout` regardless of
            # what the pool was constructed with; honor the pool's own configured `timeout` instead so
            # a caller-supplied pool's shorter setting (e.g. for fast-failing tests) actually applies.
            # Serialized by `_open_lock`: concurrent `wait=True` calls on the same pool are not safe
            # (see the lock's own docstring in `__init__`), which concurrent first-use through
            # `ready()` hits routinely, not just a caller's own explicit call racing the lazy path.
            await self._pool.open(wait=True, timeout=self._pool.timeout)
            if self._create_schema:
                await create_schema(self._pool, self._schema)
            await self._require_tables()
            # Only after the probe succeeds: a failed probe must be retried on the next `ready()`, not
            # latched as "opened" -- otherwise a later call would skip re-probing entirely and let a
            # request through with no tables underneath it.
            self._opened = True

    async def _require_tables(self) -> None:
        """Fail at startup, naming what is missing, rather than on the first request."""
        missing: list[str] = []
        async with self._checkout() as conn:
            for table in TABLES:
                cur = await conn.execute('SELECT to_regclass(%s) AS oid', (f'{self._schema}.{table}',))
                row = await cur.fetchone()
                if row is None or row['oid'] is None:
                    missing.append(table)
        if missing:
            raise RegistryUnavailable(
                f'schema {self._schema!r} is missing {", ".join(missing)} -- apply the DDL shipped as '
                'pydantic_ai_conversations/postgres/schema.sql, or construct the registry with create_schema=True'
            )

    async def close(self) -> None:
        self._closed = True
        self._opened = False
        if self._own_pool:
            await self._pool.close()
