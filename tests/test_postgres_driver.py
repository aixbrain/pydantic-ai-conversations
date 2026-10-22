"""Pool ownership, the startup schema probe, and the error contract."""

from typing import Any
from uuid import uuid4

import psycopg
import pytest
from psycopg import AsyncConnection, IsolationLevel, sql
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from pydantic_ai_conversations.errors import RegistryUnavailable
from pydantic_ai_conversations.postgres._driver import PostgresDriver, maps_backend_errors


def test_exactly_one_connection_source_is_required() -> None:
    with pytest.raises(ValueError):
        PostgresDriver()
    with pytest.raises(ValueError):
        PostgresDriver(pool=None, dsn=None, schema='public')


async def test_open_without_a_schema_names_the_missing_tables(postgres_dsn: str) -> None:
    """A registry pointed at a database nobody migrated must say so at startup, by name."""
    driver = PostgresDriver(dsn=postgres_dsn, schema=uuid4().hex[:8])
    with pytest.raises(RegistryUnavailable) as caught:
        await driver.open()
    assert 'conversations' in str(caught.value) and 'turns' in str(caught.value)
    await driver.close()


async def test_open_names_only_the_tables_actually_missing(postgres_dsn: str) -> None:
    """The probe reports a partial gap precisely, not just 'something is missing' -- a schema
    with `conversations` but not `turns` (e.g. a half-applied migration) must name only `turns`."""
    schema = uuid4().hex[:8]
    async with await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True) as conn:
        await conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        await conn.execute(sql.SQL('CREATE TABLE {}.conversations (id int)').format(sql.Identifier(schema)))
    driver = PostgresDriver(dsn=postgres_dsn, schema=schema)
    with pytest.raises(RegistryUnavailable) as caught:
        await driver.open()
    named = str(caught.value).split('is missing ', 1)[1].split(' --', 1)[0]
    assert named == 'turns'
    await driver.close()


async def test_create_schema_true_applies_the_ddl(postgres_dsn: str) -> None:
    driver = PostgresDriver(dsn=postgres_dsn, schema=uuid4().hex[:8], create_schema=True)
    await driver.open()
    async with driver.connection() as conn:
        cur = await conn.execute(sql.SQL('SELECT count(*) AS n FROM {}').format(driver.table('conversations')))
        row = await cur.fetchone()
        assert row is not None and row['n'] == 0
    await driver.close()


async def test_connection_refuses_while_closed(postgres_dsn: str) -> None:
    """`connection()`'s own guard, not just `ready()`'s: a caller reaching for a connection on a
    closed driver must see `RegistryUnavailable`, not a pool error from a torn-down pool."""
    driver = PostgresDriver(dsn=postgres_dsn, schema=uuid4().hex[:8], create_schema=True)
    await driver.open()
    await driver.close()
    with pytest.raises(RegistryUnavailable):
        async with driver.connection():
            pass


async def test_table_quotes_schema_and_name() -> None:
    """`table()` is pure string composition and needs no live connection, but `PostgresDriver(dsn=)`
    still builds a real (unopened) pool in `__init__` -- close it via the driver's own `close()`
    rather than leaving it for `__del__`."""
    driver = PostgresDriver(dsn='postgresql://localhost/db', schema='my schema')
    composed = driver.table('my table')
    rendered = composed.as_string(None)
    assert rendered == '"my schema"."my table"'
    await driver.close()


async def test_a_borrowed_pool_survives_close(postgres_dsn: str) -> None:
    """`close()` shuts the driver, not a pool it did not open -- the owner closes that."""
    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(postgres_dsn, open=False)
    await pool.open(wait=True)
    driver = PostgresDriver(pool=pool, schema=uuid4().hex[:8], create_schema=True)
    await driver.open()
    await driver.close()
    with pytest.raises(RegistryUnavailable):
        await driver.ready()
    async with pool.connection() as conn:  # still usable
        await conn.execute('SELECT 1')
    await pool.close()


async def test_connection_yields_dict_rows_from_a_plain_pool(postgres_dsn: str) -> None:
    """`connection()` promises `dict_row` rows regardless of the caller's own pool config -- a
    borrowed pool with the default `tuple_row` factory must still hand back dict-shaped rows, and
    must be left exactly as it was found once the driver is done with the connection.

    Checks the restore on the *same* connection object the driver touched, not a second checkout:
    with the pool's default `min_size=4` and FIFO handout, a second `pool.connection()` call could
    return a different, untouched connection and pass regardless of whether `_driver.py` actually
    restores `row_factory` -- this would not fail if the restore were removed.
    """
    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(postgres_dsn, open=False)  # default tuple_row
    await pool.open(wait=True)
    schema = uuid4().hex[:8]
    driver = PostgresDriver(pool=pool, schema=schema, create_schema=True)
    await driver.open()
    async with driver.connection() as conn:
        assert conn.row_factory is dict_row
        cur = await conn.execute(sql.SQL('SELECT count(*) AS n FROM {}').format(driver.table('conversations')))
        row = await cur.fetchone()
        assert row is not None and row['n'] == 0
    assert conn.row_factory is not dict_row  # restored on the very connection the driver used
    await driver.close()
    await pool.close()


async def test_an_owned_pool_is_closed_and_rebuilt_on_reopen(postgres_dsn: str) -> None:
    schema = uuid4().hex[:8]
    driver = PostgresDriver(dsn=postgres_dsn, schema=schema, create_schema=True)
    await driver.open()
    await driver.close()
    with pytest.raises(RegistryUnavailable):
        await driver.ready()
    await driver.open()  # a pool it opened, it can reopen
    await driver.ready()
    await driver.close()


async def test_a_driver_error_surfaces_as_registry_unavailable() -> None:
    """`psycopg.Error` -> `RegistryUnavailable`, on an address nothing answers.

    The pool is built with a short `timeout=`, and passed in via `pool=` rather than `dsn=`: the
    DSN's own `connect_timeout=1` only bounds a single connection attempt, while
    `AsyncConnectionPool.open(wait=True)` retries in the background for its own hardcoded 30s
    default regardless -- `_driver.py` passes the pool's configured `timeout` through to that
    call, so this proves the same mapping in about a second instead of waiting out that default.
    """
    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        'postgresql://nobody@127.0.0.1:1/none?connect_timeout=1', open=False, timeout=1
    )
    driver = PostgresDriver(pool=pool, schema='public', create_schema=True)
    with pytest.raises(RegistryUnavailable):
        await driver.open()
    await pool.close()


async def test_ready_lazily_opens_the_pool_on_first_use(postgres_dsn: str) -> None:
    """`open()` is optional -- a caller that goes straight for `connection()` must still be served,
    matching the SQLite and Mongo kits' own lazy connect."""
    driver = PostgresDriver(dsn=postgres_dsn, schema=uuid4().hex[:8], create_schema=True)
    async with driver.connection() as conn:
        cur = await conn.execute('SELECT 1 AS ok')
        row = await cur.fetchone()
        assert row is not None and row['ok'] == 1
    await driver.close()


async def test_a_failed_lazy_open_is_retried_not_latched(postgres_dsn: str) -> None:
    """If a failed `open()` still set `_opened = True`, the *next* `ready()` would skip re-running
    the probe entirely and let a request through with no tables underneath it -- two consecutive
    failures, not a failure followed by a false success, is the only way to tell them apart."""
    driver = PostgresDriver(dsn=postgres_dsn, schema=uuid4().hex[:8])  # create_schema=False
    with pytest.raises(RegistryUnavailable):
        async with driver.connection():
            pass
    with pytest.raises(RegistryUnavailable):
        async with driver.connection():
            pass
    await driver.close()


async def test_connection_rolls_back_and_restores_on_a_failing_query(postgres_dsn: str) -> None:
    """`_checkout`'s exception path: a failing statement's own error must propagate unmasked, and
    the connection must still be usable for the next checkout afterward.

    Asserts the *specific* exception type, not just `psycopg.Error`: without the explicit
    `rollback()`, the connection is left `INTRANS`/`INERROR`, and the `finally`'s own
    `set_isolation_level` call then raises its own `ProgrammingError` -- also a `psycopg.Error`,
    so a bare `pytest.raises(psycopg.Error)` would pass either way and never catch the masking.
    """
    driver = PostgresDriver(dsn=postgres_dsn, schema=uuid4().hex[:8], create_schema=True)
    await driver.open()
    with pytest.raises(psycopg.errors.DivisionByZero):
        async with driver.connection() as conn:
            await conn.execute('SELECT 1/0')
    async with driver.connection() as conn:
        cur = await conn.execute('SELECT 1 AS ok')
        row = await cur.fetchone()
        assert row is not None and row['ok'] == 1
    await driver.close()


async def test_a_failing_commit_is_the_error_the_caller_sees(postgres_dsn: str) -> None:
    """When `commit()` fails, the connection is still inside the transaction, where psycopg refuses to
    change the isolation level; the restore in `finally` must not replace the commit's own error."""
    driver = PostgresDriver(dsn=postgres_dsn, schema=uuid4().hex[:8], create_schema=True)
    await driver.open()

    async def failing_commit() -> None:
        raise psycopg.OperationalError('connection lost during commit')

    with pytest.raises(psycopg.OperationalError, match='during commit'):
        async with driver.connection() as conn:
            await conn.execute('SELECT 1')
            conn.commit = failing_commit  # type: ignore[method-assign]
    async with driver.connection() as conn:
        cur = await conn.execute('SELECT 1 AS ok')
        row = await cur.fetchone()
        assert row is not None and row['ok'] == 1
    await driver.close()


async def test_a_connection_lost_inside_the_checkout_leaves_the_callers_error_in_place(postgres_dsn: str) -> None:
    """Rollback and the isolation-level restore are skipped on a closed connection, so the caller's
    own error comes out; the pool replaces the connection on return."""
    driver = PostgresDriver(dsn=postgres_dsn, schema=uuid4().hex[:8], create_schema=True)
    await driver.open()
    with pytest.raises(RuntimeError, match='boom'):
        async with driver.connection() as conn:
            await conn.close()
            raise RuntimeError('boom')
    with pytest.raises(psycopg.OperationalError):  # the commit on a connection closed inside the block
        async with driver.connection() as conn:
            await conn.close()
    async with driver.connection() as conn:
        cur = await conn.execute('SELECT 1 AS ok')
        row = await cur.fetchone()
        assert row is not None and row['ok'] == 1
    await driver.close()


async def test_checkout_forces_read_committed_regardless_of_pool_default(postgres_dsn: str) -> None:
    """The isolation-level override, not just `dict_row`: a borrowed pool configured for a stricter
    default must still see `READ COMMITTED` for the span of a checkout."""

    async def configure(conn: AsyncConnection[DictRow]) -> None:
        await conn.set_isolation_level(IsolationLevel.REPEATABLE_READ)

    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        postgres_dsn, kwargs={'row_factory': dict_row}, configure=configure, open=False
    )
    driver = PostgresDriver(pool=pool, schema=uuid4().hex[:8], create_schema=True)
    await driver.open()
    async with driver.connection() as conn:
        assert conn.isolation_level == IsolationLevel.READ_COMMITTED
    await driver.close()
    await pool.close()


async def test_checkout_forces_read_committed_so_a_concurrent_commit_stays_visible(postgres_dsn: str) -> None:
    """The override's actual effect, not just the property value -- the exact guarantee `register_conversation`'s
    claim-then-re-select needs: a statement issued after a concurrent commit must still see it, even
    though the pool defaults to REPEATABLE READ.

    Under REPEATABLE READ this would not hold: a transaction's snapshot is fixed at its first
    statement, so a later statement in the *same* transaction cannot see a commit another session
    made afterward -- which is exactly how `register_conversation`'s re-select can miss a concurrent winner's row
    despite `ON CONFLICT` correctly detecting the conflict (constraint checks are not bound by the
    snapshot). Verified by temporarily removing the `set_isolation_level` call in `_checkout` and
    re-running this test: it fails with `row is None`, confirming it would catch a regression.
    """

    async def configure(conn: AsyncConnection[DictRow]) -> None:
        await conn.set_isolation_level(IsolationLevel.REPEATABLE_READ)

    schema = uuid4().hex[:8]
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        postgres_dsn, kwargs={'row_factory': dict_row}, configure=configure, open=False
    )
    driver = PostgresDriver(pool=pool, schema=schema, create_schema=True)
    await driver.open()
    cid = uuid4().hex
    async with driver.connection() as conn:
        # The transaction's first statement: under REPEATABLE READ this is where the snapshot
        # would be fixed, the same as `register_conversation`'s own INSERT would be.
        await conn.execute(sql.SQL('SELECT 1 FROM {}').format(driver.table('conversations')))
        # A concurrent session commits a row this checkout's transaction has not seen yet.
        async with await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True) as other:
            await other.execute(
                sql.SQL(
                    'INSERT INTO {} (conversation_id, namespace_key, created_by, created_at, attrs) '
                    "VALUES (%s, 'ns', 'alice', now(), '{{}}'::jsonb)"
                ).format(driver.table('conversations')),
                (cid,),
            )
        cur = await conn.execute(
            sql.SQL('SELECT 1 FROM {} WHERE conversation_id = %s').format(driver.table('conversations')), (cid,)
        )
        row = await cur.fetchone()
    assert row is not None, "checkout's forced READ COMMITTED must see the concurrent commit"
    await driver.close()
    await pool.close()


async def test_maps_backend_errors_translates_psycopg_errors_only() -> None:
    """Every registry method carries the decorator, but exercises it only incidentally,
    mixed in with each method's own SQL; this test isolates it directly: a `psycopg.Error`
    becomes `RegistryUnavailable`, a normal return passes through untouched, and an unrelated
    exception is left alone."""

    @maps_backend_errors
    async def ok() -> str:
        return 'fine'

    @maps_backend_errors
    async def boom() -> None:
        raise psycopg.OperationalError('boom')

    @maps_backend_errors
    async def unrelated() -> None:
        raise ValueError('not ours to translate')

    assert await ok() == 'fine'
    with pytest.raises(RegistryUnavailable):
        await boom()
    with pytest.raises(ValueError):
        await unrelated()
