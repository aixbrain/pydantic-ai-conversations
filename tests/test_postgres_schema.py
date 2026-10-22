"""The shipped DDL applies cleanly into a named schema and is idempotent."""

from uuid import uuid4

import psycopg
from psycopg import AsyncConnection, sql
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from pydantic_ai_conversations.postgres._schema import SCHEMA_SQL, TABLES, create_schema, schema_statements

_SCHEMA = 'chatns'


async def test_the_ddl_applies_into_a_named_schema_and_is_idempotent(postgres_dsn: str) -> None:
    async with await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True) as conn:
        await conn.execute(schema_statements(_SCHEMA))
        await conn.execute(schema_statements(_SCHEMA))  # second run must be a no-op, not an error
        for table in TABLES:
            cur = await conn.execute('SELECT to_regclass(%s)', (f'{_SCHEMA}.{table}',))
            row = await cur.fetchone()
            assert row is not None and row[0] is not None, f'{table} missing after applying the DDL'


async def test_a_schema_name_is_quoted_not_interpolated(postgres_dsn: str) -> None:
    """The schema name reaches SQL as an identifier, so an awkward one cannot reshape the DDL."""
    awkward = 'weird schema"name'
    async with await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True) as conn:
        await conn.execute(schema_statements(awkward))
        quoted = f'"{awkward.replace(chr(34), chr(34) * 2)}".conversations'
        cur = await conn.execute('SELECT to_regclass(%s)', (quoted,))
        row = await cur.fetchone()
        assert row is not None and row[0] is not None


async def test_the_shipped_text_applies_as_a_migration_tool_would_and_the_attrs_default_fires(
    postgres_dsn: str,
) -> None:
    """A migration tool never calls `schema_statements` -- it substitutes `{schema}` itself (a
    plain find-and-replace) and runs the rest of the file verbatim, no Python string-formatting
    involved. Reproduce exactly that, independent of `_schema.py`'s own substitution, so a
    regression to `str.format`-shaped placeholders (which would demand doubled braces this file
    must not carry) is caught here too.

    Also exercises the branch no other test reaches: an INSERT that does not name `attrs`, so the
    column's `DEFAULT` clause itself fires -- not just parses -- rather than a caller-supplied
    value masking a broken default.
    """
    schema = 'chatns_raw'
    ddl = SCHEMA_SQL.replace('{schema}', schema)
    async with await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True) as conn:
        # `ddl` is computed with `str.replace`, so it is a plain `str`, not a `LiteralString`; safe
        # regardless, since it is our own shipped file's text, not caller input.
        await conn.execute(ddl)  # pyright: ignore[reportCallIssue, reportArgumentType]
        for table in TABLES:
            cur = await conn.execute('SELECT to_regclass(%s)', (f'{schema}.{table}',))
            row = await cur.fetchone()
            assert row is not None and row[0] is not None, f'{table} missing after applying the raw file'
        cid = uuid4().hex
        await conn.execute(
            sql.SQL(
                'INSERT INTO {}.conversations (conversation_id, namespace_key, created_by, created_at) '
                "VALUES (%s, 'ns', 'alice', now())"
            ).format(sql.Identifier(schema)),
            (cid,),
        )
        cur = await conn.execute(
            sql.SQL('SELECT attrs FROM {}.conversations WHERE conversation_id = %s').format(sql.Identifier(schema)),
            (cid,),
        )
        row = await cur.fetchone()
        assert row is not None and row[0] == {}, 'attrs DEFAULT did not fire for an INSERT that omitted the column'


def test_the_sql_file_ships_beside_the_module() -> None:
    """Apps paste the file into a migration; it must be in the wheel, not only in the repo."""
    from importlib.resources import files

    text = files('pydantic_ai_conversations.postgres').joinpath('schema.sql').read_text()
    assert 'CREATE TABLE IF NOT EXISTS' in text and '{schema}' in text


async def test_create_schema_applies_the_ddl_through_a_pool(postgres_dsn: str) -> None:
    """`create_schema` is the escape hatch for tests and local dev, used through a pool rather
    than a bare connection -- exercising the one code path this module ships that the other tests
    do not touch."""
    schema = 'chatns_pool'
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        postgres_dsn, kwargs={'row_factory': dict_row}, open=False
    )
    async with pool:
        await pool.open(wait=True)
        await create_schema(pool, schema)
        async with pool.connection() as conn:
            for table in TABLES:
                cur = await conn.execute('SELECT to_regclass(%s)', (f'{schema}.{table}',))
                row = await cur.fetchone()
                assert row is not None and row['to_regclass'] is not None, f'{table} missing after create_schema'
