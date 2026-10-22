"""The registry's DDL, shipped rather than executed.

Postgres deployments usually establish schema in an ordered step before the app process -- Alembic
in an entrypoint, or init SQL baked into the database image. Runtime DDL from the app would bypass
that step, needs privileges the app's database user often lacks, and lets concurrent replicas race
`CREATE TABLE IF NOT EXISTS` into a catalog deadlock. So this module hands the statements over;
`create_schema` exists for tests and local development.
"""

from __future__ import annotations

from importlib.resources import files
from typing import cast

from psycopg import AsyncConnection, sql
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

TABLES = ('conversations', 'turns')
"""What `_driver.py`'s startup probe looks for, and what an app's migration must have created."""

# `__package__` is `str | None` in the stubs (a module run as `__main__` has none), but a module
# that imports normally always has one; `files` needs the plain `str` this always is at runtime.
SCHEMA_SQL = files(cast(str, __package__)).joinpath('schema.sql').read_text(encoding='utf-8')
"""The DDL as shipped, with `{schema}` still a placeholder. Paste it into your own migration."""


def schema_statements(schema: str) -> sql.SQL:
    """`SCHEMA_SQL` with the schema name substituted as a quoted identifier.

    Substitution is a plain `str.replace` of the `{schema}` placeholder, not `sql.SQL(...).format`
    -- the same thing a migration tool's find-and-replace does when a human pastes the file in.
    That keeps `schema.sql` ordinary SQL: nothing else in the file needs brace-escaping to survive
    Python's own formatting mini-language, on either that path or this one. The name itself still
    goes through `sql.Identifier`, never bare string interpolation, so a schema whose name holds a
    quote or a dot cannot reshape the DDL.
    """
    quoted = sql.Identifier(schema).as_string(None)
    # `sql.SQL` wants a `LiteralString`; the substituted text is a plain `str` because
    # `SCHEMA_SQL` is read from a file at import time, not written as a literal, so pyright cannot
    # narrow it. Safe regardless: `SCHEMA_SQL` is our own shipped file, not caller input.
    return sql.SQL(SCHEMA_SQL.replace('{schema}', quoted))  # pyright: ignore[reportArgumentType]


async def create_schema(pool: AsyncConnectionPool[AsyncConnection[DictRow]], schema: str = 'public') -> None:
    """Apply the shipped DDL. For tests and local development -- production applies it through the
    app's own migration tool, which is why the registry does not run DDL by default."""
    async with pool.connection() as conn:
        await conn.execute(schema_statements(schema))
        await conn.commit()
