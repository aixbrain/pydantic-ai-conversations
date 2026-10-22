"""The container fixtures hand back a server that actually answers."""

from typing import Any

import psycopg
from pymongo import AsyncMongoClient


async def test_mongo_container_answers(mongo_url: str) -> None:
    client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(mongo_url)
    try:
        assert (await client.admin.command('ping'))['ok'] == 1.0
    finally:
        await client.close()


async def test_postgres_container_answers(postgres_dsn: str) -> None:
    async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
        cur = await conn.execute('SELECT 1 AS one')
        row = await cur.fetchone()
        assert row is not None and row[0] == 1
