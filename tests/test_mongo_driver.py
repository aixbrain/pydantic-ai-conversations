"""Connection ownership, index setup, and the error contract -- the driver's own concerns."""

from typing import Any
from uuid import uuid4

import pytest
from pymongo import AsyncMongoClient
from pymongo.errors import PyMongoError

from pydantic_ai_conversations.errors import RegistryUnavailable
from pydantic_ai_conversations.mongo._driver import MongoDriver, maps_backend_errors


def test_exactly_one_connection_source_is_required() -> None:
    """A `ValueError` from argument validation, before any connection is attempted -- no server
    needed, so this does not take the `mongo_url` fixture and start a container for it."""
    with pytest.raises(ValueError):
        MongoDriver(database='d')
    with pytest.raises(ValueError):
        MongoDriver(client=AsyncMongoClient('mongodb://localhost/'), db_url='mongodb://localhost/', database='d')


async def test_a_borrowed_client_survives_close(mongo_url: str) -> None:
    """`close()` shuts the driver, not a client it did not create -- the owner closes that."""
    client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(mongo_url)
    driver = MongoDriver(client=client, database=uuid4().hex)
    await driver.open()
    await driver.close()
    with pytest.raises(RegistryUnavailable):
        await driver.ready()
    assert (await client.admin.command('ping'))['ok'] == 1.0  # still usable
    await client.close()


async def test_an_owned_client_is_closed_and_rebuilt_on_reopen(mongo_url: str) -> None:
    driver = MongoDriver(db_url=mongo_url, database=uuid4().hex)
    await driver.open()
    await driver.close()
    with pytest.raises(RegistryUnavailable):
        await driver.ready()
    await driver.open()  # a client it created, it can recreate
    await driver.ready()
    await driver.close()


async def test_indexes_are_created_once(mongo_url: str) -> None:
    driver = MongoDriver(db_url=mongo_url, database=uuid4().hex)
    await driver.ready()
    await driver.ready()  # idempotent, and the second call issues nothing
    names = [index['name'] async for index in await driver.conversations.list_indexes()]
    assert any('namespace_key' in name for name in names)
    await driver.close()


async def test_maps_backend_errors_translates_pymongo_errors_only() -> None:
    """Exercised directly, apart from any registry method: a `PyMongoError` becomes
    `RegistryUnavailable`, a normal return passes through untouched."""

    @maps_backend_errors
    async def ok() -> str:
        return 'fine'

    @maps_backend_errors
    async def boom() -> None:
        raise PyMongoError('boom')

    assert await ok() == 'fine'
    with pytest.raises(RegistryUnavailable):
        await boom()
