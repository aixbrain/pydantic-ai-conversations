"""The MongoDB feedback store passes the contract against a live mongod."""

from typing import Any
from uuid import uuid4

import pytest
from pymongo import AsyncMongoClient
from pymongo.errors import PyMongoError

from pydantic_ai_conversations.feedback import TurnFeedbackStore
from pydantic_ai_conversations.mongo import MongoTurnFeedbackStore
from pydantic_ai_conversations.mongo._driver import STEP_STORE_TRANSIENT_ERRORS
from pydantic_ai_conversations.testing import TurnFeedbackStoreContract


class TestMongoTurnFeedbackStoreConformance(TurnFeedbackStoreContract):
    @pytest.fixture
    def feedback_store(self, mongo_url: str) -> TurnFeedbackStore:
        return MongoTurnFeedbackStore(db_url=mongo_url, database=uuid4().hex)


def test_transient_errors_are_the_kits() -> None:
    """The kit's set, not every `PyMongoError`: a duplicate key is not retryable."""
    assert MongoTurnFeedbackStore.transient_errors == STEP_STORE_TRANSIENT_ERRORS


def test_client_or_url_exactly_one() -> None:
    with pytest.raises(ValueError, match='exactly one'):
        MongoTurnFeedbackStore(database='x')


async def test_open_creates_the_indexes_and_close_closes_an_owned_client(mongo_url: str) -> None:
    """With `db_url=` the store owns its client: `close()` shuts it, and a further call fails."""
    store = MongoTurnFeedbackStore(db_url=mongo_url, database=uuid4().hex)
    await store.open()
    assert (await store.ping()) == {'ok': True}
    await store.close()
    with pytest.raises(PyMongoError):
        await store.ping()


async def test_close_leaves_a_borrowed_client_open(mongo_url: str) -> None:
    client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(mongo_url)
    store = MongoTurnFeedbackStore(client=client, database=uuid4().hex)
    await store.close()
    assert (await store.ping()) == {'ok': True}
    await client.close()


async def test_an_owned_client_can_be_reopened_after_close(mongo_url: str) -> None:
    """A second lifespan (tests, a restart of the feature) opens the store again."""
    store = MongoTurnFeedbackStore(db_url=mongo_url, database=uuid4().hex)
    await store.open()
    await store.close()
    await store.open()
    record = await store.upsert(
        kind='r', namespace_key='ns', conversation_id='c', run_id=uuid4().hex, author_user_id='a', data={}
    )
    assert record.kind == 'r'
    await store.close()
