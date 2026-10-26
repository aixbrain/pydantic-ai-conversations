"""The PostgreSQL feedback store passes the contract against a live server."""

from uuid import uuid4

import psycopg
import pytest
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from pydantic_ai_conversations.feedback import TurnFeedbackStore
from pydantic_ai_conversations.postgres import FEEDBACK_SCHEMA_SQL, PostgresTurnFeedbackStore
from pydantic_ai_conversations.testing import TurnFeedbackStoreContract


class TestPostgresTurnFeedbackStoreConformance(TurnFeedbackStoreContract):
    @pytest.fixture
    def feedback_store(self, postgres_dsn: str) -> TurnFeedbackStore:
        return PostgresTurnFeedbackStore(dsn=postgres_dsn, schema=uuid4().hex[:8], create_schema=True)


def test_ddl_ships_with_a_schema_placeholder() -> None:
    assert '{schema}.feedback' in FEEDBACK_SCHEMA_SQL


async def test_missing_table_is_named_at_open(postgres_dsn: str) -> None:
    store = PostgresTurnFeedbackStore(dsn=postgres_dsn, schema=uuid4().hex[:8])
    with pytest.raises(RuntimeError, match='feedback.sql'):
        await store.open()
    await store.close()


def test_pool_or_dsn_exactly_one() -> None:
    with pytest.raises(ValueError, match='exactly one'):
        PostgresTurnFeedbackStore()


def test_transient_errors_are_the_drivers() -> None:
    assert PostgresTurnFeedbackStore.transient_errors == (psycopg.OperationalError,)


async def test_open_is_idempotent_and_close_leaves_a_borrowed_pool_open(postgres_dsn: str) -> None:
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        postgres_dsn, open=False, kwargs={'row_factory': dict_row}
    )
    store = PostgresTurnFeedbackStore(pool=pool, schema=uuid4().hex[:8], create_schema=True)
    await store.open()
    await store.open()
    await store.close()
    assert not pool.closed
    assert (await store.ping()) == {'ok': True}
    await pool.close()


async def test_an_owned_pool_can_be_reopened_after_close(postgres_dsn: str) -> None:
    """A closed psycopg pool cannot be reopened, so the store makes a new one, as the registry does."""
    store = PostgresTurnFeedbackStore(dsn=postgres_dsn, schema=uuid4().hex[:8], create_schema=True)
    await store.open()
    await store.close()
    await store.open()
    assert (await store.ping()) == {'ok': True}
    await store.close()
