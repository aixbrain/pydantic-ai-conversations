"""The SQLite feedback store passes the contract and survives reopen."""

import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from pydantic_ai_conversations.feedback import TurnFeedbackStore
from pydantic_ai_conversations.sqlite import SqliteTurnFeedbackStore
from pydantic_ai_conversations.testing import TurnFeedbackStoreContract

_NS = '[["user.tenant_id","t1"]]'


class TestSqliteTurnFeedbackStoreConformance(TurnFeedbackStoreContract):
    @pytest.fixture
    def feedback_store(self, tmp_path: Path) -> TurnFeedbackStore:
        return SqliteTurnFeedbackStore(tmp_path / 'app.db')


async def test_entries_survive_reopen(tmp_path: Path) -> None:
    database = tmp_path / 'app.db'
    first = SqliteTurnFeedbackStore(database)
    cid, run_id = uuid4().hex, uuid4().hex
    await first.upsert(
        kind='report', namespace_key=_NS, conversation_id=cid, run_id=run_id, author_user_id='alice', data={'k': 'v'}
    )
    await first.close()

    second = SqliteTurnFeedbackStore(database)
    (row,) = await second.list(kind='report', namespace_key=_NS, conversation_id=cid)
    assert row.data == {'k': 'v'}
    await second.close()


def test_transient_errors_are_the_kits() -> None:
    assert SqliteTurnFeedbackStore.transient_errors == (sqlite3.OperationalError, OSError)
