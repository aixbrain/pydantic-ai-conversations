"""The shipped SQLite registry passes the same conformance suite as the in-memory reference --
plus what only a durable backend can promise: state, soft-deletes, and purge tombstones survive
close + reopen."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from pydantic_ai_conversations.errors import ConversationNotFound, RegistryUnavailable
from pydantic_ai_conversations.registry import ConversationRegistry
from pydantic_ai_conversations.sqlite import SqliteConversationRegistry
from pydantic_ai_conversations.testing import RegistryContract

_NS = '[["user.tenant_id","t1"]]'


class TestSqliteConversationRegistryConformance(RegistryContract):
    @pytest.fixture
    def registry(self, tmp_path: Path) -> ConversationRegistry:
        return SqliteConversationRegistry(tmp_path / 'conversations.db')


async def test_state_survives_reopen_from_the_same_file(tmp_path: Path) -> None:
    database = tmp_path / 'conversations.db'
    first = SqliteConversationRegistry(database)
    cid, deleted_cid = uuid4().hex, uuid4().hex
    await first.register_conversation(_NS, cid, created_by='alice', attrs={'source': 'web'})
    await first.set_title(_NS, cid, 'Pump diagnosis')
    await first.start_turn(_NS, cid, run_id='r1', author_user_id='alice')
    await first.register_conversation(_NS, deleted_cid, created_by='alice')
    await first.soft_delete(_NS, deleted_cid)
    await first.close()

    second = SqliteConversationRegistry(database)
    meta = await second.get(_NS, cid)
    assert meta is not None and (meta.title, meta.attrs['source']) == ('Pump diagnosis', 'web')
    assert [t.run_id for t in await second.list_turns(_NS, cid)] == ['r1']
    # The soft-delete survives too: hidden, and still unrevivable.
    assert await second.get(_NS, deleted_cid) is None
    with pytest.raises(ConversationNotFound):
        await second.register_conversation(_NS, deleted_cid, created_by='alice')


async def test_purge_tombstone_survives_reopen(tmp_path: Path) -> None:
    database = tmp_path / 'conversations.db'
    first = SqliteConversationRegistry(database)
    cid = uuid4().hex
    await first.register_conversation(_NS, cid, created_by='alice')
    assert await first.purge(_NS, cid) is True
    await first.close()

    second = SqliteConversationRegistry(database)
    with pytest.raises(ConversationNotFound):
        await second.register_conversation(_NS, cid, created_by='alice')  # dead forever, restart included


async def test_closed_registry_raises_unavailable_until_reopened(tmp_path: Path) -> None:
    registry = SqliteConversationRegistry(tmp_path / 'conversations.db')
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.close()
    with pytest.raises(RegistryUnavailable):
        await registry.get(_NS, cid)
    with pytest.raises(RegistryUnavailable):
        await registry.ping()
    await registry.open()
    assert await registry.get(_NS, cid) is not None


async def test_a_driver_error_surfaces_as_registry_unavailable(tmp_path: Path) -> None:
    """The `sqlite3.Error` -> `RegistryUnavailable` mapping every backend owes `errors.py`.

    A directory is not a database file, so the driver's own `connect` raises.
    """
    registry = SqliteConversationRegistry(tmp_path)
    with pytest.raises(RegistryUnavailable):
        await registry.ping()
    with pytest.raises(RegistryUnavailable):
        await registry.open()


async def test_open_creates_the_schema_before_the_first_request(tmp_path: Path) -> None:
    database = tmp_path / 'conversations.db'
    registry = SqliteConversationRegistry(database)
    await registry.open()
    assert await registry.ping() == {'ok': True}


async def test_close_before_first_use_marks_the_registry_unavailable(tmp_path: Path) -> None:
    """The feature's lifespan closes whatever it opened, including a registry no request touched."""
    registry = SqliteConversationRegistry(tmp_path / 'conversations.db')
    await registry.close()  # nothing was ever connected -- must not raise
    with pytest.raises(RegistryUnavailable):
        await registry.ping()


async def test_paging_a_filtered_list_stays_inside_the_filter(tmp_path: Path) -> None:
    """`attrs` and `page_after` together: the cursor is resolved inside the filtered set.

    A cursor naming a conversation the filter excludes ends the list, the same answer an unknown
    cursor gets -- `InMemoryConversationRegistry` looks the cursor up in its filtered list and
    lands there too.
    """
    registry = SqliteConversationRegistry(tmp_path / 'conversations.db')
    web = [uuid4().hex for _ in range(3)]
    twilio = uuid4().hex
    for index, cid in enumerate([web[0], twilio, web[1], web[2]]):
        source = 'twilio' if cid == twilio else 'web'
        await registry.register_conversation(_NS, cid, created_by='alice', attrs={'source': source})
        await registry.update_message_stats(
            _NS, cid, last_message_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index), message_count_delta=1
        )

    newest_first = [web[2], web[1], web[0]]  # twilio sorts between web[0] and web[1], and is filtered out
    page1 = await registry.list(_NS, attrs={'source': 'web'}, limit=2)
    assert [meta.conversation_id for meta in page1] == newest_first[:2]

    page2 = await registry.list(_NS, attrs={'source': 'web'}, page_after=page1[-1].conversation_id)
    assert [meta.conversation_id for meta in page2] == newest_first[2:]

    assert await registry.list(_NS, attrs={'source': 'web'}, page_after=twilio) == []


async def test_attrs_keys_are_matched_literally_not_as_a_path(tmp_path: Path) -> None:
    """A key holding a dot or a quote filters on itself -- it is a bound value, never JSON syntax."""
    registry = SqliteConversationRegistry(tmp_path / 'conversations.db')
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice', attrs={'a.b"c': 'v', 'a': 'no'})
    assert [meta.conversation_id for meta in await registry.list(_NS, attrs={'a.b"c': 'v'})] == [cid]
    assert await registry.list(_NS, attrs={'a.b': 'v'}) == []


async def test_a_file_from_before_the_lineage_column_is_migrated_on_connect(tmp_path: Path) -> None:
    """`CREATE TABLE IF NOT EXISTS` leaves an older `turns` table as it is, so the registry adds the
    missing column itself the first time it connects."""
    database = tmp_path / 'conversations.db'
    with sqlite3.connect(database) as raw:
        raw.executescript(
            'CREATE TABLE turns (seq INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, '
            'run_id TEXT NOT NULL, author_user_id TEXT NOT NULL, at REAL NOT NULL, '
            "phase TEXT NOT NULL DEFAULT 'started');"
        )
    registry = SqliteConversationRegistry(database)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.start_turn(_NS, cid, run_id='r1', author_user_id='alice')
    await registry.start_turn(_NS, cid, run_id='r2', author_user_id='alice', previous_run_id='r1')
    assert [t.previous_run_id for t in await registry.list_turns(_NS, cid)] == [None, 'r1']
    await registry.close()
