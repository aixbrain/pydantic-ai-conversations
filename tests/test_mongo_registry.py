"""What only a real mongod can prove, and what the conformance suite does not cover."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError, PyMongoError

from pydantic_ai_conversations.errors import ConversationNotFound, RegistryUnavailable
from pydantic_ai_conversations.mongo import MongoConversationRegistry
from pydantic_ai_conversations.registry import ConversationRegistry
from pydantic_ai_conversations.testing import RegistryContract

_NS = '[["user.tenant_id","t1"]]'


def _registry(mongo_url: str) -> MongoConversationRegistry:
    return MongoConversationRegistry(db_url=mongo_url, database=uuid4().hex)


async def test_register_returns_aware_utc_timestamps(mongo_url: str) -> None:
    """MongoDB hands back naive datetimes unless the client is `tz_aware`; the port promises aware."""
    registry = _registry(mongo_url)
    meta = await registry.register_conversation(_NS, uuid4().hex, created_by='alice')
    assert meta.created_at.tzinfo is not None
    offset = meta.created_at.utcoffset()
    assert offset is not None and offset.total_seconds() == 0
    await registry.close()


async def test_concurrent_registers_survive_the_duplicate_key_race(mongo_url: str) -> None:
    """A concurrent upsert on one `_id` can raise `DuplicateKeyError` even with `upsert=True`.

    Real MongoDB behaviour that no in-memory fake produces -- the registry must re-read rather
    than surface it.
    """
    registry = _registry(mongo_url)
    cid = uuid4().hex
    metas = await asyncio.gather(*(registry.register_conversation(_NS, cid, created_by=f'u{i}') for i in range(20)))
    assert len({m.created_by for m in metas}) == 1  # everyone got the winner's document
    await registry.close()


async def test_register_recovers_from_a_duplicate_key_error(mongo_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The race above is real but timing-dependent -- a local container rarely loses it inside one
    test run. Drive the same recovery path deterministically by forcing the upsert to raise."""
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='winner')

    async def boom(*_args: object, **_kwargs: object) -> None:
        raise DuplicateKeyError('boom')

    monkeypatch.setattr(AsyncCollection, 'find_one_and_update', boom)
    meta = await registry.register_conversation(_NS, cid, created_by='loser')
    assert meta.created_by == 'winner'
    await registry.close()


async def test_ping_reports_status_only(mongo_url: str) -> None:
    """The report is public (`/readyz` is unauthenticated), so it carries no counts."""
    registry = _registry(mongo_url)
    assert await registry.ping() == {'ok': True}
    await registry.register_conversation(_NS, uuid4().hex, created_by='alice')
    assert await registry.ping() == {'ok': True}
    await registry.close()


async def test_ping_detects_a_dead_server(mongo_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """`ping` must make a real server round-trip, not only the local readiness guard, so `/readyz`
    catches a server outage that happens after startup."""
    registry = _registry(mongo_url)
    await registry.register_conversation(_NS, uuid4().hex, created_by='alice')  # connect and build indexes

    async def boom(*_args: object, **_kwargs: object) -> None:
        raise PyMongoError('server down')

    monkeypatch.setattr(AsyncDatabase, 'command', boom)
    with pytest.raises(RegistryUnavailable):
        await registry.ping()
    await registry.close()


async def test_state_survives_reopen(mongo_url: str) -> None:
    database = uuid4().hex
    first = MongoConversationRegistry(db_url=mongo_url, database=database)
    cid = uuid4().hex
    await first.register_conversation(_NS, cid, created_by='alice', attrs={'source': 'web'})
    await first.close()

    second = MongoConversationRegistry(db_url=mongo_url, database=database)
    meta = await second.get(_NS, cid)
    assert meta is not None and meta.attrs['source'] == 'web'
    await second.close()


async def test_closed_registry_raises_unavailable_until_reopened(mongo_url: str) -> None:
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.close()
    with pytest.raises(RegistryUnavailable):
        await registry.get(_NS, cid)
    with pytest.raises(RegistryUnavailable):
        await registry.ping()
    await registry.open()
    assert await registry.get(_NS, cid) is not None
    await registry.close()


async def test_a_driver_error_surfaces_as_registry_unavailable() -> None:
    """The `PyMongoError` -> `RegistryUnavailable` mapping, on an address nothing answers."""
    registry = MongoConversationRegistry(db_url='mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=50', database='d')
    with pytest.raises(RegistryUnavailable):
        await registry.ping()
    await registry.close()


async def test_open_on_an_unreachable_address_raises_registry_unavailable() -> None:
    """`open()` is `ConversationsFeature.lifespan`'s startup probe (`feature.py`): it must map the
    same `PyMongoError` -> `RegistryUnavailable` as every other method, not leak a raw pymongo
    error to the app's lifespan."""
    registry = MongoConversationRegistry(db_url='mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=50', database='d')
    with pytest.raises(RegistryUnavailable):
        await registry.open()
    await registry.close()


async def test_a_foreign_or_deleted_id_is_not_found(mongo_url: str) -> None:
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    with pytest.raises(ConversationNotFound):
        await registry.register_conversation('[["user.tenant_id","t2"]]', cid, created_by='mallory')
    await registry.close()


async def test_stats_updates_move_the_sort_key(mongo_url: str) -> None:
    """`activity_at` is denormalized, so every write that touches `last_message_at` must set it
    too -- otherwise the list orders by a stale field."""
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    at = datetime(2026, 5, 1, tzinfo=UTC)
    await registry.update_message_stats(_NS, cid, last_message_at=at, message_count_delta=4)
    listed = await registry.list(_NS, limit=1)
    assert [m.conversation_id for m in listed] == [cid]
    meta = await registry.get(_NS, cid)
    assert meta is not None and meta.message_count == 4 and meta.last_message_at == at
    # `list`'s sort key is `activity_at`, which `get`'s `ConversationMeta` never surfaces -- read
    # the raw document to confirm the write kept the denormalized copy in sync, rather than
    # waiting for a `list` test to notice a stale field.
    raw = await registry._driver.conversations.find_one({'_id': cid})
    assert raw is not None and raw['activity_at'].replace(tzinfo=UTC) == at
    # A single increment from zero cannot tell an atomic `$inc` from an accidental `$set` to the
    # raw delta -- both leave 4. A second call with a different delta only lands on 7 if the first
    # write's 4 was actually accumulated rather than overwritten.
    await registry.update_message_stats(_NS, cid, last_message_at=at, message_count_delta=3)
    meta = await registry.get(_NS, cid)
    assert meta is not None and meta.message_count == 7
    await registry.close()


async def test_turns_keep_record_order(mongo_url: str) -> None:
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    for index in range(5):
        await registry.start_turn(_NS, cid, run_id=f'r{index}', author_user_id='alice')
    assert [t.run_id for t in await registry.list_turns(_NS, cid)] == [f'r{i}' for i in range(5)]
    await registry.close()


async def test_set_title_updates_and_rejects_an_unknown_or_foreign_id(mongo_url: str) -> None:
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.set_title(_NS, cid, 'Pump diagnosis')
    meta = await registry.get(_NS, cid)
    assert meta is not None and meta.title == 'Pump diagnosis'
    with pytest.raises(ConversationNotFound):
        await registry.set_title('[["user.tenant_id","t2"]]', cid, 'Stolen title')
    with pytest.raises(ConversationNotFound):
        await registry.set_title(_NS, uuid4().hex, 'Nobody home')
    await registry.close()


async def test_update_message_stats_rejects_an_unknown_id(mongo_url: str) -> None:
    registry = _registry(mongo_url)
    with pytest.raises(ConversationNotFound):
        await registry.update_message_stats(
            _NS, uuid4().hex, last_message_at=datetime(2026, 5, 1, tzinfo=UTC), message_count_delta=1
        )
    await registry.close()


async def test_soft_delete_returns_the_deleted_meta_and_rejects_a_repeat(mongo_url: str) -> None:
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    deleted = await registry.soft_delete(_NS, cid)
    assert deleted.deleted_at is not None
    assert await registry.get(_NS, cid) is None
    with pytest.raises(ConversationNotFound):
        await registry.soft_delete(_NS, cid)  # already gone
    await registry.close()


async def test_start_turn_rejects_a_soft_deleted_conversation(mongo_url: str) -> None:
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.soft_delete(_NS, cid)
    with pytest.raises(ConversationNotFound):
        await registry.start_turn(_NS, cid, run_id='r1', author_user_id='alice')
    await registry.close()


async def test_set_turn_phase_updates_the_named_run_and_ignores_an_unknown_one(mongo_url: str) -> None:
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.start_turn(_NS, cid, run_id='r1', author_user_id='alice')
    await registry.start_turn(_NS, cid, run_id='r2', author_user_id='alice')
    await registry.set_turn_phase(_NS, cid, run_id='r2', phase='failed')
    # A terminal hook can fire for a turn whose `start_turn` never landed -- a silent no-op.
    await registry.set_turn_phase(_NS, cid, run_id='never-recorded', phase='answered')
    by_run = {t.run_id: t.phase for t in await registry.list_turns(_NS, cid)}
    assert by_run == {'r1': 'started', 'r2': 'failed'}
    with pytest.raises(ConversationNotFound):
        await registry.set_turn_phase('[["user.tenant_id","t2"]]', cid, run_id='r1', phase='answered')
    await registry.close()


async def test_list_turns_rejects_an_unknown_conversation(mongo_url: str) -> None:
    registry = _registry(mongo_url)
    with pytest.raises(ConversationNotFound):
        await registry.list_turns(_NS, uuid4().hex)
    await registry.close()


async def test_turns_are_ordered_by_stamped_time_over_insertion_order(mongo_url: str) -> None:
    """`list_turns` sorts by `(at, _id)`, not insertion or `_id` order: a turn recorded with an
    earlier stamped time must still sort first even if it lands after another turn's `ObjectId`."""
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.start_turn(_NS, cid, run_id='second', author_user_id='alice')
    later = datetime.now(UTC) + timedelta(days=1)
    await registry._driver.turns.update_one({'run_id': 'second'}, {'$set': {'at': later}})
    await registry.start_turn(_NS, cid, run_id='first', author_user_id='alice')
    assert [t.run_id for t in await registry.list_turns(_NS, cid)] == ['first', 'second']
    await registry.close()


async def test_turns_with_equal_stamped_time_break_ties_by_object_id(mongo_url: str) -> None:
    """No SQL-style autoincrement `seq` exists here; the `(at, _id)` sort relies on `ObjectId`
    ordering to break a tie -- exercise it directly rather than trust it holds."""
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.start_turn(_NS, cid, run_id='r1', author_user_id='alice')
    await registry.start_turn(_NS, cid, run_id='r2', author_user_id='alice')
    same_time = datetime(2026, 1, 1, tzinfo=UTC)
    await registry._driver.turns.update_many({'conversation_id': cid}, {'$set': {'at': same_time}})
    assert [t.run_id for t in await registry.list_turns(_NS, cid)] == ['r1', 'r2']
    await registry.close()


class TestMongoConversationRegistryConformance(RegistryContract):
    @pytest.fixture
    def registry(self, mongo_url: str) -> ConversationRegistry:
        return MongoConversationRegistry(db_url=mongo_url, database=uuid4().hex)


async def test_paging_a_filtered_list_stays_inside_the_filter(mongo_url: str) -> None:
    """`attrs` and `page_after` together: the cursor is resolved inside the filtered set, so one
    naming an excluded conversation ends the list -- the same answer an unknown cursor gets."""
    registry = _registry(mongo_url)
    web = [uuid4().hex for _ in range(3)]
    twilio = uuid4().hex
    for index, cid in enumerate([web[0], twilio, web[1], web[2]]):
        source = 'twilio' if cid == twilio else 'web'
        await registry.register_conversation(_NS, cid, created_by='alice', attrs={'source': source})
        await registry.update_message_stats(
            _NS,
            cid,
            last_message_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
            message_count_delta=1,
        )

    newest_first = [web[2], web[1], web[0]]
    page1 = await registry.list(_NS, attrs={'source': 'web'}, limit=2)
    assert [m.conversation_id for m in page1] == newest_first[:2]
    page2 = await registry.list(_NS, attrs={'source': 'web'}, page_after=page1[-1].conversation_id)
    assert [m.conversation_id for m in page2] == newest_first[2:]
    assert await registry.list(_NS, attrs={'source': 'web'}, page_after=twilio) == []
    await registry.close()


async def test_attrs_keys_are_matched_literally_not_as_a_path(mongo_url: str) -> None:
    """A key holding a dot filters on itself -- pairs are bound values, never a field path."""
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice', attrs={'a.b': 'v', 'a': 'no'})
    assert [m.conversation_id for m in await registry.list(_NS, attrs={'a.b': 'v'})] == [cid]
    assert await registry.list(_NS, attrs={'a.b.c': 'v'}) == []
    await registry.close()


async def test_purge_of_a_never_registered_id_is_a_noop(mongo_url: str) -> None:
    """The conformance suite only exercises a foreign purge against an id some namespace already
    owns (`doc['namespace_key'] != namespace_key`); coverage.py folds that into the same branch as
    a wholly unknown `_id` because both sit behind one `or` on one `if`. Drive `doc is None`
    directly so the early return is proven, not merely arithmetically covered."""
    registry = _registry(mongo_url)
    assert await registry.purge(_NS, uuid4().hex) is False
    await registry.close()


async def test_purge_excludes_the_conversation_from_list_and_clears_its_fields(mongo_url: str) -> None:
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice', attrs={'source': 'web'})
    await registry.set_title(_NS, cid, 'Pump diagnosis')
    assert await registry.purge(_NS, cid) is True
    assert [m.conversation_id for m in await registry.list(_NS)] == []
    raw = await registry._driver.conversations.find_one({'_id': cid})
    assert raw is not None
    assert (raw['purged'], raw['title'], raw['attrs']) == (True, None, [])
    await registry.close()


async def test_purge_preserves_an_earlier_soft_delete_timestamp(mongo_url: str) -> None:
    """The tombstone's `deleted_at` keeps an existing timestamp (`doc['deleted_at'] if ... is not
    None else _now()`): a conversation already soft-deleted keeps its original timestamp rather
    than having purge bump it forward."""
    registry = _registry(mongo_url)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    deleted = await registry.soft_delete(_NS, cid)
    assert await registry.purge(_NS, cid) is True
    raw = await registry._driver.conversations.find_one({'_id': cid})
    assert raw is not None
    assert raw['deleted_at'].replace(tzinfo=UTC) == deleted.deleted_at
    await registry.close()
