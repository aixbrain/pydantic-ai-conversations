"""The shipped MongoDB step store: the two extras `stores.py` discovers on the harness store."""

import json
from typing import Any
from uuid import uuid4

import pytest
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, UserPromptPart
from pydantic_ai_app_factory import AgentError
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import PyMongoError

from pydantic_ai_conversations.mongo import MongoConversationStepStore
from pydantic_ai_conversations.mongo._store import _REFS_PER_QUERY
from pydantic_ai_conversations.stores import (
    PreparingStepStore,
    SupportsPing,
    SupportsRunDeletion,
    SupportsTransientErrors,
)


def _store(mongo_url: str) -> MongoConversationStepStore:
    return MongoConversationStepStore(db_url=mongo_url, database=uuid4().hex)


def _prompt(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


async def _save(store: MongoConversationStepStore, run_id: str, messages: list[ModelMessage]) -> None:
    await store.register_run(RunRecord(run_id=run_id, conversation_id=f'c-{run_id}'))
    await store.save_snapshot(ContinuableSnapshot(run_id=run_id, step_index=0, messages=messages))


async def _stored_texts(store: MongoConversationStepStore) -> list[str]:
    return [str(doc['message']) async for doc in store._db['snapshot_messages'].find({})]


def test_the_store_satisfies_all_three_optional_protocols(mongo_url: str) -> None:
    store = _store(mongo_url)
    assert isinstance(store, SupportsRunDeletion) and isinstance(store, SupportsPing)
    assert isinstance(store, SupportsTransientErrors)


async def test_ping_reports_health_before_anything_is_written(mongo_url: str) -> None:
    assert (await _store(mongo_url).ping())['ok'] is True


async def test_max_snapshots_per_run_reaches_the_base_store(mongo_url: str) -> None:
    """`max_snapshots_per_run` must reach `MongoStepStore.__init__` -- without pass-through, this
    subclass could not prune snapshots at all, even though the shipped store is supposed to be a
    strict superset of the harness one."""
    store = MongoConversationStepStore(db_url=mongo_url, database=uuid4().hex, max_snapshots_per_run=1)
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[]))
    await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=[]))
    assert len(await store.list_snapshots(run_id='r1')) == 1


async def test_delete_runs_removes_the_run_and_ignores_an_empty_id_list(mongo_url: str) -> None:
    store = _store(mongo_url)
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    assert await store.get_run(run_id='r1') is not None

    await store.delete_runs(run_ids=[])  # nothing to do, and no statement issued
    assert await store.get_run(run_id='r1') is not None

    await store.delete_runs(run_ids=['r1'])
    assert await store.get_run(run_id='r1') is None
    await store.delete_runs(run_ids=['r1'])  # idempotent: purge may be retried


async def test_delete_runs_clears_the_replay_ledger_so_the_run_id_can_be_written_again(mongo_url: str) -> None:
    """The harness keeps snapshot idempotency keys in a ledger that survives snapshot pruning, so
    a purge that skipped it would leave the run's ids in the database and suppress every later
    save under a recurring key."""
    store = _store(mongo_url)
    snapshot = ContinuableSnapshot(run_id='r1', step_index=0, messages=[], idempotency_key='0:0:complete')
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(snapshot)

    await store.delete_runs(run_ids=['r1'])

    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(snapshot)
    assert len(await store.list_snapshots(run_id='r1')) == 1


async def test_delete_runs_leaves_the_run_when_a_later_delete_fails(
    mongo_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`runs` must be deleted last, so a failure partway through leaves the run discoverable and a
    retry can finish the purge (the feature re-finds the run via `list_runs`). Deleting `runs`
    first would orphan its dependents for good."""
    store = _store(mongo_url)
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[]))
    assert await store.get_run(run_id='r1') is not None

    original = AsyncCollection.delete_many
    calls = 0

    async def flaky(self: AsyncCollection[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PyMongoError('delete failed')
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(AsyncCollection, 'delete_many', flaky)
    with pytest.raises(PyMongoError):
        await store.delete_runs(run_ids=['r1'])

    assert await store.get_run(run_id='r1') is not None  # survives for a retry


async def test_delete_runs_deletes_the_runs_stored_messages(mongo_url: str) -> None:
    """`MongoStepStore` keeps snapshot messages in their own collection, which no harness call
    clears, so a purge that left it alone would keep the conversation's text."""
    store = _store(mongo_url)
    await _save(store, 'r1', [_prompt('first')])
    await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=[_prompt('second')]))

    await store.delete_runs(run_ids=['r1'])

    assert await store.get_run(run_id='r1') is None
    assert await store._db['snapshots'].count_documents({}) == 0
    assert await _stored_texts(store) == []


async def test_delete_runs_keeps_a_message_another_run_still_references(mongo_url: str) -> None:
    """Identical messages are stored once, so the other run's snapshot must stay readable."""
    store = _store(mongo_url)
    shared = _prompt('shared')
    await _save(store, 'r1', [shared, _prompt('own')])
    await _save(store, 'r2', [shared])

    await store.delete_runs(run_ids=['r1'])

    assert [doc for doc in await _stored_texts(store) if 'own' in doc] == []
    snapshot = await store.latest_snapshot(run_id='r2')
    assert snapshot is not None and snapshot.messages == [shared]


async def test_delete_runs_checks_every_message_of_a_long_run(mongo_url: str) -> None:
    """A run with more messages than one query holds is checked batch by batch, and each batch must
    reach the server as a valid pattern."""
    store = _store(mongo_url)
    messages: list[ModelMessage] = [_prompt(f'message {i}') for i in range(_REFS_PER_QUERY + 1)]
    await _save(store, 'r1', messages)
    await _save(store, 'r2', [messages[-1]])

    await store.delete_runs(run_ids=['r1'])

    assert len(await _stored_texts(store)) == 1
    snapshot = await store.latest_snapshot(run_id='r2')
    assert snapshot is not None and snapshot.messages == [messages[-1]]


async def test_delete_runs_ignores_message_keys_in_a_history_held_inline(mongo_url: str) -> None:
    """Snapshots written before messages were stored once hold the history as a JSON list. They
    reference no stored message, even when their text contains a message key, and must not stop
    the purge."""
    store = _store(mongo_url)
    await _save(store, 'r1', [_prompt('own')])
    doc = await store._db['snapshots'].find_one({'run_id': 'r1'})
    assert doc is not None
    key = json.loads(str(doc['messages']))['message_refs'][0]
    inline = ModelMessagesTypeAdapter.dump_json([_prompt(key)]).decode()
    await store._db['snapshots'].insert_many(
        [{'run_id': run_id, 'seq': 0, 'step_index': 0, 'messages': inline} for run_id in ('r1', 'r2')]
    )

    await store.delete_runs(run_ids=['r1'])

    assert await store._db['snapshots'].count_documents({'run_id': 'r1'}) == 0
    assert await _stored_texts(store) == []


async def test_an_unreachable_server_is_reported_as_temporary() -> None:
    unreachable = MongoConversationStepStore(
        db_url='mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=200', database=uuid4().hex
    )
    with pytest.raises(AgentError) as exc_info:
        await PreparingStepStore(unreachable).get_run(run_id='r1')
    assert (exc_info.value.code, exc_info.value.retryable) == ('conversations/unavailable', True)


async def test_a_duplicate_run_id_is_not_reported_as_temporary(mongo_url: str) -> None:
    """The harness turns the server's duplicate-key error into `ValueError`, and a `DuplicateKeyError`
    is no connection failure either, so neither is retried."""
    seam = PreparingStepStore(_store(mongo_url))
    await seam.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    with pytest.raises(ValueError, match='already in the store'):
        await seam.register_run(RunRecord(run_id='r1', conversation_id='c1'))
