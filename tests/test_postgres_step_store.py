"""The shipped PostgreSQL step store: the two extras `stores.py` discovers on the harness store."""

from collections.abc import AsyncIterator
from typing import cast
from uuid import uuid4

import asyncpg
import pytest
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, UserPromptPart
from pydantic_ai_app_factory import AgentError
from pydantic_ai_harness.media import PostgresPool
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepEvent, ToolEffectRecord

from pydantic_ai_conversations.postgres import PostgresConversationStepStore
from pydantic_ai_conversations.stores import (
    PreparingStepStore,
    SupportsPing,
    SupportsRunDeletion,
    SupportsTransientErrors,
)


@pytest.fixture
async def pool(postgres_dsn: str) -> AsyncIterator[asyncpg.Pool]:
    pool = await asyncpg.create_pool(postgres_dsn)
    assert pool is not None
    yield pool
    await pool.close()


@pytest.fixture
def table() -> str:
    return f't_{uuid4().hex[:8]}'


@pytest.fixture
def store(pool: asyncpg.Pool, table: str) -> PostgresConversationStepStore:
    # asyncpg's own annotations do not spell out the harness protocols, so the pool is cast.
    return PostgresConversationStepStore(cast(PostgresPool, pool), table=table)


def _prompt(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


async def _save(store: PostgresConversationStepStore, run_id: str, messages: list[ModelMessage]) -> None:
    await store.register_run(RunRecord(run_id=run_id, conversation_id=f'c-{run_id}'))
    await store.save_snapshot(ContinuableSnapshot(run_id=run_id, step_index=0, messages=messages))


async def _rows(pool: asyncpg.Pool, table: str, suffix: str, run_id: str | None = None) -> int:
    where = '' if run_id is None else f" WHERE run_id = '{run_id}'"
    return cast(int, await pool.fetchval(f'SELECT count(*) FROM {table}_{suffix}{where}'))


async def _stored_texts(pool: asyncpg.Pool, table: str) -> list[str]:
    return [cast(str, row['message']) for row in await pool.fetch(f'SELECT message FROM {table}_messages')]


def test_the_store_satisfies_all_three_optional_protocols(store: PostgresConversationStepStore) -> None:
    assert isinstance(store, SupportsRunDeletion) and isinstance(store, SupportsPing)
    assert isinstance(store, SupportsTransientErrors)


async def test_ping_reports_health_before_anything_is_written(store: PostgresConversationStepStore) -> None:
    assert (await store.ping())['ok'] is True


async def test_delete_runs_removes_the_run_and_ignores_an_empty_id_list(store: PostgresConversationStepStore) -> None:
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    assert await store.get_run(run_id='r1') is not None

    await store.delete_runs(run_ids=[])  # nothing to do, and no statement issued
    assert await store.get_run(run_id='r1') is not None

    await store.delete_runs(run_ids=['r1'])
    assert await store.get_run(run_id='r1') is None
    await store.delete_runs(run_ids=['r1'])  # idempotent: purge may be retried


async def test_delete_runs_removes_every_row_of_the_run(
    store: PostgresConversationStepStore, pool: asyncpg.Pool, table: str
) -> None:
    """Events, snapshots, the replay ledger, tool effects and the stored messages all go, and the
    other run keeps all of its rows."""
    for run_id in ('r1', 'r2'):
        await _save(store, run_id, [_prompt(run_id)])
        await store.append_event(StepEvent(run_id=run_id, kind='run_started', step_index=0))
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id=f'{run_id}-t', tool_name='t', run_id=run_id, status='started')
        )
        await store.save_snapshot(
            ContinuableSnapshot(run_id=run_id, step_index=1, messages=[_prompt(run_id)], idempotency_key='1:1')
        )

    await store.delete_runs(run_ids=['r1'])

    for suffix in ('runs', 'events', 'snapshots', 'snapshot_keys', 'tool_effects'):
        assert await _rows(pool, table, suffix, 'r1') == 0, suffix
        assert await _rows(pool, table, suffix, 'r2') > 0, suffix
    texts = await _stored_texts(pool, table)
    assert texts and all('"r2"' in text for text in texts)


async def test_delete_runs_clears_the_replay_ledger_so_the_run_id_can_be_written_again(
    store: PostgresConversationStepStore,
) -> None:
    """The harness keeps snapshot idempotency keys in a ledger that survives snapshot pruning, so a
    purge that skipped it would leave the run's ids in the database and suppress every later save
    under a recurring key."""
    snapshot = ContinuableSnapshot(run_id='r1', step_index=0, messages=[_prompt('hi')], idempotency_key='0:0')
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(snapshot)

    await store.delete_runs(run_ids=['r1'])

    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(snapshot)
    assert len(await store.list_snapshots(run_id='r1')) == 1


async def test_delete_runs_keeps_a_message_another_run_still_references(
    store: PostgresConversationStepStore, pool: asyncpg.Pool, table: str
) -> None:
    """Identical messages are stored once, so the other run's snapshot must stay readable."""
    shared = _prompt('shared')
    await _save(store, 'r1', [shared, _prompt('own')])
    await _save(store, 'r2', [shared])

    await store.delete_runs(run_ids=['r1'])

    assert [text for text in await _stored_texts(pool, table) if 'own' in text] == []
    snapshot = await store.latest_snapshot(run_id='r2')
    assert snapshot is not None and snapshot.messages == [shared]


async def test_delete_runs_ignores_message_keys_in_a_history_held_inline(
    store: PostgresConversationStepStore, pool: asyncpg.Pool, table: str
) -> None:
    """Snapshots written before messages were stored once hold the history as a JSON list. They
    reference no stored message, even when their text contains a message key, and must not stop
    the purge or keep the message alive."""
    await _save(store, 'r1', [_prompt('own')])
    refs = await pool.fetchval(f"SELECT messages FROM {table}_snapshots WHERE run_id = 'r1'")
    key = cast(str, refs)[len('{"message_refs": ["') : -len('"]}')]
    inline = ModelMessagesTypeAdapter.dump_json([_prompt(key)]).decode()
    for run_id in ('r1', 'r2'):
        await pool.execute(
            f'INSERT INTO {table}_snapshots (run_id, step_index, timestamp, messages) VALUES ($1, 0, $2, $3)',
            run_id,
            '2026-01-01T00:00:00+00:00',
            inline,
        )

    await store.delete_runs(run_ids=['r1'])

    assert await _rows(pool, table, 'snapshots', 'r1') == 0
    assert await _stored_texts(pool, table) == []


async def test_delete_runs_is_one_transaction(
    store: PostgresConversationStepStore, pool: asyncpg.Pool, table: str
) -> None:
    """A failure partway through leaves every row of the run in place for a retried purge, instead
    of orphaning the rest once the run record is gone."""
    await _save(store, 'r1', [_prompt('own')])
    await pool.execute(f'DROP TABLE {table}_tool_effects')

    with pytest.raises(asyncpg.PostgresError):
        await store.delete_runs(run_ids=['r1'])

    assert await store.get_run(run_id='r1') is not None
    assert await _rows(pool, table, 'snapshots', 'r1') == 1
    assert len(await _stored_texts(pool, table)) == 1


async def test_an_unreachable_server_is_reported_as_temporary() -> None:
    pool = await asyncpg.create_pool('postgresql://app@127.0.0.1:1/app', min_size=0)
    assert pool is not None
    try:
        store = PreparingStepStore(PostgresConversationStepStore(cast(PostgresPool, pool)))
        with pytest.raises(AgentError) as exc_info:
            await store.get_run(run_id='r1')
        assert (exc_info.value.code, exc_info.value.retryable) == ('conversations/unavailable', True)
    finally:
        await pool.close()


async def test_a_duplicate_run_id_is_not_reported_as_temporary(store: PostgresConversationStepStore) -> None:
    seam = PreparingStepStore(store)
    await seam.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    with pytest.raises(ValueError, match='already in the store'):
        await seam.register_run(RunRecord(run_id='r1', conversation_id='c1'))
