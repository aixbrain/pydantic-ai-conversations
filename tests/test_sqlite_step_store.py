"""The shipped SQLite step store: the two extras `stores.py` discovers on top of the harness store."""

import sqlite3
from pathlib import Path

import pytest
from pydantic_ai_app_factory import AgentError
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord

from pydantic_ai_conversations.sqlite import SqliteConversationStepStore
from pydantic_ai_conversations.stores import (
    PreparingStepStore,
    SupportsPing,
    SupportsRunDeletion,
    SupportsTransientErrors,
)


def test_the_store_satisfies_all_three_optional_protocols(tmp_path: Path) -> None:
    store = SqliteConversationStepStore(tmp_path / 'app.db')
    assert isinstance(store, SupportsRunDeletion) and isinstance(store, SupportsPing)
    assert isinstance(store, SupportsTransientErrors)


async def test_ping_answers_before_the_schema_exists(tmp_path: Path) -> None:
    store = SqliteConversationStepStore(tmp_path / 'app.db')
    assert await store.ping() == {'ok': True}


async def test_delete_runs_removes_the_run_and_ignores_an_empty_id_list(tmp_path: Path) -> None:
    store = SqliteConversationStepStore(tmp_path / 'app.db')
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    assert await store.get_run(run_id='r1') is not None

    await store.delete_runs(run_ids=[])  # nothing to do, and no statement issued
    assert await store.get_run(run_id='r1') is not None

    await store.delete_runs(run_ids=['r1'])
    assert await store.get_run(run_id='r1') is None
    await store.delete_runs(run_ids=['r1'])  # idempotent: purge may be retried


async def test_delete_runs_clears_the_replay_ledger_so_the_run_id_can_be_written_again(tmp_path: Path) -> None:
    """The harness keeps snapshot idempotency keys in a ledger that survives snapshot pruning, so
    a purge that skipped it would leave the run's ids on disk and suppress every later save under
    a recurring key."""
    store = SqliteConversationStepStore(tmp_path / 'app.db')
    snapshot = ContinuableSnapshot(run_id='r1', step_index=0, messages=[], idempotency_key='0:0:complete')
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(snapshot)

    await store.delete_runs(run_ids=['r1'])

    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(snapshot)
    assert len(await store.list_snapshots(run_id='r1')) == 1


async def test_a_database_that_cannot_be_opened_is_reported_as_temporary(tmp_path: Path) -> None:
    store = PreparingStepStore(SqliteConversationStepStore(tmp_path))  # a directory, not a file
    with pytest.raises(AgentError) as exc_info:
        await store.get_run(run_id='r1')
    assert (exc_info.value.code, exc_info.value.retryable) == ('conversations/unavailable', True)


async def test_a_duplicate_run_id_is_not_reported_as_temporary(tmp_path: Path) -> None:
    store = PreparingStepStore(SqliteConversationStepStore(tmp_path / 'app.db'))
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    with pytest.raises(sqlite3.IntegrityError):
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))


async def test_delete_runs_on_a_database_without_the_harness_tables_is_a_no_op(tmp_path: Path) -> None:
    """The harness creates its tables on first write, so a purge can run before any run was stored;
    unknown ids are a no-op by contract, tables or not."""
    store = SqliteConversationStepStore(tmp_path / 'fresh.db')
    await store.delete_runs(run_ids=['unknown'])
    assert await store.ping() == {'ok': True}


async def test_a_failing_delete_rolls_back_and_leaves_the_connection_usable(tmp_path: Path) -> None:
    """The driver keeps one connection open for its life, so a transaction left open by a failed
    statement would make every later call fail with "cannot start a transaction within a
    transaction" until the process restarts."""
    store = SqliteConversationStepStore(tmp_path / 'app.db')
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    with sqlite3.connect(tmp_path / 'app.db') as raw:
        raw.execute('DROP TABLE tool_effects')

    with pytest.raises(sqlite3.OperationalError, match='tool_effects'):
        await store.delete_runs(run_ids=['r1'])

    assert await store.get_run(run_id='r1') is not None  # rolled back, so a retried purge finds it
    with pytest.raises(sqlite3.OperationalError, match='tool_effects'):
        await store.delete_runs(run_ids=['r1'])  # the same failure again, not a stuck transaction


async def test_media_settings_reach_the_base_store(tmp_path: Path) -> None:
    store = SqliteConversationStepStore(tmp_path / 'app.db', media_store=None, media_threshold_bytes=7)
    assert store._media_store is None  # pyright: ignore[reportPrivateUsage]
    assert store._media_threshold_bytes == 7  # pyright: ignore[reportPrivateUsage]


async def test_close_releases_the_drivers_connection_and_a_later_call_reconnects(tmp_path: Path) -> None:
    store = SqliteConversationStepStore(tmp_path / 'app.db')
    assert await store.ping() == {'ok': True}
    await store.close()
    assert store._driver._connection is None  # pyright: ignore[reportPrivateUsage]
    assert await store.ping() == {'ok': True}
