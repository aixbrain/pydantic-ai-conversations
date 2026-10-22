"""Performance contracts of the SQL stack, pinned deterministically where possible.

Wall-clock bounds are deliberately generous (CI machines vary); the load-bearing assertions are
the deterministic ones -- store round-trip counts (`max_history_runs` caps the documented N+1
`latest_snapshot` cost) and linear history growth across many turns (the write seam must cut
the loaded history out of every snapshot, or snapshots compound superlinearly)."""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_app_factory import AppDeps, UserContext
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord

from pydantic_ai_conversations import ConversationsFeature
from pydantic_ai_conversations._loader import load_conversation
from pydantic_ai_conversations.sqlite import SqliteConversationRegistry, SqliteConversationStepStore

_NS = '[["user.tenant_id","t1"]]'


class CountingStepStore(SqliteConversationStepStore):
    """Counts `latest_snapshot` round trips -- the unit `max_history_runs` promises to cap."""

    def __init__(self, database: str | Path) -> None:
        super().__init__(database)
        self.latest_snapshot_calls = 0

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        self.latest_snapshot_calls += 1
        return await super().latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)


async def _seed_runs(store: SqliteConversationStepStore, conversation_id: str, count: int) -> None:
    base = datetime.now(UTC)
    for index in range(count):
        run_id = f'run-{index:03d}'
        await store.register_run(
            RunRecord(run_id=run_id, conversation_id=conversation_id, started_at=base + timedelta(seconds=index))
        )
        await store.save_snapshot(
            ContinuableSnapshot(
                run_id=run_id,
                step_index=1,
                messages=[
                    ModelRequest(parts=[UserPromptPart(content=f'q{index}')]),
                    ModelResponse(parts=[TextPart(content=f'a{index}')]),
                ],
                conversation_id=conversation_id,
                state='complete',
            )
        )


async def test_max_history_runs_caps_the_store_round_trips(tmp_path: Path) -> None:
    store = CountingStepStore(tmp_path / 'perf.db')
    cid = uuid4().hex
    await _seed_runs(store, cid, 60)

    store.latest_snapshot_calls = 0
    run_ids = [f'run-{index:03d}' for index in range(60)]
    capped = await load_conversation(store, conversation_id=cid, run_ids=run_ids, max_history_runs=10)
    # Exactly the last 10 contributing runs were read -- not 60 reads then a trim.
    assert store.latest_snapshot_calls == 10
    assert len(capped) == 20 and 'q50' in str(capped[0])

    store.latest_snapshot_calls = 0
    full = await load_conversation(store, conversation_id=cid, run_ids=run_ids)
    assert store.latest_snapshot_calls == 60  # the documented N+1 cost the cap exists to bound
    assert len(full) == 120


async def test_listing_a_large_namespace_stays_fast(tmp_path: Path) -> None:
    registry = SqliteConversationRegistry(tmp_path / 'perf.db')
    last_cid = ''
    for index in range(1500):
        last_cid = f'{index:04d}-{uuid4().hex}'
        await registry.register_conversation(_NS, last_cid, created_by='alice')
        await registry.update_message_stats(
            _NS, last_cid, last_message_at=datetime.now(UTC) + timedelta(seconds=index), message_count_delta=2
        )
    started = time.perf_counter()
    page = await registry.list(_NS, limit=100)
    elapsed = time.perf_counter() - started
    assert len(page) == 100 and page[0].conversation_id == last_cid  # newest activity first
    assert elapsed < 0.5, f'one listing page over 1500 rows took {elapsed:.3f}s'


async def test_concurrent_stats_updates_sustain_throughput(tmp_path: Path) -> None:
    registry = SqliteConversationRegistry(tmp_path / 'perf.db')
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    started = time.perf_counter()
    await asyncio.gather(
        *(
            registry.update_message_stats(_NS, cid, last_message_at=datetime.now(UTC), message_count_delta=1)
            for _ in range(300)
        )
    )
    elapsed = time.perf_counter() - started
    meta = await registry.get(_NS, cid)
    assert meta is not None and meta.message_count == 300  # atomic increments: nothing lost under load
    assert elapsed < 5.0, f'300 concurrent stats updates took {elapsed:.3f}s'


async def test_history_growth_stays_linear_across_many_turns(tmp_path: Path) -> None:
    """12 multi-step turns end to end: model context grows by exactly one 4-message batch per turn.

    THE compounding regression this stack documents against (`_capability.py`'s module docstring):
    were the loaded history stored with each turn, snapshots would re-carry it and the model
    request size would grow quadratically -- exact counts catch that, wall clocks only eventually.
    """
    database = tmp_path / 'perf.db'
    registry, store = SqliteConversationRegistry(database), SqliteConversationStepStore(database)
    seen: list[list[ModelMessage]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
            return ModelResponse(parts=[TextPart(content='noted')])
        return ModelResponse(parts=[ToolCallPart(tool_name='lookup_part', args={'query': 'x'})])

    feature = ConversationsFeature(
        registry=registry,
        step_store=store,
        scope=('user.tenant_id',),
    )
    agent: Agent[AppDeps, Any] = Agent(FunctionModel(model_fn), deps_type=AppDeps, capabilities=feature.capabilities())

    @agent.tool_plain
    def lookup_part(query: str) -> str:
        return f'part-info:{query}'

    user = UserContext(user_id='alice', tenant_id='t1')
    deps = AppDeps(user=user, extras={})
    cid = uuid4().hex
    started = time.perf_counter()
    for turn in range(12):
        result = await agent.run(f'turn {turn}', deps=deps, conversation_id=cid)
        deps = AppDeps(user=user, extras={'previous_run_id': result.run_id})
    elapsed = time.perf_counter() - started

    # Turn N's first model request: 4 messages per earlier turn, plus its own prompt.
    first_request_sizes = [len(seen[2 * turn]) for turn in range(12)]
    assert first_request_sizes == [4 * turn + 1 for turn in range(12)]
    meta = await registry.get(feature.namespace_for(deps.user), cid)
    assert meta is not None and meta.message_count == 48  # every stored batch stayed per-turn: 12 * 4
    assert elapsed < 15.0, f'12 SQLite-backed turns took {elapsed:.3f}s'
