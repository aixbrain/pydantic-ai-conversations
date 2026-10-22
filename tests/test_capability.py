"""ConversationsCapability: registry upkeep + loading the conversation into run state."""

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolOutput
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_app_factory import AgentError, AppDeps, UserContext
from pydantic_ai_harness.compaction import SlidingWindowCompaction
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, RunRecord, StepPersistence

from pydantic_ai_conversations._capability import ConversationsCapability, _prompt_in_history, agent_error_mapper
from pydantic_ai_conversations._loader import load_conversation, load_transcript
from pydantic_ai_conversations.errors import ConversationNotFound, RegistryUnavailable
from pydantic_ai_conversations.feature import ConversationsFeature
from pydantic_ai_conversations.registry import ConversationMeta, InMemoryConversationRegistry, TurnError
from pydantic_ai_conversations.scope import build_namespace, parse_scope
from pydantic_ai_conversations.stores import NewMessagesPreparer, PreparingStepStore

SCOPE = parse_scope(('user.tenant_id', 'user.user_id'))


def _agent(
    registry: InMemoryConversationRegistry, store: InMemoryStepStore, seen: list[list[ModelMessage]]
) -> Agent[AppDeps, str]:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart(content='ok')])

    return Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            # The feature's order: the capability stamps the prompt before StepPersistence's
            # frontier save of it (`_capability.py` module docstring).
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )


def _agent_with_capability(
    store: InMemoryStepStore,
    seen: list[list[ModelMessage]],
    capability: ConversationsCapability[AppDeps],
    *,
    extra_capabilities: tuple[AbstractCapability[AppDeps], ...] = (),
) -> Agent[AppDeps, str]:
    """Like `_agent`, but takes an already-built `ConversationsCapability` and lets a caller
    insert extra agent-level capabilities ahead of it. Both helpers compose the feature's write
    seam (`NewMessagesPreparer`) in front of the store, as `ConversationsFeature` does.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart(content='ok')])

    return Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            *extra_capabilities,
            capability,
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )


_OMITTED = object()


def _deps(user_id: str = 'alice', tenant_id: str = 't1', *, previous: object = _OMITTED) -> AppDeps:
    """Deps for one run; `previous` is the `previous_run_id` the client sends from the second
    turn of a conversation on. Left out, the request carries no such field; `None` is the
    explicit value a retry of a first turn sends."""
    extras = {} if previous is _OMITTED else {'previous_run_id': previous}
    return AppDeps(user=UserContext(user_id=user_id, tenant_id=tenant_id), extras=extras)


async def test_second_turn_sees_first_turns_history() -> None:
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    await agent.run('first prompt', deps=_deps(), conversation_id='c1', run_id='r1')
    await agent.run('second prompt', deps=_deps(previous='r1'), conversation_id='c1')
    # The second run's model request starts with the first turn's messages.
    second_request_messages = seen[1]
    assert len(second_request_messages) > len(seen[0])
    flattened = str(second_request_messages)
    assert 'first prompt' in flattened and 'second prompt' in flattened


async def test_run_state_lives_on_the_for_run_copy_not_the_agent_bound_instance() -> None:
    """`for_run` hands the graph a fresh copy per run and every hook runs on it, so run state
    cannot alias across runs sharing one `AppDeps`. If pydantic-ai ever stopped calling `for_run`, run
    state would land on the agent-bound original and this fails before anything aliases.
    """
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    capability = ConversationsCapability[AppDeps](registry=registry, step_store=store, scope=SCOPE)
    agent = _agent_with_capability(store, seen, capability)
    await agent.run('hello', deps=_deps(), conversation_id='c1')
    assert capability._run_namespace is None


async def test_snapshots_are_batches_not_full_history() -> None:
    # THE invariant this feature stands on, pinned exactly: each run's snapshot equals that
    # run's own messages. Run state holds the conversation from `before_run` on, and the
    # write seam cuts every snapshot back to the run's own turn; if either side drifts, this
    # test fails before any data compounds.
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    result_one = await agent.run('first prompt', deps=_deps(), conversation_id='c1', run_id='r1')
    result_two = await agent.run('second prompt', deps=_deps(previous='r1'), conversation_id='c1')
    runs = sorted(await store.list_runs(conversation_id='c1'), key=lambda r: r.started_at)
    assert len(runs) == 2
    for run, result in zip(runs, (result_one, result_two), strict=True):
        snapshot = await store.latest_snapshot(run_id=run.run_id)
        assert snapshot is not None
        assert snapshot.messages == result.new_messages()
    # And the second turn's snapshot does not contain the first turn's prompt.
    second = await store.latest_snapshot(run_id=runs[1].run_id)
    assert second is not None and 'first prompt' not in str(second.messages)


async def test_registry_entry_created_and_activity_updated() -> None:
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    await agent.run('first prompt', deps=_deps(), conversation_id='c1', run_id='r1')
    namespace = build_namespace(SCOPE, _deps().user, {})
    meta = await registry.get(namespace, 'c1')
    assert meta is not None
    assert meta.created_by == 'alice'
    assert meta.last_message_at is not None and meta.message_count > 0
    turns = await registry.list_turns(namespace, 'c1')
    assert [t.author_user_id for t in turns] == ['alice']


async def test_precreated_attrs_survive_the_implicit_registration() -> None:
    """The escape hatch for the implicit-create path (no `enable_create_route`, see
    `ConversationsFeature`'s class docstring): pre-create the entry with `attrs` before the first
    run touches it. `register_conversation` is idempotent -- an undeleted entry already in the namespace is returned
    as-is, `attrs` on the call ignored, not merged -- so the capability's own implicit `register_conversation` in
    `before_run` cannot clobber attrs set ahead of time.
    """
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    namespace = build_namespace(SCOPE, _deps().user, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice', attrs={'source': 'web'})

    await agent.run('hello', deps=_deps(), conversation_id='c1')

    filtered = await registry.list(namespace, attrs={'source': 'web'})
    assert [m.conversation_id for m in filtered] == ['c1']


async def test_foreign_conversation_id_is_rejected_as_not_found() -> None:
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    await agent.run('first prompt', deps=_deps('alice', 't1'), conversation_id='c1')
    with pytest.raises(AgentError) as exc_info:
        await agent.run('intrude', deps=_deps('mallory', 't2'), conversation_id='c1')
    assert exc_info.value.code == 'conversations/not-found'
    assert exc_info.value.http_status == 404


async def test_run_without_conversation_id_still_gets_a_registry_entry() -> None:
    """`RunContext.conversation_id` is never `None` in installed pydantic-ai 2.28: `Agent.run()`
    resolves it before any capability hook fires (`_agent_graph.resolve_conversation_id`), as the
    explicit value if given, else the most recent id on `message_history`, else a fresh UUID7. So
    `ctx.conversation_id is None` is unreachable and the capability keeps that guard only as
    defensive dead code. There is no signal that lets a capability distinguish "caller wants
    conversation tracking" from "caller didn't ask": every run participates in the registry, keyed
    by whatever conversation_id pydantic-ai resolved for it, and two unrelated bare calls each get
    their own fresh id.
    """
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    result = await agent.run('hello', deps=_deps())
    assert result.output == 'ok'
    namespace = build_namespace(SCOPE, _deps().user, {})
    assert len(await registry.list(namespace)) == 1
    await agent.run('hello again', deps=_deps())
    assert len(await registry.list(namespace)) == 2


# --- Direct unit tests below ------------------------------------------------------------------
#
# These exercise capability methods against synthetic `RunContext`/`ModelRequestContext` objects
# rather than a full `agent.run()`, for two kinds of branches a full run cannot reach:
#
# 1. `conversation_id is None`: unreachable through `agent.run()` on this pydantic-ai version (see
#    the invariant note above), but a directly-constructed `RunContext` -- one the graph never
#    built -- defaults `conversation_id` to `None`, same as `usage_limits` and other fields the
#    `RunContext` docstring calls out as "only `None` on a bare/synthetic RunContext".
# 2. Registry backend failures (`RegistryUnavailable`) and the `_joined` dangling-request merge:
#    not naturally producible through `InMemoryConversationRegistry` / a clean two-turn run.


def _bare_ctx(
    *, conversation_id: str | None = None, run_id: str | None = None, previous: str | None = None
) -> RunContext[AppDeps]:
    return RunContext(
        deps=_deps(previous=previous),
        model=TestModel(),
        usage=RunUsage(),
        conversation_id=conversation_id,
        run_id=run_id,
    )


class _RegisterUnavailableRegistry(InMemoryConversationRegistry):
    """Raises on `register_conversation`, as a real backend does on a transient outage."""

    async def register_conversation(
        self, namespace_key: str, conversation_id: str, *, created_by: str, attrs: Mapping[str, str] | None = None
    ) -> ConversationMeta:
        raise RegistryUnavailable('backend down')


class _UnavailableMessageStatsRegistry(InMemoryConversationRegistry):
    """Raises on `update_message_stats` only -- `register_conversation`/`start_turn` succeed,
    mirroring a backend whose read path degrades independently of its write path."""

    async def update_message_stats(
        self, namespace_key: str, conversation_id: str, *, last_message_at: datetime, message_count_delta: int
    ) -> None:
        raise RegistryUnavailable('backend down')


class _DeletedMidRunMessageStatsRegistry(InMemoryConversationRegistry):
    """Raises `ConversationNotFound` on `update_message_stats` only, as if a `DELETE` raced the run to
    completion: the entry existed when the run started but is gone by the time it finishes."""

    async def update_message_stats(
        self, namespace_key: str, conversation_id: str, *, last_message_at: datetime, message_count_delta: int
    ) -> None:
        raise ConversationNotFound(conversation_id)


async def test_before_run_is_a_noop_when_conversation_id_is_none() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    capability = ConversationsCapability(registry=registry, step_store=store, scope=SCOPE)
    ctx = _bare_ctx(conversation_id=None)

    await capability.before_run(ctx)

    assert await registry.list(build_namespace(SCOPE, _deps().user, {})) == []
    assert capability._run_namespace is None  # no run state recorded either


async def test_after_run_is_a_noop_when_conversation_id_is_none() -> None:
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    real_result = await agent.run('hello', deps=_deps())
    capability = ConversationsCapability(
        registry=InMemoryConversationRegistry(), step_store=InMemoryStepStore(), scope=SCOPE
    )
    ctx = _bare_ctx(conversation_id=None)

    returned = await capability.after_run(ctx, result=real_result)

    assert returned is real_result


class _UnavailableReadStepStore(InMemoryStepStore):
    """Raises on `latest_snapshot`, as a real step-store backend does on a transient read-path
    outage: the first read `load_conversation` makes for a chain with a turn on it."""

    transient_errors = (OSError,)
    failure: Exception = ConnectionError('backend down')

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        raise self.failure


async def test_before_run_maps_step_store_failure_to_agent_error() -> None:
    """A step-store read failure during `load_conversation` must not escape as a bare
    exception -- it maps to the same retryable `conversations/unavailable` `AgentError` a registry
    outage produces (see `test_before_run_maps_registry_unavailable_to_agent_error` below).
    """
    registry = InMemoryConversationRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    await registry.start_turn(namespace, 'c1', run_id='r0', author_user_id='alice')
    capability = ConversationsCapability(registry=registry, step_store=_UnavailableReadStepStore(), scope=SCOPE)
    ctx = _bare_ctx(conversation_id='c1', run_id='r1', previous='r0')

    with pytest.raises(AgentError) as exc_info:
        await capability.before_run(ctx)

    assert exc_info.value.code == 'conversations/unavailable'
    assert exc_info.value.retryable is True


async def test_before_run_lets_a_permanent_step_store_failure_through() -> None:
    """A malformed stored row fails every load the same way, so it is not reported as retryable."""
    registry = InMemoryConversationRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    await registry.start_turn(namespace, 'c1', run_id='r0', author_user_id='alice')
    store = _UnavailableReadStepStore()
    store.failure = ValueError('snapshot row has wrong types')
    capability = ConversationsCapability(registry=registry, step_store=store, scope=SCOPE)

    with pytest.raises(ValueError, match='wrong types'):
        await capability.before_run(_bare_ctx(conversation_id='c1', run_id='r1', previous='r0'))


def test_prompt_in_history_is_the_trailing_user_request() -> None:
    """The UI adapters put the new prompt inside `message_history`: the client's list ends in a
    request carrying a `UserPromptPart`, and that request alone is the run's own input."""
    ctx = _bare_ctx(conversation_id='c1', run_id='r1')
    new_prompt = ModelRequest(parts=[UserPromptPart(content='new')])
    ctx.messages.extend(
        [
            ModelRequest(parts=[UserPromptPart(content='old')]),
            ModelResponse(parts=[TextPart(content='old a')]),
            new_prompt,
        ]
    )
    assert _prompt_in_history(ctx) is new_prompt


def test_prompt_in_history_is_none_when_the_prompt_is_passed_separately() -> None:
    """A separately passed prompt is appended by the graph after `before_run`; the client's list
    then holds replayed history only."""
    ctx = replace(_bare_ctx(conversation_id='c1', run_id='r1'), prompt='new')
    ctx.messages.extend(
        [ModelRequest(parts=[UserPromptPart(content='old')]), ModelResponse(parts=[TextPart(content='old a')])]
    )
    assert _prompt_in_history(ctx) is None


def test_prompt_in_history_is_none_when_the_client_list_ends_in_a_tool_return() -> None:
    """A trailing request without a `UserPromptPart` (a tool return) is not a new prompt."""
    ctx = _bare_ctx(conversation_id='c1', run_id='r1')
    ctx.messages.extend(
        [
            ModelRequest(parts=[UserPromptPart(content='do the thing')]),
            ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id='tc-1')]),
            ModelRequest(parts=[ToolReturnPart(tool_name='lookup', content='result', tool_call_id='tc-1')]),
        ]
    )
    assert _prompt_in_history(ctx) is None


def test_prompt_in_history_is_none_for_an_empty_client_list() -> None:
    assert _prompt_in_history(_bare_ctx(conversation_id='c1', run_id='r1')) is None


async def test_before_run_maps_registry_unavailable_to_agent_error() -> None:
    capability = ConversationsCapability(
        registry=_RegisterUnavailableRegistry(), step_store=InMemoryStepStore(), scope=SCOPE
    )
    ctx = _bare_ctx(conversation_id='c1', run_id='r1')

    with pytest.raises(AgentError) as exc_info:
        await capability.before_run(ctx)

    assert exc_info.value.code == 'conversations/unavailable'
    assert exc_info.value.retryable is True


async def test_after_run_logs_and_continues_when_registry_message_stats_update_is_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    real_result = await agent.run('hello', deps=_deps())
    flaky = _UnavailableMessageStatsRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    # Seed the entry first: `update_message_stats`'s override raises unconditionally, but seeding means this test
    # fails for the RIGHT reason (an existing entry whose backend degrades) rather than accidentally
    # also covering a missing-entry scenario if the override were ever removed.
    await flaky.register_conversation(namespace, 'c1', created_by='alice')
    capability = ConversationsCapability(registry=flaky, step_store=InMemoryStepStore(), scope=SCOPE)
    ctx = _bare_ctx(conversation_id='c1', run_id='r1')

    with caplog.at_level('WARNING'):
        returned = await capability.after_run(ctx, result=real_result)

    assert returned is real_result
    assert 'registry message-stats update failed' in caplog.text


async def test_after_run_logs_and_continues_when_conversation_deleted_mid_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A `DELETE /conversations/{id}` racing a run must not turn a successful run into a
    client-facing 500: the answer is already persisted (the snapshot landed) by the time
    `after_run`'s `update_message_stats` finds the entry gone, so the failure is logged and swallowed, not raised.
    """
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    real_result = await agent.run('hello', deps=_deps())
    flaky = _DeletedMidRunMessageStatsRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await flaky.register_conversation(namespace, 'c1', created_by='alice')
    capability = ConversationsCapability(registry=flaky, step_store=InMemoryStepStore(), scope=SCOPE)
    ctx = _bare_ctx(conversation_id='c1', run_id='r1')

    with caplog.at_level('INFO'):
        returned = await capability.after_run(ctx, result=real_result)

    assert returned is real_result
    assert 'deleted mid-run' in caplog.text


# --- Broader run shapes ------------------------------------------------------------------------
#
# The tests above use a single-step, non-streaming `agent.run()`. These pin the same invariants
# against shapes that stress the capability's per-run caching and `wrap_model_request` seam
# differently: a streamed run, a run that makes more than one model request (a tool call), and two
# runs on one conversation racing each other.


async def test_streamed_runs_keep_the_batch_invariant() -> None:
    """The snapshot == `all_messages()` invariant (see `test_snapshots_are_batches_not_full_history`)
    must hold under `run_stream` too: `wrap_model_request` fires identically regardless of whether
    the caller streams, so loaded history must never leak into a streamed run's own snapshot.
    """
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        seen.append(list(messages))
        yield 'ok'

    agent = Agent(
        FunctionModel(stream_function=stream_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )

    async with agent.run_stream('first prompt', deps=_deps(), conversation_id='c1', run_id='r1') as stream_one:
        await stream_one.get_output()
        messages_one = stream_one.new_messages()
    async with agent.run_stream('second prompt', deps=_deps(previous='r1'), conversation_id='c1') as stream_two:
        await stream_two.get_output()
        messages_two = stream_two.new_messages()

    runs = sorted(await store.list_runs(conversation_id='c1'), key=lambda r: r.started_at)
    assert len(runs) == 2
    for run, messages in zip(runs, (messages_one, messages_two), strict=True):
        snapshot = await store.latest_snapshot(run_id=run.run_id)
        assert snapshot is not None
        assert snapshot.messages == messages
    second = await store.latest_snapshot(run_id=runs[1].run_id)
    assert second is not None and 'first prompt' not in str(second.messages)


async def test_two_step_tool_run_does_not_duplicate_history() -> None:
    """A run that calls a tool makes two model requests. Per-run history caching (see
    `test_wrap_model_request_loads_history_once_per_run`) must mean both of turn 2's requests see
    the loaded history exactly once each -- never duplicated across the extra request a tool call
    adds -- and turn 2's own snapshot must still equal `all_messages()`.
    """
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
            return ModelResponse(parts=[TextPart(content='final answer')])
        return ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id='tc-x')])

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )

    @agent.tool
    async def lookup(ctx: RunContext[AppDeps]) -> str:
        return 'tool result'

    result_one = await agent.run('first prompt', deps=_deps(), conversation_id='c1', run_id='r1')
    result_two = await agent.run('second prompt', deps=_deps(previous='r1'), conversation_id='c1')

    turn_two_requests = seen[2:]
    assert len(turn_two_requests) == 2  # tool-call request, then the post-tool-return request
    for messages in turn_two_requests:
        assert str(messages).count('first prompt') == 1

    runs = sorted(await store.list_runs(conversation_id='c1'), key=lambda r: r.started_at)
    assert len(runs) == 2
    for run, result in zip(runs, (result_one, result_two), strict=True):
        snapshot = await store.latest_snapshot(run_id=run.run_id)
        assert snapshot is not None
        assert snapshot.messages == result.new_messages()


async def test_server_mode_replaces_a_client_resending_full_history() -> None:
    """A client that ignores the server-authoritative
    contract and re-sends its entire local history (the Vercel AI SDK default transport) must not
    duplicate the first turn -- neither in the stored snapshot nor at the model.
    """
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    first = await agent.run('first prompt', deps=_deps(), conversation_id='c1', run_id='r1')
    # A misbehaving client re-sends everything it has, plus the new turn, as message_history.
    replay = [*first.all_messages(), ModelRequest(parts=[UserPromptPart(content='second prompt')])]
    await agent.run(deps=_deps(previous='r1'), conversation_id='c1', message_history=replay)
    runs = sorted(await store.list_runs(conversation_id='c1'), key=lambda r: r.started_at)
    snapshot = await store.latest_snapshot(run_id=runs[1].run_id)
    assert snapshot is not None
    # The snapshot holds only the second turn -- the replayed first turn was trimmed.
    assert 'first prompt' not in str(snapshot.messages)
    assert 'second prompt' in str(snapshot.messages)
    # And the model saw the first turn exactly once (injected), not twice.
    assert str(seen[1]).count('first prompt') == 1


async def test_server_mode_replaces_a_streamed_client_resending_full_history() -> None:
    """The batch invariant under `run_stream` (see `test_streamed_runs_keep_the_batch_invariant`)
    must hold against a replaying client too: `before_model_request` fires identically whether the
    caller streams or not, so the trim must keep the streamed run's own snapshot batch-only.
    """
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        seen.append(list(messages))
        yield 'ok'

    agent = Agent(
        FunctionModel(stream_function=stream_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )

    async with agent.run_stream('first prompt', deps=_deps(), conversation_id='c1', run_id='r1') as stream_one:
        await stream_one.get_output()
        first_messages = stream_one.all_messages()

    replay = [*first_messages, ModelRequest(parts=[UserPromptPart(content='second prompt')])]
    async with agent.run_stream(deps=_deps(previous='r1'), conversation_id='c1', message_history=replay) as stream_two:
        await stream_two.get_output()

    runs = sorted(await store.list_runs(conversation_id='c1'), key=lambda r: r.started_at)
    assert len(runs) == 2
    snapshot = await store.latest_snapshot(run_id=runs[1].run_id)
    assert snapshot is not None
    assert 'first prompt' not in str(snapshot.messages)
    assert 'second prompt' in str(snapshot.messages)
    assert str(seen[1]).count('first prompt') == 1


async def test_server_mode_keeps_run_state_coherent_with_the_run_result() -> None:
    """The trimmed messages are what the graph writes back into run state (`ctx.state.message_history
    [:] = messages`, per this hook's own docstring), so the run's own result object -- not just the
    stored snapshot -- must reflect the trim: `all_messages()` (the write-back plus the response)
    must not contain the replayed first turn, and must be exactly the trimmed request plus response
    -- two messages, no leftover replay fragments.

    `new_messages()` is deliberately NOT asserted to equal that same pair here: this call passes no
    separate `user_prompt`, matching the Vercel-AI-SDK-style client this feature defends against
    (the whole array, newest message included, arrives as one `message_history`). Empirically,
    pydantic-ai's own graph then treats the trailing request in that history as `resumed_request`
    (`ModelRequestNode.is_resuming_without_prompt`, `_agent_graph.py`) -- "prior context", the same
    bookkeeping a deferred-tool resume gets -- so `new_messages()` returns only the model's response.
    That framework behavior is orthogonal to this capability's hook; pinning it here would test
    pydantic-ai's bookkeeping, not our replacement of the client's list.

    `all_messages()` holds the conversation: the stored first turn (replacing the client's
    copy of it) and the run's own request and response.
    """
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    first = await agent.run('first prompt', deps=_deps(), conversation_id='c1', run_id='r1')
    replay = [*first.all_messages(), ModelRequest(parts=[UserPromptPart(content='second prompt')])]

    second = await agent.run(deps=_deps(previous='r1'), conversation_id='c1', message_history=replay)

    all_messages = second.all_messages()
    assert str(all_messages).count('first prompt') == 1
    assert len(all_messages) == 4
    request, response = all_messages[2:]
    assert isinstance(request, ModelRequest)
    assert len(request.parts) == 1
    assert isinstance(request.parts[0], UserPromptPart)
    assert request.parts[0].content == 'second prompt'
    assert isinstance(response, ModelResponse)


async def test_server_mode_on_a_multi_step_tool_run_keeps_the_runs_own_tool_messages() -> None:
    """A replaying client on turn 2 of a run that also calls a tool: the client's copy of turn 1
    is replaced once, in `before_run`, and the run's own tool-call/tool-return messages then
    accumulate in run state as on any pydantic-ai run. Pin: step 2 of the run keeps them, and the
    final snapshot has the tool call but not the replayed turn-1 prompt.
    """
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
            return ModelResponse(parts=[TextPart(content='final answer')])
        return ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id='tc-x')])

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )

    @agent.tool
    async def lookup(ctx: RunContext[AppDeps]) -> str:
        return 'tool result'

    first = await agent.run('first prompt', deps=_deps(), conversation_id='c1', run_id='r1')
    replay = [*first.all_messages(), ModelRequest(parts=[UserPromptPart(content='second prompt')])]
    result_two = await agent.run(deps=_deps(previous='r1'), conversation_id='c1', message_history=replay)

    turn_two_requests = seen[2:]
    assert len(turn_two_requests) == 2  # tool-call request, then the post-tool-return request
    for messages in turn_two_requests:
        # The client's replay contributes nothing extra: 'first prompt' appears exactly once, from
        # the loaded (server-side) turn-1 history -- not twice, which is what keeping the client's
        # copy beside it would have produced.
        assert str(messages).count('first prompt') == 1
        assert str(messages).count('second prompt') == 1

    runs = sorted(await store.list_runs(conversation_id='c1'), key=lambda r: r.started_at)
    assert len(runs) == 2
    snapshot = await store.latest_snapshot(run_id=runs[1].run_id)
    assert snapshot is not None
    # A prompt sent inside `message_history` counts as prior context for `new_messages()` (see above),
    # and the graph rebuilds that request when it cleans the history; the snapshot is the whole
    # turn, prompt included.
    assert snapshot.messages[1:] == result_two.new_messages()
    flattened = str(snapshot.messages)
    assert 'second prompt' in str(snapshot.messages[0])
    assert 'tc-x' in flattened  # the run's own tool call survives
    assert 'first prompt' not in flattened  # the replayed turn-1 prompt was trimmed


async def test_concurrent_writers_on_one_tip_branch() -> None:
    """Two runs that build on the same turn at the same time (`asyncio.gather`) both land: no lost
    update, no duplicate registry entry. They are siblings, and a later turn naming one of them
    sees that one's branch, not the other's. `InMemoryConversationRegistry`'s methods are atomic
    with respect to concurrent coroutines (no `await` between read and write within a method).
    """
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(0)
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart(content='ok')])

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )

    await agent.run('prompt 0', deps=_deps(), conversation_id='c1', run_id='r0')
    await asyncio.gather(
        agent.run('prompt A', deps=_deps(previous='r0'), conversation_id='c1', run_id='rA'),
        agent.run('prompt B', deps=_deps(previous='r0'), conversation_id='c1', run_id='rB'),
    )

    namespace = build_namespace(SCOPE, _deps().user, {})
    turns = await registry.list_turns(namespace, 'c1')
    assert len(turns) == 3
    assert {t.run_id: t.previous_run_id for t in turns} == {'r0': None, 'rA': 'r0', 'rB': 'r0'}

    await agent.run('prompt C', deps=_deps(previous='rA'), conversation_id='c1')
    # What the model received, loaded history included, is `seen[-1]`: branch A, without B.
    flattened = str(seen[-1])
    assert 'prompt 0' in flattened and 'prompt A' in flattened
    assert 'prompt B' not in flattened


# --- Pending turns: a run that fails before its first model response must not
# lose the user's typed message -- the harness's frontier save keeps it before the first model request.
# ------------------------------------------------------------------------------------------------


def _flaky_agent(
    registry: InMemoryConversationRegistry, store: InMemoryStepStore, *, fail_on_call: int
) -> Agent[AppDeps, str]:
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == fail_on_call:
            raise RuntimeError('boom')
        return ModelResponse(parts=[TextPart(content='ok')])

    return Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )


async def test_failed_first_request_preserves_the_user_message() -> None:
    """Turn 2's model raises before producing a `ModelResponse`; the frontier save holds its
    prompt, so `load_transcript` shows the failed turn with it and a run naming the failed turn
    continues from that prompt. A retry names the failed turn's predecessor as
    `previous_run_id` instead, so the failed turn stays a visible side branch and the retry
    builds on the turn before it, never seeing the failed prompt.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    agent = _flaky_agent(registry, store, fail_on_call=2)

    await agent.run('first prompt', deps=_deps(), conversation_id='c1', run_id='r1')
    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('second prompt', deps=_deps(previous='r1'), conversation_id='c1', run_id='r2')

    transcript = await load_transcript(store, conversation_id='c1')
    assert [t.phase for t in transcript] == ['answered', 'failed']
    assert 'second prompt' in str(transcript[1].messages)

    continuing = await load_conversation(store, conversation_id='c1', run_ids=['r1', 'r2'])
    assert 'second prompt' in str(continuing[-1])  # what a run building on r2 continues from
    retrying = await load_conversation(store, conversation_id='c1', run_ids=['r1'])
    assert 'second prompt' not in str(retrying)  # the retry's branch never had it

    # The retry re-sends the text as a new run building on r1, the failed turn's predecessor.
    await agent.run('second prompt', deps=_deps(previous='r1'), conversation_id='c1', run_id='r3')
    transcript_after_retry = await load_transcript(store, conversation_id='c1')
    assert [t.phase for t in transcript_after_retry] == ['answered', 'failed', 'answered']
    assert 'second prompt' in str(transcript_after_retry[2].messages)
    namespace = build_namespace(SCOPE, _deps().user, {})
    turns = await registry.list_turns(namespace, 'c1')
    assert [(t.run_id, t.previous_run_id, t.phase) for t in turns] == [
        ('r1', None, 'answered'),
        ('r2', 'r1', 'failed'),
        ('r3', 'r1', 'answered'),
    ]


async def test_pending_turn_with_different_text_than_any_later_turn_stays_pending() -> None:
    """A failed turn the user abandoned stays visible: the next turn builds on the turn before
    it, and the reader never hides a turn on its own."""
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    agent = _flaky_agent(registry, store, fail_on_call=2)

    await agent.run('first prompt', deps=_deps(), conversation_id='c1', run_id='r1')
    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('second prompt', deps=_deps(previous='r1'), conversation_id='c1')
    # The user gives up on "second prompt" and asks something unrelated instead.
    await agent.run('a completely different question', deps=_deps(previous='r1'), conversation_id='c1')

    transcript = await load_transcript(store, conversation_id='c1')
    assert [t.phase for t in transcript] == ['answered', 'failed', 'answered']
    assert 'second prompt' in str(transcript[1].messages)


async def test_mid_tool_crash_leaves_the_prompt_as_the_turn() -> None:
    """A tool that raises fires AFTER a model response (a `ToolCallPart`), and the harness's own
    at-failure save lands `interrupted` with that partial response. The turn is the saved
    prompt alone: the interrupted save is not read, so a run building on this turn starts over
    from the prompt instead of repeating a tool call whose effect is unknown.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='boom_tool', args={}, tool_call_id='tc1')])
        return ModelResponse(parts=[TextPart(content='should not get here')])

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )

    @agent.tool
    async def boom_tool(ctx: RunContext[AppDeps]) -> str:
        raise RuntimeError('tool boom')

    with pytest.raises(RuntimeError, match='tool boom'):
        await agent.run('do the thing', deps=_deps(), conversation_id='c1')

    transcript = await load_transcript(store, conversation_id='c1')
    assert len(transcript) == 1
    assert transcript[0].phase == 'failed'
    flattened = str(transcript[0].messages)
    assert 'do the thing' in flattened  # the prompt
    assert 'boom_tool' not in flattened  # the partial tool call is not part of the turn


async def test_a_prompt_rewriting_capability_reaches_the_model_not_storage() -> None:
    """A `before_model_request` capability that rewrites the user's prompt (e.g. a
    redaction/safety processor) changes what the model receives; pydantic-ai does not write a
    hook's reassigned list back into run state, so the `complete` snapshot keeps the original
    text. The transcript reader trusts a complete snapshot without a second store read -- pinned
    with a store whose `list_snapshots` raises if ever called.
    """

    class _FailsIfListSnapshotsCalledStore(InMemoryStepStore):
        async def list_snapshots(self, *, run_id: str, include_interrupted: bool = False) -> list[ContinuableSnapshot]:
            raise AssertionError('list_snapshots must not be called for a step>=1 complete snapshot')

    @dataclass
    class _RedactingCapability(AbstractCapability[AppDeps]):
        async def before_model_request(
            self, ctx: RunContext[AppDeps], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            rewritten: list[ModelMessage] = []
            for message in request_context.messages:
                if isinstance(message, ModelRequest):
                    parts = [
                        replace(part, content=f'[redacted] {part.content}')
                        if isinstance(part, UserPromptPart)
                        else part
                        for part in message.parts
                    ]
                    rewritten.append(replace(message, parts=parts))
                else:
                    rewritten.append(message)
            request_context.messages = rewritten
            return request_context

    store = _FailsIfListSnapshotsCalledStore()
    registry = InMemoryConversationRegistry()
    seen: list[list[ModelMessage]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart(content='ok')])

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
            _RedactingCapability(),
        ],
    )

    result = await agent.run('sensitive prompt', deps=_deps(), conversation_id='c1')
    assert result.output == 'ok'
    assert '[redacted] sensitive prompt' in str(seen[0])  # the rewrite reached the model

    transcript = await load_transcript(store, conversation_id='c1')
    assert len(transcript) == 1
    assert transcript[0].phase == 'answered'
    assert '[redacted]' not in str(transcript[0].messages)  # but not storage
    assert 'sensitive prompt' in str(transcript[0].messages)


async def test_a_before_model_request_hook_of_another_capability_sees_the_whole_conversation() -> None:
    """History lives in run state from `before_run` on, so a hook of a capability registered
    beside this one sees the conversation in the request and in `ctx.messages` alike. Its
    view grows with the conversation; the stored snapshots do not.
    """
    request_sizes: list[int] = []
    state_sizes: list[int] = []

    @dataclass
    class _ObservingCapability(AbstractCapability[AppDeps]):
        async def before_model_request(
            self, ctx: RunContext[AppDeps], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            request_sizes.append(len(request_context.messages))
            state_sizes.append(len(ctx.messages))
            return request_context

    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content='ok')])),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
            _ObservingCapability(),
        ],
    )
    previous: str | None = None
    for index in range(3):
        result = await agent.run(f'prompt {index}', deps=_deps(previous=previous), conversation_id='c1')
        previous = result.run_id

    assert request_sizes == [1, 3, 5]
    assert state_sizes == [1, 3, 5]
    runs = await store.list_runs(conversation_id='c1')
    stored = [await store.latest_snapshot(run_id=run.run_id) for run in runs]
    assert [len(snapshot.messages) for snapshot in stored if snapshot is not None] == [2, 2, 2]


async def test_streamed_run_resolves_to_the_answer() -> None:
    """A plain `run_stream` text turn ends through `SetFinalResult`, not a settled `CallToolsNode`
    boundary, so its answer lands with `StepPersistence.after_run`'s fallback save. That save is
    the run's latest, so the turn is the answer, not the prompt the frontier save holds.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello '
        yield 'world'

    agent = Agent(
        FunctionModel(stream_function=stream_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )

    async with agent.run_stream('hi there', deps=_deps(), conversation_id='c1') as stream:
        await stream.get_output()

    runs = await store.list_runs(conversation_id='c1')
    assert len(runs) == 1
    snapshot = await store.latest_snapshot(run_id=runs[0].run_id)
    assert snapshot is not None

    transcript = await load_transcript(store, conversation_id='c1')
    assert len(transcript) == 1
    assert transcript[0].phase == 'answered'
    assert 'hi there' in str(transcript[0].messages) and 'hello world' in str(transcript[0].messages)


async def test_history_borne_prompt_is_stamped_and_kept_on_failure() -> None:
    """The prompt inside `message_history`: a client (e.g. the Vercel adapter, or any replaying client) sends no
    separate `user_prompt` and instead passes the new message as the trailing entry of
    `message_history`. `ctx.prompt` is `None` in that shape, so `before_model_request` stamps the
    trailing request in `ctx.messages` before the frontier save.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(
            getattr(part, 'content', None) == 'prompt in history' for part in last.parts
        ):
            raise RuntimeError('boom')
        return ModelResponse(parts=[TextPart(content='ok')])

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )

    trailing_request = ModelRequest(parts=[UserPromptPart(content='prompt in history')])
    with pytest.raises(RuntimeError, match='boom'):
        await agent.run(deps=_deps(), conversation_id='c1', message_history=[trailing_request])

    transcript = await load_transcript(store, conversation_id='c1')
    assert len(transcript) == 1
    assert transcript[0].phase == 'failed'
    (stored,) = transcript[0].messages
    assert 'prompt in history' in str(stored)
    # A trailing request built by a UI adapter carries no timestamp; the capability fills one in and
    # stamps the run, so the turn is this run's and renders under its own id.
    assert stored.timestamp is not None
    assert stored.run_id == transcript[0].run_id


async def test_history_borne_prompt_gets_a_fresh_timestamp() -> None:
    """A trailing request that already carries a timestamp is re-stamped. A retry re-sends the
    previous attempt's timestamp (the Vercel adapter round-trips it through the UI message
    metadata), so a kept stamp would give every attempt of one message the same UI message id.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('boom')

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )

    sent_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    trailing_request = ModelRequest(parts=[UserPromptPart(content='prompt in history')], timestamp=sent_at)
    with pytest.raises(RuntimeError, match='boom'):
        await agent.run(deps=_deps(), conversation_id='c1', message_history=[trailing_request])

    transcript = await load_transcript(store, conversation_id='c1')
    (stored,) = transcript[0].messages
    assert stored.timestamp is not None
    assert stored.timestamp > sent_at
    assert stored.run_id == transcript[0].run_id


async def test_rejected_run_persists_nothing() -> None:
    """A run rejected in `before_run` (foreign-namespace `conversation_id`) persists nothing: the
    rejection happens before `StepPersistence.before_run` and before any model request.
    """
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    await agent.run('first prompt', deps=_deps('alice', 't1'), conversation_id='c1', run_id='run-owner')

    with pytest.raises(AgentError) as exc_info:
        await agent.run('intrude', deps=_deps('mallory', 't2'), conversation_id='c1', run_id='run-intruder')
    assert exc_info.value.code == 'conversations/not-found'

    snapshot = await store.latest_snapshot(run_id='run-intruder', include_interrupted=True)
    assert snapshot is None


async def test_rejected_run_leaves_no_run_record() -> None:
    """Under the feature's `[ConversationsCapability, StepPersistence]` order, a run rejected at
    the namespace wall never reaches `StepPersistence.before_run`: no `RunRecord`, no event, no
    snapshot."""
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content='ok')])),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )
    await agent.run('first prompt', deps=_deps('alice', 't1'), conversation_id='c1', run_id='run-owner')

    with pytest.raises(AgentError) as exc_info:
        await agent.run('intrude', deps=_deps('mallory', 't2'), conversation_id='c1', run_id='run-intruder')
    assert exc_info.value.code == 'conversations/not-found'

    assert await store.get_run(run_id='run-intruder') is None
    assert await store.list_events(run_id='run-intruder') == []
    assert await store.latest_snapshot(run_id='run-intruder', include_interrupted=True) is None


async def test_register_run_failure_leaves_a_started_turn_and_no_content() -> None:
    """`StepPersistence.before_run` runs after the capability's, so a `register_run` failure
    leaves the turn recorded with no `RunRecord` and no content behind it. The turn stays
    `started`: no terminal hook runs for a run whose `before_run` chain failed. The transcript
    lists runs from the step store and never shows it."""

    class _RejectsRegisterStore(InMemoryStepStore):
        async def register_run(self, record: RunRecord) -> None:
            if record.run_id == 'boom-run':
                raise RuntimeError('register backend down')
            await super().register_run(record)

    store = _RejectsRegisterStore()
    registry = InMemoryConversationRegistry()
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content='ok')])),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )

    with pytest.raises(RuntimeError, match='register backend down'):
        await agent.run('hello', deps=_deps(), conversation_id='c1', run_id='boom-run')

    assert await store.get_run(run_id='boom-run') is None
    assert await store.latest_snapshot(run_id='boom-run', include_interrupted=True) is None
    namespace = build_namespace(SCOPE, _deps().user, {})
    assert [(t.run_id, t.phase) for t in await registry.list_turns(namespace, 'c1')] == [('boom-run', 'started')]
    assert await load_transcript(store, conversation_id='c1') == []


# ---------------------------------------------------------------------------------------------
# Part B: in-seam compaction
# ---------------------------------------------------------------------------------------------


async def test_agent_level_compaction_bounds_what_the_model_sees() -> None:
    """Harness compaction registered on the agent works as it does on any pydantic-ai run: run
    state holds the conversation, so the strategy sees it and bounds it.

    8 turns, `SlidingWindowCompaction(max_messages=6, keep_messages=4)`. Uncompacted, the model
    sees 1, 3, 5, 7, 9, 11, 13, 15 messages. Compacted, the first three turns are identical
    (below the trigger); from turn 4 on `find_safe_cutoff` keeps the last 4 messages and
    `preserve_first_user_message` (default `True`) re-prepends the very first user turn, so every
    triggered turn settles at exactly 5.
    """
    uncompacted_registry, uncompacted_store, uncompacted_seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    uncompacted_agent = _agent(uncompacted_registry, uncompacted_store, uncompacted_seen)
    previous: str | None = None
    for i in range(8):
        result = await uncompacted_agent.run(f'turn {i}', deps=_deps(previous=previous), conversation_id='c1')
        previous = result.run_id
    assert [len(messages) for messages in uncompacted_seen] == [1, 3, 5, 7, 9, 11, 13, 15]

    compacted_store: InMemoryStepStore = InMemoryStepStore()
    compacted_seen: list[list[ModelMessage]] = []
    compacted_agent = _agent_with_capability(
        compacted_store,
        compacted_seen,
        ConversationsCapability[AppDeps](
            registry=InMemoryConversationRegistry(), step_store=compacted_store, scope=SCOPE
        ),
        extra_capabilities=(SlidingWindowCompaction(max_messages=6, keep_messages=4),),
    )
    previous = None
    for i in range(8):
        result = await compacted_agent.run(f'turn {i}', deps=_deps(previous=previous), conversation_id='c1')
        previous = result.run_id

    counts = [len(messages) for messages in compacted_seen]
    assert counts[:3] == [1, 3, 5]  # below the trigger: identical to the uncompacted run
    assert counts[-3:] == [5, 5, 5]  # growth stopped
    assert counts[-1] < len(uncompacted_seen[-1])  # strictly bounded relative to the uncompacted run


async def test_agent_level_compaction_does_not_touch_the_snapshots() -> None:
    """Harness compaction rewrites run state, but the feature's write seam cuts every snapshot
    back to the run's own turn, so the stored transcript stays complete while the model-facing
    history is capped.
    """
    inner = InMemoryStepStore()
    seen: list[list[ModelMessage]] = []
    agent = _agent_with_capability(
        inner,
        seen,
        ConversationsCapability[AppDeps](registry=InMemoryConversationRegistry(), step_store=inner, scope=SCOPE),
        extra_capabilities=(SlidingWindowCompaction(max_messages=6, keep_messages=4),),
    )
    results: list[AgentRunResult[str]] = []
    for i in range(8):
        previous = results[-1].run_id if results else None
        results.append(await agent.run(f'turn {i}', deps=_deps(previous=previous), conversation_id='c1'))

    runs = sorted(await inner.list_runs(conversation_id='c1'), key=lambda run: run.started_at)
    assert len(runs) == 8
    for run, result in zip(runs, results, strict=True):
        snapshot = await inner.latest_snapshot(run_id=run.run_id)
        assert snapshot is not None
        assert snapshot.messages == result.new_messages()


def _identity_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages


async def test_process_history_on_the_agent_keeps_snapshots_per_turn() -> None:
    """pydantic-ai's `ProcessHistory` writes its result back into `ctx.messages`; with the
    conversation in run state that is its normal job, and the feature's write seam still
    cuts every snapshot back to the run's own turn.
    """
    from pydantic_ai.capabilities import ProcessHistory

    inner = InMemoryStepStore()
    seen: list[list[ModelMessage]] = []
    agent = _agent_with_capability(
        inner,
        seen,
        ConversationsCapability(registry=InMemoryConversationRegistry(), step_store=inner, scope=SCOPE),
        extra_capabilities=(ProcessHistory[AppDeps](_identity_processor),),
    )
    previous: str | None = None
    for index in range(3):
        result = await agent.run(f'prompt {index}', deps=_deps(previous=previous), conversation_id='c1')
        previous = result.run_id

    assert [len(messages) for messages in seen] == [1, 3, 5]
    runs = await inner.list_runs(conversation_id='c1')
    stored = [await inner.latest_snapshot(run_id=run.run_id) for run in runs]
    assert [len(snapshot.messages) for snapshot in stored if snapshot is not None] == [2, 2, 2]


def test_bad_max_history_runs_value_raises_at_capability_construction() -> None:
    with pytest.raises(ValueError, match='max_history_runs must be positive'):
        ConversationsCapability(
            registry=InMemoryConversationRegistry(), step_store=InMemoryStepStore(), scope=SCOPE, max_history_runs=0
        )


class _CountingStepStore(InMemoryStepStore):
    """Counts `latest_snapshot` calls, to prove `max_history_runs` reduces store READS -- not merely
    what reaches the model, which a compaction capability already does downstream of the load and
    therefore cannot achieve (see `_loader.py`'s `load_conversation` docstring)."""

    def __init__(self) -> None:
        super().__init__()
        self.latest_snapshot_calls = 0

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        self.latest_snapshot_calls += 1
        return await super().latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)


async def test_max_history_runs_caps_reads_and_context() -> None:
    store = _CountingStepStore()
    for i in range(6):
        run_id = f'r{i}'
        await store.register_run(
            RunRecord(run_id=run_id, conversation_id='c1', started_at=datetime(2026, 8, 12, 10 + i, tzinfo=UTC))
        )
        await store.save_snapshot(
            ContinuableSnapshot(
                run_id=run_id,
                step_index=1,
                messages=[
                    ModelRequest(parts=[UserPromptPart(content=f'turn{i}')]),
                    ModelResponse(parts=[TextPart(content=f're{i}')]),
                ],
                conversation_id='c1',
            )
        )
    registry = InMemoryConversationRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    for i in range(6):
        await registry.start_turn(
            namespace, 'c1', run_id=f'r{i}', author_user_id='alice', previous_run_id=f'r{i - 1}' if i else None
        )
    capability = ConversationsCapability(registry=registry, step_store=store, scope=SCOPE, max_history_runs=2)
    ctx = replace(_bare_ctx(conversation_id='c1', run_id='r-live', previous='r5'), prompt='new')

    await capability.before_run(ctx)

    assert store.latest_snapshot_calls == 2
    flattened = str(ctx.messages)
    assert 'turn4' in flattened and 'turn5' in flattened
    assert (
        'turn0' not in flattened and 'turn1' not in flattened and 'turn2' not in flattened and 'turn3' not in flattened
    )


# --- Turn phase recording (answered / failed) -------------------------------------------------


class _UnavailableSetOutcomeRegistry(InMemoryConversationRegistry):
    """Raises on `set_turn_phase` only -- the write path a terminal hook drives degrades
    independently of `register_conversation`/`start_turn`."""

    async def set_turn_phase(
        self, namespace_key: str, conversation_id: str, *, run_id: str, phase: str, error: TurnError | None = None
    ) -> None:
        raise RegistryUnavailable('backend down')


class _DeletedMidRunSetOutcomeRegistry(InMemoryConversationRegistry):
    """Raises `ConversationNotFound` on `set_turn_phase`, as if a `DELETE` erased the entry
    between `before_run` and the terminal hook."""

    async def set_turn_phase(
        self, namespace_key: str, conversation_id: str, *, run_id: str, phase: str, error: TurnError | None = None
    ) -> None:
        raise ConversationNotFound(conversation_id)


def _raising_agent(
    registry: InMemoryConversationRegistry,
    store: InMemoryStepStore,
    describe_failure: Callable[[BaseException], TurnError | None] | None = None,
) -> Agent[AppDeps, str]:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('boom')

    return Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(
                registry=registry, step_store=store, scope=SCOPE, describe_failure=describe_failure
            ),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )


async def test_successful_run_marks_the_turn_answered() -> None:
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    await agent.run('hello', deps=_deps(), conversation_id='c1')
    namespace = build_namespace(SCOPE, _deps().user, {})
    assert [t.phase for t in await registry.list_turns(namespace, 'c1')] == ['answered']


async def test_failed_run_marks_the_turn_failed_and_propagates_the_error() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    agent = _raising_agent(registry, store)
    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('hello', deps=_deps(), conversation_id='c1')
    namespace = build_namespace(SCOPE, _deps().user, {})
    # The turn was recorded 'started' in before_run and resolved 'failed' by on_run_error -- NOT
    # left at 'started' (which would read as "still in flight" for a run that is definitively dead).
    assert [t.phase for t in await registry.list_turns(namespace, 'c1')] == ['failed']


async def test_failed_run_updates_the_listing_digest_with_what_it_stored() -> None:
    """A conversation whose only turn failed lists with that turn's activity, because its prompt is
    stored and the transcript shows it."""
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    agent = _raising_agent(registry, store)
    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('hello', deps=_deps(), conversation_id='c1')
    meta = await registry.get(build_namespace(SCOPE, _deps().user, {}), 'c1')
    assert meta is not None
    assert meta.message_count == 1
    assert meta.last_message_at is not None


async def test_failed_run_that_stored_nothing_leaves_the_listing_digest_alone() -> None:
    """A run that fails before its first model request leaves no message in the store, so the
    listing must not show activity the transcript cannot show."""
    registry = InMemoryConversationRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    capability = ConversationsCapability(registry=registry, step_store=InMemoryStepStore(), scope=SCOPE)
    ctx = _bare_ctx(conversation_id='c1', run_id='r1')

    with pytest.raises(RuntimeError, match='boom'):
        await capability.on_run_error(ctx, error=RuntimeError('boom'))

    meta = await registry.get(namespace, 'c1')
    assert meta is not None
    assert (meta.message_count, meta.last_message_at) == (0, None)


class _UnreadableStepStore(InMemoryStepStore):
    """Raises on `latest_snapshot`, as a backend does when its server is gone mid-run."""

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        raise ConnectionError('store down')


async def test_failed_run_still_raises_its_own_error_when_the_store_read_for_the_digest_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The digest update after a failure is bookkeeping: a store that cannot be read is logged, the
    listing stays as it was, and the caller sees the run's error, not the store's."""
    registry = InMemoryConversationRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    capability = ConversationsCapability(registry=registry, step_store=_UnreadableStepStore(), scope=SCOPE)
    ctx = _bare_ctx(conversation_id='c1', run_id='r1')

    with caplog.at_level('WARNING'), pytest.raises(RuntimeError, match='boom'):
        await capability.on_run_error(ctx, error=RuntimeError('boom'))

    assert 'step store read failed' in caplog.text
    meta = await registry.get(namespace, 'c1')
    assert meta is not None and meta.message_count == 0


async def test_failed_run_counts_its_turn_the_way_the_transcript_reads_it() -> None:
    """The digest counts the failed turn from its first message stamped with the run's id to the
    end, the rule the store and the transcript use, so a trailing tool return saved before
    pydantic-ai stamped it is counted too."""
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hi')], run_id='r1'),
        ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='tc1')], run_id='r1'),
        ModelRequest(parts=[ToolReturnPart(tool_name='t', content='done', tool_call_id='tc1')]),
    ]
    await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=messages))
    capability = ConversationsCapability(registry=registry, step_store=store, scope=SCOPE)

    with pytest.raises(RuntimeError, match='boom'):
        await capability.on_run_error(_bare_ctx(conversation_id='c1', run_id='r1'), error=RuntimeError('boom'))

    meta = await registry.get(namespace, 'c1')
    assert meta is not None and meta.message_count == 3


async def test_a_store_outage_the_seam_already_logged_is_not_logged_again_for_the_digest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = InMemoryConversationRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    seam = PreparingStepStore(_UnavailableReadStepStore())
    capability = ConversationsCapability(registry=registry, step_store=seam, scope=SCOPE)

    with caplog.at_level('WARNING'), pytest.raises(RuntimeError, match='boom'):
        await capability.on_run_error(_bare_ctx(conversation_id='c1', run_id='r1'), error=RuntimeError('boom'))

    assert caplog.text.count('step store latest_snapshot failed with a temporary error') == 1
    assert 'step store read failed' not in caplog.text


class _CountingUnavailableReadStepStore(_UnavailableReadStepStore):
    def __init__(self) -> None:
        super().__init__()
        self.latest_snapshot_calls = 0

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        self.latest_snapshot_calls += 1
        raise self.failure


async def test_a_run_failed_by_a_store_outage_does_not_read_the_store_again_for_the_digest() -> None:
    """When the error on its way out is the write seam's own `unavailable`, the store has just
    reported itself down; a digest read would only wait out a second driver timeout before the
    error reaches the client."""
    registry = InMemoryConversationRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    inner = _CountingUnavailableReadStepStore()
    capability = ConversationsCapability(registry=registry, step_store=PreparingStepStore(inner), scope=SCOPE)
    outage = AgentError('conversations/unavailable', 'Conversation storage is temporarily unavailable.', retryable=True)

    with pytest.raises(AgentError) as exc_info:
        await capability.on_run_error(_bare_ctx(conversation_id='c1', run_id='r1'), error=outage)

    assert exc_info.value is outage
    assert inner.latest_snapshot_calls == 0


async def test_failed_run_stores_the_reason_from_describe_failure() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    seen: list[BaseException] = []

    def describe(exc: BaseException) -> TurnError:
        seen.append(exc)
        return TurnError('model-unavailable', 'The model is temporarily unavailable.', transient=True)

    agent = _raising_agent(registry, store, describe)
    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('hello', deps=_deps(), conversation_id='c1')
    namespace = build_namespace(SCOPE, _deps().user, {})
    [turn] = await registry.list_turns(namespace, 'c1')
    assert turn.error == TurnError('model-unavailable', 'The model is temporarily unavailable.', transient=True)
    assert [str(exc) for exc in seen] == ['boom']


async def test_a_raising_describe_failure_stores_no_reason_and_keeps_the_run_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()

    def describe(exc: BaseException) -> TurnError:
        raise ValueError('mapper bug')

    agent = _raising_agent(registry, store, describe)
    with caplog.at_level('WARNING'), pytest.raises(RuntimeError, match='boom'):
        await agent.run('hello', deps=_deps(), conversation_id='c1')
    namespace = build_namespace(SCOPE, _deps().user, {})
    assert [(t.phase, t.error) for t in await registry.list_turns(namespace, 'c1')] == [('failed', None)]
    assert 'describe_failure raised' in caplog.text


def test_agent_error_mapper_translates_with_the_app_error_handlers() -> None:
    class QuotaExceeded(Exception):
        pass

    describe = agent_error_mapper(((QuotaExceeded, AgentError('limits/exceeded', 'Quota used up.')),))
    assert describe(QuotaExceeded()) == TurnError('limits/exceeded', 'Quota used up.', transient=False)
    assert describe(AgentError('x/busy', 'Busy.', retryable=True)) == TurnError('x/busy', 'Busy.', transient=True)
    # Without a matching handler the no-leak fallback applies, as on the live path.
    assert describe(RuntimeError('secret detail')) == TurnError('internal', 'An internal error occurred.')


def test_agent_error_mapper_names_cancellation_and_skips_other_base_exceptions() -> None:
    describe = agent_error_mapper()
    assert describe(asyncio.CancelledError()) == TurnError(
        'conversations/cancelled', 'The run was cancelled.', transient=True
    )
    assert describe(KeyboardInterrupt()) is None


async def test_on_run_error_is_a_noop_when_conversation_id_is_none() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    capability = ConversationsCapability(registry=registry, step_store=store, scope=SCOPE)
    ctx = _bare_ctx(conversation_id=None)

    with pytest.raises(RuntimeError, match='boom'):
        await capability.on_run_error(ctx, error=RuntimeError('boom'))

    assert await registry.list(build_namespace(SCOPE, _deps().user, {})) == []


async def test_on_run_error_reraises_the_original_error_even_when_the_registry_update_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A registry hiccup on the error path must never mask the ORIGINAL run error -- the caller has
    to see why the run failed, not a secondary bookkeeping failure. The phase update is best-
    effort: logged, then the original error re-raised unchanged.
    """
    flaky = _UnavailableSetOutcomeRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await flaky.register_conversation(namespace, 'c1', created_by='alice')
    capability = ConversationsCapability(registry=flaky, step_store=InMemoryStepStore(), scope=SCOPE)
    ctx = _bare_ctx(conversation_id='c1', run_id='r1')

    with caplog.at_level('WARNING'), pytest.raises(RuntimeError, match='boom'):
        await capability.on_run_error(ctx, error=RuntimeError('boom'))

    assert 'failed to record run phase' in caplog.text


async def test_after_run_logs_and_continues_when_setting_the_outcome_is_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    real_result = await agent.run('hello', deps=_deps())
    flaky = _UnavailableSetOutcomeRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await flaky.register_conversation(namespace, 'c1', created_by='alice')
    capability = ConversationsCapability(registry=flaky, step_store=InMemoryStepStore(), scope=SCOPE)
    ctx = _bare_ctx(conversation_id='c1', run_id='r1')

    with caplog.at_level('WARNING'):
        returned = await capability.after_run(ctx, result=real_result)

    assert returned is real_result
    assert 'failed to record run phase' in caplog.text


async def test_record_outcome_logs_and_continues_when_the_conversation_was_deleted_mid_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A `DELETE` racing a failing run: the entry is gone by the time `on_run_error` records the
    phase, so `set_turn_phase` raises `ConversationNotFound`. That is logged and swallowed --
    the original run error must still propagate unchanged, not be replaced by the bookkeeping miss.
    """
    flaky = _DeletedMidRunSetOutcomeRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await flaky.register_conversation(namespace, 'c1', created_by='alice')
    capability = ConversationsCapability(registry=flaky, step_store=InMemoryStepStore(), scope=SCOPE)
    ctx = _bare_ctx(conversation_id='c1', run_id='r1')

    with caplog.at_level('INFO'), pytest.raises(RuntimeError, match='boom'):
        await capability.on_run_error(ctx, error=RuntimeError('boom'))

    assert 'gone while recording run phase' in caplog.text


# --- Turn lineage: `previous_run_id` names the turn a run builds on ---------------------------


async def test_second_turn_without_previous_run_id_is_rejected_and_persists_nothing() -> None:
    """A left-out `previous_run_id` is a request error: the run is rejected before its turn is
    recorded or its prompt stored."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    await agent.run('first prompt', deps=_deps(), conversation_id='c1', run_id='r1')

    with pytest.raises(AgentError) as exc_info:
        await agent.run('second prompt', deps=_deps(), conversation_id='c1', run_id='r2')

    assert exc_info.value.code == 'conversations/previous-run-required'
    assert exc_info.value.http_status == 422
    namespace = build_namespace(SCOPE, _deps().user, {})
    assert [t.run_id for t in await registry.list_turns(namespace, 'c1')] == ['r1']
    assert await store.latest_snapshot(run_id='r2', include_interrupted=True) is None


@pytest.mark.parametrize('previous', ['never-recorded', 'r1'])
async def test_a_previous_run_id_that_is_not_a_turn_of_the_conversation_is_rejected(previous: str) -> None:
    """Unknown ids, and any id on the first turn of a conversation (there is nothing to build on
    yet), answer `run-not-found`; a run of another conversation is as unknown as a made-up id."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    await agent.run('elsewhere', deps=_deps(), conversation_id='other', run_id='r1')

    with pytest.raises(AgentError) as exc_info:
        await agent.run('hello', deps=_deps(previous=previous), conversation_id='c1', run_id='r2')

    assert exc_info.value.code == 'conversations/run-not-found'
    assert exc_info.value.http_status == 404
    assert await store.latest_snapshot(run_id='r2', include_interrupted=True) is None


async def test_a_fork_builds_on_the_named_turn_and_leaves_the_later_ones_aside() -> None:
    """Naming an earlier turn starts a branch: the run sees that turn's history and nothing of
    the turns recorded after it; every branch stays in the registry and the transcript."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    await agent.run('first prompt', deps=_deps(), conversation_id='c1', run_id='r1')
    await agent.run('second prompt', deps=_deps(previous='r1'), conversation_id='c1', run_id='r2')

    await agent.run('second prompt, edited', deps=_deps(previous='r1'), conversation_id='c1', run_id='r3')

    assert 'first prompt' in str(seen[-1]) and 'second prompt' not in str(seen[-1][:-1])
    namespace = build_namespace(SCOPE, _deps().user, {})
    turns = await registry.list_turns(namespace, 'c1')
    assert [(t.run_id, t.previous_run_id) for t in turns] == [('r1', None), ('r2', 'r1'), ('r3', 'r1')]
    transcript = await load_transcript(store, conversation_id='c1', turns=turns)
    assert [(t.run_id, t.previous_run_id) for t in transcript] == [('r1', None), ('r2', 'r1'), ('r3', 'r1')]


async def test_the_chain_is_read_in_order_and_a_cycle_cannot_loop() -> None:
    """`_resolve_chain` on records alone: first turn first, and a corrupted link back into the
    chain ends the walk instead of looping."""
    from pydantic_ai_conversations._lineage import resolve_chain
    from pydantic_ai_conversations.registry import TurnRecord

    records = [
        TurnRecord(run_id='r1', author_user_id='a'),
        TurnRecord(run_id='r2', author_user_id='a', previous_run_id='r1'),
        TurnRecord(run_id='r3', author_user_id='a', previous_run_id='r2'),
    ]
    assert resolve_chain(records, 'r3') == ['r1', 'r2', 'r3']
    assert resolve_chain(records, 'r2') == ['r1', 'r2']
    assert resolve_chain([], None) == []
    looped = [*records, TurnRecord(run_id='r4', author_user_id='a', previous_run_id='r4')]
    assert resolve_chain(looped, 'r4') == ['r4']


def test_resolve_chain_raises_package_errors_that_the_capability_maps() -> None:
    """The resolver knows nothing of HTTP: it raises the package's own errors, and `before_run`
    turns them into the `previous-run-required` and `run-not-found` responses."""
    from pydantic_ai_conversations._lineage import UNSET, PreviousRunRequired, RunNotFound, resolve_chain
    from pydantic_ai_conversations.registry import TurnRecord

    records = [TurnRecord(run_id='r1', author_user_id='a')]
    with pytest.raises(PreviousRunRequired):
        resolve_chain(records, UNSET)
    with pytest.raises(RunNotFound):
        resolve_chain(records, 'r9')
    with pytest.raises(RunNotFound):
        resolve_chain(records, 7)


async def test_an_explicit_none_starts_from_nothing_after_a_failed_first_turn() -> None:
    """A retry of a first turn has no turn to name: it sends `previous_run_id: null`, which is
    accepted where an omitted field is not, and starts a second root."""
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    agent = _flaky_agent(registry, store, fail_on_call=1)
    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('hello', deps=_deps(), conversation_id='c1', run_id='r1')

    await agent.run('hello', deps=_deps(previous=None), conversation_id='c1', run_id='r2')

    namespace = build_namespace(SCOPE, _deps().user, {})
    turns = await registry.list_turns(namespace, 'c1')
    assert [(t.run_id, t.previous_run_id, t.phase) for t in turns] == [('r1', None, 'failed'), ('r2', None, 'answered')]


# --- Resuming a failed turn: the run that names it continues from what it left ----------------


def _resumable_agent(
    registry: InMemoryConversationRegistry, store: InMemoryStepStore, seen: list[list[ModelMessage]]
) -> tuple[Agent[AppDeps, str], dict[str, bool]]:
    """Answers from a tool round; `faults` crash the model before the first response or after
    the tool round, once each, so a turn can be left with only its prompt or with settled work."""
    faults = {'before_first': False, 'after_tool': False}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
            if faults['after_tool']:
                faults['after_tool'] = False
                raise RuntimeError('down after tool')
            return ModelResponse(parts=[TextPart(content='answer')])
        if faults['before_first']:
            faults['before_first'] = False
            raise RuntimeError('down before first response')
        return ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id=f'tc-{len(seen)}')])

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )

    @agent.tool
    async def lookup(ctx: RunContext[AppDeps]) -> str:
        return 'result'

    return agent, faults


def _prompts(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


async def _chain_of(
    store: InMemoryStepStore, registry: InMemoryConversationRegistry, run_id: str
) -> list[ModelMessage]:
    """What a run naming `run_id` would be loaded with: the chain's snapshots, assembled."""
    from pydantic_ai_conversations._lineage import resolve_chain

    namespace = build_namespace(SCOPE, _deps().user, {})
    records = await registry.list_turns(namespace, 'c1')
    chain = resolve_chain(records, run_id)
    answered = {record.run_id for record in records if record.phase == 'answered'}
    return await load_conversation(store, conversation_id='c1', run_ids=chain, answered_runs=answered)


async def _failed_before_first_response(
    registry: InMemoryConversationRegistry, store: InMemoryStepStore, seen: list[list[ModelMessage]]
) -> Agent[AppDeps, str]:
    agent, faults = _resumable_agent(registry, store, seen)
    await agent.run('first', deps=_deps(), conversation_id='c1', run_id='r1')
    faults['before_first'] = True
    with pytest.raises(RuntimeError, match='down before first response'):
        await agent.run('second', deps=_deps(previous='r1'), conversation_id='c1', run_id='r2')
    return agent


async def test_resuming_without_a_prompt_continues_from_the_saved_prompt() -> None:
    """A Vercel client resumes a failed turn by naming it and sending an empty list: the model
    gets the turn's prompt, and the new run stores only its answer; the prompt stays the failed
    turn's."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = await _failed_before_first_response(registry, store, seen)

    await agent.run(deps=_deps(previous='r2'), conversation_id='c1', run_id='r3', message_history=[])

    assert _prompts(seen[-1]) == ['first', 'second']
    stored = await store.latest_snapshot(run_id='r3')
    assert stored is not None and _prompts(stored.messages) == [] and stored.messages[0].kind == 'response'
    assert _prompts(await _chain_of(store, registry, 'r3')) == ['first', 'second']
    turns = await load_transcript(store, conversation_id='c1')
    assert [(t.run_id, t.phase, _prompts(t.messages)) for t in turns] == [
        ('r1', 'answered', ['first']),
        ('r2', 'failed', ['second']),
        ('r3', 'answered', []),
    ]


async def test_resuming_with_a_prompt_passed_separately_keeps_the_saved_one_in_context() -> None:
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = await _failed_before_first_response(registry, store, seen)

    await agent.run('more', deps=_deps(previous='r2'), conversation_id='c1', run_id='r3')

    assert _prompts(seen[-1]) == ['first', 'second', 'more']
    stored = await store.latest_snapshot(run_id='r3')
    assert stored is not None and _prompts(stored.messages) == ['more']
    assert _prompts(await _chain_of(store, registry, 'r3')) == ['first', 'second', 'more']


async def test_resuming_with_a_prompt_inside_the_history_stores_the_same_turn() -> None:
    """The UI adapters' path: the new prompt arrives inside `message_history` and pydantic-ai merges
    it with the loaded open prompt into one request; the capability splits that request back, so
    the stored turn and the branch read the same as with a separately passed prompt."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = await _failed_before_first_response(registry, store, seen)

    await agent.run(
        deps=_deps(previous='r2'),
        conversation_id='c1',
        run_id='r3',
        message_history=[ModelRequest(parts=[UserPromptPart(content='more')])],
    )

    assert _prompts(seen[-1]) == ['first', 'second', 'more']
    stored = await store.latest_snapshot(run_id='r3')
    assert stored is not None and _prompts(stored.messages) == ['more']
    assert _prompts(await _chain_of(store, registry, 'r3')) == ['first', 'second', 'more']


async def test_resending_the_saved_prompt_sends_it_twice() -> None:
    # Not guarded: a client that re-sends what the server already holds gets it twice.
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = await _failed_before_first_response(registry, store, seen)

    await agent.run(
        deps=_deps(previous='r2'),
        conversation_id='c1',
        run_id='r3',
        message_history=[ModelRequest(parts=[UserPromptPart(content='second')])],
    )

    assert _prompts(seen[-1]) == ['first', 'second', 'second']


async def test_continuing_a_turn_that_left_two_open_prompts_stores_only_the_new_text() -> None:
    """r3 continued failed r2 and failed before answering itself, so the branch ends in two open
    prompts. r4 sends new text inside `message_history`; pydantic-ai merges all three into one
    request, and the split restores r2's and r3's prompts with their stamps."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent, faults = _resumable_agent(registry, store, seen)
    await agent.run('first', deps=_deps(), conversation_id='c1', run_id='r1')
    faults['before_first'] = True
    with pytest.raises(RuntimeError, match='down before first response'):
        await agent.run('second', deps=_deps(previous='r1'), conversation_id='c1', run_id='r2')
    faults['before_first'] = True
    with pytest.raises(RuntimeError, match='down before first response'):
        await agent.run(
            deps=_deps(previous='r2'),
            conversation_id='c1',
            run_id='r3',
            message_history=[ModelRequest(parts=[UserPromptPart(content='third')])],
        )
    stored = await store.latest_snapshot(run_id='r3')
    assert stored is not None and _prompts(stored.messages) == ['third']

    await agent.run(
        deps=_deps(previous='r3'),
        conversation_id='c1',
        run_id='r4',
        message_history=[ModelRequest(parts=[UserPromptPart(content='fourth')])],
    )

    assert _prompts(seen[-1]) == ['first', 'second', 'third', 'fourth']
    stored = await store.latest_snapshot(run_id='r4')
    assert stored is not None and _prompts(stored.messages) == ['fourth']
    chain = await _chain_of(store, registry, 'r4')
    assert _prompts(chain) == ['first', 'second', 'third', 'fourth']
    prompts = [m for m in chain if isinstance(m, ModelRequest) and any(isinstance(p, UserPromptPart) for p in m.parts)]
    assert [m.run_id for m in prompts] == ['r1', 'r2', 'r3', 'r4']


async def test_resuming_a_turn_that_left_two_open_prompts_stores_only_the_answer() -> None:
    """A Vercel client resumes with an empty list after two failed attempts; pydantic-ai merges
    the two loaded prompts into one request on its own, and the split hands them back to r2 and
    r3, so r4 stores only its answer."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent, faults = _resumable_agent(registry, store, seen)
    await agent.run('first', deps=_deps(), conversation_id='c1', run_id='r1')
    faults['before_first'] = True
    with pytest.raises(RuntimeError, match='down before first response'):
        await agent.run('second', deps=_deps(previous='r1'), conversation_id='c1', run_id='r2')
    faults['before_first'] = True
    with pytest.raises(RuntimeError, match='down before first response'):
        await agent.run('third', deps=_deps(previous='r2'), conversation_id='c1', run_id='r3')

    await agent.run(deps=_deps(previous='r3'), conversation_id='c1', run_id='r4', message_history=[])

    assert _prompts(seen[-1]) == ['first', 'second', 'third']
    stored = await store.latest_snapshot(run_id='r4')
    assert stored is not None and _prompts(stored.messages) == []
    assert _prompts(await _chain_of(store, registry, 'r4')) == ['first', 'second', 'third']


async def test_resuming_a_turn_that_settled_a_tool_round_answers_from_its_result() -> None:
    """The run building on a failed turn with a settled tool round answers from the stored tool
    result without calling the tool again; the tool return stays the failed turn's, and the branch
    carries it once."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent, faults = _resumable_agent(registry, store, seen)
    await agent.run('first', deps=_deps(), conversation_id='c1', run_id='r1')
    faults['after_tool'] = True
    with pytest.raises(RuntimeError, match='down after tool'):
        await agent.run('second', deps=_deps(previous='r1'), conversation_id='c1', run_id='r2')
    tool_calls_so_far = sum(1 for request in seen if isinstance(request[-1], ModelRequest))

    await agent.run(
        deps=_deps(previous='r2'),
        conversation_id='c1',
        run_id='r3',
        message_history=[ModelRequest(parts=[UserPromptPart(content='more')])],
    )

    assert sum(1 for request in seen if isinstance(request[-1], ModelRequest)) == tool_calls_so_far + 1
    stored = await store.latest_snapshot(run_id='r3')
    assert stored is not None
    assert [type(part).__name__ for part in stored.messages[0].parts] == ['UserPromptPart']
    chain = await _chain_of(store, registry, 'r3')
    assert sum(1 for m in chain if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)) == 2
    from pydantic_ai_harness.step_persistence import is_provider_valid

    assert is_provider_valid(chain)


@pytest.mark.parametrize(
    'sent', [[], [ModelRequest(parts=[UserPromptPart(content='more')])]], ids=['resume', 'new-text']
)
async def test_a_continuation_does_not_store_a_tool_return_the_failed_turn_saved_unstamped(
    sent: list[ModelMessage],
) -> None:
    """A run killed between settling a tool round and sending its returns leaves them stored
    without a run id. The continuation loads them as the failed turn's, so it stores only its own
    message and the branch carries each tool return once."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent, faults = _resumable_agent(registry, store, seen)
    await agent.run('first', deps=_deps(), conversation_id='c1', run_id='r1')
    faults['after_tool'] = True
    with pytest.raises(RuntimeError, match='down after tool'):
        await agent.run('second', deps=_deps(previous='r1'), conversation_id='c1', run_id='r2')
    failed = await store.latest_snapshot(run_id='r2')
    assert failed is not None
    *settled, returns = failed.messages
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r2',
            step_index=failed.step_index + 1,
            messages=[*settled, replace(returns, run_id=None)],
            conversation_id='c1',
        )
    )

    await agent.run(deps=_deps(previous='r2'), conversation_id='c1', run_id='r3', message_history=list(sent))

    stored = await store.latest_snapshot(run_id='r3')
    assert stored is not None
    first = stored.messages[0]
    if sent:
        assert isinstance(first, ModelRequest)
        assert [type(part).__name__ for part in first.parts] == ['UserPromptPart']
    else:
        assert isinstance(first, ModelResponse)
    chain = await _chain_of(store, registry, 'r3')
    assert sum(1 for m in chain if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)) == 2
    from pydantic_ai_harness.step_persistence import is_provider_valid

    assert is_provider_valid(chain)


def _approval_agent(
    registry: InMemoryConversationRegistry, store: InMemoryStepStore, seen: list[list[ModelMessage]]
) -> Agent[AppDeps, str | DeferredToolRequests]:
    """Calls a tool that needs approval, then answers once its result is in."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='delete_file', args={'path': 'x'}, tool_call_id='tc-del')])

    agent: Agent[AppDeps, str | DeferredToolRequests] = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        output_type=[str, DeferredToolRequests],
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )

    @agent.tool_plain(requires_approval=True)
    def delete_file(path: str) -> str:
        return f'deleted {path}'

    return agent


async def test_a_turn_paused_for_approval_shows_its_calls_and_the_approval_continues_it() -> None:
    """A run that ends in tool calls awaiting approval is a finished run (`answered`) whose newest
    snapshot the harness marks `interrupted`, the calls being open. Both readers take it anyway:
    the transcript shows the pending calls, and the run that names the turn with the approval
    executes the tool and answers, storing the tool return as its own."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _approval_agent(registry, store, seen)

    paused = await agent.run('delete x', deps=_deps(), conversation_id='c1', run_id='r1')
    assert isinstance(paused.output, DeferredToolRequests)
    namespace = build_namespace(SCOPE, _deps().user, {})
    records = await registry.list_turns(namespace, 'c1')
    assert [record.phase for record in records] == ['answered']
    (turn,) = await load_transcript(store, conversation_id='c1', turns=records)
    assert [type(message).__name__ for message in turn.messages] == ['ModelRequest', 'ModelResponse']
    assert (await load_transcript(store, conversation_id='c1'))[0].messages == turn.messages[:1]  # store-only read

    approved = await agent.run(
        deps=_deps(previous='r1'),
        conversation_id='c1',
        run_id='r2',
        message_history=[],
        deferred_tool_results=DeferredToolResults(approvals={'tc-del': True}),
    )

    assert approved.output == 'done'
    assert [type(message).__name__ for message in seen[-1]] == ['ModelRequest', 'ModelResponse', 'ModelRequest']
    stored = await store.latest_snapshot(run_id='r2')
    assert stored is not None
    assert [part.part_kind for message in stored.messages for part in message.parts] == ['tool-return', 'text']
    from pydantic_ai_harness.step_persistence import is_provider_valid

    chain = await _chain_of(store, registry, 'r2')
    assert is_provider_valid(chain) and len(chain) == 4
    turns = await load_transcript(store, conversation_id='c1', turns=await registry.list_turns(namespace, 'c1'))
    assert [(t.run_id, t.phase, t.previous_run_id) for t in turns] == [
        ('r1', 'answered', None),
        ('r2', 'answered', 'r1'),
    ]


async def test_a_turn_paused_for_approval_is_counted_the_way_the_transcript_shows_it() -> None:
    """The digest reads an answered turn as the transcript does, its newest snapshot `interrupted`
    or not, so a turn paused on open tool calls counts the response holding them too."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _approval_agent(registry, store, seen)

    paused = await agent.run('delete x', deps=_deps(), conversation_id='c1', run_id='r1')
    assert isinstance(paused.output, DeferredToolRequests)
    namespace = build_namespace(SCOPE, _deps().user, {})
    (turn,) = await load_transcript(store, conversation_id='c1', turns=await registry.list_turns(namespace, 'c1'))
    meta = await registry.get(namespace, 'c1')
    assert meta is not None and meta.message_count == len(turn.messages) == 2


async def test_a_run_cut_off_mid_call_is_not_mistaken_for_a_pause() -> None:
    """The same snapshot shape, a response with open tool calls, under `phase='failed'` stays
    unread: a continuation starts from the last complete snapshot, never from a tool call whose
    effect is unknown."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _approval_agent(registry, store, seen)

    @agent.tool_plain
    def crash(path: str) -> str:
        raise RuntimeError('tool crashed')

    agent._function_toolset.tools['delete_file'] = agent._function_toolset.tools.pop('crash')  # pyright: ignore[reportPrivateUsage]
    agent._function_toolset.tools['delete_file'].name = 'delete_file'  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError, match='tool crashed'):
        await agent.run('delete x', deps=_deps(), conversation_id='c1', run_id='r1')
    namespace = build_namespace(SCOPE, _deps().user, {})
    records = await registry.list_turns(namespace, 'c1')
    assert [record.phase for record in records] == ['failed']
    (turn,) = await load_transcript(store, conversation_id='c1', turns=records)
    assert _prompts(turn.messages) == ['delete x'] and len(turn.messages) == 1
    assert _prompts(await _chain_of(store, registry, 'r1')) == ['delete x']


async def test_a_request_without_a_prompt_on_an_answered_turn_is_rejected() -> None:
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent, _ = _resumable_agent(registry, store, seen)
    await agent.run('first', deps=_deps(), conversation_id='c1', run_id='r1')

    with pytest.raises(AgentError) as exc_info:
        await agent.run(deps=_deps(previous='r1'), conversation_id='c1', run_id='r2', message_history=[])

    assert exc_info.value.code == 'conversations/prompt-required'
    assert exc_info.value.http_status == 422
    namespace = build_namespace(SCOPE, _deps().user, {})
    assert [t.run_id for t in await registry.list_turns(namespace, 'c1')] == ['r1']


async def test_a_first_request_without_a_prompt_is_rejected() -> None:
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent, _ = _resumable_agent(registry, store, seen)

    with pytest.raises(AgentError) as exc_info:
        await agent.run(deps=_deps(), conversation_id='c1', run_id='r1', message_history=[])

    assert exc_info.value.code == 'conversations/prompt-required'
    assert await registry.list(build_namespace(SCOPE, _deps().user, {})) != []  # registered, no turn
    assert await registry.list_turns(build_namespace(SCOPE, _deps().user, {}), 'c1') == []


def test_resumable_recognizes_open_tool_calls_and_suspended_turns() -> None:
    from pydantic_ai_conversations._lineage import is_resumable

    call = ModelResponse(parts=[ToolCallPart(tool_name='t', args={}, tool_call_id='tc')])
    suspended = ModelResponse(parts=[TextPart(content='...')], state='suspended')
    answer = ModelResponse(parts=[TextPart(content='done')])
    request = ModelRequest(parts=[ToolReturnPart(tool_name='final_result', content='done', tool_call_id='tc')])
    assert is_resumable([call]) and is_resumable([suspended])
    assert not is_resumable([answer]) and not is_resumable([])
    assert is_resumable([request]) and not is_resumable([request], answered=True)


async def test_a_stream_that_fails_on_open_leaves_the_prompt_as_the_failed_turn() -> None:
    """A streamed model call that fails before its first part makes pydantic-ai append an empty
    `interrupted` response, which the harness saves as `complete`; the write seam drops it, so
    the failed turn is its prompt and a resume without a new message re-sends it."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    fail = {'next': False}

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        seen.append(list(messages))
        if fail['next']:
            fail['next'] = False
            raise RuntimeError('stream refused')
        yield 'ok'

    agent = Agent(
        FunctionModel(stream_function=stream_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )
    async with agent.run_stream('first', deps=_deps(), conversation_id='c1', run_id='r1') as stream:
        await stream.get_output()
    fail['next'] = True
    with pytest.raises(RuntimeError, match='stream refused'):
        async with agent.run_stream('second', deps=_deps(previous='r1'), conversation_id='c1', run_id='r2') as stream:
            await stream.get_output()

    stored = await store.latest_snapshot(run_id='r2')
    assert stored is not None and [m.kind for m in stored.messages] == ['request']

    async with agent.run_stream(
        deps=_deps(previous='r2'), conversation_id='c1', run_id='r3', message_history=[]
    ) as stream:
        await stream.get_output()
    assert _prompts(seen[-1]) == ['first', 'second']
    assert seen[-1][-1].kind == 'request'


async def test_an_answered_tool_output_turn_needs_a_new_prompt_to_continue() -> None:
    """With `ToolOutput` a run ends in the final-result tool's return, a request. That turn is
    answered, so a continuation sending nothing new has nothing to resume and is rejected like one
    after a plain answer, instead of calling the model again on the finished turn."""

    class Answer(BaseModel):
        text: str

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args={'text': 'done'}, tool_call_id='t1')])

    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    feature = ConversationsFeature(registry=registry, step_store=store, scope=('user.tenant_id', 'user.user_id'))
    agent: Agent[AppDeps, Answer] = Agent(
        FunctionModel(model_fn), deps_type=AppDeps, output_type=ToolOutput(Answer), capabilities=feature.capabilities()
    )
    first = await agent.run('hello', deps=_deps(), conversation_id='c1')

    with pytest.raises(AgentError) as exc_info:
        await agent.run(None, deps=_deps(previous=first.run_id), conversation_id='c1')

    assert exc_info.value.code == 'conversations/prompt-required'
    assert [t.phase for t in await registry.list_turns(build_namespace(SCOPE, _deps().user, {}), 'c1')] == ['answered']


async def test_the_digest_counts_what_the_preparers_left_in_the_store() -> None:
    """`message_count` is read back from the stored turn after the preparer chain, on success as on
    failure, so a preparer that drops messages cannot make the listing disagree with the store."""

    class DropResponses:
        def prepare_snapshot(self, snapshot: ContinuableSnapshot) -> ContinuableSnapshot:
            return replace(snapshot, messages=[m for m in snapshot.messages if not isinstance(m, ModelResponse)])

    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    feature = ConversationsFeature(
        registry=registry,
        step_store=store,
        scope=('user.tenant_id', 'user.user_id'),
        snapshot_preparers=[DropResponses()],
    )
    agent: Agent[AppDeps, str] = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content='ok')])),
        deps_type=AppDeps,
        capabilities=feature.capabilities(),
    )
    result = await agent.run('hello', deps=_deps(), conversation_id='c1')

    snapshot = await store.latest_snapshot(run_id=result.run_id)
    meta = await registry.get(build_namespace(SCOPE, _deps().user, {}), 'c1')
    assert snapshot is not None and meta is not None
    assert meta.message_count == len(snapshot.messages) == 1
