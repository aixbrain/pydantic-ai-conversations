"""PreparingStepStore: preparer chain, empty-skip, exception propagation, delegation."""

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai_app_factory import AgentError
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    InMemoryStepStore,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)

from pydantic_ai_conversations.stores import NewMessagesPreparer, PreparingStepStore


def _snapshot(run_id: str = 'r1', **kwargs: Any) -> ContinuableSnapshot:
    defaults: dict[str, Any] = {
        'run_id': run_id,
        'step_index': 1,
        'messages': [ModelRequest(parts=[UserPromptPart(content='hi')])],
        'conversation_id': 'c1',
    }
    defaults.update(kwargs)
    return ContinuableSnapshot(**defaults)


class _TagPreparer:
    """Stamps the snapshot's agent_name so the transform (and its order) is observable."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def prepare_snapshot(self, snapshot: ContinuableSnapshot) -> ContinuableSnapshot:
        return replace(snapshot, agent_name=(snapshot.agent_name or '') + self.tag)


class _EmptyingPreparer:
    def prepare_snapshot(self, snapshot: ContinuableSnapshot) -> ContinuableSnapshot:
        return replace(snapshot, messages=[])


class _RaisingPreparer:
    def prepare_snapshot(self, snapshot: ContinuableSnapshot) -> ContinuableSnapshot:
        raise RuntimeError('policy bug')


async def test_save_runs_preparer_chain_in_order():
    inner = InMemoryStepStore()
    store = PreparingStepStore(inner, preparers=(_TagPreparer('a'), _TagPreparer('b')))
    await store.save_snapshot(_snapshot())
    saved = await inner.latest_snapshot(run_id='r1')
    assert saved is not None
    assert saved.agent_name == 'ab'


async def test_emptied_snapshot_is_not_persisted():
    inner = InMemoryStepStore()
    store = PreparingStepStore(inner, preparers=(_EmptyingPreparer(),))
    await store.save_snapshot(_snapshot())
    assert await inner.latest_snapshot(run_id='r1', include_interrupted=True) is None


async def test_preparer_exception_propagates_and_nothing_is_saved():
    inner = InMemoryStepStore()
    store = PreparingStepStore(inner, preparers=(_RaisingPreparer(),))
    with pytest.raises(RuntimeError, match='policy bug'):
        await store.save_snapshot(_snapshot())
    assert await inner.latest_snapshot(run_id='r1', include_interrupted=True) is None


async def test_empty_chain_saves_unchanged():
    inner = InMemoryStepStore()
    store = PreparingStepStore(inner)
    await store.save_snapshot(_snapshot())
    saved = await inner.latest_snapshot(run_id='r1')
    assert saved is not None
    assert saved.agent_name is None


async def test_all_other_methods_delegate():
    inner = InMemoryStepStore()
    store = PreparingStepStore(inner, preparers=())
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    assert (await store.get_run(run_id='r1')) is not None
    assert [r.run_id for r in await store.list_runs(conversation_id='c1')] == ['r1']
    await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
    assert [e.kind for e in await store.list_events(run_id='r1')] == ['run_started']
    await store.record_tool_effect(ToolEffectRecord(tool_call_id='t1', tool_name='t', run_id='r1', status='started'))
    assert await store.get_tool_effect(run_id='r1', tool_call_id='t1') is not None
    assert [r.tool_call_id for r in await store.list_unresolved_tool_effects(run_id='r1')] == ['t1']
    await store.save_snapshot(_snapshot())
    assert await store.latest_snapshot(run_id='r1') is not None


def _stamped(message: ModelRequest | ModelResponse, run_id: str) -> ModelRequest | ModelResponse:
    return replace(message, run_id=run_id)


def test_new_messages_keeps_the_suffix_from_the_first_own_stamp():
    # Run state: the loaded history (stamped by earlier runs, or unstamped client input) ahead of
    # the run's own messages. The boundary save folds in the not-yet-stamped tool-return request,
    # which belongs to the turn: a suffix keeps it, a stamp filter would not.
    replayed = ModelRequest(parts=[UserPromptPart(content='old')])
    foreign = _stamped(ModelResponse(parts=[TextPart(content='old a')]), 'r1')
    own_request = _stamped(ModelRequest(parts=[UserPromptPart(content='new')]), 'r2')
    own_response = _stamped(ModelResponse(parts=[TextPart(content='a')]), 'r2')
    tool_return = ModelRequest(parts=[ToolReturnPart(tool_name='t', content='done', tool_call_id='tc1')])
    snapshot = _snapshot(run_id='r2', messages=[replayed, foreign, own_request, own_response, tool_return])
    assert NewMessagesPreparer().prepare_snapshot(snapshot).messages == [own_request, own_response, tool_return]


def test_new_messages_leaves_a_fully_own_snapshot_untouched():
    own_request = _stamped(ModelRequest(parts=[UserPromptPart(content='new')]), 'r2')
    own_response = _stamped(ModelResponse(parts=[TextPart(content='a')]), 'r2')
    snapshot = _snapshot(run_id='r2', messages=[own_request, own_response])
    assert NewMessagesPreparer().prepare_snapshot(snapshot) is snapshot


def test_new_messages_cuts_an_interrupted_snapshot_with_own_stamps_too():
    # The harness's at-failure save of a mid-tool-call crash carries the whole run state; only
    # the run's own part is the turn a human should see.
    foreign = _stamped(ModelResponse(parts=[TextPart(content='old a')]), 'r1')
    own_request = _stamped(ModelRequest(parts=[UserPromptPart(content='new')]), 'r2')
    own_call = _stamped(ModelResponse(parts=[TextPart(content='partial')]), 'r2')
    snapshot = _snapshot(run_id='r2', state='interrupted', messages=[foreign, own_request, own_call])
    assert NewMessagesPreparer().prepare_snapshot(snapshot).messages == [own_request, own_call]


def test_new_messages_passes_an_unstamped_interrupted_snapshot_through():
    # An interrupted save with no stamp of the run is kept as is; only an unstamped `complete`
    # save is the at-failure replay worth suppressing.
    snapshot = _snapshot(run_id='r2', step_index=0, state='interrupted')
    assert NewMessagesPreparer().prepare_snapshot(snapshot) is snapshot


def test_new_messages_empties_a_complete_snapshot_with_no_own_stamp():
    # The harness's at-failure save of a run that died at its first model request: the run's
    # start-of-run input, nothing of the run's own. Emptied, which the seam turns into a skipped
    # save, so the run keeps its frontier save.
    artifact = _snapshot(
        run_id='r2',
        step_index=0,
        messages=[
            ModelRequest(parts=[UserPromptPart(content='old')]),
            ModelResponse(parts=[TextPart(content='old a')]),
            ModelRequest(parts=[UserPromptPart(content='new')]),
        ],
    )
    assert NewMessagesPreparer().prepare_snapshot(artifact).messages == []


def test_new_messages_keeps_a_complete_snapshot_of_the_prompt_alone():
    # The frontier save: the run's prompt, stamped, saved before the first model request. It is what a
    # run building on this one continues from, so it is stored as the run's turn.
    foreign = _stamped(ModelResponse(parts=[TextPart(content='old a')]), 'r1')
    own_request = _stamped(ModelRequest(parts=[UserPromptPart(content='new')]), 'r2')
    snapshot = _snapshot(run_id='r2', messages=[foreign, own_request])
    assert NewMessagesPreparer().prepare_snapshot(snapshot).messages == [own_request]


def test_new_messages_keeps_results_of_the_runs_own_tool_calls():
    own_call = _stamped(ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='tc1')]), 'r2')
    own_return = ModelRequest(parts=[ToolReturnPart(tool_name='t', content='done', tool_call_id='tc1')])
    snapshot = _snapshot(run_id='r2', messages=[own_call, own_return])
    assert NewMessagesPreparer().prepare_snapshot(snapshot) is snapshot


class _DownStepStore(InMemoryStepStore):
    """Every store method raises `failure`, as a driver does when its server is gone. The store
    declares connection errors temporary, the way the shipped stores declare their driver's."""

    transient_errors: tuple[type[Exception], ...] = (OSError,)

    def __init__(self, failure: Exception | None = None) -> None:
        super().__init__()
        self.failure = failure or ConnectionError('backend down')

    def __getattribute__(self, name: str) -> Any:
        attribute = super().__getattribute__(name)
        if name.startswith('_') or name == 'failure' or not callable(attribute):
            return attribute

        async def down(*args: Any, **kwargs: Any) -> Any:
            raise self.failure

        return down


_OPERATIONS: dict[str, Callable[[PreparingStepStore], Awaitable[object]]] = {
    'save_snapshot': lambda store: store.save_snapshot(_snapshot()),
    'register_run': lambda store: store.register_run(RunRecord(run_id='r1', conversation_id='c1')),
    'get_run': lambda store: store.get_run(run_id='r1'),
    'list_runs': lambda store: store.list_runs(conversation_id='c1'),
    'append_event': lambda store: store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0)),
    'list_events': lambda store: store.list_events(run_id='r1'),
    'latest_snapshot': lambda store: store.latest_snapshot(run_id='r1'),
    'record_tool_effect': lambda store: store.record_tool_effect(
        ToolEffectRecord(tool_call_id='t1', tool_name='t', run_id='r1', status='started')
    ),
    'get_tool_effect': lambda store: store.get_tool_effect(run_id='r1', tool_call_id='t1'),
    'list_unresolved_tool_effects': lambda store: store.list_unresolved_tool_effects(run_id='r1'),
}


@pytest.mark.parametrize('operation', sorted(_OPERATIONS))
async def test_a_failing_store_call_becomes_the_retryable_unavailable_error(operation: str) -> None:
    """The harness calls the store from inside the run, where no capability hook of this package can
    catch a driver exception; mapped here, a store outage reaches the caller as 503 with
    `retryable: true` on every backend, instead of 500 `internal`."""
    store = PreparingStepStore(_DownStepStore(), preparers=())
    with pytest.raises(AgentError) as exc_info:
        await _OPERATIONS[operation](store)
    assert (exc_info.value.code, exc_info.value.retryable) == ('conversations/unavailable', True)
    assert isinstance(exc_info.value.__cause__, ConnectionError)


async def test_a_failure_the_store_does_not_declare_temporary_passes_through() -> None:
    """The harness raises `ValueError` for a run id already stored or a malformed row; retrying
    either cannot succeed, so the caller must not be told to."""
    store = PreparingStepStore(_DownStepStore(ValueError('run_id already in the store')), preparers=())
    with pytest.raises(ValueError, match='already in the store'):
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))


async def test_a_store_that_declares_nothing_has_no_failure_mapped() -> None:
    class _Undeclared(InMemoryStepStore):
        async def get_run(self, *, run_id: str) -> RunRecord | None:
            raise ConnectionError('backend down')

    with pytest.raises(ConnectionError):
        await PreparingStepStore(_Undeclared(), preparers=()).get_run(run_id='r1')


async def test_an_agent_error_from_the_store_keeps_its_own_retry_decision() -> None:
    final = AgentError('quota-exceeded', 'The storage quota is used up.', retryable=False)
    down = _DownStepStore(final)
    down.transient_errors = (Exception,)
    with pytest.raises(AgentError) as exc_info:
        await PreparingStepStore(down, preparers=()).get_run(run_id='r1')
    assert exc_info.value is final


async def test_structural_extras_reachable_through_wrapper():
    class _PingStore(InMemoryStepStore):
        async def ping(self) -> dict[str, Any]:
            return {'ok': True}

    store = PreparingStepStore(_PingStore(), preparers=())
    assert await store.ping() == {'ok': True}
    assert not hasattr(store, 'delete_runs')
