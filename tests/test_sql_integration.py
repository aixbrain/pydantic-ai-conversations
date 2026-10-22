"""Full-stack integration: the factory app over an all-SQLite persistence stack.

One database file carries the whole feature -- the harness store's runs/events/snapshots AND the
registry's conversations/turns (see `pydantic_ai_conversations/sqlite/`). A multi-step agent (tool
roundtrip via `FunctionModel`) drives every turn, and faults are injected at each seam the design
documents a behavior for: the registry (whole-backend outage via `close()`, surgical
`update_message_stats` failure), the step store (the frontier save, history reads, substrate
deletion), the model (before the first response, after a settled tool boundary), and the tool.

Error-shape note: the run path renders `AgentError` flat (`{'code': ...}`) while the management
routes raise `HTTPException` with a structured `detail` -- assertions below mirror that split.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_app_factory import AppDeps

from pydantic_ai_conversations import ConversationsFeature
from tests.sql_stack import (
    ALICE,
    BOB,
    CAROL,
    Backends,
    SqlStack,
    last_user_prompt,
    run_async,
    sql_stack,
    user_prompt_count,
)


@dataclass
class Faults:
    """Switches the model function and tool consult on every call."""

    fail_first_model_request: bool = False
    fail_after_tool_boundary: bool = False
    fail_tool: bool = False


OutageStack = SqlStack[Faults]


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[OutageStack]:
    faults = Faults()
    seen: list[list[ModelMessage]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        last = messages[-1]
        after_tool_boundary = isinstance(last, ModelRequest) and any(
            isinstance(part, ToolReturnPart) for part in last.parts
        )
        if not after_tool_boundary:
            if faults.fail_first_model_request:
                raise RuntimeError('injected: model down before first response')
            return ModelResponse(
                parts=[ToolCallPart(tool_name='lookup_part', args={'query': last_user_prompt(messages)})]
            )
        if faults.fail_after_tool_boundary:
            raise RuntimeError('injected: model down after tool boundary')
        return ModelResponse(parts=[TextPart(content=f'answer:{last_user_prompt(messages)}')])

    def build(backends: Backends) -> tuple[Agent[AppDeps, Any], ConversationsFeature]:
        feature = ConversationsFeature(
            registry=backends.registry,
            step_store=backends.store,
            scope=('user.tenant_id',),  # tenant-wide sharing: alice and bob co-write, carol hits the wall
            on_delete=[backends.deletions],
        )
        agent: Agent[AppDeps, Any] = Agent(FunctionModel(model_fn), deps_type=AppDeps)

        @agent.tool_plain
        def lookup_part(query: str) -> str:
            if faults.fail_tool:
                raise RuntimeError('injected: tool backend down')
            return f'part-info:{query}'

        return agent, feature

    with sql_stack(tmp_path / 'stack.db', build, faults=faults, seen=seen) as stack:
        yield stack


def _phases(stack: OutageStack, conversation_id: str) -> list[str]:
    return list(stack.turn_phases(conversation_id))


def _user_prompt_count(messages: list[dict[str, Any]], text: str) -> int:
    return user_prompt_count(messages, text)


def test_multi_turn_history_flows_into_later_runs(stack: OutageStack) -> None:
    cid = 'flow-chat'
    first = stack.run('pump is leaking', cid)
    assert first.status_code == 200 and first.json()['output'] == 'answer:pump is leaking'
    second = stack.run('replaced the seal', cid, after=first.json()['run_id'])
    assert second.status_code == 200

    # Each turn is a real multi-step run: tool-call request, then the post-tool answer request.
    assert len(stack.seen) == 4
    # Turn 2's FIRST model request already carries turn 1 whole: prompt, tool call, and tool return.
    turn2_first = str(stack.seen[2])
    assert 'pump is leaking' in turn2_first and 'part-info:pump is leaking' in turn2_first
    # Injection never re-enters run state: turn 2's own batch stays 4 messages, so the digest counts 8.
    listed = stack.client.get('/conversations').json()['data']
    assert [c['conversation_id'] for c in listed] == [cid]
    assert listed[0]['message_count'] == 8 and listed[0]['last_message_at'] is not None

    detail = stack.client.get(f'/conversations/{cid}').json()
    assert [turn['phase'] for turn in detail['turns']] == ['answered', 'answered']
    assert [turn['author_user_id'] for turn in detail['turns']] == ['alice', 'alice']
    assert len(detail['messages']) == 8  # both turns' batches, flat, in run-start order


def test_two_authors_share_the_conversation_and_the_wall_holds(stack: OutageStack) -> None:
    cid = 'case-chat'
    first = stack.run('pump is leaking', cid)
    assert first.status_code == 200
    stack.auth.user = BOB
    assert stack.run('replaced the seal', cid, after=first.json()['run_id']).status_code == 200
    # Bob's first model request saw alice's turn.
    assert 'pump is leaking' in str(stack.seen[2])

    turns = run_async(stack.registry.list_turns(stack.namespace, cid))
    assert [turn.author_user_id for turn in turns] == ['alice', 'bob']
    detail = stack.client.get(f'/conversations/{cid}').json()
    assert [turn['author_user_id'] for turn in detail['turns']] == ['alice', 'bob']

    # Carol's tenant derives a different namespace: everything is 404-shaped, nothing is touched.
    stack.auth.user = CAROL
    assert stack.client.get(f'/conversations/{cid}').status_code == 404
    assert stack.client.delete(f'/conversations/{cid}').status_code == 404
    rejected = stack.run('let me in', cid)
    assert rejected.status_code == 404 and rejected.json()['code'] == 'conversations/not-found'
    stack.auth.user = ALICE
    assert stack.client.get(f'/conversations/{cid}').status_code == 200
    assert _phases(stack, cid) == ['answered', 'answered']  # carol's rejected run never recorded a turn


def test_registry_outage_is_retryable_and_clears(stack: OutageStack) -> None:
    cid = 'outage-chat'
    first = stack.run('pump is leaking', cid)
    assert first.status_code == 200
    run_async(stack.registry.close())

    run = stack.run('replaced the seal', cid, after=first.json()['run_id'])
    assert (run.status_code, run.json()['code']) == (503, 'conversations/unavailable')
    listed = stack.client.get('/conversations')
    assert (listed.status_code, listed.json()['detail']['code']) == (503, 'conversations/unavailable')
    assert stack.client.get(f'/conversations/{cid}').status_code == 503
    assert stack.client.delete(f'/conversations/{cid}').status_code == 503
    ready = stack.client.get('/readyz')
    assert ready.status_code == 503 and 'conversations' in ready.json()['failures']

    # `retryable: true` means exactly that: reopen, resend unchanged, everything works.
    run_async(stack.registry.open())
    assert stack.run('replaced the seal', cid, after=first.json()['run_id']).status_code == 200
    assert _phases(stack, cid) == ['answered', 'answered']
    assert stack.client.get('/readyz').status_code == 200


def test_stats_update_failure_never_costs_the_answer(stack: OutageStack) -> None:
    cid = 'stats-chat'
    first = stack.run('pump is leaking', cid)
    assert first.status_code == 200
    stack.registry.fail_update_message_stats = True
    second = stack.run('replaced the seal', cid, after=first.json()['run_id'])
    # The answer is persisted and returned; only the listing digest goes stale.
    assert second.status_code == 200 and second.json()['output'] == 'answer:replaced the seal'
    meta = stack.client.get('/conversations').json()['data'][0]
    assert meta['message_count'] == 4  # still turn 1's count -- stale, not lost
    assert _phases(stack, cid) == ['answered', 'answered']  # the phase write is a separate, unfaulted seam
    assert len(stack.client.get(f'/conversations/{cid}').json()['messages']) == 8  # transcript is complete


def test_model_failure_before_first_response_keeps_the_prompt(stack: OutageStack) -> None:
    cid = 'early-fail-chat'
    stack.faults.fail_first_model_request = True
    failed = stack.run('pump is leaking', cid)
    assert (failed.status_code, failed.json()['code']) == (500, 'internal')
    assert _phases(stack, cid) == ['failed']

    detail = stack.client.get(f'/conversations/{cid}').json()
    (turn,) = detail['turns']
    # No model response ever existed, yet the user's message is not lost: the frontier save holds it.
    assert turn['phase'] == 'failed'
    assert 'pump is leaking' in str(turn['messages'])
    assert _user_prompt_count(detail['messages'], 'pump is leaking') == 1  # the branch ends in the open question

    stack.faults.fail_first_model_request = False
    # A retry of a first turn starts from nothing again: an explicit `null`, not an omitted field.
    retry = stack.run('pump is leaking', cid, after=None)
    assert retry.status_code == 200
    # The failed attempt never polluted model context: the retry's first request is prompt-only.
    assert sum(isinstance(m, ModelResponse) for m in stack.seen[1]) == 0
    detail = stack.client.get(f'/conversations/{cid}').json()
    failed, answered = detail['turns']  # both attempts stay listed, as two roots
    assert (failed['phase'], answered['phase']) == ('failed', 'answered')
    assert [turn['previous_run_id'] for turn in detail['turns']] == [None, None]
    assert _user_prompt_count(detail['messages'], 'pump is leaking') == 1  # the flat view: the latest root alone
    assert _phases(stack, cid) == ['failed', 'answered']  # the registry still remembers both runs


def test_model_failure_after_tool_boundary_keeps_the_settled_work(stack: OutageStack) -> None:
    cid = 'boundary-fail-chat'
    stack.faults.fail_after_tool_boundary = True
    assert stack.run('pump is leaking', cid).status_code == 500
    detail = stack.client.get(f'/conversations/{cid}').json()
    (turn,) = detail['turns']
    # The tool round settled before the crash: the turn shows the settled work, and the branch a
    # reload shows ends in the dangling tool return, which is where the run stopped.
    assert turn['phase'] == 'failed'
    assert [message['kind'] for message in detail['messages']] == ['request', 'response', 'request']

    stack.faults.fail_after_tool_boundary = False
    # Naming the failed attempt builds on its settled work: the retry answers from the stored tool
    # result instead of calling the tool again; its stored turn is the re-sent prompt and the answer.
    assert stack.run('pump is leaking', cid, after=stack.last_turn_id(cid)).status_code == 200
    detail = stack.client.get(f'/conversations/{cid}').json()
    failed, answered = detail['turns']
    assert (failed['phase'], answered['phase']) == ('failed', 'answered')
    assert answered['previous_run_id'] == failed['run_id']
    assert [message['kind'] for message in failed['messages']] == ['request', 'response', 'request']
    assert [message['kind'] for message in answered['messages']] == ['request', 'response']
    # `/run` always sends a prompt, so this client re-sent the text: once in the attempt, once in
    # the retry, and the branch shows both. A Vercel client resumes with an empty list instead.
    assert _user_prompt_count(detail['messages'], 'pump is leaking') == 2
    # Two requests for the failed attempt, one for the retry: no second tool round. The retry's
    # prompt reached the model joined onto the attempt's dangling tool return.
    assert len(stack.seen) == 3
    assert [type(part).__name__ for part in stack.seen[2][-1].parts] == ['ToolReturnPart', 'UserPromptPart']


def test_tool_failure_fails_the_run_and_resolves_the_turn(stack: OutageStack) -> None:
    cid = 'tool-fail-chat'
    stack.faults.fail_tool = True
    assert stack.run('pump is leaking', cid).status_code == 500
    assert _phases(stack, cid) == ['failed']
    stack.faults.fail_tool = False
    assert stack.run('pump is leaking', cid, after=None).status_code == 200
    assert _phases(stack, cid) == ['failed', 'answered']


def test_store_failure_at_the_first_save_is_retryable_and_stores_nothing(stack: OutageStack) -> None:
    cid = 'first-save-fail-chat'
    stack.store.fail_next_save = True  # the run's first save is the harness's frontier save of the prompt
    response = stack.run('pump is leaking', cid)
    assert (response.status_code, response.json()['code']) == (503, 'conversations/unavailable')
    assert response.json()['retryable'] is True
    assert stack.client.get(f'/conversations/{cid}').json()['turns'] == []
    assert _phases(stack, cid) == ['failed']

    # The same request, resent, becomes the conversation's first turn.
    assert stack.run('pump is leaking', cid, after=None).status_code == 200
    assert _phases(stack, cid) == ['failed', 'answered']


def test_history_load_failure_is_retryable_and_clears(stack: OutageStack) -> None:
    cid = 'load-fail-chat'
    first = stack.run('pump is leaking', cid)
    assert first.status_code == 200
    stack.store.fail_reads = True
    failed = stack.run('replaced the seal', cid, after=first.json()['run_id'])
    assert (failed.status_code, failed.json()['code']) == (503, 'conversations/unavailable')
    assert stack.client.get(f'/conversations/{cid}').status_code == 503  # transcript read hits the same seam

    stack.store.fail_reads = False
    assert stack.run('replaced the seal', cid, after=first.json()['run_id']).status_code == 200
    detail = stack.client.get(f'/conversations/{cid}').json()
    # The rejected request was turned away before anything of it was recorded: no turn, no snapshot.
    assert [turn['phase'] for turn in detail['turns']] == ['answered', 'answered']
    assert [turn['previous_run_id'] for turn in detail['turns']] == [None, first.json()['run_id']]
    assert len(detail['messages']) == 8
    assert _phases(stack, cid) == ['answered', 'answered']


def test_soft_delete_hides_everywhere_without_revival(stack: OutageStack) -> None:
    cid = 'delete-chat'
    assert stack.run('pump is leaking', cid).status_code == 200
    assert stack.client.delete(f'/conversations/{cid}').status_code == 204
    assert [(event.kind, event.conversation_id) for event in stack.deletions.events] == [('soft_delete', cid)]
    assert stack.deletions.events[0].meta is not None

    assert stack.client.get(f'/conversations/{cid}').status_code == 404
    assert stack.client.get('/conversations').json()['data'] == []
    assert stack.client.delete(f'/conversations/{cid}').status_code == 404  # already gone
    revived = stack.run('knock knock', cid)
    assert (revived.status_code, revived.json()['code']) == (404, 'conversations/not-found')


def test_purge_survives_a_substrate_failure_and_retries_clean(stack: OutageStack) -> None:
    cid = 'purge-chat'
    assert stack.run('pump is leaking', cid).status_code == 200
    (turn,) = run_async(stack.registry.list_turns(stack.namespace, cid))
    namespace = stack.namespace

    stack.store.fail_delete_runs_once = True
    with pytest.raises(RuntimeError, match='substrate delete down'):
        run_async(stack.feature.purge(namespace_key=namespace, conversation_id=cid))
    # The registry purge ran FIRST and held: unreachable through the feature, substrate rows intact.
    assert stack.client.get(f'/conversations/{cid}').status_code == 404
    assert len(run_async(stack.store.list_runs(conversation_id=cid))) == 1

    run_async(stack.feature.purge(namespace_key=namespace, conversation_id=cid))
    assert run_async(stack.store.list_runs(conversation_id=cid)) == []
    assert run_async(stack.store.latest_snapshot(run_id=turn.run_id, include_interrupted=True)) is None
    # The failed attempt aborted before its hook loop; only the completed retry ran the hooks.
    assert [event.kind for event in stack.deletions.events] == ['purge']
    # The id stays claimed forever, restart-independent (see test_sqlite_registry for the reopen case).
    reused = stack.run('brand new?', cid)
    assert (reused.status_code, reused.json()['code']) == (404, 'conversations/not-found')


def test_readyz_reports_both_sql_backends(stack: OutageStack) -> None:
    ready = stack.client.get('/readyz')
    assert ready.status_code == 200
    # Status only: the probe is unauthenticated, so the report carries no counts.
    assert ready.json()['reports']['conversations'] == {'ok': True, 'step_store': {'ok': True}}
    assert stack.run('pump is leaking', 'health-chat').status_code == 200
    assert stack.client.get('/readyz').json()['reports']['conversations'] == {'ok': True, 'step_store': {'ok': True}}
