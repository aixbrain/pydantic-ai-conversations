"""Tool error recovery over the SQL stack -- each mechanism, and the mechanisms combined.

pydantic-ai gives a tool four distinct ways to end badly, and they end the RUN differently:

| Tool raises | Model sees | Run outcome | Registry phase |
|---|---|---|---|
| `ModelRetry` (budget left) | `ToolReturnPart(outcome='retried')`, calls again | answer | `answered` |
| `ModelRetry` (budget spent) | -- | `UnexpectedModelBehavior` | `failed` |
| `ToolFailed` | the failure as a tool return, adapts | answer | `answered` |
| a plain exception | -- | the exception | `failed` |

Schema-invalid arguments travel the same channel as `ModelRetry`: a retried `ToolReturnPart` for
the call, which the model answers with corrected arguments.

What this suite pins is not that pydantic-ai implements those (upstream's job) but what the
conversation feature does WITH them: the recovery trail is part of the run's own batch, so it is
stored, it reaches later turns' model context, and it shows up in transcripts -- unless a view
seam says otherwise. The final test wires all three seams at once over one recovery trail and
verifies the three audiences see three different things from one stored truth.
"""

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import ToolFailed
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
from pydantic_ai_harness.step_persistence import ContinuableSnapshot

from pydantic_ai_conversations import ConversationsFeature, StripRejectedDrafts, TranscriptTurn
from tests.sql_stack import Backends, SqlStack, is_retry_feedback, last_user_prompt, part_kinds, sql_stack

SECRET = 'serial-4711'
"""Planted in every tool return so the write seam has something real to redact."""


@dataclass
class Counters:
    """What actually happened, for assertions the message shapes alone cannot make."""

    tool_calls: int = 0
    drafts: int = 0
    bad_arguments_sent: bool = False


@dataclass
class RecoveryFaults:
    """How the tool and the model misbehave this run, and what that produced."""

    tool_retries_before_success: int = 0
    """Raise `ModelRetry` this many times, then return normally."""
    tool_always_retries: bool = False
    """Raise `ModelRetry` on every call -- exhausts the budget and fails the run."""
    tool_failed: bool = False
    """Raise `ToolFailed`: terminal for the tool, but the model gets to adapt."""
    reject_first_draft: bool = False
    """Have the output validator reject the run's first answer once."""
    send_bad_arguments_once: bool = False
    """Emit a schema-invalid tool call once, so validation produces the retried return."""
    fail_after_tool_boundary: bool = False
    """Crash the model once a tool round has settled -- `complete` snapshot, failed run."""
    counters: Counters = field(default_factory=Counters)


RecoveryStack = SqlStack[RecoveryFaults]

_RETRY_BUDGET = 2
"""Small enough that `tool_always_retries` exhausts it in one test, large enough that a single
`ModelRetry` (and a single rejected draft) still recovers."""


def _has(parts: Any, kind: type, **attrs: Any) -> bool:
    return any(isinstance(part, kind) and all(getattr(part, k) == v for k, v in attrs.items()) for part in parts)


def _tool_retry_pending(messages: list[ModelMessage]) -> bool:
    """Whether the newest request asks the model to redo a TOOL call (schema or `ModelRetry`)."""
    last = messages[-1]
    return isinstance(last, ModelRequest) and _has(last.parts, ToolReturnPart, outcome='retried')


def _draft_rejected(messages: list[ModelMessage]) -> bool:
    """Whether the newest request is an output-validator rejection (feedback bound to no call)."""
    last = messages[-1]
    return isinstance(last, ModelRequest) and any(is_retry_feedback(part) for part in last.parts)


def _tool_settled(messages: list[ModelMessage]) -> bool:
    last = messages[-1]
    return isinstance(last, ModelRequest) and any(
        isinstance(part, ToolReturnPart) and part.outcome != 'retried' for part in last.parts
    )


class RedactSecrets:
    """Write seam: the secret must never reach the store, on any path.

    Rebuilds messages with `replace(..., parts=...)` so pydantic-ai's `run_id` stamps survive;
    the write seam cuts each snapshot to its run by them.
    """

    def prepare_snapshot(self, snapshot: ContinuableSnapshot) -> ContinuableSnapshot:
        messages = [
            replace(
                message,
                parts=[
                    replace(part, content=part.content.replace(SECRET, '[redacted]'))
                    if isinstance(part, ToolReturnPart) and isinstance(part.content, str)
                    else part
                    for part in message.parts
                ],
            )
            if isinstance(message, ModelRequest)
            else message
            for message in snapshot.messages
        ]
        return dataclasses.replace(snapshot, messages=messages)


def hide_tool_fumbling(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
    """Transcript seam: humans get the answer, not the tool's stumbles."""
    return [_without_tool_fumbling(turn) for turn in turns]


def _without_tool_fumbling(turn: TranscriptTurn) -> TranscriptTurn:
    """Drops every tool-bound retried return and the call it answers. Orphaning a tool call is fatal
    for a provider but harmless here -- nothing in a transcript is ever sent to a model.
    """
    retried_call_ids = {
        part.tool_call_id
        for message in turn.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.outcome == 'retried'
    }
    messages: list[ModelMessage] = []
    for message in turn.messages:
        kept = [
            part
            for part in message.parts
            if not (isinstance(part, ToolReturnPart) and part.outcome == 'retried')
            and not (isinstance(part, ToolCallPart) and part.tool_call_id in retried_call_ids)
        ]
        if kept:
            messages.append(replace(message, parts=kept))
    return replace(turn, messages=messages)


@dataclass
class _Wiring:
    """Which optional seams this fixture wires -- parametrized per test via `stack_for`."""

    snapshot_preparers: tuple[Any, ...] = ()
    agent_capabilities: tuple[Any, ...] = ()
    transcript_view: Any = None
    retry_budget: int = _RETRY_BUDGET


def _build_stack(tmp_path: Path, wiring: _Wiring) -> Iterator[RecoveryStack]:
    faults = RecoveryFaults()
    counters = faults.counters
    seen: list[list[ModelMessage]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        prompt = last_user_prompt(messages)
        if _draft_rejected(messages):
            return ModelResponse(parts=[TextPart(content=f'GOOD:{prompt}')])
        if _tool_retry_pending(messages):
            # The model corrects itself: same tool, now with arguments that validate.
            return ModelResponse(parts=[ToolCallPart(tool_name='lookup_part', args={'query': prompt})])
        if _tool_settled(messages):
            if faults.fail_after_tool_boundary:
                raise RuntimeError('injected: model down after tool boundary')
            counters.drafts += 1
            rejected = faults.reject_first_draft and counters.drafts == 1
            return ModelResponse(parts=[TextPart(content=f'{"BAD" if rejected else "GOOD"}:{prompt}')])
        if faults.send_bad_arguments_once and not counters.bad_arguments_sent:
            counters.bad_arguments_sent = True
            return ModelResponse(parts=[ToolCallPart(tool_name='lookup_part', args={'quarry': prompt})])
        return ModelResponse(parts=[ToolCallPart(tool_name='lookup_part', args={'query': prompt})])

    def build(backends: Backends) -> tuple[Agent[AppDeps, Any], ConversationsFeature]:
        feature = ConversationsFeature(
            registry=backends.registry,
            step_store=backends.store,
            scope=('user.tenant_id',),
            on_delete=[backends.deletions],
            snapshot_preparers=list(wiring.snapshot_preparers),
            transcript_view=wiring.transcript_view,
        )
        agent: Agent[AppDeps, Any] = Agent(
            FunctionModel(model_fn),
            deps_type=AppDeps,
            retries={'tools': wiring.retry_budget, 'output': wiring.retry_budget},
            capabilities=list(wiring.agent_capabilities),
        )

        @agent.tool_plain
        def lookup_part(query: str) -> str:
            counters.tool_calls += 1
            if faults.tool_always_retries:
                raise ModelRetry('the parts index is not answering; try again')
            if counters.tool_calls <= faults.tool_retries_before_success:
                raise ModelRetry('that part number is malformed; use the full serial')
            if faults.tool_failed:
                raise ToolFailed('the parts index is permanently gone')
            return f'part-info:{query} ({SECRET})'

        @agent.output_validator
        def reject_bad_drafts(text: str) -> str:
            if text.startswith('BAD'):
                raise ModelRetry('that answer violates policy; rewrite it')
            return text

        return agent, feature

    with sql_stack(tmp_path / 'recovery.db', build, faults=faults, seen=seen) as stack:
        yield stack


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[RecoveryStack]:
    yield from _build_stack(tmp_path, _Wiring())


@pytest.fixture
def viewed_stack(tmp_path: Path) -> Iterator[RecoveryStack]:
    """All three view seams wired at once -- the combination test's fixture."""
    yield from _build_stack(
        tmp_path,
        _Wiring(
            snapshot_preparers=(RedactSecrets(),),
            agent_capabilities=(StripRejectedDrafts(),),
            transcript_view=hide_tool_fumbling,
        ),
    )


def _counters(stack: RecoveryStack) -> Counters:
    return stack.faults.counters


def test_model_retry_recovers_inside_the_turn_and_is_stored(stack: RecoveryStack) -> None:
    cid = 'retry-chat'
    stack.faults.tool_retries_before_success = 1
    response = stack.run('pump is leaking', cid)

    assert response.status_code == 200 and response.json()['output'] == 'GOOD:pump is leaking'
    assert _counters(stack).tool_calls == 2  # the failed attempt and the successful redo
    assert stack.turn_phases(cid) == ['answered']  # recovery inside a turn is not a failed turn
    # The recovery trail is the run's OWN contribution, so it is part of the stored batch.
    assert part_kinds(stack.stored_messages(cid)) == [
        'UserPromptPart',
        'ToolCallPart',
        'ToolReturnPart',  # the tool's ModelRetry, bound to the call above
        'ToolCallPart',
        'ToolReturnPart',
        'TextPart',
    ]


def test_the_retry_trail_reaches_the_next_turn_by_default(stack: RecoveryStack) -> None:
    cid = 'retry-history-chat'
    stack.faults.tool_retries_before_success = 1
    first = stack.run('pump is leaking', cid)
    assert first.status_code == 200
    stack.faults.tool_retries_before_success = 0
    assert stack.run('replaced the seal', cid, after=first.json()['run_id']).status_code == 200

    # Turn 2's first model request carries turn 1 whole, retried return included: with no view seam
    # wired, nothing hides it. `StripRejectedDrafts` would NOT remove it either -- a tool-bound
    # retry answers a tool call, and dropping it would orphan that call.
    turn2_first = stack.seen[3]
    assert sum(_has(message.parts, ToolReturnPart, outcome='retried') for message in turn2_first) == 1
    # It sits in history, not the new request.
    assert _has(turn2_first[-1].parts, ToolReturnPart, outcome='retried') is False


def test_retry_exhaustion_fails_the_turn_but_keeps_the_settled_work(stack: RecoveryStack) -> None:
    cid = 'exhausted-chat'
    stack.faults.tool_always_retries = True
    response = stack.run('pump is leaking', cid)

    assert response.status_code == 500 and response.json()['code'] == 'internal'
    assert _counters(stack).tool_calls == _RETRY_BUDGET + 1  # the budget, then the attempt that broke it
    assert stack.turn_phases(cid) == ['failed']
    detail = stack.detail(cid)
    (turn,) = detail['turns']
    # A settled boundary snapshot exists even though the run never answered: the turn shows the
    # work, and the branch a reload shows ends where the run stopped.
    assert turn['phase'] == 'failed'
    assert len(detail['messages']) == len(turn['messages'])


def test_a_turn_after_retry_exhaustion_joins_provider_valid(stack: RecoveryStack) -> None:
    cid = 'exhausted-then-continue-chat'
    stack.faults.tool_always_retries = True
    assert stack.run('pump is leaking', cid).status_code == 500
    stack.faults.tool_always_retries = False

    # The stored history now ENDS in a dangling request (a retried return nobody answered). A turn
    # that builds on the failed one has to repair that seam: tool-bound parts first, then the new
    # user prompt.
    assert stack.run('replaced the seal', cid, after=stack.last_turn_id(cid)).status_code == 200
    joined = stack.seen[_RETRY_BUDGET + 1]
    leading = joined[-1]
    assert isinstance(leading, ModelRequest)
    assert [type(part).__name__ for part in leading.parts] == ['ToolReturnPart', 'UserPromptPart']
    assert stack.turn_phases(cid) == ['failed', 'answered']


def test_tool_failed_is_terminal_for_the_tool_but_not_for_the_turn(stack: RecoveryStack) -> None:
    cid = 'tool-failed-chat'
    stack.faults.tool_failed = True
    response = stack.run('pump is leaking', cid)

    # `ToolFailed` never retries: the model is handed the failure and adapts, so the run answers.
    assert response.status_code == 200 and response.json()['output'] == 'GOOD:pump is leaking'
    assert _counters(stack).tool_calls == 1
    assert stack.turn_phases(cid) == ['answered']
    stored = stack.stored_messages(cid)
    assert part_kinds(stored) == ['UserPromptPart', 'ToolCallPart', 'ToolReturnPart', 'TextPart']
    returns = [part for message in stored for part in message.parts if isinstance(part, ToolReturnPart)]
    assert returns[0].content == 'the parts index is permanently gone'


def test_invalid_arguments_are_corrected_through_the_same_retry_channel(stack: RecoveryStack) -> None:
    cid = 'bad-args-chat'
    stack.faults.send_bad_arguments_once = True
    response = stack.run('pump is leaking', cid)

    assert response.status_code == 200
    assert _counters(stack).tool_calls == 1  # the invalid call never reached the tool body
    assert stack.turn_phases(cid) == ['answered']
    assert part_kinds(stack.stored_messages(cid)) == [
        'UserPromptPart',
        'ToolCallPart',  # the schema-invalid call
        'ToolReturnPart',  # validation's own retried return, same channel as ModelRetry
        'ToolCallPart',
        'ToolReturnPart',
        'TextPart',
    ]


def test_tool_retry_and_a_rejected_draft_combine_in_one_turn(stack: RecoveryStack) -> None:
    cid = 'combined-recovery-chat'
    stack.faults.tool_retries_before_success = 1
    stack.faults.reject_first_draft = True
    response = stack.run('pump is leaking', cid)

    # Two independent recovery mechanisms fire in the same run and both resolve.
    assert response.status_code == 200 and response.json()['output'] == 'GOOD:pump is leaking'
    assert stack.turn_phases(cid) == ['answered']
    assert part_kinds(stack.stored_messages(cid)) == [
        'UserPromptPart',
        'ToolCallPart',
        'ToolReturnPart',  # tool-bound: the tool asked for a redo
        'ToolCallPart',
        'ToolReturnPart',
        'TextPart',  # the rejected draft
        'RetryFeedbackPart',  # output-bound: the validator rejected it
        'TextPart',  # the accepted answer
    ]


def test_recovery_survives_a_registry_digest_outage(stack: RecoveryStack) -> None:
    cid = 'recovery-under-outage-chat'
    stack.faults.tool_retries_before_success = 1
    stack.faults.reject_first_draft = True
    stack.registry.fail_update_message_stats = True

    # Recovery inside the run and a failing registry write are independent: the answer survives
    # both, and only the listing digest goes stale.
    response = stack.run('pump is leaking', cid)
    assert response.status_code == 200 and response.json()['output'] == 'GOOD:pump is leaking'
    assert stack.turn_phases(cid) == ['answered']
    assert stack.client.get('/conversations').json()['data'][0]['message_count'] == 0


def test_a_retry_that_names_the_failed_attempt_inherits_its_recovery_work(stack: RecoveryStack) -> None:
    cid = 'recovered-then-failed-chat'
    stack.faults.tool_retries_before_success = 1
    stack.faults.fail_after_tool_boundary = True
    assert stack.run('pump is leaking', cid).status_code == 500
    assert _counters(stack).tool_calls == 2

    stack.faults.fail_after_tool_boundary = False
    stack.faults.tool_retries_before_success = 0
    assert stack.run('pump is leaking', cid, after=stack.last_turn_id(cid)).status_code == 200

    # The settled tool result reached the retry through the loaded history, so it had no reason
    # to call the tool again: recovery work survives a failure without being redone.
    assert _counters(stack).tool_calls == 2
    failed, answered = stack.detail(cid)['turns']
    assert (failed['phase'], answered['phase']) == ('failed', 'answered')
    assert answered['previous_run_id'] == failed['run_id']
    assert [part['part_kind'] for message in failed['messages'] for part in message['parts']] == [
        'user-prompt',
        'tool-call',
        'tool-return',  # the first attempt's tool retry
        'tool-call',
        'tool-return',
    ]
    assert [part['part_kind'] for message in answered['messages'] for part in message['parts']] == [
        'user-prompt',
        'text',
    ]
    # The retry's prompt reached the model joined onto the attempt's dangling tool return.
    assert [type(part).__name__ for part in stack.seen[-1][-1].parts] == ['ToolReturnPart', 'UserPromptPart']
    assert stack.turn_phases(cid) == ['failed', 'answered']


def test_three_view_seams_diverge_over_one_recovery_trail(viewed_stack: RecoveryStack) -> None:
    """The combination: one stored truth, three audiences, three different shapes.

    `RedactSecrets` (write seam) keeps the secret out of the store entirely, so it is absent for
    everyone. `StripRejectedDrafts` (on the agent) hides the rejected draft from the MODEL only.
    `hide_tool_fumbling` (transcript_view) hides the tool retry from HUMANS only. Each removal
    lands in exactly one place -- that is what makes the seams composable rather than overlapping.
    """
    stack = viewed_stack
    cid = 'three-views-chat'
    stack.faults.tool_retries_before_success = 1
    stack.faults.reject_first_draft = True
    first = stack.run('pump is leaking', cid)
    assert first.status_code == 200

    stored = stack.stored_messages(cid)
    assert part_kinds(stored) == [
        'UserPromptPart',
        'ToolCallPart',
        'ToolReturnPart',
        'ToolCallPart',
        'ToolReturnPart',
        'TextPart',
        'RetryFeedbackPart',
        'TextPart',
    ]
    returns = [part for message in stored for part in message.parts if isinstance(part, ToolReturnPart)]
    assert SECRET not in str(returns) and '[redacted]' in str(returns)  # write seam: never stored

    # THE MODEL's view of turn 1, as loaded for turn 2: the rejected draft and its marker are gone,
    # the tool-bound retry stays (dropping it would orphan the call it answers).
    stack.faults.tool_retries_before_success = 0
    stack.faults.reject_first_draft = False
    assert stack.run('replaced the seal', cid, after=first.json()['run_id']).status_code == 200
    # Turn 1 issued four model requests (call, redo after the tool retry, draft, rewrite after the
    # rejection), so turn 2's first request -- the one carrying the loaded history -- is index 4.
    turn2_first_request = stack.seen[4]
    assert part_kinds(turn2_first_request) == [
        'UserPromptPart',
        'ToolCallPart',
        'ToolReturnPart',  # tool-bound: kept, or the call above would be orphaned
        'ToolCallPart',
        'ToolReturnPart',
        'TextPart',  # only the ACCEPTED answer -- the rejected draft and its marker are stripped
        'UserPromptPart',  # turn 2's own prompt
    ]
    assert 'BAD:pump is leaking' not in str(turn2_first_request)
    assert SECRET not in str(turn2_first_request)

    # THE HUMAN's view: the tool fumbling is hidden, the rejected draft is still there (this app
    # streamed it live, so hiding it on reload would contradict what the user already saw).
    (turn, _later) = stack.detail(cid)['turns']
    human_kinds = [part['part_kind'] for message in turn['messages'] for part in message['parts']]
    assert human_kinds == ['user-prompt', 'tool-call', 'tool-return', 'text', 'retry-feedback', 'text']
    assert 'BAD:pump is leaking' in str(turn['messages'])
    assert SECRET not in str(turn['messages'])
