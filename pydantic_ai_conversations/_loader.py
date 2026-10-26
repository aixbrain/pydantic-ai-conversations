"""Rebuild a conversation's history from the harness substrate.

Precondition, owned by the write seam (`stores.NewMessagesPreparer`): each run's latest complete
snapshot contains exactly that run's own messages -- the snapshots ARE the per-turn batches, and
concatenating them along the turn chain is provider-valid. Reading via `latest_snapshot` (not a
snapshot union) keeps this path immune to `max_snapshots_per_run` pruning. One reader serves the
model and the transcript: the latest complete snapshot of each run, or the newest one of any
state for a run the registry resolved to `answered`.

Messages are returned by reference, or as shallow copies sharing the stored parts where a run id
or a part list had to change; treat them as read-only. Cost is one `latest_snapshot`
call per run read. Agents used with this feature must declare `instructions=` (re-rendered per
request, never re-sent from history) rather than legacy `system_prompt=`, which would repeat
per turn in the concatenation.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai_harness.step_persistence import RunRecord, StepStore

from pydantic_ai_conversations.registry import RunPhase, TurnError, TurnRecord


async def load_conversation(
    step_store: StepStore,
    *,
    conversation_id: str,
    run_ids: Sequence[str],
    max_history_runs: int | None = None,
    answered_runs: Collection[str] = (),
) -> list[ModelMessage]:
    """The latest complete snapshot of each run in `run_ids`, concatenated in that order: the
    turn chain from the first turn to the one the new turn builds on.

    A run in `answered_runs` contributes its newest snapshot whatever its state, because the harness
    saves the snapshot holding the open calls of a run that ended in `DeferredToolRequests` as
    `interrupted`, the same as one cut off mid-call. A message without a run id gets the id of the
    run whose snapshot holds it: pydantic-ai stamps a request when it sends it, so the harness's
    boundary save holds the trailing tool-return request unstamped.

    `max_history_runs` keeps the last N runs whose snapshot starts with a request; a continuation
    whose snapshot starts with its answer shares the slot of the run it continues, and a run
    without a snapshot takes none, so a stretch of failed turns cannot blank the history. The
    result starts with a `ModelRequest` or is empty.
    """
    if not conversation_id:
        raise ValueError('conversation_id must be non-empty')
    if max_history_runs is not None and max_history_runs <= 0:
        raise ValueError(f'max_history_runs must be positive, got {max_history_runs!r}')
    # Walked backwards so a `max_history_runs` cut skips snapshot-less candidates without
    # consuming a slot; collected in reverse, flipped back to chain order at the end.
    collected: list[list[ModelMessage]] = []
    slots = 0
    for run_id in reversed(run_ids):
        if max_history_runs is not None and slots >= max_history_runs:
            break
        snapshot = await step_store.latest_snapshot(run_id=run_id, include_interrupted=run_id in answered_runs)
        if snapshot is None:
            continue
        collected.append([m if m.run_id else replace(m, run_id=run_id) for m in snapshot.messages])
        if snapshot.messages and isinstance(snapshot.messages[0], ModelRequest):
            slots += 1
    messages: list[ModelMessage] = []
    for snapshot_messages in reversed(collected):
        messages.extend(snapshot_messages)
    return _drop_leading_assistant_turns(_drop_orphaned_leading_tool_results(messages))


def _is_orphaned_tool_result(part: ModelRequestPart, call_ids: set[str]) -> bool:
    """Whether `part` answers some `ToolCallPart` missing from `call_ids` -- i.e. cut away.

    `ToolReturnPart.tool_call_id` always names a real preceding call, for a retried return as much
    as a settled one. A `RetryFeedbackPart` answers no call and carries no `tool_call_id`.
    """
    return isinstance(part, ToolReturnPart) and part.tool_call_id not in call_ids


def _drop_orphaned_leading_tool_results(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Strip a leading tool result whose `ToolCallPart` is nowhere in `messages`.

    The `max_history_runs` cut can leave out the run that made the call. pydantic-ai drops such a
    result at run start itself; stripping it here can empty the first request and expose the
    response behind it, which `_drop_leading_assistant_turns` then removes. Only the first message
    can carry one, since every other stored batch is provider-valid on its own; `messages` comes
    back unchanged when there is nothing to strip.
    """
    if not messages:
        return messages
    first = messages[0]
    if not isinstance(first, ModelRequest):
        return messages
    call_ids = {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_call_id
    }
    kept_parts = [part for part in first.parts if not _is_orphaned_tool_result(part, call_ids)]
    if len(kept_parts) == len(first.parts):
        return messages
    if kept_parts:
        return [replace(first, parts=kept_parts), *messages[1:]]
    return messages[1:]


def _drop_leading_assistant_turns(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop a leading `ModelResponse` left behind once the request before it was cut away or
    emptied out by `_drop_orphaned_leading_tool_results`.

    Providers reject a message list that starts with the assistant (Anthropic's API among them).
    One leading response is the only shape that reaches here; the loop covers more.
    """
    while messages and isinstance(messages[0], ModelResponse):
        messages = messages[1:]
    return messages


@dataclass(frozen=True)
class TranscriptTurn:
    """One run's contribution to the transcript: its latest complete snapshot, the same
    messages the model path loads for it.

    `phase` is the lifecycle of the run in `run_id`, authoritative from the registry when
    `load_transcript` is given the conversation's `turns`, else derived from the snapshot shape
    (which never yields `'started'`). Render on `phase`: a failed turn's messages are its prompt,
    or its prompt and the tool rounds it settled before failing.

    `previous_run_id` is the turn this one builds on, from the registry: `None` for a first turn,
    and for every turn when `load_transcript` was given no `turns`. Following the links back from
    any turn gives one branch; a retry or a fork is a second turn with the same `previous_run_id`.

    `messages` is the run's best snapshot, returned by reference, treat as read-only.
    `author_user_id` is `None` unless the `turns` covered this run.
    """

    run_id: str
    started_at: datetime
    phase: RunPhase
    messages: list[ModelMessage]
    author_user_id: str | None = None
    previous_run_id: str | None = None
    error: TurnError | None = None
    """Why a `'failed'` run failed, from the registry; `None` without a `describe_failure` mapper
    or without `turns`."""


TranscriptView = Callable[[list[TranscriptTurn]], list[TranscriptTurn]]
"""Shapes the transcript for human-facing reads: receives every resolved turn of the
conversation in run-start order, every branch included, and returns the turns to show. A turn
left out of the result is omitted; `dataclasses.replace` rewrites one. All turns are given
so a view can decide across turns, say hide a failed attempt once a sibling has answered."""


def _derive_phase(messages: list[ModelMessage]) -> RunPhase:
    """Infer the run phase from the snapshot when no registry record is available: a snapshot
    ending in a `ModelResponse` is `'answered'`, everything else `'failed'`.

    A store alone cannot tell an in-flight run from a hard-killed one, so this never yields
    `'started'`. pydantic-ai ends a run with tool output in the final-result tool's return, so
    such a run derives as `'failed'`; the registry record, when present, wins over both."""
    if messages and isinstance(messages[-1], ModelResponse):
        return 'answered'
    return 'failed'


def _make_turn(run: RunRecord, *, messages: list[ModelMessage], record: TurnRecord | None) -> TranscriptTurn:
    """Build a per-run turn; the registry `record`, when there is one, supplies phase, author,
    and lineage, else the phase is derived from the snapshot."""
    messages = list(messages)
    return TranscriptTurn(
        run_id=run.run_id,
        started_at=run.started_at,
        phase=record.phase if record is not None else _derive_phase(messages),
        messages=messages,
        author_user_id=record.author_user_id if record is not None else None,
        previous_run_id=record.previous_run_id if record is not None else None,
        error=record.error if record is not None else None,
    )


async def _resolve_run_turn(
    step_store: StepStore, run: RunRecord, *, record: TurnRecord | None
) -> TranscriptTurn | None:
    """One run's `TranscriptTurn`: its latest complete snapshot, else nothing (a run that
    crashed before its prompt was saved, or a resume that had no prompt and never answered). A
    run the registry resolved to `answered` shows its newest snapshot whatever its state: the
    harness saves the snapshot holding tool calls awaiting approval as `interrupted`.

    A complete snapshot is trusted as is: the write seam stores only a run's own turn, so a
    replayed-input artifact never reaches the store (docs/http-api.md).
    """
    answered = record is not None and record.phase == 'answered'
    snapshot = await step_store.latest_snapshot(run_id=run.run_id, include_interrupted=answered)
    if snapshot is None:
        return None
    return _make_turn(run, messages=snapshot.messages, record=record)


async def load_transcript(
    step_store: StepStore,
    *,
    conversation_id: str,
    turns: Sequence[TurnRecord] | None = None,
    transcript_view: TranscriptView | None = None,
) -> list[TranscriptTurn]:
    """The conversation for HUMAN eyes: every run's best snapshot as a `TranscriptTurn`, failed
    and in-flight turns included, every branch included, in run-start order (what a client
    renders from it is in docs/http-api.md).

    `turns` are the registry's records for the conversation: they stamp each turn's `phase` (the
    only way `'started'` surfaces), `author_user_id`, and `previous_run_id`. A store-only caller
    passes none and gets the documented degradations. `transcript_view` runs once over the
    resolved turns and returns the turns to show.

    Unlike `load_conversation` this reads every run (no cap) and every branch. Delegate runs,
    `parent_run_id` pointing at another run of this conversation (a helper agent spawned inside a
    tool call), are left out; a parent outside the conversation means the agent itself ran
    nested, and its turn stays.
    """
    if not conversation_id:
        raise ValueError('conversation_id must be non-empty')
    runs = await step_store.list_runs(conversation_id=conversation_id)
    own = {run.run_id for run in runs}
    records = {record.run_id: record for record in turns} if turns is not None else {}
    resolved: list[TranscriptTurn] = []
    for run in runs:
        if run.parent_run_id in own:
            continue
        turn = await _resolve_run_turn(step_store, run, record=records.get(run.run_id))
        if turn is not None:
            resolved.append(turn)
    return list(transcript_view(resolved)) if transcript_view is not None else resolved


def active_run_ids(turns: Sequence[TranscriptTurn]) -> set[str]:
    """The run ids on the branch a reload shows: the latest-started turn and every turn it builds
    on, following `previous_run_id`. A turn off that branch is a retried or forked-away attempt,
    or a root a retry of a first turn replaced; without lineage (no `turns` given to
    `load_transcript`) the branch is the latest turn alone."""
    by_id = {turn.run_id: turn for turn in turns}
    active: set[str] = set()
    cursor: str | None = turns[-1].run_id if turns else None
    while cursor is not None and cursor not in active:
        active.add(cursor)
        turn = by_id.get(cursor)
        cursor = turn.previous_run_id if turn is not None else None
    return active


def path_messages(turns: Sequence[TranscriptTurn]) -> list[ModelMessage]:
    """The branch a reload shows, flat: the messages of the turns in `active_run_ids`, in
    run-start order, assembled as the model path assembles a chain."""
    active = active_run_ids(turns)
    messages: list[ModelMessage] = []
    for turn in turns:
        if turn.run_id in active:
            messages.extend(turn.messages)
    return messages
