"""Turn lineage: the turn a request names, and the chain of turns behind it.

A request names the turn it builds on through `previous_run_id` in its extras. The field is
required once the conversation has a turn, so a client cannot continue "whatever came last" by
leaving it out; an explicit `None` starts a branch from nothing, as the first turn does.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, cast

from pydantic import BaseModel, Field, create_model, field_serializer
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from pydantic_ai_conversations.registry import TurnRecord

PREVIOUS_RUN_ID = 'previous_run_id'
"""The extras key naming the turn a run builds on; `ConversationsFeature` declares it on its
`extras_type`, so it arrives in `/run`'s `extras` and at the top level of a `/chat` body."""


class _Unset:
    """The value of an absent `previous_run_id`, as opposed to an explicit `None`."""


UNSET = _Unset()


class LineageExtras(BaseModel):
    """The chat body's `previous_run_id`, optional in the schema.

    An omitted field defaults to `UNSET` rather than `None` because the two are different
    requests: leaving it out is an error once the conversation has a turn, while an explicit
    `None` starts a branch from nothing."""

    # The declared type is what a client may send; the default is the internal sentinel.
    previous_run_id: Annotated[str | None, Field(min_length=1)] = Field(
        default_factory=cast('Callable[[], str | None]', lambda: UNSET)
    )

    @field_serializer(PREVIOUS_RUN_ID)
    def _keep_unset(self, value: object) -> object:
        # `model_dump()` hands the capability the sentinel as it is; a `str` serializer would
        # warn about it.
        return value


def extras_model(scope_extras: type[BaseModel] | None) -> type[BaseModel]:
    """The chat body's extras: the scope's `extras.*` fields plus the lineage field."""
    bases = (scope_extras, LineageExtras) if scope_extras is not None else (LineageExtras,)
    return create_model('ConversationsExtras', __base__=bases)


class PreviousRunRequired(Exception):
    """`previous_run_id` was left out of a request to a conversation that already has a turn."""


class RunNotFound(LookupError):
    """`previous_run_id` names no recorded turn of this conversation."""


def resolve_chain(turns: Sequence[TurnRecord], previous_run_id: object) -> list[str]:
    """The run ids the new turn builds on, first turn first: `previous_run_id` and every turn
    behind it, following the registry's links.

    `previous_run_id` must name a recorded turn of this conversation, or be an explicit `None`
    to start from nothing (the first turn, or a retry of one). It may be left out only while the
    conversation has no turns, so a left-out id is a request error rather than a continuation of
    the latest turn. The walk stops at a link it has seen, so a corrupted chain cannot loop.
    """
    by_id = {record.run_id: record for record in turns}
    if previous_run_id is UNSET:
        if by_id:
            raise PreviousRunRequired(previous_run_id)
        return []
    if previous_run_id is None:
        return []
    if not isinstance(previous_run_id, str) or previous_run_id not in by_id:
        raise RunNotFound(previous_run_id)
    chain: list[str] = []
    cursor: str | None = previous_run_id
    while cursor is not None and cursor not in chain:
        chain.append(cursor)
        record = by_id.get(cursor)
        cursor = record.previous_run_id if record is not None else None
    chain.reverse()
    return chain


def loaded_tail(history: Sequence[ModelMessage]) -> list[ModelRequest]:
    """The requests `history` ends in, in order: what the named turn left unanswered. One
    request normally (a prompt, or the tool returns of a settled round), and one more for each
    continuation that sent a new message and failed before answering.

    pydantic-ai merges consecutive requests of `message_history` into one at run start, so the
    capability keeps these to restore them with `unmerge_loaded_tail`."""
    tail: list[ModelRequest] = []
    for message in reversed(history):
        if not isinstance(message, ModelRequest):
            break
        tail.insert(0, message)
    return tail


def unmerge_loaded_tail(messages: list[ModelMessage], tail: Sequence[ModelRequest]) -> bool:
    """Split the request pydantic-ai merged out of `tail` back into `tail` (the original objects,
    with their stamps) and one request holding the remainder, if any.

    pydantic-ai merges consecutive requests of `message_history` into one unstamped request at run
    start: the loaded open request with a new message sent inside `message_history`, or two loaded
    open requests with each other when no message was sent. Only a trailing unstamped request whose
    parts begin with those of the tail requests missing from `messages` is split, since requests
    with differing `instructions` stay apart; `False` otherwise.
    """
    if not tail or not messages:
        return False
    merged = messages[-1]
    if not isinstance(merged, ModelRequest) or merged.run_id is not None:
        return False
    pending = [request for request in tail if not any(message is request for message in messages)]
    if not pending:
        return False
    tail_parts = [part for request in pending for part in request.parts]
    if len(merged.parts) < len(tail_parts):
        return False
    for own, loaded in zip(merged.parts, tail_parts, strict=False):
        if own is not loaded and own != loaded:
            return False
    restored: list[ModelMessage] = [*pending]
    if rest := merged.parts[len(tail_parts) :]:
        restored.append(ModelRequest(parts=list(rest), instructions=merged.instructions, metadata=merged.metadata))
    messages[-1:] = restored
    return True


def has_user_prompt(message: ModelMessage) -> bool:
    """Whether `message` is a request carrying a user prompt part, whatever else it carries.

    UI adapters send the new message as the last entry of `message_history` instead of as the
    run's prompt, and may add tool returns to that request, so this is how the capability finds
    it."""
    return isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart) for part in message.parts)


def is_resumable(history: Sequence[ModelMessage], *, answered: bool = False) -> bool:
    """Whether a run can proceed from `history` without a new prompt.

    pydantic-ai, given no prompt, re-sends a trailing request, runs open tool calls, or continues
    a suspended turn. On a plain answer it would only send an instructions-only request or replay
    the answer, so the capability asks for a new message there. `answered` says the turn the
    history ends with was answered: a trailing request is then the final-result tool's return of a
    `ToolOutput` run, which has nothing to resume either."""
    if not history:
        return False
    last = history[-1]
    if isinstance(last, ModelRequest):
        return not answered
    return bool(last.tool_calls) or last.state == 'suspended'
