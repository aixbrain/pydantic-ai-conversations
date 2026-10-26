"""Recognizers for output-validator-rejected drafts.

A `ModelResponse` directly followed by a request carrying a `RetryFeedbackPart` was rejected as
a final answer; that covers every text-path rejection, plain text and `NativeOutput`/
`PromptedOutput` alike. A `ToolOutput`-mode rejection answers its call as a
`ToolReturnPart(outcome='retried')`, indistinguishable from a function tool's retry, and is left
alone. On retry exhaustion pydantic-ai raises before appending the last feedback, so the final
rejected draft carries no marker and is not found here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, RetryFeedbackPart, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext


def _is_rejection_marker(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(isinstance(part, RetryFeedbackPart) for part in message.parts)


def _is_tool_call_free_response(message: ModelMessage) -> bool:
    """Whether `message` is a `ModelResponse` carrying no `ToolCallPart` (text/thinking only).

    Only such responses are safe to drop: removing one WITH tool calls would orphan the parts
    answering its calls -- a tool return, retried or not, which carries the call's
    `tool_call_id`."""
    return isinstance(message, ModelResponse) and not any(isinstance(part, ToolCallPart) for part in message.parts)


def rejected_draft_pairs(messages: Sequence[ModelMessage]) -> list[tuple[int, int]]:
    """`(draft_index, retry_request_index)` for every validator-rejected draft in `messages`."""
    return [
        (index, index + 1)
        for index in range(len(messages) - 1)
        if _is_tool_call_free_response(messages[index]) and _is_rejection_marker(messages[index + 1])
    ]


def strip_rejected_drafts(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove every rejected draft together with its retry feedback.

    The draft is dropped whole; the retry request keeps its other parts (rebuilt via `replace`,
    so metadata survives) and is dropped only when emptied. Idempotent; returns `messages`
    unchanged when there is nothing to strip. `StripRejectedDrafts` applies it per model request.
    """
    pairs = rejected_draft_pairs(messages)
    if not pairs:
        return messages
    drafts = {draft_index for draft_index, _ in pairs}
    markers = {marker_index for _, marker_index in pairs}
    result: list[ModelMessage] = []
    for index, message in enumerate(messages):
        if index in drafts:
            continue
        if index in markers and isinstance(message, ModelRequest):
            kept = [part for part in message.parts if not isinstance(part, RetryFeedbackPart)]
            if kept:
                result.append(replace(message, parts=kept))
            continue
        result.append(message)
    return result


@dataclass
class StripRejectedDrafts(AbstractCapability[Any]):
    """Keeps validator-rejected drafts of earlier turns out of the model's context.

    pydantic-ai's output-validator retry leaves each rejected draft and its retry feedback in the
    history, so a later turn would send them to the model again. Applies
    `strip_rejected_drafts` over the loaded history ahead of the run's own messages, on every
    model request. Request-only: run state keeps the drafts, so stored snapshots stay complete
    and transcripts show them, and the run's own messages (stamped with its `run_id`, or not yet
    stamped) are never touched, because an in-flight retry needs its draft/feedback pair intact.
    Register it on the agent beside `ConversationsCapability`.
    """

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        messages = request_context.messages
        own_from = next(
            (index for index, message in enumerate(messages) if message.run_id in (ctx.run_id, None)), len(messages)
        )
        history = strip_rejected_drafts(list(messages[:own_from]))
        if len(history) == own_from:
            return request_context
        request_context.messages = [*history, *messages[own_from:]]
        return request_context
