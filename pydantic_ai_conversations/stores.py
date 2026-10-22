"""Optional service-side extensions a StepStore implementation may provide -- and the write seam.

The harness `StepStore` protocol has no deletion, no ping, and no error contract; this package
discovers all three structurally at runtime instead of requiring them, and apps opt in with a thin
subclass of their harness store (see docs/extending.md for an example). `PreparingStepStore` is
the feature's one declared write seam in front of the substrate.

`isinstance` against a `runtime_checkable` protocol only checks that the named members exist --
not their signature or behavior. An incompatible `delete_runs`/`ping` passes the check and fails
at call time with a `TypeError`.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic_ai.messages import ModelResponse
from pydantic_ai_app_factory import AgentError
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    StepStore,
    ToolEffectRecord,
)

from pydantic_ai_conversations._transient import is_transient

_logger = logging.getLogger(__name__)

_T = TypeVar('_T')


@runtime_checkable
class SupportsRunDeletion(Protocol):
    """A step store that can hard-delete runs (events, snapshots, tool effects included).

    Discovered by `ConversationsFeature.purge`: when present, a purge also removes the
    conversation's runs from the substrate. Must be idempotent (unknown run ids are a no-op) --
    purge may be retried. Shared media blobs stay with the store's own garbage collection.
    """

    async def delete_runs(self, *, run_ids: Sequence[str]) -> None: ...  # pragma: no cover


@runtime_checkable
class SupportsPing(Protocol):
    """A step store that can answer a cheap liveness round-trip for `/readyz`.

    Same contract as the registry's ping: must be cheap, raising means unhealthy, and the report
    is public -- operational status only, no counts.
    """

    async def ping(self) -> Mapping[str, Any]: ...  # pragma: no cover


@runtime_checkable
class SupportsTransientErrors(Protocol):
    """A step store that names the exceptions meaning "temporarily unreachable".

    Harness stores raise their driver's exceptions, and a run id already stored or a malformed row
    raises `ValueError`. The write seam and the transcript read turn only the exceptions named here
    into the retryable `conversations/unavailable` (503); anything else reaches the caller
    unchanged, because retrying a permanent failure cannot succeed. A store without the attribute
    has no failure mapped.
    """

    transient_errors: tuple[type[Exception], ...]


class SnapshotPreparer(Protocol):
    """A write-seam transform applied to every snapshot before it is persisted.

    Wired via `ConversationsFeature(snapshot_preparers=...)`, after `NewMessagesPreparer`, so a
    custom preparer sees the run's own turn only. Implementations must be idempotent, key
    per-run state by `run_id` (unknown runs pass through), and rebuild messages with
    `replace(msg, parts=...)` so `run_id` stamps survive. Never remove the run's own
    `UserPromptPart`s, tool returns, or tool-bound retries.
    """

    def prepare_snapshot(self, snapshot: ContinuableSnapshot) -> ContinuableSnapshot: ...  # pragma: no cover


class PreparingStepStore:
    """Wraps a `StepStore`, runs the preparer chain on every snapshot before it is saved, and turns
    a temporary failure of the wrapped store into the retryable `conversations/unavailable` error;
    everything else delegates to the wrapped store.

    Sits below the capability hooks, so every write path is covered -- boundary snapshots, the
    streaming fallback, and the at-failure save, which no capability hook can intercept. A
    snapshot the chain empties is not persisted (logged at DEBUG); preparer exceptions propagate
    and fail the save. The harness calls the store from inside the run, where a driver exception
    would otherwise reach the caller as `internal` and not retryable; a failure the wrapped store
    declares temporary (`SupportsTransientErrors`) raises the app factory's `AgentError` instead,
    logged at WARNING with the cause. Structural extras of the inner store stay reachable via
    `__getattr__` and are not translated, because purge and readiness handle their own failures.
    """

    def __init__(self, inner: StepStore, preparers: Sequence[SnapshotPreparer] = ()) -> None:
        self._inner = inner
        self._preparers = tuple(preparers)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        for preparer in self._preparers:
            snapshot = preparer.prepare_snapshot(snapshot)
        if not snapshot.messages:
            _logger.debug(
                'run %s step %s: snapshot emptied by the preparer chain; skipping save',
                snapshot.run_id,
                snapshot.step_index,
            )
            return
        await self._guarded('save_snapshot', self._inner.save_snapshot(snapshot))

    async def register_run(self, record: RunRecord) -> None:
        await self._guarded('register_run', self._inner.register_run(record))

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        return await self._guarded('get_run', self._inner.get_run(run_id=run_id))

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None
    ) -> list[RunRecord]:
        return await self._guarded(
            'list_runs', self._inner.list_runs(parent_run_id=parent_run_id, conversation_id=conversation_id)
        )

    async def append_event(self, event: StepEvent) -> None:
        await self._guarded('append_event', self._inner.append_event(event))

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        return await self._guarded('list_events', self._inner.list_events(run_id=run_id))

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        return await self._guarded(
            'latest_snapshot', self._inner.latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)
        )

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._guarded('record_tool_effect', self._inner.record_tool_effect(record))

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        return await self._guarded(
            'get_tool_effect', self._inner.get_tool_effect(run_id=run_id, tool_call_id=tool_call_id)
        )

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        return await self._guarded(
            'list_unresolved_tool_effects', self._inner.list_unresolved_tool_effects(run_id=run_id)
        )

    async def _guarded(self, operation: str, call: Awaitable[_T]) -> _T:
        """Await one store call; a temporary failure becomes the retryable `conversations/unavailable`
        error with the driver exception as its cause, so the log keeps the detail the client never
        sees. Any other failure propagates unchanged.
        """
        try:
            return await call
        except Exception as exc:
            if not is_transient(self._inner, exc):
                raise
            _logger.warning('step store %s failed with a temporary error', operation, exc_info=True)
            raise AgentError(
                'conversations/unavailable', 'Conversation storage is temporarily unavailable.', retryable=True
            ) from exc

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class NewMessagesPreparer:
    """Reduces every snapshot to the run's new messages: the suffix from the first message
    stamped with the snapshot's `run_id`. That is the run's `result.new_messages()`, except that
    a prompt sent inside `message_history` is kept, since it is the turn's prompt.

    A suffix rather than a stamp filter because pydantic-ai stamps a request only when it sends
    it, so a boundary save's trailing tool return is still unstamped and belongs to the turn. A
    `complete` snapshot with no message of the run is emptied, which skips the save and keeps the
    run's frontier save: the harness's at-failure save can hold the list from before pydantic-ai
    rebuilt it, where a prompt sent inside `message_history` is the client's unstamped original.
    An unstamped `interrupted` snapshot passes through. A trailing empty `interrupted` response is
    dropped: pydantic-ai appends it when a streamed model call fails before its first part, and it
    would otherwise end the failed turn in place of its prompt. `ConversationsFeature` wires this
    preparer ahead of your own.
    """

    def prepare_snapshot(self, snapshot: ContinuableSnapshot) -> ContinuableSnapshot:
        messages = snapshot.messages
        new_start = next((index for index, message in enumerate(messages) if message.run_id == snapshot.run_id), None)
        if new_start is None:
            # No message of the run: the at-failure replay of an unstamped start-of-run input.
            return snapshot if snapshot.state == 'interrupted' else dataclasses.replace(snapshot, messages=[])
        new_messages = list(messages[new_start:])
        last = new_messages[-1]
        if isinstance(last, ModelResponse) and not last.parts and last.state == 'interrupted':
            # pydantic-ai's marker for a stream that failed before its first part.
            new_messages.pop()
            return dataclasses.replace(snapshot, messages=new_messages)
        if new_start == 0:
            return snapshot
        return dataclasses.replace(snapshot, messages=new_messages)
