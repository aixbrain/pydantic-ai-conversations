"""Agent-side of the feature: registry upkeep and loading the conversation into the run.

Each run names the turn it builds on (`previous_run_id` in the run's extras); the registry keeps
that link per turn, and the run's history is the chain of links from the first turn to the named
one, so a retry or a fork is a second turn naming the same `previous_run_id`. The history is loaded in
`before_run`, into run state: from the first model request on, the run is a plain pydantic-ai
run over the conversation, and every other capability sees it that way. Stored snapshots stay
per-turn batches because the write seam (`stores.NewMessagesPreparer`) cuts each one back to the
run's own messages. The resolved namespace and chain, this capability's only run state, live on
the per-run copy `for_run` returns; the agent-bound instance stays stateless.

`ctx.conversation_id is None` guards below are defensive dead code: pydantic-ai resolves a
conversation id for every run before any capability hook fires.

Composition order: this capability BEFORE `StepPersistence(capture_frontier=True)`. The harness
saves the request about to be sent in its own `before_model_request`, and pydantic-ai stamps a
prompt that arrived inside `message_history` only after those hooks, so `before_model_request`
here must stamp it first for that save to carry the run's prompt. A run rejected here persists
neither a run record nor a snapshot. Leave `StepPersistence.run_id` unset: an explicit id there
decouples its `RunRecord` from the `ctx.run_id` everything here is keyed on.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext
from pydantic_ai_app_factory import AgentError, ErrorHandlers, to_agent_error
from pydantic_ai_harness.step_persistence import StepStore

from pydantic_ai_conversations._lineage import (
    PREVIOUS_RUN_ID,
    UNSET,
    PreviousRunRequired,
    RunNotFound,
    has_user_prompt,
    is_resumable,
    loaded_tail,
    resolve_chain,
    unmerge_loaded_tail,
)
from pydantic_ai_conversations._loader import load_conversation
from pydantic_ai_conversations._transient import is_transient
from pydantic_ai_conversations.errors import ConversationNotFound, RegistryUnavailable
from pydantic_ai_conversations.registry import ConversationRegistry, RunPhase, TurnError
from pydantic_ai_conversations.scope import HasUserId, ScopeField, build_namespace

_logger = logging.getLogger(__name__)
# The metadata namespace pydantic-ai's UI adapters reserve for their own state.
_RESERVED_METADATA = '__pydantic_ai__'


FailureDescriber = Callable[[BaseException], TurnError | None]
"""Turns a run's exception into the `TurnError` stored on its failed turn; `None` stores none.

`agent_error_mapper` builds one from the app's `error_handlers`, so the stored reason matches the
error the client saw live."""


def agent_error_mapper(error_handlers: ErrorHandlers | None = None) -> FailureDescriber:
    """A `FailureDescriber` that translates with the app factory's `to_agent_error`.

    Pass the same `error_handlers` as `create_agent_app`; without them every app-specific error
    is stored as `internal`. A cancelled run is stored as `conversations/cancelled`; other
    non-`Exception` errors (`KeyboardInterrupt`, `SystemExit`) store no reason.
    """

    def describe(exc: BaseException) -> TurnError | None:
        if isinstance(exc, asyncio.CancelledError):
            return TurnError('conversations/cancelled', 'The run was cancelled.', transient=True)
        if not isinstance(exc, Exception):
            return None
        error = to_agent_error(exc, error_handlers)
        return TurnError(error.code, error.message, transient=error.retryable)

    return describe


def _turn_length(ctx: RunContext[Any], messages: Sequence[ModelMessage]) -> int:
    """How many of `messages` are this run's turn, tool traffic included: everything from the first
    message stamped with its `run_id`, the rule `NewMessagesPreparer` stores and the transcript
    reads. A trailing tool return pydantic-ai has not stamped yet still belongs to the turn."""
    start = next((index for index, message in enumerate(messages) if message.run_id == ctx.run_id), None)
    return 0 if start is None else len(messages) - start


class SupportsConversationDeps(Protocol):
    """What the capability needs from a run's deps object, structurally.

    Read-only properties so any dataclass/attrs/plain class satisfies it; the app-factory's
    `AppDeps` does.
    """

    @property
    def user(self) -> HasUserId: ...

    @property
    def extras(self) -> Mapping[str, Any]: ...


DepsT = TypeVar('DepsT', bound=SupportsConversationDeps)


def _not_found() -> AgentError:
    return AgentError('conversations/not-found', 'The conversation does not exist.', http_status=404)


def _run_not_found() -> AgentError:
    return AgentError(
        'conversations/run-not-found', 'The previous run does not exist in this conversation.', http_status=404
    )


def _previous_run_required() -> AgentError:
    return AgentError(
        'conversations/previous-run-required',
        'previous_run_id is required after the first turn of a conversation.',
        http_status=422,
    )


def _prompt_required() -> AgentError:
    return AgentError(
        'conversations/prompt-required',
        'A message is required: the named turn left nothing to resume.',
        http_status=422,
    )


def _unavailable() -> AgentError:
    return AgentError('conversations/unavailable', 'Conversation storage is temporarily unavailable.', retryable=True)


def _is_store_outage(error: BaseException) -> bool:
    """Whether `error` is the retryable outage the write seam (or `before_run`) raised itself."""
    return isinstance(error, AgentError) and error.code == 'conversations/unavailable'


def _prompt_in_history(ctx: RunContext[Any]) -> ModelRequest | None:
    """The client list's trailing user request, or `None` when the prompt was passed separately.

    UI adapters send the new message inside `message_history` instead of as the run's prompt, so
    `before_run` keeps it when it replaces the client's list with the stored branch. A prompt
    passed separately is appended by the graph after `before_run`, and the client's list then
    holds only turns the store already has.
    """
    messages = ctx.messages
    if ctx.prompt is None and messages and has_user_prompt(messages[-1]):
        last = messages[-1]
        assert isinstance(last, ModelRequest)
        return last
    return None


@dataclass
class ConversationsCapability(AbstractCapability[DepsT]):
    """Maintains the registry around each run and loads the stored history into run state."""

    registry: ConversationRegistry
    step_store: StepStore
    scope: tuple[ScopeField, ...]
    describe_failure: FailureDescriber | None = None
    """Maps a failed run's exception to the `TurnError` stored on its turn (see
    `agent_error_mapper`). `None` (default) stores no reason, only `phase='failed'`. A mapper that
    raises is logged and stores no reason."""

    max_history_runs: int | None = None
    """Caps `load_conversation` to the last `max_history_runs` runs whose snapshot starts with a
    request.

    A continuation whose snapshot starts with its answer shares the slot of the run it continues,
    and a run without a snapshot takes none.

    The only lever on the per-run snapshot reads; a compaction capability on the agent shrinks
    what the model sees after those reads happened. `None` (default) reads every run; `<= 0`
    raises `ValueError`.
    """

    # Per-run state -- lives only on the copy `for_run` returns (`init=False`, so `replace`
    # resets it). The agent-bound original never carries it.
    _run_namespace: str | None = field(init=False, default=None, repr=False)
    _run_chain: list[str] = field(init=False, default_factory=list, repr=False)
    """The run ids this run's history is loaded from, first turn first."""
    _run_loaded_tail: list[ModelRequest] = field(init=False, default_factory=list, repr=False)
    """The requests the loaded history ends in, kept until the first model request so the split
    in `before_model_request` can restore them."""

    def __post_init__(self) -> None:
        if self.max_history_runs is not None and self.max_history_runs <= 0:
            raise ValueError(f'max_history_runs must be positive, got {self.max_history_runs!r}')

    async def for_run(self, ctx: RunContext[DepsT]) -> ConversationsCapability[DepsT]:
        """A fresh copy for this run, on which every lifecycle hook then runs.

        pydantic-ai shares a capability instance across runs unless `for_run` returns a new one,
        so run state lives on this copy. `replace` re-runs `__init__`, resetting the `init=False`
        fields; a nested run delegated with `deps=ctx.deps` gets its own copy too.
        """
        return replace(self)

    async def before_run(self, ctx: RunContext[DepsT]) -> None:
        """Register the conversation, resolve the turn the run builds on, load that branch,
        record the turn, then put the branch into run state.

        Every check runs before `start_turn`: a run naming an unknown or missing
        `previous_run_id`, one with nothing to send, or one whose history cannot be read is
        rejected before anything of it is recorded or saved.
        """
        conversation_id = ctx.conversation_id
        if conversation_id is None:
            return
        namespace = build_namespace(self.scope, ctx.deps.user, ctx.deps.extras)
        previous_run_id = ctx.deps.extras.get(PREVIOUS_RUN_ID, UNSET)
        try:
            await self.registry.register_conversation(namespace, conversation_id, created_by=ctx.deps.user.user_id)
            records = await self.registry.list_turns(namespace, conversation_id)
            chain = resolve_chain(records, previous_run_id)
            answered = {record.run_id for record in records if record.phase == 'answered'}
            history = await self._load_branch(conversation_id, chain, answered)
            prompt = _prompt_in_history(ctx)
            last_answered = bool(chain) and chain[-1] in answered
            if ctx.prompt is None and prompt is None and not is_resumable(history, answered=last_answered):
                raise _prompt_required()
            await self.registry.start_turn(
                namespace,
                conversation_id,
                # `ctx.run_id` is always set on a real run; the `''` fallback only satisfies the
                # type checker for a synthetic RunContext that never went through graph construction.
                run_id=ctx.run_id or '',
                author_user_id=ctx.deps.user.user_id,
                previous_run_id=chain[-1] if chain else None,
            )
        except ConversationNotFound as exc:
            # Our own exception, not the backend's -- safe to log verbatim (see errors.py).
            _logger.info('conversation %s not accessible in namespace: %s', conversation_id, exc)
            raise _not_found() from None
        except PreviousRunRequired:
            raise _previous_run_required() from None
        except RunNotFound:
            raise _run_not_found() from None
        except RegistryUnavailable as exc:
            _logger.warning(
                'registry unavailable while starting conversation %s: %s', conversation_id, exc, exc_info=True
            )
            raise _unavailable() from None
        self._run_namespace = namespace
        self._run_chain = chain
        self._run_loaded_tail = loaded_tail(history)
        # Replaced, never merged, so a client re-sending its full local copy cannot duplicate a
        # turn; the graph then stamps the run's own messages and appends a separately passed
        # prompt. The client's list is not inspected, so sanitizing it is the entry point's job
        # (the UI adapters and the app factory do; a hand-written route must).
        ctx.messages[:] = [*history, prompt] if prompt is not None else history

    async def _load_branch(
        self, conversation_id: str, chain: Sequence[str], answered: Collection[str]
    ) -> list[ModelMessage]:
        """The branch's stored turns, ready to become run state. `answered` names the runs that
        ended normally, whose newest snapshot counts even when marked `interrupted`. A failure the
        store declares temporary is retryable `conversations/unavailable`; any other propagates.
        """
        try:
            history = await load_conversation(
                self.step_store,
                conversation_id=conversation_id,
                run_ids=chain,
                max_history_runs=self.max_history_runs,
                answered_runs=answered,
            )
        except AgentError:
            raise  # the write seam already mapped a temporary failure
        except Exception as exc:
            if not is_transient(self.step_store, exc):
                raise
            _logger.warning(
                'step store unavailable while loading history for conversation %s', conversation_id, exc_info=True
            )
            raise _unavailable() from None
        return history

    async def before_model_request(
        self, ctx: RunContext[DepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Split the loaded open requests back out of the request pydantic-ai merged them into,
        then stamp the run's unstamped trailing requests with its run id and a fresh timestamp.

        Without the split, the merged request would take this run's id and store the loaded open
        requests again as this run's turn. pydantic-ai stamps a prompt sent inside
        `message_history` only after this hook, and a frontier save holding no message of the run
        is dropped at the write seam. `__pydantic_ai__` metadata is removed because the Vercel
        adapter derives a message's id from the client id kept there, so a retry of one message
        would render under the previous attempt's id.
        """
        messages = ctx.messages
        if self._run_loaded_tail:
            # Only the first model request can hold the merged request; later ones follow a response.
            unmerge_loaded_tail(messages, self._run_loaded_tail)
            self._run_loaded_tail = []
        start = len(messages)
        while start and isinstance(messages[start - 1], ModelRequest) and messages[start - 1].run_id is None:
            start -= 1
        for index in range(start, len(messages)):
            request = messages[index]
            assert isinstance(request, ModelRequest)
            # Mutated in place, as pydantic-ai's own stamping does: run state, the request about to
            # be sent, and the harness's at-failure stash all hold this object. A remainder split
            # off above is held by run state alone; pydantic-ai stamps the merged request it sends.
            request.run_id = ctx.run_id
            request.timestamp = datetime.now(UTC)
            if request.metadata and _RESERVED_METADATA in request.metadata:
                metadata = {key: value for key, value in request.metadata.items() if key != _RESERVED_METADATA}
                request.metadata = metadata or None
        return request_context

    async def after_run(self, ctx: RunContext[DepsT], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        """Update the listing digest and resolve the turn to `answered`.

        `message_count_delta` is read back from the stored turn (`_stored_count`), tool traffic
        included, so the listing counts what the preparer chain left in the store, the same figure
        the transcript shows.
        """
        conversation_id = ctx.conversation_id
        if conversation_id is None:
            return result
        await self._update_digest(ctx, conversation_id, await self._stored_count(ctx, answered=True))
        await self._record_phase(ctx, 'answered')
        return result

    async def on_run_error(self, ctx: RunContext[DepsT], *, error: BaseException) -> AgentRunResult[Any]:
        """Resolve the run's turn to `failed` with the `describe_failure` reason, then re-raise
        the original error unchanged.

        Without this a failed run would stay `'started'`, indistinguishable from one still in
        flight. The phase update is best-effort -- a bookkeeping failure must never mask the run
        error the caller needs to see.
        """
        turn_error = None
        if self.describe_failure is not None:
            try:
                turn_error = self.describe_failure(error)
            except Exception:
                _logger.warning('describe_failure raised for run %s; no reason stored', ctx.run_id, exc_info=True)
        # What the run stored before failing shows in the transcript: the prompt, and whatever the
        # model returned before the error. The listing counts it too, so a conversation whose turns
        # all failed does not list as empty. When the error is the write seam's own outage, the
        # store just reported itself down, and another read would only wait out a second driver
        # timeout before the error reaches the client.
        stored = 0 if _is_store_outage(error) else await self._stored_count(ctx, answered=False)
        if stored and ctx.conversation_id is not None:
            await self._update_digest(ctx, ctx.conversation_id, stored)
        await self._record_phase(ctx, 'failed', turn_error)
        raise error

    async def _stored_count(self, ctx: RunContext[DepsT], *, answered: bool) -> int:
        """How many messages the run left in the store, read the way the transcript reads a turn:
        its newest snapshot, an `interrupted` one included when the turn is answered (a run paused
        on open tool calls ends in one), the latest complete one when it failed. The run's own
        message list would count what the preparer chain may have dropped, and the error hook's
        context carries no history at all; the store is what the transcript shows either way. A
        store failure counts as nothing stored and is logged once, because a run error on its way
        out must not be replaced by it.
        """
        if ctx.run_id is None:
            return 0
        try:
            snapshot = await self.step_store.latest_snapshot(run_id=ctx.run_id, include_interrupted=answered)
        except AgentError:
            return 0  # the write seam already logged the temporary failure
        except Exception:
            _logger.warning('step store read failed after run %s failed; digest not updated', ctx.run_id, exc_info=True)
            return 0
        return 0 if snapshot is None else _turn_length(ctx, snapshot.messages)

    async def _update_digest(self, ctx: RunContext[DepsT], conversation_id: str, stored: int) -> None:
        """Overwrite `last_message_at` and add the run's `stored` messages to `message_count`,
        best-effort: registry failures are logged and swallowed, because the snapshots already
        landed and a stale listing row must not change the run's outcome.
        """
        # The `or build_namespace(...)` fallback is unreachable on a real run (`before_run` always
        # sets it first); it only matters for a synthetic RunContext driven straight into a hook.
        namespace = self._run_namespace or build_namespace(self.scope, ctx.deps.user, ctx.deps.extras)
        try:
            await self.registry.update_message_stats(
                namespace, conversation_id, last_message_at=datetime.now(UTC), message_count_delta=stored
            )
        except ConversationNotFound:
            # Deleted mid-run (a DELETE racing this run): the messages are persisted, only the
            # digest row is gone.
            _logger.info('conversation %s deleted mid-run; message-stats update skipped', conversation_id)
        except RegistryUnavailable:
            # Snapshots landed; the listing shows stale activity until the next update.
            _logger.warning(
                'registry message-stats update failed for conversation %s; digest is stale',
                conversation_id,
                exc_info=True,
            )

    async def _record_phase(self, ctx: RunContext[DepsT], phase: RunPhase, error: TurnError | None = None) -> None:
        """Resolve this run's turn to its terminal phase, best-effort: registry failures are
        logged and swallowed. Keyed by `ctx.run_id`, the same id `start_turn` used; an unknown
        run is the registry's documented no-op.
        """
        conversation_id = ctx.conversation_id
        if conversation_id is None or ctx.run_id is None:
            return
        namespace = self._run_namespace or build_namespace(self.scope, ctx.deps.user, ctx.deps.extras)
        try:
            await self.registry.set_turn_phase(namespace, conversation_id, run_id=ctx.run_id, phase=phase, error=error)
        except ConversationNotFound:
            _logger.info('conversation %s gone while recording run phase', conversation_id)
        except RegistryUnavailable:
            _logger.warning('failed to record run phase for conversation %s', conversation_id, exc_info=True)
