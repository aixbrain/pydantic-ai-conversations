"""FastAPI binding of the feature.

Route parameters expose exactly the client-side (`extras.*`) part of the scope -- the user side
comes from auth and never appears in the API. The `extras.*` query parameters are UNVERIFIED
client claims, validated for shape only: an app whose scope contains `extras.*` MUST verify the
claim in its own `AuthProvider`, or any authenticated caller can read and delete any partition by
naming it in the query string.

Mount this router with the exact same `auth` callable object the app itself guards with:
FastAPI's dependency cache is per callable object, so a second, behaviorally identical callable
runs the provider twice. `ConversationsFeature` gets that object from the app factory; only a
standalone mount has to pass it by hand.
"""

# No `from __future__ import annotations`: FastAPI resolves `Depends(...)` from real annotation
# objects, and stringized annotations cannot see a closure-scoped dependency like `scope_params`.

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal, TypeAlias
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, StringConstraints, create_model
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai_harness.step_persistence import StepStore
from pydantic_core import to_jsonable_python

from pydantic_ai_conversations._loader import TranscriptTurn, TranscriptView, load_transcript, path_messages
from pydantic_ai_conversations._transient import is_transient
from pydantic_ai_conversations.errors import ConversationNotFound, RegistryUnavailable
from pydantic_ai_conversations.registry import ConversationMeta, ConversationRegistry, RunPhase
from pydantic_ai_conversations.scope import HasUserId, ScopeField, build_namespace, scope_extras_model

DeletionKind: TypeAlias = Literal['soft_delete', 'purge']
"""Which deletion path invoked a cleanup hook."""


@dataclass(frozen=True)
class DeletionEvent:
    """What a cleanup hook is told about a deletion."""

    namespace_key: str
    conversation_id: str
    kind: DeletionKind
    meta: ConversationMeta | None
    """The entry as it was, or `None` when purging one that was already soft-deleted or purged
    (a retried purge) -- the registry's `get` hides both by contract, so the purge path cannot
    recover it."""


OnDeleteHook: TypeAlias = Callable[[DeletionEvent], Awaitable[None]]

UserDependency: TypeAlias = Callable[..., HasUserId | Awaitable[HasUserId]]
"""Any FastAPI dependency resolving the request to an object with the user-side scope attributes.
The app-factory's `AuthProvider` satisfies this."""

UIMessageEncoder: TypeAlias = Callable[[Sequence[ModelMessage]], Sequence[Any]]
"""Renders a turn's messages into the app's UI protocol for `TranscriptTurnOut.ui_messages`.

A UI adapter's `dump_messages` classmethod satisfies it -- `VercelAIAdapter.dump_messages`,
`AGUIAdapter.dump_messages`, or either wrapped in `functools.partial` to pin its options. The
package calls the encoder and JSON-encodes the result; it never inspects the protocol."""

_logger = logging.getLogger(__name__)


class ConversationOut(BaseModel):
    conversation_id: str
    title: str | None
    created_by: str
    created_at: datetime
    last_message_at: datetime | None
    message_count: int
    attrs: Mapping[str, str]


class ConversationList(BaseModel):
    data: list[ConversationOut]
    more_after: str | None


class TurnErrorOut(BaseModel):
    """Why a `'failed'` turn failed, as the app's `describe_failure` mapper reported it."""

    code: str
    message: str
    retryable: bool
    """`TurnError.transient`: re-sending the message may succeed unchanged, as after a model
    outage. A rejected API key is not retryable; the turn can still be re-sent once it is fixed."""


class TranscriptTurnOut(BaseModel):
    run_id: str
    started_at: datetime
    phase: RunPhase
    """The lifecycle of the run in `run_id`, authoritative from the registry: render `'answered'`
    normally, `'failed'` with a retry affordance, `'started'` as in-flight. A hard-killed run
    keeps `'started'` forever -- with no own request open, treat a stale `'started'` as failed."""
    previous_run_id: str | None
    """The turn this one builds on, `None` for the first turn. Every turn of every branch is
    listed; a client renders the branch ending in the latest turn by following these links back,
    and names the turn it continues from as `previous_run_id` on its next request."""
    author_user_id: str | None
    """Who wrote this turn, from the registry's `TurnRecord`s; `None` when the registry has no
    attribution for the run."""
    error: TurnErrorOut | None = None
    """Why the run failed; `None` unless the feature was configured with `describe_failure` and
    the turn is `'failed'`."""
    messages: list[Any]
    """The turn's messages: what the run stored, which for a failed turn is its prompt, or its
    prompt and the tool rounds it settled before failing."""
    ui_messages: list[Any] | None = None
    """The same messages in the app's UI protocol, present only when the router was built with a
    `ui_messages` encoder; `None` otherwise -- the field never replaces `messages`.

    A frontend seeding its chat state reads this per turn, not the flat `messages` of the
    enclosing `ConversationDetail`: that field carries no `phase`, while a seed needs the `failed`
    turn to carry a retry affordance and the `started` one to render as in-flight. A failed
    turn's messages end in its prompt or a tool return, never in a tool call without a result,
    so `phase` is what tells the frontend the turn failed."""


class ConversationDetail(BaseModel):
    conversation: ConversationOut
    turns: list[TranscriptTurnOut]
    messages: list[Any]
    """The branch ending in the latest turn, flat: every turn on it concatenated in run-start
    order, as the model path assembles a chain. A compatibility field for callers reading a flat
    history; `turns` is canonical and also carries the other branches, which a flat list cannot
    represent."""


class CreateConversationBody(BaseModel):
    """Optional JSON body for `POST /conversations`.

    The explicit-create path's write hook for the create-time-only `attrs`; the implicit-create
    path has no body and always creates with none (pre-create the entry to set them there).
    """

    attrs: Mapping[str, str] = Field(default_factory=dict)


class SetTitleBody(BaseModel):
    """JSON body for `PUT /conversations/{id}/title`.

    Stripped before the length check, so a whitespace-only title is a 422 rather than a blank row
    in the listing. The cap is a sanity bound on a sidebar label, not a product decision -- an app
    wanting a stricter one validates before calling, and one wanting a longer label stores it in
    `attrs`. There is no clearing: `set_title` takes a `str`, and a conversation that never had a
    title is already the `None` state.
    """

    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


def _to_out(meta: ConversationMeta) -> ConversationOut:
    return ConversationOut(
        conversation_id=meta.conversation_id,
        title=meta.title,
        created_by=meta.created_by,
        created_at=meta.created_at,
        last_message_at=meta.last_message_at,
        message_count=meta.message_count,
        attrs=meta.attrs,
    )


def _encode_ui(encoder: UIMessageEncoder | None, messages: Sequence[ModelMessage]) -> list[Any] | None:
    """One turn's messages in the app's UI protocol, JSON-ready, or `None` with no encoder wired.

    The UI adapters' message models are camelCase-aliased and a `list[Any]` response field
    serializes them by attribute name, so the result is dumped `by_alias=True` to emit
    `toolCallId` rather than `tool_call_id`. An encoder raising fails the request rather than
    degrading to `None`, which would be indistinguishable from "no encoder configured".
    """
    if encoder is None:
        return None
    return list(to_jsonable_python(encoder(messages), by_alias=True))


def _filter_params_model(filterable_attrs: Sequence[str]) -> type[BaseModel]:
    """Declared attrs keys as OPTIONAL, non-empty-string query parameters (undeclared attrs are
    not queryable)."""
    fields: dict[str, Any] = {name: (Annotated[str | None, Field(min_length=1)], None) for name in filterable_attrs}
    return create_model('FilterParams', **fields)


def _not_found_404() -> HTTPException:
    # `HTTPException` with a structured `detail`, not `AgentError`: a standalone mount has no
    # `AgentError` exception handler, so the same router stays correct under both mounts.
    return HTTPException(
        status_code=404,
        detail={'code': 'conversations/not-found', 'message': 'The conversation does not exist.'},
    )


def _unavailable_503() -> HTTPException:
    # Same shape decision as `_not_found_404`.
    return HTTPException(
        status_code=503,
        detail={
            'code': 'conversations/unavailable',
            'message': 'Conversation storage is temporarily unavailable.',
            'retryable': True,
        },
    )


async def _fetch_transcript(
    registry: ConversationRegistry,
    step_store: StepStore,
    namespace: str,
    conversation_id: str,
    transcript_view: TranscriptView | None = None,
) -> list[TranscriptTurn]:
    """The conversation's transcript, each turn stamped with its author, phase, and lineage from
    the registry.

    Both reads map a temporary failure to a retryable 503: the registry's `RegistryUnavailable`,
    and whatever the step store declares temporary (`stores.SupportsTransientErrors`). Any other
    step-store failure is a 500, because retrying a malformed row cannot help.
    """
    try:
        turn_records = await registry.list_turns(namespace, conversation_id)
    except RegistryUnavailable:
        _logger.warning(
            'registry unavailable while fetching turn records for conversation %s', conversation_id, exc_info=True
        )
        raise _unavailable_503() from None
    # `list_turns`'s ConversationNotFound is unreachable behind `get_conversation`'s `get` gate.
    try:
        return await load_transcript(
            step_store, conversation_id=conversation_id, turns=turn_records, transcript_view=transcript_view
        )
    except Exception as exc:
        if not is_transient(step_store, exc):
            raise
        _logger.warning(
            'step store unavailable while loading transcript for conversation %s', conversation_id, exc_info=True
        )
        raise _unavailable_503() from None


async def _soft_delete(registry: ConversationRegistry, namespace: str, conversation_id: str) -> ConversationMeta:
    """Soft-delete, mapping the registry's two failure modes onto the routes' error contract."""
    try:
        return await registry.soft_delete(namespace, conversation_id)
    except ConversationNotFound:
        raise _not_found_404() from None
    except RegistryUnavailable:
        _logger.warning('registry unavailable while deleting conversation %s', conversation_id, exc_info=True)
        raise _unavailable_503() from None


async def _write_title(registry: ConversationRegistry, namespace: str, conversation_id: str, title: str) -> None:
    """Set the title, mapping the registry's two failure modes onto the routes' error contract."""
    try:
        await registry.set_title(namespace, conversation_id, title)
    except ConversationNotFound:
        # A foreign id answers 404 like a missing one -- a write route must not become the
        # existence oracle the read routes refuse to be.
        raise _not_found_404() from None
    except RegistryUnavailable:
        _logger.warning('registry unavailable while titling conversation %s', conversation_id, exc_info=True)
        raise _unavailable_503() from None


def _check_filterable_attrs(scope: tuple[ScopeField, ...], filterable_attrs: Sequence[str]) -> None:
    """Reject a `filterable_attrs` name that would silently misbehave once wired.

    A name colliding with a scope field or a pagination param binds twice (the second binding
    wins arbitrarily); a non-identifier degrades the generated model into an always-required
    `extra_data` param. Both fail here, at router-build time.
    """
    reserved = {f.name for f in scope if f.source == 'extras'} | {'pageAfter', 'pageSize'}
    for name in filterable_attrs:
        if not name.isidentifier():
            raise ValueError(f'filterable attr {name!r} is not a valid identifier')
        if name in reserved:
            raise ValueError(f'filterable attr {name!r} collides with a scope field or pagination param')


def build_conversation_router(
    *,
    registry: ConversationRegistry,
    step_store: StepStore,
    scope: tuple[ScopeField, ...],
    auth: UserDependency,
    enable_create: bool = False,
    enable_title: bool = False,
    filterable_attrs: Sequence[str] = (),
    on_delete: Sequence[OnDeleteHook] = (),
    transcript_view: TranscriptView | None = None,
    ui_messages: UIMessageEncoder | None = None,
) -> APIRouter:
    """Build the `/conversations` router.

    `on_delete` hooks run inline on the DELETE path, after the soft-delete commits and before the
    204 returns -- a slow hook delays the response; a failing one does not fail it.

    `enable_title` mounts the write route for `registry.set_title`. It is opt-in because WHO may
    rename a conversation is a product decision: an app that titles from the model or the first
    message wants no user-facing rename at all, and an unmounted route cannot be called.

    `ui_messages` populates each turn's `ui_messages` field; without one the field stays `None`
    and every turn's content is carried once.
    """
    _check_filterable_attrs(scope, filterable_attrs)

    router = APIRouter(prefix='/conversations', tags=['conversations'])
    scope_params = scope_extras_model(scope) or create_model('ScopeParams')
    filter_params = _filter_params_model(filterable_attrs)

    @router.get('')
    async def list_conversations(
        user: Annotated[HasUserId, Depends(auth)],
        params: Annotated[Any, Depends(scope_params)],
        filters: Annotated[Any, Depends(filter_params)],
        page_after: Annotated[str | None, Query(alias='pageAfter')] = None,
        page_size: Annotated[int, Query(ge=1, le=500, alias='pageSize')] = 100,
    ) -> ConversationList:
        namespace = build_namespace(scope, user, params.model_dump())
        attrs = {key: value for key, value in filters.model_dump().items() if value is not None}
        try:
            rows = await registry.list(namespace, attrs=attrs or None, page_after=page_after, limit=page_size + 1)
        except RegistryUnavailable:
            _logger.warning('registry unavailable while listing conversations', exc_info=True)
            raise _unavailable_503() from None
        more_after = rows[page_size - 1].conversation_id if len(rows) > page_size else None
        return ConversationList(data=[_to_out(m) for m in rows[:page_size]], more_after=more_after)

    @router.get('/{conversation_id}')
    async def get_conversation(
        conversation_id: str,
        user: Annotated[HasUserId, Depends(auth)],
        params: Annotated[Any, Depends(scope_params)],
    ) -> ConversationDetail:
        namespace = build_namespace(scope, user, params.model_dump())
        try:
            meta = await registry.get(namespace, conversation_id)
        except RegistryUnavailable:
            _logger.warning('registry unavailable while fetching conversation %s', conversation_id, exc_info=True)
            raise _unavailable_503() from None
        if meta is None:
            raise _not_found_404()
        transcript = await _fetch_transcript(registry, step_store, namespace, conversation_id, transcript_view)
        turns = [
            TranscriptTurnOut(
                run_id=turn.run_id,
                started_at=turn.started_at,
                phase=turn.phase,
                previous_run_id=turn.previous_run_id,
                author_user_id=turn.author_user_id,
                error=None
                if turn.error is None
                else TurnErrorOut(code=turn.error.code, message=turn.error.message, retryable=turn.error.transient),
                messages=ModelMessagesTypeAdapter.dump_python(turn.messages, mode='json'),
                ui_messages=_encode_ui(ui_messages, turn.messages),
            )
            for turn in transcript
        ]
        return ConversationDetail(
            conversation=_to_out(meta),
            turns=turns,
            messages=ModelMessagesTypeAdapter.dump_python(path_messages(transcript), mode='json'),
        )

    @router.delete('/{conversation_id}', status_code=204)
    async def delete_conversation(
        conversation_id: str,
        user: Annotated[HasUserId, Depends(auth)],
        params: Annotated[Any, Depends(scope_params)],
    ) -> None:
        namespace = build_namespace(scope, user, params.model_dump())
        meta = await _soft_delete(registry, namespace, conversation_id)
        event = DeletionEvent(namespace_key=namespace, conversation_id=conversation_id, kind='soft_delete', meta=meta)
        for hook in on_delete:
            try:
                await hook(event)
            except Exception:
                # The conversation is gone from the caller's view either way; cleanup debt is an
                # operational concern, not the caller's.
                _logger.exception('on_delete hook failed for conversation %s', conversation_id)

    if enable_create:

        @router.post('', status_code=201)
        async def create_conversation(
            user: Annotated[HasUserId, Depends(auth)],
            params: Annotated[Any, Depends(scope_params)],
            body: CreateConversationBody | None = None,
        ) -> ConversationOut:
            namespace = build_namespace(scope, user, params.model_dump())
            try:
                meta = await registry.register_conversation(
                    namespace, uuid4().hex, created_by=user.user_id, attrs=body.attrs if body else None
                )
            except RegistryUnavailable:
                _logger.warning('registry unavailable while creating a conversation', exc_info=True)
                raise _unavailable_503() from None
            return _to_out(meta)

    if enable_title:

        @router.put('/{conversation_id}/title', status_code=204)
        async def set_title(
            conversation_id: str,
            body: SetTitleBody,
            user: Annotated[HasUserId, Depends(auth)],
            params: Annotated[Any, Depends(scope_params)],
        ) -> None:
            await _write_title(registry, build_namespace(scope, user, params.model_dump()), conversation_id, body.title)

    return router
