"""The add-on contract: what `ConversationsFeature(addons=...)` composes, and the capabilities an
add-on may opt into.

An add-on is a name plus whichever of the `Supports*` protocols it implements; the feature detects
each with `isinstance`, so an add-on that brings routes and no inline data declares no turn
fields, and a new capability never widens `ConversationAddon` itself. The mechanics
(how turn fields reach `GET /conversations/{id}`, how routers are mounted) live with the routes
and the feature; this module holds only the contract an add-on author reads.

As with the store extensions in `stores.py`, `isinstance` against a `runtime_checkable` protocol
only checks that the named members exist, not their signatures: a `load_turns` with the wrong
shape passes the check and fails at call time.
"""

# No `from __future__ import annotations`: FastAPI resolves `Depends(...)` from real annotation
# objects, and `AddonRouteGuard.caller()` builds such a dependency.

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, create_model

from pydantic_ai_conversations._hooks import DeletionEvent
from pydantic_ai_conversations._http import not_found_404, unavailable_503
from pydantic_ai_conversations._loader import TranscriptTurn
from pydantic_ai_conversations.errors import RegistryUnavailable
from pydantic_ai_conversations.registry import ConversationMeta, ConversationRegistry, TurnRecord
from pydantic_ai_conversations.scope import (
    HasUserId,
    ScopeField,
    UserDependency,
    build_namespace,
    scope_extras_model,
)

_logger = logging.getLogger(__name__)


class ConversationAddon(Protocol):
    """Something composed into the feature through `addons=`: a name, plus whichever of the
    optional capabilities it implements (`SupportsTurnFields`, `SupportsRouters`, `SupportsLifespan`,
    `SupportsDeletion`)."""

    # A read-only property, not an attribute: a mutable protocol attribute is invariant under
    # pyright, so a class attribute would not satisfy a wider declared type.
    @property
    def name(self) -> str:
        """A slug; the key under `addons`, a field name of the generated response model, and what
        `config()` reports. Unique per feature, checked when the app is built."""
        ...


@dataclass(frozen=True)
class AddonTurnQuery:
    """What a turn add-on's loader is told about one detail query."""

    namespace_key: str
    conversation_id: str
    user: HasUserId
    """The caller, for add-ons whose entry depends on who reads (my rating of a shared turn)."""
    turns: Sequence[TranscriptTurn]
    """The turns the response will show, after `transcript_view`; a loader keys its result by
    their `run_id`s and need not query for turns the view hid."""


@runtime_checkable
class SupportsTurnFields(Protocol):
    """An add-on contributing per-turn data to `GET /conversations/{id}`, under `turns[].addons[name]`.

    Stored snapshots are never rewritten, so the add-on's own table is joined to the transcript by
    `run_id` at read time. `transient_errors` (`stores.SupportsTransientErrors`), when declared,
    are mapped to the retryable 503.
    """

    @property
    def turn_model(self) -> type[BaseModel]:
        """One turn's entry as the wire shows it; what keeps the add-on's fields typed in OpenAPI."""
        ...

    async def load_turns(self, query: AddonTurnQuery, /) -> Mapping[str, BaseModel]:
        """Entries for the turns in `query.turns`, keyed by `run_id`, in one query; a turn missing from
        the result is sent as `null`. Raising fails the request, because `null` already means "no
        entry" and cannot double as "the loader failed"."""
        ...


@dataclass(frozen=True)
class AddonCaller:
    """The resolved request identity an add-on route works with: the user, and the namespace
    the feature's scope derives for them, the same one the conversation routes use."""

    user: HasUserId
    namespace_key: str


@dataclass(frozen=True)
class AddonRouteGuard:
    """What an add-on gets to build its routes, with the namespace wall of the conversation routes.

    An add-on's route enforces the same rule (a foreign or missing conversation is a 404, never a
    403) through `caller()`, `require_conversation` and `require_turn` rather than its own checks.
    `auth` is the app's own dependency object, so FastAPI runs the provider once per request.
    """

    registry: ConversationRegistry
    scope: tuple[ScopeField, ...]
    auth: UserDependency

    def caller(self) -> Callable[..., Awaitable[AddonCaller]]:
        """A FastAPI dependency: `caller: Annotated[AddonCaller, Depends(guard.caller())]`."""
        scope_params = scope_extras_model(self.scope) or create_model('ScopeParams')
        scope, auth = self.scope, self.auth

        async def resolve(
            user: Annotated[HasUserId, Depends(auth)], params: Annotated[Any, Depends(scope_params)]
        ) -> AddonCaller:
            return AddonCaller(user=user, namespace_key=build_namespace(scope, user, params.model_dump()))

        return resolve

    async def require_conversation(self, namespace_key: str, conversation_id: str) -> ConversationMeta:
        """The entry, or the 404 (foreign, missing, soft-deleted) / 503 the conversation routes raise."""
        try:
            meta = await self.registry.get(namespace_key, conversation_id)
        except RegistryUnavailable:
            _logger.warning('registry unavailable while resolving conversation %s', conversation_id, exc_info=True)
            raise unavailable_503() from None
        if meta is None:
            raise not_found_404()
        return meta

    async def require_turn(self, namespace_key: str, conversation_id: str, run_id: str) -> TurnRecord:
        """The turn's record, after `require_conversation`; a run that is not one of the
        conversation's turns is a 404 like a foreign conversation. Two registry reads per call, so
        a route needs no `get` of its own on top."""
        await self.require_conversation(namespace_key, conversation_id)
        try:
            turns = await self.registry.list_turns(namespace_key, conversation_id)
        except RegistryUnavailable:
            _logger.warning('registry unavailable while resolving turn %s', run_id, exc_info=True)
            raise unavailable_503() from None
        for turn in turns:
            if turn.run_id == run_id:
                return turn
        raise not_found_404()

    @staticmethod
    def not_found() -> HTTPException:
        """The package's 404, for an add-on's own not-found cases (same shape as the routes')."""
        return not_found_404()

    @staticmethod
    def unavailable() -> HTTPException:
        """The package's retryable 503, for an add-on's own temporary failures."""
        return unavailable_503()


@runtime_checkable
class SupportsRouters(Protocol):
    """An add-on that mounts routes of its own next to the conversation router, behind the same auth."""

    def routers(self, guard: AddonRouteGuard, /) -> Sequence[APIRouter]: ...  # pragma: no cover


@runtime_checkable
class SupportsDeletion(Protocol):
    """An add-on whose rows follow the conversation. Runs as one more `on_delete` hook, after the
    app's own, with the same event and the same failure rules (purge raises them together,
    soft delete logs them)."""

    async def on_delete(self, event: DeletionEvent) -> None: ...  # pragma: no cover


@runtime_checkable
class SupportsLifespan(Protocol):
    """An add-on with a store to open at startup and close at shutdown. Opened after the registry
    and closed before it, so an add-on may consult the registry while it is open."""

    async def open(self) -> None: ...  # pragma: no cover

    async def close(self) -> None: ...  # pragma: no cover
