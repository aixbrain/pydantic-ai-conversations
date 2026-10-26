"""The app-factory composition unit: registry + capability + StepPersistence + routes.

The extras schema and the route query parameters are both generated from the same scope fields,
so the chat path and the management routes cannot drift.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from types import MappingProxyType
from typing import Any

from fastapi import APIRouter, FastAPI
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_app_factory import AppDeps, AuthProvider, Feature
from pydantic_ai_harness.step_persistence import StepPersistence, StepStore

from pydantic_ai_conversations._capability import ConversationsCapability, FailureDescriber
from pydantic_ai_conversations._lineage import extras_model
from pydantic_ai_conversations._loader import TranscriptView
from pydantic_ai_conversations._routes import (
    DeletionEvent,
    OnDeleteHook,
    UIMessageEncoder,
    build_conversation_router,
)
from pydantic_ai_conversations.addons import AddonRouteGuard, ConversationAddon, SupportsRouters
from pydantic_ai_conversations.registry import ConversationRegistry
from pydantic_ai_conversations.scope import (
    HasUserId,
    build_namespace,
    parse_scope,
    scope_extras_model,
)
from pydantic_ai_conversations.stores import (
    NewMessagesPreparer,
    PreparingStepStore,
    SnapshotPreparer,
    SupportsPing,
    SupportsRunDeletion,
)

_DEFAULT_SCOPE = ('user.tenant_id', 'user.user_id')
_EMPTY: Mapping[str, Any] = MappingProxyType({})
_logger = logging.getLogger(__name__)


class ConversationsFeature(Feature):
    """Conversation persistence over a StepStore, with a namespace-scoped registry.

    The routes take the app's auth provider from `routers(auth=...)`, so the namespace and the
    access decision derive from the one identity the app guards with. `attrs` are create-time-only;
    the implicit create path sets them by pre-registering the entry before the first run
    (docs/http-api.md). Add-ons passed as `addons` contribute per-turn data to the detail read
    and routes of their own; see `ConversationAddon` and the `Supports*` protocols next to it.
    """

    name = 'conversations'

    def __init__(
        self,
        *,
        registry: ConversationRegistry,
        step_store: StepStore,
        scope: Iterable[str] = _DEFAULT_SCOPE,
        enable_create_route: bool = False,
        enable_title_route: bool = False,
        filterable_attrs: Sequence[str] = (),
        on_delete: Sequence[OnDeleteHook] = (),
        snapshot_preparers: Sequence[SnapshotPreparer] = (),
        transcript_view: TranscriptView | None = None,
        ui_messages: UIMessageEncoder | None = None,
        max_history_runs: int | None = None,
        describe_failure: FailureDescriber | None = None,
        addons: Sequence[ConversationAddon] = (),
    ) -> None:
        self._registry = registry
        self._step_store = step_store
        self._scope = parse_scope(scope)
        self._enable_create_route = enable_create_route
        self._enable_title_route = enable_title_route
        self._filterable_attrs = tuple(filterable_attrs)
        self._on_delete = tuple(on_delete)
        self._snapshot_preparers = tuple(snapshot_preparers)
        self._transcript_view = transcript_view
        self._ui_messages = ui_messages
        self._max_history_runs = max_history_runs
        self._describe_failure = describe_failure
        self._addons = tuple(addons)
        # One recipe for the chat body AND the route query params: the scope fields cannot drift.
        # The chat body additionally carries `previous_run_id`, which no route takes.
        self._scope_extras_type = scope_extras_model(self._scope)
        self.extras_type = extras_model(self._scope_extras_type)

    def capabilities(self) -> list[AbstractCapability[AppDeps]]:
        """`ConversationsCapability` first, `StepPersistence` second.

        The harness saves the request about to be sent in its own `before_model_request`, so the
        capability's hook must run first to stamp a prompt that arrived inside `message_history`;
        that save is what keeps a prompt whose run fails before answering.
        """
        # Run state carries the conversation; the seam keeps each stored snapshot to the run's
        # own turn.
        preparers: tuple[SnapshotPreparer, ...] = (NewMessagesPreparer(), *self._snapshot_preparers)
        # Routes and purge keep the raw store: they never write snapshots.
        write_store = PreparingStepStore(self._step_store, preparers)
        return [
            ConversationsCapability[AppDeps](
                registry=self._registry,
                step_store=write_store,
                scope=self._scope,
                max_history_runs=self._max_history_runs,
                describe_failure=self._describe_failure,
            ),
            # No `agent_name` here: it would make StepPersistence derive its own run id instead
            # of adopting `ctx.run_id`, desynchronizing turn records and purge's run walk. Helper
            # agents outside this feature may still use it.
            StepPersistence(store=write_store, capture_frontier=True),
        ]

    def routers(self, *, auth: AuthProvider) -> list[APIRouter]:
        routers = [
            build_conversation_router(
                registry=self._registry,
                step_store=self._step_store,
                scope=self._scope,
                auth=auth,
                enable_create=self._enable_create_route,
                enable_title=self._enable_title_route,
                filterable_attrs=self._filterable_attrs,
                on_delete=self._on_delete,
                transcript_view=self._transcript_view,
                ui_messages=self._ui_messages,
                addons=self._addons,
            )
        ]
        guard = AddonRouteGuard(registry=self._registry, scope=self._scope, auth=auth)
        for addon in self._addons:
            if isinstance(addon, SupportsRouters):
                routers.extend(addon.routers(guard))
        return routers

    def lifespan(self, app: FastAPI) -> AbstractAsyncContextManager[None]:
        @asynccontextmanager
        async def _registry_lifespan() -> AsyncGenerator[None]:
            await self._registry.open()
            try:
                yield
            finally:
                await self._registry.close()

        return _registry_lifespan()

    async def health(self) -> Mapping[str, Any]:
        report = dict(await self._registry.ping())
        if isinstance(self._step_store, SupportsPing):
            if 'step_store' in report:
                _logger.warning(
                    "registry ping() returned the reserved key 'step_store'; "
                    'overwriting it with the step store report (see ConversationRegistry.ping)'
                )
            report['step_store'] = dict(await self._step_store.ping())
        return report

    def config(self) -> Mapping[str, Any]:
        """The switches a client branches on; see docs/operations.md for what is deliberately absent."""
        return {
            'create_route': self._enable_create_route,
            # A client cannot offer "rename" it is not allowed to perform, and it should not have
            # to probe a PUT to find out.
            'title_route': self._enable_title_route,
            # Presence, not identity: the feature holds an opaque callable and does not know which
            # UI protocol it encodes. A client only needs to know whether `ui_messages` will be null.
            'ui_messages': self._ui_messages is not None,
            'filterable_attrs': list(self._filterable_attrs),
            # Every add-on by name, with room under each for the add-on's own switches.
            'addons': {addon.name: {} for addon in self._addons},
        }

    def namespace_for(self, user: HasUserId, extras: Mapping[str, Any] = _EMPTY) -> str:
        """The namespace key this feature's scope derives for a caller; the companion to `purge()`
        and the pre-create pattern.

        `extras` is validated against the scope's `extras.*` fields -- the same schema the chat
        body and the routes use for them -- so an omitted scope field raises `ValidationError`
        instead of silently keying an empty partition where nothing lives. Other keys pass, so a
        run's `deps.extras` can be handed over as it is; its `previous_run_id` is an internal
        sentinel on a first turn.
        """
        if self._scope_extras_type is not None:
            self._scope_extras_type.model_validate(dict(extras))
        return build_namespace(self._scope, user, extras)

    async def purge(self, *, namespace_key: str, conversation_id: str) -> None:
        """Hard-delete the conversation: registry entry, turn records, and -- when the step store
        opts in via `SupportsRunDeletion` -- the runs and snapshots in the substrate.

        The registry purge runs first, so the conversation is unreachable even if the substrate
        delete fails midway; retrying re-derives the run closure and tries again. A foreign purge
        is a silent no-op and runs no hooks. `on_delete` hooks run with `kind='purge'`; a raising
        hook does not stop the others, and their failures are raised together as one
        `ExceptionGroup` afterwards, so a caller erasing personal data learns the erasure is
        incomplete and retries. Hooks must be idempotent, and `meta` is `None` when the entry was
        already soft-deleted or purged.
        """
        meta = await self._registry.get(namespace_key, conversation_id)
        owned = await self._registry.purge(namespace_key, conversation_id)
        if not owned:
            return
        if isinstance(self._step_store, SupportsRunDeletion):
            run_ids = await self._delegate_closure(conversation_id)
            if run_ids:
                await self._step_store.delete_runs(run_ids=run_ids)
        event = DeletionEvent(namespace_key=namespace_key, conversation_id=conversation_id, kind='purge', meta=meta)
        failures: list[Exception] = []
        for hook in self._on_delete:
            try:
                await hook(event)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ExceptionGroup(f'on_delete hooks failed while purging conversation {conversation_id}', failures)

    async def _delegate_closure(self, conversation_id: str) -> list[str]:
        """This conversation's own run ids, plus every run spawned from them transitively via
        `parent_run_id`, since delegates carry content forwarded from this conversation.

        The harness sets `parent_run_id` on a run started inside another run, which is what makes
        delegates findable here. A nested agent with its own `ConversationsCapability` looks the
        same and is deleted with it, which docs/deleting.md lists as unsupported. Each run id is
        expanded once, so the walk ends on cyclic store data.
        """
        seen = {run.run_id for run in await self._step_store.list_runs(conversation_id=conversation_id)}
        frontier = list(seen)
        while frontier:
            next_frontier: list[str] = []
            for parent_id in frontier:
                for child in await self._step_store.list_runs(parent_run_id=parent_id):
                    if child.run_id not in seen:
                        seen.add(child.run_id)
                        next_frontier.append(child.run_id)
            frontier = next_frontier
        return list(seen)
