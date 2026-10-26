"""ConversationsFeature: the app-factory composition unit."""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Annotated, Any

import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_app_factory import UserContext
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, RunRecord, StepPersistence

from pydantic_ai_conversations._capability import ConversationsCapability
from pydantic_ai_conversations._routes import DeletionEvent
from pydantic_ai_conversations.addons import AddonCaller, AddonRouteGuard, AddonTurnQuery
from pydantic_ai_conversations.feature import ConversationsFeature
from pydantic_ai_conversations.registry import InMemoryConversationRegistry
from pydantic_ai_conversations.scope import build_namespace, parse_scope


def _encode_ui(messages: Sequence[ModelMessage]) -> Sequence[Any]:
    """A `ui_messages` encoder; `config()` reports only that one is configured, never which."""
    return []


def _auth() -> UserContext:
    return UserContext(user_id='alice', tenant_id='t1')


def _feature(**kwargs: Any) -> ConversationsFeature:
    return ConversationsFeature(registry=InMemoryConversationRegistry(), step_store=InMemoryStepStore(), **kwargs)


async def _seed_run(
    store: InMemoryStepStore,
    *,
    run_id: str,
    conversation_id: str,
    at: datetime,
    parent_run_id: str | None = None,
) -> None:
    await store.register_run(
        RunRecord(run_id=run_id, conversation_id=conversation_id, parent_run_id=parent_run_id, started_at=at)
    )
    await store.save_snapshot(
        ContinuableSnapshot(run_id=run_id, step_index=1, messages=[], conversation_id=conversation_id)
    )


class DeletingStepStore(InMemoryStepStore):
    """A step store that opts into `SupportsRunDeletion`, actually removing the run's data --
    real deletion (not just recording) so `list_runs` after a purge tells the truth."""

    def __init__(self) -> None:
        super().__init__()
        self.deleted_run_ids: list[str] = []

    async def delete_runs(self, *, run_ids: Sequence[str]) -> None:
        self.deleted_run_ids.extend(run_ids)
        for run_id in run_ids:
            self._runs.pop(run_id, None)
            self._snapshots.pop(run_id, None)
            self._events.pop(run_id, None)


class PingingStepStore(InMemoryStepStore):
    """A step store that opts into `SupportsPing`."""

    async def ping(self) -> Mapping[str, Any]:
        return {'ok': True}


def _list_query_param_names(feature: ConversationsFeature) -> set[str]:
    """The `GET /conversations` query parameter names, read off the generated OpenAPI schema.

    This is the honest way to check a param was wired in: it is what the framework actually
    exposes to callers, not an implementation detail of how FastAPI's `Dependant` is shaped.
    """
    app = FastAPI()
    app.include_router(feature.routers(auth=_auth)[0])
    parameters = app.openapi()['paths']['/conversations']['get']['parameters']
    return {p['name'] for p in parameters}


def test_name_and_default_scope_declare_only_the_lineage_field() -> None:
    feature = _feature()
    assert feature.name == 'conversations'
    # The default scope is user-side only; the chat body still carries the turn a run builds on.
    assert feature.extras_type is not None and set(feature.extras_type.model_fields) == {'previous_run_id'}
    assert not feature.extras_type.model_fields['previous_run_id'].is_required()


def test_extras_scope_fields_generate_the_extras_slice() -> None:
    feature = _feature(scope=('user.tenant_id', 'extras.case_id'))
    assert feature.extras_type is not None
    fields = feature.extras_type.model_fields
    assert set(fields) == {'case_id', 'previous_run_id'}
    assert fields['case_id'].is_required()


def test_capabilities_bundle_ours_and_step_persistence() -> None:
    capabilities = _feature().capabilities()
    assert any(isinstance(c, ConversationsCapability) for c in capabilities)
    assert any(isinstance(c, StepPersistence) for c in capabilities)
    assert all(isinstance(c, AbstractCapability) for c in capabilities)


def test_routers_mount_the_conversation_router() -> None:
    routers = _feature().routers(auth=_auth)
    assert len(routers) == 1
    paths = {getattr(route, 'path', None) for route in routers[0].routes}
    assert '/conversations' in paths and '/conversations/{conversation_id}' in paths


async def test_lifespan_opens_and_closes_the_registry() -> None:
    opened: list[str] = []

    class TrackingRegistry(InMemoryConversationRegistry):
        async def open(self) -> None:
            opened.append('open')

        async def close(self) -> None:
            opened.append('close')

    feature = ConversationsFeature(registry=TrackingRegistry(), step_store=InMemoryStepStore())
    lifespan = feature.lifespan(FastAPI())
    assert lifespan is not None
    async with lifespan:
        assert opened == ['open']
    assert opened == ['open', 'close']


async def test_health_pings_the_registry() -> None:
    feature = _feature()
    assert await feature.health() == {'ok': True}


async def test_health_includes_a_pinging_step_store() -> None:
    feature = ConversationsFeature(registry=InMemoryConversationRegistry(), step_store=PingingStepStore())
    assert await feature.health() == {'ok': True, 'step_store': {'ok': True}}


async def test_health_warns_when_the_registry_claims_the_reserved_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ChattyRegistry(InMemoryConversationRegistry):
        async def ping(self) -> Mapping[str, Any]:
            return {'ok': True, 'step_store': 'postgres-eu1'}

    feature = ConversationsFeature(registry=ChattyRegistry(), step_store=PingingStepStore())
    with caplog.at_level('WARNING'):
        report = await feature.health()
    # The step store still wins the key (that is the contract), and the loss is logged.
    assert report['step_store'] == {'ok': True}
    assert "reserved key 'step_store'" in caplog.text


async def test_purge_without_deletion_support_keeps_snapshots() -> None:
    # Registry-only by contract when the store doesn't opt in: the id becomes unreachable through
    # this feature, but the substrate is untouched.
    registry = InMemoryConversationRegistry()
    step_store = InMemoryStepStore()
    feature = ConversationsFeature(registry=registry, step_store=step_store)
    await registry.register_conversation('[]', 'c1', created_by='alice')
    await _seed_run(step_store, run_id='r1', conversation_id='c1', at=datetime(2026, 8, 12, 10, tzinfo=UTC))
    await feature.purge(namespace_key='[]', conversation_id='c1')
    assert await registry.get('[]', 'c1') is None
    assert await step_store.list_runs(conversation_id='c1') != []


async def test_purge_deletes_runs_when_the_store_supports_deletion() -> None:
    registry = InMemoryConversationRegistry()
    step_store = DeletingStepStore()
    feature = ConversationsFeature(registry=registry, step_store=step_store)
    await registry.register_conversation('[]', 'c1', created_by='alice')
    await registry.register_conversation('[]', 'c2', created_by='alice')
    await _seed_run(step_store, run_id='r1', conversation_id='c1', at=datetime(2026, 8, 12, 10, tzinfo=UTC))
    await _seed_run(step_store, run_id='r2', conversation_id='c1', at=datetime(2026, 8, 12, 11, tzinfo=UTC))
    await _seed_run(step_store, run_id='other', conversation_id='c2', at=datetime(2026, 8, 12, 12, tzinfo=UTC))

    await feature.purge(namespace_key='[]', conversation_id='c1')

    assert sorted(step_store.deleted_run_ids) == ['r1', 'r2']
    assert await step_store.list_runs(conversation_id='c1') == []
    assert await step_store.list_runs(conversation_id='c2') != []  # untouched
    assert await registry.get('[]', 'c1') is None


async def test_purge_takes_delegate_runs_transitively() -> None:
    # A delegate spawned inside a tool call without an explicit conversation_id gets a FRESH
    # conversation id of its own, with parent_run_id pointing at the run that spawned it -- so
    # list_runs(conversation_id='c1') alone never finds it. Its snapshots still carry content
    # forwarded from this conversation, so the purge must walk parent_run_id transitively:
    # r1 (the conversation's own top-level run) -> delegate (child) -> grandchild (its own child).
    registry = InMemoryConversationRegistry()
    step_store = DeletingStepStore()
    feature = ConversationsFeature(registry=registry, step_store=step_store)
    await registry.register_conversation('[]', 'c1', created_by='alice')
    await _seed_run(step_store, run_id='r1', conversation_id='c1', at=datetime(2026, 8, 12, 10, tzinfo=UTC))
    await _seed_run(
        step_store,
        run_id='delegate',
        conversation_id='delegate-conv',  # fresh, unrelated conversation id
        at=datetime(2026, 8, 12, 11, tzinfo=UTC),
        parent_run_id='r1',
    )
    await _seed_run(
        step_store,
        run_id='grandchild',
        conversation_id='grandchild-conv',  # another fresh conversation id, one level deeper
        at=datetime(2026, 8, 12, 12, tzinfo=UTC),
        parent_run_id='delegate',
    )
    # A same-conversation child, the other delegate shape: already present in the
    # initial list_runs(conversation_id='c1') pass, and reached again via r1's parent_run_id walk
    # -- exercises the closure's already-seen dedup, not just the new-id path.
    await _seed_run(
        step_store,
        run_id='same-conv-child',
        conversation_id='c1',
        at=datetime(2026, 8, 12, 10, 30, tzinfo=UTC),
        parent_run_id='r1',
    )
    await _seed_run(
        step_store,
        run_id='unrelated',
        conversation_id='other-conv',
        at=datetime(2026, 8, 12, 13, tzinfo=UTC),
    )

    await feature.purge(namespace_key='[]', conversation_id='c1')

    assert sorted(step_store.deleted_run_ids) == ['delegate', 'grandchild', 'r1', 'same-conv-child']
    assert await step_store.list_runs(conversation_id='c1') == []
    assert await step_store.list_runs(conversation_id='delegate-conv') == []
    assert await step_store.list_runs(conversation_id='grandchild-conv') == []
    assert await step_store.list_runs(conversation_id='other-conv') != []  # untouched


async def test_purge_retries_after_a_transient_substrate_failure() -> None:
    class FlakyStepStore(DeletingStepStore):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def delete_runs(self, *, run_ids: Sequence[str]) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError('transient substrate failure')
            await super().delete_runs(run_ids=run_ids)

    registry = InMemoryConversationRegistry()
    step_store = FlakyStepStore()
    feature = ConversationsFeature(registry=registry, step_store=step_store)
    await registry.register_conversation('[]', 'c1', created_by='alice')
    await _seed_run(step_store, run_id='r1', conversation_id='c1', at=datetime(2026, 8, 12, 10, tzinfo=UTC))

    with pytest.raises(RuntimeError, match='transient substrate failure'):
        await feature.purge(namespace_key='[]', conversation_id='c1')

    # The registry purge already landed -- and holds -- regardless of the substrate failure.
    assert await registry.get('[]', 'c1') is None
    assert step_store.deleted_run_ids == []

    # Retry: registry.purge is idempotent for an id this namespace already owns (still True), and
    # the substrate delete succeeds this time.
    await feature.purge(namespace_key='[]', conversation_id='c1')
    assert step_store.deleted_run_ids == ['r1']
    assert await step_store.list_runs(conversation_id='c1') == []


async def test_purge_with_deletion_support_and_no_runs_skips_the_delete_call() -> None:
    # Owned conversation, deletion-capable store, but no runs were ever registered (e.g. purged
    # right after creation): `delete_runs` must not be called with an empty list.
    registry = InMemoryConversationRegistry()
    step_store = DeletingStepStore()
    feature = ConversationsFeature(registry=registry, step_store=step_store)
    await registry.register_conversation('[]', 'c1', created_by='alice')

    await feature.purge(namespace_key='[]', conversation_id='c1')

    assert step_store.deleted_run_ids == []
    assert await registry.get('[]', 'c1') is None


async def test_foreign_purge_deletes_nothing_from_the_substrate() -> None:
    registry = InMemoryConversationRegistry()
    step_store = DeletingStepStore()
    feature = ConversationsFeature(registry=registry, step_store=step_store)
    owner_namespace = '[["user.tenant_id","t1"]]'
    foreign_namespace = '[["user.tenant_id","t2"]]'
    await registry.register_conversation(owner_namespace, 'c1', created_by='alice')
    await _seed_run(step_store, run_id='r1', conversation_id='c1', at=datetime(2026, 8, 12, 10, tzinfo=UTC))

    await feature.purge(namespace_key=foreign_namespace, conversation_id='c1')

    assert step_store.deleted_run_ids == []
    assert await step_store.list_runs(conversation_id='c1') != []
    assert await registry.get(owner_namespace, 'c1') is not None  # the owner's entry survives


async def test_purge_runs_the_cleanup_hooks() -> None:
    events: list[DeletionEvent] = []

    async def hook(event: DeletionEvent) -> None:
        events.append(event)

    registry = InMemoryConversationRegistry()
    step_store = DeletingStepStore()
    feature = ConversationsFeature(registry=registry, step_store=step_store, on_delete=(hook,))
    await registry.register_conversation('[]', 'c1', created_by='alice')
    await _seed_run(step_store, run_id='r1', conversation_id='c1', at=datetime(2026, 8, 12, 10, tzinfo=UTC))

    await feature.purge(namespace_key='[]', conversation_id='c1')

    assert len(events) == 1
    event = events[0]
    assert event.kind == 'purge'
    assert event.namespace_key == '[]'
    assert event.conversation_id == 'c1'
    assert event.meta is not None


async def test_purge_hook_sees_no_meta_for_an_already_soft_deleted_entry() -> None:
    events: list[DeletionEvent] = []

    async def hook(event: DeletionEvent) -> None:
        events.append(event)

    registry = InMemoryConversationRegistry()
    feature = ConversationsFeature(registry=registry, step_store=InMemoryStepStore(), on_delete=(hook,))
    await registry.register_conversation('[]', 'c1', created_by='alice')
    await registry.soft_delete('[]', 'c1')

    await feature.purge(namespace_key='[]', conversation_id='c1')

    assert len(events) == 1
    assert events[0].kind == 'purge'
    assert events[0].meta is None


async def test_foreign_purge_runs_no_hooks() -> None:
    events: list[DeletionEvent] = []

    async def hook(event: DeletionEvent) -> None:
        events.append(event)

    registry = InMemoryConversationRegistry()
    feature = ConversationsFeature(registry=registry, step_store=InMemoryStepStore(), on_delete=(hook,))
    owner_namespace = '[["user.tenant_id","t1"]]'
    foreign_namespace = '[["user.tenant_id","t2"]]'
    await registry.register_conversation(owner_namespace, 'c1', created_by='alice')

    await feature.purge(namespace_key=foreign_namespace, conversation_id='c1')

    assert events == []
    assert await registry.get(owner_namespace, 'c1') is not None  # the owner's entry survives


async def test_failing_purge_hook_runs_the_rest_then_raises() -> None:
    calls: list[str] = []

    async def bad_hook(event: DeletionEvent) -> None:
        calls.append('bad')
        raise RuntimeError('cleanup exploded')

    async def good_hook(event: DeletionEvent) -> None:
        calls.append('good')

    registry = InMemoryConversationRegistry()
    feature = ConversationsFeature(registry=registry, step_store=InMemoryStepStore(), on_delete=(bad_hook, good_hook))
    await registry.register_conversation('[]', 'c1', created_by='alice')

    with pytest.raises(ExceptionGroup) as raised:
        await feature.purge(namespace_key='[]', conversation_id='c1')

    assert calls == ['bad', 'good']
    assert [str(exc) for exc in raised.value.exceptions] == ['cleanup exploded']
    assert await registry.get('[]', 'c1') is None


async def test_purge_retried_after_a_hook_failure_reruns_the_hooks() -> None:
    events: list[DeletionEvent] = []
    failures = [RuntimeError('cleanup exploded')]

    async def flaky_hook(event: DeletionEvent) -> None:
        events.append(event)
        if failures:
            raise failures.pop()

    registry = InMemoryConversationRegistry()
    step_store = DeletingStepStore()
    feature = ConversationsFeature(registry=registry, step_store=step_store, on_delete=(flaky_hook,))
    await registry.register_conversation('[]', 'c1', created_by='alice')
    await _seed_run(step_store, run_id='r1', conversation_id='c1', at=datetime(2026, 8, 12, 10, tzinfo=UTC))

    with pytest.raises(ExceptionGroup):
        await feature.purge(namespace_key='[]', conversation_id='c1')
    await feature.purge(namespace_key='[]', conversation_id='c1')

    assert [event.kind for event in events] == ['purge', 'purge']
    assert events[1].meta is None  # the entry is gone by the retry


def test_namespace_for_matches_build_namespace() -> None:
    feature = _feature(scope=('user.tenant_id', 'extras.case_id'))
    user = UserContext(user_id='u', tenant_id='t')
    extras = {'case_id': 'c1'}
    assert feature.namespace_for(user, extras) == build_namespace(
        parse_scope(('user.tenant_id', 'extras.case_id')), user, extras
    )


def test_namespace_for_accepts_the_extras_of_a_first_turn() -> None:
    """The app factory hands the capability the validated chat body, whose `previous_run_id` is the
    internal sentinel on a first turn; `namespace_for(deps.user, deps.extras)` is the documented way
    to read the registry in a test, so it must take that mapping as it is."""
    feature = _feature(scope=('user.tenant_id', 'extras.case_id'))
    user = UserContext(user_id='u', tenant_id='t')
    assert feature.extras_type is not None
    first_turn_extras = feature.extras_type.model_validate({'case_id': 'c1'}).model_dump()
    assert feature.namespace_for(user, first_turn_extras) == build_namespace(
        parse_scope(('user.tenant_id', 'extras.case_id')), user, {'case_id': 'c1'}
    )


def test_namespace_for_defaults_extras_to_empty() -> None:
    feature = _feature()
    user = UserContext(user_id='u', tenant_id='t')
    assert feature.namespace_for(user) == build_namespace(parse_scope(('user.tenant_id', 'user.user_id')), user, {})


def test_namespace_for_rejects_a_missing_extras_scope_field() -> None:
    # Without validation this keys an empty partition nothing lives in, and the purge that follows
    # is a silent no-op indistinguishable from a foreign one.
    feature = _feature(scope=('user.tenant_id', 'extras.case_id'))
    with pytest.raises(ValidationError):
        feature.namespace_for(UserContext(user_id='u', tenant_id='t'))


def test_namespace_for_rejects_an_empty_extras_claim() -> None:
    feature = _feature(scope=('user.tenant_id', 'extras.case_id'))
    with pytest.raises(ValidationError):
        feature.namespace_for(UserContext(user_id='u', tenant_id='t'), {'case_id': ''})


def test_optional_route_config_reaches_the_router() -> None:
    routes = _feature(enable_create_route=True, enable_title_route=True).routers(auth=_auth)[0].routes
    mounted = {(method, getattr(route, 'path', None)) for route in routes for method in getattr(route, 'methods', ())}
    assert ('POST', '/conversations') in mounted
    assert ('PUT', '/conversations/{conversation_id}/title') in mounted


def test_scope_reaches_the_router_and_the_capability() -> None:
    feature = _feature(scope=('extras.case_id',))
    capability = feature.capabilities()[0]
    assert isinstance(capability, ConversationsCapability)
    assert capability.scope == parse_scope(('extras.case_id',))
    # Non-drift: the router's required query param and the extras slice come from one recipe;
    # the lineage field is chat-body only, no route takes it.
    assert feature.extras_type is not None and set(feature.extras_type.model_fields) == {'case_id', 'previous_run_id'}
    assert 'case_id' in _list_query_param_names(feature)
    assert 'previous_run_id' not in _list_query_param_names(feature)


def test_max_history_runs_defaults_to_none_and_reaches_the_capability() -> None:
    capability = _feature().capabilities()[0]
    assert isinstance(capability, ConversationsCapability)
    assert capability.max_history_runs is None


def test_max_history_runs_reaches_the_capability() -> None:
    capability = _feature(max_history_runs=3).capabilities()[0]
    assert isinstance(capability, ConversationsCapability)
    assert capability.max_history_runs == 3


def test_bad_max_history_runs_value_raises_when_capabilities_are_built() -> None:
    """The feature does not re-validate `max_history_runs` itself in `__init__` -- it passes
    straight through and `ConversationsCapability.__post_init__` is what raises, so the failure
    surfaces at `capabilities()` (typically still app startup, since `create_agent_app` calls it
    eagerly), not at `ConversationsFeature(...)` itself.
    """
    feature = _feature(max_history_runs=0)
    with pytest.raises(ValueError, match='max_history_runs must be positive'):
        feature.capabilities()


def test_step_persistence_gets_no_agent_name() -> None:
    # `agent_name` would make StepPersistence derive its own run id instead of adopting
    # `ctx.run_id`, which every id this feature keeps is keyed on (turn records, purge's run
    # walk). The feature has no such parameter.
    capability = _feature().capabilities()[1]
    assert isinstance(capability, StepPersistence)
    assert capability.agent_name is None


def test_conversations_capability_precedes_step_persistence_with_frontier_saves() -> None:
    """The capability's `before_model_request` stamps the run's prompt before `StepPersistence`'s
    frontier save of the first request, so the order is capability first, and the frontier save
    is what keeps the prompt of a run that fails before answering."""
    capabilities = _feature().capabilities()
    assert isinstance(capabilities[0], ConversationsCapability)
    assert isinstance(capabilities[1], StepPersistence)
    assert capabilities[1].capture_frontier is True


def test_filterable_attrs_reach_the_router() -> None:
    feature = _feature(filterable_attrs=('machine_id',))
    assert 'machine_id' in _list_query_param_names(feature)


async def test_on_delete_reaches_the_router() -> None:
    seen: list[str] = []

    async def hook(event: DeletionEvent) -> None:
        seen.append(event.conversation_id)

    registry = InMemoryConversationRegistry()
    feature = ConversationsFeature(registry=registry, step_store=InMemoryStepStore(), on_delete=(hook,))
    namespace = '[["user.tenant_id","t1"],["user.user_id","alice"]]'
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    app = FastAPI()
    app.include_router(feature.routers(auth=_auth)[0])
    assert TestClient(app).delete('/conversations/c1').status_code == 204
    assert seen == ['c1']


async def test_ui_messages_reaches_the_router() -> None:
    """The encoder given to the feature is what renders each turn's `ui_messages`."""
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    feature = ConversationsFeature(
        registry=registry,
        step_store=store,
        ui_messages=lambda messages: [{'seen': len(messages)}],
    )
    namespace = '[["user.tenant_id","t1"],["user.user_id","alice"]]'
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r1',
            step_index=1,
            messages=[ModelRequest(parts=[UserPromptPart(content='hi')]), ModelResponse(parts=[TextPart('hello')])],
            conversation_id='c1',
        )
    )
    app = FastAPI()
    app.include_router(feature.routers(auth=_auth)[0])

    body = TestClient(app).get('/conversations/c1').json()

    assert body['turns'][0]['ui_messages'] == [{'seen': 2}]


def test_config_reports_the_defaults() -> None:
    assert _feature().config() == {
        'create_route': False,
        'title_route': False,
        'ui_messages': False,
        'filterable_attrs': [],
        'addons': {},
    }


def test_config_reports_a_fully_configured_feature() -> None:
    feature = _feature(
        enable_create_route=True,
        enable_title_route=True,
        ui_messages=_encode_ui,
        filterable_attrs=('case_id', 'status'),
    )
    assert feature.config() == {
        'create_route': True,
        'title_route': True,
        'ui_messages': True,
        'filterable_attrs': ['case_id', 'status'],
        'addons': {},
    }


def test_config_is_json_serializable() -> None:
    # It goes out over `/config/features`; a callable or a `Path` in here would 500 the endpoint.
    config = _feature(ui_messages=_encode_ui, filterable_attrs=('case_id',)).config()
    assert config['ui_messages'] is True  # the encoder is reported as presence, never as itself
    assert json.loads(json.dumps(config)) == config


# --- turn add-ons ---------------------------------------------------------------------------------


class _MarkOut(BaseModel):
    seen: bool


class _MarkAddon:
    name = 'mark'
    turn_model = _MarkOut

    async def load_turns(self, query: AddonTurnQuery) -> Mapping[str, BaseModel]:
        return {t.run_id: _MarkOut(seen=True) for t in query.turns}


def test_config_lists_the_configured_addons_by_name() -> None:
    feature = ConversationsFeature(
        registry=InMemoryConversationRegistry(), step_store=InMemoryStepStore(), addons=[_MarkAddon()]
    )
    assert feature.config()['addons'] == {'mark': {}}


def test_routers_pass_addons_to_the_detail_route() -> None:
    feature = ConversationsFeature(
        registry=InMemoryConversationRegistry(), step_store=InMemoryStepStore(), addons=[_MarkAddon()]
    )
    app = FastAPI()
    for router in feature.routers(auth=_auth):
        app.include_router(router)
    components = TestClient(app).get('/openapi.json').json()['components']['schemas']
    assert 'mark' in components['TurnAddonsOut']['properties']


class _RoutedAddon:
    """Brings a route of its own, built on the context's helpers; no turn fields."""

    name = 'mark_routes'

    def routers(self, guard: AddonRouteGuard) -> list[APIRouter]:
        router = APIRouter(prefix='/conversations')

        @router.get('/{conversation_id}/turns/{run_id}/mark')
        async def mark(
            conversation_id: str, run_id: str, caller: Annotated[AddonCaller, Depends(guard.caller())]
        ) -> dict[str, str]:
            turn = await guard.require_turn(caller.namespace_key, conversation_id, run_id)
            return {'run_id': turn.run_id, 'by': caller.user.user_id}

        return [router]


async def test_routers_mount_the_add_ons_routes_behind_the_same_auth() -> None:
    registry = InMemoryConversationRegistry()
    feature = ConversationsFeature(registry=registry, step_store=InMemoryStepStore(), addons=[_RoutedAddon()])
    namespace = feature.namespace_for(UserContext(user_id='alice', tenant_id='t1'))
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    await registry.start_turn(namespace, 'c1', run_id='r1', author_user_id='alice')
    app = FastAPI()
    for router in feature.routers(auth=_auth):
        app.include_router(router)
    client = TestClient(app)

    assert client.get('/conversations/c1/turns/r1/mark').json() == {'run_id': 'r1', 'by': 'alice'}
    assert client.get('/conversations/c1/turns/r9/mark').status_code == 404
    assert feature.config()['addons'] == {'mark_routes': {}}


def test_an_add_on_without_routers_mounts_nothing_extra() -> None:
    feature = ConversationsFeature(
        registry=InMemoryConversationRegistry(), step_store=InMemoryStepStore(), addons=[_MarkAddon()]
    )
    assert len(feature.routers(auth=_auth)) == 1


async def test_add_on_routes_take_the_scope_query_parameters() -> None:
    """`guard.caller()` derives the namespace from the same scope as the conversation routes, so an
    `extras.*` field is a required query parameter on an add-on route too."""
    registry = InMemoryConversationRegistry()
    feature = ConversationsFeature(
        registry=registry,
        step_store=InMemoryStepStore(),
        scope=('user.tenant_id', 'extras.case_id'),
        addons=[_RoutedAddon()],
    )
    namespace = feature.namespace_for(UserContext(user_id='alice', tenant_id='t1'), {'case_id': 'k1'})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    await registry.start_turn(namespace, 'c1', run_id='r1', author_user_id='alice')
    app = FastAPI()
    for router in feature.routers(auth=_auth):
        app.include_router(router)
    client = TestClient(app)

    assert client.get('/conversations/c1/turns/r1/mark').status_code == 422
    assert client.get('/conversations/c1/turns/r1/mark', params={'case_id': 'k1'}).status_code == 200
    assert client.get('/conversations/c1/turns/r1/mark', params={'case_id': 'k2'}).status_code == 404
