"""The feedback add-on end to end: typed payload, own routes, inline entries, cleanup."""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_app_factory import UserContext
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, RunRecord

from pydantic_ai_conversations import AddonTurnQuery, ConversationsFeature, TranscriptTurn
from pydantic_ai_conversations.feedback import InMemoryTurnFeedbackStore, TurnFeedbackAddon, TurnFeedbackStore
from pydantic_ai_conversations.registry import InMemoryConversationRegistry

ALICE = UserContext(user_id='alice', tenant_id='t1')
BOB = UserContext(user_id='bob', tenant_id='t1')


class Report(BaseModel):
    category: str
    remarks: str = ''


def _auth_as(user: UserContext) -> Callable[[], UserContext]:
    def auth() -> UserContext:
        return user

    return auth


async def _app(
    *,
    user: UserContext = ALICE,
    store: TurnFeedbackStore | None = None,
    name: str = 'report',
    scope: tuple[str, ...] = ('user.tenant_id', 'user.user_id'),
) -> tuple[TestClient, ConversationsFeature, InMemoryConversationRegistry, TurnFeedbackAddon]:
    registry, step_store = InMemoryConversationRegistry(), InMemoryStepStore()
    addon = TurnFeedbackAddon(payload_model=Report, store=store or InMemoryTurnFeedbackStore(), name=name)
    feature = ConversationsFeature(registry=registry, step_store=step_store, scope=scope, addons=[addon])
    namespace = feature.namespace_for(ALICE)
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    await registry.start_turn(namespace, 'c1', run_id='r1', author_user_id='alice')
    await step_store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await step_store.save_snapshot(
        ContinuableSnapshot(
            run_id='r1',
            step_index=1,
            messages=[
                ModelRequest(parts=[UserPromptPart(content='hi')]),
                ModelResponse(parts=[TextPart(content='hello')]),
            ],
            conversation_id='c1',
        )
    )
    app = FastAPI()
    for router in feature.routers(auth=_auth_as(user)):
        app.include_router(router)
    return TestClient(app, raise_server_exceptions=False), feature, registry, addon


async def test_put_stores_the_payload_and_the_detail_shows_it_inline() -> None:
    client, _, _, _ = await _app()

    put = client.put('/conversations/c1/turns/r1/report', json={'category': 'wrong'})

    assert put.status_code == 200
    assert put.json()['data'] == {'category': 'wrong', 'remarks': ''}
    assert put.json()['author_user_id'] == 'alice'
    turn = client.get('/conversations/c1').json()['turns'][0]
    assert turn['addons']['report']['data'] == {'category': 'wrong', 'remarks': ''}


async def test_put_replaces_the_callers_own_entry() -> None:
    client, _, _, _ = await _app()
    client.put('/conversations/c1/turns/r1/report', json={'category': 'wrong'})

    client.put('/conversations/c1/turns/r1/report', json={'category': 'rude', 'remarks': 'tone'})

    entries = client.get('/conversations/c1/report').json()
    assert [e['data'] for e in entries] == [{'category': 'rude', 'remarks': 'tone'}]


async def test_payload_is_validated_against_the_apps_model() -> None:
    client, _, _, _ = await _app()
    assert client.put('/conversations/c1/turns/r1/report', json={'remarks': 'no category'}).status_code == 422


async def test_openapi_types_the_body_and_the_entry() -> None:
    client, _, _, _ = await _app()
    schema = client.get('/openapi.json').json()
    put = schema['paths']['/conversations/{conversation_id}/turns/{run_id}/report']['put']
    assert put['requestBody']['content']['application/json']['schema'] == {'$ref': '#/components/schemas/Report'}
    components = schema['components']['schemas']
    assert components['ReportEntry']['properties']['data'] == {'$ref': '#/components/schemas/Report'}
    assert components['TurnAddonsOut']['properties']['report']['anyOf'][0] == {
        '$ref': '#/components/schemas/ReportEntry'
    }


async def test_detail_shows_only_the_callers_entry() -> None:
    """On a conversation shared by a tenant-wide scope, the detail shows the caller's own entry."""
    store = InMemoryTurnFeedbackStore()
    alice_client, _, _, _ = await _app(store=store, scope=('user.tenant_id',))
    alice_client.put('/conversations/c1/turns/r1/report', json={'category': 'wrong'})

    bob_client, _, _, _ = await _app(user=BOB, store=store, scope=('user.tenant_id',))

    assert bob_client.get('/conversations/c1').json()['turns'][0]['addons']['report'] is None
    assert len(bob_client.get('/conversations/c1/report').json()) == 1


async def test_unknown_run_or_foreign_conversation_is_404() -> None:
    client, _, _, _ = await _app()
    assert client.put('/conversations/c1/turns/nope/report', json={'category': 'x'}).status_code == 404
    assert client.put('/conversations/c9/turns/r1/report', json={'category': 'x'}).status_code == 404
    assert client.get('/conversations/c9/report').status_code == 404


async def test_delete_is_idempotent() -> None:
    client, _, _, _ = await _app()
    client.put('/conversations/c1/turns/r1/report', json={'category': 'wrong'})

    assert client.delete('/conversations/c1/turns/r1/report').status_code == 204
    assert client.delete('/conversations/c1/turns/r1/report').status_code == 204
    assert client.get('/conversations/c1').json()['turns'][0]['addons']['report'] is None


async def test_purge_removes_the_entries_and_soft_delete_keeps_them() -> None:
    store = InMemoryTurnFeedbackStore()
    client, feature, _, _ = await _app(store=store)
    client.put('/conversations/c1/turns/r1/report', json={'category': 'wrong'})
    namespace = feature.namespace_for(ALICE)

    assert client.delete('/conversations/c1').status_code == 204
    assert len(await store.list(kind='report', namespace_key=namespace, conversation_id='c1')) == 1

    await feature.purge(namespace_key=namespace, conversation_id='c1')
    assert await store.list(kind='report', namespace_key=namespace, conversation_id='c1') == []


class _Down(InMemoryTurnFeedbackStore):
    """A store whose declared transient failure fires on every operation."""

    transient_errors = (ConnectionError,)

    async def upsert(self, **kwargs: Any) -> Any:
        raise ConnectionError('down')

    async def by_run_ids(self, **kwargs: Any) -> Any:
        raise ConnectionError('down')


async def test_a_transient_store_failure_is_503_on_the_routes_and_the_detail_read() -> None:
    client, _, _, _ = await _app(store=_Down())

    put = client.put('/conversations/c1/turns/r1/report', json={'category': 'x'})
    detail = client.get('/conversations/c1')

    assert put.status_code == 503 and put.json()['detail']['code'] == 'conversations/unavailable'
    assert detail.status_code == 503 and detail.json()['detail']['code'] == 'conversations/unavailable'


async def test_a_store_failure_outside_the_declared_set_is_a_500() -> None:
    class _Buggy(InMemoryTurnFeedbackStore):
        transient_errors = (ConnectionError,)

        async def upsert(self, **kwargs: Any) -> Any:
            raise KeyError('column')

    client, _, _, _ = await _app(store=_Buggy())
    assert client.put('/conversations/c1/turns/r1/report', json={'category': 'x'}).status_code == 500


async def test_two_feedback_addons_coexist_by_name() -> None:
    class Rating(BaseModel):
        up: bool

    registry, step_store = InMemoryConversationRegistry(), InMemoryStepStore()
    feature = ConversationsFeature(
        registry=registry,
        step_store=step_store,
        addons=[
            TurnFeedbackAddon(payload_model=Report, store=InMemoryTurnFeedbackStore(), name='report'),
            TurnFeedbackAddon(payload_model=Rating, store=InMemoryTurnFeedbackStore(), name='rating'),
        ],
    )
    assert feature.config()['addons'] == {'report': {}, 'rating': {}}


def test_name_must_be_a_slug() -> None:
    with pytest.raises(ValueError, match='slug'):
        TurnFeedbackAddon(payload_model=Report, store=InMemoryTurnFeedbackStore(), name='my-report')


async def test_load_turns_without_routes_still_works() -> None:
    """The loader is usable on its own (an app composing the detail read by hand)."""
    store = InMemoryTurnFeedbackStore()
    await store.upsert(
        kind='report',
        namespace_key='ns',
        conversation_id='c1',
        run_id='r1',
        author_user_id='alice',
        data={'category': 'x'},
    )
    addon = TurnFeedbackAddon(payload_model=Report, store=store, name='report')
    turn = TranscriptTurn(
        run_id='r1', started_at=datetime.now(UTC), phase='answered', messages=[], author_user_id='alice'
    )

    entries = await addon.load_turns(AddonTurnQuery(namespace_key='ns', conversation_id='c1', user=ALICE, turns=[turn]))

    assert entries['r1'].model_dump()['data'] == {'category': 'x', 'remarks': ''}


class _RecordingStore(InMemoryTurnFeedbackStore):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    async def open(self) -> None:
        self.events.append('open')

    async def close(self) -> None:
        self.events.append('close')


async def test_the_feature_lifespan_opens_and_closes_the_store() -> None:
    store = _RecordingStore()
    addon = TurnFeedbackAddon(payload_model=Report, store=store, name='report')
    feature = ConversationsFeature(
        registry=InMemoryConversationRegistry(), step_store=InMemoryStepStore(), addons=[addon]
    )

    async with feature.lifespan(FastAPI()):
        assert store.events == ['open']
    assert store.events == ['open', 'close']


async def test_the_in_memory_store_opens_and_closes_without_effect() -> None:
    store = InMemoryTurnFeedbackStore()
    await store.open()
    await store.upsert(kind='r', namespace_key='ns', conversation_id='c', run_id='r1', author_user_id='a', data={})
    await store.close()
    assert len(await store.list(kind='r', namespace_key='ns', conversation_id='c')) == 1


async def test_an_aliased_payload_model_round_trips() -> None:
    """A camelCase wire model is stored by alias and validated back by alias on every read."""

    class Issue(BaseModel):
        model_config = ConfigDict(alias_generator=to_camel)
        issue_category: str

    registry, step_store = InMemoryConversationRegistry(), InMemoryStepStore()
    addon = TurnFeedbackAddon(payload_model=Issue, store=InMemoryTurnFeedbackStore(), name='issue')
    feature = ConversationsFeature(registry=registry, step_store=step_store, addons=[addon])
    namespace = feature.namespace_for(ALICE)
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    await registry.start_turn(namespace, 'c1', run_id='r1', author_user_id='alice')
    app = FastAPI()
    for router in feature.routers(auth=_auth_as(ALICE)):
        app.include_router(router)
    client = TestClient(app, raise_server_exceptions=False)

    put = client.put('/conversations/c1/turns/r1/issue', json={'issueCategory': 'wrong'})

    assert put.status_code == 200 and put.json()['data'] == {'issueCategory': 'wrong'}
    assert client.get('/conversations/c1/issue').json()[0]['data'] == {'issueCategory': 'wrong'}


async def test_a_row_the_current_model_rejects_is_skipped_not_fatal() -> None:
    """A row written under an older payload model must not take the whole transcript down."""
    store = InMemoryTurnFeedbackStore()
    client, feature, _, _ = await _app(store=store)
    namespace = feature.namespace_for(ALICE)
    await store.upsert(
        kind='report',
        namespace_key=namespace,
        conversation_id='c1',
        run_id='r1',
        author_user_id='alice',
        data={'old': 1},
    )

    detail = client.get('/conversations/c1')
    entries = client.get('/conversations/c1/report')

    assert detail.status_code == 200 and detail.json()['turns'][0]['addons']['report'] is None
    assert entries.status_code == 200 and entries.json() == []


async def test_the_list_keeps_every_author_and_names_the_turn() -> None:
    """Two authors on one turn and two turns of one author are four entries, each with its run_id."""
    store = InMemoryTurnFeedbackStore()
    alice_client, feature, registry, _ = await _app(store=store, scope=('user.tenant_id',))
    namespace = feature.namespace_for(ALICE)
    await registry.start_turn(namespace, 'c1', run_id='r2', author_user_id='alice')
    alice_client.put('/conversations/c1/turns/r1/report', json={'category': 'a1'})
    alice_client.put('/conversations/c1/turns/r2/report', json={'category': 'a2'})
    await store.upsert(
        kind='report',
        namespace_key=namespace,
        conversation_id='c1',
        run_id='r1',
        author_user_id='bob',
        data={'category': 'b1'},
    )

    entries = alice_client.get('/conversations/c1/report').json()

    assert sorted((e['run_id'], e['author_user_id'], e['data']['category']) for e in entries) == [
        ('r1', 'alice', 'a1'),
        ('r1', 'bob', 'b1'),
        ('r2', 'alice', 'a2'),
    ]


async def test_readiness_includes_the_feedback_store() -> None:
    class _DownPing(InMemoryTurnFeedbackStore):
        async def ping(self) -> Mapping[str, Any]:
            raise ConnectionError('down')

    _, feature, _, _ = await _app()
    assert await feature.health() == {'ok': True, 'addons': {'report': {'ok': True}}}

    _, down, _, _ = await _app(store=_DownPing())
    with pytest.raises(ConnectionError):
        await down.health()
