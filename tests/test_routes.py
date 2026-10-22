"""The /conversations management routes: list, get, optional create/title, soft-delete + hooks."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai_app_factory import AppDeps, UserContext
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, RunRecord, StepPersistence

from pydantic_ai_conversations import ConversationsCapability, load_transcript
from pydantic_ai_conversations._routes import DeletionEvent, OnDeleteHook, build_conversation_router
from pydantic_ai_conversations.errors import RegistryUnavailable
from pydantic_ai_conversations.registry import (
    ConversationMeta,
    ConversationRegistry,
    InMemoryConversationRegistry,
    TurnError,
    TurnRecord,
)
from pydantic_ai_conversations.scope import ScopeField, build_namespace, parse_scope
from pydantic_ai_conversations.stores import NewMessagesPreparer, PreparingStepStore

SCOPE = parse_scope(('user.tenant_id', 'user.user_id'))
ALICE = UserContext(user_id='alice', tenant_id='t1')


def _auth() -> UserContext:
    return ALICE


class _UnavailableListRegistry(InMemoryConversationRegistry):
    """Raises on `list` only, as a real backend does on a transient read-path outage."""

    async def list(
        self,
        namespace_key: str,
        *,
        attrs: Mapping[str, str] | None = None,
        page_after: str | None = None,
        limit: int = 100,
    ) -> Sequence[ConversationMeta]:
        raise RegistryUnavailable('backend down')


class _UnavailableGetRegistry(InMemoryConversationRegistry):
    """Raises on `get` only."""

    async def get(self, namespace_key: str, conversation_id: str) -> ConversationMeta | None:
        raise RegistryUnavailable('backend down')


class _UnavailableSoftDeleteRegistry(InMemoryConversationRegistry):
    """Raises on `soft_delete` only."""

    async def soft_delete(self, namespace_key: str, conversation_id: str) -> ConversationMeta:
        raise RegistryUnavailable('backend down')


class _UnavailableListTurnsRegistry(InMemoryConversationRegistry):
    """Raises on `list_turns` only -- `get` succeeds (the entry exists), but the turn-authorship
    read degrades independently."""

    async def list_turns(self, namespace_key: str, conversation_id: str) -> Sequence[TurnRecord]:
        raise RegistryUnavailable('backend down')


class _RegisterUnavailableRegistry(InMemoryConversationRegistry):
    """Raises on `register_conversation` only, as hit by the opt-in create route."""

    async def register_conversation(
        self, namespace_key: str, conversation_id: str, *, created_by: str, attrs: Mapping[str, str] | None = None
    ) -> ConversationMeta:
        raise RegistryUnavailable('backend down')


class _UnavailableSetTitleRegistry(InMemoryConversationRegistry):
    """Raises on `set_title` only, as hit by the opt-in title route."""

    async def set_title(self, namespace_key: str, conversation_id: str, title: str) -> None:
        raise RegistryUnavailable('backend down')


class _UnavailableReadStepStore(InMemoryStepStore):
    """Raises on `list_runs` only, as a real step-store backend does on a transient read-path
    outage. The failure type is store-specific, so the store declares which ones are temporary.
    """

    transient_errors = (OSError,)
    failure: Exception = ConnectionError('backend down')

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None
    ) -> list[RunRecord]:
        raise self.failure


def _client(
    registry: ConversationRegistry,
    store: InMemoryStepStore,
    *,
    scope: tuple[ScopeField, ...] = SCOPE,
    enable_create: bool = False,
    enable_title: bool = False,
    filterable_attrs: tuple[str, ...] = (),
    on_delete: tuple[OnDeleteHook, ...] = (),
    transcript_view: Any = None,
    ui_messages: Any = None,
) -> TestClient:
    app = FastAPI()
    app.include_router(
        build_conversation_router(
            registry=registry,
            step_store=store,
            scope=scope,
            auth=_auth,
            enable_create=enable_create,
            enable_title=enable_title,
            filterable_attrs=filterable_attrs,
            on_delete=on_delete,
            transcript_view=transcript_view,
            ui_messages=ui_messages,
        )
    )
    return TestClient(app, raise_server_exceptions=False)


async def _seed(registry: InMemoryConversationRegistry, store: InMemoryStepStore, conversation_id: str = 'c1') -> str:
    namespace = build_namespace(SCOPE, ALICE, {})
    await registry.register_conversation(namespace, conversation_id, created_by='alice')
    await store.register_run(RunRecord(run_id=f'{conversation_id}-r1', conversation_id=conversation_id))
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id=f'{conversation_id}-r1',
            step_index=1,
            messages=[
                ModelRequest(parts=[UserPromptPart(content='hi')]),
                ModelResponse(parts=[TextPart(content='hello')]),
            ],
            conversation_id=conversation_id,
        )
    )
    return namespace


async def test_list_returns_namespace_conversations() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)
    response = _client(registry, store).get('/conversations')
    assert response.status_code == 200
    body = response.json()
    assert [item['conversation_id'] for item in body['data']] == ['c1']
    assert body['more_after'] is None


async def test_list_paginates_with_cursor() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    namespace = build_namespace(SCOPE, ALICE, {})
    now = datetime.now(UTC)
    for cid in ('c1', 'c2', 'c3'):
        await registry.register_conversation(namespace, cid, created_by='alice')
        await registry.update_message_stats(namespace, cid, last_message_at=now, message_count_delta=1)
    response = _client(registry, store).get('/conversations', params={'pageSize': 2})
    body = response.json()
    assert len(body['data']) == 2
    assert body['more_after'] == body['data'][-1]['conversation_id']


async def test_list_filters_on_declared_attrs() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    namespace = build_namespace(SCOPE, ALICE, {})
    await registry.register_conversation(namespace, 'm42-chat', created_by='alice', attrs={'machine_id': 'm-42'})
    await registry.register_conversation(namespace, 'm7-chat', created_by='alice', attrs={'machine_id': 'm-7'})
    client = _client(registry, store, filterable_attrs=('machine_id',))
    filtered = client.get('/conversations', params={'machine_id': 'm-42'})
    assert [item['conversation_id'] for item in filtered.json()['data']] == ['m42-chat']
    unfiltered = client.get('/conversations')
    assert len(unfiltered.json()['data']) == 2
    # An UNDECLARED attr is not queryable: the param is ignored, not applied.
    undeclared = _client(registry, store).get('/conversations', params={'machine_id': 'm-42'})
    assert len(undeclared.json()['data']) == 2


async def test_get_returns_messages() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)
    response = _client(registry, store).get('/conversations/c1')
    assert response.status_code == 200
    body = response.json()
    assert body['conversation']['conversation_id'] == 'c1'
    assert len(body['messages']) == 2


async def test_get_returns_turns_with_status() -> None:
    """`turns` is the canonical field: a complete-only conversation shows one `complete` turn, and
    `messages` (kept for compatibility) is unchanged -- built from the same transcript fetch, not a
    second store roundtrip.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)
    response = _client(registry, store).get('/conversations/c1')
    assert response.status_code == 200
    body = response.json()
    assert len(body['turns']) == 1
    turn = body['turns'][0]
    assert turn['run_id'] == 'c1-r1'
    assert len(turn['messages']) == 2
    assert len(body['messages']) == 2


async def test_get_returns_author_user_id_per_turn() -> None:
    """`get_conversation` fetches the registry's `TurnRecord`s and stamps each turn with
    its author -- what a case-app UI needs to render "who wrote this".
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    namespace = await _seed(registry, store)
    await registry.start_turn(namespace, 'c1', run_id='c1-r1', author_user_id='alice')
    response = _client(registry, store).get('/conversations/c1')
    assert response.status_code == 200
    assert response.json()['turns'][0]['author_user_id'] == 'alice'


async def test_get_returns_none_author_user_id_when_registry_has_no_attribution() -> None:
    # A run the registry has no `TurnRecord` for is attributed to nobody: `None`, as
    # `load_transcript` leaves it for a run absent from `turns`.
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)
    response = _client(registry, store).get('/conversations/c1')
    assert response.status_code == 200
    assert response.json()['turns'][0]['author_user_id'] is None


async def test_get_surfaces_a_failed_turn_with_its_prompt() -> None:
    """A turn whose only snapshot is its saved prompt appears in `turns` as `failed` with that
    prompt and, as the latest turn, at the end of the flat `messages`, where the branch a reload
    shows ends.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    namespace = await _seed(registry, store)
    await registry.start_turn(namespace, 'c1', run_id='c1-r1', author_user_id='alice')
    await registry.set_turn_phase(namespace, 'c1', run_id='c1-r1', phase='answered')
    await registry.start_turn(namespace, 'c1', run_id='c1-r2', author_user_id='alice', previous_run_id='c1-r1')
    await registry.set_turn_phase(namespace, 'c1', run_id='c1-r2', phase='failed')
    await store.register_run(RunRecord(run_id='c1-r2', conversation_id='c1'))
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='c1-r2',
            step_index=0,
            messages=[ModelRequest(parts=[UserPromptPart(content='are you there')])],
            conversation_id='c1',
        )
    )

    response = _client(registry, store).get('/conversations/c1')

    assert response.status_code == 200
    body = response.json()
    assert [t['phase'] for t in body['turns']] == ['answered', 'failed']
    pending = body['turns'][1]
    assert pending['run_id'] == 'c1-r2'
    assert 'are you there' in str(pending['messages'])
    assert len(body['messages']) == 3 and 'are you there' in str(body['messages'][-1])


async def test_get_returns_outcome_per_turn() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    namespace = await _seed(registry, store)
    await registry.start_turn(namespace, 'c1', run_id='c1-r1', author_user_id='alice')
    await registry.set_turn_phase(namespace, 'c1', run_id='c1-r1', phase='answered')

    response = _client(registry, store).get('/conversations/c1')

    assert response.status_code == 200
    assert response.json()['turns'][0]['phase'] == 'answered'
    assert response.json()['turns'][0]['error'] is None


async def test_get_returns_the_stored_failure_reason() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    namespace = await _seed(registry, store)
    await registry.start_turn(namespace, 'c1', run_id='c1-r1', author_user_id='alice')
    error = TurnError('model-unauthorized', 'The model provider rejected the credentials.')
    await registry.set_turn_phase(namespace, 'c1', run_id='c1-r1', phase='failed', error=error)

    response = _client(registry, store).get('/conversations/c1')

    assert response.status_code == 200
    assert response.json()['turns'][0]['error'] == {
        'code': 'model-unauthorized',
        'message': 'The model provider rejected the credentials.',
        'retryable': False,
    }


async def test_get_surfaces_a_failed_registry_outcome_over_the_derived_one() -> None:
    """The registry's authoritative phase state must reach the wire: a run that settled a tool
    round then crashed leaves a `complete` snapshot (so structural derivation would say `answered`),
    but its registry `TurnRecord` says `failed` -- and that is what the FE must see to offer retry.
    Proves the route threads the `phases` map from the registry into `load_transcript`.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    namespace = await _seed(registry, store)  # c1-r1 has a complete hi/hello snapshot
    await registry.start_turn(namespace, 'c1', run_id='c1-r1', author_user_id='alice')
    await registry.set_turn_phase(namespace, 'c1', run_id='c1-r1', phase='failed')

    response = _client(registry, store).get('/conversations/c1')

    assert response.status_code == 200
    turn = response.json()['turns'][0]
    assert turn['phase'] == 'failed'


async def test_foreign_or_missing_is_404() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    # c-foreign lives in another namespace -> invisible.
    foreign_namespace = '[["user.tenant_id","t2"],["user.user_id","bob"]]'
    await registry.register_conversation(foreign_namespace, 'c-foreign', created_by='bob')
    client = _client(registry, store)
    for response in (
        client.get('/conversations/c-foreign'),
        client.get('/conversations/missing'),
        client.delete('/conversations/missing'),
    ):
        assert response.status_code == 404
        assert response.json()['detail']['code'] == 'conversations/not-found'


async def test_delete_soft_deletes_and_runs_hooks() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    namespace = await _seed(registry, store)
    events: list[DeletionEvent] = []

    async def hook(event: DeletionEvent) -> None:
        events.append(event)

    client = _client(registry, store, on_delete=(hook,))
    assert client.delete('/conversations/c1').status_code == 204
    assert len(events) == 1
    event = events[0]
    assert (event.namespace_key, event.conversation_id) == (namespace, 'c1')
    assert event.kind == 'soft_delete'
    assert event.meta is not None and event.meta.deleted_at is not None
    assert await registry.get(namespace, 'c1') is None
    after_delete = client.get('/conversations/c1')
    assert after_delete.status_code == 404
    assert after_delete.json()['detail']['code'] == 'conversations/not-found'


async def test_failing_hook_does_not_undo_the_delete() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)

    async def bad_hook(event: DeletionEvent) -> None:
        raise RuntimeError('cleanup exploded')

    client = _client(registry, store, on_delete=(bad_hook,))
    assert client.delete('/conversations/c1').status_code == 204
    assert client.get('/conversations/c1').status_code == 404


async def test_create_route_is_opt_in() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    assert _client(registry, store).post('/conversations').status_code in (404, 405)
    response = _client(registry, store, enable_create=True).post('/conversations')
    assert response.status_code == 201
    conversation_id = response.json()['conversation_id']
    namespace = build_namespace(SCOPE, ALICE, {})
    assert await registry.get(namespace, conversation_id) is not None


async def test_create_route_accepts_an_attrs_body() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    client = _client(registry, store, enable_create=True, filterable_attrs=('source',))
    response = client.post('/conversations', json={'attrs': {'source': 'web'}})
    assert response.status_code == 201
    body = response.json()
    assert body['attrs'] == {'source': 'web'}
    # The attr just landed via the create body is filterable, same as a pre-seeded one.
    filtered = client.get('/conversations', params={'source': 'web'})
    assert [item['conversation_id'] for item in filtered.json()['data']] == [body['conversation_id']]


async def test_title_route_is_opt_in() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)
    assert _client(registry, store).put('/conversations/c1/title', json={'title': 'Hi'}).status_code in (404, 405)
    client = _client(registry, store, enable_title=True)
    assert client.put('/conversations/c1/title', json={'title': 'Quarterly numbers'}).status_code == 204
    assert client.get('/conversations/c1').json()['conversation']['title'] == 'Quarterly numbers'


@pytest.mark.parametrize('title', ['', '   ', 'x' * 201])
async def test_title_route_rejects_a_title_that_would_not_render(title: str) -> None:
    """Blank, whitespace-only and overlong titles never reach the registry: the listing shows this
    string, and a row titled `'   '` is indistinguishable from an untitled one.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)
    client = _client(registry, store, enable_title=True)
    assert client.put('/conversations/c1/title', json={'title': title}).status_code == 422
    assert client.get('/conversations/c1').json()['conversation']['title'] is None


async def test_title_route_strips_surrounding_whitespace() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)
    client = _client(registry, store, enable_title=True)
    assert client.put('/conversations/c1/title', json={'title': '  Quarterly numbers  '}).status_code == 204
    assert client.get('/conversations/c1').json()['conversation']['title'] == 'Quarterly numbers'


async def test_title_route_is_404_for_a_foreign_conversation() -> None:
    """The write route obeys the same access wall as the read routes: a live conversation in
    another namespace answers 404, not 403 -- anything else would confirm the id exists.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await registry.register_conversation('[["user.tenant_id","t2"],["user.user_id","bob"]]', 'c1', created_by='bob')
    client = _client(registry, store, enable_title=True)
    assert client.put('/conversations/c1/title', json={'title': 'Mine now'}).status_code == 404
    assert client.put('/conversations/nonexistent/title', json={'title': 'Mine now'}).status_code == 404


async def test_title_route_is_404_after_a_soft_delete() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)
    client = _client(registry, store, enable_title=True)
    assert client.delete('/conversations/c1').status_code == 204
    response = client.put('/conversations/c1/title', json={'title': 'Back from the dead'})
    assert response.status_code == 404
    assert response.json()['detail']['code'] == 'conversations/not-found'


def _assert_unavailable_503(response: Any) -> None:
    # `Any`, not `httpx.Response`: `TestClient` resolves to whichever of `httpx`/`httpx2` is
    # installed (see starlette.testclient), so the concrete response type is an environment detail,
    # not a stable import target for a test helper.
    assert response.status_code == 503
    detail = response.json()['detail']
    assert detail['code'] == 'conversations/unavailable'
    assert detail['retryable'] is True


async def test_registry_unavailable_is_503_on_list() -> None:
    response = _client(_UnavailableListRegistry(), InMemoryStepStore()).get('/conversations')
    _assert_unavailable_503(response)


async def test_registry_unavailable_is_503_on_get() -> None:
    response = _client(_UnavailableGetRegistry(), InMemoryStepStore()).get('/conversations/c1')
    _assert_unavailable_503(response)


async def test_registry_unavailable_is_503_on_list_turns() -> None:
    """`get_conversation`'s `registry.list_turns` read (the author lookup) gets the same
    `RegistryUnavailable` -> 503 handling as its `registry.get` call above it, even though the
    entry itself resolves fine.
    """
    registry = _UnavailableListTurnsRegistry()
    namespace = build_namespace(SCOPE, ALICE, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    response = _client(registry, InMemoryStepStore()).get('/conversations/c1')
    _assert_unavailable_503(response)


async def test_registry_unavailable_is_503_on_delete() -> None:
    response = _client(_UnavailableSoftDeleteRegistry(), InMemoryStepStore()).delete('/conversations/c1')
    _assert_unavailable_503(response)


async def test_registry_unavailable_is_503_on_create() -> None:
    client = _client(_RegisterUnavailableRegistry(), InMemoryStepStore(), enable_create=True)
    _assert_unavailable_503(client.post('/conversations'))


async def test_registry_unavailable_is_503_on_title() -> None:
    registry = _UnavailableSetTitleRegistry()
    await registry.register_conversation(build_namespace(SCOPE, ALICE, {}), 'c1', created_by='alice')
    client = _client(registry, InMemoryStepStore(), enable_title=True)
    _assert_unavailable_503(client.put('/conversations/c1/title', json={'title': 'Hi'}))


async def test_step_store_unavailable_is_503_on_get() -> None:
    """A step-store read failure (here, `load_transcript`'s `list_runs` call) must map
    to the same retryable `conversations/unavailable` contract as a registry outage, not escape as
    a bare 500. The registry entry resolves (`registry.get` succeeds); only the step-store read
    fails.
    """
    registry = InMemoryConversationRegistry()
    namespace = build_namespace(SCOPE, ALICE, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    response = _client(registry, _UnavailableReadStepStore()).get('/conversations/c1')
    _assert_unavailable_503(response)


async def test_a_permanent_step_store_failure_on_get_is_not_reported_as_retryable() -> None:
    registry = InMemoryConversationRegistry()
    await registry.register_conversation(build_namespace(SCOPE, ALICE, {}), 'c1', created_by='alice')
    store = _UnavailableReadStepStore()
    store.failure = ValueError('snapshot row has wrong types')
    response = _client(registry, store).get('/conversations/c1')
    assert response.status_code == 500


def test_filterable_attr_rejects_non_identifier() -> None:
    with pytest.raises(ValueError, match='not a valid identifier'):
        build_conversation_router(
            registry=InMemoryConversationRegistry(),
            step_store=InMemoryStepStore(),
            scope=SCOPE,
            auth=_auth,
            filterable_attrs=('not an identifier',),
        )


def test_filterable_attr_rejects_collision_with_scope_or_pagination() -> None:
    scope = parse_scope(('user.tenant_id', 'extras.case_id'))
    with pytest.raises(ValueError, match='collides'):
        build_conversation_router(
            registry=InMemoryConversationRegistry(),
            step_store=InMemoryStepStore(),
            scope=scope,
            auth=_auth,
            filterable_attrs=('case_id',),
        )
    with pytest.raises(ValueError, match='collides'):
        build_conversation_router(
            registry=InMemoryConversationRegistry(),
            step_store=InMemoryStepStore(),
            scope=SCOPE,
            auth=_auth,
            filterable_attrs=('pageSize',),
        )


# --- extras-bearing scope: the query-string half of the namespace ------------------------------
#
# `case_id` here is an UNVERIFIED CLIENT CLAIM (see `_routes.py`'s module docstring): these tests
# only pin that the claim participates in the namespace and is required + non-empty. Actually
# VERIFYING the claim (e.g. that the caller belongs to `case_id`) is the AuthProvider's job, not
# this router's -- there is no such AuthProvider here, deliberately, so these tests can't and don't
# claim to cover that verification step.

CASE_SCOPE = parse_scope(('user.tenant_id', 'extras.case_id'))


async def test_extras_scope_requires_the_claim_on_every_route() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    client = _client(registry, store, scope=CASE_SCOPE, enable_create=True, enable_title=True)
    assert client.get('/conversations').status_code == 422
    assert client.get('/conversations/c1').status_code == 422
    assert client.delete('/conversations/c1').status_code == 422
    assert client.post('/conversations').status_code == 422
    assert client.put('/conversations/c1/title', json={'title': 'Hi'}).status_code == 422


async def test_extras_scope_empty_claim_is_422() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    client = _client(registry, store, scope=CASE_SCOPE)
    assert client.get('/conversations', params={'case_id': ''}).status_code == 422


async def test_extras_scope_partitions_conversations_by_claim() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    namespace_a = build_namespace(CASE_SCOPE, ALICE, {'case_id': 'case-A'})
    await registry.register_conversation(namespace_a, 'c1', created_by='alice')
    client = _client(registry, store, scope=CASE_SCOPE)
    # Same conversation, addressed under a DIFFERENT claim -> invisible, same as a foreign namespace.
    assert client.get('/conversations/c1', params={'case_id': 'case-B'}).status_code == 404
    assert client.get('/conversations/c1', params={'case_id': 'case-A'}).status_code == 200


async def test_extras_scope_create_lands_in_the_claimed_namespace() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    client = _client(registry, store, scope=CASE_SCOPE, enable_create=True)
    response = client.post('/conversations', params={'case_id': 'case-A'})
    assert response.status_code == 201
    conversation_id = response.json()['conversation_id']
    namespace_a = build_namespace(CASE_SCOPE, ALICE, {'case_id': 'case-A'})
    assert await registry.get(namespace_a, conversation_id) is not None
    namespace_b = build_namespace(CASE_SCOPE, ALICE, {'case_id': 'case-B'})
    assert await registry.get(namespace_b, conversation_id) is None


async def test_transcript_view_shapes_the_wire_transcript() -> None:
    """A view wired via `build_conversation_router` shapes the turns before serialization."""
    from dataclasses import replace as dc_replace

    from pydantic_ai_conversations._loader import TranscriptTurn

    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)

    def scrub(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
        return [
            dc_replace(
                turn,
                messages=[
                    message
                    for message in turn.messages
                    if not any(
                        isinstance(part, TextPart) and part.content == 'hello' for part in getattr(message, 'parts', [])
                    )
                ],
            )
            for turn in turns
        ]

    response = _client(registry, store, transcript_view=scrub).get('/conversations/c1')
    assert response.status_code == 200
    body = response.text
    assert 'hi' in body
    assert 'hello' not in body


async def test_ui_messages_absent_without_an_encoder() -> None:
    """No encoder wired: the field stays `None` and each turn carries its content exactly once."""
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)

    response = _client(registry, store).get('/conversations/c1')

    assert response.status_code == 200
    assert response.json()['turns'][0]['ui_messages'] is None


async def test_ui_messages_are_serialized_by_protocol_alias() -> None:
    """The encoder returns protocol models, and those are camelCase-aliased: serializing them
    through the `list[Any]` field duck-typed would emit `tool_call_id`, which no Vercel AI
    frontend reads. The harness-native `messages` stays untouched next to it, and the flat
    compat field gains no UI twin -- a seed needs the turn structure.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    namespace = build_namespace(SCOPE, ALICE, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    await store.register_run(RunRecord(run_id='c1-r1', conversation_id='c1'))
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='c1-r1',
            step_index=1,
            messages=[
                ModelRequest(parts=[UserPromptPart(content='weather?')]),
                ModelResponse(parts=[ToolCallPart(tool_name='forecast', args={'city': 'Berlin'}, tool_call_id='tc1')]),
                ModelRequest(parts=[ToolReturnPart(tool_name='forecast', content='sunny', tool_call_id='tc1')]),
                ModelResponse(parts=[TextPart(content='it is sunny')]),
            ],
            conversation_id='c1',
        )
    )

    response = _client(registry, store, ui_messages=VercelAIAdapter.dump_messages).get('/conversations/c1')

    assert response.status_code == 200
    body = response.json()
    turn = body['turns'][0]
    tool_part = next(part for msg in turn['ui_messages'] for part in msg['parts'] if part['type'] == 'tool-forecast')
    assert tool_part['toolCallId'] == 'tc1'
    assert 'tool_call_id' not in tool_part
    assert len(turn['messages']) == 4
    assert 'ui_messages' not in body


async def test_ui_messages_cover_a_non_answered_turn() -> None:
    """Rendered per turn, failed turns included: a frontend seeding its state from `turns` gets
    the failed turn's prompt in its own protocol, with `phase` beside it."""
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)
    await store.register_run(RunRecord(run_id='c1-r2', conversation_id='c1'))
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='c1-r2',
            step_index=0,
            messages=[ModelRequest(parts=[UserPromptPart(content='are you there')])],
            conversation_id='c1',
        )
    )

    response = _client(registry, store, ui_messages=VercelAIAdapter.dump_messages).get('/conversations/c1')

    assert response.status_code == 200
    body = response.json()
    assert 'are you there' in str(body['turns'][1]['ui_messages'])
    assert body['turns'][1]['phase'] == 'failed'


async def test_ui_messages_of_two_failed_turns_carry_distinct_ids() -> None:
    """Two runs of one conversation fail before their first model response, leaving two saved
    prompts. A client keys its bubbles on the UI message id, which the Vercel adapter derives
    from the run id the capability stamps, so the two attempts render under distinct ids.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('boom')

    agent: Agent[AppDeps, str] = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )
    for prompt, extras in (('first try', {}), ('second try', {'previous_run_id': None})):
        with pytest.raises(RuntimeError, match='boom'):
            await agent.run(prompt, deps=AppDeps(user=ALICE, extras=extras), conversation_id='c1')

    response = _client(registry, store, ui_messages=VercelAIAdapter.dump_messages).get('/conversations/c1')

    assert response.status_code == 200
    turns = response.json()['turns']
    assert [t['phase'] for t in turns] == ['failed', 'failed']
    assert [t['ui_messages'][0]['role'] for t in turns] == ['user', 'user']
    first_id, second_id = (t['ui_messages'][0]['id'] for t in turns)
    assert first_id != second_id
    assert [t['messages'][0]['run_id'] for t in turns] == [t['run_id'] for t in turns]


async def test_ui_messages_of_a_retried_failed_turn_carry_distinct_ids() -> None:
    """A retry re-sends the failed turn's user message as the Vercel adapter dumped it, timestamp
    metadata included, and the adapter restores that timestamp on the way in. Both attempts then
    arrive with the first attempt's timestamp; each attempt must still be stamped afresh, or the
    adapter's id fallback renders both attempts under one id.
    """
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('boom')

    agent: Agent[AppDeps, str] = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )
    client = _client(registry, store, ui_messages=VercelAIAdapter.dump_messages)
    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('same message', deps=AppDeps(user=ALICE), conversation_id='c1')
    (first_turn,) = await load_transcript(store, conversation_id='c1')
    resent = VercelAIAdapter.load_messages(VercelAIAdapter.dump_messages(first_turn.messages))
    assert resent[0].timestamp == first_turn.messages[0].timestamp  # the round-trip keeps the stamp
    with pytest.raises(RuntimeError, match='boom'):
        await agent.run(
            deps=AppDeps(user=ALICE, extras={'previous_run_id': None}), conversation_id='c1', message_history=resent
        )

    response = client.get('/conversations/c1')

    assert response.status_code == 200
    turns = response.json()['turns']
    assert [t['phase'] for t in turns] == ['failed', 'failed']
    first_id, second_id = (t['ui_messages'][0]['id'] for t in turns)
    assert first_id != second_id


async def test_stamping_keeps_application_metadata() -> None:
    """Only the UI adapters' reserved namespace is dropped from the stamped request; application keys stay."""
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('boom')

    agent: Agent[AppDeps, str] = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )
    sent = ModelRequest(
        parts=[UserPromptPart(content='hi')],
        metadata={'source': 'web', '__pydantic_ai__': {'ui_message_id': 'client-id'}},
    )
    with pytest.raises(RuntimeError, match='boom'):
        await agent.run(deps=AppDeps(user=ALICE), conversation_id='c1', message_history=[sent])

    (turn,) = await load_transcript(store, conversation_id='c1')
    assert turn.messages[0].metadata == {'source': 'web'}


async def test_raising_ui_encoder_fails_the_request() -> None:
    """A broken encoder surfaces as a fault rather than a `None` indistinguishable from
    "no encoder configured"."""

    def boom(messages: Sequence[ModelMessage]) -> Sequence[Any]:
        raise RuntimeError('encoder down')

    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)

    response = _client(registry, store, ui_messages=boom).get('/conversations/c1')

    assert response.status_code == 500
