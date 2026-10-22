"""Standalone usage: the capability on a plain Agent, the router in a plain FastAPI app -- this
file deliberately contains no pydantic_ai_app_factory import; that absence IS the regression pin.
"""

from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, RunRecord, StepPersistence

from pydantic_ai_conversations import (
    ConversationsCapability,
    HasUserId,
    InMemoryConversationRegistry,
    NewMessagesPreparer,
    PreparingStepStore,
    SupportsConversationDeps,
    build_conversation_router,
    build_namespace,
    parse_scope,
)

SCOPE = parse_scope(('user.org_id',))


@dataclass(frozen=True)
class StandaloneUser:
    user_id: str
    org_id: str


@dataclass
class StandaloneDeps:
    user: StandaloneUser
    extras: dict[str, Any] = field(default_factory=dict)


def _deps(previous: str | None = None) -> StandaloneDeps:
    extras = {'previous_run_id': previous} if previous is not None else {}
    return StandaloneDeps(user=StandaloneUser(user_id='u1', org_id='acme'), extras=extras)


def _agent(
    registry: InMemoryConversationRegistry, store: InMemoryStepStore, seen: list[list[ModelMessage]]
) -> Agent[StandaloneDeps, str]:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart(content='ok')])

    return Agent(
        FunctionModel(model_fn),
        deps_type=StandaloneDeps,
        capabilities=[
            # The write seam the feature would otherwise wire: snapshots stay per-turn batches.
            ConversationsCapability[StandaloneDeps](registry=registry, step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
        ],
    )


async def test_capability_on_a_plain_agent() -> None:
    """The capability runs on a bare `Agent[StandaloneDeps, str]` -- no `AppDeps`, no `Feature`."""
    registry, store, seen = InMemoryConversationRegistry(), InMemoryStepStore(), []
    agent = _agent(registry, store, seen)
    first = await agent.run('first prompt', deps=_deps(), conversation_id='c1')
    await agent.run('second prompt', deps=_deps(previous=first.run_id), conversation_id='c1')
    # The second run's model request starts with the first turn's messages.
    second_request_messages = seen[1]
    assert len(second_request_messages) > len(seen[0])
    flattened = str(second_request_messages)
    assert 'first prompt' in flattened and 'second prompt' in flattened
    namespace = build_namespace(SCOPE, _deps().user, {})
    entry = await registry.get(namespace, 'c1')
    assert entry is not None
    assert 'acme' in namespace


def _router_auth() -> StandaloneUser:
    return StandaloneUser(user_id='u1', org_id='acme')


def _router_client(registry: InMemoryConversationRegistry, store: InMemoryStepStore) -> TestClient:
    app = FastAPI()
    app.include_router(build_conversation_router(registry=registry, step_store=store, scope=SCOPE, auth=_router_auth))
    return TestClient(app, raise_server_exceptions=False)


async def _seed(registry: InMemoryConversationRegistry, store: InMemoryStepStore, conversation_id: str = 'c1') -> str:
    namespace = build_namespace(SCOPE, StandaloneUser(user_id='u1', org_id='acme'), {})
    await registry.register_conversation(namespace, conversation_id, created_by='u1')
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


async def test_router_in_a_plain_fastapi_app() -> None:
    """`build_conversation_router` mounts into a bare `FastAPI()` -- no `create_agent_app`, no
    `AuthProvider`, just a plain callable returning something with the right attributes."""
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    await _seed(registry, store)
    client = _router_client(registry, store)

    listed = client.get('/conversations')
    assert listed.status_code == 200
    assert [item['conversation_id'] for item in listed.json()['data']] == ['c1']

    detail = client.get('/conversations/c1')
    assert detail.status_code == 200
    assert len(detail.json()['messages']) == 2

    assert client.delete('/conversations/c1').status_code == 204
    assert client.get('/conversations/c1').status_code == 404


# Compile-time pins: pyright must accept `StandaloneUser`/`StandaloneDeps` as satisfying
# `HasUserId`/`SupportsConversationDeps` purely structurally -- neither class inherits from
# either protocol. This IS the assertion; the test function below only keeps pytest from
# flagging an empty file.
_pinned_user: HasUserId = StandaloneUser(user_id='u', org_id='o')
_pinned_deps: SupportsConversationDeps = StandaloneDeps(user=StandaloneUser(user_id='u', org_id='o'))


def test_typing_is_structural_not_nominal() -> None:
    assert _pinned_user.user_id == 'u'
    assert _pinned_deps.user.user_id == 'u'
