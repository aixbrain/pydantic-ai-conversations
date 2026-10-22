"""Smoke test: ConversationsFeature composed into a real app-factory app.

`create_agent_app` mounts feature routers behind its own `auth` dependency and reports every
feature on `/config` and `/readyz`. The feature takes no `auth` of its own -- the factory hands
its provider to `routers()` -- so these tests pass one only to the app.
"""

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import Header, HTTPException
from fastapi.testclient import TestClient
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_app_factory import AppDeps, UserContext, create_agent_app
from pydantic_ai_harness.step_persistence import InMemoryStepStore

from pydantic_ai_conversations import ConversationsFeature, InMemoryConversationRegistry


def _auth() -> UserContext:
    return UserContext(user_id='alice', tenant_id='t1')


def _model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='noted')])


def test_conversations_feature_composes_into_the_factory_app() -> None:
    feature = ConversationsFeature(registry=InMemoryConversationRegistry(), step_store=InMemoryStepStore())
    agent: Agent[AppDeps, Any] = Agent(FunctionModel(_model_fn), deps_type=AppDeps)
    app = create_agent_app(agent, auth=_auth, features=[feature])
    client = TestClient(app)

    ready = client.get('/readyz')
    assert ready.status_code == 200 and 'conversations' in ready.json()['reports']

    listed = client.get('/conversations')
    assert listed.status_code == 200 and listed.json() == {'data': [], 'more_after': None}

    config = client.get('/config')
    assert config.status_code == 200 and 'conversations' in config.json()['features']


def _strict_auth(x_api_key: Annotated[str | None, Header()] = None) -> UserContext:
    if x_api_key != 'k':
        raise HTTPException(status_code=401)
    return UserContext(user_id='alice', tenant_id='t1')


def test_feature_config_reports_the_switches_a_client_branches_on() -> None:
    feature = ConversationsFeature(
        registry=InMemoryConversationRegistry(),
        step_store=InMemoryStepStore(),
        enable_create_route=True,
        enable_title_route=True,
    )
    agent: Agent[AppDeps, Any] = Agent(FunctionModel(_model_fn), deps_type=AppDeps)
    client = TestClient(create_agent_app(agent, auth=_strict_auth, features=[feature]))

    assert client.get('/config/features').status_code == 401
    body = client.get('/config/features', headers={'x-api-key': 'k'}).json()
    assert body['feature_config']['conversations'] == {
        'create_route': True,
        'title_route': True,
        'ui_messages': False,
        'filterable_attrs': [],
    }


def _chunks(client: TestClient, messages: list[dict[str, Any]], **extra: Any) -> list[dict[str, Any]]:
    """POST a Vercel AI SDK chat body and return the streamed chunks."""
    response = client.post('/chat', json={'trigger': 'submit-message', 'id': 'c1', 'messages': messages, **extra})
    assert response.status_code == 200
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: {')]


def _chat(client: TestClient, text: str, **extra: Any) -> tuple[str | None, str | None]:
    """POST one user message and return the streamed assistant message id and error text."""
    message = {'id': 'u', 'role': 'user', 'parts': [{'type': 'text', 'text': text}]}
    run_id, error = None, None
    for chunk in _chunks(client, [message], **extra):
        if chunk.get('type') == 'start':
            run_id = chunk.get('messageId')
        if chunk.get('type') == 'error':
            error = chunk.get('errorText')
    return run_id, error


def test_chat_names_the_previous_turn_with_the_camel_case_field() -> None:
    """The Vercel body carries `previousRunId` at the top level, bound to `previous_run_id` by the
    factory's camelCase alias; the streamed assistant message id is the run id a client sends back.
    A missing or unknown id is answered inside the stream, as the factory renders every run error.
    """
    seen: list[list[ModelMessage]] = []

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        seen.append(list(messages))
        yield 'noted'

    feature = ConversationsFeature(registry=InMemoryConversationRegistry(), step_store=InMemoryStepStore())
    agent: Agent[AppDeps, Any] = Agent(FunctionModel(stream_function=stream_fn), deps_type=AppDeps)
    client = TestClient(create_agent_app(agent, auth=_auth, features=[feature]))

    first, error = _chat(client, 'first')
    assert first is not None and error is None
    _, error = _chat(client, 'second, without the field')
    assert error is not None and 'previous_run_id is required' in error
    second, error = _chat(client, 'second', previousRunId=first)
    assert error is None and len(seen[-1]) == 3  # turn 1 plus the new prompt
    root, error = _chat(client, 'again from nothing', previousRunId=None)
    assert error is None and len(seen[-1]) == 1
    _, error = _chat(client, 'bogus', previousRunId='nope')
    assert error is not None and 'does not exist in this conversation' in error

    turns = client.get('/conversations/c1').json()['turns']
    assert [t['run_id'] for t in turns] == [first, second, root]
    assert [t['previous_run_id'] for t in turns] == [None, first, None]
    assert [t['phase'] for t in turns] == ['answered', 'answered', 'answered']


def test_chat_approves_a_paused_tool_call_by_naming_the_paused_turn() -> None:
    """Human in the loop over `/chat`: a tool that needs approval pauses the run, the client sends
    the approval with `previousRunId` naming the paused turn, and the server loads that turn's
    pending calls from the store, so no client-held history is needed."""

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(part.part_kind == 'tool-return' for part in last.parts):
            yield 'deleted'
        else:
            yield {0: DeltaToolCall(name='delete_file', json_args='{"path": "x"}', tool_call_id='tc-1')}

    feature = ConversationsFeature(registry=InMemoryConversationRegistry(), step_store=InMemoryStepStore())
    agent: Agent[AppDeps, Any] = Agent(
        FunctionModel(stream_function=stream_fn), deps_type=AppDeps, output_type=[str, DeferredToolRequests]
    )

    @agent.tool_plain(requires_approval=True)
    def delete_file(path: str) -> str:
        return f'deleted {path}'

    # Approval responses are an AI SDK v6 shape; on v5 the adapter reads no deferred results.
    client = TestClient(create_agent_app(agent, auth=_auth, features=[feature], vercel_sdk_version=6))
    user = {'id': 'u', 'role': 'user', 'parts': [{'type': 'text', 'text': 'delete x'}]}

    chunks = _chunks(client, [user])
    paused = next(chunk['messageId'] for chunk in chunks if chunk['type'] == 'start')
    assert not any(chunk['type'] == 'error' for chunk in chunks), chunks
    assert any(chunk.get('toolCallId') == 'tc-1' for chunk in chunks), chunks

    approval = {
        'id': paused,
        'role': 'assistant',
        'parts': [
            {
                'type': 'tool-delete_file',
                'toolCallId': 'tc-1',
                'state': 'approval-responded',
                'input': {'path': 'x'},
                'approval': {'id': 'tc-1', 'approved': True},
            }
        ],
    }
    chunks = _chunks(client, [user, approval], previousRunId=paused)

    assert not any(chunk['type'] == 'error' for chunk in chunks), chunks
    assert ''.join(chunk.get('delta', '') for chunk in chunks if chunk['type'] == 'text-delta') == 'deleted'
    turns = client.get('/conversations/c1').json()['turns']
    assert [(t['phase'], t['previous_run_id']) for t in turns] == [('answered', None), ('answered', paused)]
    assert [m['kind'] for m in turns[0]['messages']] == ['request', 'response']
    assert [p['part_kind'] for m in turns[1]['messages'] for p in m['parts']] == ['tool-return', 'text']
