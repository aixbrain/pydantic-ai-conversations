# Testing

This guide shows you how to test an app that uses the feature, and how to prove a registry of
your own.

Prerequisites: the app factory's
[Testing](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/testing.md), which
covers scripting the model and asserting on errors.

## An app with conversations

Use the in-memory registry and the harness's in-memory step store. Both start empty and need no
cleanup:

```python
from fastapi.testclient import TestClient
from pydantic_ai_app_factory import UserContext, create_agent, create_agent_app
from pydantic_ai_harness.step_persistence import InMemoryStepStore
from pydantic_ai_conversations import ConversationsFeature, InMemoryConversationRegistry


def auth() -> UserContext:
    return UserContext(user_id='alice', tenant_id='t1')


def test_second_turn_sees_the_first() -> None:
    feature = ConversationsFeature(registry=InMemoryConversationRegistry(), step_store=InMemoryStepStore())
    agent = create_agent('test', instructions='You are a support assistant.')
    with TestClient(create_agent_app(agent, auth=auth, features=[feature])) as client:
        first = client.post('/run', json={'prompt': 'My printer is offline.', 'conversation_id': 'c1'}).json()
        second = client.post(
            '/run',
            json={
                'prompt': 'It is a Brother.',
                'conversation_id': 'c1',
                'extras': {'previous_run_id': first['run_id']},
            },
        ).json()
        turns = client.get('/conversations/c1').json()['turns']
        assert [t['previous_run_id'] for t in turns] == [None, first['run_id']]
        assert second['run_id'] == turns[1]['run_id']
```

`TestClient` needs `httpx` installed. Use `with TestClient(...)` so the lifespan runs and the
registry is opened. A `FunctionModel`
that records the messages it receives is the way to assert what the model saw on the second
turn; see the pydantic-ai
[testing guide](https://pydantic.dev/docs/ai/guides/unit-testing/).

The error codes to assert on are listed in the [reference](api-reference.md#error-codes).
On `/run` they arrive as the status and a flat body; on `/chat` as a `data-error` chunk inside
a 200 stream.

## Driving the capability without HTTP

The feature's capabilities can go straight onto an agent, which keeps a test to `agent.run`:

```python
from pydantic_ai import Agent
from pydantic_ai_app_factory import AppDeps, UserContext

agent = Agent('test', deps_type=AppDeps, capabilities=feature.capabilities())
deps = AppDeps(user=UserContext(user_id='alice', tenant_id='t1'))
first = await agent.run('My printer is offline.', deps=deps, conversation_id='c1')
deps = AppDeps(user=UserContext(user_id='alice', tenant_id='t1'), extras={'previous_run_id': first.run_id})
await agent.run('It is a Brother.', deps=deps, conversation_id='c1')
```

`feature.namespace_for(deps.user, deps.extras)` gives the key for reading the registry
directly, so a test never spells the scope twice.

## A registry of your own

Subclass `RegistryContract` from `pydantic_ai_conversations.testing` and override its
`registry` fixture; see [Extending](extending.md#your-own-registry). The suite runs against
a real backend, because the properties it checks -- the atomic id claim, lost increments
under concurrency, index ordering -- are what an in-memory fake cannot reproduce.

Going deeper: [Extending](extending.md).
