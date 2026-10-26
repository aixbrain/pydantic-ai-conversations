# Conversations for Pydantic AI App Factory

`pydantic-ai-conversations` adds persistent, multi-turn conversations to an app built with
[pydantic-ai-app-factory](https://github.com/aixbrain/pydantic-ai-app-factory). Messages are
stored through [pydantic-ai-harness](https://pydantic.dev/docs/ai/harness/)'s Step
Persistence, the harness's durable record of an agent run. This package adds what a chat
product needs on top: a registry that lists, titles, deletes, and isolates conversations per
tenant; a capability that loads the stored conversation into each run and records each turn,
so retries, edits, and tool approvals are ordinary turns; and the `/conversations` routes a
frontend reads.

> [!NOTE]
> This package is in early development, the API might still change before v1.0.

## Quick Start

Install the package and the FastAPI CLI:

```bash
pip install pydantic-ai-conversations "fastapi[standard]"
```

Create `main.py`. The agent uses pydantic-ai's test model and the auth function is a stub, so
the app runs without a model key:

```python
from pydantic_ai_app_factory import UserContext, create_agent, create_agent_app
from pydantic_ai_conversations import ConversationsFeature
from pydantic_ai_conversations.sqlite import SqliteConversationRegistry, SqliteConversationStepStore


def auth() -> UserContext:
    return UserContext(user_id='alice', tenant_id='t1')


conversations = ConversationsFeature(
    registry=SqliteConversationRegistry('app.db'),
    step_store=SqliteConversationStepStore('app.db'),
)
agent = create_agent('test', instructions='You are a support assistant.')
app = create_agent_app(agent, auth=auth, features=[conversations])
```

Run it with `fastapi dev`. A conversation starts with the first request that names a
`conversation_id`; the id is yours to choose. Every later request names the turn (one agent
run) it builds on, through the `previous_run_id` the previous answer returned:

```bash
curl -s http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "My printer is offline.", "conversation_id": "c1"}' | jq -r .run_id
```

```bash
curl -s http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "It is a Brother.", "conversation_id": "c1", "extras": {"previous_run_id": "<the run_id printed above>"}}'
```

`GET /conversations` lists the caller's conversations and `GET /conversations/c1` returns the
transcript. Over `/chat`, the streaming endpoint for Vercel AI SDK frontends, the chat `id` is
the conversation, `previousRunId` names the previous turn, and the streamed message's id is the
run id to send back.

## What stays with the app

Who the caller is, and whether they may act in a tenant or case, is the app's `AuthProvider`.
Sanitising the messages a client sends is the endpoint's job; the app factory does it. Message
durability is the harness's. This package enforces one thing: a conversation is reachable only
inside the partition its caller's identity maps to.

## Documentation

The quick start uses a test model, a stub auth function, and SQLite. The documentation covers
the rest.

1. [Getting started](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/docs/getting-started.md):
   install, wire, first conversation over `/run` and `/chat`.
   1. [Example app](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/pydantic_ai_conversations_examples/quickstart.py)
1. [How a conversation is stored](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/docs/concepts.md):
   the terms, and what the harness, this package, and the app each own.
1. [Continuing a conversation](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/docs/continuing-a-conversation.md):
   `previous_run_id`, retries, edits, approvals, history limits.
1. [Scope and access](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/docs/scope-and-access.md):
   tenants, users, shared cases, and where the access wall ends.
   1. [Example: conversations shared per case](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/pydantic_ai_conversations_examples/shared_cases.py)
1. [The /conversations routes](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/docs/http-api.md):
   listing, transcript, and what a frontend renders.
   1. [Example: attrs set by the server](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/pydantic_ai_conversations_examples/ticket_attrs.py)
1. [Deleting conversations](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/docs/deleting.md):
   soft delete, purge, cleanup hooks, what remains.
1. [Backends](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/docs/backends.md):
   SQLite, MongoDB, PostgreSQL, schema and upgrades.
1. [Extending](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/docs/extending.md):
   your own registry or store, transforms, add-ons, use without the app factory.
1. [Operations](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/docs/operations.md):
   readiness, concurrency, logging, limits, version pins.
1. [Testing](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/docs/testing.md):
   testing an app with conversations; the registry conformance suite.
1. [Reference](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/docs/api-reference.md):
   signatures, wire shapes, error codes.

## Compatibility

Python 3.11 or later. The package depends on a narrow range of `pydantic-ai-app-factory` and
`pydantic-ai-harness` versions, and the harness range fixes the pydantic-ai version with it;
[Operations](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/docs/operations.md#version-pins)
says why, and
[pyproject.toml](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/pyproject.toml)
has the current ranges.

## Contributing

Apache-2.0. Set up with `make install`, run `make all` before a pull request, and see
[AGENTS.md](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/AGENTS.md) for the
module layout and boundaries.

## License

[Apache-2.0](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/LICENSE)
