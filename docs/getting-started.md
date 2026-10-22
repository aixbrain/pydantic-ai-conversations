# Getting Started

This guide shows you how to add conversation persistence to an existing app-factory app and
drive a first conversation through it, over `/run` and over `/chat`.

Prerequisites: an app built with
[pydantic-ai-app-factory](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/building-an-app.md),
whose `auth` function returns the caller's identity on every request. The terms used here
(turn, phase, branch) are defined in [How a conversation is stored](concepts.md).

## Install

```bash
pip install pydantic-ai-conversations
```

The SQLite backend needs nothing else. MongoDB and PostgreSQL ship behind extras:

```bash
pip install "pydantic-ai-conversations[mongodb]"
pip install "pydantic-ai-conversations[postgres]"
```

Installing pulls in `pydantic-ai-app-factory` and `pydantic-ai-harness`. The harness
dependency fixes the pydantic-ai version as well; [Operations](operations.md#version-pins)
says why. Running the app with `fastapi dev` needs `fastapi[standard]` on top.

## Add the feature to the app

In `main.py`:

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

The `auth` stub stands in for a real provider that reads the request; `tenant_id` and
`user_id` decide which conversations the caller sees ([Scope and access](scope-and-access.md)).
`app.db` is created in the working directory on first use, and one file carries everything:
the harness's runs and snapshots next to this package's conversations and turns. Delete the
file to start over.

The agent must declare its system text as `instructions=`, not the legacy `system_prompt=`.
Instructions are re-rendered per request, while a system prompt would be stored with every
turn and repeated when the history is concatenated.

Run it like any FastAPI app:

```bash
fastapi dev
```

The same app ships as `pydantic_ai_conversations_examples/quickstart.py`; start it with
`uv run -m pydantic_ai_conversations_examples.quickstart`.

## The first conversation over `/run`

`/run` answers with JSON, which makes the contract easy to see. The first turn needs only a
prompt and a conversation id. The id is yours to choose; the conversation is created when the
first run starts:

```bash
curl -s http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "My printer is offline.", "conversation_id": "c1"}'
```

The response carries `output`, `conversation_id`, `run_id`, and the run's `messages`. With
the test model the output is a canned string. Every later turn names the turn it builds on
through the `run_id` of the previous answer:

```bash
curl -s http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "It is a Brother.", "conversation_id": "c1", "extras": {"previous_run_id": "<run_id from the first answer>"}}'
```

The second run loads the first turn from the store and appends the new prompt, so the model
request holds both. Leaving `previous_run_id` out after the first turn answers 422 with the
code `conversations/previous-run-required`;
[Continuing a conversation](continuing-a-conversation.md#naming-the-previous-turn) says why.

List and read:

```bash
curl -s http://127.0.0.1:8000/conversations
curl -s http://127.0.0.1:8000/conversations/c1
```

The listing shows one entry with `message_count: 4`, two messages per turn. The detail lists
both turns with their `run_id`, `previous_run_id` (`null` on the first), and their `phase`,
here `answered` on both.

## The same over `/chat`

`/chat` speaks the Vercel AI SDK protocol and streams. The chat `id` is the conversation id,
and `previousRunId` is a top-level field of the body:

```bash
curl -N http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"id": "c2", "trigger": "submit-message", "messages": [{"id": "m1", "role": "user", "parts": [{"type": "text", "text": "My printer is offline."}]}]}'
```

The stream's `start` chunk carries `messageId`; that is the run id. The next request sends it
back:

```bash
curl -N http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"id": "c2", "trigger": "submit-message", "previousRunId": "<messageId from the start chunk>", "messages": [{"id": "m2", "role": "user", "parts": [{"type": "text", "text": "It is a Brother."}]}]}'
```

Both spellings bind on both endpoints: `previous_run_id` or `previousRunId` inside `/run`'s
`extras`, and at the top level of a `/chat` body. On `/chat` a rejected request answers 200
with the error inside the stream; see the [error codes](api-reference.md#error-codes).

## What you end up with

- Two capabilities on every run: `ConversationsCapability`, which loads the conversation and
  records the turn, and the harness's `StepPersistence`, which saves the snapshots.
- `GET /conversations`, `GET /conversations/{id}`, and `DELETE /conversations/{id}`, mounted
  behind the app's `auth`. Create and title routes are opt-in; see
  [The /conversations routes](http-api.md).
- A `previous_run_id` field in the request extras. `GET /config` lists it in the extras
  schema under its camelCase name, `previousRunId`.
- The registry opened on startup and closed on shutdown, and a `conversations` entry in the
  reports of `/readyz` and `/config/features`; see [Operations](operations.md).

Going deeper: [Continuing a conversation](continuing-a-conversation.md) for retries, edits,
and approvals; [Scope and access](scope-and-access.md) for multi-tenant and shared conversations;
[Backends](backends.md) for MongoDB and PostgreSQL.
