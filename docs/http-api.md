# The /conversations Routes

This guide covers the management routes the feature mounts, what a frontend reads from them,
and how it rebuilds its state from a transcript. Wire shapes and status codes in full are in
the [reference](api-reference.md#wire-models).

Prerequisites: [Getting started](getting-started.md). The chat and run endpoints themselves
belong to the app factory; see its
[Agent endpoints](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/agent-endpoints.md).

## Routes

| Route | Mounted | Answers |
|---|---|---|
| `GET /conversations` | always | A page of the caller's conversations, newest activity first |
| `GET /conversations/{id}` | always | The entry plus its transcript |
| `DELETE /conversations/{id}` | always | 204; soft-deletes the entry |
| `POST /conversations` | `enable_create_route=True` | 201 with the new entry; the only route that sets `attrs` |
| `PUT /conversations/{id}/title` | `enable_title_route=True` | 204; sets a title |

All routes sit behind the app's `auth`, under the `conversations` tag. Every `extras.*` scope
field is a required query parameter on each of them, in snake_case; see
[Scope and access](scope-and-access.md#client-side-scope-fields).

The create and title routes are opt-in because who may create or rename is a product decision.
`GET /config/features`, which needs auth, reports both switches:

```json
{"feature_config": {"conversations": {"create_route": true, "title_route": false, "ui_messages": false, "filterable_attrs": []}}}
```

A client reads them to hide a rename it is not allowed to perform instead of probing. There is
no hook that runs after an answer, so a title generated from the first exchange is set by your
own code, through this route or `registry.set_title`.

## Listing

`GET /conversations?pageSize=100&pageAfter=<conversation_id>` returns
`{"data": [...], "more_after": <id or null>}`. `pageSize` is 1 to 500. Pass `more_after` back
as `pageAfter` for the next page; an unknown or since-deleted cursor ends the list.

Each entry carries `conversation_id`, `title`, `created_by`, `created_at`, `last_message_at`,
`message_count`, and `attrs`. `message_count` counts stored messages including tool traffic,
and a failed turn counts what it stored before failing, so treat it and `last_message_at` as a
listing digest, not a figure to display as "N messages".

Declared `attrs` keys become optional query filters:

```python
ConversationsFeature(..., filterable_attrs=('project',))
```

`GET /conversations?project=atlas` then returns entries whose `attrs["project"]` equals
`atlas`; several filters combine with AND, and an empty value is a 422. An undeclared key is
not queryable. A name that is not an identifier, or collides with a scope field or with
`pageAfter`/`pageSize`, raises `ValueError` when the app is built.

The cursor is the conversation id alone, so a page can skip or repeat a row that changed
position while the client paged. That is accepted; the same rule holds on every backend.

## Reading a transcript

`GET /conversations/{id}` returns the entry, a `turns` list, and a flat `messages` list.

`turns` is canonical. It lists every turn of every branch in run-start order, each with:

- `run_id`, `started_at`, `author_user_id`;
- `previous_run_id`, the link a client follows to render one branch;
- `phase`: render `answered` normally, `failed` with a retry affordance, `started` as in
  flight. A run killed hard keeps `started` forever, so a `started` turn whose `started_at`
  is older than any run could last is a failed one;
- `error`, `null` unless the turn failed and the feature stores reasons
  ([Extending](extending.md#storing-why-a-turn-failed)); then `code`, `message`, `retryable`;
- `messages`, the turn's stored messages in pydantic-ai's JSON form;
- `ui_messages`, the same messages in the app's UI protocol, `null` unless the feature was
  built with an encoder;
- `addons`, per-turn data from the feature's add-ons keyed by add-on name, `{}` with none
  configured; a name maps to `null` when its add-on has no entry for the turn, and an add-on
  without turn fields has no key.

`messages` on the detail is the branch ending in the latest turn, flattened: the latest turn
and its ancestors, in order. It is a compatibility field for callers that want one list; it
carries no phase and cannot show branches.

Delegate runs, helper agents spawned inside a tool call, do not appear as turns. A turn whose
run saved no snapshot does not appear either.

### Rebuilding a chat frontend

Wire a UI adapter's `dump_messages` as the encoder, pinned to the SDK version the app
streams:

```python
from functools import partial

from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai_conversations import ConversationsFeature

feature = ConversationsFeature(
    registry=..., step_store=..., ui_messages=partial(VercelAIAdapter.dump_messages, sdk_version=6)
)
```

On reload the frontend fetches the detail, takes the latest turn, follows `previous_run_id`
back to the root, and seeds its state with those turns' `ui_messages` in order. It reads
`ui_messages` per turn rather than the flat `messages` because only the turn carries `phase`:
a `failed` turn gets a retry affordance, a `started` turn renders as in flight. A run in flight
cannot be re-attached to; its answer appears as an `answered` turn on the next fetch.
`GET /config/features` reports `ui_messages: true` when an encoder is wired. The result
carries the protocol's camelCase aliases, `toolCallId` for example.

### Shaping the transcript

A `transcript_view` sees every resolved turn and returns the turns to show, for example hiding
a failed attempt once a sibling answered. See [Extending](extending.md#transcript-views).

### Per-turn data from add-ons

An add-on joins its own records to the transcript by `run_id` and the detail shows them under
`turns[].addons.<name>`, typed per add-on in the OpenAPI schema. `GET /config/features` lists
every configured add-on under `addons`, whether or not it has turn fields. A turn streamed in
the current session carries no add-on data until the client fetches the detail again. See
[Extending](extending.md#add-ons); the shipped [turn feedback add-on](feedback.md) is one.

## Deleting

`DELETE /conversations/{id}` marks the entry deleted and hides it. Messages stay in the store;
a hard delete is a Python call. Both are covered in
[Deleting conversations](deleting.md).

## Attrs and the pre-create pattern

`attrs` are string pairs set when the entry is created and never changed afterwards. The
implicit create on a first run sets none. To attach them, create the entry before the first
run. `POST /conversations` with `{"attrs": {"project": "atlas"}}` creates it with a
server-generated id, returned as `conversation_id`, which the client then uses as the chat
`id`. Server-side code can choose the id itself:

```python
namespace = feature.namespace_for(user, extras)
await registry.register_conversation(namespace, conversation_id, created_by=user.user_id, attrs={'project': 'atlas'})
```

The first run then finds the entry and keeps its `attrs`.
`pydantic_ai_conversations_examples/ticket_attrs.py` is a runnable app that creates each entry
this way from a support ticket, with the create route left off so a client cannot choose `attrs`.

## Errors

The routes raise FastAPI's `HTTPException` with a structured `detail`:

| Status | `detail` |
|---|---|
| 404 | `{"code": "conversations/not-found", "message": "The conversation does not exist."}` |
| 503 | `{"code": "conversations/unavailable", "message": "Conversation storage is temporarily unavailable.", "retryable": true}` |
| 422 | FastAPI's validation shape, for a missing scope parameter or an invalid body |

A foreign or already-deleted id answers 404 like a missing one. 503 comes from the registry on
every route, and on the detail route also from a step store that reports a temporary failure.
The run path renders its codes flat, as the app factory's `AgentError` does, without the
`detail` wrapper; the codes are listed in the [reference](api-reference.md#error-codes).

What you end up with: a listing, a transcript a frontend can seed from, and switches the
client reads from `/config/features`. Going deeper: [Deleting conversations](deleting.md).
