# Turn feedback add-on

Feedback on a turn, with a payload the app defines, stored in the add-on's own table and shown
inline on `GET /conversations/{id}` under `turns[].addons.<name>`.

Prerequisites: [Add-ons](extending.md#add-ons) for how an add-on is composed into the feature.

## Setup

```python
from pydantic import BaseModel

from pydantic_ai_conversations import ConversationsFeature
from pydantic_ai_conversations.feedback import TurnFeedbackAddon
from pydantic_ai_conversations.sqlite import (
    SqliteConversationRegistry,
    SqliteConversationStepStore,
    SqliteTurnFeedbackStore,
)


class Report(BaseModel):
    category: str
    remarks: str = ''


feature = ConversationsFeature(
    registry=SqliteConversationRegistry('app.db'),
    step_store=SqliteConversationStepStore('app.db'),
    addons=[TurnFeedbackAddon(payload_model=Report, store=SqliteTurnFeedbackStore('app.db'), name='report')],
)
```

`payload_model` is validated on every write and read; the store holds JSON and has no schema of
its own, so changing the model is the app's migration: a stored entry the current model rejects
is logged and left out (`null` on the turn, absent from the list) rather than failing the read.
`name` is the key under `addons`, the last segment of the routes, and the `kind` the store files
entries under: two add-ons with different names (a report form and a thumbs rating) share one
table.

## Routes

- `PUT /conversations/{id}/turns/{run_id}/<name>`, body = the payload: creates or replaces the
  caller's own entry and returns it (`run_id`, `author_user_id`, `created_at`, `updated_at`,
  `data`).
- `DELETE` on the same path removes the caller's entry; 204 whether or not one existed.
- `GET /conversations/{id}/<name>` lists every author's entries for the conversation, oldest
  first, each naming its `run_id`.

A conversation outside the caller's namespace, or a `run_id` that is not one of its turns, is a
404. The detail read shows the caller's own entry per turn and `null` where there is none; a
turn streamed in the current session shows its entry on the next fetch. Entries are deleted
with a purge and kept through a soft delete ([Deleting](deleting.md#cleanup-hooks)). A store
failure the store declares temporary is the retryable 503 the conversation routes use, and the
store's `ping` is nested under the add-on's name in `/readyz`
([Operations](operations.md#startup-readiness-config)).

## Stores

`InMemoryTurnFeedbackStore` for tests; `SqliteTurnFeedbackStore(path)`,
`MongoTurnFeedbackStore(client=|db_url=, database=)`, and
`PostgresTurnFeedbackStore(pool=|dsn=, schema=, create_schema=)` take the same handle the
registry takes, so the feedback table lands next to the conversations
([Backends](backends.md)). The Postgres DDL ships as `feedback.sql` with
`create_feedback_schema` for tests. A store of your own implements `TurnFeedbackStore` and
passes `testing.TurnFeedbackStoreContract`.
