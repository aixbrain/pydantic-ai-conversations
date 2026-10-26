# Extending

This guide shows you the seams the feature leaves open: your own registry or step store,
transforms on what is stored and shown, add-ons, the stored failure reason, and use without the
app factory.

Prerequisites: [How a conversation is stored](concepts.md). Signatures are in the
[reference](api-reference.md).

## Your own registry

`ConversationRegistry` is a protocol; any object with its methods works. The methods and
their contracts are listed in the [reference](api-reference.md#conversationregistry). Beyond
the signatures, an implementation owes these guarantees:

- Every method takes a namespace key and answers only within it. A foreign id is answered
  like a missing one: `ConversationNotFound` raised with the conversation id as its only
  argument, `None` from `get`, or absence from a list. A soft-deleted entry counts as missing
  on every method but `purge`.
- `register_conversation` claims the global id atomically, so concurrent registers of one id
  produce a single winner and everyone gets the same entry.
- `update_message_stats` adds its delta atomically; a read-modify-write loses increments.
- A soft-deleted entry is hidden and cannot be revived. A purged id stays claimed, and a purge
  of a soft-deleted entry still reports it as owned.
- `list` returns a list sorted by `(last_message_at or created_at, conversation_id)`
  descending and pages on the id; an unknown cursor ends the list, and `limit <= 0` raises
  `ValueError`.
- `list_turns` returns turns in `start_turn` order. `set_turn_phase` is an idempotent
  single-row update and ignores an unknown run.
- Timestamps are returned timezone-aware in UTC.
- Methods work before `open()` is called, by connecting lazily; after `close()` they raise
  `RegistryUnavailable` until `open()` is called again.
- Backend failures become `RegistryUnavailable`; its message goes to the log only, so it may
  carry connection detail. `ConversationNotFound` and `ValueError` pass through untranslated.
  Any other exception is not caught: it fails the request, or fails a run that had already
  answered.

The conformance suite checks all of these but the last. Subclass it in your test suite and
provide the fixture:

```python
import pytest

from pydantic_ai_conversations.testing import RegistryContract


class TestMyRegistry(RegistryContract):
    @pytest.fixture
    def registry(self):
        return MyRegistry(...)
```

It needs pytest and pytest-asyncio in auto mode, and it fires twenty concurrent calls at one
instance, so the fixture's connection limit must absorb that. Hand it an empty store.
`InMemoryConversationRegistry` passes the suite and shows the semantics, but it is atomic only
because nothing awaits between read and write; a real backend must use atomic statements.

New registry operations never extend the protocol itself. They ship as a separate
`runtime_checkable` protocol that callers detect with `isinstance`, so an app's own registry
keeps satisfying `ConversationRegistry` after an upgrade.

## Your own step store

Any harness `StepStore` works. Two optional methods and one attribute are discovered
structurally:

```python
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai_harness.step_persistence import SqliteStepStore


class MyStepStore(SqliteStepStore):
    transient_errors = (sqlite3.OperationalError, OSError)  # the driver's temporary failures

    async def delete_runs(
        self, *, run_ids: Sequence[str]
    ) -> None: ...  # remove everything the store keeps per run; unknown ids are a no-op

    async def ping(self) -> Mapping[str, Any]:
        return {'ok': True}
```

`delete_runs` lets a purge remove the conversation's runs. It must be idempotent, because a
purge may be retried, and it must cover every table the store keeps per run: run records,
events, snapshots, tool effects, and the harness's record of saved snapshot keys, which
outlives snapshot pruning and would otherwise keep the purged run's keys. A store that keeps
snapshot messages apart from the snapshots, shared between runs, must also delete the ones no
other run's snapshot references; `SqliteStepStore` does that itself by trigger. `ping` must be cheap and public:
`/readyz` is unauthenticated, so report status only, no counts. `transient_errors` names the
exceptions that mean the database is temporarily unreachable; during a run and on the transcript
read they become the retryable `conversations/unavailable`, and every other exception passes
through, because a run id already stored or a malformed row fails the same way on every retry.
The `isinstance` check sees that the names exist, not their signatures; a wrong signature fails
at call time.

## Snapshot preparers

A preparer transforms every snapshot before it is saved. The feature wires its own first,
which cuts the snapshot to the run's turn, so yours sees the turn only:

```python
from dataclasses import replace

from pydantic_ai.messages import ToolReturnPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot

from pydantic_ai_conversations import ConversationsFeature


class TruncateToolReturns:
    """Keep oversized tool output out of the stored turn."""

    def prepare_snapshot(self, snapshot: ContinuableSnapshot) -> ContinuableSnapshot:
        messages = [
            replace(
                message,
                parts=[
                    replace(part, content=part.content[:4000] + '...')
                    if isinstance(part, ToolReturnPart) and isinstance(part.content, str) and len(part.content) > 4000
                    else part
                    for part in message.parts
                ],
            )
            for message in snapshot.messages
        ]
        return replace(snapshot, messages=messages)


feature = ConversationsFeature(registry=..., step_store=..., snapshot_preparers=[TruncateToolReturns()])
```

A preparer must be idempotent, key any per-run state by `run_id`, and rebuild messages with
`replace(message, parts=...)` so the `run_id` each message carries survives. It must never
remove the run's own user prompt, tool returns, or retry prompts that answer a tool call,
since the next turn is loaded from what it leaves behind. The same holds for parts a provider needs intact on resume, such as thinking
blocks beside a tool call. A snapshot emptied by the chain is not saved at all.

This is the one place stored content is shaped. What the preparers leave in the store is what
every reader sees afterwards; snapshots are never rewritten later.

## Transcript views

A view shapes what `GET /conversations/{id}` shows. It receives every resolved turn of the
conversation in run-start order, all branches included, and returns the turns to show:

```python
from pydantic_ai_conversations import ConversationsFeature, TranscriptTurn


def hide_superseded_failures(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
    """Hide a failed turn once a sibling built on the same predecessor has answered."""
    answered_after = {t.previous_run_id for t in turns if t.phase == 'answered'}
    return [t for t in turns if not (t.phase == 'failed' and t.previous_run_id in answered_after)]


feature = ConversationsFeature(registry=..., step_store=..., transcript_view=hide_superseded_failures)
```

A turn left out is omitted; `dataclasses.replace` rewrites one. The view affects reads only,
never what the model loads.

## UI message encoders

`ui_messages` takes any callable from a turn's messages to the app's UI protocol. A UI
adapter's `dump_messages` fits, wrapped in `functools.partial` to pin its SDK version to the
one the app streams. The feature calls it per turn and JSON-encodes the result; it never
inspects the protocol. See [Rebuilding a chat frontend](http-api.md#rebuilding-a-chat-frontend).

## Add-ons

An add-on is composed into the feature through `addons=`. It has a name, and whichever of the
optional capabilities it implements, each a protocol the feature detects with `isinstance`:
`SupportsTurnFields` for per-turn data on the detail read, `SupportsRouters` for routes of
its own. The name is the key under `addons`, what `config()` reports, and must be a unique slug.

### Turn fields

An add-on contributes per-turn data to `GET /conversations/{id}` from its own table. It
declares one turn's entry as a pydantic model and loads the entries for a read in one query
keyed by `run_id`:

```python
from collections.abc import Mapping

from pydantic import BaseModel

from pydantic_ai_conversations import AddonTurnQuery, ConversationsFeature


class FeedbackOut(BaseModel):
    rating: int
    comment: str | None = None


class FeedbackAddon:
    name = 'feedback'
    turn_model = FeedbackOut

    def __init__(self, db: FeedbackTable) -> None:
        self._db = db

    async def load_turns(self, query: AddonTurnQuery) -> Mapping[str, BaseModel]:
        rows = await self._db.by_run_ids(query.namespace_key, [t.run_id for t in query.turns])
        return {row.run_id: FeedbackOut(rating=row.rating, comment=row.comment) for row in rows}


feature = ConversationsFeature(registry=..., step_store=..., addons=[FeedbackAddon(db)])
```

The loader runs after `transcript_view`, over the turns the response shows, with the caller in
`query.user` for entries that depend on who reads. A turn missing from the result is sent as
`null`; a raising loader fails the request, because `null` already means "no entry". An add-on
that declares `transient_errors` (as a step store does) gets those mapped to the retryable 503
instead. Stored snapshots are never rewritten, so joining at read time is the way per-turn data
reaches the transcript. The narrowed response models keep the base names in OpenAPI; a second
conversation router in the same app gets module-qualified names for them.

### Routes

An add-on that implements `routers(guard)` gets its routers mounted behind the app's auth next to
the conversation router. `guard` is an `AddonRouteGuard`: `guard.caller()` is a FastAPI
dependency resolving the user and their namespace, and `guard.require_conversation` /
`guard.require_turn`
apply the namespace wall with the package's own 404 and 503, so an add-on does not rebuild it:

```python
from typing import Annotated

from fastapi import APIRouter, Depends

from pydantic_ai_conversations import AddonCaller, AddonRouteGuard


class FeedbackAddon:  # the class from above, with routes
    name = 'feedback'
    turn_model = FeedbackOut

    def routers(self, guard: AddonRouteGuard) -> list[APIRouter]:
        router = APIRouter(prefix='/conversations')

        @router.put('/{conversation_id}/turns/{run_id}/feedback', status_code=204)
        async def rate(
            conversation_id: str,
            run_id: str,
            body: FeedbackOut,
            caller: Annotated[AddonCaller, Depends(guard.caller())],
        ) -> None:
            turn = await guard.require_turn(caller.namespace_key, conversation_id, run_id)
            await self._db.put(caller.namespace_key, conversation_id, turn.run_id, caller.user.user_id, body)

        return [router]
```

## Storing why a turn failed

`describe_failure` maps a run's exception to the `TurnError` stored on its turn. The mapper
built from the app's error handlers stores the same code and message the client saw live:

```python
from pydantic_ai_conversations import ConversationsFeature, agent_error_mapper

feature = ConversationsFeature(registry=..., step_store=..., describe_failure=agent_error_mapper(error_handlers))
```

Pass the same `error_handlers` as `create_agent_app`; without them every app-specific error
is stored as `internal`. A cancelled run is stored as `conversations/cancelled`. A mapper that
raises is logged and stores no reason. Without a mapper the turn carries `phase='failed'` and
nothing else.

## Without the app factory

The capability and the router work on a plain agent and a plain FastAPI app. The deps object
needs a `user` with a `user_id` and an `extras` mapping; the scope is parsed by hand:

```python
from fastapi import FastAPI
from pydantic_ai import Agent
from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepPersistence

from pydantic_ai_conversations import (
    ConversationsCapability,
    InMemoryConversationRegistry,
    NewMessagesPreparer,
    PreparingStepStore,
    build_conversation_router,
    parse_scope,
)

scope = parse_scope(('user.org_id',))
registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
agent = Agent(
    'test',
    deps_type=MyDeps,
    capabilities=[
        ConversationsCapability[MyDeps](registry=registry, step_store=store, scope=scope),
        StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
    ],
)
app = FastAPI()
app.include_router(build_conversation_router(registry=registry, step_store=store, scope=scope, auth=current_user))
```

Three things the feature does for you have to be kept by hand:

- The capability goes before `StepPersistence`, so the prompt is stamped before the harness
  saves it. `capture_frontier=True` makes the harness save the request before the first model
  call.
- `StepPersistence` takes no `agent_name` and no `run_id`, so its run id stays the one the
  registry keys on.
- `auth` is the exact callable the app guards with, because FastAPI caches a dependency per
  callable and a second one runs the provider twice.

The run path raises the app factory's `AgentError`, the seam included when the store reports a
temporary failure; a standalone app needs a handler for it, while the routes raise
`HTTPException` and need none.

What you end up with: a backend the conformance suite accepts, and the stored and shown
content shaped at the places meant for it. Going deeper: [Testing](testing.md).
