# Reference

Signatures, defaults, and wire shapes. For what these are *for*, start at
[Getting started](getting-started.md).

## `ConversationsFeature`

```python
from pydantic_ai_conversations import ConversationsFeature

ConversationsFeature(
    *,
    registry: ConversationRegistry,
    step_store: StepStore,
    scope: Iterable[str] = ('user.tenant_id', 'user.user_id'),
    enable_create_route: bool = False,
    enable_title_route: bool = False,
    filterable_attrs: Sequence[str] = (),
    on_delete: Sequence[OnDeleteHook] = (),
    snapshot_preparers: Sequence[SnapshotPreparer] = (),
    transcript_view: TranscriptView | None = None,
    ui_messages: UIMessageEncoder | None = None,
    max_history_runs: int | None = None,
    describe_failure: FailureDescriber | None = None,
    addons: Sequence[ConversationAddon] = (),
)
```

| Parameter | Effect |
|---|---|
| `registry` | Opened on startup, closed on shutdown, pinged for `/readyz` |
| `step_store` | The harness store holding runs and snapshots; wrapped in `PreparingStepStore` for writes |
| `scope` | The partition fields; see [Scope and access](scope-and-access.md). Invalid entries raise `ValueError` |
| `enable_create_route` | Mounts `POST /conversations` |
| `enable_title_route` | Mounts `PUT /conversations/{id}/title` |
| `filterable_attrs` | `attrs` keys exposed as optional list filters; a non-identifier, or a collision with a scope field or paging parameter, raises `ValueError` when the app is built |
| `on_delete` | Cleanup hooks for soft delete and purge |
| `snapshot_preparers` | Write-seam transforms, applied after `NewMessagesPreparer` |
| `transcript_view` | Shapes the turns `GET /conversations/{id}` shows |
| `ui_messages` | Encoder for `TranscriptTurnOut.ui_messages` |
| `max_history_runs` | Caps the turns loaded per run; `<= 0` raises `ValueError` when the app is built |
| `describe_failure` | Maps a failed run's exception to the stored `TurnError` |
| `addons` | Add-ons: `SupportsTurnFields` ones fill `TranscriptTurnOut.addons`, `SupportsRouters` ones mount routes, `SupportsLifespan` ones open and close with the registry, `SupportsDeletion` ones run with the delete hooks; a duplicate or non-slug name raises `ValueError` when the app is built |

Members: `name = 'conversations'`; `extras_type`, a pydantic model with `previous_run_id` plus
one required field per `extras.*` scope entry; `capabilities()`, `routers(auth=)`,
`lifespan(app)`, `health()`, `config()` as the app factory's `Feature` defines them;
`namespace_for(user, extras={})`; `async purge(*, namespace_key, conversation_id)`.

`config()` returns `{'create_route': bool, 'title_route': bool, 'ui_messages': bool,
'filterable_attrs': list[str], 'addons': {name: {}}}`.

## `ConversationsCapability`

```python
ConversationsCapability[DepsT](
    registry: ConversationRegistry,
    step_store: StepStore,
    scope: tuple[ScopeField, ...],
    describe_failure: FailureDescriber | None = None,
    max_history_runs: int | None = None,
)
```

`DepsT` must satisfy `SupportsConversationDeps`: a `user: HasUserId` property and an
`extras: Mapping[str, Any]` property. The app factory's `AppDeps` does. The capability must
precede `StepPersistence(capture_frontier=True)` on the agent; it only reads from `step_store`.

## `build_conversation_router`

```python
build_conversation_router(
    *,
    registry: ConversationRegistry,
    step_store: StepStore,
    scope: tuple[ScopeField, ...],
    auth: UserDependency,
    enable_create: bool = False,
    enable_title: bool = False,
    filterable_attrs: Sequence[str] = (),
    on_delete: Sequence[OnDeleteHook] = (),
    transcript_view: TranscriptView | None = None,
    ui_messages: UIMessageEncoder | None = None,
    addons: Sequence[ConversationAddon] = (),
) -> APIRouter
```

Returns an `APIRouter` with prefix `/conversations` and tag `conversations`. `auth` is any
FastAPI dependency resolving to an object with `user_id` and the `user.*` scope attributes;
pass the exact callable the app guards with.

## Routes

| Method and path | Query | Body | Response |
|---|---|---|---|
| `GET /conversations` | `pageAfter?`, `pageSize=100` (1..500), one optional parameter per `filterable_attrs` entry, scope fields | | 200 `ConversationList` |
| `GET /conversations/{conversation_id}` | scope fields | | 200 `ConversationDetail`; 404 |
| `DELETE /conversations/{conversation_id}` | scope fields | | 204; 404 |
| `POST /conversations` | scope fields | `{"attrs": {str: str}}`, optional | 201 `ConversationOut` |
| `PUT /conversations/{conversation_id}/title` | scope fields | `{"title": str}`, 1 to 200 characters after stripping | 204; 404; 422 |

Every route answers 503 when the registry is unavailable, the detail route also when the step
store reports a temporary failure, and every route 422 when a scope parameter is missing or
empty.

## Wire models

```
ConversationOut      {conversation_id, title: str | null, created_by, created_at, last_message_at: datetime | null, message_count: int, attrs: {str: str}}
ConversationList     {data: [ConversationOut], more_after: str | null}
ConversationDetail   {conversation: ConversationOut, turns: [TranscriptTurnOut], messages: [ModelMessage]}
TranscriptTurnOut    {run_id, started_at, phase: 'started' | 'answered' | 'failed', previous_run_id: str | null,
                      author_user_id: str | null, error: TurnErrorOut | null, messages: [ModelMessage], ui_messages: [any] | null,
                      addons: {name: object | null}}
                     (`error`, `ui_messages` and `addons` are always present; `addons` is `{}` without add-ons)
TurnErrorOut         {code, message, retryable: bool}
```

`messages` are pydantic-ai messages as `ModelMessagesTypeAdapter` dumps them. `ui_messages` is
the encoder's output dumped `by_alias=True`.

## Error codes

| Code | Where | Status | Retryable | When |
|---|---|---|---|---|
| `conversations/not-found` | run path, routes | 404 | no | The id is unknown, foreign, or deleted |
| `conversations/run-not-found` | run path | 404 | no | `previous_run_id` names no turn of this conversation |
| `conversations/previous-run-required` | run path | 422 | no | `previous_run_id` left out after the first turn |
| `conversations/prompt-required` | run path (`/chat` or `agent.run`; `/run` requires a prompt) | 422 | no | No prompt and nothing to resume |
| `conversations/unavailable` | run path, routes | 503 | yes | The registry or the step store is temporarily unreachable |
| `conversations/cancelled` | stored `TurnError` only | | yes | The run was cancelled; written by `agent_error_mapper` |

The run path raises the app factory's `AgentError`, rendered as its
[error contract](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/error-contract.md)
describes: the status and a flat body on `/run`, HTTP 200 with a `data-error` chunk inside the
stream on `/chat`. The routes raise `HTTPException` with `{code, message, retryable?}` as
`detail`.

## Registry types

```python
RunPhase = Literal['started', 'answered', 'failed']


@dataclass(frozen=True)
class ConversationMeta:
    namespace_key: str
    conversation_id: str
    created_by: str
    created_at: datetime
    last_message_at: datetime | None = None
    message_count: int = 0
    title: str | None = None
    deleted_at: datetime | None = None
    attrs: Mapping[str, str] = <empty, frozen on construction>


@dataclass(frozen=True)
class TurnError:
    code: str
    message: str
    transient: bool = False


@dataclass(frozen=True)
class TurnRecord:
    run_id: str
    author_user_id: str
    at: datetime = <now, UTC>
    phase: RunPhase = 'started'
    previous_run_id: str | None = None
    error: TurnError | None = None
```

### `ConversationRegistry`

| Method | Contract |
|---|---|
| `open()` | Acquire resources; optional, methods connect lazily; may be called again after `close()` |
| `close()` | Release them; methods raise `RegistryUnavailable` until the next `open()` |
| `ping() -> Mapping[str, Any]` | Cheap; raising means unhealthy; `{'ok': True}` and nothing data-derived; the key `step_store` is reserved |
| `register_conversation(ns, id, *, created_by, attrs=None) -> ConversationMeta` | Create, or return the live entry as is; atomic global claim; foreign, deleted, or purged raises `ConversationNotFound` |
| `get(ns, id) -> ConversationMeta \| None` | `None` for missing, foreign, or deleted; never raises `ConversationNotFound` |
| `list(ns, *, attrs=None, page_after=None, limit=100) -> Sequence[ConversationMeta]` | Live entries, newest activity first; `attrs` is an AND equality filter; id cursor; `limit <= 0` raises `ValueError` |
| `set_title(ns, id, title)` | `ConversationNotFound` for unknown, foreign, or deleted |
| `update_message_stats(ns, id, *, last_message_at, message_count_delta)` | Overwrite the timestamp, add the delta atomically; `ConversationNotFound` for unknown, foreign, or deleted |
| `soft_delete(ns, id) -> ConversationMeta` | Mark and return; a second call raises; the id stays claimed |
| `start_turn(ns, id, *, run_id, author_user_id, previous_run_id=None)` | Record a `started` turn; `previous_run_id` stored as given; `ConversationNotFound` as above |
| `set_turn_phase(ns, id, *, run_id, phase, error=None)` | Idempotent single-row update; unknown run is a no-op; `error` overwrites, `None` included; `ConversationNotFound` as above |
| `list_turns(ns, id) -> Sequence[TurnRecord]` | In `start_turn` order; `ConversationNotFound` as above |
| `purge(ns, id) -> bool` | Tombstone the entry, delete its turns; id stays claimed forever; returns whether `ns` owned it (soft-deleted counts), `True` again on repeat |

`InMemoryConversationRegistry()` implements it for tests and single-process use.

## Scope

```python
@dataclass(frozen=True)
class ScopeField:
    source: Literal['user', 'extras']
    name: str

parse_scope(spec: Iterable[str]) -> tuple[ScopeField, ...]
build_namespace(scope: tuple[ScopeField, ...], user: HasUserId, extras: Mapping[str, Any]) -> str
scope_extras_model(scope: tuple[ScopeField, ...]) -> type[BaseModel] | None
```

`HasUserId` is a protocol with a `user_id: str` property.

## Stores

```python
class PreparingStepStore:  # satisfies StepStore by delegation
    def __init__(self, inner: StepStore, preparers: Sequence[SnapshotPreparer] = ()) -> None

class SnapshotPreparer(Protocol):
    def prepare_snapshot(self, snapshot: ContinuableSnapshot) -> ContinuableSnapshot

class NewMessagesPreparer:  # the feature's own first preparer

@runtime_checkable
class SupportsRunDeletion(Protocol):
    async def delete_runs(self, *, run_ids: Sequence[str]) -> None

@runtime_checkable
class SupportsPing(Protocol):
    async def ping(self) -> Mapping[str, Any]

@runtime_checkable
class SupportsTransientErrors(Protocol):
    transient_errors: tuple[type[Exception], ...]
```

`PreparingStepStore` runs the chain on `save_snapshot`, skips a save whose snapshot the chain
emptied, and forwards every other `StepStore` method. A failure the wrapped store lists in
`transient_errors` is raised as the retryable `conversations/unavailable` `AgentError`, so a
database outage during a run reaches the caller as 503; any other exception, an `AgentError`
included, passes through unchanged. The shipped stores list their driver's connection and
timeout errors. Attributes outside the protocol pass through untranslated.

## Transcript and hooks

```python
@dataclass(frozen=True)
class TranscriptTurn:
    run_id: str
    started_at: datetime
    phase: RunPhase
    messages: list[ModelMessage]
    author_user_id: str | None = None
    previous_run_id: str | None = None
    error: TurnError | None = None

TranscriptView = Callable[[list[TranscriptTurn]], list[TranscriptTurn]]
UIMessageEncoder = Callable[[Sequence[ModelMessage]], Sequence[Any]]
AddonTurnQuery(namespace_key, conversation_id, user: HasUserId, turns: Sequence[TranscriptTurn])
ConversationAddon   Protocol: name: str (read-only)
SupportsTurnFields  Protocol (runtime_checkable): turn_model: type[BaseModel] (read-only),
                    async load_turns(query: AddonTurnQuery, /) -> Mapping[run_id, BaseModel]
SupportsRouters Protocol (runtime_checkable): routers(guard: AddonRouteGuard, /) -> Sequence[APIRouter]
SupportsLifespan Protocol (runtime_checkable): async open(), async close()
SupportsDeletion Protocol (runtime_checkable): async on_delete(event: DeletionEvent)
AddonCaller(user: HasUserId, namespace_key: str)
AddonRouteGuard(registry, scope, auth): caller() -> FastAPI dependency yielding AddonCaller,
                    async require_conversation(namespace_key, conversation_id) -> ConversationMeta,
                    async require_turn(namespace_key, conversation_id, run_id) -> TurnRecord,
                    not_found() / unavailable() -> HTTPException
FailureDescriber = Callable[[BaseException], TurnError | None]
UserDependency = Callable[..., HasUserId | Awaitable[HasUserId]]

load_transcript(step_store, *, conversation_id, turns=None, transcript_view=None) -> list[TranscriptTurn]
agent_error_mapper(error_handlers: ErrorHandlers | None = None) -> FailureDescriber

DeletionKind = Literal['soft_delete', 'purge']

@dataclass(frozen=True)
class DeletionEvent:
    namespace_key: str
    conversation_id: str
    kind: DeletionKind
    meta: ConversationMeta | None

OnDeleteHook = Callable[[DeletionEvent], Awaitable[None]]
```

`StripRejectedDrafts()` is a capability; `strip_rejected_drafts(messages)` and
`rejected_draft_pairs(messages)` are the functions behind it.

## Errors

`ConversationNotFound` and `RegistryUnavailable`, both plain `Exception` subclasses, in
`pydantic_ai_conversations.errors`.

## Backends

```python
from pydantic_ai_conversations.sqlite import SqliteConversationRegistry, SqliteConversationStepStore
SqliteConversationRegistry(database: str | Path)
SqliteConversationStepStore(database: str | Path, *, media_store='auto', media_threshold_bytes=65536,
                            max_snapshots_per_run=None)  # plus `async close()`

from pydantic_ai_conversations.mongo import MongoConversationRegistry, MongoConversationStepStore
MongoConversationRegistry(*, client: AsyncMongoClient | None = None, db_url: str | None = None, database: str)
MongoConversationStepStore(*, client=None, db_url=None, database: str, media_store='auto',
                           media_threshold_bytes=65536, max_snapshots_per_run=None)

from pydantic_ai_conversations.postgres import (
    PostgresConversationRegistry, PostgresConversationStepStore, SCHEMA_SQL, create_schema,
)
PostgresConversationRegistry(*, pool: AsyncConnectionPool | None = None, dsn: str | None = None,
                             schema: str = 'public', create_schema: bool = False)
PostgresConversationStepStore(pool: PostgresPool, *, table: str = 'step_persistence', media_store='auto',
                              media_threshold_bytes=65536, max_snapshots_per_run=None)
create_schema(pool: AsyncConnectionPool, schema: str = 'public') -> None
```

Passing both or neither of the paired connection arguments raises `ValueError`.

## Testing

`pydantic_ai_conversations.testing.RegistryContract`: a pytest class; override the `registry`
fixture.

## Package exports

From `pydantic_ai_conversations`: `ConversationsFeature`, `ConversationsCapability`,
`SupportsConversationDeps`, `FailureDescriber`, `agent_error_mapper`, `StripRejectedDrafts`,
`strip_rejected_drafts`, `rejected_draft_pairs`, `TranscriptTurn`, `TranscriptView`,
`load_transcript`, `build_conversation_router`, `ConversationDetail`, `ConversationList`,
`ConversationOut`, `TranscriptTurnOut`, `TurnErrorOut`, `ConversationAddon`, `SupportsTurnFields`,
`SupportsRouters`, `SupportsLifespan`, `SupportsDeletion`, `AddonTurnQuery`,
`AddonRouteGuard`, `AddonCaller`, `DeletionEvent`, `DeletionKind`,
`OnDeleteHook`, `UIMessageEncoder`, `UserDependency`, `ConversationNotFound`,
`RegistryUnavailable`, `ConversationMeta`, `ConversationRegistry`,
`InMemoryConversationRegistry`, `RunPhase`, `TurnError`, `TurnRecord`, `HasUserId`,
`ScopeField`, `build_namespace`, `parse_scope`, `scope_extras_model`, `NewMessagesPreparer`,
`PreparingStepStore`, `SnapshotPreparer`, `SupportsPing`, `SupportsRunDeletion`,
`SupportsTransientErrors`.

The backends and `RegistryContract` are imported from their submodules. Private modules
(underscore-prefixed) are not part of the API.
