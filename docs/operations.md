# Operations

This guide covers what the feature reports at runtime, how it behaves under concurrency, what
it logs, its known limits, and why its dependencies are pinned the way they are.

Prerequisites: [Getting started](getting-started.md).

## Startup, readiness, config

The feature opens the registry on startup and closes it on shutdown through the app's
lifespan. The step store is not opened or closed by the feature; who closes a pool or client
the store borrows is covered per backend in [Backends](backends.md).

`/readyz` is unauthenticated and nests each feature's report:

```json
{"status": "ok", "reports": {"conversations": {"ok": true, "step_store": {"ok": true}}}}
```

The `conversations` report is the registry's `ping()`, plus a `step_store` key when the
store supports `ping`. Reports carry status only and no counts, because the endpoint is
public. A `ping` that raises marks the app not ready. Every shipped `ping` runs a statement
against the database, so an outage after startup shows.

`/config/features` requires auth and reports the switches a client branches on:
`create_route`, `title_route`, `ui_messages` (whether an encoder is wired), `filterable_attrs`,
and `addons` (the configured add-ons by name, each mapping to its own switches). Scope, history
limits, and hooks are not reported.

## Concurrency

Registry operations are individually atomic: the id claim, the stats increment, and the
phase update. There is no version check and no lock across a turn. Two runs building on the
same turn at once both succeed and become siblings; the design branches instead of detecting
conflicts. `last_message_at` is last-write-wins.

A delete racing a run does not fail the run: the answer is stored, the stats update is
skipped and logged. A registry outage after the model answered never fails the run either;
the snapshots already landed, and the listing shows stale activity until the next update.

The shipped SQLite registry serialises its work on one connection and a lock; the Postgres
registry runs each call at `READ COMMITTED`; the MongoDB registry relies on per-operation
atomicity.

## Logging

The package uses the standard `logging` module under `pydantic_ai_conversations.*`:

| Level | When |
|---|---|
| INFO | A conversation is not accessible, or was deleted while a run was in flight or its phase was being recorded |
| WARNING | The registry or store is unavailable at the start of a run, a store call failed with a temporary error, a history load failed, a stats or phase update failed, a failure mapper raised, or a registry `ping` returned the reserved `step_store` key |
| DEBUG | A snapshot was emptied by the preparer chain and not saved |
| ERROR | A cleanup hook failed during a soft delete |

Hook failures during a purge are not logged; they are raised as the `ExceptionGroup`. Tracebacks
go to the log with the backend's own message, which may include connection detail;
client-facing messages are fixed strings. The package emits no spans of its own. The SQLite
driver routes statements through a real cursor so `opentelemetry-instrumentation-sqlite3`
sees them; run tracing comes from pydantic-ai and the harness.

## Known limits

- The listing cursor is the conversation id alone, so a row that changed position while a
  client paged can be skipped or repeated.
- History loading issues one store read per turn on the chain. `max_history_runs` bounds it.
- `message_count` counts stored messages including tool traffic.
- A run killed hard keeps `phase='started'` forever.
- A purged id is never released, in any namespace, and a purge leaves a tombstone; see
  [What remains after a purge](deleting.md#what-remains-after-a-purge).
- A nested agent that carries its own `ConversationsCapability` is not supported; see
  [Deleting conversations](deleting.md#purge).

## Version pins

`pydantic-ai-harness` is pinned to one 0.x minor. The package relies on four harness behaviours
that a later minor may change: `list_runs` sorted by start time, `latest_snapshot` returning
complete snapshots unless asked otherwise, the absence of a delete primitive, and
`StepPersistence` adopting the run's own id when given no `agent_name`. Each harness release
0.N pins pydantic-ai-slim 2.N, so the harness pin fixes the pydantic-ai version too.

`pydantic-ai-app-factory` is pinned to a 0.x minor as well. The current ranges are in
[pyproject.toml](https://github.com/aixbrain/pydantic-ai-conversations/blob/main/pyproject.toml).

Going deeper: [Backends](backends.md) for per-backend connection ownership.
