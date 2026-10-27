# pydantic-ai-conversations

Conversation persistence as a `pydantic-ai-app-factory` feature: per-turn message batches are
`pydantic-ai-harness` `StepPersistence` snapshots; this package adds the conversation registry
(listing, soft-delete, title, namespace access wall), the load/append capability, and the
`/conversations` routes.

## Module layout

Public modules are the API: `feature.py`, `registry.py`, `scope.py`, `errors.py`,
`testing.py`, `stores.py`, `addons.py` (the add-on contract), `feedback/` (the turn feedback
add-on), and the `sqlite/`, `mongo/`, and `postgres/` backend packages.
Every other module is a private implementation detail and underscore-prefixed; nothing outside
this package may import one -- inside `sqlite/`, `mongo/`, and `postgres/` the same rule applies to
their own modules (`_driver.py`, `_registry.py`, and `_store.py`, plus `_schema.py` in
`postgres/`, which ships the registry's DDL), whose names reach users only through
`pydantic_ai_conversations.sqlite`, `.mongo`, and `.postgres` respectively. Some public
names -- `ConversationsCapability`, `build_conversation_router`,
the wire models (`ConversationDetail`/`ConversationList`/`ConversationOut`), and the delete hook
types (`DeletionEvent`, `OnDeleteHook`) -- are implemented in private modules (`_capability.py`,
`_routes.py`, `_hooks.py`) and re-exported from `__init__.py`; the package root, not the module a
name happens to live in, is the supported import surface.

## Boundaries

- Substrate *mechanics* (durability, events, retention, resume) are `pydantic-ai-harness` -- never
  rebuilt here. This package owns ONE declared write seam in front of it
  (`stores.PreparingStepStore`): `save_snapshot` runs through a static preparer chain, and what
  lies in the store afterwards is the single truth every reader sees -- stored snapshots are never
  rewritten after the fact.
- Authorization (scopes, HMAC, case membership) is the app's `AuthProvider` -- never decided here.
- The registry's namespace is the only access wall this package enforces; a foreign
  `conversation_id` resolves to nothing (404), never 403/409.
- New registry operations never extend the `ConversationRegistry` protocol itself. They ship as a
  separate `runtime_checkable` protocol that callers detect with `isinstance`, as
  `stores.SupportsRunDeletion` does for step stores, so an app's own registry keeps satisfying the
  protocol after an upgrade. The rule is for operations the load and read paths can do without;
  what they need from every registry (the turn lineage, the turn's phase and failure reason) is
  part of the core protocol.
- `sqlite/`, `mongo/`, and `postgres/` are backend kits, each kept liftable into its own
  distribution: each imports nothing from this package but `errors`, the `registry` types, and the
  store port in `feedback/store.py`, and driver plumbing (connection ownership, schema/index
  setup, error mapping) stays in its own `_driver.py`, apart from the port implementations.
  `feedback/` builds on the public add-on API only (`addons.AddonRouteGuard`,
  `stores.SupportsTransientErrors`, `stores.SupportsPing`), so it could be lifted out like a kit;
  the kits ship its stores because a store belongs with its driver, and `feedback/store.py` is
  the one file a lifted kit would carry along. `sqlite/` needs no extra -- stdlib `sqlite3`
  only. `mongo/` has a real driver dependency, so it ships behind the `mongodb` extra; `postgres/`
  ships behind the `postgres` extra the same way.

## Coding standards

- Python 3.11+
- pyright strict; avoid `Any` in public signatures where practical
- ruff: line-length 120, single quotes, max-complexity 15
- 100% branch coverage, enforced by `make testcov`
- docstrings use single backticks (markdown), not RST double backticks
- no em-dashes in prose, docstrings, comments, or commit messages; use `--`
- document the why and the non-obvious; don't restate what the code says

## Commands

```bash
make install      # uv sync
make format       # ruff format (rewrites files)
make format-check # ruff format --check
make lint         # ruff check
make typecheck    # pyright strict
make test         # pytest
make testcov      # pytest with 100% branch coverage
make all          # lint + format-check + typecheck + testcov
```

Run `make all` before every commit.

## Testing patterns

- No real model calls: `ALLOW_MODEL_REQUESTS = False` in `tests/conftest.py`; the one wire test
  that drives a real model class over a mock transport lifts it for its own runs
- `FunctionModel` scripts model behaviour (tool calls, failures, streams)
- The backend suites run against real MongoDB and PostgreSQL servers (`tests/containers.py`,
  testcontainers, so Docker must be running); a fake cannot prove the atomic claim or the
  duplicate-key race
- Standalone test functions; the only test classes are the backends' `testing.RegistryContract`
  subclasses
- Tests may import private members of this package (pyright's `reportPrivateUsage` is off under
  `tests/`); mark genuinely unreachable branches `# pragma: no cover`
