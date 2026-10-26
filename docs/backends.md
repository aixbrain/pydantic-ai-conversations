# Backends

This guide shows you how to choose and configure a storage backend: SQLite, MongoDB, or
PostgreSQL, each with a registry for the conversations and a step store for the messages.

Prerequisites: [How a conversation is stored](concepts.md), for what the registry and the step
store each hold.

## What goes where

The feature takes two stores. The registry holds conversations and turn records; the step
store, a harness `StepStore`, holds runs and snapshots. They share only the `conversation_id`.

| Backend | Registry | Step store | Install |
|---|---|---|---|
| SQLite | `SqliteConversationRegistry` | `SqliteConversationStepStore` | nothing extra |
| MongoDB | `MongoConversationRegistry` | `MongoConversationStepStore` | `pydantic-ai-conversations[mongodb]` |
| PostgreSQL | `PostgresConversationRegistry` | `PostgresConversationStepStore` | `pydantic-ai-conversations[postgres]` |

The shipped step stores subclass the harness's own `SqliteStepStore`, `MongoStepStore`, and
`PostgresStepStore` and add what the harness protocol lacks: `delete_runs`, which a purge uses;
`ping`, which `/readyz` reports; and `transient_errors`, the driver errors that mean the
database is temporarily unreachable and reach the client as a retryable 503. Any other harness
`StepStore` works as well. Without `delete_runs` a purge leaves the conversation's runs in
place, without `ping` the readiness report says nothing about the store, and without
`transient_errors` a database outage reaches the client as a 500.
[Extending](extending.md#your-own-step-store) shows how to add all three to a store of your own.

## SQLite

```python
from pydantic_ai_conversations.sqlite import SqliteConversationRegistry, SqliteConversationStepStore

registry = SqliteConversationRegistry('app.db')
step_store = SqliteConversationStepStore('app.db', max_snapshots_per_run=None)
```

Both may point at one file: the harness's tables and this package's tables sit side by side.
The registry creates its tables on first use and adds columns a file from an earlier version
lacks. It switches the file to SQLite's write-ahead log mode, so its connection and the step
store's coexist; its blocking calls run on a worker thread. A SQLite file serves one host;
replicas need MongoDB or PostgreSQL.

`max_snapshots_per_run` prunes old snapshots of a run. History loading reads only the latest
snapshot of each run, so pruning does not affect it. `media_store` and `media_threshold_bytes`
pass through to the harness store unchanged. The store keeps one connection of its own open for
`delete_runs` and `ping`; `close()` releases it on shutdown, and a later call reopens it.
`SqliteTurnFeedbackStore('app.db')` puts the [feedback add-on](feedback.md)'s table in the same
file.

## MongoDB

```python
from pymongo import AsyncMongoClient

from pydantic_ai_conversations.mongo import MongoConversationRegistry, MongoConversationStepStore

client = AsyncMongoClient('mongodb://localhost:27017')
registry = MongoConversationRegistry(client=client, database='chat')
step_store = MongoConversationStepStore(client=client, database='chat')
```

Pass exactly one of `client=` or `db_url=`. With `client=` the registry borrows it and leaves it
open on `close()`, so the app, the step store, and the registry can share one client, and the
app closes it on shutdown. With `db_url=` the registry owns the client it creates and closes
it; a step store built with `db_url=` owns its client the same way, and nothing closes that one
for you. `database` is required on both classes, with no default, so conversations and
messages cannot land in different databases by accident.

The registry uses the `conversations` and `turns` collections and creates its indexes on
`open()` or first use. `MongoConversationStepStore` passes `media_store`,
`media_threshold_bytes`, and `max_snapshots_per_run` through to the harness store unchanged.
Timestamps come back in UTC whether or not the client is `tz_aware`; MongoDB stores
milliseconds. `MongoTurnFeedbackStore(client=client, database='chat')` puts the
[feedback add-on](feedback.md)'s collection in the same database, with the same ownership rule
for the client.

## PostgreSQL

```python
import asyncpg
from psycopg_pool import AsyncConnectionPool

from pydantic_ai_conversations.postgres import PostgresConversationRegistry, PostgresConversationStepStore

pool = AsyncConnectionPool('postgresql://localhost/app', open=False)
registry = PostgresConversationRegistry(pool=pool, schema='chat')
asyncpg_pool = await asyncpg.create_pool(  # in the app's startup
    'postgresql://localhost/app', server_settings={'search_path': 'chat'}
)
step_store = PostgresConversationStepStore(asyncpg_pool)
```

The registry and the step store use different drivers. The registry runs on psycopg 3, so an
app can hand over the pool it already runs; the step store subclasses the harness's
`PostgresStepStore`, which takes an asyncpg pool, so the `postgres` extra installs both drivers
and the app opens two pools on one database. Behind a transaction pooler such as PgBouncer, set
`prepare_threshold=None` on the psycopg pool and `statement_cache_size=0` on the asyncpg pool,
because both drivers prepare statements the pooler cannot keep. asyncpg's own annotations do not
spell out the harness's pool protocol, so a strictly type-checked app casts the pool to
`PostgresPool` from `pydantic_ai_harness.media`. The step store may also be paired with the
SQLite or MongoDB registry, and the registry with any other harness store.

For the registry, pass exactly one of `pool=` or `dsn=`. A passed pool is opened by the registry
on startup if it is not open yet, shared, and left open on `close()`; the app closes it. A `dsn`
makes the registry own a pool it opens and closes itself. Connections are checked out at
`READ COMMITTED` for the duration of a call and restored afterwards. The step store never
closes the pool it is given and creates its tables on first use, unlike the registry's shipped
DDL below. `table` is a prefix for their names (`step_persistence_runs`, ... by default) and
cannot name a schema, so they land in the pool's search path; the `search_path` setting above
puts them in `chat` beside the registry's. `media_store`, `media_threshold_bytes`, and
`max_snapshots_per_run` pass through to the harness store unchanged.
`PostgresTurnFeedbackStore(pool=pool, schema='chat')` takes the registry's pool and schema for
the [feedback add-on](feedback.md)'s table.

### Schema and upgrades

The registry's DDL is shipped, not executed. `open()` probes for the `conversations` and
`turns` tables in the configured schema and raises `RegistryUnavailable` naming the missing
ones. Apply the schema through your own migration step:

```python
from pydantic_ai_conversations.postgres import SCHEMA_SQL

ddl = SCHEMA_SQL.replace('{schema}', '"chat"')  # paste into an Alembic revision or init script
```

`{schema}` is a plain placeholder for the quoted schema name. Every statement is
`IF NOT EXISTS`, and columns added in later versions come as `ADD COLUMN IF NOT EXISTS`, so
re-applying the file is the upgrade path. There is no version table, and the startup probe
checks that the tables exist, not that they have every column: an app upgraded without
re-applying the file starts and fails on the first query that needs a new column. The role
that applies the file needs DDL rights; the app's role needs the usual table and sequence
grants.

`create_schema=True` on the constructor, or `create_schema(pool, schema=...)`, applies the DDL
at startup. Both exist for tests and local development: production deployments usually apply
schema in an ordered step before the app starts, the app's database user often lacks DDL
rights, and replicas racing `CREATE TABLE` can deadlock in the catalog. The feedback add-on's
DDL is a separate file, `FEEDBACK_SCHEMA_SQL` from `feedback.sql`, applied the same way, with
`create_feedback_schema` as its test-time shortcut.

## Choosing

SQLite serves one process on one host; it is the right default for development and
single-instance deployments. MongoDB and PostgreSQL each serve both stores from one deployment
and work across replicas; PostgreSQL costs a second driver and a second pool.

Every shipped registry passes the same conformance suite.

What you end up with: a registry and a step store wired into the feature, and a migration file
where the deployment expects one. Going deeper: [Operations](operations.md).
