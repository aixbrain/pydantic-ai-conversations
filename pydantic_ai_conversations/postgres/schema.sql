-- Schema for pydantic_ai_conversations' PostgreSQL registry.
-- Apply this yourself: paste it into an Alembic revision, an init script, or call
-- `pydantic_ai_conversations.postgres.create_schema`. The registry does not run DDL by default.
-- `{schema}` is a plain text placeholder -- replace every occurrence with your schema name (as a
-- quoted identifier). It is substituted with `str.replace`, not Python `str.format`, so this file
-- is ordinary SQL: no other brace needs escaping, on either the migration-tool path or ours.

CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.conversations (
    conversation_id TEXT PRIMARY KEY,
    namespace_key   TEXT        NOT NULL,
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL,
    last_message_at TIMESTAMPTZ,
    message_count   BIGINT      NOT NULL DEFAULT 0,
    title           TEXT,
    deleted_at      TIMESTAMPTZ,
    attrs           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    purged          BOOLEAN     NOT NULL DEFAULT FALSE
);

-- `list`'s exact shape: the namespace as an equality prefix, then its sort key, over live rows
-- only. Postgres walks it in order and stops at LIMIT instead of sorting the namespace. The
-- ordering expression is COALESCE, never a bare column: under DESC, Postgres sorts NULLs FIRST,
-- so a bare `last_message_at DESC` would float never-used conversations to the top of the list.
CREATE INDEX IF NOT EXISTS ix_conversations_activity ON {schema}.conversations (
    namespace_key, (COALESCE(last_message_at, created_at)) DESC, conversation_id DESC
) WHERE deleted_at IS NULL AND NOT purged;

-- Containment lookups for the `attrs` filter.
CREATE INDEX IF NOT EXISTS ix_conversations_attrs ON {schema}.conversations USING GIN (attrs)
    WHERE deleted_at IS NULL AND NOT purged;

CREATE TABLE IF NOT EXISTS {schema}.turns (
    seq             BIGSERIAL PRIMARY KEY,
    conversation_id TEXT        NOT NULL,
    run_id          TEXT        NOT NULL,
    author_user_id  TEXT        NOT NULL,
    at              TIMESTAMPTZ NOT NULL,
    phase           TEXT        NOT NULL DEFAULT 'started',
    previous_run_id TEXT,
    error_code      TEXT,
    error_message   TEXT,
    error_transient BOOLEAN
);

-- A table created by an earlier version of this file lacks these columns; CREATE TABLE IF NOT
-- EXISTS leaves it as it is, so the file stays re-runnable on upgrade.
ALTER TABLE {schema}.turns ADD COLUMN IF NOT EXISTS previous_run_id TEXT;
ALTER TABLE {schema}.turns ADD COLUMN IF NOT EXISTS error_code TEXT;
ALTER TABLE {schema}.turns ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE {schema}.turns ADD COLUMN IF NOT EXISTS error_transient BOOLEAN;

CREATE INDEX IF NOT EXISTS ix_turns_conversation ON {schema}.turns (conversation_id, seq);
