-- Schema for the feedback add-on's PostgreSQL store. Apply it like schema.sql: paste it into your
-- migration, or call `pydantic_ai_conversations.postgres.create_feedback_schema`. `{schema}` is
-- substituted with `str.replace`; it is kept apart from schema.sql so the registry's DDL stays as
-- it is.

CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.feedback (
    kind            TEXT        NOT NULL,
    namespace_key   TEXT        NOT NULL,
    conversation_id TEXT        NOT NULL,
    run_id          TEXT        NOT NULL,
    author_user_id  TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL,
    data            JSONB       NOT NULL,
    PRIMARY KEY (kind, namespace_key, conversation_id, run_id, author_user_id)
);

CREATE INDEX IF NOT EXISTS ix_feedback_conversation ON {schema}.feedback (conversation_id, kind);
