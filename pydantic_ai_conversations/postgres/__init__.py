"""PostgreSQL backend: a `ConversationRegistry` and a step store for apps that already run Postgres.

Needs the `postgres` extra (`pip install pydantic-ai-conversations[postgres]`).

    from pydantic_ai_conversations.postgres import PostgresConversationRegistry, PostgresConversationStepStore

    feature = ConversationsFeature(
        registry=PostgresConversationRegistry(pool=app_pool, schema='chat'),
        step_store=PostgresConversationStepStore(asyncpg_pool),
    )

The registry runs on psycopg 3 and the step store, a subclass of the harness's `PostgresStepStore`,
on an asyncpg pool; the two share only the `conversation_id`. The registry's DDL is shipped, not
executed: apply `schema.sql` through your own migration tool, or pass `create_schema=True` for
tests and local development. The step store creates its own tables on first use, in the asyncpg
pool's search path (`server_settings={'search_path': 'chat'}` puts them beside the registry's).
"""

from pydantic_ai_conversations.postgres._registry import PostgresConversationRegistry
from pydantic_ai_conversations.postgres._schema import SCHEMA_SQL, create_schema
from pydantic_ai_conversations.postgres._store import PostgresConversationStepStore

__all__ = ['SCHEMA_SQL', 'PostgresConversationRegistry', 'PostgresConversationStepStore', 'create_schema']
