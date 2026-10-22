"""MongoDB backend: a `ConversationRegistry`, and the step store extras `ConversationsFeature` uses.

Needs the `mongodb` extra (`pip install pydantic-ai-conversations[mongodb]`). Both classes reach
the same database, so one MongoDB deployment can carry the harness's runs and snapshots next to
this package's conversations and turns.

    from pydantic_ai_conversations.mongo import MongoConversationRegistry, MongoConversationStepStore

    feature = ConversationsFeature(
        registry=MongoConversationRegistry(client=app_client, database='chat'),
        step_store=MongoConversationStepStore(client=app_client, database='chat'),
    )
"""

from pydantic_ai_conversations.mongo._registry import MongoConversationRegistry
from pydantic_ai_conversations.mongo._store import MongoConversationStepStore

__all__ = ['MongoConversationRegistry', 'MongoConversationStepStore']
