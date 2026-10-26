"""MongoDB backend: a `ConversationRegistry`, the step store extras `ConversationsFeature` uses, and
the feedback add-on's store.

Needs the `mongodb` extra (`pip install pydantic-ai-conversations[mongodb]`). All three reach the
same database, so one MongoDB deployment can carry the harness's runs and snapshots next to this
package's conversations, turns, and feedback.

    from pydantic_ai_conversations.mongo import MongoConversationRegistry, MongoConversationStepStore

    feature = ConversationsFeature(
        registry=MongoConversationRegistry(client=app_client, database='chat'),
        step_store=MongoConversationStepStore(client=app_client, database='chat'),
        addons=[
            TurnFeedbackAddon(
                payload_model=Report, store=MongoTurnFeedbackStore(client=app_client, database='chat'), name='report'
            )
        ],
    )
"""

from pydantic_ai_conversations.mongo._feedback import MongoTurnFeedbackStore
from pydantic_ai_conversations.mongo._registry import MongoConversationRegistry
from pydantic_ai_conversations.mongo._store import MongoConversationStepStore

__all__ = ['MongoConversationRegistry', 'MongoConversationStepStore', 'MongoTurnFeedbackStore']
