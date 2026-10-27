"""SQLite backend: a `ConversationRegistry`, the step store extras `ConversationsFeature` uses, and
the feedback add-on's store.

Stdlib `sqlite3` only -- nothing extra to install. All three reach the file the same way (see
`_driver.py`), so one SQLite file can carry a whole deployment: the harness's runs and snapshots
next to this package's conversations, turns, and feedback.

    from pydantic_ai_conversations.sqlite import SqliteConversationRegistry, SqliteConversationStepStore

    feature = ConversationsFeature(
        registry=SqliteConversationRegistry('app.db'),
        step_store=SqliteConversationStepStore('app.db'),
        addons=[TurnFeedbackAddon(payload_model=Report, store=SqliteTurnFeedbackStore('app.db'), name='report')],
    )
"""

from pydantic_ai_conversations.sqlite._feedback import SqliteTurnFeedbackStore
from pydantic_ai_conversations.sqlite._registry import SqliteConversationRegistry
from pydantic_ai_conversations.sqlite._store import SqliteConversationStepStore

__all__ = ['SqliteConversationRegistry', 'SqliteConversationStepStore', 'SqliteTurnFeedbackStore']
