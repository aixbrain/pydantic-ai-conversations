"""SQLite backend: a `ConversationRegistry`, and the step store extras `ConversationsFeature` uses.

Stdlib `sqlite3` only -- nothing extra to install. Both classes reach the file the same way (see
`_driver.py`), so one SQLite file can carry a whole deployment: the harness's runs and snapshots
next to this package's conversations and turns.

    from pydantic_ai_conversations.sqlite import SqliteConversationRegistry, SqliteConversationStepStore

    feature = ConversationsFeature(
        registry=SqliteConversationRegistry('app.db'),
        step_store=SqliteConversationStepStore('app.db'),
    )
"""

from pydantic_ai_conversations.sqlite._registry import SqliteConversationRegistry
from pydantic_ai_conversations.sqlite._store import SqliteConversationStepStore

__all__ = ['SqliteConversationRegistry', 'SqliteConversationStepStore']
