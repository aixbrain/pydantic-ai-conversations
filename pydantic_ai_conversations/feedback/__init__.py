"""Turn feedback add-on: an app-defined payload per turn and author, stored in its own table and
shown inline on the detail read. See docs/feedback.md."""

from pydantic_ai_conversations.feedback._addon import TurnFeedbackAddon
from pydantic_ai_conversations.feedback._wire import TurnFeedbackEntryOut
from pydantic_ai_conversations.feedback.store import InMemoryTurnFeedbackStore, TurnFeedbackRecord, TurnFeedbackStore

__all__ = [
    'InMemoryTurnFeedbackStore',
    'TurnFeedbackAddon',
    'TurnFeedbackEntryOut',
    'TurnFeedbackRecord',
    'TurnFeedbackStore',
]
