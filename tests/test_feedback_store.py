"""The in-memory feedback store is the reference implementation and passes the contract."""

import pytest

from pydantic_ai_conversations.feedback import InMemoryTurnFeedbackStore, TurnFeedbackStore
from pydantic_ai_conversations.testing import TurnFeedbackStoreContract


class TestInMemoryTurnFeedbackStoreConformance(TurnFeedbackStoreContract):
    @pytest.fixture
    def feedback_store(self) -> TurnFeedbackStore:
        return InMemoryTurnFeedbackStore()
