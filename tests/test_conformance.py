"""The in-memory reference is the conformance suite's first consumer."""

import pytest

from pydantic_ai_conversations.registry import ConversationRegistry, InMemoryConversationRegistry
from pydantic_ai_conversations.testing import RegistryContract


class TestInMemoryConversationRegistryConformance(RegistryContract):
    @pytest.fixture
    def registry(self) -> ConversationRegistry:
        return InMemoryConversationRegistry()
