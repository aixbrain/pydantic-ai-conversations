"""The registry's exception vocabulary."""

from pydantic_ai_conversations.errors import ConversationNotFound, RegistryUnavailable


def test_not_found_is_an_exception_carrying_no_client_message() -> None:
    error = ConversationNotFound('conv-1')
    assert isinstance(error, Exception)
    assert 'conv-1' in str(error)


def test_unavailable_is_an_exception() -> None:
    error = RegistryUnavailable('pool exhausted')
    assert isinstance(error, Exception)
