"""Exceptions the registry raises; the capability and routes translate them at the boundary
(`conversations/not-found` -> 404, `conversations/unavailable` -> 503 retryable).

`str(exc)` goes to the log only and never reaches the client, so backends may put connection
detail in the message. Caller bugs stay plain builtins: a non-positive
`ConversationRegistry.list` limit raises `ValueError`.
"""

from __future__ import annotations


class ConversationNotFound(Exception):
    """The conversation does not exist in the caller's namespace.

    Also raised when the id exists under a foreign namespace or is soft-deleted: anything but
    not-found would make the API an existence oracle.
    """


class RegistryUnavailable(Exception):
    """A transient registry backend failure; the caller may retry unchanged."""
