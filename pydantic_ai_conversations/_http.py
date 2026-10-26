"""The error shapes the routes and add-on routes raise, in one place so both agree.

They are `HTTPException`s with a structured `detail` because a standalone mount has no
`AgentError` exception handler; the same router then stays correct under both mounts.
"""

from __future__ import annotations

from fastapi import HTTPException


def not_found_404() -> HTTPException:
    """A conversation, or a turn of it, the caller cannot see. Foreign, missing and soft-deleted
    look the same, so the API is no existence oracle."""
    return HTTPException(
        status_code=404,
        detail={'code': 'conversations/not-found', 'message': 'The conversation does not exist.'},
    )


def unavailable_503() -> HTTPException:
    """A temporary backend failure; the client may retry the request unchanged."""
    return HTTPException(
        status_code=503,
        detail={
            'code': 'conversations/unavailable',
            'message': 'Conversation storage is temporarily unavailable.',
            'retryable': True,
        },
    )
