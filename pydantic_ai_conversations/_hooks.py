"""What a cleanup hook is told about a deletion; shared by the routes, the feature's purge, and the
add-on contract, so none of them has to import another for it."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic_ai_conversations.registry import ConversationMeta

DeletionKind: TypeAlias = Literal['soft_delete', 'purge']
"""Which deletion path invoked a cleanup hook."""


@dataclass(frozen=True)
class DeletionEvent:
    """What a cleanup hook is told about a deletion."""

    namespace_key: str
    conversation_id: str
    kind: DeletionKind
    meta: ConversationMeta | None
    """The entry as it was, or `None` when purging one that was already soft-deleted or purged
    (a retried purge) -- the registry's `get` hides both by contract, so the purge path cannot
    recover it."""


OnDeleteHook: TypeAlias = Callable[[DeletionEvent], Awaitable[None]]
