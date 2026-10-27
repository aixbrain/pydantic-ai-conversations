"""The feedback add-on's storage port and its in-memory reference.

The store holds the payload as plain JSON and knows no shape: the add-on validates against the
app's model on the way in and out, so a store serves every payload model and several feedback
add-ons at once (`kind` is the add-on's name). `namespace_key` is stored and filtered even
though the routes already resolved the conversation in the caller's namespace, so a purge and a
list cannot reach another namespace's rows through a guessed id.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class TurnFeedbackRecord:
    kind: str
    namespace_key: str
    conversation_id: str
    run_id: str
    author_user_id: str
    created_at: datetime
    updated_at: datetime
    data: Mapping[str, Any]


class TurnFeedbackStore(Protocol):
    """One entry per (`kind`, `namespace_key`, `conversation_id`, `run_id`, `author_user_id`); `upsert`
    replaces the caller's own and never moves a row to another namespace or conversation."""

    async def open(self) -> None: ...  # pragma: no cover

    async def close(self) -> None: ...  # pragma: no cover

    async def upsert(
        self,
        *,
        kind: str,
        namespace_key: str,
        conversation_id: str,
        run_id: str,
        author_user_id: str,
        data: Mapping[str, Any],
    ) -> TurnFeedbackRecord: ...  # pragma: no cover

    async def delete(
        self, *, kind: str, namespace_key: str, conversation_id: str, run_id: str, author_user_id: str
    ) -> bool: ...  # pragma: no cover

    async def by_run_ids(
        self,
        *,
        kind: str,
        namespace_key: str,
        conversation_id: str,
        run_ids: Sequence[str],
        author_user_id: str | None = None,
    ) -> Sequence[TurnFeedbackRecord]: ...  # pragma: no cover

    async def list(
        self, *, kind: str, namespace_key: str, conversation_id: str
    ) -> Sequence[TurnFeedbackRecord]: ...  # pragma: no cover

    async def delete_conversation(self, *, namespace_key: str, conversation_id: str) -> int:
        """Every kind's entries for the conversation; the count removed, for a purge's report."""
        ...  # pragma: no cover

    async def ping(self) -> Mapping[str, Any]: ...  # pragma: no cover


def _now() -> datetime:
    return datetime.now(UTC)


class InMemoryTurnFeedbackStore:
    """The reference implementation: a dict keyed by (kind, namespace, conversation, run_id, author)."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str, str, str, str], TurnFeedbackRecord] = {}

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert(
        self,
        *,
        kind: str,
        namespace_key: str,
        conversation_id: str,
        run_id: str,
        author_user_id: str,
        data: Mapping[str, Any],
    ) -> TurnFeedbackRecord:
        key = (kind, namespace_key, conversation_id, run_id, author_user_id)
        now = _now()
        existing = self._rows.get(key)
        if existing is None:
            record = TurnFeedbackRecord(
                kind, namespace_key, conversation_id, run_id, author_user_id, now, now, dict(data)
            )
        else:
            record = replace(existing, updated_at=now, data=dict(data))
        self._rows[key] = record
        return record

    async def delete(
        self, *, kind: str, namespace_key: str, conversation_id: str, run_id: str, author_user_id: str
    ) -> bool:
        key = (kind, namespace_key, conversation_id, run_id, author_user_id)
        if key not in self._rows:
            return False
        del self._rows[key]
        return True

    async def by_run_ids(
        self,
        *,
        kind: str,
        namespace_key: str,
        conversation_id: str,
        run_ids: Sequence[str],
        author_user_id: str | None = None,
    ) -> Sequence[TurnFeedbackRecord]:
        wanted = set(run_ids)
        return [
            row
            for row in self._rows.values()
            if row.kind == kind
            and row.namespace_key == namespace_key
            and row.conversation_id == conversation_id
            and row.run_id in wanted
            and (author_user_id is None or row.author_user_id == author_user_id)
        ]

    async def list(self, *, kind: str, namespace_key: str, conversation_id: str) -> Sequence[TurnFeedbackRecord]:
        rows = [
            row
            for row in self._rows.values()
            if row.kind == kind and row.namespace_key == namespace_key and row.conversation_id == conversation_id
        ]
        return sorted(rows, key=lambda row: (row.created_at, row.author_user_id))

    async def delete_conversation(self, *, namespace_key: str, conversation_id: str) -> int:
        keys = [
            key
            for key, row in self._rows.items()
            if row.namespace_key == namespace_key and row.conversation_id == conversation_id
        ]
        for key in keys:
            del self._rows[key]
        return len(keys)

    async def ping(self) -> Mapping[str, Any]:
        return {'ok': True}
