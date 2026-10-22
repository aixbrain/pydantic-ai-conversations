"""The conversation registry: metadata, turn records, store protocol, in-memory reference.

Every method takes the `namespace_key` that `scope.build_namespace` derives from the declared
scope, and answers only within it, so a foreign `conversation_id` resolves to nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal, Protocol

from pydantic_ai_conversations.errors import ConversationNotFound

RunPhase = Literal['started', 'answered', 'failed']
"""Where a run stands: `start_turn` writes `'started'`; the run's terminal hooks resolve it to
`'answered'` or `'failed'`.

Not a liveness signal -- a hard-killed run fires no terminal hook and stays `'started'` forever.
Distinct from the snapshot axis: a `'failed'` run can still have left a `complete` boundary
snapshot (a crash after a settled tool round)."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ConversationMeta:
    """One conversation in the registry; no messages live here."""

    namespace_key: str
    conversation_id: str
    created_by: str
    created_at: datetime
    last_message_at: datetime | None = None
    message_count: int = 0
    title: str | None = None
    deleted_at: datetime | None = None
    attrs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Backends hydrate metas from DB rows with plain dicts; freeze regardless of caller.
        object.__setattr__(self, 'attrs', MappingProxyType(dict(self.attrs)))


@dataclass(frozen=True)
class TurnError:
    """Why a run failed, in client-safe terms, as `ConversationsCapability.describe_failure`
    reported it.

    `transient` says the failure may clear on its own, such as a model outage, as the live
    error's `retryable` does."""

    code: str
    message: str
    transient: bool = False


@dataclass(frozen=True)
class TurnRecord:
    """Which run wrote a turn, who authored it (multi-writer attribution), what it builds on, and
    how it ended."""

    run_id: str
    author_user_id: str
    at: datetime = field(default_factory=_utcnow)
    phase: RunPhase = 'started'
    """See `RunPhase`: `'started'` until a terminal hook resolves it."""
    previous_run_id: str | None = None
    """The turn this one builds on; `None` for the first turn of a conversation. The chain of
    these links is the conversation's history, so a retry or a fork is a second turn naming the
    same `previous_run_id`."""
    error: TurnError | None = None
    """Why a `'failed'` run failed; `None` when the capability has no `describe_failure` mapper."""


class ConversationRegistry(Protocol):
    """The feature's registry port. No namespace-free method exists: a conversation_id alone is
    never proof of access.

    No method may distinguish a foreign-namespace conversation_id from a missing one -- both
    answer `ConversationNotFound` / `None` / absence-from-list, identically shaped. `attrs` are
    create-time-only; there is no `set_attrs`.
    """

    async def open(self) -> None:  # pragma: no cover
        """Acquire backend resources (pool, connection, client). Idempotent-safe to call once
        before first use; not required before every method call."""
        ...

    async def close(self) -> None:  # pragma: no cover
        """Release backend resources acquired by `open`."""
        ...

    async def ping(self) -> Mapping[str, Any]:  # pragma: no cover
        """Liveness round-trip for `/readyz`: must be cheap, raising means unhealthy. `/readyz` is
        unauthenticated, so the report carries operational status only (`{'ok': True}`) and
        nothing derived from stored data, counts included. The key `step_store` is reserved for
        `ConversationsFeature.health`'s own step-store report."""
        ...

    async def register_conversation(
        self,
        namespace_key: str,
        conversation_id: str,
        *,
        created_by: str,
        attrs: Mapping[str, str] | None = None,
    ) -> ConversationMeta:  # pragma: no cover
        """Idempotent create-or-return: create the entry, or return an undeleted entry already in
        `namespace_key` as-is (`created_by`/`attrs` on the call are ignored, not merged).

        A foreign, soft-deleted, or purged conversation_id raises `ConversationNotFound` -- the
        three cases are indistinguishable to the caller. Must claim the global id atomically
        (SQL: `INSERT ... ON CONFLICT DO NOTHING`, then re-select). Because this doubles as a
        global-id existence probe, conversation ids must be unguessable (uuid4).
        """
        ...

    async def get(self, namespace_key: str, conversation_id: str) -> ConversationMeta | None:  # pragma: no cover
        """`None` for missing, foreign-namespace, or soft-deleted -- never raises."""
        ...

    async def list(
        self,
        namespace_key: str,
        *,
        attrs: Mapping[str, str] | None = None,
        page_after: str | None = None,
        limit: int = 100,
    ) -> Sequence[ConversationMeta]:  # pragma: no cover
        """The namespace's live (non-deleted) entries, newest activity first, at most `limit`
        rows; `attrs` keeps only entries containing every given pair (equality, AND semantics).

        Sorted by `(last_message_at or created_at, conversation_id)` descending -- the id
        tie-break keeps equal-timestamp ordering deterministic. `page_after` is a conversation_id
        cursor; an unknown or since-deleted cursor means end-of-list, not an error. The id-only
        cursor is unstable under concurrent activity (rows can be skipped or repeated across
        pages) -- an accepted tradeoff, keep it rather than fixing it divergently per backend. A
        non-positive `limit` raises `ValueError`.
        """
        ...

    async def set_title(self, namespace_key: str, conversation_id: str, title: str) -> None:  # pragma: no cover
        """`ConversationNotFound` for an unknown, foreign, or deleted id."""
        ...

    async def update_message_stats(
        self,
        namespace_key: str,
        conversation_id: str,
        *,
        last_message_at: datetime,
        message_count_delta: int,
    ) -> None:  # pragma: no cover
        """Overwrite `last_message_at` and atomically add `message_count_delta` to `message_count`.

        `ConversationNotFound` for an unknown, foreign, or deleted id. The count must be an
        atomic increment (`SET count = count + delta`), never read-modify-write;
        `last_message_at` is last-write-wins with no monotonicity requirement.
        """
        ...

    async def soft_delete(self, namespace_key: str, conversation_id: str) -> ConversationMeta:  # pragma: no cover
        """Mark the entry deleted (`deleted_at`) and return it; hidden from `get`/`list` and
        unrevivable via `register_conversation` thereafter.

        `ConversationNotFound` for an unknown, foreign, or deleted id -- a second delete raises
        too. The id stays claimed: no revival, no takeover."""
        ...

    async def start_turn(
        self,
        namespace_key: str,
        conversation_id: str,
        *,
        run_id: str,
        author_user_id: str,
        previous_run_id: str | None = None,
    ) -> None:  # pragma: no cover
        """`ConversationNotFound` for an unknown, foreign, or deleted id. Records the turn as
        `'started'`, building on `previous_run_id`, which is stored as given: the caller has
        checked that it names a turn of this conversation, or is `None` for the first turn.
        """
        ...

    async def set_turn_phase(
        self,
        namespace_key: str,
        conversation_id: str,
        *,
        run_id: str,
        phase: RunPhase,
        error: TurnError | None = None,
    ) -> None:  # pragma: no cover
        """Resolve a recorded turn's phase (see `RunPhase`) and failure reason from the run's
        terminal hooks; `error` overwrites the stored one, `None` included.

        `ConversationNotFound` for an unknown, foreign, or deleted id. An unknown `run_id` under
        an existing entry is a silent no-op -- a terminal hook can fire for a turn whose
        `start_turn` never landed. Must be an idempotent single-row update
        (`SET phase = ..., error = ... WHERE run_id = ...`)."""
        ...

    async def list_turns(self, namespace_key: str, conversation_id: str) -> Sequence[TurnRecord]:  # pragma: no cover
        """All turns recorded for the conversation, in `start_turn` order.

        `ConversationNotFound` for an unknown, foreign, or deleted id."""
        ...

    async def purge(self, namespace_key: str, conversation_id: str) -> bool:  # pragma: no cover
        """Remove the entry and its turns; the global id stays claimed forever, against every
        namespace, the previous owner included -- substrate deletion is best-effort, so a
        released id could inherit surviving history. A foreign purge is a silent no-op.

        Returns whether `namespace_key` owned `conversation_id` (soft-deleted counts as owned):
        `True` even on a repeated purge, `False` for a foreign or unknown id. Callers gate
        substrate deletion on this -- a purge must never delete another namespace's runs.
        """
        ...


class InMemoryConversationRegistry:
    """In-memory reference implementation of `ConversationRegistry`.

    Its methods are atomic only because no `await` sits between read and write -- a single
    coroutine step never yields mid-mutation. That is NOT a template: a real backend must use
    atomic statements (`SET count = count + delta`, `INSERT ... ON CONFLICT`) rather than
    read-then-write, or concurrent callers will race.
    """

    def __init__(self) -> None:
        self._conversations: dict[tuple[str, str], ConversationMeta] = {}
        self._owner_namespace: dict[str, str] = {}  # conversation_id -> namespace_key
        self._turns: dict[tuple[str, str], list[TurnRecord]] = {}
        self._purged: set[str] = set()  # conversation_id tombstones -- permanently dead ids

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def ping(self) -> Mapping[str, Any]:
        return {'ok': True}

    def _require_entry(self, namespace_key: str, conversation_id: str) -> ConversationMeta:
        meta = self._conversations.get((namespace_key, conversation_id))
        if meta is None or meta.deleted_at is not None:
            raise ConversationNotFound(conversation_id)
        return meta

    async def register_conversation(
        self,
        namespace_key: str,
        conversation_id: str,
        *,
        created_by: str,
        attrs: Mapping[str, str] | None = None,
    ) -> ConversationMeta:
        if conversation_id in self._purged:
            raise ConversationNotFound(conversation_id)  # dead forever, even for the original owner
        owner = self._owner_namespace.get(conversation_id)
        if owner is not None and owner != namespace_key:
            raise ConversationNotFound(conversation_id)
        existing = self._conversations.get((namespace_key, conversation_id))
        if existing is not None:
            if existing.deleted_at is not None:
                raise ConversationNotFound(conversation_id)
            return existing
        meta = ConversationMeta(
            namespace_key=namespace_key,
            conversation_id=conversation_id,
            created_by=created_by,
            created_at=_utcnow(),
            attrs=dict(attrs) if attrs else {},
        )
        self._conversations[(namespace_key, conversation_id)] = meta
        self._owner_namespace[conversation_id] = namespace_key
        return meta

    async def get(self, namespace_key: str, conversation_id: str) -> ConversationMeta | None:
        meta = self._conversations.get((namespace_key, conversation_id))
        return None if meta is None or meta.deleted_at is not None else meta

    async def list(
        self,
        namespace_key: str,
        *,
        attrs: Mapping[str, str] | None = None,
        page_after: str | None = None,
        limit: int = 100,
    ) -> Sequence[ConversationMeta]:
        if limit <= 0:
            raise ValueError(f'limit must be positive, got {limit}')
        rows = [
            meta
            for (ns, _), meta in self._conversations.items()
            if ns == namespace_key
            and meta.deleted_at is None
            and (attrs is None or all(meta.attrs.get(k) == v for k, v in attrs.items()))
        ]
        rows.sort(
            key=lambda m: (
                m.last_message_at or m.created_at,
                m.conversation_id,
            ),
            reverse=True,
        )
        if page_after is not None:
            ids = [m.conversation_id for m in rows]
            rows = rows[ids.index(page_after) + 1 :] if page_after in ids else []
        return rows[:limit]

    async def set_title(self, namespace_key: str, conversation_id: str, title: str) -> None:
        meta = self._require_entry(namespace_key, conversation_id)
        self._conversations[(namespace_key, conversation_id)] = replace(meta, title=title)

    async def update_message_stats(
        self,
        namespace_key: str,
        conversation_id: str,
        *,
        last_message_at: datetime,
        message_count_delta: int,
    ) -> None:
        meta = self._require_entry(namespace_key, conversation_id)
        self._conversations[(namespace_key, conversation_id)] = replace(
            meta,
            last_message_at=last_message_at,
            message_count=meta.message_count + message_count_delta,
        )

    async def soft_delete(self, namespace_key: str, conversation_id: str) -> ConversationMeta:
        meta = self._require_entry(namespace_key, conversation_id)
        deleted = replace(meta, deleted_at=_utcnow())
        self._conversations[(namespace_key, conversation_id)] = deleted
        return deleted

    async def start_turn(
        self,
        namespace_key: str,
        conversation_id: str,
        *,
        run_id: str,
        author_user_id: str,
        previous_run_id: str | None = None,
    ) -> None:
        self._require_entry(namespace_key, conversation_id)
        self._turns.setdefault((namespace_key, conversation_id), []).append(
            TurnRecord(
                run_id=run_id,
                author_user_id=author_user_id,
                previous_run_id=previous_run_id,
            )
        )

    async def set_turn_phase(
        self,
        namespace_key: str,
        conversation_id: str,
        *,
        run_id: str,
        phase: RunPhase,
        error: TurnError | None = None,
    ) -> None:
        self._require_entry(namespace_key, conversation_id)
        records = self._turns.get((namespace_key, conversation_id), [])
        for index, record in enumerate(records):
            if record.run_id == run_id:
                records[index] = replace(record, phase=phase, error=error)
                return
        # Unknown run under an existing entry: a no-op -- its `start_turn` never landed.

    async def list_turns(self, namespace_key: str, conversation_id: str) -> Sequence[TurnRecord]:
        self._require_entry(namespace_key, conversation_id)
        return list(self._turns.get((namespace_key, conversation_id), ()))

    async def purge(self, namespace_key: str, conversation_id: str) -> bool:
        if self._owner_namespace.get(conversation_id) != namespace_key:
            return False  # foreign purge is a silent no-op: neither the entry nor the claim is ours to touch
        self._conversations.pop((namespace_key, conversation_id), None)
        self._turns.pop((namespace_key, conversation_id), None)
        self._purged.add(conversation_id)
        # The tombstone in `_purged` keeps the id permanently dead against every namespace;
        # the stale owner_namespace entry is harmless.
        return True
