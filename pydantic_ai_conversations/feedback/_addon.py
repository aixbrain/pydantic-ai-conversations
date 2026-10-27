"""`TurnFeedbackAddon`: the composition of store, routes, and the per-turn read."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from fastapi import APIRouter
from pydantic import BaseModel, ValidationError, create_model

from pydantic_ai_conversations import DeletionEvent
from pydantic_ai_conversations.addons import AddonRouteGuard, AddonTurnQuery
from pydantic_ai_conversations.feedback._routes import build_feedback_router
from pydantic_ai_conversations.feedback._wire import TurnFeedbackEntryOut
from pydantic_ai_conversations.feedback.store import TurnFeedbackRecord, TurnFeedbackStore

_NAME = re.compile(r'^[a-z][a-z0-9_]*$')
_logger = logging.getLogger(__name__)


class TurnFeedbackAddon:
    """Feedback with an app-defined payload, one entry per turn and author.

    `payload_model` is validated on every write and on every read, so the store never sees a
    shape and the app's model is the only schema. `name` is the add-on's key under `addons`, the
    last path segment of its routes, and the `kind` column in the store, which is how two feedback
    add-ons (a report form and a thumbs rating) share one table.
    """

    def __init__(self, *, payload_model: type[BaseModel], store: TurnFeedbackStore, name: str = 'feedback') -> None:
        if not _NAME.fullmatch(name):
            raise ValueError(f'feedback add-on name {name!r} must be a slug matching {_NAME.pattern!r}')
        self._name = name
        self._payload_model = payload_model
        self._store = store
        fields: dict[str, Any] = {'data': (payload_model, ...)}
        self._entry_model: type[TurnFeedbackEntryOut] = create_model(
            f'{payload_model.__name__}Entry', __base__=TurnFeedbackEntryOut, **fields
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def turn_model(self) -> type[BaseModel]:
        return self._entry_model

    @property
    def payload_model(self) -> type[BaseModel]:
        return self._payload_model

    @property
    def store(self) -> TurnFeedbackStore:
        return self._store

    @property
    def transient_errors(self) -> tuple[type[Exception], ...]:
        """The store's, so the detail read maps a store outage to the retryable 503."""
        return cast('tuple[type[Exception], ...]', getattr(self._store, 'transient_errors', ()))

    def to_entry(self, record: TurnFeedbackRecord) -> TurnFeedbackEntryOut:
        """A stored row as the wire shows it, the payload re-validated against the app's model;
        raises `ValidationError` for a row written under a model the current one rejects."""
        return self._entry_model(
            run_id=record.run_id,
            author_user_id=record.author_user_id,
            created_at=record.created_at,
            updated_at=record.updated_at,
            data=self._payload_model.model_validate(dict(record.data)),
        )

    def valid_entries(self, records: Sequence[TurnFeedbackRecord]) -> list[TurnFeedbackEntryOut]:
        """Wire entries in store order. A row the current model rejects is logged and left out,
        because one stale feedback row must not take the conversation's transcript down with it."""
        entries: list[TurnFeedbackEntryOut] = []
        for record in records:
            try:
                entries.append(self.to_entry(record))
            except ValidationError:
                _logger.warning(
                    'feedback %s on run %s by %s does not fit %s; left out',
                    self._name,
                    record.run_id,
                    record.author_user_id,
                    self._payload_model.__name__,
                    exc_info=True,
                )
        return entries

    async def load_turns(self, query: AddonTurnQuery, /) -> Mapping[str, BaseModel]:
        records = await self._store.by_run_ids(
            kind=self._name,
            namespace_key=query.namespace_key,
            conversation_id=query.conversation_id,
            run_ids=[turn.run_id for turn in query.turns],
            author_user_id=query.user.user_id,
        )
        # One entry per turn: the query is filtered to the caller, whose entry per turn is unique.
        return {entry.run_id: entry for entry in self.valid_entries(records)}

    def routers(self, guard: AddonRouteGuard, /) -> Sequence[APIRouter]:
        return [build_feedback_router(self, guard)]

    async def open(self) -> None:
        await self._store.open()

    async def close(self) -> None:
        await self._store.close()

    async def ping(self) -> Mapping[str, Any]:
        """The store's liveness, nested under the add-on's name in `/readyz`."""
        return await self._store.ping()

    async def on_delete(self, event: DeletionEvent) -> None:
        # Entries follow the conversation: gone on purge, kept (and hidden with it) on soft delete.
        if event.kind == 'purge':
            await self._store.delete_conversation(
                namespace_key=event.namespace_key, conversation_id=event.conversation_id
            )
