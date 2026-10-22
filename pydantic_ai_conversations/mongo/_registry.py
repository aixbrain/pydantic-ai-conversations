"""`ConversationRegistry` on MongoDB: the port implementation, in Mongo operations.

Documents are keyed by the conversation id itself, so the global claim is the collection's own
`_id` uniqueness. Two shapes differ from the SQL backends on purpose. `activity_at` is stored
rather than computed, because Mongo has no expression indexes and the list's sort key must be a
real field. `attrs` is an array of `{k, v}` pairs rather than a subdocument, so a key holding a
dot is a bound value instead of a field path.

Connection ownership, index setup and the `PyMongoError` -> `RegistryUnavailable` mapping are
`_driver.py`'s.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from pymongo import AsyncMongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

from pydantic_ai_conversations.errors import ConversationNotFound
from pydantic_ai_conversations.mongo._driver import MongoDriver, maps_backend_errors
from pydantic_ai_conversations.registry import ConversationMeta, RunPhase, TurnError, TurnRecord

_Document = dict[str, Any]

_LIVE: dict[str, Any] = {'deleted_at': None, 'purged': False}
"""The live-entry predicate every read and mutation carries, alongside the namespace."""


def _now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime | None) -> datetime | None:
    """MongoDB stores UTC but returns it naive unless the client is `tz_aware`, and a
    caller-supplied client may be either. Stamp the zone rather than assume the configuration."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _meta(doc: _Document) -> ConversationMeta:
    # NOT NULL by construction: every insert sets it, so the cast is a type narrowing, not a
    # runtime check -- no code path in this module ever produces a document without it.
    created_at = cast(datetime, _utc(doc['created_at']))
    return ConversationMeta(
        namespace_key=doc['namespace_key'],
        conversation_id=doc['_id'],
        created_by=doc['created_by'],
        created_at=created_at,
        last_message_at=_utc(doc['last_message_at']),
        message_count=doc['message_count'],
        title=doc['title'],
        deleted_at=_utc(doc['deleted_at']),
        attrs={pair['k']: pair['v'] for pair in doc['attrs']},
    )


class MongoConversationRegistry:
    """`ConversationRegistry` on a MongoDB database.

    Methods lazy-connect, so `open()` is optional; after `close()` every method raises
    `RegistryUnavailable` until `open()` reopens. Safe to point at the same database as the
    harness's `MongoStepStore`: one database carries a deployment's runs and snapshots alongside
    its conversations and turns, and passing both the same client shares one connection pool.
    `database` has no default, like `MongoStepStore`'s, because a default on one side would put the
    conversations and their messages in different databases whenever only the other side is set.
    """

    def __init__(
        self,
        *,
        client: AsyncMongoClient[_Document] | None = None,
        db_url: str | None = None,
        database: str,
    ) -> None:
        self._driver = MongoDriver(client=client, db_url=db_url, database=database)

    @maps_backend_errors
    async def open(self) -> None:
        """Connect and create the indexes, so a bad address fails at startup rather than on the
        first request; clears a previous `close()`."""
        await self._driver.open()

    @maps_backend_errors
    async def close(self) -> None:
        await self._driver.close()

    @maps_backend_errors
    async def ping(self) -> Mapping[str, Any]:
        """Round-trip to the server so `/readyz` catches an outage after startup. `ready()` alone
        returns without contacting the server once the indexes exist."""
        await self._driver.ready()
        await self._driver.db.command('ping')
        return {'ok': True}

    @maps_backend_errors
    async def register_conversation(
        self, namespace_key: str, conversation_id: str, *, created_by: str, attrs: Mapping[str, str] | None = None
    ) -> ConversationMeta:
        await self._driver.ready()
        now = _now()
        on_insert: _Document = {
            'namespace_key': namespace_key,
            'created_by': created_by,
            'created_at': now,
            'activity_at': now,
            'last_message_at': None,
            'message_count': 0,
            'title': None,
            'deleted_at': None,
            'attrs': [{'k': key, 'v': value} for key, value in (attrs or {}).items()],
            'purged': False,
        }
        try:
            doc = await self._driver.conversations.find_one_and_update(
                {'_id': conversation_id},
                {'$setOnInsert': on_insert},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError:
            # Concurrent upserts on one `_id` can collide even with `upsert=True`; the winner's
            # document is already committed, so re-reading is the whole recovery.
            doc = await self._driver.conversations.find_one({'_id': conversation_id})
        if doc is None or doc['purged'] or doc['namespace_key'] != namespace_key or doc['deleted_at'] is not None:
            raise ConversationNotFound(conversation_id)
        return _meta(doc)

    @maps_backend_errors
    async def get(self, namespace_key: str, conversation_id: str) -> ConversationMeta | None:
        await self._driver.ready()
        doc = await self._driver.conversations.find_one(
            {'_id': conversation_id, 'namespace_key': namespace_key, **_LIVE}
        )
        return None if doc is None else _meta(doc)

    async def _require_live(self, namespace_key: str, conversation_id: str) -> None:
        found = await self._driver.conversations.find_one(
            {'_id': conversation_id, 'namespace_key': namespace_key, **_LIVE}, {'_id': 1}
        )
        if found is None:
            raise ConversationNotFound(conversation_id)

    @maps_backend_errors
    async def set_title(self, namespace_key: str, conversation_id: str, title: str) -> None:
        await self._driver.ready()
        result = await self._driver.conversations.update_one(
            {'_id': conversation_id, 'namespace_key': namespace_key, **_LIVE}, {'$set': {'title': title}}
        )
        if result.matched_count == 0:
            raise ConversationNotFound(conversation_id)

    @maps_backend_errors
    async def update_message_stats(
        self, namespace_key: str, conversation_id: str, *, last_message_at: datetime, message_count_delta: int
    ) -> None:
        await self._driver.ready()
        result = await self._driver.conversations.update_one(
            {'_id': conversation_id, 'namespace_key': namespace_key, **_LIVE},
            {
                # `activity_at` mirrors the same value: it is the list's sort key, and Mongo
                # cannot index the `COALESCE` the SQL backends order by.
                '$set': {'last_message_at': last_message_at, 'activity_at': last_message_at},
                '$inc': {'message_count': message_count_delta},
            },
        )
        if result.matched_count == 0:
            raise ConversationNotFound(conversation_id)

    @maps_backend_errors
    async def soft_delete(self, namespace_key: str, conversation_id: str) -> ConversationMeta:
        await self._driver.ready()
        doc = await self._driver.conversations.find_one_and_update(
            {'_id': conversation_id, 'namespace_key': namespace_key, **_LIVE},
            {'$set': {'deleted_at': _now()}},
            return_document=ReturnDocument.AFTER,
        )
        if doc is None:
            raise ConversationNotFound(conversation_id)
        return _meta(doc)

    @maps_backend_errors
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
        await self._driver.ready()
        query: _Document = {'namespace_key': namespace_key, **_LIVE}
        if attrs:
            # One `$elemMatch` per pair under `$all` gives AND semantics with key and value bound,
            # so a key holding a dot filters on itself instead of reading as a field path.
            query['attrs'] = {'$all': [{'$elemMatch': {'k': key, 'v': value}} for key, value in attrs.items()]}
        if page_after is not None:
            # Resolved inside the same filtered set, so an unknown, deleted, or attrs-excluded
            # cursor ends the list rather than raising.
            cursor_doc = await self._driver.conversations.find_one({**query, '_id': page_after})
            if cursor_doc is None:
                return []
            activity_at, cursor_id = cursor_doc['activity_at'], cursor_doc['_id']
            query['$or'] = [
                {'activity_at': {'$lt': activity_at}},
                {'activity_at': activity_at, '_id': {'$lt': cursor_id}},
            ]
        found = self._driver.conversations.find(query).sort([('activity_at', -1), ('_id', -1)]).limit(limit)
        return [_meta(doc) for doc in await found.to_list(length=limit)]

    @maps_backend_errors
    async def purge(self, namespace_key: str, conversation_id: str) -> bool:
        await self._driver.ready()
        doc = await self._driver.conversations.find_one({'_id': conversation_id})
        if doc is None or doc['namespace_key'] != namespace_key:
            return False
        # A tombstone, never a delete: the `_id` stays claimed against every namespace forever,
        # the previous owner included. It keeps `namespace_key`, so a repeated purge by the owner
        # still answers True. `deleted_at` keeps a soft delete's stamp when there is one, decided in
        # the statement itself, so a soft delete racing this purge is not overwritten.
        await self._driver.conversations.update_one(
            {'_id': conversation_id},
            [
                {
                    '$set': {
                        'purged': True,
                        'title': None,
                        'attrs': [],
                        'deleted_at': {'$ifNull': ['$deleted_at', _now()]},
                    }
                }
            ],
        )
        await self._driver.turns.delete_many({'conversation_id': conversation_id})
        return True

    @maps_backend_errors
    async def start_turn(
        self,
        namespace_key: str,
        conversation_id: str,
        *,
        run_id: str,
        author_user_id: str,
        previous_run_id: str | None = None,
    ) -> None:
        await self._driver.ready()
        await self._require_live(namespace_key, conversation_id)
        await self._driver.turns.insert_one(
            {
                'conversation_id': conversation_id,
                'run_id': run_id,
                'author_user_id': author_user_id,
                'at': _now(),
                'phase': 'started',
                'previous_run_id': previous_run_id,
            }
        )

    @maps_backend_errors
    async def set_turn_phase(
        self, namespace_key: str, conversation_id: str, *, run_id: str, phase: RunPhase, error: TurnError | None = None
    ) -> None:
        await self._driver.ready()
        await self._require_live(namespace_key, conversation_id)
        stored_error = None if error is None else dataclasses.asdict(error)
        # An unknown run_id matches nothing: the contract's silent no-op.
        await self._driver.turns.update_one(
            {'conversation_id': conversation_id, 'run_id': run_id}, {'$set': {'phase': phase, 'error': stored_error}}
        )

    @maps_backend_errors
    async def list_turns(self, namespace_key: str, conversation_id: str) -> Sequence[TurnRecord]:
        await self._driver.ready()
        await self._require_live(namespace_key, conversation_id)
        # `(at, _id)` stands in for SQLite's autoincrement `seq`: ObjectId ordering is only
        # second-granular across processes, while BSON keeps `at` at millisecond precision.
        cursor = self._driver.turns.find({'conversation_id': conversation_id}).sort([('at', 1), ('_id', 1)])
        docs = await cursor.to_list(length=None)
        return [
            TurnRecord(
                run_id=doc['run_id'],
                author_user_id=doc['author_user_id'],
                # NOT NULL by construction: `start_turn` always stamps `at` -- see `_meta`'s
                # identical `created_at` cast for the same reasoning.
                at=cast(datetime, _utc(doc['at'])),
                phase=doc['phase'],
                # Documents written by an earlier version carry neither `previous_run_id` nor
                # `error`, and `start_turn` writes no `error`, hence `.get`.
                previous_run_id=doc.get('previous_run_id'),
                error=None if doc.get('error') is None else TurnError(**doc['error']),
            )
            for doc in docs
        ]
