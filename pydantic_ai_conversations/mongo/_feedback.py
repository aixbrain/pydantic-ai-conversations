"""`TurnFeedbackStore` on MongoDB: one `feedback` collection, documents keyed by (kind, run_id, author)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from pymongo import AsyncMongoClient, ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from pydantic_ai_conversations.feedback.store import TurnFeedbackRecord
from pydantic_ai_conversations.mongo._driver import STEP_STORE_TRANSIENT_ERRORS

_Document = dict[str, Any]


def _utc(value: datetime) -> datetime:
    # MongoDB returns UTC naive unless the client is `tz_aware`; stamp the zone rather than assume.
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _record(doc: _Document) -> TurnFeedbackRecord:
    return TurnFeedbackRecord(
        kind=doc['kind'],
        namespace_key=doc['namespace_key'],
        conversation_id=doc['conversation_id'],
        run_id=doc['run_id'],
        author_user_id=doc['author_user_id'],
        created_at=_utc(doc['created_at']),
        updated_at=_utc(doc['updated_at']),
        data=doc['data'],
    )


class MongoTurnFeedbackStore:
    """`TurnFeedbackStore` on a MongoDB database, sharing the registry's client when given the same one.

    Whoever opens the client closes it: with `db_url=` this store created it, `close()` closes it,
    and `open()` makes a new one (a closed pymongo client is not reusable); with `client=` it is
    the app's and stays open. Indexes are created on `open()` or first use. `transient_errors` is
    the kit's set: a duplicate key or an operation failure is not retryable.
    """

    transient_errors = STEP_STORE_TRANSIENT_ERRORS

    def __init__(
        self, *, client: AsyncMongoClient[_Document] | None = None, db_url: str | None = None, database: str
    ) -> None:
        if (client is None) == (db_url is None):
            raise ValueError('provide exactly one of `client=` or `db_url=`')
        self._own_client = client is None
        self._db_url = db_url
        self._client: AsyncMongoClient[_Document] = client if client is not None else AsyncMongoClient(db_url)
        self._database = database
        self._indexes_ready = False
        self._closed = False

    @property
    def _collection(self) -> AsyncCollection[_Document]:
        return self._client[self._database]['feedback']

    async def _ready(self) -> None:
        if self._indexes_ready:
            return
        await self._collection.create_index(
            [('kind', 1), ('namespace_key', 1), ('conversation_id', 1), ('run_id', 1), ('author_user_id', 1)],
            unique=True,
        )
        await self._collection.create_index([('conversation_id', 1), ('kind', 1)])
        self._indexes_ready = True

    async def open(self) -> None:
        if self._closed and self._own_client:
            self._client = AsyncMongoClient(self._db_url)
            self._indexes_ready = False
        self._closed = False
        await self._ready()

    async def close(self) -> None:
        self._closed = True
        if self._own_client:
            await self._client.close()

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
        await self._ready()
        now = datetime.now(UTC)
        doc = await self._collection.find_one_and_update(
            {
                'kind': kind,
                'namespace_key': namespace_key,
                'conversation_id': conversation_id,
                'run_id': run_id,
                'author_user_id': author_user_id,
            },
            {
                '$set': {'updated_at': now, 'data': dict(data)},
                '$setOnInsert': {'created_at': now},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if doc is None:  # pragma: no cover - `upsert=True` with AFTER always returns the document
            raise RuntimeError('upsert returned no document')
        return _record(doc)

    async def delete(
        self, *, kind: str, namespace_key: str, conversation_id: str, run_id: str, author_user_id: str
    ) -> bool:
        await self._ready()
        result = await self._collection.delete_one(
            {
                'kind': kind,
                'run_id': run_id,
                'author_user_id': author_user_id,
                'namespace_key': namespace_key,
                'conversation_id': conversation_id,
            }
        )
        return result.deleted_count > 0

    async def by_run_ids(
        self,
        *,
        kind: str,
        namespace_key: str,
        conversation_id: str,
        run_ids: Sequence[str],
        author_user_id: str | None = None,
    ) -> Sequence[TurnFeedbackRecord]:
        if not run_ids:
            return []
        await self._ready()
        query: _Document = {
            'kind': kind,
            'namespace_key': namespace_key,
            'conversation_id': conversation_id,
            'run_id': {'$in': list(run_ids)},
        }
        if author_user_id is not None:
            query['author_user_id'] = author_user_id
        return [_record(doc) async for doc in self._collection.find(query)]

    async def list(self, *, kind: str, namespace_key: str, conversation_id: str) -> Sequence[TurnFeedbackRecord]:
        await self._ready()
        cursor = self._collection.find(
            {'kind': kind, 'namespace_key': namespace_key, 'conversation_id': conversation_id}
        ).sort([('created_at', 1), ('author_user_id', 1)])
        return [_record(doc) async for doc in cursor]

    async def delete_conversation(self, *, namespace_key: str, conversation_id: str) -> int:
        await self._ready()
        result = await self._collection.delete_many(
            {'namespace_key': namespace_key, 'conversation_id': conversation_id}
        )
        return result.deleted_count

    async def ping(self) -> Mapping[str, Any]:
        await self._ready()
        await self._client[self._database].command('ping')
        return {'ok': True}
