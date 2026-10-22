"""The harness's MongoDB step store plus the three service-side extras `stores.py` discovers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic_ai_harness.media import MediaStore
from pydantic_ai_harness.step_persistence import MongoStepStore
from pymongo import AsyncMongoClient
from pymongo.asynchronous.cursor import AsyncCursor

from pydantic_ai_conversations.mongo._driver import STEP_STORE_TRANSIENT_ERRORS

_Document = dict[str, Any]

_RUN_COLLECTIONS = ('runs', 'events', 'snapshots', 'tool_effects')

_REFERENCING = r'^\{'
"""Matches a snapshot's `messages` when it holds `{"message_refs": [sha256, ...]}`. Documents from
harness releases that kept the history inline hold a JSON list there instead."""

_REFS_PER_QUERY = 300
"""Message keys per `$regex` alternation or `$in` list. Each key adds 65 characters, so a pattern
stays well below the roughly 32 KB MongoDB accepts."""

_DEFAULT_MEDIA_THRESHOLD_BYTES = 64 * 1024
"""Mirrors `MongoStepStore`'s own default (`pydantic_ai_harness.step_persistence._store`), kept
as a literal here rather than imported: that name is private to the harness."""


class MongoConversationStepStore(MongoStepStore):
    """The harness's `MongoStepStore` plus `delete_runs`, `ping` and `transient_errors`.

    The harness `StepStore` protocol carries none of them, and `stores.py` discovers all three
    structurally: without `delete_runs` a purge leaves the conversation's runs in the substrate,
    without `ping` `/readyz` says nothing about the store, and without `transient_errors` a
    database outage reaches the client as a 500 instead of a retryable 503. `transient_errors` comes
    from `_driver.py`.

    Media payloads are content-addressed and may be shared across runs, so `delete_runs` leaves
    them to the store's own garbage collection (`stores.py`). Snapshot messages are
    content-addressed too, and `delete_runs` deletes those no other run's snapshot references.

    `database` is required, as in `MongoStepStore`. `media_store`, `media_threshold_bytes` and
    `max_snapshots_per_run` pass straight through to it with its same defaults, so this subclass
    is a strict superset of it.
    """

    transient_errors = STEP_STORE_TRANSIENT_ERRORS

    def __init__(
        self,
        *,
        client: AsyncMongoClient[_Document] | None = None,
        db_url: str | None = None,
        database: str,
        media_store: MediaStore | None | Literal['auto'] = 'auto',
        media_threshold_bytes: int = _DEFAULT_MEDIA_THRESHOLD_BYTES,
        max_snapshots_per_run: int | None = None,
    ) -> None:
        super().__init__(
            client=client,
            db_url=db_url,
            database=database,
            media_store=media_store,
            media_threshold_bytes=media_threshold_bytes,
            max_snapshots_per_run=max_snapshots_per_run,
        )

    async def delete_runs(self, *, run_ids: Sequence[str]) -> None:
        ids = list(run_ids)
        if not ids:
            return
        # `runs` is deleted last. The purge is several deletes with no cross-collection transaction,
        # and `ConversationsFeature.purge` re-finds a run to retry via `list_runs`, which reads
        # `runs`; deleting it first would orphan its dependents for good on a mid-failure. SQLite
        # wraps the same deletes in one transaction and needs no order. The messages go first, as
        # they are found through the snapshots.
        await self._delete_unshared_messages(ids)
        for collection in _RUN_COLLECTIONS[1:]:
            await self._db[collection].delete_many({'run_id': {'$in': ids}})
        # The replay ledger keys its documents by the compound `_id` `{run_id, key}`, so it is
        # matched by a dotted path rather than a `run_id` field. It outlives snapshot pruning by
        # design, so nothing else ever clears it.
        await self._db['snapshot_idempotency_keys'].delete_many({'_id.run_id': {'$in': ids}})
        # `MongoStepStore.register_run` keys `runs` by `_id = run_id` (the module docstring of
        # `pydantic_ai_harness.step_persistence._mongo`), so `runs` is matched by `_id` where the
        # dependents carry a `run_id` field.
        await self._db['runs'].delete_many({'_id': {'$in': ids}})

    async def _delete_unshared_messages(self, run_ids: list[str]) -> None:
        """Delete the stored messages the runs' snapshots reference and no other run's snapshot does.

        `MongoStepStore` keeps each message once in `snapshot_messages`, keyed by its SHA-256, so
        identical messages in two runs are one document. A concurrent save in another run can end up
        referencing a message deleted here; docs/deleting.md says why that is accepted.
        """
        snapshots = self._db['snapshots']
        refs = await _message_refs(
            snapshots.find({'run_id': {'$in': run_ids}, 'messages': {'$regex': _REFERENCING}}, {'messages': 1})
        )
        shared: set[str] = set()
        for batch in _batches(refs):
            pattern = _REFERENCING + '.*(?:' + '|'.join(map(re.escape, batch)) + ')'
            shared |= await _message_refs(
                snapshots.find({'run_id': {'$nin': run_ids}, 'messages': {'$regex': pattern}}, {'messages': 1})
            )
        for batch in _batches(refs - shared):
            await self._db['snapshot_messages'].delete_many({'_id': {'$in': batch}})

    async def ping(self) -> Mapping[str, Any]:
        await self._db.command('ping')
        return {'ok': True}


async def _message_refs(cursor: AsyncCursor[dict[str, object]]) -> set[str]:
    """The stored-message keys the snapshot documents reference."""
    refs: set[str] = set()
    async for doc in cursor:
        refs.update(json.loads(str(doc['messages']))['message_refs'])
    return refs


def _batches(keys: set[str]) -> list[list[str]]:
    ordered = sorted(keys)
    return [ordered[start : start + _REFS_PER_QUERY] for start in range(0, len(ordered), _REFS_PER_QUERY)]
