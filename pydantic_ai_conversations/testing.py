"""Conformance suite every ConversationRegistry implementation must pass.

Usage in a backend's test suite:

    class TestPostgresRegistry(RegistryContract):
        @pytest.fixture
        def registry(self) -> ConversationRegistry:
            return PostgresRegistry(pool)

Requires pytest (and pytest-asyncio in auto mode) in the consumer's dev environment.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from pydantic_ai_conversations.errors import ConversationNotFound
from pydantic_ai_conversations.feedback.store import TurnFeedbackStore
from pydantic_ai_conversations.registry import ConversationRegistry, TurnError

_NS_A = '[["user.tenant_id","t1"]]'
_NS_B = '[["user.tenant_id","t2"]]'

_CONCURRENCY = 20
"""Callers raced against one entry in the concurrency tests. Enough to make a lost update certain
rather than probable, small enough for a modest connection pool to absorb."""


def _now() -> datetime:
    return datetime.now(UTC)


class RegistryContract:
    """Subclass and override the `registry` fixture with your implementation.

    Tests mint fresh conversation ids, so the suite is safe against persistent backends without
    per-test cleanup; the `registry` fixture may hand back a shared, non-empty store.
    """

    @pytest.fixture
    def registry(self) -> ConversationRegistry:  # pragma: no cover - consumers must override
        raise NotImplementedError('override the `registry` fixture')

    async def test_register_is_idempotent(self, registry: ConversationRegistry) -> None:
        cid = uuid4().hex
        created = await registry.register_conversation(_NS_A, cid, created_by='alice')
        again = await registry.register_conversation(_NS_A, cid, created_by='bob')
        assert (created.created_by, again.created_by) == ('alice', 'alice')

    async def test_foreign_namespace_is_invisible(self, registry: ConversationRegistry) -> None:
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        with pytest.raises(ConversationNotFound):
            await registry.register_conversation(_NS_B, cid, created_by='mallory')
        assert await registry.get(_NS_B, cid) is None
        assert list(await registry.list(_NS_B)) == []

    async def test_every_mutation_is_not_found_shaped_for_foreign_namespace(
        self, registry: ConversationRegistry
    ) -> None:
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        with pytest.raises(ConversationNotFound):
            await registry.set_title(_NS_B, cid, 'x')
        with pytest.raises(ConversationNotFound):
            await registry.update_message_stats(_NS_B, cid, last_message_at=_now(), message_count_delta=1)
        with pytest.raises(ConversationNotFound):
            await registry.soft_delete(_NS_B, cid)
        with pytest.raises(ConversationNotFound):
            await registry.start_turn(_NS_B, cid, run_id='r', author_user_id='mallory')
        with pytest.raises(ConversationNotFound):
            await registry.set_turn_phase(_NS_B, cid, run_id='r', phase='answered')
        with pytest.raises(ConversationNotFound):
            await registry.list_turns(_NS_B, cid)
        # Foreign purge is a silent no-op: the owner's entry survives, and the return says so.
        assert await registry.purge(_NS_B, cid) is False
        assert await registry.get(_NS_A, cid) is not None

    async def test_soft_delete_hides_without_revival(self, registry: ConversationRegistry) -> None:
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        meta = await registry.soft_delete(_NS_A, cid)
        assert meta.deleted_at is not None
        assert await registry.get(_NS_A, cid) is None
        with pytest.raises(ConversationNotFound):
            await registry.register_conversation(_NS_A, cid, created_by='alice')

    async def test_double_soft_delete_raises(self, registry: ConversationRegistry) -> None:
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        await registry.soft_delete(_NS_A, cid)
        with pytest.raises(ConversationNotFound):
            await registry.soft_delete(_NS_A, cid)  # already gone

    async def test_listing_orders_by_activity(self, registry: ConversationRegistry) -> None:
        cid1, cid2 = uuid4().hex, uuid4().hex
        await registry.register_conversation(_NS_A, cid1, created_by='alice')
        await registry.register_conversation(_NS_A, cid2, created_by='alice')
        # One second ahead, because MongoDB stores milliseconds and a tie with cid2's creation sorts by id.
        later = _now() + timedelta(seconds=1)
        await registry.update_message_stats(_NS_A, cid1, last_message_at=later, message_count_delta=2)
        listed = await registry.list(_NS_A)
        assert [m.conversation_id for m in listed][0] == cid1

    async def test_unknown_cursor_means_end_of_list(self, registry: ConversationRegistry) -> None:
        cid1, cid2 = uuid4().hex, uuid4().hex
        await registry.register_conversation(_NS_A, cid1, created_by='alice')
        await registry.register_conversation(_NS_A, cid2, created_by='alice')
        assert await registry.list(_NS_A, page_after=uuid4().hex) == []

    async def test_pagination_after_cursor(self, registry: ConversationRegistry) -> None:
        cid1, cid2, cid3 = uuid4().hex, uuid4().hex, uuid4().hex
        for cid in (cid1, cid2, cid3):
            await registry.register_conversation(_NS_A, cid, created_by='alice')
        # Activity times one second apart, so millisecond storage cannot tie them. cid3 sorts first.
        base = _now()
        for offset, cid in enumerate((cid1, cid2, cid3), start=1):
            at = base + timedelta(seconds=offset)
            await registry.update_message_stats(_NS_A, cid, last_message_at=at, message_count_delta=1)
        page1 = await registry.list(_NS_A, limit=2)
        assert [m.conversation_id for m in page1] == [cid3, cid2]
        page2 = await registry.list(_NS_A, page_after=cid2)
        page2_ids = [m.conversation_id for m in page2]
        assert page2_ids[0] == cid1
        assert cid2 not in page2_ids and cid3 not in page2_ids

    async def test_limit_must_be_positive(self, registry: ConversationRegistry) -> None:
        with pytest.raises(ValueError):
            await registry.list(_NS_A, limit=0)
        with pytest.raises(ValueError):
            await registry.list(_NS_A, limit=-1)

    async def test_list_filters_on_attrs(self, registry: ConversationRegistry) -> None:
        cid_web, cid_twilio = uuid4().hex, uuid4().hex
        await registry.register_conversation(_NS_A, cid_web, created_by='alice', attrs={'source': 'web', 'lang': 'en'})
        await registry.register_conversation(_NS_A, cid_twilio, created_by='alice', attrs={'source': 'twilio'})
        listed = await registry.list(_NS_A, attrs={'source': 'web'})
        assert [m.conversation_id for m in listed] == [cid_web]
        # AND semantics: a second key that doesn't match must exclude the row entirely.
        assert await registry.list(_NS_A, attrs={'source': 'web', 'lang': 'de'}) == []

    async def test_equal_timestamps_order_deterministically(self, registry: ConversationRegistry) -> None:
        cid1, cid2, cid3 = uuid4().hex, uuid4().hex, uuid4().hex
        same_time = _now()
        for cid in (cid1, cid2, cid3):
            await registry.register_conversation(_NS_A, cid, created_by='alice')
            await registry.update_message_stats(_NS_A, cid, last_message_at=same_time, message_count_delta=1)
        listed = await registry.list(_NS_A, limit=3)
        assert [m.conversation_id for m in listed] == sorted((cid1, cid2, cid3), reverse=True)

    async def test_update_message_stats_and_title_roundtrip(self, registry: ConversationRegistry) -> None:
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        await registry.update_message_stats(_NS_A, cid, last_message_at=_now(), message_count_delta=2)
        await registry.set_title(_NS_A, cid, 'Pump diagnosis')
        meta = await registry.get(_NS_A, cid)
        assert meta is not None and (meta.message_count, meta.title) == (2, 'Pump diagnosis')

    async def test_turn_attribution_roundtrip(self, registry: ConversationRegistry) -> None:
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        await registry.start_turn(_NS_A, cid, run_id='r1', author_user_id='alice')
        await registry.start_turn(_NS_A, cid, run_id='r2', author_user_id='bob')
        assert [t.author_user_id for t in await registry.list_turns(_NS_A, cid)] == ['alice', 'bob']

    async def test_turn_lineage_roundtrip(self, registry: ConversationRegistry) -> None:
        # `previous_run_id` is stored as given and read back; the first turn carries `None`.
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        await registry.start_turn(_NS_A, cid, run_id='r1', author_user_id='alice')
        await registry.start_turn(_NS_A, cid, run_id='r2', author_user_id='alice', previous_run_id='r1')
        turns = await registry.list_turns(_NS_A, cid)
        assert [(t.run_id, t.previous_run_id) for t in turns] == [('r1', None), ('r2', 'r1')]

    async def test_turn_error_roundtrip(self, registry: ConversationRegistry) -> None:
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        await registry.start_turn(_NS_A, cid, run_id='r1', author_user_id='alice')
        await registry.start_turn(_NS_A, cid, run_id='r2', author_user_id='alice', previous_run_id='r1')
        assert [t.error for t in await registry.list_turns(_NS_A, cid)] == [None, None]
        error = TurnError('model-unavailable', 'The model is temporarily unavailable.', transient=True)
        await registry.set_turn_phase(_NS_A, cid, run_id='r1', phase='failed', error=error)
        await registry.set_turn_phase(_NS_A, cid, run_id='r2', phase='failed')
        assert [(t.phase, t.error) for t in await registry.list_turns(_NS_A, cid)] == [
            ('failed', error),
            ('failed', None),
        ]

    async def test_set_turn_phase_overwrites_the_stored_error(self, registry: ConversationRegistry) -> None:
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        await registry.start_turn(_NS_A, cid, run_id='r1', author_user_id='alice')
        await registry.set_turn_phase(_NS_A, cid, run_id='r1', phase='failed', error=TurnError('internal', 'x'))
        await registry.set_turn_phase(_NS_A, cid, run_id='r1', phase='failed')
        assert [t.error for t in await registry.list_turns(_NS_A, cid)] == [None]

    async def test_turn_phase_defaults_to_started_and_updates(self, registry: ConversationRegistry) -> None:
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        await registry.start_turn(_NS_A, cid, run_id='r1', author_user_id='alice')
        # A freshly recorded turn is in flight until a terminal hook resolves it.
        assert [t.phase for t in await registry.list_turns(_NS_A, cid)] == ['started']
        await registry.set_turn_phase(_NS_A, cid, run_id='r1', phase='answered')
        assert [t.phase for t in await registry.list_turns(_NS_A, cid)] == ['answered']

    async def test_set_turn_outcome_only_touches_the_named_run(self, registry: ConversationRegistry) -> None:
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        await registry.start_turn(_NS_A, cid, run_id='r1', author_user_id='alice')
        await registry.start_turn(_NS_A, cid, run_id='r2', author_user_id='alice')
        await registry.set_turn_phase(_NS_A, cid, run_id='r2', phase='failed')
        by_run = {t.run_id: t.phase for t in await registry.list_turns(_NS_A, cid)}
        assert by_run == {'r1': 'started', 'r2': 'failed'}

    async def test_set_turn_outcome_on_an_unknown_run_is_a_noop(self, registry: ConversationRegistry) -> None:
        # A terminal hook may fire for a run whose `start_turn` never landed (a crash between the
        # two registry writes): resolving an absent turn must not raise, just do nothing.
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        await registry.set_turn_phase(_NS_A, cid, run_id='never-recorded', phase='answered')
        assert list(await registry.list_turns(_NS_A, cid)) == []

    async def test_set_turn_outcome_is_idempotent(self, registry: ConversationRegistry) -> None:
        # `after_run` and `on_run_error` each write once, and a backend must tolerate the same write
        # arriving twice from a retried hook: it lands on one row rather than appending a second.
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        await registry.start_turn(_NS_A, cid, run_id='r1', author_user_id='alice')
        await registry.set_turn_phase(_NS_A, cid, run_id='r1', phase='failed')
        await registry.set_turn_phase(_NS_A, cid, run_id='r1', phase='failed')
        assert [(t.run_id, t.phase) for t in await registry.list_turns(_NS_A, cid)] == [('r1', 'failed')]

    async def test_concurrent_message_stats_updates_do_not_lose_increments(
        self, registry: ConversationRegistry
    ) -> None:
        # `update_message_stats` must be an atomic increment (`SET count = count + delta`). A read-modify-write
        # backend loses updates here while passing every sequential test in this suite.
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        await asyncio.gather(
            *(
                registry.update_message_stats(_NS_A, cid, last_message_at=_now(), message_count_delta=1)
                for _ in range(_CONCURRENCY)
            )
        )
        meta = await registry.get(_NS_A, cid)
        assert meta is not None and meta.message_count == _CONCURRENCY

    async def test_concurrent_registers_agree_on_one_winner(self, registry: ConversationRegistry) -> None:
        # The global id claim must be atomic (`INSERT ... ON CONFLICT DO NOTHING` then re-select):
        # exactly one creator wins and every other caller is handed that same entry. Two distinct
        # `created_by` values mean two callers each believe they own the conversation.
        cid = uuid4().hex
        metas = await asyncio.gather(
            *(registry.register_conversation(_NS_A, cid, created_by=f'creator-{i}') for i in range(_CONCURRENCY))
        )
        assert len({m.created_by for m in metas}) == 1

    async def test_list_turns_requires_a_live_entry(self, registry: ConversationRegistry) -> None:
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        await registry.start_turn(_NS_A, cid, run_id='r1', author_user_id='alice')
        await registry.soft_delete(_NS_A, cid)
        with pytest.raises(ConversationNotFound):
            await registry.list_turns(_NS_A, cid)

    async def test_purge_keeps_the_global_id_claimed(self, registry: ConversationRegistry) -> None:
        # Substrate deletion is optional and best-effort (`SupportsRunDeletion`); a released id
        # would let a foreign namespace inherit any surviving history.
        cid = uuid4().hex
        await registry.register_conversation(_NS_A, cid, created_by='alice')
        assert await registry.purge(_NS_A, cid) is True
        assert await registry.get(_NS_A, cid) is None
        with pytest.raises(ConversationNotFound):
            await registry.register_conversation(_NS_B, cid, created_by='bob')
        with pytest.raises(ConversationNotFound):
            await registry.register_conversation(_NS_A, cid, created_by='alice')  # owner included
        # A repeat purge of an id this namespace already owns (now tombstoned) still reports True:
        # idempotent from the caller's perspective, not a "did I just delete something new" signal.
        assert await registry.purge(_NS_A, cid) is True

    async def test_rejection_shape_is_indistinguishable_between_foreign_and_deleted(
        self, registry: ConversationRegistry
    ) -> None:
        foreign_cid, deleted_cid = uuid4().hex, uuid4().hex
        await registry.register_conversation(_NS_A, foreign_cid, created_by='alice')
        await registry.register_conversation(_NS_A, deleted_cid, created_by='alice')
        await registry.soft_delete(_NS_A, deleted_cid)
        with pytest.raises(ConversationNotFound) as foreign:
            await registry.register_conversation(_NS_B, foreign_cid, created_by='mallory')
        with pytest.raises(ConversationNotFound) as deleted:
            await registry.register_conversation(_NS_A, deleted_cid, created_by='alice')
        assert foreign.value.args == (foreign_cid,)
        assert deleted.value.args == (deleted_cid,)

    async def test_ping_reports_health(self, registry: ConversationRegistry) -> None:
        await registry.open()
        assert await registry.ping() is not None
        await registry.close()


class TurnFeedbackStoreContract:
    """Subclass and override the `feedback_store` fixture with your implementation.

    Tests mint fresh conversation ids and run ids, so the suite is safe against a persistent,
    non-empty store.
    """

    @pytest.fixture
    def feedback_store(self) -> TurnFeedbackStore:  # pragma: no cover - consumers must override
        raise NotImplementedError('override the `feedback_store` fixture')

    async def test_upsert_creates_then_replaces_the_callers_entry(self, feedback_store: TurnFeedbackStore) -> None:
        cid, run_id = uuid4().hex, uuid4().hex
        first = await feedback_store.upsert(
            kind='report',
            namespace_key=_NS_A,
            conversation_id=cid,
            run_id=run_id,
            author_user_id='alice',
            data={'category': 'wrong', 'remarks': 'a'},
        )
        second = await feedback_store.upsert(
            kind='report',
            namespace_key=_NS_A,
            conversation_id=cid,
            run_id=run_id,
            author_user_id='alice',
            data={'category': 'wrong', 'remarks': 'b'},
        )
        assert second.data == {'category': 'wrong', 'remarks': 'b'}
        assert second.created_at == first.created_at and second.updated_at >= first.updated_at
        assert len(await feedback_store.list(kind='report', namespace_key=_NS_A, conversation_id=cid)) == 1

    async def test_entries_are_per_author_and_per_kind(self, feedback_store: TurnFeedbackStore) -> None:
        cid, run_id = uuid4().hex, uuid4().hex
        for author in ('alice', 'bob'):
            await feedback_store.upsert(
                kind='report', namespace_key=_NS_A, conversation_id=cid, run_id=run_id, author_user_id=author, data={}
            )
        await feedback_store.upsert(
            kind='rating',
            namespace_key=_NS_A,
            conversation_id=cid,
            run_id=run_id,
            author_user_id='alice',
            data={'up': True},
        )
        reports = await feedback_store.by_run_ids(
            kind='report', namespace_key=_NS_A, conversation_id=cid, run_ids=[run_id]
        )
        assert sorted(r.author_user_id for r in reports) == ['alice', 'bob']
        mine = await feedback_store.by_run_ids(
            kind='report', namespace_key=_NS_A, conversation_id=cid, run_ids=[run_id], author_user_id='alice'
        )
        assert [r.author_user_id for r in mine] == ['alice']
        assert len(await feedback_store.list(kind='rating', namespace_key=_NS_A, conversation_id=cid)) == 1

    async def test_by_run_ids_returns_only_the_named_runs(self, feedback_store: TurnFeedbackStore) -> None:
        cid, wanted, other = uuid4().hex, uuid4().hex, uuid4().hex
        for run_id in (wanted, other):
            await feedback_store.upsert(
                kind='report', namespace_key=_NS_A, conversation_id=cid, run_id=run_id, author_user_id='alice', data={}
            )
        found = await feedback_store.by_run_ids(
            kind='report', namespace_key=_NS_A, conversation_id=cid, run_ids=[wanted]
        )
        assert [r.run_id for r in found] == [wanted]
        assert (
            await feedback_store.by_run_ids(kind='report', namespace_key=_NS_A, conversation_id=cid, run_ids=[]) == []
        )

    async def test_foreign_namespace_sees_nothing(self, feedback_store: TurnFeedbackStore) -> None:
        cid, run_id = uuid4().hex, uuid4().hex
        await feedback_store.upsert(
            kind='report', namespace_key=_NS_A, conversation_id=cid, run_id=run_id, author_user_id='alice', data={}
        )
        assert await feedback_store.list(kind='report', namespace_key=_NS_B, conversation_id=cid) == []
        assert not await feedback_store.delete(
            kind='report', namespace_key=_NS_B, conversation_id=cid, run_id=run_id, author_user_id='alice'
        )

    async def test_delete_reports_whether_an_entry_existed(self, feedback_store: TurnFeedbackStore) -> None:
        cid, run_id = uuid4().hex, uuid4().hex
        await feedback_store.upsert(
            kind='report', namespace_key=_NS_A, conversation_id=cid, run_id=run_id, author_user_id='alice', data={}
        )
        assert (
            await feedback_store.delete(
                kind='report', namespace_key=_NS_A, conversation_id=cid, run_id=run_id, author_user_id='alice'
            )
            is True
        )
        assert (
            await feedback_store.delete(
                kind='report', namespace_key=_NS_A, conversation_id=cid, run_id=run_id, author_user_id='alice'
            )
            is False
        )

    async def test_delete_conversation_removes_every_kind(self, feedback_store: TurnFeedbackStore) -> None:
        cid, run_id = uuid4().hex, uuid4().hex
        for kind in ('report', 'rating'):
            await feedback_store.upsert(
                kind=kind, namespace_key=_NS_A, conversation_id=cid, run_id=run_id, author_user_id='alice', data={}
            )
        assert await feedback_store.delete_conversation(namespace_key=_NS_A, conversation_id=cid) == 2
        assert await feedback_store.delete_conversation(namespace_key=_NS_A, conversation_id=cid) == 0
        assert await feedback_store.list(kind='report', namespace_key=_NS_A, conversation_id=cid) == []

    async def test_timestamps_are_aware_utc(self, feedback_store: TurnFeedbackStore) -> None:
        record = await feedback_store.upsert(
            kind='report',
            namespace_key=_NS_A,
            conversation_id=uuid4().hex,
            run_id=uuid4().hex,
            author_user_id='alice',
            data={},
        )
        assert record.created_at.tzinfo is not None and record.updated_at.tzinfo is not None

    async def test_ping_reports_health(self, feedback_store: TurnFeedbackStore) -> None:
        assert (await feedback_store.ping())['ok'] is True

    async def test_the_same_turn_and_author_in_another_namespace_is_another_entry(
        self, feedback_store: TurnFeedbackStore
    ) -> None:
        """Identity includes the namespace and the conversation: an upsert never moves a row."""
        cid, run_id = uuid4().hex, uuid4().hex
        await feedback_store.upsert(
            kind='report',
            namespace_key=_NS_A,
            conversation_id=cid,
            run_id=run_id,
            author_user_id='alice',
            data={'n': 'a'},
        )
        await feedback_store.upsert(
            kind='report',
            namespace_key=_NS_B,
            conversation_id=cid,
            run_id=run_id,
            author_user_id='alice',
            data={'n': 'b'},
        )
        (in_a,) = await feedback_store.list(kind='report', namespace_key=_NS_A, conversation_id=cid)
        (in_b,) = await feedback_store.list(kind='report', namespace_key=_NS_B, conversation_id=cid)
        assert (in_a.data, in_b.data) == ({'n': 'a'}, {'n': 'b'})

    async def test_list_is_ordered_by_creation_then_author(self, feedback_store: TurnFeedbackStore) -> None:
        cid, run_id = uuid4().hex, uuid4().hex
        for author in ('bob', 'alice'):
            await feedback_store.upsert(
                kind='report', namespace_key=_NS_A, conversation_id=cid, run_id=run_id, author_user_id=author, data={}
            )
        rows = await feedback_store.list(kind='report', namespace_key=_NS_A, conversation_id=cid)
        assert [r.author_user_id for r in rows] == sorted(
            (r.author_user_id for r in rows),
            key=lambda a: (next(x.created_at for x in rows if x.author_user_id == a), a),
        )
