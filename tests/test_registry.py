"""InMemoryConversationRegistry behavior: the wall, lifecycle, listing, attribution."""

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from pydantic_ai_conversations.errors import ConversationNotFound
from pydantic_ai_conversations.registry import ConversationMeta, ConversationRegistry, InMemoryConversationRegistry

NS_A = '[["user.tenant_id","t1"]]'
NS_B = '[["user.tenant_id","t2"]]'

# Structural conformance pin: InMemoryConversationRegistry must keep satisfying the ConversationRegistry Protocol.
_conforms: ConversationRegistry = InMemoryConversationRegistry()


def _now() -> datetime:
    return datetime.now(UTC)


async def test_register_creates_then_returns_existing() -> None:
    registry = InMemoryConversationRegistry()
    created = await registry.register_conversation(NS_A, 'c1', created_by='alice')
    again = await registry.register_conversation(NS_A, 'c1', created_by='bob')
    assert created.created_by == 'alice'
    assert again.created_by == 'alice'  # existing entry wins; register is idempotent


async def test_foreign_namespace_collision_resolves_to_nothing() -> None:
    registry = InMemoryConversationRegistry()
    await registry.register_conversation(NS_A, 'c1', created_by='alice')
    with pytest.raises(ConversationNotFound):
        await registry.register_conversation(NS_B, 'c1', created_by='mallory')
    assert await registry.get(NS_B, 'c1') is None


async def test_soft_delete_hides_from_get_list_and_register() -> None:
    registry = InMemoryConversationRegistry()
    await registry.register_conversation(NS_A, 'c1', created_by='alice')
    meta = await registry.soft_delete(NS_A, 'c1')
    assert meta.deleted_at is not None
    assert await registry.get(NS_A, 'c1') is None
    assert await registry.list(NS_A) == []
    with pytest.raises(ConversationNotFound):
        await registry.register_conversation(NS_A, 'c1', created_by='alice')  # no revival
    with pytest.raises(ConversationNotFound):
        await registry.soft_delete(NS_A, 'c1')  # second delete: already gone


async def test_list_orders_by_activity_and_paginates() -> None:
    registry = InMemoryConversationRegistry()
    for cid in ('c1', 'c2', 'c3'):
        await registry.register_conversation(NS_A, cid, created_by='alice')
    await registry.update_message_stats(NS_A, 'c1', last_message_at=_now(), message_count_delta=2)
    listed = await registry.list(NS_A)
    # c1's activity was just updated, so it sorts first; c2/c3 fall back to created_at (c3 created later).
    assert [m.conversation_id for m in listed] == ['c1', 'c3', 'c2']
    page = await registry.list(NS_A, page_after='c1', limit=1)
    assert [m.conversation_id for m in page] == ['c3']
    assert await registry.list(NS_A, page_after='unknown') == []


async def test_list_rejects_non_positive_limit() -> None:
    registry = InMemoryConversationRegistry()
    with pytest.raises(ValueError, match='limit must be positive'):
        await registry.list(NS_A, limit=0)
    with pytest.raises(ValueError, match='limit must be positive'):
        await registry.list(NS_A, limit=-1)


async def test_list_filters_on_attrs() -> None:
    registry = InMemoryConversationRegistry()
    await registry.register_conversation(NS_A, 'web-1', created_by='alice', attrs={'source': 'web'})
    await registry.register_conversation(NS_A, 'tw-1', created_by='alice', attrs={'source': 'twilio'})
    listed = await registry.list(NS_A, attrs={'source': 'web'})
    assert [m.conversation_id for m in listed] == ['web-1']


async def test_update_message_stats_and_set_title_require_a_live_entry() -> None:
    registry = InMemoryConversationRegistry()
    await registry.register_conversation(NS_A, 'c1', created_by='alice')
    await registry.update_message_stats(NS_A, 'c1', last_message_at=_now(), message_count_delta=2)
    await registry.set_title(NS_A, 'c1', 'Pump diagnosis')
    meta = await registry.get(NS_A, 'c1')
    assert meta is not None and meta.message_count == 2 and meta.title == 'Pump diagnosis'
    with pytest.raises(ConversationNotFound):
        await registry.update_message_stats(NS_A, 'missing', last_message_at=_now(), message_count_delta=1)
    with pytest.raises(ConversationNotFound):
        await registry.set_title(NS_A, 'missing', 'x')


async def test_turn_attribution() -> None:
    registry = InMemoryConversationRegistry()
    await registry.register_conversation(NS_A, 'c1', created_by='alice')
    await registry.start_turn(NS_A, 'c1', run_id='r1', author_user_id='alice')
    await registry.start_turn(NS_A, 'c1', run_id='r2', author_user_id='bob')
    turns = await registry.list_turns(NS_A, 'c1')
    assert [t.author_user_id for t in turns] == ['alice', 'bob']


async def test_turns_requires_a_live_entry() -> None:
    registry = InMemoryConversationRegistry()
    await registry.register_conversation(NS_A, 'c1', created_by='alice')
    await registry.soft_delete(NS_A, 'c1')
    with pytest.raises(ConversationNotFound):
        await registry.list_turns(NS_A, 'c1')  # soft-deleted
    with pytest.raises(ConversationNotFound):
        await registry.list_turns(NS_B, 'c1')  # foreign


async def test_purge_removes_entry_but_keeps_the_id_claimed() -> None:
    # The substrate (StepStore) keys history globally by conversation_id and has no delete
    # primitive: releasing a purged id would let a foreign namespace inherit its snapshots
    # through the loader. Purged ids are therefore permanently unclaimable.
    registry = InMemoryConversationRegistry()
    await registry.register_conversation(NS_A, 'c1', created_by='alice')
    assert await registry.purge(NS_A, 'c1') is True
    assert await registry.get(NS_A, 'c1') is None
    with pytest.raises(ConversationNotFound):
        await registry.register_conversation(NS_B, 'c1', created_by='bob')
    with pytest.raises(ConversationNotFound):
        await registry.register_conversation(NS_A, 'c1', created_by='alice')  # not even the original owner
    # A repeat purge by the owning namespace is idempotent and still reports ownership.
    assert await registry.purge(NS_A, 'c1') is True


async def test_purge_on_unknown_conversation_is_a_noop() -> None:
    registry = InMemoryConversationRegistry()
    assert await registry.purge(NS_A, 'never-existed') is False  # owner_namespace has no entry to clean up
    assert await registry.get(NS_A, 'never-existed') is None


async def test_conversation_meta_attrs_default_to_an_empty_immutable_mapping() -> None:
    meta = ConversationMeta(namespace_key=NS_A, conversation_id='c1', created_by='alice', created_at=_now())
    assert meta.attrs == {}
    with pytest.raises(TypeError):
        cast(Any, meta.attrs)['x'] = 'y'  # attrs is a Mapping statically; assert the runtime is truly read-only


async def test_register_attrs_are_immutable_and_not_shared_with_the_caller() -> None:
    registry = InMemoryConversationRegistry()
    caller_attrs = {'source': 'web'}
    meta = await registry.register_conversation(NS_A, 'c1', created_by='alice', attrs=caller_attrs)
    caller_attrs['source'] = 'mutated'  # mutating the caller's dict must not affect the stored copy
    assert meta.attrs == {'source': 'web'}
    with pytest.raises(TypeError):
        cast(Any, meta.attrs)['source'] = 'y'


async def test_every_method_answers_not_found_shaped_for_foreign_namespace() -> None:
    registry = InMemoryConversationRegistry()
    await registry.register_conversation(NS_A, 'c1', created_by='alice')
    assert await registry.get(NS_B, 'c1') is None
    assert await registry.list(NS_B) == []
    for op in (
        registry.set_title(NS_B, 'c1', 'x'),
        registry.update_message_stats(NS_B, 'c1', last_message_at=_now(), message_count_delta=1),
        registry.soft_delete(NS_B, 'c1'),
        registry.start_turn(NS_B, 'c1', run_id='r', author_user_id='mallory'),
        registry.list_turns(NS_B, 'c1'),
    ):
        with pytest.raises(ConversationNotFound):
            await op
    # Foreign purge is a silent no-op: entry and claim survive, and the return says so.
    assert await registry.purge(NS_B, 'c1') is False
    assert await registry.get(NS_A, 'c1') is not None
    # The two register rejections are indistinguishable.
    await registry.register_conversation(NS_A, 'c2', created_by='alice')
    await registry.soft_delete(NS_A, 'c2')
    with pytest.raises(ConversationNotFound) as foreign:
        await registry.register_conversation(NS_B, 'c1', created_by='mallory')
    with pytest.raises(ConversationNotFound) as deleted:
        await registry.register_conversation(NS_A, 'c2', created_by='alice')
    assert foreign.value.args == ('c1',) and deleted.value.args == ('c2',)


async def test_lifecycle_and_ping() -> None:
    registry = InMemoryConversationRegistry()
    await registry.open()
    await registry.register_conversation(NS_A, 'c1', created_by='alice')
    assert await registry.ping() == {'ok': True}
    await registry.close()
