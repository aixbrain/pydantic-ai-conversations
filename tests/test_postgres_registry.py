"""What only a real Postgres can prove, and what the conformance suite does not cover."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from psycopg import AsyncConnection, sql
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from pydantic_ai_conversations.errors import ConversationNotFound, RegistryUnavailable
from pydantic_ai_conversations.postgres import PostgresConversationRegistry
from pydantic_ai_conversations.registry import ConversationRegistry
from pydantic_ai_conversations.testing import RegistryContract

_NS = '[["user.tenant_id","t1"]]'
_NS_B = '[["user.tenant_id","t2"]]'


def _registry(dsn: str) -> PostgresConversationRegistry:
    return PostgresConversationRegistry(dsn=dsn, schema=uuid4().hex[:8], create_schema=True)


async def test_get_lazily_opens_the_pool_without_an_explicit_open_call(postgres_dsn: str) -> None:
    """`open()` is optional: a registry nobody opened must still serve, opening (and probing the
    schema) on first use via `PostgresDriver.ready()`."""
    registry = _registry(postgres_dsn)
    assert await registry.get(_NS, uuid4().hex) is None
    await registry.close()


async def test_ping_reports_status_only(postgres_dsn: str) -> None:
    """The report is public (`/readyz` is unauthenticated), so it carries no counts."""
    registry = _registry(postgres_dsn)
    assert await registry.ping() == {'ok': True}
    await registry.register_conversation(_NS, uuid4().hex, created_by='alice')
    assert await registry.ping() == {'ok': True}
    await registry.close()


async def test_ping_raises_registry_unavailable_when_the_table_is_dropped_after_open(postgres_dsn: str) -> None:
    """`ready()`'s schema probe runs once, at `open()` -- not on every call. If `conversations` is
    dropped out from under an already-open registry, `ping`'s catalog query returns no row; this
    must surface as `RegistryUnavailable`, not a healthy report over a table that is gone.
    `ping()` is what `/readyz` calls to report backend health, and `errors.py` allows only
    `ConversationNotFound` and `RegistryUnavailable` to escape a registry method."""
    registry = _registry(postgres_dsn)
    assert await registry.ping() == {'ok': True}  # succeeds once, while the table exists
    driver = registry._driver  # pyright: ignore[reportPrivateUsage]
    async with driver.connection() as conn:
        await conn.execute(sql.SQL('DROP TABLE {table}').format(table=driver.table('conversations')))
    with pytest.raises(RegistryUnavailable):
        await registry.ping()
    await registry.close()


async def test_attrs_survive_the_round_trip(postgres_dsn: str) -> None:
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice', attrs={'source': 'web'})
    meta = await registry.get(_NS, cid)
    assert meta is not None and dict(meta.attrs) == {'source': 'web'}
    await registry.close()


async def test_registering_without_attrs_stores_an_empty_object(postgres_dsn: str) -> None:
    """`attrs` is NOT NULL DEFAULT '{}'; an INSERT naming the column must pass Jsonb({}), not None."""
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    meta = await registry.get(_NS, cid)
    assert meta is not None and dict(meta.attrs) == {}
    await registry.close()


async def test_state_survives_reopen(postgres_dsn: str) -> None:
    schema = uuid4().hex[:8]
    first = PostgresConversationRegistry(dsn=postgres_dsn, schema=schema, create_schema=True)
    cid = uuid4().hex
    await first.register_conversation(_NS, cid, created_by='alice', attrs={'source': 'web'})
    await first.close()

    second = PostgresConversationRegistry(dsn=postgres_dsn, schema=schema)  # no create_schema
    meta = await second.get(_NS, cid)
    assert meta is not None and meta.attrs['source'] == 'web'
    await second.close()


async def test_closed_registry_raises_unavailable_until_reopened(postgres_dsn: str) -> None:
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.close()
    with pytest.raises(RegistryUnavailable):
        await registry.get(_NS, cid)
    with pytest.raises(RegistryUnavailable):
        await registry.ping()
    await registry.open()
    assert await registry.get(_NS, cid) is not None
    await registry.close()


async def test_closed_registry_with_a_borrowed_pool_still_refuses(postgres_dsn: str) -> None:
    """The registry's own closed-state guard, not the pool's: `close()` never tears down a borrowed
    pool, so only `PostgresDriver.ready()`'s `_closed` flag stands between a closed registry and a
    pool that would otherwise happily serve the request. A test built on an owned pool cannot tell
    this guard apart from the pool simply being shut, because `close()` shuts that pool too."""
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        postgres_dsn, kwargs={'row_factory': dict_row}, open=False
    )
    await pool.open(wait=True)
    registry = PostgresConversationRegistry(pool=pool, schema=uuid4().hex[:8], create_schema=True)
    assert await registry.get(_NS, uuid4().hex) is None  # lazily opens: genuinely working first
    await registry.close()
    with pytest.raises(RegistryUnavailable):
        await registry.get(_NS, uuid4().hex)
    async with pool.connection() as conn:  # the pool itself is untouched by the registry's close()
        await conn.execute('SELECT 1')
    await pool.close()


async def test_open_on_an_unmigrated_schema_raises_registry_unavailable(postgres_dsn: str) -> None:
    """The default is create_schema=False; a database nobody migrated must fail at startup."""
    registry = PostgresConversationRegistry(dsn=postgres_dsn, schema=uuid4().hex[:8])
    with pytest.raises(RegistryUnavailable):
        await registry.open()
    await registry.close()


async def test_get_on_an_unmigrated_schema_raises_registry_unavailable_without_open(postgres_dsn: str) -> None:
    """The lazy path reports the same failure as the explicit one above -- a caller that skips
    `open()` entirely must not get a confusing error from a bare query against a missing table."""
    registry = PostgresConversationRegistry(dsn=postgres_dsn, schema=uuid4().hex[:8])  # create_schema=False
    with pytest.raises(RegistryUnavailable):
        await registry.get(_NS, uuid4().hex)
    await registry.close()


async def test_message_stats_increment_atomically(postgres_dsn: str) -> None:
    """Two different deltas separate `count = count + n` from a `SET` of the raw delta."""
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    at = datetime(2026, 5, 1, tzinfo=UTC)
    await registry.update_message_stats(_NS, cid, last_message_at=at, message_count_delta=4)
    await registry.update_message_stats(_NS, cid, last_message_at=at, message_count_delta=3)
    meta = await registry.get(_NS, cid)
    assert meta is not None and meta.message_count == 7 and meta.last_message_at == at
    await registry.close()


async def test_update_message_stats_is_last_write_wins_not_monotonic(postgres_dsn: str) -> None:
    """The protocol makes no monotonicity promise for `last_message_at`: a later call with an
    *earlier* timestamp must still win, exactly as a plain `SET` (not a `GREATEST`-guarded one)
    behaves. Distinct from the sibling test above, which reuses the same `at` on both calls and so
    cannot tell a last-write-wins `SET` apart from a monotonic `SET ... WHERE ... >` guard."""
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    later = datetime(2026, 6, 1, tzinfo=UTC)
    earlier = later - timedelta(days=1)
    await registry.update_message_stats(_NS, cid, last_message_at=later, message_count_delta=1)
    await registry.update_message_stats(_NS, cid, last_message_at=earlier, message_count_delta=1)
    meta = await registry.get(_NS, cid)
    assert meta is not None and meta.last_message_at == earlier
    await registry.close()


async def test_every_mutation_is_not_found_for_a_foreign_namespace(postgres_dsn: str) -> None:
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    with pytest.raises(ConversationNotFound):
        await registry.set_title(_NS_B, cid, 'x')
    with pytest.raises(ConversationNotFound):
        await registry.update_message_stats(_NS_B, cid, last_message_at=datetime.now(UTC), message_count_delta=1)
    with pytest.raises(ConversationNotFound):
        await registry.soft_delete(_NS_B, cid)
    with pytest.raises(ConversationNotFound):
        await registry.start_turn(_NS_B, cid, run_id='r', author_user_id='mallory')
    with pytest.raises(ConversationNotFound):
        await registry.set_turn_phase(_NS_B, cid, run_id='r', phase='answered')
    with pytest.raises(ConversationNotFound):
        await registry.list_turns(_NS_B, cid)
    await registry.close()


async def test_every_mutation_is_not_found_for_an_unknown_id(postgres_dsn: str) -> None:
    """The foreign-namespace test above and the soft-delete test below each cover one shape of
    'not ours'; this covers the third -- an id that was never registered at all, in a namespace
    that does hold *some* other live row. That sibling row matters: in a schema with nothing
    registered at all, a query that accidentally drops its `conversation_id = %s` filter (matching
    on `namespace_key` alone) would still correctly find zero rows and this test would not notice --
    confirmed by deleting the filter and rerunning, which passed regardless. With a sibling
    present, the same bug would spuriously match it instead of raising."""
    registry = _registry(postgres_dsn)
    await registry.register_conversation(_NS, uuid4().hex, created_by='alice')  # sibling row in the same namespace
    cid = uuid4().hex
    with pytest.raises(ConversationNotFound):
        await registry.set_title(_NS, cid, 'x')
    with pytest.raises(ConversationNotFound):
        await registry.update_message_stats(_NS, cid, last_message_at=datetime.now(UTC), message_count_delta=1)
    with pytest.raises(ConversationNotFound):
        await registry.soft_delete(_NS, cid)
    with pytest.raises(ConversationNotFound):
        await registry.start_turn(_NS, cid, run_id='r', author_user_id='mallory')
    with pytest.raises(ConversationNotFound):
        await registry.set_turn_phase(_NS, cid, run_id='r', phase='answered')
    with pytest.raises(ConversationNotFound):
        await registry.list_turns(_NS, cid)
    await registry.close()


async def test_turns_keep_record_order(postgres_dsn: str) -> None:
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    for index in range(5):
        await registry.start_turn(_NS, cid, run_id=f'r{index}', author_user_id='alice')
    assert [t.run_id for t in await registry.list_turns(_NS, cid)] == [f'r{i}' for i in range(5)]
    await registry.close()


async def test_set_turn_phase_touches_only_the_named_run(postgres_dsn: str) -> None:
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.start_turn(_NS, cid, run_id='r1', author_user_id='alice')
    await registry.start_turn(_NS, cid, run_id='r2', author_user_id='alice')
    await registry.set_turn_phase(_NS, cid, run_id='r1', phase='answered')
    assert [t.phase for t in await registry.list_turns(_NS, cid)] == ['answered', 'started']
    await registry.set_turn_phase(_NS, cid, run_id='nope', phase='failed')  # silent no-op
    assert [t.phase for t in await registry.list_turns(_NS, cid)] == ['answered', 'started']
    await registry.close()


async def test_soft_delete_hides_and_never_revives(postgres_dsn: str) -> None:
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    deleted = await registry.soft_delete(_NS, cid)
    assert deleted.deleted_at is not None
    assert await registry.get(_NS, cid) is None
    with pytest.raises(ConversationNotFound):
        await registry.soft_delete(_NS, cid)
    with pytest.raises(ConversationNotFound):
        await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.close()


async def test_set_title_round_trips(postgres_dsn: str) -> None:
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.set_title(_NS, cid, 'Pump diagnosis')
    meta = await registry.get(_NS, cid)
    assert meta is not None and meta.title == 'Pump diagnosis'
    await registry.close()


class TestPostgresConversationRegistryConformance(RegistryContract):
    @pytest.fixture
    def registry(self, postgres_dsn: str) -> ConversationRegistry:
        return PostgresConversationRegistry(dsn=postgres_dsn, schema=uuid4().hex[:8], create_schema=True)


async def test_a_never_used_conversation_does_not_float_to_the_top(postgres_dsn: str) -> None:
    """Postgres sorts NULLs FIRST under DESC. A bare `last_message_at DESC` would put a
    conversation nobody has written to at the head of the chat list; COALESCE prevents it."""
    registry = _registry(postgres_dsn)
    used, never = uuid4().hex, uuid4().hex
    await registry.register_conversation(_NS, used, created_by='alice')
    await registry.register_conversation(_NS, never, created_by='alice')
    # Relative to "now", not an absolute date: `never`'s `created_at` is stamped at registration
    # time (real wall-clock "now"), so a hardcoded past-looking literal would eventually sort
    # after it as the calendar caught up -- exactly the false-positive-turned-false-failure this
    # test must not become.
    await registry.update_message_stats(
        _NS, used, last_message_at=datetime.now(UTC) + timedelta(days=1), message_count_delta=1
    )
    assert [m.conversation_id for m in await registry.list(_NS, limit=2)][0] == used
    await registry.close()


async def test_paging_a_filtered_list_stays_inside_the_filter(postgres_dsn: str) -> None:
    """`attrs` and `page_after` together: the cursor resolves inside the filtered set, so one
    naming an excluded conversation ends the list -- the same answer an unknown cursor gets."""
    registry = _registry(postgres_dsn)
    web = [uuid4().hex for _ in range(3)]
    twilio = uuid4().hex
    for index, cid in enumerate([web[0], twilio, web[1], web[2]]):
        source = 'twilio' if cid == twilio else 'web'
        await registry.register_conversation(_NS, cid, created_by='alice', attrs={'source': source})
        await registry.update_message_stats(
            _NS,
            cid,
            last_message_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
            message_count_delta=1,
        )

    newest_first = [web[2], web[1], web[0]]
    page1 = await registry.list(_NS, attrs={'source': 'web'}, limit=2)
    assert [m.conversation_id for m in page1] == newest_first[:2]
    page2 = await registry.list(_NS, attrs={'source': 'web'}, page_after=page1[-1].conversation_id)
    assert [m.conversation_id for m in page2] == newest_first[2:]
    assert await registry.list(_NS, attrs={'source': 'web'}, page_after=twilio) == []
    await registry.close()


async def test_attrs_keys_are_matched_literally_not_as_a_path(postgres_dsn: str) -> None:
    """JSONB containment binds the whole object, so a key holding a dot filters on itself."""
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice', attrs={'a.b': 'v', 'a': 'no'})
    assert [m.conversation_id for m in await registry.list(_NS, attrs={'a.b': 'v'})] == [cid]
    assert await registry.list(_NS, attrs={'a.b.c': 'v'}) == []
    await registry.close()


async def test_purge_keeps_the_id_claimed_and_is_namespace_scoped(postgres_dsn: str) -> None:
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.start_turn(_NS, cid, run_id='r1', author_user_id='alice')
    assert await registry.purge(_NS_B, cid) is False  # foreign purge is a silent no-op
    assert await registry.purge(_NS, cid) is True
    assert await registry.purge(_NS, cid) is True  # repeated purge by the owner still answers True
    assert await registry.purge(_NS_B, uuid4().hex) is False  # unknown id
    with pytest.raises(ConversationNotFound):
        await registry.register_conversation(_NS, cid, created_by='alice')  # dead forever
    await registry.close()


async def test_register_against_a_purged_id_is_not_found_even_for_the_original_owner(postgres_dsn: str) -> None:
    """`register_conversation`'s re-select checks `row['purged']`, a flag only `purge` sets, so this is
    the test that reaches that branch. It must reject the original owner too: purge is
    permanent, not a takeover-blocker."""
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    assert await registry.purge(_NS, cid) is True
    with pytest.raises(ConversationNotFound):
        await registry.register_conversation(_NS, cid, created_by='alice')
    with pytest.raises(ConversationNotFound):
        await registry.register_conversation(_NS_B, cid, created_by='mallory')
    await registry.close()


async def test_register_rejects_purged_even_when_deleted_at_is_independently_null(postgres_dsn: str) -> None:
    """`purge()` always stamps `deleted_at` alongside `purged` (`COALESCE(deleted_at, now())`), so a
    purge through the public API can never leave `purged=TRUE` with `deleted_at IS NULL` -- meaning
    the test above never actually exercises `register_conversation`'s `row['purged']` check on its own: `deleted_at
    IS NOT NULL` alone already explains the rejection. Constructing that combination directly, below
    the registry, proves the `purged` check is doing its own work rather than merely riding along
    with `deleted_at` -- a defense-in-depth line a future writer that clears `deleted_at` without
    knowing purge must stay permanent would otherwise silently defeat."""
    registry = _registry(postgres_dsn)
    cid = uuid4().hex
    await registry.register_conversation(_NS, cid, created_by='alice')
    # Reach below the port, through the same driver the registry itself uses, rather than opening a
    # second connection with its own row-factory typing to juggle.
    driver = registry._driver  # pyright: ignore[reportPrivateUsage]
    table = driver.table('conversations')
    async with driver.connection() as conn:
        await conn.execute(
            sql.SQL('UPDATE {table} SET purged = TRUE WHERE conversation_id = %s').format(table=table), (cid,)
        )
        cur = await conn.execute(
            sql.SQL('SELECT deleted_at FROM {table} WHERE conversation_id = %s').format(table=table), (cid,)
        )
        row = await cur.fetchone()
    assert row is not None and row['deleted_at'] is None  # the constructed state: purged, but not deleted
    with pytest.raises(ConversationNotFound):
        await registry.register_conversation(_NS, cid, created_by='alice')
    await registry.close()
