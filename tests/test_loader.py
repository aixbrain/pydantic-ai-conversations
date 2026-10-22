"""Conversation history = the latest complete snapshot of each run on the turn chain, in chain order."""

from datetime import UTC, datetime

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    InMemoryStepStore,
    RunRecord,
    is_provider_valid,
)

from pydantic_ai_conversations._loader import (
    TranscriptTurn,
    _drop_orphaned_leading_tool_results,
    active_run_ids,
    load_conversation,
    load_transcript,
    path_messages,
)
from pydantic_ai_conversations.registry import TurnRecord


def _partial(text: str) -> list[ModelMessage]:
    """A complete boundary snapshot of a run that settled a tool round but never produced a final
    answer -- ends in a dangling `ModelRequest` (tool returns), exactly the shape a run leaves when
    it crashes on the model request AFTER a settled `CallToolsNode`."""
    return [
        ModelRequest(parts=[UserPromptPart(content=text)]),
        ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='c1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='t', content='result', tool_call_id='c1')]),
    ]


def _turn(text: str) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=text)]),
        ModelResponse(parts=[TextPart(content=f're: {text}')]),
    ]


async def _seed_run(store: InMemoryStepStore, run_id: str, conversation_id: str, text: str, at: datetime) -> None:
    await store.register_run(RunRecord(run_id=run_id, conversation_id=conversation_id, started_at=at))
    await store.save_snapshot(
        ContinuableSnapshot(run_id=run_id, step_index=1, messages=_turn(text), conversation_id=conversation_id)
    )


async def test_unions_snapshots_in_run_start_order() -> None:
    store = InMemoryStepStore()
    await _seed_run(store, 'r2', 'c1', 'second', datetime(2026, 8, 12, 11, tzinfo=UTC))
    await _seed_run(store, 'r1', 'c1', 'first', datetime(2026, 8, 12, 10, tzinfo=UTC))
    messages = await load_conversation(store, conversation_id='c1', run_ids=['r1', 'r2'])
    assert len(messages) == 4
    first_request = messages[0]
    assert isinstance(first_request, ModelRequest)
    assert getattr(first_request.parts[0], 'content', None) == 'first'


async def test_a_message_stored_without_a_run_id_loads_with_its_snapshots_run_id() -> None:
    """A request saved before pydantic-ai sent it has no run id; it is loaded as the run's own, so
    a continuation does not take it for one of its own requests and store it again."""
    store = InMemoryStepStore()
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1', started_at=datetime(2026, 8, 12, tzinfo=UTC)))
    stamped = ModelRequest(parts=[UserPromptPart(content='first')], run_id='r1')
    unstamped = ModelRequest(parts=[ToolReturnPart(tool_name='t', content='result', tool_call_id='c1')])
    call = ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='c1')], run_id='r1')
    await store.save_snapshot(
        ContinuableSnapshot(run_id='r1', step_index=1, messages=[stamped, call, unstamped], conversation_id='c1')
    )

    messages = await load_conversation(store, conversation_id='c1', run_ids=['r1'])

    assert [message.run_id for message in messages] == ['r1', 'r1', 'r1']
    assert messages[0] is stamped and messages[2].parts == unstamped.parts


async def test_max_history_runs_selects_the_newest_runs() -> None:
    """Tail selection: `max_history_runs` keeps the LAST N contributing runs, not the first N -- a
    reversed slice here would silently serve the oldest history instead of the most recent.
    """
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'first', datetime(2026, 8, 12, 10, tzinfo=UTC))
    await _seed_run(store, 'r2', 'c1', 'second', datetime(2026, 8, 12, 11, tzinfo=UTC))
    await _seed_run(store, 'r3', 'c1', 'third', datetime(2026, 8, 12, 12, tzinfo=UTC))
    messages = await load_conversation(store, conversation_id='c1', run_ids=['r1', 'r2', 'r3'], max_history_runs=2)
    flattened = str(messages)
    assert 'first' not in flattened
    assert 'second' in flattened and 'third' in flattened


async def test_max_history_runs_must_be_positive() -> None:
    store = InMemoryStepStore()
    with pytest.raises(ValueError, match='max_history_runs must be positive'):
        await load_conversation(store, conversation_id='c1', run_ids=[], max_history_runs=0)


async def test_max_history_runs_counts_contributing_runs_not_candidates() -> None:
    """A run with no complete snapshot must not consume a `max_history_runs` slot: with 3
    answered turns followed by 2 failed ones that left no complete snapshot, and
    `max_history_runs=2`, the last 2 answered turns load. Slicing candidate indices before reading
    snapshots would give both slots to the failed candidates and return an empty history."""
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'first', datetime(2026, 8, 12, 10, tzinfo=UTC))
    await _seed_run(store, 'r2', 'c1', 'second', datetime(2026, 8, 12, 11, tzinfo=UTC))
    await _seed_run(store, 'r3', 'c1', 'third', datetime(2026, 8, 12, 12, tzinfo=UTC))
    for index, hour in enumerate((13, 14)):
        run_id = f'failed{index}'
        await store.register_run(
            RunRecord(run_id=run_id, conversation_id='c1', started_at=datetime(2026, 8, 12, hour, tzinfo=UTC))
        )
        # A run that left only interrupted saves (a resume without a prompt that crashed), invisible
        # to `latest_snapshot`'s complete-only default: the shape that must not eat a
        # `max_history_runs` slot.
        await store.save_snapshot(
            ContinuableSnapshot(
                run_id=run_id, step_index=0, messages=_turn('failed'), conversation_id='c1', state='interrupted'
            )
        )

    messages = await load_conversation(
        store, conversation_id='c1', run_ids=['r1', 'r2', 'r3', 'failed0', 'failed1'], max_history_runs=2
    )

    flattened = str(messages)
    assert 'failed' not in flattened
    assert 'first' not in flattened
    assert 'second' in flattened and 'third' in flattened


async def test_max_history_runs_keeps_a_continuations_answer_with_the_prompt_it_answers() -> None:
    """A run that resumed a failed turn with an empty list stored only its answer, so its
    snapshot starts with a response and shares the slot of the run holding the prompt; counted
    on its own, a cut of 1 would keep the bare answer, and the leading-response repair would
    leave nothing."""
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'first', datetime(2026, 8, 12, 10, tzinfo=UTC))
    for run_id, hour, messages in (
        ('r2', 11, [ModelRequest(parts=[UserPromptPart(content='second')])]),
        ('r3', 12, [ModelResponse(parts=[TextPart(content='re: second')])]),
    ):
        await store.register_run(
            RunRecord(run_id=run_id, conversation_id='c1', started_at=datetime(2026, 8, 12, hour, tzinfo=UTC))
        )
        await store.save_snapshot(
            ContinuableSnapshot(run_id=run_id, step_index=0, messages=list(messages), conversation_id='c1')
        )

    one = await load_conversation(store, conversation_id='c1', run_ids=['r1', 'r2', 'r3'], max_history_runs=1)
    assert [type(m).__name__ for m in one] == ['ModelRequest', 'ModelResponse']
    assert "content='second'" in str(one) and "content='re: second'" in str(one)
    assert 'first' not in str(one)

    two = await load_conversation(store, conversation_id='c1', run_ids=['r1', 'r2', 'r3'], max_history_runs=2)
    assert [type(m).__name__ for m in two] == ['ModelRequest', 'ModelResponse'] * 2
    assert 'first' in str(two) and "content='second'" in str(two)


async def test_max_history_runs_cut_does_not_leave_an_orphaned_leading_tool_return() -> None:
    """A cut that keeps only the run resuming a deferred tool call drops the run's leading tool
    return, whose call lies outside the window, and then its response, which no request precedes.
    Providers reject a history that starts with either, and pydantic-ai's own history repair runs
    ahead of this loader, so the loader drops both and the loaded history is empty.
    """
    store = InMemoryStepStore()
    await store.register_run(
        RunRecord(run_id='rA', conversation_id='c1', started_at=datetime(2026, 8, 12, 10, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='rA',
            step_index=1,
            messages=[
                ModelRequest(parts=[UserPromptPart(content='do the thing')]),
                ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='call1')]),
            ],
            conversation_id='c1',
        )
    )
    await store.register_run(
        RunRecord(run_id='rB', conversation_id='c1', started_at=datetime(2026, 8, 12, 11, tzinfo=UTC))
    )
    tail_response = ModelResponse(parts=[TextPart(content='done')])
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='rB',
            step_index=1,
            messages=[
                ModelRequest(parts=[ToolReturnPart(tool_name='t', content='result', tool_call_id='call1')]),
                tail_response,
            ],
            conversation_id='c1',
        )
    )

    messages = await load_conversation(store, conversation_id='c1', run_ids=['rA', 'rB'], max_history_runs=1)

    assert messages == []


async def test_max_history_runs_cut_never_leaves_a_leading_assistant_turn_when_a_later_run_survives() -> None:
    """When a run after the emptied one is also in the window, the loaded history starts with that
    run's own leading `ModelRequest`, not with the emptied run's orphaned `ModelResponse`. Same
    deferred-tool shape as above (rA's call, rB's orphaned return) plus a third, ordinary run rC,
    with `max_history_runs=2` keeping rB and rC.
    """
    store = InMemoryStepStore()
    await store.register_run(
        RunRecord(run_id='rA', conversation_id='c1', started_at=datetime(2026, 8, 12, 10, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='rA',
            step_index=1,
            messages=[
                ModelRequest(parts=[UserPromptPart(content='do the thing')]),
                ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='call1')]),
            ],
            conversation_id='c1',
        )
    )
    await store.register_run(
        RunRecord(run_id='rB', conversation_id='c1', started_at=datetime(2026, 8, 12, 11, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='rB',
            step_index=1,
            messages=[
                ModelRequest(parts=[ToolReturnPart(tool_name='t', content='result', tool_call_id='call1')]),
                ModelResponse(parts=[TextPart(content='done')]),
            ],
            conversation_id='c1',
        )
    )
    await store.register_run(
        RunRecord(run_id='rC', conversation_id='c1', started_at=datetime(2026, 8, 12, 12, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='rC',
            step_index=1,
            messages=_turn('later'),
            conversation_id='c1',
        )
    )

    messages = await load_conversation(store, conversation_id='c1', run_ids=['rA', 'rB', 'rC'], max_history_runs=2)

    assert messages
    assert isinstance(messages[0], ModelRequest)
    flattened = str(messages)
    assert 'later' in flattened and 'done' not in flattened


async def test_loaded_history_is_provider_valid_where_the_raw_cut_window_is_not() -> None:
    """The loader's contract -- each run's latest snapshot, concatenated in run-start order, is a
    sendable history -- is exactly what the harness's own `is_provider_valid` predicate checks
    (balanced tool call/return pairing). A `max_history_runs` cut through a deferred-tool boundary
    is the one shape where the RAW window is NOT provider-valid: it would start with a
    `ToolReturnPart` whose `ToolCallPart` sat in the cut-away run. The loader's boundary repair is
    what restores the invariant, so asserting the predicate on the loaded result -- and its FAILURE
    on the raw window -- gives that repair teeth rather than just checking the emptied-out shape.
    """
    store = InMemoryStepStore()
    await store.register_run(
        RunRecord(run_id='rA', conversation_id='c1', started_at=datetime(2026, 8, 12, 10, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='rA',
            step_index=1,
            messages=[
                ModelRequest(parts=[UserPromptPart(content='do the thing')]),
                ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='call1')]),
            ],
            conversation_id='c1',
        )
    )
    raw_window: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name='t', content='result', tool_call_id='call1')]),
        ModelResponse(parts=[TextPart(content='done')]),
    ]
    await store.register_run(
        RunRecord(run_id='rB', conversation_id='c1', started_at=datetime(2026, 8, 12, 11, tzinfo=UTC))
    )
    await store.save_snapshot(ContinuableSnapshot(run_id='rB', step_index=1, messages=raw_window, conversation_id='c1'))

    # Negative control: the cut window on its own is precisely the pairing no provider accepts.
    assert not is_provider_valid(raw_window)

    messages = await load_conversation(store, conversation_id='c1', run_ids=['rA', 'rB'], max_history_runs=1)

    assert is_provider_valid(messages)


def test_drop_orphaned_leading_tool_results_noop_on_empty_list() -> None:
    assert _drop_orphaned_leading_tool_results([]) == []


def test_drop_orphaned_leading_tool_results_noop_when_first_message_is_a_response() -> None:
    # Only the FIRST message can carry the artifact; a batch that happens to start with a
    # `ModelResponse` (not realistic through `load_conversation`, but a defensive branch) has
    # nothing this function acts on.
    messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content='x')])]
    assert _drop_orphaned_leading_tool_results(messages) is messages


def test_drop_orphaned_leading_tool_results_noop_when_nothing_is_orphaned() -> None:
    messages: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name='t', content='result', tool_call_id='call1')]),
        ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='call1')]),
    ]
    assert _drop_orphaned_leading_tool_results(messages) is messages


def test_drop_orphaned_leading_tool_results_ignores_a_non_tool_retry_prompt() -> None:
    # `RetryPromptPart.tool_name` is `None` for a plain output-validation retry -- it answers no
    # tool call at all, and its (randomly generated) `tool_call_id` must not be mistaken for one.
    messages: list[ModelMessage] = [
        ModelRequest(parts=[RetryPromptPart(content='invalid output', tool_call_id='not-a-real-call')]),
        ModelResponse(parts=[TextPart(content='retry')]),
    ]
    assert _drop_orphaned_leading_tool_results(messages) is messages


def test_drop_orphaned_leading_tool_results_strips_only_the_orphaned_part() -> None:
    # Same reasoning as the `load_conversation`-level test above: kept references, not freshly
    # reconstructed equal-content objects, since every part auto-generates its own `timestamp`.
    kept_part = UserPromptPart(content='and also this')
    tail_response = ModelResponse(parts=[TextPart(content='done')])
    messages: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name='t', content='orphaned', tool_call_id='gone'), kept_part]),
        tail_response,
    ]
    result = _drop_orphaned_leading_tool_results(messages)
    first = result[0]
    assert isinstance(first, ModelRequest)
    assert first.parts == [kept_part]
    assert result[1] is tail_response


def test_drop_orphaned_leading_tool_results_strips_an_orphaned_tool_retry() -> None:
    tail_response = ModelResponse(parts=[TextPart(content='done')])
    messages: list[ModelMessage] = [
        ModelRequest(parts=[RetryPromptPart(content='bad args', tool_name='t', tool_call_id='gone')]),
        tail_response,
    ]
    assert _drop_orphaned_leading_tool_results(messages) == [tail_response]


def test_drop_orphaned_leading_tool_results_drops_the_message_when_nothing_survives() -> None:
    tail_response = ModelResponse(parts=[TextPart(content='done')])
    messages: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name='t', content='orphaned', tool_call_id='gone')]),
        tail_response,
    ]
    assert _drop_orphaned_leading_tool_results(messages) == [tail_response]


async def test_run_without_snapshot_contributes_nothing() -> None:
    store = InMemoryStepStore()
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    assert await load_conversation(store, conversation_id='c1', run_ids=['r1']) == []


async def test_empty_conversation_id_raises() -> None:
    store = InMemoryStepStore()
    with pytest.raises(ValueError):
        await load_conversation(store, conversation_id='', run_ids=[])


async def test_interrupted_only_run_contributes_nothing() -> None:
    store = InMemoryStepStore()
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r1', step_index=1, messages=_turn('pending'), conversation_id='c1', state='interrupted'
        )
    )
    assert await load_conversation(store, conversation_id='c1', run_ids=['r1']) == []


async def test_latest_snapshot_wins_over_earlier_boundaries() -> None:
    store = InMemoryStepStore()
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(
        ContinuableSnapshot(run_id='r1', step_index=1, messages=_turn('first'), conversation_id='c1')
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r1',
            step_index=2,
            messages=[*_turn('first'), *_turn('second')],
            conversation_id='c1',
        )
    )
    messages = await load_conversation(store, conversation_id='c1', run_ids=['r1'])
    assert len(messages) == 4


async def test_transcript_shows_a_failed_turn_with_its_saved_prompt() -> None:
    """A run that failed before its first model response left its prompt as a complete
    snapshot (the frontier save); the transcript shows it as a failed turn."""
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'first', datetime(2026, 8, 12, 10, tzinfo=UTC))
    await store.register_run(
        RunRecord(run_id='r2', conversation_id='c1', started_at=datetime(2026, 8, 12, 11, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(run_id='r2', step_index=0, messages=_turn('pending')[:1], conversation_id='c1')
    )

    turns = await load_transcript(store, conversation_id='c1')

    assert [(t.run_id, t.phase) for t in turns] == [('r1', 'answered'), ('r2', 'failed')]
    assert 'pending' in str(turns[1].messages)


async def test_transcript_omits_a_run_with_only_interrupted_snapshots() -> None:
    """The harness's interrupted saves are not turns: a resume that had no prompt and
    never answered stays out."""
    store = InMemoryStepStore()
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r1', step_index=1, messages=_turn('partial'), conversation_id='c1', state='interrupted'
        )
    )

    assert await load_transcript(store, conversation_id='c1') == []


async def test_transcript_prefers_complete_when_both_exist() -> None:
    """A run with a `complete` snapshot shows that snapshot, even if a later `interrupted` save
    also exists for the same run (a crash after a settled boundary) -- the same read gate
    `load_conversation` applies.
    """
    store = InMemoryStepStore()
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(
        ContinuableSnapshot(run_id='r1', step_index=1, messages=_turn('first'), conversation_id='c1', state='complete')
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r1',
            step_index=2,
            messages=[*_turn('first'), *_turn('interrupted-tail')],
            conversation_id='c1',
            state='interrupted',
        )
    )

    turns = await load_transcript(store, conversation_id='c1')

    assert len(turns) == 1
    assert 'interrupted-tail' not in str(turns[0].messages)


async def test_transcript_excludes_delegates() -> None:
    store = InMemoryStepStore()
    await _seed_run(store, 'r-top', 'c1', 'top-level', datetime(2026, 8, 12, 10, tzinfo=UTC))
    await store.register_run(
        RunRecord(
            run_id='r-delegate',
            conversation_id='c1',
            parent_run_id='r-top',
            started_at=datetime(2026, 8, 12, 11, tzinfo=UTC),
        )
    )
    await store.save_snapshot(
        ContinuableSnapshot(run_id='r-delegate', step_index=1, messages=_turn('delegate'), conversation_id='c1')
    )

    turns = await load_transcript(store, conversation_id='c1')

    assert [t.run_id for t in turns] == ['r-top']


async def test_transcript_empty_conversation_id_raises() -> None:
    store = InMemoryStepStore()
    with pytest.raises(ValueError):
        await load_transcript(store, conversation_id='')


async def test_transcript_run_without_any_snapshot_contributes_nothing() -> None:
    # A registered run with no snapshot at all (neither complete nor interrupted) -- e.g. one that
    # crashed before `before_run` finished, or hasn't reached its first boundary yet -- contributes
    # no turn, same as `load_conversation`'s equivalent case.
    store = InMemoryStepStore()
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    assert await load_transcript(store, conversation_id='c1') == []


# --- Failed turns stay listed; the lineage, not the reader, decides what a client shows ---------


async def test_transcript_keeps_a_pending_turn_visible_next_to_later_turns() -> None:
    """A failed turn stays listed next to later turns: the reader hides nothing, the client picks
    the branch from the lineage."""
    store = InMemoryStepStore()
    await store.register_run(
        RunRecord(run_id='r-failed', conversation_id='c1', started_at=datetime(2026, 8, 12, 10, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r-failed',
            step_index=0,
            messages=[ModelRequest(parts=[UserPromptPart(content='X')])],
            conversation_id='c1',
        )
    )
    await store.register_run(
        RunRecord(run_id='r-later', conversation_id='c1', started_at=datetime(2026, 8, 12, 11, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(run_id='r-later', step_index=1, messages=_turn('Y'), conversation_id='c1', state='complete')
    )

    turns = await load_transcript(store, conversation_id='c1')

    assert [(t.run_id, t.phase) for t in turns] == [('r-failed', 'failed'), ('r-later', 'answered')]


async def test_transcript_turn_author_user_id_populated_from_the_turn_records() -> None:
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'hi', datetime(2026, 8, 12, 10, tzinfo=UTC))

    with_records = await load_transcript(
        store, conversation_id='c1', turns=[TurnRecord(run_id='r1', author_user_id='alice')]
    )
    without_records = await load_transcript(store, conversation_id='c1')

    assert with_records[0].author_user_id == 'alice'
    assert without_records[0].author_user_id is None


def _count_prompt(messages: list[ModelMessage], text: str) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, UserPromptPart) and p.content == text
    )


async def test_transcript_stamps_the_phase_from_the_turn_records() -> None:
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'hi', datetime(2026, 8, 12, 10, tzinfo=UTC))

    turns = await load_transcript(
        store, conversation_id='c1', turns=[TurnRecord(run_id='r1', author_user_id='alice', phase='failed')]
    )

    # The registry's authoritative phase wins over any structural derivation.
    assert turns[0].phase == 'failed'


async def test_transcript_derives_answered_when_no_map_and_the_turn_ends_in_a_response() -> None:
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'hi', datetime(2026, 8, 12, 10, tzinfo=UTC))

    turns = await load_transcript(store, conversation_id='c1')

    assert turns[0].phase == 'answered'


async def test_transcript_derives_failed_for_a_prompt_only_turn_without_a_map() -> None:
    store = InMemoryStepStore()
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r1',
            step_index=0,
            messages=[ModelRequest(parts=[UserPromptPart(content='M')])],
            conversation_id='c1',
        )
    )

    turns = await load_transcript(store, conversation_id='c1')

    assert turns[0].phase == 'failed'


async def test_transcript_derives_failed_for_a_complete_but_unanswered_turn() -> None:
    """A run with a settled tool boundary that never produced a final answer (ends in a dangling
    tool-return request) is `failed`, not `answered`.
    """
    store = InMemoryStepStore()
    await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
    await store.save_snapshot(
        ContinuableSnapshot(run_id='r1', step_index=1, messages=_partial('M'), conversation_id='c1', state='complete')
    )

    turns = await load_transcript(store, conversation_id='c1')

    assert turns[0].phase == 'failed'


async def test_transcript_partial_failure_without_a_retry_stays_visible_as_failed() -> None:
    store = InMemoryStepStore()
    await store.register_run(
        RunRecord(run_id='r-partial', conversation_id='c1', started_at=datetime(2026, 8, 12, 10, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r-partial', step_index=1, messages=_partial('M'), conversation_id='c1', state='complete'
        )
    )

    turns = await load_transcript(store, conversation_id='c1')

    assert len(turns) == 1
    assert turns[0].run_id == 'r-partial'
    assert turns[0].phase == 'failed'
    assert 'result' in str(turns[0].messages)  # its tool work stays shown


async def test_transcript_view_sees_every_turn_at_once() -> None:
    """The view runs once over all resolved turns, failed and answered alike, so scrubbed
    material is gone from every branch a client may render."""
    from dataclasses import replace as dc_replace

    from pydantic_ai_conversations._loader import TranscriptTurn

    store = InMemoryStepStore()
    await store.register_run(
        RunRecord(run_id='r1', conversation_id='c1', started_at=datetime(2026, 8, 17, 10, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r1',
            step_index=1,
            conversation_id='c1',
            messages=[
                ModelRequest(parts=[UserPromptPart(content='q')]),
                ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='tc1')]),
                ModelRequest(parts=[ToolReturnPart(tool_name='t', content='MARKER', tool_call_id='tc1')]),
            ],
        )
    )
    await store.register_run(
        RunRecord(run_id='r2', conversation_id='c1', started_at=datetime(2026, 8, 17, 11, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r2',
            step_index=1,
            conversation_id='c1',
            messages=[
                ModelRequest(parts=[UserPromptPart(content='q')]),
                ModelResponse(parts=[TextPart(content='answer')]),
            ],
        )
    )

    def scrub(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
        return [
            dc_replace(
                turn,
                messages=[
                    message
                    for message in turn.messages
                    if not any(
                        isinstance(part, ToolReturnPart) and part.content == 'MARKER'
                        for part in getattr(message, 'parts', [])
                    )
                ],
            )
            for turn in turns
        ]

    turns = await load_transcript(
        store,
        conversation_id='c1',
        turns=[
            TurnRecord(run_id='r1', author_user_id='a', phase='failed'),
            TurnRecord(run_id='r2', author_user_id='a', phase='answered'),
        ],
        transcript_view=scrub,
    )
    assert [t.phase for t in turns] == ['failed', 'answered']
    assert not any(
        isinstance(part, ToolReturnPart) and part.content == 'MARKER'
        for turn in turns
        for message in turn.messages
        for part in getattr(message, 'parts', [])
    )


async def test_transcript_view_can_drop_turns() -> None:
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'q', datetime(2026, 8, 17, 10, tzinfo=UTC))
    turns = await load_transcript(store, conversation_id='c1', transcript_view=lambda turns: [])
    assert turns == []


async def test_transcript_view_can_decide_across_turns() -> None:
    """The view gets every turn, so a rule that needs other turns is expressible: hide
    a failed attempt once a sibling (same `previous_run_id`) has answered."""
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'first', datetime(2026, 8, 12, 10, tzinfo=UTC))
    await store.register_run(
        RunRecord(run_id='r2', conversation_id='c1', started_at=datetime(2026, 8, 12, 11, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r2',
            step_index=0,
            messages=[ModelRequest(parts=[UserPromptPart(content='second')])],
            conversation_id='c1',
        )
    )
    await _seed_run(store, 'r3', 'c1', 'second', datetime(2026, 8, 12, 12, tzinfo=UTC))
    records = [
        TurnRecord(run_id='r1', author_user_id='a', phase='answered'),
        TurnRecord(run_id='r2', author_user_id='a', phase='failed', previous_run_id='r1'),
        TurnRecord(run_id='r3', author_user_id='a', phase='answered', previous_run_id='r1'),
    ]

    def hide_retried_attempts(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
        answered_after = {turn.previous_run_id for turn in turns if turn.phase == 'answered'}
        return [turn for turn in turns if not (turn.phase == 'failed' and turn.previous_run_id in answered_after)]

    turns = await load_transcript(store, conversation_id='c1', turns=records, transcript_view=hide_retried_attempts)

    assert [t.run_id for t in turns] == ['r1', 'r3']


async def test_transcript_view_absent_keeps_turns_unchanged() -> None:
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'q', datetime(2026, 8, 17, 10, tzinfo=UTC))
    assert len(await load_transcript(store, conversation_id='c1')) == 1


# --- Lineage: every branch is listed, the active one is the latest turn's chain ----------------


async def test_transcript_stamps_previous_run_id_from_the_turn_records() -> None:
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'first', datetime(2026, 8, 12, 10, tzinfo=UTC))
    await _seed_run(store, 'r2', 'c1', 'second', datetime(2026, 8, 12, 11, tzinfo=UTC))
    records = [
        TurnRecord(run_id='r1', author_user_id='a'),
        TurnRecord(run_id='r2', author_user_id='a', previous_run_id='r1'),
    ]

    with_records = await load_transcript(store, conversation_id='c1', turns=records)
    without_records = await load_transcript(store, conversation_id='c1')

    assert [t.previous_run_id for t in with_records] == [None, 'r1']
    assert [t.previous_run_id for t in without_records] == [None, None]


async def test_transcript_lists_every_branch_in_run_start_order() -> None:
    """A retry and the attempt it replaces are two turns with the same `previous_run_id`; the
    reader lists both, and the client picks the branch to show."""
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'first', datetime(2026, 8, 12, 10, tzinfo=UTC))
    await store.register_run(
        RunRecord(run_id='r2', conversation_id='c1', started_at=datetime(2026, 8, 12, 11, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r2',
            step_index=0,
            messages=[ModelRequest(parts=[UserPromptPart(content='second')])],
            conversation_id='c1',
        )
    )
    await _seed_run(store, 'r3', 'c1', 'second', datetime(2026, 8, 12, 12, tzinfo=UTC))
    records = [
        TurnRecord(run_id='r1', author_user_id='a', phase='answered'),
        TurnRecord(run_id='r2', author_user_id='a', phase='failed', previous_run_id='r1'),
        TurnRecord(run_id='r3', author_user_id='a', phase='answered', previous_run_id='r1'),
    ]

    turns = await load_transcript(store, conversation_id='c1', turns=records)

    assert [(t.run_id, t.previous_run_id, t.phase) for t in turns] == [
        ('r1', None, 'answered'),
        ('r2', 'r1', 'failed'),
        ('r3', 'r1', 'answered'),
    ]
    assert active_run_ids(turns) == {'r1', 'r3'}


def _turn_stub(run_id: str, previous_run_id: str | None, messages: list[ModelMessage] | None = None) -> TranscriptTurn:
    return TranscriptTurn(
        run_id=run_id,
        started_at=datetime(2026, 8, 12, 10, tzinfo=UTC),
        phase='answered',
        messages=messages or [],
        previous_run_id=previous_run_id,
    )


def test_active_run_ids_follows_the_chain_from_the_latest_turn() -> None:
    turns = [_turn_stub('r1', None), _turn_stub('r2', 'r1'), _turn_stub('r3', 'r1'), _turn_stub('r4', 'r3')]
    assert active_run_ids(turns) == {'r1', 'r3', 'r4'}


def test_active_run_ids_is_the_latest_root_without_links() -> None:
    # Two roots are a failed first turn and its retry (sent with an explicit `null`), or a
    # store-only read without lineage; either way only the latest root is on the branch.
    assert active_run_ids([]) == set()
    assert active_run_ids([_turn_stub('r1', None), _turn_stub('r2', None)]) == {'r2'}


def test_active_run_ids_stops_at_a_link_it_has_seen() -> None:
    assert active_run_ids([_turn_stub('r1', 'r2'), _turn_stub('r2', 'r1')]) == {'r1', 'r2'}


# --- Open requests: a failed turn keeps the request it left unanswered -----------------------


async def test_a_failed_turns_prompt_is_loaded_as_stored_on_every_chain() -> None:
    """A failed turn's prompt is loaded whether the turn is the chain's last run or an ancestor;
    the run that continued it stored only what it added, so the prompt is on the branch once."""
    store = InMemoryStepStore()
    await _seed_run(store, 'r1', 'c1', 'first', datetime(2026, 8, 12, 10, tzinfo=UTC))
    await store.register_run(
        RunRecord(run_id='r2', conversation_id='c1', started_at=datetime(2026, 8, 12, 11, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r2',
            step_index=0,
            messages=[ModelRequest(parts=[UserPromptPart(content='second')])],
            conversation_id='c1',
        )
    )
    await store.register_run(
        RunRecord(run_id='r3', conversation_id='c1', started_at=datetime(2026, 8, 12, 12, tzinfo=UTC))
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r3',
            step_index=1,
            messages=[ModelResponse(parts=[TextPart(content='re: second')])],
            conversation_id='c1',
        )
    )

    continuing = await load_conversation(store, conversation_id='c1', run_ids=['r1', 'r2'])
    assert str(continuing).count("content='second'") == 1 and isinstance(continuing[-1], ModelRequest)

    after = await load_conversation(store, conversation_id='c1', run_ids=['r1', 'r2', 'r3'])
    assert str(after).count("content='second'") == 1 and isinstance(after[-1], ModelResponse)


def test_path_messages_follows_the_latest_turns_branch() -> None:
    prompt: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='q')])]
    turns = [
        _turn_stub('r1', None, _turn('first')),
        _turn_stub('r2', 'r1', prompt),  # failed, continued by r3
        _turn_stub('r3', 'r2', [ModelResponse(parts=[TextPart(content='a')])]),
        _turn_stub('r4', 'r1', _turn('fork')),  # another branch, off the path
        _turn_stub('r5', 'r3', _turn('next')),
    ]
    flat = path_messages(turns)
    assert str(flat).count("content='q'") == 1
    assert 'fork' not in str(flat)
    assert [type(message).__name__ for message in flat] == ['ModelRequest', 'ModelResponse'] * 3


def test_path_messages_shows_only_the_latest_root_of_a_retried_first_turn() -> None:
    # The failed root settled a tool round before dying; a retry rooted with `null` must not
    # inherit its dangling tool return, which would sit mid-list without the call that follows.
    failed_root: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='q')]),
        ModelResponse(parts=[ToolCallPart(tool_name='t', args={}, tool_call_id='c1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='t', content='x', tool_call_id='c1')]),
    ]
    retry = _turn('q')
    turns = [_turn_stub('r1', None, failed_root), _turn_stub('r2', None, retry)]
    assert path_messages(turns) == retry


def test_path_messages_keeps_a_prompt_only_latest_turn() -> None:
    turns = [
        _turn_stub('r1', None, _turn('first')),
        _turn_stub('r2', 'r1', [ModelRequest(parts=[UserPromptPart(content='q')])]),
    ]
    assert len(path_messages(turns)) == 3
