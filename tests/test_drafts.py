"""rejected_draft_pairs / strip_rejected_drafts: recognizing validator-rejected drafts."""

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai_conversations import rejected_draft_pairs, strip_rejected_drafts


def _draft(text: str = 'wrong') -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _retry() -> ModelRequest:
    return ModelRequest(parts=[RetryPromptPart(content='search first')])


def test_pair_is_recognized_and_stripped():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content='q')]),
        _draft(),
        _retry(),
        ModelResponse(parts=[TextPart(content='right')]),
    ]
    assert rejected_draft_pairs(msgs) == [(1, 2)]
    stripped = strip_rejected_drafts(msgs)
    assert [type(m).__name__ for m in stripped] == ['ModelRequest', 'ModelResponse']
    final = stripped[1].parts[0]
    assert isinstance(final, TextPart)
    assert final.content == 'right'


def test_consecutive_pairs_all_stripped():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content='q')]),
        _draft('d1'),
        _retry(),
        _draft('d2'),
        _retry(),
        ModelResponse(parts=[TextPart(content='ok')]),
    ]
    stripped = strip_rejected_drafts(msgs)
    assert len(stripped) == 2


def test_trailing_unmarked_draft_is_not_touched():
    # Retry exhaustion: the final draft has no retry prompt after it (documented trap).
    msgs = [ModelRequest(parts=[UserPromptPart(content='q')]), _draft('d1'), _retry(), _draft('final')]
    stripped = strip_rejected_drafts(msgs)
    assert len(stripped) == 2
    assert isinstance(stripped[-1], ModelResponse)


def test_tool_bound_retry_is_not_a_rejection_marker():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content='q')]),
        ModelResponse(parts=[ToolCallPart(tool_name='t', args={}, tool_call_id='tc1')]),
        ModelRequest(parts=[RetryPromptPart(content='bad args', tool_name='t', tool_call_id='tc1')]),
    ]
    assert rejected_draft_pairs(msgs) == []
    assert strip_rejected_drafts(msgs) == msgs


def test_response_with_tool_calls_is_never_treated_as_draft():
    # Defensive: dropping it would orphan the tool returns answering it.
    msgs = [
        ModelResponse(parts=[TextPart(content='x'), ToolCallPart(tool_name='t', args={}, tool_call_id='tc1')]),
        ModelRequest(
            parts=[ToolReturnPart(tool_name='t', content='r', tool_call_id='tc1'), RetryPromptPart(content='retry')]
        ),
    ]
    assert rejected_draft_pairs(msgs) == []


def test_retry_request_with_extra_parts_keeps_them():
    req = ModelRequest(parts=[RetryPromptPart(content='retry'), UserPromptPart(content='kept')])
    msgs = [_draft(), req]
    stripped = strip_rejected_drafts(msgs)
    assert len(stripped) == 1
    kept_request = stripped[0]
    assert isinstance(kept_request, ModelRequest)
    assert [type(p).__name__ for p in kept_request.parts] == ['UserPromptPart']


def test_metadata_survives_the_part_surgery():
    req = ModelRequest(parts=[RetryPromptPart(content='retry'), UserPromptPart(content='kept')], run_id='r9')
    stripped = strip_rejected_drafts([_draft(), req])
    assert stripped[0].run_id == 'r9'


def test_idempotent_and_noop_returns_same_list():
    msgs = [ModelRequest(parts=[UserPromptPart(content='q')]), ModelResponse(parts=[TextPart(content='a')])]
    assert strip_rejected_drafts(msgs) is msgs
    once = strip_rejected_drafts([_draft(), _retry()])
    assert strip_rejected_drafts(once) == once == []
