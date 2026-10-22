"""Turn lineage helpers: the loaded tail and its restoration after pydantic-ai merges it."""

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai_conversations._lineage import loaded_tail, unmerge_loaded_tail


def _req(*parts: UserPromptPart | ToolReturnPart, run_id: str | None = None) -> ModelRequest:
    return ModelRequest(parts=list(parts), run_id=run_id)


def test_loaded_tail_is_the_trailing_run_of_requests() -> None:
    p1, p2 = _req(UserPromptPart(content='a'), run_id='r1'), _req(UserPromptPart(content='b'), run_id='r2')
    answer = ModelResponse(parts=[TextPart(content='x')], run_id='r0')
    assert loaded_tail([]) == []
    assert loaded_tail([p1, answer]) == []
    assert loaded_tail([answer, p1]) == [p1]
    assert loaded_tail([answer, p1, p2]) == [p1, p2]


def test_unmerge_restores_the_tail_and_leaves_the_remainder_unstamped() -> None:
    old = _req(UserPromptPart(content='old'), run_id='r1')
    new_part = UserPromptPart(content='new')
    merged = ModelRequest(parts=[*old.parts, new_part], instructions='sys', metadata={'source': 'web'})
    messages: list[ModelMessage] = [merged]

    assert unmerge_loaded_tail(messages, [old]) is True

    assert messages[0] is old
    rest = messages[1]
    assert isinstance(rest, ModelRequest) and rest.parts == [new_part] and rest.run_id is None
    assert rest.instructions == 'sys' and rest.metadata == {'source': 'web'}


def test_unmerge_restores_two_open_requests() -> None:
    # A continuation of a failed turn that itself failed before answering leaves two.
    first = _req(UserPromptPart(content='a'), run_id='r1')
    second = _req(UserPromptPart(content='b'), run_id='r2')
    new_part = UserPromptPart(content='c')
    answer = ModelResponse(parts=[TextPart(content='x')], run_id='r0')
    messages: list[ModelMessage] = [answer, ModelRequest(parts=[*first.parts, *second.parts, new_part])]

    assert unmerge_loaded_tail(messages, [first, second]) is True

    assert messages[1:3] == [first, second] and messages[3].parts == [new_part]


def test_unmerge_restores_two_open_requests_merged_without_a_new_message() -> None:
    # A resume with an empty list: pydantic-ai merges the two loaded requests on their own.
    first = _req(UserPromptPart(content='a'), run_id='r1')
    second = _req(UserPromptPart(content='b'), run_id='r2')
    messages: list[ModelMessage] = [ModelRequest(parts=[*first.parts, *second.parts])]

    assert unmerge_loaded_tail(messages, [first, second]) is True

    assert messages == [first, second] and messages[0] is first and messages[1] is second


def test_unmerge_restores_a_tool_return_tail() -> None:
    call = ModelResponse(parts=[ToolCallPart(tool_name='t', args={}, tool_call_id='c1')], run_id='r1')
    ret = _req(ToolReturnPart(tool_name='t', content='x', tool_call_id='c1'), run_id='r1')
    new_part = UserPromptPart(content='new')
    messages: list[ModelMessage] = [call, ModelRequest(parts=[*ret.parts, new_part])]

    assert unmerge_loaded_tail(messages, [ret]) is True

    assert messages[1] is ret and messages[2].parts == [new_part]


def test_unmerge_does_nothing_without_a_merge() -> None:
    old = _req(UserPromptPart(content='old'), run_id='r1')
    # The resumed request keeps its stamp: nothing to split.
    resumed: list[ModelMessage] = [old]
    assert unmerge_loaded_tail(resumed, [old]) is False and resumed == [old]
    # A separately passed prompt sits in its own request: nothing to split.
    separate: list[ModelMessage] = [old, _req(UserPromptPart(content='new'), run_id='r2')]
    assert unmerge_loaded_tail(separate, [old]) is False and len(separate) == 2
    # The same with the new prompt not yet stamped: the tail is in place, nothing was merged.
    unsent = _req(UserPromptPart(content='new'))
    pending: list[ModelMessage] = [old, unsent]
    assert unmerge_loaded_tail(pending, [old]) is False and pending == [old, unsent] and pending[1] is unsent
    # No tail (the history ended in a response): the client's request is left alone.
    fresh: list[ModelMessage] = [_req(UserPromptPart(content='new'))]
    assert unmerge_loaded_tail(fresh, []) is False and len(fresh) == 1
    # The trailing unstamped request does not start with the tail: left alone.
    other: list[ModelMessage] = [_req(UserPromptPart(content='other'), UserPromptPart(content='new'))]
    assert unmerge_loaded_tail(other, [old]) is False and len(other) == 1
    # Fewer parts than the tail: not a merge of it.
    second = _req(UserPromptPart(content='second'), run_id='r2')
    short: list[ModelMessage] = [ModelRequest(parts=list(old.parts))]
    assert unmerge_loaded_tail(short, [old, second]) is False and len(short) == 1


def test_unmerge_matches_parts_by_identity_then_equality() -> None:
    old_part = UserPromptPart(content='old')
    old = _req(old_part, run_id='r1')
    copy = UserPromptPart(content='old', timestamp=old_part.timestamp)  # equal, not identical
    messages: list[ModelMessage] = [ModelRequest(parts=[copy, UserPromptPart(content='new')])]
    assert unmerge_loaded_tail(messages, [old]) is True and messages[0] is old


def test_unmerge_restores_only_the_tail_requests_that_were_merged() -> None:
    # Requests with differing instructions stay apart, so only the last open request is merged.
    first = ModelRequest(parts=[UserPromptPart(content='a')], run_id='r2', instructions='one')
    second = ModelRequest(parts=[UserPromptPart(content='b')], run_id='r3')
    new_part = UserPromptPart(content='c')
    messages: list[ModelMessage] = [first, ModelRequest(parts=[*second.parts, new_part], instructions='two')]

    assert unmerge_loaded_tail(messages, [first, second]) is True

    assert messages[0] is first and messages[1] is second
    assert len(messages) == 3 and messages[2].parts == [new_part] and messages[2].run_id is None
