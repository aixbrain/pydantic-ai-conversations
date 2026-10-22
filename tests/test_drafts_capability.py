"""`StripRejectedDrafts`: rejected drafts of earlier turns leave the model's context, the live
run's own draft/retry pair and the stored snapshots stay intact."""

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_app_factory import AppDeps, UserContext
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    InMemoryStepStore,
    RunRecord,
    StepPersistence,
)

from pydantic_ai_conversations import (
    ConversationsCapability,
    InMemoryConversationRegistry,
    NewMessagesPreparer,
    PreparingStepStore,
    StripRejectedDrafts,
)
from pydantic_ai_conversations.scope import build_namespace, parse_scope

SCOPE = parse_scope(('user.tenant_id', 'user.user_id'))


def _deps(previous: str | None = None) -> AppDeps:
    extras = {'previous_run_id': previous} if previous is not None else {}
    return AppDeps(user=UserContext(user_id='alice', tenant_id='t1'), extras=extras)


def _agent(
    inner: InMemoryStepStore,
    seen: list[list[ModelMessage]],
    *,
    reject_drafts: bool = False,
    registry: InMemoryConversationRegistry | None = None,
) -> Agent[AppDeps, str]:
    store = PreparingStepStore(inner, [NewMessagesPreparer()])

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        last = messages[-1]
        retrying = isinstance(last, ModelRequest) and any(isinstance(part, RetryPromptPart) for part in last.parts)
        return ModelResponse(parts=[TextPart(content='final' if retrying or not reject_drafts else 'draft')])

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=AppDeps,
        capabilities=[
            ConversationsCapability(registry=registry or InMemoryConversationRegistry(), step_store=store, scope=SCOPE),
            StepPersistence(store=PreparingStepStore(store, [NewMessagesPreparer()]), capture_frontier=True),
            StripRejectedDrafts(),
        ],
    )
    if reject_drafts:

        @agent.output_validator
        def reject(text: str) -> str:
            if text == 'draft':
                raise ModelRetry('rejected')
            return text

    return agent


def _texts(messages: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, TextPart)
    ]


async def test_a_stored_pair_is_filtered_out_of_the_next_turn() -> None:
    inner = InMemoryStepStore()
    await inner.register_run(RunRecord(run_id='r0', conversation_id='c1'))
    await inner.save_snapshot(
        ContinuableSnapshot(
            run_id='r0',
            step_index=1,
            conversation_id='c1',
            messages=[
                ModelRequest(parts=[UserPromptPart(content='old q')]),
                ModelResponse(parts=[TextPart(content='rejected draft')]),
                ModelRequest(parts=[RetryPromptPart(content='scold')]),
                ModelResponse(parts=[TextPart(content='real answer')]),
            ],
        )
    )
    for message in (await inner.latest_snapshot(run_id='r0')).messages:  # type: ignore[union-attr]
        message.run_id = 'r0'
    registry = InMemoryConversationRegistry()
    namespace = build_namespace(SCOPE, _deps().user, {})
    await registry.register_conversation(namespace, 'c1', created_by='alice')
    await registry.start_turn(namespace, 'c1', run_id='r0', author_user_id='alice')
    seen: list[list[ModelMessage]] = []
    await _agent(inner, seen, registry=registry).run('new q', deps=_deps(previous='r0'), conversation_id='c1')
    assert _texts(seen[0]) == ['real answer']


async def test_the_live_runs_own_pair_stays_and_the_retry_succeeds() -> None:
    """The validator rejects every first draft: the retry prompt and the draft it answers belong
    to the live run and must reach the model, or the retry cannot correct anything. Earlier
    turns' pairs are stripped from the same request."""
    inner = InMemoryStepStore()
    seen: list[list[ModelMessage]] = []
    agent = _agent(inner, seen, reject_drafts=True)
    previous: str | None = None
    for index in range(2):
        result = await agent.run(f'q{index}', deps=_deps(previous), conversation_id='c1')
        assert result.output == 'final'
        previous = result.run_id

    # Turn 2's retry request: turn 1 appears without its draft, turn 2's own draft is still there.
    assert _texts(seen[3]) == ['final', 'draft']
    assert isinstance(seen[3][-1], ModelRequest)
    assert any(isinstance(part, RetryPromptPart) for part in seen[3][-1].parts)


async def test_stored_snapshots_keep_the_drafts() -> None:
    inner = InMemoryStepStore()
    seen: list[list[ModelMessage]] = []
    agent = _agent(inner, seen, reject_drafts=True)
    await agent.run('q0', deps=_deps(), conversation_id='c1')
    (run,) = await inner.list_runs(conversation_id='c1')
    snapshot = await inner.latest_snapshot(run_id=run.run_id)
    assert snapshot is not None
    assert _texts(snapshot.messages) == ['draft', 'final']


async def test_nothing_to_strip_leaves_the_request_untouched() -> None:
    inner = InMemoryStepStore()
    seen: list[list[ModelMessage]] = []
    agent = _agent(inner, seen)
    first = await agent.run('q0', deps=_deps(), conversation_id='c1')
    await agent.run('q1', deps=_deps(previous=first.run_id), conversation_id='c1')
    assert _texts(seen[1]) == ['final']
