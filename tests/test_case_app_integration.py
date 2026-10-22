"""The case-app walkthrough: one case conversation, several authors, membership via auth.

Scope `('user.tenant_id', 'extras.case_id')` partitions per case within a tenant. WHO may act in
a case is the AuthProvider's decision (membership lookup) and happens before the feature runs;
here both alice and bob are members, carol's tenant differs so the namespace wall alone stops her.
"""

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_app_factory import AgentError, AppDeps, UserContext
from pydantic_ai_harness.step_persistence import InMemoryStepStore

from pydantic_ai_conversations import ConversationsFeature, InMemoryConversationRegistry

SCOPE_SPEC = ('user.tenant_id', 'extras.case_id')


async def test_two_authors_share_one_case_conversation() -> None:
    registry, store = InMemoryConversationRegistry(), InMemoryStepStore()
    feature = ConversationsFeature(registry=registry, step_store=store, scope=SCOPE_SPEC)
    assert feature.extras_type is not None and 'case_id' in feature.extras_type.model_fields

    seen: list[list[ModelMessage]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart(content='noted')])

    agent = Agent(FunctionModel(model_fn), deps_type=AppDeps, capabilities=feature.capabilities())

    alice = AppDeps(user=UserContext(user_id='alice', tenant_id='t1'), extras={'case_id': 'case-9'})
    first = await agent.run('pump is leaking', deps=alice, conversation_id='case-9-chat')
    bob = AppDeps(
        user=UserContext(user_id='bob', tenant_id='t1'), extras={'case_id': 'case-9', 'previous_run_id': first.run_id}
    )
    await agent.run('replaced the seal', deps=bob, conversation_id='case-9-chat')

    # Bob's model request saw alice's turn.
    assert 'pump is leaking' in str(seen[1])

    # Attribution: the registry knows who wrote each turn. The namespace comes from the feature
    # itself -- apps never re-declare the scope spec, a second literal is a drift vector.
    namespace = feature.namespace_for(alice.user, alice.extras)
    turns = await registry.list_turns(namespace, 'case-9-chat')
    assert [t.author_user_id for t in turns] == ['alice', 'bob']

    # A caller outside the namespace (other tenant, same claimed case) hits the wall.
    carol = AppDeps(user=UserContext(user_id='carol', tenant_id='t2'), extras={'case_id': 'case-9'})
    with pytest.raises(AgentError) as exc_info:
        await agent.run('let me in', deps=carol, conversation_id='case-9-chat')
    assert exc_info.value.code == 'conversations/not-found'
