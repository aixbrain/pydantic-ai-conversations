"""Wire-level proof that loaded history uses the model's STANDARD channel.

Every other test observes the loaded history through `FunctionModel`'s `messages` argument or
through model behavior. This one goes to the wire: a real `AnthropicModel` over an httpx2 `MockTransport`, and
the assertion reads the raw JSON body that would be POSTed to `/v1/messages`. Turn 2 must carry
turn 1 inside the payload's ordinary `messages` array -- the same field `message_history=` feeds --
not in any side channel. Traces show the same list but not what was posted.
"""

import json
from typing import Any

import httpx2
import pytest

pytest.importorskip('anthropic', reason='wire test needs the anthropic SDK from the dev group')

from pydantic_ai import Agent, models  # noqa: E402
from pydantic_ai.models.anthropic import AnthropicModel  # noqa: E402
from pydantic_ai.providers.anthropic import AnthropicProvider  # noqa: E402
from pydantic_ai.settings import ModelSettings  # noqa: E402
from pydantic_ai_app_factory import AppDeps, UserContext  # noqa: E402
from pydantic_ai_harness.step_persistence import InMemoryStepStore  # noqa: E402

from pydantic_ai_conversations import ConversationsFeature, InMemoryConversationRegistry  # noqa: E402


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return ' '.join(block.get('text', '') for block in content)


async def test_injected_history_is_sent_in_the_standard_messages_array() -> None:
    captured: list[dict[str, Any]] = []

    def transport_handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                'id': f'msg_{len(captured)}',
                'type': 'message',
                'role': 'assistant',
                'model': 'claude-sonnet-4-6',
                'content': [{'type': 'text', 'text': f'answer number {len(captured)}'}],
                'stop_reason': 'end_turn',
                'usage': {'input_tokens': 10, 'output_tokens': 5},
            },
        )

    provider = AnthropicProvider(
        api_key='test-key', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(transport_handler))
    )
    feature = ConversationsFeature(
        registry=InMemoryConversationRegistry(),
        step_store=InMemoryStepStore(),
        scope=('user.tenant_id',),
    )
    agent: Agent[AppDeps, Any] = Agent(
        AnthropicModel('claude-sonnet-4-6', provider=provider), deps_type=AppDeps, capabilities=feature.capabilities()
    )
    deps = AppDeps(user=UserContext(user_id='u', tenant_id='t'), extras={})

    # The default `max_tokens` is the model's maximum, which the SDK only sends streamed; the mock
    # transport answers with a plain JSON message.
    settings: ModelSettings = {'max_tokens': 1024}
    # A real model class, so `conftest.py`'s block on model requests is lifted; the mock transport
    # answers, and nothing leaves the process.
    with models.override_allow_model_requests(True):
        first = await agent.run('erste frage', deps=deps, conversation_id='wire-test', model_settings=settings)
        deps = AppDeps(user=deps.user, extras={'previous_run_id': first.run_id})
        await agent.run('zweite frage', deps=deps, conversation_id='wire-test', model_settings=settings)

    first, second = captured
    # Turn 1 carries exactly its own message -- nothing leaks in from anywhere.
    assert [(m['role'], _content_text(m['content'])) for m in first['messages']] == [('user', 'erste frage')]
    # Turn 2's payload carries turn 1 whole, in the provider's ordinary messages field.
    assert [(m['role'], _content_text(m['content'])) for m in second['messages']] == [
        ('user', 'erste frage'),
        ('assistant', 'answer number 1'),
        ('user', 'zweite frage'),
    ]
