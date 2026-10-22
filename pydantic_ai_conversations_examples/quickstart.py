"""The quick-start app from the README: one SQLite file holds every conversation.

The model is pydantic-ai's test model: every reply is "success (no tool calls)", no provider is
called. Any pydantic-ai model string works in its place, with the provider's extra installed and
its key exported.

Run with:

    uv run -m pydantic_ai_conversations_examples.quickstart

`app.db` is created in the working directory; delete it to start over.
"""

from pydantic_ai_app_factory import UserContext, create_agent, create_agent_app

from pydantic_ai_conversations import ConversationsFeature
from pydantic_ai_conversations.sqlite import SqliteConversationRegistry, SqliteConversationStepStore


def auth() -> UserContext:
    """Stands in for real authentication. Every request is the same user in the same tenant."""
    return UserContext(user_id='alice', tenant_id='t1')


# Registry and step store may share one file; each keeps its own tables.
conversations = ConversationsFeature(
    registry=SqliteConversationRegistry('app.db'),
    step_store=SqliteConversationStepStore('app.db'),
)
agent = create_agent('test', instructions='You are a support assistant.')
app = create_agent_app(agent, auth=auth, features=[conversations])


if __name__ == '__main__':
    import uvicorn

    print(
        'Start a conversation; the reply carries the run_id the next turn names:\n'
        '  curl -s http://127.0.0.1:8000/run -H "Content-Type: application/json" \\\n'
        '    -d \'{"prompt": "My printer is offline.", "conversation_id": "c1"}\'\n'
        'Continue it:\n'
        '  curl -s http://127.0.0.1:8000/run -H "Content-Type: application/json" \\\n'
        '    -d \'{"prompt": "It is a Brother.", "conversation_id": "c1", '
        '"extras": {"previous_run_id": "<run_id>"}}\'\n'
        'Read it back:\n'
        '  curl -s http://127.0.0.1:8000/conversations/c1\n'
        'Docs at http://127.0.0.1:8000/docs\n',
        flush=True,  # before uvicorn's log lines, even when stdout is a pipe
    )
    uvicorn.run(app, host='127.0.0.1', port=8000)
