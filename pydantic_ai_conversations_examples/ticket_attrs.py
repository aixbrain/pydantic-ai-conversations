"""Conversations tagged with `attrs` the server derives, so a client can filter on them but not forge them.

Each conversation belongs to a support ticket. The app's own `POST /tickets/{ticket_id}/conversations`
looks the ticket up and creates the registry entry with the ticket's id, product and priority as
`attrs` before the first run. `enable_create_route` stays off, because `POST /conversations`
would let the client choose any `attrs`.

The tickets are a dict. A real app reads them from its ticket system.

Run it with `uv run -m pydantic_ai_conversations_examples.ticket_attrs`. `tickets.db` is created
in the working directory.
"""

from typing import Annotated
from uuid import uuid4

from fastapi import Depends, HTTPException
from pydantic_ai_app_factory import UserContext, create_agent, create_agent_app

from pydantic_ai_conversations import ConversationOut, ConversationsFeature
from pydantic_ai_conversations.sqlite import SqliteConversationRegistry, SqliteConversationStepStore

TICKETS = {
    'T-17': {'product': 'printer', 'priority': 'high'},
    'T-18': {'product': 'scanner', 'priority': 'low'},
}
"""The ticket system: what the server knows about each ticket."""


def auth() -> UserContext:
    """Stands in for real authentication. Every request is the same user in the same tenant."""
    return UserContext(user_id='alice', tenant_id='t1')


registry = SqliteConversationRegistry('tickets.db')
conversations = ConversationsFeature(
    registry=registry,
    step_store=SqliteConversationStepStore('tickets.db'),
    # `GET /conversations?ticket=T-17&priority=high` lists by ticket, priority, or both.
    filterable_attrs=('ticket', 'product', 'priority'),
)
agent = create_agent('test', instructions='You help resolve support tickets.')
app = create_agent_app(agent, auth=auth, features=[conversations])


@app.post('/tickets/{ticket_id}/conversations', status_code=201)
async def open_ticket_conversation(ticket_id: str, user: Annotated[UserContext, Depends(auth)]) -> ConversationOut:
    """Create a conversation for a ticket; the client uses the returned id as `conversation_id`.

    `attrs` are set once at creation and never change, so the entry has to exist before the first
    run. A run on an unknown id creates its entry without `attrs`.
    """
    ticket = TICKETS.get(ticket_id)
    if ticket is None:
        raise HTTPException(404, 'Unknown ticket.')
    meta = await registry.register_conversation(
        conversations.namespace_for(user),
        uuid4().hex,
        created_by=user.user_id,
        attrs={'ticket': ticket_id, **ticket},
    )
    return ConversationOut(
        conversation_id=meta.conversation_id,
        title=meta.title,
        created_by=meta.created_by,
        created_at=meta.created_at,
        last_message_at=meta.last_message_at,
        message_count=meta.message_count,
        attrs=meta.attrs,
    )


if __name__ == '__main__':
    import uvicorn

    print(
        'Open a conversation on ticket T-17; the reply carries its conversation_id:\n'
        '  curl -s -X POST http://127.0.0.1:8000/tickets/T-17/conversations\n'
        'Talk in it:\n'
        '  curl -s http://127.0.0.1:8000/run -H "Content-Type: application/json" \\\n'
        '    -d \'{"prompt": "The printer is offline.", "conversation_id": "<conversation_id>"}\'\n'
        'List the high-priority ones:\n'
        '  curl -s "http://127.0.0.1:8000/conversations?priority=high"\n'
        'Docs at http://127.0.0.1:8000/docs\n',
        flush=True,  # before uvicorn's log lines, even when stdout is a pipe
    )
    uvicorn.run(app, host='127.0.0.1', port=8000)
