"""Conversations shared by the members of a case, with the feature's optional settings filled in.

Every conversation belongs to a case. Members of the case read and continue the same
conversations, and each turn records who wrote it. The case arrives with the request as the
`case_id` extras field, so the client names it, and `auth` checks that the caller is a member
before anything else runs.

The members are a dict and the cleanup hook appends to a list. A real app looks the membership
up in its own system and deletes attachments from its own storage.

Run it with `uv run -m pydantic_ai_conversations_examples.shared_cases`. `cases.db` is created
in the working directory.
"""

from functools import partial
from typing import Annotated, Any

from fastapi import Header, HTTPException, Request
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai_app_factory import AgentError, ErrorHandlers, UserContext, create_agent, create_agent_app

from pydantic_ai_conversations import (
    ConversationsFeature,
    DeletionEvent,
    TranscriptTurn,
    agent_error_mapper,
)
from pydantic_ai_conversations.sqlite import SqliteConversationRegistry, SqliteConversationStepStore

TENANT_OF = {'alice': 't1', 'bob': 't1', 'carol': 't1'}
"""Which tenant each demo user belongs to."""

CASE_MEMBERS = {'case-9': {'alice', 'bob'}}
"""Who may act in which case. Carol is in the tenant but not on the case."""

forgotten: list[str] = []
"""Conversation ids the cleanup hook received."""


async def requested_cases(request: Request) -> set[str]:
    """Every case a request names: a query parameter on the `/conversations` routes, a top-level body
    field on `/chat`, and a field inside `extras` on `/run`, in either spelling.

    All of them are collected, because checking only the first would let a request name one case
    in the query and act on another in the body.
    """
    cases = set(request.query_params.getlist('case_id'))
    if request.method != 'POST':
        return cases
    try:
        body: Any = await request.json()
    except ValueError:  # not JSON; the endpoint rejects it
        return cases
    if isinstance(body, dict):
        extras = body.get('extras')
        for source in (body, extras if isinstance(extras, dict) else {}):
            cases.update(str(source[key]) for key in ('case_id', 'caseId') if key in source)
    return cases


async def auth(request: Request, x_user: Annotated[str, Header()] = 'alice') -> UserContext:
    """Demo authentication: the `X-User` header names the caller.

    The case is a claim the client makes, so the membership check belongs here. Without it any
    caller in the tenant could read and delete any case's conversations by naming the case.
    """
    if x_user not in TENANT_OF:
        raise HTTPException(401, 'Unknown user.')
    if any(x_user not in CASE_MEMBERS.get(case, set()) for case in await requested_cases(request)):
        raise HTTPException(403, 'Not a member of this case.')
    return UserContext(user_id=x_user, tenant_id=TENANT_OF[x_user])


class CaseSystemDown(Exception):
    """Raised by a tool when the case system is unreachable. The demo agent has no such tool."""


error_handlers: ErrorHandlers = [
    (CaseSystemDown, AgentError('cases/unavailable', 'The case system is unavailable.', retryable=True)),
]


async def forget_attachments(event: DeletionEvent) -> None:
    """Runs on soft delete and on purge. Must be idempotent, since a failed purge is retried."""
    forgotten.append(event.conversation_id)


def hide_superseded_failures(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
    """Hide a failed turn once a retry built on the same predecessor has answered."""
    answered_after = {t.previous_run_id for t in turns if t.phase == 'answered'}
    return [t for t in turns if not (t.phase == 'failed' and t.previous_run_id in answered_after)]


conversations = ConversationsFeature(
    registry=SqliteConversationRegistry('cases.db'),
    step_store=SqliteConversationStepStore('cases.db'),
    # One partition per case within a tenant, instead of the default one per user.
    scope=('user.tenant_id', 'extras.case_id'),
    # `POST /conversations` creates an entry with `attrs` before its first run; the rename route
    # lets members set a title. Both are off by default because who may do it is the app's call.
    enable_create_route=True,
    enable_title_route=True,
    # `GET /conversations?topic=billing` lists the case's conversations on one topic.
    filterable_attrs=('topic',),
    on_delete=[forget_attachments],
    transcript_view=hide_superseded_failures,
    # Each transcript turn carries its messages in the protocol `/chat` streams, so a frontend
    # rebuilds its chat state from them. Pinned to the SDK version the app streams.
    ui_messages=partial(VercelAIAdapter.dump_messages, sdk_version=5),
    # Long case threads load the last 20 turns into each run.
    max_history_runs=20,
    # A failed turn stores the same code and message the client saw live.
    describe_failure=agent_error_mapper(error_handlers),
)

agent = create_agent('test', instructions='You help a team work through a support case.')
app = create_agent_app(agent, auth=auth, features=[conversations], error_handlers=error_handlers)


if __name__ == '__main__':
    import uvicorn

    print(
        'Alice opens a conversation on case-9; Bob continues it with her run_id:\n'
        '  curl -s http://127.0.0.1:8000/run -H "Content-Type: application/json" -H "X-User: alice" \\\n'
        '    -d \'{"prompt": "The customer reports a double charge.", "conversation_id": "c1", '
        '"extras": {"case_id": "case-9"}}\'\n'
        '  curl -s http://127.0.0.1:8000/run -H "Content-Type: application/json" -H "X-User: bob" \\\n'
        '    -d \'{"prompt": "Refund issued.", "conversation_id": "c1", '
        '"extras": {"case_id": "case-9", "previous_run_id": "<run_id>"}}\'\n'
        'Carol is not on the case and gets 403:\n'
        '  curl -s "http://127.0.0.1:8000/conversations?case_id=case-9" -H "X-User: carol"\n'
        'Docs at http://127.0.0.1:8000/docs\n',
        flush=True,  # before uvicorn's log lines, even when stdout is a pipe
    )
    uvicorn.run(app, host='127.0.0.1', port=8000)
