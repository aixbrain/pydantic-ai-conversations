"""The feedback add-on's routes: write and delete the caller's own entry, list a conversation's."""

# No `from __future__ import annotations`: FastAPI resolves `Depends(...)` from real annotation
# objects, and `caller_dep` below is closure-scoped.

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends

from pydantic_ai_conversations.addons import AddonCaller, AddonRouteGuard
from pydantic_ai_conversations.feedback._wire import TurnFeedbackEntryOut

if TYPE_CHECKING:
    from pydantic_ai_conversations.feedback._addon import TurnFeedbackAddon


def build_feedback_router(addon: 'TurnFeedbackAddon', guard: AddonRouteGuard) -> APIRouter:
    """Routes under `/conversations/{id}/turns/{run_id}/<name>` and `/conversations/{id}/<name>`.

    The namespace wall is the guard's (`require_turn`, `require_conversation`): a foreign or
    missing conversation, or a run that is not one of its turns, is a 404. Writes touch the
    caller's own entry only, so no second permission decision is made here.
    """
    name, caller_dep = addon.name, guard.caller()
    router = APIRouter(prefix='/conversations', tags=[name])
    entry_model = addon.turn_model
    # Built as a value: pyright rejects a variable inside a type expression.
    list_model: Any = list[entry_model]
    turn_path = '/{conversation_id}/turns/{run_id}/' + name

    async def _store_call(coro: Any) -> Any:
        try:
            return await coro
        except Exception as exc:
            if isinstance(exc, addon.transient_errors):
                raise guard.unavailable() from None
            raise

    async def put_entry(
        conversation_id: str, run_id: str, caller: Annotated[AddonCaller, Depends(caller_dep)], body: Any
    ) -> TurnFeedbackEntryOut:
        turn = await guard.require_turn(caller.namespace_key, conversation_id, run_id)
        record = await _store_call(
            addon.store.upsert(
                kind=name,
                namespace_key=caller.namespace_key,
                conversation_id=conversation_id,
                run_id=turn.run_id,
                author_user_id=caller.user.user_id,
                data=body.model_dump(mode='json', by_alias=True),
            )
        )
        return addon.to_entry(record)

    # FastAPI reads the body type from the function's annotations when the route is registered,
    # and the payload model is only known here; so the annotation is set before `add_api_route`.
    put_entry.__annotations__['body'] = addon.payload_model
    router.add_api_route(turn_path, put_entry, methods=['PUT'], response_model=entry_model)

    @router.delete(turn_path, status_code=204)
    async def delete_entry(
        conversation_id: str, run_id: str, caller: Annotated[AddonCaller, Depends(caller_dep)]
    ) -> None:
        turn = await guard.require_turn(caller.namespace_key, conversation_id, run_id)
        await _store_call(
            addon.store.delete(
                kind=name,
                namespace_key=caller.namespace_key,
                conversation_id=conversation_id,
                run_id=turn.run_id,
                author_user_id=caller.user.user_id,
            )
        )

    @router.get('/{conversation_id}/' + name, response_model=list_model)
    async def list_entries(
        conversation_id: str, caller: Annotated[AddonCaller, Depends(caller_dep)]
    ) -> list[TurnFeedbackEntryOut]:
        await guard.require_conversation(caller.namespace_key, conversation_id)
        records = await _store_call(
            addon.store.list(kind=name, namespace_key=caller.namespace_key, conversation_id=conversation_id)
        )
        return addon.valid_entries(records)

    return router
