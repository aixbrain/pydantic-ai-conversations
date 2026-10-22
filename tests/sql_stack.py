"""Shared harness for the SQL-backed integration suites.

Builds the whole stack the way an app would: one SQLite file carrying both the harness store's
runs/events/snapshots and the registry's conversations/turns, a `create_agent_app` over it, and
one auth callable object serving every caller. The faultable subclasses add per-seam switches so a
test can take a backend down mid-conversation; every fault travels the real error path
(`RegistryUnavailable`, a raising store) rather than a patched-out boundary.

Each suite brings its own agent and feature configuration through the `build` callback -- the
recovery suite needs retrying tools and view seams, the outage suite needs a plain tool -- while
the backends, wiring, and inspection helpers stay identical between them.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Coroutine, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai_app_factory import AppDeps, UserContext, create_agent_app
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord

from pydantic_ai_conversations import ConversationsFeature, DeletionEvent
from pydantic_ai_conversations.errors import RegistryUnavailable
from pydantic_ai_conversations.registry import RunPhase
from pydantic_ai_conversations.sqlite import SqliteConversationRegistry, SqliteConversationStepStore

_T = TypeVar('_T')


class _Omitted:
    """Marks a `previous_run_id` the request leaves out, as opposed to an explicit `None`."""


OMITTED = _Omitted()

ALICE = UserContext(user_id='alice', tenant_id='t1')
BOB = UserContext(user_id='bob', tenant_id='t1')
CAROL = UserContext(user_id='carol', tenant_id='t2')


def run_async(coroutine: Coroutine[Any, Any, _T]) -> _T:
    """Run an async stack call from a sync `TestClient` test (no loop is running there)."""
    return asyncio.run(coroutine)


class FaultableRegistry(SqliteConversationRegistry):
    """The SQL registry plus one surgical switch: stats update fails while the answer succeeds.

    A whole-backend outage needs no switch -- `close()` is the real thing, and every method then
    raises `RegistryUnavailable` until `open()` reconnects.
    """

    def __init__(self, database: str | Path) -> None:
        super().__init__(database)
        self.fail_update_message_stats = False

    async def update_message_stats(
        self, namespace_key: str, conversation_id: str, *, last_message_at: datetime, message_count_delta: int
    ) -> None:
        if self.fail_update_message_stats:
            raise RegistryUnavailable('injected: stats backend down')
        await super().update_message_stats(
            namespace_key, conversation_id, last_message_at=last_message_at, message_count_delta=message_count_delta
        )


class FaultableStepStore(SqliteConversationStepStore):
    """The shipped SQLite store plus per-seam switches for the save, read, and substrate-delete paths."""

    def __init__(self, database: str | Path) -> None:
        super().__init__(database)
        self.fail_next_save = False
        self.fail_reads = False
        self.fail_delete_runs_once = False

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        if self.fail_next_save:
            self.fail_next_save = False
            raise sqlite3.OperationalError('injected: database is locked')
        await super().save_snapshot(snapshot)

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None
    ) -> list[RunRecord]:
        if self.fail_reads:
            raise sqlite3.OperationalError('injected: database is locked')
        return await super().list_runs(parent_run_id=parent_run_id, conversation_id=conversation_id)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        if self.fail_reads:
            raise sqlite3.OperationalError('injected: database is locked')
        return await super().latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)

    async def delete_runs(self, *, run_ids: Sequence[str]) -> None:
        if self.fail_delete_runs_once:
            self.fail_delete_runs_once = False
            raise RuntimeError('injected: substrate delete down')
        await super().delete_runs(run_ids=run_ids)


@dataclass(eq=False)  # identity hash: FastAPI keys its dependency cache on the callable object
class AuthHolder:
    """One callable object (the feature/factory contract) whose resolved user is swappable."""

    user: UserContext

    def __call__(self) -> UserContext:
        return self.user


class DeletionRecorder:
    """An `on_delete` hook that records what it was told, for assertions."""

    def __init__(self) -> None:
        self.events: list[DeletionEvent] = []

    async def __call__(self, event: DeletionEvent) -> None:
        self.events.append(event)


def last_user_prompt(messages: list[ModelMessage]) -> str:
    """The most recent user-authored text in `messages`; `''` when there is none."""
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return part.content
    return ''


def part_kinds(messages: list[ModelMessage]) -> list[str]:
    """Every part's class name, flattened -- the shape assertions read against."""
    return [type(part).__name__ for message in messages for part in message.parts]


def json_part_kinds(messages: list[dict[str, Any]]) -> list[str]:
    """`part_kinds` for messages already dumped to JSON by the HTTP routes."""
    return [part['part_kind'] for message in messages for part in message['parts']]


def user_prompt_count(messages: list[dict[str, Any]], text: str) -> int:
    """How often `text` appears as a USER message in a JSON-dumped list.

    The merge invariant counts prompts, not substrings: the same text legitimately recurs in tool
    arguments, tool returns, and the answer.
    """
    return sum(
        1
        for message in messages
        for part in message['parts']
        if part.get('part_kind') == 'user-prompt' and part.get('content') == text
    )


@dataclass
class Backends:
    """What the harness creates before the suite builds its agent and feature over them."""

    registry: FaultableRegistry
    store: FaultableStepStore
    auth: AuthHolder
    deletions: DeletionRecorder
    """Ready to pass as `ConversationsFeature(on_delete=[backends.deletions])`."""


StackBuilder = Callable[[Backends], tuple['Agent[AppDeps, Any]', ConversationsFeature]]
"""Builds the suite's agent and feature over the backends the harness created."""

FaultsT = TypeVar('FaultsT')
"""The suite's own agent-side fault switches -- what its model function and tools consult."""


@dataclass
class SqlStack(Generic[FaultsT]):
    """Everything a test needs to drive the app and inspect the stack underneath it."""

    client: TestClient
    registry: FaultableRegistry
    store: FaultableStepStore
    auth: AuthHolder
    feature: ConversationsFeature
    faults: FaultsT
    deletions: DeletionRecorder
    seen: list[list[ModelMessage]] = field(default_factory=list)
    """Every model request the agent issued, in order -- what the model actually saw."""

    def run(self, prompt: str, conversation_id: str, *, after: str | None | _Omitted = OMITTED):  # noqa: ANN201 -- TestClient's Response type is env-specific
        """POST `/run`; `after` is the `previous_run_id` the client sends, required from the second
        turn on (`None` starts from nothing, as a retry of a first turn does)."""
        body: dict[str, Any] = {'prompt': prompt, 'conversation_id': conversation_id}
        if not isinstance(after, _Omitted):
            body['extras'] = {'previous_run_id': after}
        return self.client.post('/run', json=body)

    def last_turn_id(self, conversation_id: str) -> str:
        """The run id of the conversation's latest recorded turn, failed ones included -- what a
        client reads off a transcript when the run itself returned no `run_id`."""
        turns = run_async(self.registry.list_turns(self.namespace, conversation_id))
        return turns[-1].run_id

    def detail(self, conversation_id: str) -> dict[str, Any]:
        response = self.client.get(f'/conversations/{conversation_id}')
        assert response.status_code == 200, response.text
        return response.json()

    @property
    def namespace(self) -> str:
        return self.feature.namespace_for(self.auth.user)

    def turn_phases(self, conversation_id: str) -> list[RunPhase]:
        """The registry's run-lifecycle phases for the conversation, in `start_turn` order."""
        turns = run_async(self.registry.list_turns(self.namespace, conversation_id))
        return [turn.phase for turn in turns]

    def stored_messages(self, conversation_id: str, *, run_index: int = 0) -> list[ModelMessage]:
        """One run's latest complete snapshot -- the stored truth, read straight from the store."""

        async def read() -> list[ModelMessage]:
            runs = await self.store.list_runs(conversation_id=conversation_id)
            snapshot = await self.store.latest_snapshot(run_id=runs[run_index].run_id)
            return list(snapshot.messages) if snapshot else []

        return run_async(read())


@contextmanager
def sql_stack(
    database: Path,
    build: StackBuilder,
    *,
    faults: FaultsT,
    seen: list[list[ModelMessage]] | None = None,
    user: UserContext = ALICE,
) -> Generator[SqlStack[FaultsT]]:
    """Wire backends, agent, feature, and a running `create_agent_app` into one `SqlStack`.

    `faults` and `seen` are the suite's own objects -- the switches its tools consult and the list
    its model function appends to. Passing them here puts the same objects on the stack, so tests
    read them as `stack.faults` / `stack.seen` without keeping a second handle.
    """
    backends = Backends(
        registry=FaultableRegistry(database),
        store=FaultableStepStore(database),
        auth=AuthHolder(user),
        deletions=DeletionRecorder(),
    )
    agent, feature = build(backends)
    app = create_agent_app(agent, auth=backends.auth, features=[feature])
    with TestClient(app) as client:
        yield SqlStack(
            client=client,
            registry=backends.registry,
            store=backends.store,
            auth=backends.auth,
            feature=feature,
            faults=faults,
            deletions=backends.deletions,
            seen=seen if seen is not None else [],
        )
