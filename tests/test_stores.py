"""SupportsRunDeletion / SupportsPing: runtime_checkable structural discovery."""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai_harness.step_persistence import InMemoryStepStore

from pydantic_ai_conversations.stores import SupportsPing, SupportsRunDeletion


class _DeletingStepStore(InMemoryStepStore):
    async def delete_runs(self, *, run_ids: Sequence[str]) -> None:
        pass


class _PingingStepStore(InMemoryStepStore):
    async def ping(self) -> Mapping[str, Any]:
        return {'ok': True}


def test_plain_store_satisfies_neither_extension() -> None:
    store = InMemoryStepStore()
    assert isinstance(store, SupportsRunDeletion) is False
    assert isinstance(store, SupportsPing) is False


def test_a_store_implementing_delete_runs_satisfies_supports_run_deletion() -> None:
    assert isinstance(_DeletingStepStore(), SupportsRunDeletion) is True
    # Deletion support alone does not imply ping support.
    assert isinstance(_DeletingStepStore(), SupportsPing) is False


def test_a_store_implementing_ping_satisfies_supports_ping() -> None:
    assert isinstance(_PingingStepStore(), SupportsPing) is True
    # Ping support alone does not imply deletion support.
    assert isinstance(_PingingStepStore(), SupportsRunDeletion) is False
