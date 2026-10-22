"""The harness's PostgreSQL step store plus the three service-side extras `stores.py` discovers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai_harness.media import PostgresConnection
from pydantic_ai_harness.step_persistence import PostgresStepStore

from pydantic_ai_conversations.postgres._driver import STEP_STORE_TRANSIENT_ERRORS


class PostgresConversationStepStore(PostgresStepStore):
    """The harness's `PostgresStepStore` plus `delete_runs`, `ping` and `transient_errors`.

    The harness `StepStore` protocol carries none of them, and `stores.py` discovers all three
    structurally: without `delete_runs` a purge leaves the conversation's runs in the substrate,
    without `ping` `/readyz` says nothing about the store, and without `transient_errors` a
    database outage reaches the client as a 500 instead of a retryable 503. `transient_errors` comes
    from `_driver.py`.

    Takes the same asyncpg-compatible pool as the harness store and passes every constructor
    argument through unchanged, so this subclass is a strict superset of it. Media payloads are
    content-addressed and may be shared across runs, so `delete_runs` leaves them to the store's
    own garbage collection (`stores.py`). Snapshot messages are content-addressed too, and
    `delete_runs` deletes those no other run's snapshot references.
    """

    transient_errors = STEP_STORE_TRANSIENT_ERRORS

    async def delete_runs(self, *, run_ids: Sequence[str]) -> None:
        ids = sorted(set(run_ids))
        if not ids:
            return
        await self._ensure_schema()
        # One transaction, so a failure partway leaves every row of the run for a retried purge.
        # The messages go first, as they are found through the snapshots.
        async with self._pool.acquire() as connection, connection.transaction():
            await self._delete_unshared_messages(connection, ids)
            for table in (
                self._events_table,
                self._snapshots_table,
                self._snapshot_keys_table,
                self._tool_effects_table,
                self._runs_table,
            ):
                await connection.execute(f'DELETE FROM {table} WHERE run_id = ANY($1::text[])', ids)

    async def _delete_unshared_messages(self, connection: PostgresConnection, run_ids: list[str]) -> None:
        """Delete the stored messages the runs' snapshots reference and no other run's snapshot does.

        `PostgresStepStore` keeps each message once in `{table}_messages`, keyed by its SHA-256, so
        identical messages in two runs are one row. A snapshot's `messages` column holds
        `{"message_refs": [...]}`; a row from a harness release that kept the history inline holds a
        JSON list, which has no `message_refs` and so references nothing. Finding the shared keys
        reads every other snapshot once, and only when the runs stored anything; a purge is rare
        enough for that. A concurrent save in another run can end up referencing a message deleted
        here; docs/deleting.md says why that is accepted.
        """
        refs = "jsonb_array_elements_text((snapshot.messages::jsonb) -> 'message_refs') AS ref(sha256)"
        own = [
            str(row[0])
            for row in await connection.fetch(
                f'SELECT DISTINCT ref.sha256 FROM {self._snapshots_table} AS snapshot, {refs} '
                f'WHERE snapshot.run_id = ANY($1::text[])',
                run_ids,
            )
        ]
        if not own:
            return
        await connection.execute(
            f'WITH shared AS ('
            f'  SELECT DISTINCT ref.sha256 FROM {self._snapshots_table} AS snapshot, {refs}'
            f'  WHERE NOT (snapshot.run_id = ANY($1::text[])) AND ref.sha256 = ANY($2::text[])'
            f') '
            f'DELETE FROM {self._messages_table} '
            f'WHERE sha256 = ANY($2::text[]) AND sha256 NOT IN (SELECT sha256 FROM shared)',
            run_ids,
            own,
        )

    async def ping(self) -> Mapping[str, Any]:
        async with self._pool.acquire() as connection:
            await connection.fetchval('SELECT 1')
        return {'ok': True}
