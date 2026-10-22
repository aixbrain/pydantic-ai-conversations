"""The harness's SQLite step store plus the three service-side extras `stores.py` discovers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic_ai_harness.media import MediaStore
from pydantic_ai_harness.step_persistence import SqliteStepStore

from pydantic_ai_conversations.sqlite._driver import STEP_STORE_TRANSIENT_ERRORS, CursorRoutedConnection, SqliteDriver

_DEFAULT_MEDIA_THRESHOLD_BYTES = 64 * 1024
"""Mirrors `SqliteStepStore`'s own default (`pydantic_ai_harness.step_persistence._store`), kept
as a literal here rather than imported: that name is private to the harness."""


class SqliteConversationStepStore(SqliteStepStore):
    """The harness's `SqliteStepStore` plus `delete_runs`, `ping` and `transient_errors`.

    The harness `StepStore` protocol carries none of them, and `stores.py` discovers all three
    structurally: without `delete_runs` a purge leaves the conversation's runs in the substrate,
    without `ping` `/readyz` says nothing about the store, and without `transient_errors` a
    database outage reaches the client as a 500 instead of a retryable 503. `transient_errors` comes
    from `_driver.py`.

    `delete_runs` and `ping` run on a worker thread over a connection of this subclass's own (see
    `_driver.py`), against the same file the base class opens; `close()` releases it, and a later
    call reopens it. Media blobs are content-addressed and may be shared across runs, so
    `delete_runs` leaves them to the store's own garbage collection (`stores.py`).

    `media_store`, `media_threshold_bytes` and `max_snapshots_per_run` pass straight through to
    `SqliteStepStore` with its same defaults, so this subclass is a strict superset of it.
    """

    transient_errors = STEP_STORE_TRANSIENT_ERRORS

    def __init__(
        self,
        database: str | Path,
        *,
        media_store: MediaStore | None | Literal['auto'] = 'auto',
        media_threshold_bytes: int = _DEFAULT_MEDIA_THRESHOLD_BYTES,
        max_snapshots_per_run: int | None = None,
    ) -> None:
        super().__init__(
            database=database,
            media_store=media_store,
            media_threshold_bytes=media_threshold_bytes,
            max_snapshots_per_run=max_snapshots_per_run,
        )
        self._driver = SqliteDriver(database)

    async def close(self) -> None:
        """Release this subclass's own connection; the base store opens its own per call."""
        await self._driver.close()

    async def delete_runs(self, *, run_ids: Sequence[str]) -> None:
        ids = list(run_ids)
        if not ids:
            return
        placeholders = ', '.join('?' for _ in ids)

        def work(connection: CursorRoutedConnection) -> None:
            # The base store creates its tables on the first write, so a purge can come first; with no
            # `runs` table nothing of any run is stored, and unknown ids are a no-op by contract.
            tables = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'runs'")
            if tables.fetchone() is None:
                return
            # The driver's connection is autocommit and lives on, so the deletes run in one
            # transaction that is rolled back on a failure: an open transaction left behind would
            # fail every later statement on the connection.
            connection.execute('BEGIN')
            try:
                # `snapshot_idempotency_keys` is the harness's replay ledger: it outlives snapshot
                # pruning by design, so nothing else ever clears it. Leaving it would keep the
                # purged run's ids on disk and make a later save under a recurring key a silent no-op.
                for table in ('runs', 'events', 'snapshots', 'snapshot_idempotency_keys', 'tool_effects'):
                    connection.execute(f'DELETE FROM {table} WHERE run_id IN ({placeholders})', ids)  # noqa: S608
            except BaseException:
                connection.execute('ROLLBACK')
                raise
            connection.execute('COMMIT')

        await self._driver.run(work)

    async def ping(self) -> Mapping[str, Any]:
        def work(connection: CursorRoutedConnection) -> Mapping[str, Any]:
            # The base store creates its schema on first use, so the runs table may not exist yet;
            # probe the connection alone. The report is public, so no count.
            connection.execute('SELECT 1').fetchone()
            return {'ok': True}

        return await self._driver.run(work)
