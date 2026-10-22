"""Driver plumbing for the SQLite backend: connection, pragmas, schema, thread dispatch, and the
registry's error mapping.

Separate from the port implementations so `_registry.py` and `_store.py` carry SQL and protocol
semantics only.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import anyio.to_thread

from pydantic_ai_conversations.errors import RegistryUnavailable

_R = TypeVar('_R')

STEP_STORE_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (sqlite3.OperationalError, OSError)
"""What the step store raises when the file is locked, cannot be opened, or the disk fails. An
`IntegrityError` is left out: the harness store raises it for a run id already stored."""


class CursorRoutedConnection:
    """A `sqlite3.Connection` facade whose `execute` goes through a real cursor.

    `opentelemetry-instrumentation-sqlite3` wraps `Cursor.execute`, not the connection-level
    `execute` shortcut, so queries issued through the shortcut carry no spans. The shortcut makes a
    throwaway cursor of its own and a cursor inherits the connection's `row_factory`, so routing
    through one is otherwise identical.
    """

    def __init__(self, raw: sqlite3.Connection) -> None:
        self._raw = raw

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self._raw.cursor().execute(sql, parameters)

    def close(self) -> None:
        self._raw.close()


class SqliteDriver:
    """Runs each unit of work on a worker thread, over one connection held for the driver's life.

    The connection is held rather than opened per call: closing the last connection to a WAL
    database checkpoints it and tears down the `-wal`/`-shm` files, a cost every statement would
    otherwise carry. Holding one open spares the harness store's short-lived connections to the
    same file as well, so do not narrow its lifetime to the call.

    The lock keeps two worker threads off that shared connection at once. SQLite admits one writer
    at a time regardless; what the lock adds is atomicity for a whole unit of work against other
    callers, which the multi-statement methods (`register_conversation`'s claim-then-select) depend on. It is a
    `threading.Lock`, taken inside the worker thread, so the event loop never waits on it and it
    stays independent of which event loop is running.
    """

    def __init__(
        self, database: str | Path, *, schema: str = '', migrate: Callable[[sqlite3.Connection], None] | None = None
    ) -> None:
        self._database = str(database)
        self._schema = schema
        self._migrate = migrate
        self._connection: CursorRoutedConnection | None = None
        self._lock = threading.Lock()

    async def run(self, work: Callable[[CursorRoutedConnection], _R]) -> _R:
        return await anyio.to_thread.run_sync(self._run_sync, work)

    async def close(self) -> None:
        await anyio.to_thread.run_sync(self._close_sync)

    def _run_sync(self, work: Callable[[CursorRoutedConnection], _R]) -> _R:
        with self._lock:
            return work(self._connect())

    def _close_sync(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _connect(self) -> CursorRoutedConnection:
        """Lazy-connect, so a caller needs no explicit `open()` before the first unit of work.
        Runs under the lock, so connecting, the schema script, and the migration happen exactly
        once."""
        if self._connection is None:
            raw = sqlite3.connect(self._database, check_same_thread=False, isolation_level=None)
            raw.row_factory = sqlite3.Row
            raw.execute('PRAGMA journal_mode=WAL')  # so the harness store's own connections coexist
            raw.execute('PRAGMA synchronous=NORMAL')  # the standard WAL pairing: no fsync per autocommit
            raw.execute('PRAGMA busy_timeout=5000')  # wait for a foreign writer instead of failing
            if self._schema:
                raw.executescript(self._schema)
            if self._migrate is not None:
                self._migrate(raw)
            self._connection = CursorRoutedConnection(raw)
        return self._connection


class RegistryDriver(SqliteDriver):
    """`SqliteDriver` with the registry port's availability and error contract.

    After `close()` every unit of work raises `RegistryUnavailable` until `open()` lifts it -- the
    port's outage state, not merely a released resource; the connection itself comes back lazily
    with the next unit of work. `sqlite3.Error` becomes `RegistryUnavailable`;
    `ConversationNotFound` and a caller's `ValueError` pass through untranslated.
    """

    def __init__(
        self, database: str | Path, *, schema: str, migrate: Callable[[sqlite3.Connection], None] | None = None
    ) -> None:
        super().__init__(database, schema=schema, migrate=migrate)
        self._closed = False

    async def open(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True
        await super().close()

    async def run(self, work: Callable[[CursorRoutedConnection], _R]) -> _R:
        if self._closed:
            raise RegistryUnavailable('registry is closed')
        try:
            return await super().run(work)
        except sqlite3.Error as exc:
            raise RegistryUnavailable(str(exc)) from exc
