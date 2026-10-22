"""Driver plumbing for the MongoDB backend: client ownership, indexes, and the error mapping.

Separate from the port implementation so `_registry.py` and `_store.py` carry Mongo operations
and protocol semantics only.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import CoroutineType
from typing import Any, ParamSpec, TypeVar

from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import ConnectionFailure, PyMongoError

from pydantic_ai_conversations.errors import RegistryUnavailable

_P = ParamSpec('_P')
_R = TypeVar('_R')

STEP_STORE_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (ConnectionFailure, OSError)
"""What the step store raises when no server is reachable, a primary steps down, or the network
times out. Other `PyMongoError`s, a duplicate key among them, are left out, because they fail the
same way again."""
_Document = dict[str, Any]


def maps_backend_errors(
    method: Callable[_P, CoroutineType[Any, Any, _R]],
) -> Callable[_P, CoroutineType[Any, Any, _R]]:
    """`PyMongoError` -> `RegistryUnavailable`, per the `errors.py` contract; everything else
    (`ConversationNotFound`, a caller's `ValueError`) passes through untranslated.

    Annotated with `types.CoroutineType` rather than the abstract `collections.abc.Coroutine`:
    pyright represents a genuine `async def` Protocol method's call signature with the concrete
    runtime type, and only the concrete type -- not its own abstract supertype -- satisfies that
    signature when a decorated method is checked against `ConversationRegistry` structurally.
    """

    @functools.wraps(method)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return await method(*args, **kwargs)
        except PyMongoError as exc:
            raise RegistryUnavailable(str(exc)) from exc

    return wrapper


class MongoDriver:
    """Holds the client, owns it or borrows it, and creates the registry's indexes once.

    Whoever opens the client closes it: with `db_url=` the driver created it and `close()` closes
    it; with `client=` it did not, and `close()` leaves it alone -- the app, the step store and the
    registry commonly share one client, and closing it from here would cut the others off mid
    shutdown. Either way `close()` puts the driver in its outage state, and `open()` lifts it.
    """

    def __init__(
        self,
        *,
        client: AsyncMongoClient[_Document] | None = None,
        db_url: str | None = None,
        database: str,
    ) -> None:
        if (client is None) == (db_url is None):
            raise ValueError('provide exactly one of `client=` or `db_url=`')
        self._db_url = db_url
        self._database = database
        self._own_client = client is None
        self._client: AsyncMongoClient[_Document] = client if client is not None else AsyncMongoClient(db_url)
        self._indexes_ready = False
        self._closed = False

    @property
    def db(self) -> AsyncDatabase[_Document]:
        return self._client[self._database]

    @property
    def conversations(self) -> AsyncCollection[_Document]:
        return self._client[self._database]['conversations']

    @property
    def turns(self) -> AsyncCollection[_Document]:
        return self._client[self._database]['turns']

    async def ready(self) -> None:
        """Every operation's guard: refuse while closed, and create the indexes on first use."""
        if self._closed:
            raise RegistryUnavailable('registry is closed')
        if self._indexes_ready:
            return
        # Serves the list's filter and its order together, so a page costs what the page holds.
        await self.conversations.create_index([('namespace_key', 1), ('activity_at', -1), ('_id', -1)])
        await self.turns.create_index([('conversation_id', 1), ('at', 1)])
        self._indexes_ready = True

    async def open(self) -> None:
        if self._closed and self._own_client:
            # A client we created, we can recreate; a closed pymongo client is not reusable.
            self._client = AsyncMongoClient(self._db_url)
            self._indexes_ready = False
        self._closed = False
        await self.ready()

    async def close(self) -> None:
        self._closed = True
        if self._own_client:
            await self._client.close()
