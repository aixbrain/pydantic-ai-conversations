"""Which step-store failures count as temporary, for the write seam and both read paths."""

from __future__ import annotations

from typing import cast

from pydantic_ai_app_factory import AgentError


def is_transient(store: object, exc: BaseException) -> bool:
    """Whether `store` declares `exc` a temporary failure (`stores.SupportsTransientErrors`).

    An `AgentError` never counts, because it carries its own retry decision. A store that declares
    nothing has no temporary failures, so its exceptions reach the caller unchanged.
    """
    if isinstance(exc, AgentError):
        return False
    # `SupportsTransientErrors` declares the attribute's type, and nothing validates it: as with
    # `delete_runs` and `ping`, a structural check sees that a name exists, not what it holds.
    declared = cast('tuple[type[BaseException], ...]', getattr(store, 'transient_errors', ()))
    return isinstance(exc, declared)
