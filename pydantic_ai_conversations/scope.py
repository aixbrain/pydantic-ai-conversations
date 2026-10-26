"""Scope declaration and the namespace key derived from it.

The namespace built from the declared scope fields is simultaneously the partition and the access
rule: every registry method takes `namespace_key` and answers only within it, so a foreign
`conversation_id` resolves to nothing. `conversation_id` itself is never part of the scope -- it is
the feature-owned identity of a conversation, the namespace is the partition it lives in.
"""

from __future__ import annotations

import json
import keyword
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, TypeAlias

from pydantic import BaseModel, Field, create_model

Source = Literal['user', 'extras']


@dataclass(frozen=True)
class ScopeField:
    """One declared partition field; the source prefix is the trust zone.

    `user.*` is read from the verified user object; `extras.*` is a shape-validated client claim.
    """

    source: Source
    name: str


def parse_scope(spec: Iterable[str]) -> tuple[ScopeField, ...]:
    """Parse `('user.tenant_id', 'extras.case_id', ...)` declarations, order-preserving.

    The prefix selects the source; everything after the first dot is one literal name, looked up
    exactly one level deep (`getattr(user, name)` / `extras[name]`) -- project nested values onto
    a flat attribute and declare that. Names must be plain public Python identifiers (no keyword,
    no leading underscore) and unique per source; every excluded shape raises `ValueError` here
    instead of failing late or silently keying the wrong partition (docs/scope-and-access.md
    walks through the failure modes).
    """
    fields: list[ScopeField] = []
    for entry in spec:
        source, _, name = entry.partition('.')
        if (
            not name.isidentifier()
            or keyword.iskeyword(name)
            or name.startswith('_')
            or (source != 'user' and source != 'extras')
        ):
            raise ValueError(
                f"scope entry must be 'user.<attr>' or 'extras.<key>' with <attr>/<key> a single "
                f'plain public Python identifier (no keyword, no leading underscore), got '
                f'{entry!r}. Nested paths are not traversed -- project the value onto one flat '
                f'attribute/key and declare that.'
            )
        scope_field = ScopeField(source, name)
        if scope_field in fields:
            raise ValueError(f'duplicate scope entry {entry!r}')
        fields.append(scope_field)
    return tuple(fields)


class HasUserId(Protocol):
    """The caller identity this package needs: a stable `user_id`, plus whatever attributes the
    configured `user.*` scope fields name. The app-factory's `UserContext` satisfies this.

    `user_id` is a property because a plain protocol attribute is invariant (read AND write),
    which frozen dataclasses can never satisfy structurally.
    """

    @property
    def user_id(self) -> str: ...


def build_namespace(scope: tuple[ScopeField, ...], user: HasUserId, extras: Mapping[str, Any]) -> str:
    """Serialize the caller's scope values into the scalar namespace key: a compact JSON list of
    `(field, value)` pairs, compared by exact string equality.

    Missing values serialize as `''` so single-tenant deployments (tenant_id None) stay
    deterministic; rejecting empty claims is the extras schema's job, not the key's.
    """
    pairs: list[tuple[str, str]] = []
    for scope_field in scope:
        value: Any = getattr(user, scope_field.name) if scope_field.source == 'user' else extras.get(scope_field.name)
        pairs.append((f'{scope_field.source}.{scope_field.name}', '' if value is None else str(value)))
    return json.dumps(pairs, separators=(',', ':'))


def scope_extras_model(scope: tuple[ScopeField, ...]) -> type[BaseModel] | None:
    """The extras-side scope fields as a required, non-empty-string schema; `None` when the scope
    has no extras fields.

    The chat body, the route query parameters, and `ConversationsFeature.namespace_for` all
    validate against a model built here, so the paths cannot drift apart.
    """
    fields: dict[str, Any] = {f.name: (Annotated[str, Field(min_length=1)], ...) for f in scope if f.source == 'extras'}
    return create_model('ScopeExtras', **fields) if fields else None


UserDependency: TypeAlias = Callable[..., HasUserId | Awaitable[HasUserId]]
"""Any FastAPI dependency resolving the request to an object with the user-side scope attributes.
The app-factory's `AuthProvider` satisfies this."""
