"""Scope declaration -> namespace key: the configurable partition that is also the access wall."""

from dataclasses import dataclass

import pytest
from pydantic_ai_app_factory import UserContext

from pydantic_ai_conversations.scope import ScopeField, build_namespace, parse_scope, scope_extras_model


def test_parse_scope_reads_source_prefixes() -> None:
    scope = parse_scope(('user.tenant_id', 'extras.case_id'))
    assert scope == (ScopeField('user', 'tenant_id'), ScopeField('extras', 'case_id'))


def test_parse_scope_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match='scratch.case_id'):
        parse_scope(('scratch.case_id',))


def test_parse_scope_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match='user\\.'):
        parse_scope(('user.',))


def test_parse_scope_rejects_a_dotted_user_name() -> None:
    # 'user.land.id' READS like a nested path but `partition('.')` cuts only at the first dot, so
    # 'land.id' stays ONE name and `getattr(user, 'land.id')` raises AttributeError on every
    # request. Nothing traverses -- reject the declaration at config time instead.
    with pytest.raises(ValueError, match='user\\.land\\.id'):
        parse_scope(('user.land.id',))


def test_parse_scope_rejects_a_dotted_extras_name() -> None:
    # Same misreading on the extras side, but it fails SILENTLY where it isn't schema-validated:
    # `extras.get('land.id')` misses a nested {'land': {'id': ...}} claim and keys the empty
    # partition rather than raising.
    with pytest.raises(ValueError, match='extras\\.land\\.id'):
        parse_scope(('extras.land.id',))


def test_parse_scope_rejects_a_name_that_is_no_python_identifier() -> None:
    # The guard is `isidentifier()`, not merely "no dot": any non-identifier degenerates
    # `scope_extras_model`'s generated `__init__` to `(**extra_data)`, which FastAPI then exposes
    # as one query parameter literally named `extra_data` -- an unsatisfiable route.
    with pytest.raises(ValueError, match='case-id'):
        parse_scope(('extras.case-id',))


def test_parse_scope_rejects_entries_without_a_valid_source_prefix() -> None:
    # No dot at all: `partition('.')` yields an empty name ('user') or an unknown source
    # ('tenant_id', '') -- all three collapse into the same rejection.
    for entry in ('', 'user', 'tenant_id'):
        with pytest.raises(ValueError, match='scope entry'):
            parse_scope((entry,))


def test_parse_scope_rejects_python_keywords() -> None:
    # 'class'/'for' pass `isidentifier()` but no dataclass can declare such an attribute
    # (user side: AttributeError per request) and pydantic's generated `__init__` cannot name
    # such a parameter, degenerating the extras schema to `(**extra_data)` -- the same
    # unsatisfiable-route failure as a non-identifier.
    with pytest.raises(ValueError, match='extras\\.class'):
        parse_scope(('extras.class',))
    with pytest.raises(ValueError, match='user\\.for'):
        parse_scope(('user.for',))


def test_parse_scope_rejects_leading_underscore_names() -> None:
    # The worst failure mode found: pydantic turns a leading-underscore field into a PRIVATE
    # attribute -- `scope_extras_model` then builds a schema with NO fields, `validate({})`
    # passes, and every caller silently keys the empty partition. Dunders ('__base__') are
    # create_model kwargs on top. Both are caught by the same rule.
    with pytest.raises(ValueError, match='extras\\._private'):
        parse_scope(('extras._private',))
    with pytest.raises(ValueError, match='user\\.__base__'):
        parse_scope(('user.__base__',))


def test_parse_scope_rejects_duplicate_fields() -> None:
    # A duplicate doubles its pair inside every namespace key. That works -- until someone
    # removes the duplicate and every existing key silently changes, orphaning all data in the
    # partition. Rejecting at config time prevents the trap before data exists. The same name
    # under DIFFERENT sources is not a duplicate: the source prefix is part of the key label.
    with pytest.raises(ValueError, match='user\\.tenant_id'):
        parse_scope(('user.tenant_id', 'user.tenant_id'))
    assert parse_scope(('user.case_id', 'extras.case_id')) == (
        ScopeField('user', 'case_id'),
        ScopeField('extras', 'case_id'),
    )


def test_namespace_key_is_deterministic_and_ordered() -> None:
    scope = parse_scope(('user.tenant_id', 'user.user_id'))
    user = UserContext(user_id='u1', tenant_id='t1')
    key = build_namespace(scope, user, {})
    assert key == build_namespace(scope, user, {})
    assert 't1' in key and 'u1' in key


def test_namespace_key_reads_extras_fields() -> None:
    scope = parse_scope(('extras.case_id',))
    user = UserContext(user_id='u1')
    assert build_namespace(scope, user, {'case_id': 'case-9'}) != build_namespace(scope, user, {'case_id': 'case-7'})


def test_empty_scope_is_the_degenerate_global_namespace() -> None:
    # An app whose AuthProvider is the entire wall (an HMAC-signed caller) declares no scope: one partition.
    user = UserContext(user_id='u1')
    assert build_namespace((), user, {}) == build_namespace((), UserContext(user_id='u2'), {})


def test_missing_values_serialize_as_empty() -> None:
    # A single-tenant deployment has tenant_id None; the key must still be deterministic.
    scope = parse_scope(('user.tenant_id',))
    key_none = build_namespace(scope, UserContext(user_id='u1'), {})
    key_set = build_namespace(scope, UserContext(user_id='u1', tenant_id='t1'), {})
    assert key_none != key_set


def test_misconfigured_user_field_fails_loudly() -> None:
    # A typo in a user.* scope declaration must not silently collapse isolation.
    scope = parse_scope(('user.tenent_id',))  # deliberate typo
    with pytest.raises(AttributeError):
        build_namespace(scope, UserContext(user_id='u1', tenant_id='t1'), {})


def test_delimiter_values_cannot_collide() -> None:
    # JSON quoting keeps embedded separators from forging a foreign namespace.
    scope = parse_scope(('extras.a', 'extras.b'))
    user = UserContext(user_id='u1')
    key_one = build_namespace(scope, user, {'a': 'x,y', 'b': 'z'})
    key_two = build_namespace(scope, user, {'a': 'x', 'b': 'y,z'})
    assert key_one != key_two


def test_missing_and_empty_extras_value_share_a_partition() -> None:
    # Deliberate: None and '' serialize identically; the extras schema owns non-empty
    # validation. Pinned so a future change cannot silently rely on them differing.
    scope = parse_scope(('extras.case_id',))
    user = UserContext(user_id='u1')
    assert build_namespace(scope, user, {}) == build_namespace(scope, user, {'case_id': ''})


def test_scope_extras_model_requires_non_empty_extras_fields() -> None:
    scope = parse_scope(('user.tenant_id', 'extras.case_id'))
    model = scope_extras_model(scope)
    assert model is not None
    assert model.model_validate({'case_id': 'case-1'}).model_dump() == {'case_id': 'case-1'}
    with pytest.raises(ValueError, match='case_id'):
        model.model_validate({})
    with pytest.raises(ValueError, match='case_id'):
        model.model_validate({'case_id': ''})


def test_scope_extras_model_is_none_for_a_user_only_scope() -> None:
    scope = parse_scope(('user.tenant_id', 'user.user_id'))
    assert scope_extras_model(scope) is None


def test_unicode_identifiers_work_end_to_end() -> None:
    # `isidentifier()` accepts non-ASCII identifiers, and everything downstream handles them:
    # a real named model field (no `(**extra_data)` degeneration) and a clean key. Pinned so a
    # future guard tightening cannot reject them by accident.
    scope = parse_scope(('extras.größe',))
    model = scope_extras_model(scope)
    assert model is not None and list(model.model_fields) == ['größe']
    key = build_namespace(scope, UserContext(user_id='u1'), {'größe': 'XL'})
    assert 'XL' in key


def test_user_scope_field_reads_properties() -> None:
    # The documented recipe for nested user data: project it onto a flat property and declare
    # that. `getattr` resolves properties like plain attributes.
    @dataclass(frozen=True)
    class RichUser(UserContext):
        land_code: str = 'DE'

        @property
        def land_id(self) -> str:
            return self.land_code.lower()

    scope = parse_scope(('user.land_id',))
    assert '"de"' in build_namespace(scope, RichUser(user_id='u1'), {})


def test_field_order_changes_the_key() -> None:
    # Order-preserving cuts both ways: the same fields in a different order key DIFFERENT
    # partitions. Reordering a deployed scope declaration is a data migration, not a cleanup.
    user = UserContext(user_id='u1', tenant_id='t1')
    ab = build_namespace(parse_scope(('user.tenant_id', 'user.user_id')), user, {})
    ba = build_namespace(parse_scope(('user.user_id', 'user.tenant_id')), user, {})
    assert ab != ba


def test_values_are_stringified_so_types_conflate() -> None:
    # `build_namespace` applies `str()`: int 7 and str '7' share a partition. The extras schema
    # enforces str on every validated path; a direct `build_namespace` caller inherits this
    # conflation. Pinned as deliberate -- the key stores scalars, not types.
    scope = parse_scope(('extras.n',))
    user = UserContext(user_id='u1')
    assert build_namespace(scope, user, {'n': 7}) == build_namespace(scope, user, {'n': '7'})


def test_model_config_is_documented_as_a_loud_config_time_failure() -> None:
    # 'model_config' is a plain public identifier the guard passes, but it is `create_model`'s
    # own config kwarg -- the schema build raises. That failure is LOUD and happens at feature
    # construction (the feature builds `extras_type` in `__init__`), which is acceptable; this
    # test documents it. Neighboring 'model_id' must keep working: no overbroad 'model_'
    # rejection.
    with pytest.raises(TypeError):
        scope_extras_model(parse_scope(('extras.model_config',)))
    model = scope_extras_model(parse_scope(('extras.model_id',)))
    assert model is not None and list(model.model_fields) == ['model_id']
