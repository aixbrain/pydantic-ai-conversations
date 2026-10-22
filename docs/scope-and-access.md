# Scope and Access

This guide shows you how conversations are kept apart per tenant, user, or case, and where the
package's access wall ends and your `AuthProvider` begins.

Prerequisites: [Getting started](getting-started.md), and the app factory's
[Requests and identity](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/auth.md).

## Declaring the scope

The scope is a tuple of field names, each prefixed with its source:

```python
ConversationsFeature(registry=..., step_store=..., scope=('user.tenant_id', 'user.user_id'))
```

That is the default: one partition per user within a tenant. `user.<attr>` is read from the
user object your `AuthProvider` returns, with `getattr`. `extras.<key>` is read from the request
extras. Lookups are one level deep; project a nested value onto a flat attribute and declare
that. A name must be a plain public identifier, and a duplicate entry raises `ValueError` when
the feature is built.

The user object needs a `user_id` and whatever `user.*` attributes the scope names. The app
factory's `UserContext` has both `user_id` and `tenant_id`.

## The namespace key

The scope values are serialised into one string, a compact JSON list of pairs:

```
[["user.tenant_id","t1"],["user.user_id","alice"]]
```

Every registry call takes this key and answers only within it. A conversation id from another
namespace resolves to nothing: 404 on the routes, `conversations/not-found` on the run path,
never 403 or 409. Anything but not-found would tell a caller that an id exists somewhere.

A value of `None` serialises as an empty string, so a single-tenant deployment with
`tenant_id=None` still keys one deterministic partition.

## Client-side scope fields

An `extras.*` field is a claim the client makes. It becomes a required, non-empty field in the
chat body and in `/run`'s `extras`, and a required query parameter on every `/conversations`
route. The package validates its shape and nothing else.

**An app whose scope contains `extras.*` must verify the claim in its own `AuthProvider`.**
Otherwise any authenticated caller can read and delete any partition by naming it.

The typical use is a conversation shared by several people on one case:

```python
feature = ConversationsFeature(
    registry=registry,
    step_store=step_store,
    scope=('user.tenant_id', 'extras.case_id'),
)
```

Alice and Bob, both in tenant `t1`, pass the same `case_id` and see one conversation; each turn
records its author. Carol in tenant `t2` passes the same `case_id` and hits the wall, because her
namespace differs. Whether Bob is a member of the case is a question the `AuthProvider` answers
before the run starts, for example by looking the membership up and rejecting the request.
`pydantic_ai_conversations_examples/shared_cases.py` is a runnable app with such a check.

On `/chat` the field is a top-level body field, `caseId` or `case_id`; on `/run` it sits in
`extras`, in either spelling; on the routes it is a query parameter in snake_case only,
`?case_id=case-9`.

## The namespace in server-side code

`ConversationsFeature.namespace_for(user, extras)` derives the key the same way the request
paths do. Use it for `purge` and for creating a conversation with `attrs` before the first run
(see [Deleting conversations](deleting.md) and
[The /conversations routes](http-api.md#attrs-and-the-pre-create-pattern)). It validates
`extras` against the same schema the requests use, so an omitted scope field raises a
`ValidationError` instead of keying an empty partition.

What you end up with: a partition key derived from verified identity, plus client claims your
`AuthProvider` has checked. Going deeper: [The /conversations routes](http-api.md).
