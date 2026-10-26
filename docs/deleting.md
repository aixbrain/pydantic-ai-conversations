# Deleting Conversations

This guide covers the two deletion paths, what each removes and what remains, and how to clean
up data of your own alongside.

Prerequisites: [How a conversation is stored](concepts.md).

## Soft delete

`DELETE /conversations/{id}` marks the entry deleted. It disappears from listing and reading,
a run naming it is rejected with `conversations/not-found`, and the id cannot be reused. The
messages stay in the step store untouched. A second delete answers 404.

Nothing expires on its own: there is no retention setting, and soft-deleted entries cannot be
listed through the routes. An app that wants erasure after a soft delete tracks the ids itself
and purges them.

## Purge

A hard delete is a Python call; there is no route for it:

```python
await feature.purge(namespace_key=feature.namespace_for(user, extras), conversation_id='c1')
```

It runs three steps in order:

1. The registry tombstones the entry and deletes its turn records. This goes first so the
   conversation is unreachable even if a later step fails.
2. If the step store supports deletion, the conversation's runs are removed: events,
   snapshots and their messages, tool effects, and the harness's record of saved snapshot
   keys. The shipped SQLite, MongoDB, and PostgreSQL stores do; a store without `delete_runs`
   keeps the snapshots. See [Extending](extending.md#your-own-step-store) for adding it.
3. The cleanup hooks run.

The runs removed are the conversation's own plus every run spawned from them, transitively:
helper agents called inside a tool forward conversation content, so their runs go too. A
nested agent that carries its own `ConversationsCapability` looks like such a helper and is
deleted with it; running one is not supported.

A purge of an id the namespace does not own is a silent no-op: nothing is deleted, no hook
runs.

### What remains after a purge

- The registry keeps a tombstone row for the id, with the namespace key (which embeds the
  tenant and user ids of the scope), `created_by`, `created_at`, `last_message_at`, and
  `message_count`. Title and `attrs` are cleared. The id stays claimed forever, in every
  namespace, because substrate deletion is best-effort and a released id could inherit
  surviving history.
- Media the step store keeps outside the snapshots (files and images, content-addressed and
  possibly shared between runs) is left to the store's own garbage collection.
- The MongoDB and PostgreSQL step stores keep each snapshot message once, shared by every run
  whose snapshots contain it, so a purge keeps the messages another run's snapshot still
  references. A snapshot another run saves while the purge runs can reference a message the
  purge deletes, and then fails to load. Each snapshot holds only its own run's turn, so runs
  rarely share a message, and the purge does not guard against that case.
- With the MongoDB step store, a `delete_runs` that fails midway leaves the rest of the run in
  place until the purge is retried. It deletes the run records last, and a retried purge finds
  the runs through those records. The SQLite and PostgreSQL stores delete in one transaction.

## Cleanup hooks

Hooks run on both paths and receive the kind, the ids, and the entry as it was:

```python
from pydantic_ai_conversations import ConversationsFeature, DeletionEvent


async def drop_attachments(event: DeletionEvent) -> None:
    await attachments.delete(event.conversation_id)  # your own storage


feature = ConversationsFeature(registry=..., step_store=..., on_delete=[drop_attachments])
```

Add-ons that implement `SupportsDeletion` run here too, after the app's hooks; the
[turn feedback add-on](feedback.md) deletes its entries on purge and keeps them through a soft
delete. On soft delete the hooks run after the mark is written and before the 204 returns; a
failing hook is logged and does not fail the response. On purge a failing hook does not stop the
others, and the failures are raised together as one `ExceptionGroup` after all have run, so a
caller erasing personal data learns the erasure is incomplete and retries. Hooks must be
idempotent for that reason. `event.meta` is `None` on a purge of an entry that was already
soft-deleted or purged, because the registry hides both.

What you end up with: a soft delete for the frontend, a purge for data erasure with a known
residue, and hooks for data of your own. Going deeper: [Backends](backends.md).
