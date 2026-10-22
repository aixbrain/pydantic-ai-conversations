# How a Conversation Is Stored

This page defines the terms the other pages use -- conversation, turn, snapshot, lineage,
phase, namespace -- and says which part of the stack owns what. Read it once; the guides assume
these words.

Prerequisites: an app built with
[pydantic-ai-app-factory](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/building-an-app.md),
and a look at the harness's
[Step Persistence](https://pydantic.dev/docs/ai/harness/step-persistence/), which this package
builds on.

## Where things live

| You want | Where it lives |
|---|---|
| A run's messages saved durably, and a run resumed after a crash | `pydantic-ai-harness`: the `StepPersistence` capability writes snapshots to a `StepStore`, its storage interface |
| Conversations listed, titled, deleted, and kept apart per user or tenant | This package: the `ConversationRegistry` |
| The stored history loaded into each run, and a turn linked to the one before it | This package: `ConversationsCapability` |
| `/conversations` routes for a frontend | This package: `build_conversation_router`, mounted by `ConversationsFeature` |
| Who the caller is, and whether they may act in a tenant or case | The app's `AuthProvider` |
| What the model sees after the history is loaded (compaction, trimming) | The agent's own capabilities |

The package adds no storage mechanics. Messages live in the harness's step store; the registry
holds metadata and turn records only.

## Conversation

A conversation is identified by its `conversation_id`, the id the client sends: the chat `id` on
`/chat`, the `conversation_id` field on `/run`. The registry keeps one entry per conversation
with who created it, when, the time of the last message, a message count, an optional title,
create-time `attrs`, and a deletion mark. No messages live there.

A conversation comes into being on its first run; the registry call that creates it is
idempotent, so no prior request is needed. An app that wants `attrs` on the entry creates it
first, through `POST /conversations` or `register_conversation` (see
[The /conversations routes](http-api.md#attrs-and-the-pre-create-pattern)).

Ids must be unguessable, uuid4 for example. The first run's create call raises not-found for an
id that exists in another partition, so a guessable id would tell a caller which ids exist.

## Turn and run

A turn is one agent run inside a conversation. Its id is the run's `run_id`, the same
`ctx.run_id` pydantic-ai gives every run. The registry records one `TurnRecord` per run: who
wrote it, when it started, the turn it builds on, its phase, and why it failed if it did.

## Snapshot and per-turn batch

The harness saves snapshots of a run's message list: before the first model request, so a
prompt survives a run that fails before answering; after each tool round, once every call in
it has returned; and when the run ends. This package puts one write seam in front of the store,
`PreparingStepStore`, whose first transform, `NewMessagesPreparer`, cuts every snapshot down
to the run's new messages before it is saved. They start at the first message stamped with the
run's `run_id`, and everything from there on is stored. A message ahead of that point is not,
such as a summary that compaction puts at the start of the list. The latest snapshot of a run is therefore that turn's message batch, and a conversation's
history is the batches of its turns concatenated. Stored snapshots are never rewritten
afterwards; what lies in the store is what every reader sees.

## Lineage: `previous_run_id`, chains, and branches

Every run names the turn it builds on through `previous_run_id`. The registry stores that link
per turn. The chain of links from the first turn to the named one is the history the run sees;
two turns naming the same `previous_run_id` are siblings, which is how a retry and an edit look.
Every turn stays stored, and the transcript lists all branches.
[Continuing a conversation](continuing-a-conversation.md) gives the rules a client follows.

## Phase

A turn's phase is `started`, `answered`, or `failed`. The capability writes `started` when the
run begins and resolves it when the run ends. It is not a liveness signal: a run killed hard
stays `started` forever. It is also independent of the snapshot: a failed run can leave a
complete snapshot behind, for example a crash after a settled tool round.

## Namespace

The feature's scope -- by default the caller's `tenant_id` and `user_id` -- is serialised into a
namespace key. Every registry call takes that key and answers only within it, so a conversation
id from another namespace resolves to nothing. The namespace is both the partition and the only
access wall this package enforces; see [Scope and access](scope-and-access.md).

## What this package does not do

- Decide who the caller is or what they may act on. That is the app's `AuthProvider`.
- Sanitise the message list a client sends. The app factory and pydantic-ai's UI adapters do;
  a hand-written route must.
- Resume a run cut off in the middle of a tool call. Only the last settled boundary is kept.
- Shrink what the model sees. A compaction capability on the agent does that after the history
  is loaded.

Going deeper: [Getting started](getting-started.md) wires the feature into an app;
[Continuing a conversation](continuing-a-conversation.md) covers the per-turn contract.
