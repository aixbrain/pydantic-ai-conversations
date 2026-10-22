# Continuing a Conversation

This guide covers what a client sends on each turn, what the run sees, and how retries, edits,
failed turns, and paused turns work.

Prerequisites: [Getting started](getting-started.md). Lineage and phase are defined in
[How a conversation is stored](concepts.md).

## Naming the previous turn

Every request names the turn it builds on. The value decides what the run sees:

| `previous_run_id` | Meaning | Rejected when |
|---|---|---|
| left out | The first turn of a conversation | The conversation already has a turn: `conversations/previous-run-required` |
| `null` | Start from nothing: a first turn, or a second root beside one | never, by lineage |
| a run id | Continue from that turn and everything behind it | The id is no turn of this conversation: `conversations/run-not-found` |

On `/run` the field sits in `extras`; on `/chat` it is a top-level field of the body. The
camelCase `previousRunId` binds on both. An empty string fails the body validation, as a
plain 422. The status codes of the package's own codes are in the
[reference](api-reference.md#error-codes); on `/chat` every one of them arrives inside the
stream with HTTP 200.

A rejected run records no turn and saves no snapshot. The conversation entry itself is
created before the check, so a rejected first request still leaves an empty entry in the
listing.

The field is required after the first turn so a client cannot continue "whatever came last".
Once a conversation has branches, that phrase has no single answer, and the client knows which
turn it rendered.

## Branches: retry, edit, resume

Two turns naming the same `previous_run_id` are siblings. The transcript lists every turn of
every branch with its link; a client renders the branch ending in the latest turn by following
the links back, and names the turn it continues from on its next request.

Three client actions follow from this:

- **Retry an answer.** Send the prompt again, naming the failed turn's own `previous_run_id`
  (`null` for a first turn). The new turn is a sibling of the failed one.
- **Edit an earlier message.** Send the edited message, naming the turn before the edited one.
  The old turn and everything after it stay stored on their branch.
- **Resume a turn.** Name the turn itself and send no new message. This continues the turn as
  a child: the saved prompt is sent again when the model never answered, tool calls the model
  made but that have no stored result yet are run, and a response pydantic-ai paused to
  continue later is continued. A resume with a new message sends both, the saved
  prompt and the new one.

A request with no new message is accepted only when the named turn left something to resume.
Otherwise it is rejected with `conversations/prompt-required`, because the model would receive
nothing new. `/run` always requires a `prompt`, so a resume without a message goes over `/chat`
or through `agent.run` directly.

## Approving a tool call

A tool that needs approval pauses the run with pydantic-ai's
[deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/). Over `/chat`
the sequence is:

1. The client sends the user message. The stream carries the tool call, its `toolCallId`, and
   a `start` chunk whose `messageId` is the paused turn's run id.
2. The client sends the approval: an assistant message whose `id` is that run id, with a
   `tool-<name>` part in state `approval-responded` carrying the `toolCallId` and
   `{"approved": true}`, and `previousRunId` set to the same run id.
3. The server loads the pending calls from the store, runs the tool, and streams the answer as
   a new turn whose `previous_run_id` is the paused turn.

The client holds no history for this. Approval parts are an AI SDK 6 shape, so the app needs
`vercel_sdk_version=6` or later on `create_agent_app`; the default is 5.

## What the model sees

Before the first model request the run's message list is replaced with the stored branch: the
latest snapshot of each turn on the chain, in order, then the new prompt. The list the client
sent is replaced, never merged, so a client that re-sends its full local history cannot
duplicate a turn. Sending only the new message is enough.

From the first model request on, the run is a plain pydantic-ai run over the conversation.
Other capabilities, including compaction, see it that way. Only the run's new messages are
stored (see [How a conversation is stored](concepts.md#snapshot-and-per-turn-batch)). A
summary of earlier turns is therefore not stored, so a summarizer runs again on the next turn;
a prompt that compaction removed is missing from the stored turn.

The client's list is not inspected or sanitised here. The app factory's endpoints and
pydantic-ai's UI adapters run
[`sanitize_messages`](https://pydantic.dev/docs/ai/api/pydantic-ai/messages/#pydantic_ai.messages.sanitize_messages)
before the run; a hand-written route must do the same.

## Failed turns

A run that raises resolves its turn to `failed`. Its stored messages are the prompt, or the
prompt and the tool rounds it settled before failing; never a tool call without a result. A
failed turn can be resumed or retried like any other. With `describe_failure` configured, the
turn also carries a client-safe reason; see [Extending](extending.md#storing-why-a-turn-failed).

## Limiting the loaded history

`max_history_runs` keeps the last N turns whose stored messages start with a user prompt or
tool results. A resume that starts with the answer shares the slot of the turn it resumed, and
a turn whose prompt was removed by compaction shares the slot of the turn before it. A turn
without a snapshot takes no slot, so a stretch of failed turns cannot blank the history. The cut drops a
tool result whose call fell outside the window and never starts the list with an assistant
message, since providers reject that.

Loading costs one store read per turn on the chain, so this setting is also the cost lever.
It does not shrink what the model sees beyond the cut; use a
[compaction](https://pydantic.dev/docs/ai/capabilities/compaction/) capability on the agent
for that.

## Rejected drafts

pydantic-ai's output-validator retry leaves each rejected draft and its retry prompt in the
history, so a later turn would send them to the model again. `StripRejectedDrafts`, registered
on the agent beside the feature's capabilities, removes them from the model request only. Run
state and stored snapshots keep them, so transcripts still show what happened, and the current
run's own draft is never touched, because an in-flight retry needs it.

```python
from pydantic_ai import Agent
from pydantic_ai_app_factory import AppDeps
from pydantic_ai_conversations import StripRejectedDrafts

agent = Agent('test', deps_type=AppDeps, capabilities=[StripRejectedDrafts()])
```

Two limits. Only text-path rejections are recognised; a rejection in `ToolOutput` mode looks
like a function tool's retry and is left alone. And when retries are exhausted, pydantic-ai
raises before appending the last retry prompt, so that final draft carries no marker and
stays.

What you end up with: a client that names the turn it continues from, retries and edits by
naming the right predecessor, and approves tool calls without holding history. Going deeper:
[The /conversations routes](http-api.md) for the transcript a frontend renders.
