# Documentation

Start at the top; each page assumes the ones above it.

1. [Getting started](getting-started.md): install the package, add the feature to an
   app-factory app, and drive a first conversation.
1. [How a conversation is stored](concepts.md): the terms (conversation, turn, snapshot,
   lineage, phase, namespace) and which part of the stack owns what.
1. [Continuing a conversation](continuing-a-conversation.md): what a client sends per turn;
   retries, edits, failed and paused turns; limiting the loaded history.
1. [Scope and access](scope-and-access.md): partitions per tenant, user, or case, and where
   the package's access wall ends.
1. [The /conversations routes](http-api.md): listing, transcript, deletion, and the switches a
   frontend reads.
1. [Deleting conversations](deleting.md): soft delete, purge, and cleanup hooks.
1. [Backends](backends.md): SQLite, MongoDB, PostgreSQL; schema and upgrades.
1. [Extending](extending.md): your own registry or step store, snapshot preparers, transcript
   views, add-ons, failure reasons, use without the app factory.
1. [Operations](operations.md): readiness, concurrency, logging, known limits, version pins.
1. [Testing](testing.md): testing an app with conversations; the registry conformance suite.
1. [Reference](api-reference.md): signatures, defaults, wire shapes, error codes.
