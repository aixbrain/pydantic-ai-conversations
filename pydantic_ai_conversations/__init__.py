"""Conversation persistence for pydantic-ai-app-factory, backed by pydantic-ai-harness.

Contract names are re-exported here from their public submodules -- some (`ConversationsCapability`,
`build_conversation_router`, the wire models, `OnDeleteHook`/`UserDependency`) are implemented in
private (underscore-prefixed) modules; the package root, not the module they happen to live in, is
the supported import surface. `testing` (the registry conformance suite) is imported from its
submodule directly: `from pydantic_ai_conversations.testing import RegistryContract`.
"""

from pydantic_ai_conversations._capability import (
    ConversationsCapability,
    FailureDescriber,
    SupportsConversationDeps,
    agent_error_mapper,
)
from pydantic_ai_conversations._drafts import StripRejectedDrafts, rejected_draft_pairs, strip_rejected_drafts
from pydantic_ai_conversations._loader import TranscriptTurn, TranscriptView, load_transcript
from pydantic_ai_conversations._routes import (
    ConversationDetail,
    ConversationList,
    ConversationOut,
    DeletionEvent,
    DeletionKind,
    OnDeleteHook,
    TranscriptTurnOut,
    TurnErrorOut,
    UIMessageEncoder,
    build_conversation_router,
)
from pydantic_ai_conversations.addons import (
    AddonCaller,
    AddonRouteGuard,
    AddonTurnQuery,
    ConversationAddon,
    SupportsRouters,
    SupportsTurnFields,
)
from pydantic_ai_conversations.errors import ConversationNotFound, RegistryUnavailable
from pydantic_ai_conversations.feature import ConversationsFeature
from pydantic_ai_conversations.registry import (
    ConversationMeta,
    ConversationRegistry,
    InMemoryConversationRegistry,
    RunPhase,
    TurnError,
    TurnRecord,
)
from pydantic_ai_conversations.scope import (
    HasUserId,
    ScopeField,
    UserDependency,
    build_namespace,
    parse_scope,
    scope_extras_model,
)
from pydantic_ai_conversations.stores import (
    NewMessagesPreparer,
    PreparingStepStore,
    SnapshotPreparer,
    SupportsPing,
    SupportsRunDeletion,
    SupportsTransientErrors,
)

__all__ = [
    'AddonCaller',
    'AddonRouteGuard',
    'AddonTurnQuery',
    'ConversationAddon',
    'ConversationDetail',
    'ConversationList',
    'ConversationMeta',
    'ConversationNotFound',
    'ConversationOut',
    'ConversationRegistry',
    'ConversationsCapability',
    'ConversationsFeature',
    'DeletionEvent',
    'DeletionKind',
    'FailureDescriber',
    'HasUserId',
    'InMemoryConversationRegistry',
    'OnDeleteHook',
    'NewMessagesPreparer',
    'RegistryUnavailable',
    'RunPhase',
    'ScopeField',
    'SnapshotPreparer',
    'StripRejectedDrafts',
    'SupportsConversationDeps',
    'SupportsPing',
    'SupportsRouters',
    'SupportsRunDeletion',
    'SupportsTransientErrors',
    'SupportsTurnFields',
    'TranscriptTurn',
    'TranscriptTurnOut',
    'TurnError',
    'TurnErrorOut',
    'TranscriptView',
    'PreparingStepStore',
    'TurnRecord',
    'UIMessageEncoder',
    'UserDependency',
    'agent_error_mapper',
    'build_conversation_router',
    'build_namespace',
    'load_transcript',
    'parse_scope',
    'rejected_draft_pairs',
    'scope_extras_model',
    'strip_rejected_drafts',
]
