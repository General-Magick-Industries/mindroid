pub mod core;
pub mod omni;
pub mod prelude;
pub mod voice;

// Re-export core submodules at crate root for convenience
pub use core::config;
pub use core::error;
pub use core::models;
pub use core::runtime;

pub mod skills;
pub use skills::{
    ActivationCriteria, LoadedSkill, ReadSkillTool, SkillManifest, SkillRegistry, SkillSet,
    SkillSource, SkillTrust, build_skill_index, prefilter_skills,
};

pub mod tools;
#[cfg(feature = "llm-client")]
pub use pipeline::stages::{ParsedToolCall, ToolCallParser, ToolExecutorStage, XmlToolCallParser};
pub use tools::{
    DelegationTool, OpenTool, ReminderRoutine, ReminderStore, SetReminderTool, ShellTool, Tool,
    ToolRegistry, new_reminder_store,
};

// Core trait modules (always available)
pub mod auth;
pub mod memory;
pub mod observer;
pub mod pipeline;
pub mod transport;

// Optional implementation modules
#[cfg(feature = "llm-client")]
pub mod llm_client;

#[cfg(feature = "persona")]
pub mod episode;
#[cfg(feature = "persona")]
pub mod persona;

#[cfg(feature = "identity")]
pub mod identity;
#[cfg(feature = "identity")]
pub use identity::{CanonicalUserId, IdentityResolutionStage, IdentityResolver};

// Re-export core types at crate root
pub use auth::Auth;
pub use config::{
    AgentConfig, MindroidConfig, ModelConfig, OpenToolConfig, ProviderConfig, ShellToolConfig,
    ToolsConfig,
};
pub use core::content::{ContentPart, ContentSource};
pub use core::context::Context;
pub use core::coordinator::{CoordinatorPermit, PerKey, SessionCoordinator};
pub use core::events::PipelineEvent;
pub use core::factory::credential_kind_from_config;
pub use core::strategy::RunStrategy;
#[cfg(feature = "persona")]
pub use episode::{EpisodeIngestStage, EpisodeReplyIngestStage};
pub use error::{MindroidError, Result};
pub use memory::{Memory, NoMemory};
pub use models::{
    ChannelType, CredentialKind, LlmMessage, Message, MessageType, Response, Role, SenderType,
    StreamEvent, TokenUsage,
};
pub use observer::{NoObserver, Observer};
#[cfg(feature = "persona")]
pub use persona::{
    ConversationHistory, MagickmindPersonaClient, MagickmindPersonaStage, PersonaContextBuilder,
    PersonaId, PersonaProvider,
};
pub use pipeline::combinators::{ApprovalStage, BranchStage, RetryStage, RouteFn, RouterStage};
pub use pipeline::context::{ContextPreparer, ContextProvider, PrepareOutcome, ProviderWarning};
pub use pipeline::coordination::EngagementTracker;
#[cfg(feature = "transport-audio")]
pub use pipeline::extensions::{AudioInput, AudioOutput, TextInput};
#[cfg(feature = "llm-client")]
pub use pipeline::presets::vision::vision_pipeline;
#[cfg(all(feature = "speech", feature = "llm-client"))]
pub use pipeline::presets::voice::voice_pipeline;
#[cfg(feature = "transport-audio")]
pub use pipeline::stages::AudioOutputStage;
#[cfg(feature = "transport-audio")]
pub use pipeline::stages::StreamingTtsTransformer;
#[cfg(feature = "llm-client")]
pub use pipeline::stages::gate::{AndGate, CoordinationGate, Gate, OrGate, RelevanceGate};
#[cfg(feature = "speech")]
pub use pipeline::stages::{DeepgramStt, DeepgramSttConfig, DeepgramTts, DeepgramTtsConfig};
#[cfg(feature = "llm-client")]
pub use pipeline::stages::{
    GenericLlmProcessor, OpenAiStt, OpenAiSttConfig, OpenAiTts, OpenAiTtsConfig, collect_stream,
};
pub use pipeline::stages::{PostProcessor, SimpleContextBuilder};
pub use pipeline::stages::{SttProvider, SttStage, TtsProvider, TtsStage};
pub use pipeline::{Pipeline, PipelineContext, PipelineStage, StreamingStage};
pub use runtime::{
    MessageContext, Routine, RoutineContext, Runtime, RuntimeBuilder, TransportSend,
    TransportSender,
};
pub use transport::Transport;
#[cfg(feature = "transport-audio")]
pub use transport::audio::{AudioTransport, AudioTransportConfig};
