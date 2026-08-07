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
    DelegationTool, OpenTool, ReminderRoutine, ReminderStore, RemoteTool, SetReminderTool,
    ShellTool, Tool, ToolContext, ToolRegistry, new_reminder_store,
};

// Core trait modules (always available)
#[cfg(feature = "artifacts")]
pub mod artifacts;
pub mod auth;
#[cfg(feature = "llm-client")]
pub mod ingest;
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
#[cfg(feature = "artifacts")]
pub use artifacts::{Artifact, ArtifactStore, LocalArtifactStore, NoArtifactStore};
pub use auth::Auth;
#[cfg(feature = "artifacts")]
pub use config::ArtifactsConfig;
pub use config::{
    AgentConfig, MindroidConfig, ModelConfig, OpenToolConfig, ProviderConfig, ShellToolConfig,
    ToolsConfig,
};
pub use core::content::{ContentPart, ContentSource};
pub use core::context::Context;
pub use core::coordinator::{CoordinatorPermit, PerKey, SessionCoordinator};
pub use core::events::PipelineEvent;
#[cfg(feature = "artifacts")]
pub use core::factory::build_artifact_store;
pub use core::factory::credential_kind_from_config;
pub use core::strategy::RunStrategy;
#[cfg(feature = "persona")]
pub use episode::{EpisodeIngestStage, EpisodeReplyIngestStage};
pub use error::{MindroidError, Result};
#[cfg(feature = "llm-client")]
pub use ingest::{Base64Source, Encoder, MediaEncoder, RawInput, ResolvedSource, Source};
pub use memory::{Memory, NoMemory};
#[allow(deprecated)]
pub use models::PersonaCaller;
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
pub use pipeline::extensions::{FileInput, FileInputs};
#[cfg(feature = "artifacts")]
pub use pipeline::presets::artifacts::{ArtifactSet, artifacts_from_store, artifacts_local};
#[cfg(feature = "llm-client")]
pub use pipeline::presets::vision::vision_pipeline;
#[cfg(all(feature = "speech", feature = "llm-client"))]
pub use pipeline::presets::voice::voice_pipeline;
#[cfg(feature = "artifacts")]
pub use pipeline::stages::ArtifactOffload;
#[cfg(feature = "llm-client")]
pub use pipeline::stages::AttachMedia;
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
    GenericLlmProcessor, IngestStage, OpenAiStt, OpenAiSttConfig, OpenAiTts, OpenAiTtsConfig,
    collect_stream,
};
pub use pipeline::stages::{PostProcessor, SimpleContextBuilder};
pub use pipeline::stages::{SttProvider, SttStage, TtsProvider, TtsStage};
pub use pipeline::{Pipeline, PipelineContext, PipelineStage, StreamingStage};
pub use runtime::{
    MessageContext, Routine, RoutineContext, Runtime, RuntimeBuilder, TransportSend,
    TransportSender,
};
#[cfg(feature = "artifacts")]
pub use tools::{GET_ARTIFACT_TOOL, GetArtifactTool};
pub use transport::Transport;
#[cfg(feature = "transport-audio")]
pub use transport::audio::{AudioTransport, AudioTransportConfig};
