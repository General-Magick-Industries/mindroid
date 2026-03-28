pub mod core;
pub mod prelude;

// Re-export core submodules at crate root for convenience
pub use core::config;
pub use core::error;
pub use core::models;
pub use core::runtime;

pub mod skills;
pub use skills::{
    ActivationCriteria, LoadedSkill, SkillManifest, SkillRegistry, SkillSet, SkillSource, SkillTrust,
    prefilter_skills, build_skill_index, ReadSkillTool,
};

pub mod tools;
pub use tools::{Tool, ToolRegistry, ShellTool, OpenTool, SetReminderTool, ReminderRoutine, ReminderStore, new_reminder_store};
#[cfg(feature = "llm-client")]
pub use pipeline::stages::{ToolExecutorStage, ToolCallParser, ParsedToolCall, XmlToolCallParser};

// Core trait modules (always available)
pub mod pipeline;
pub mod transport;
pub mod auth;
pub mod memory;
pub mod observer;

// Optional implementation modules
#[cfg(feature = "llm-client")]
pub mod llm_client;

#[cfg(feature = "persona")]
pub mod persona;

#[cfg(feature = "identity")]
pub mod identity;
#[cfg(feature = "identity")]
pub use identity::{CanonicalUserId, IdentityResolutionStage, IdentityResolver};

// Re-export core types at crate root
pub use config::{AgentConfig, MindroidConfig, ModelConfig, ProviderConfig, ToolsConfig, ShellToolConfig, OpenToolConfig};
pub use error::{MindroidError, Result};
pub use auth::Auth;
pub use memory::{Memory, NoMemory};
pub use models::{
    ChannelType, LlmMessage, Message, MessageType, Response, Role, SenderType, StreamEvent, TokenUsage,
};
pub use observer::{NoObserver, Observer};
pub use pipeline::{Pipeline, PipelineContext, PipelineStage, StreamingStage};
pub use pipeline::context::{ContextPreparer, ContextProvider};
pub use pipeline::coordination::EngagementTracker;
pub use pipeline::stages::{SimpleContextBuilder, PostProcessor};
pub use pipeline::stages::{SttProvider, SttStage, TtsProvider, TtsStage};
#[cfg(feature = "llm-client")]
pub use pipeline::stages::{GenericLlmProcessor, collect_stream, OpenAiStt, OpenAiSttConfig, OpenAiTts, OpenAiTtsConfig};
#[cfg(feature = "speech")]
pub use pipeline::stages::{DeepgramStt, DeepgramSttConfig, DeepgramTts, DeepgramTtsConfig};
#[cfg(feature = "llm-client")]
pub use pipeline::stages::gate::{AndGate, CoordinationGate, Gate, OrGate, RelevanceGate};
pub use runtime::{MessageContext, Runtime, RuntimeBuilder, TransportSend, TransportSender, Routine, RoutineContext};
pub use transport::Transport;
#[cfg(feature = "persona")]
pub use persona::{MagickmindPersonaClient, PersonaContextBuilder, PersonaProvider};
#[cfg(feature = "transport-audio")]
pub use transport::audio::{AudioTransport, AudioTransportConfig};
#[cfg(feature = "transport-audio")]
pub use pipeline::stages::AudioOutputStage;
#[cfg(feature = "transport-audio")]
pub use pipeline::stages::StreamingTtsTransformer;
#[cfg(feature = "transport-audio")]
pub use pipeline::extensions::{AudioInput, AudioOutput, TextInput};
