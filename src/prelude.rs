//! Convenience re-exports for common mindroid types and traits.
//!
//! ```ignore
//! use mindroid::prelude::*;
//! ```

// Core submodule re-exports
pub use crate::core::config;
pub use crate::core::error;
pub use crate::core::models;
pub use crate::core::runtime;

// Skills
pub use crate::skills::{
    ActivationCriteria, LoadedSkill, ReadSkillTool, SkillManifest, SkillRegistry, SkillSet,
    SkillSource, SkillTrust, build_skill_index, prefilter_skills,
};

// Tools
pub use crate::tools::{
    OpenTool, ReminderRoutine, ReminderStore, SetReminderTool, ShellTool, Tool, ToolRegistry,
    new_reminder_store,
};
#[cfg(feature = "llm-client")]
pub use crate::pipeline::stages::{
    ParsedToolCall, ToolCallParser, ToolExecutorStage, XmlToolCallParser,
};
#[cfg(feature = "mcp")]
pub use crate::tools::mcp::{McpClient, McpToolWrapper, load_mcp_tools};

// Auth
pub use crate::auth::Auth;

// Config
pub use crate::config::{
    AgentConfig, CorpusConfig, McpServerConfig, MindroidConfig, ModelConfig, OpenToolConfig,
    ProviderConfig, ResolvedProvider, ShellToolConfig, ToolsConfig,
};

// Error
pub use crate::error::{MindroidError, Result};

// Memory
pub use crate::memory::{Memory, NoMemory};

// Models
pub use crate::models::{
    ChannelType, LlmMessage, Message, MessageType, Response, Role, SenderType, StreamEvent,
    TokenUsage,
};

// Observer
pub use crate::observer::{NoObserver, Observer};

// Pipeline
pub use crate::pipeline::context::{ContextPreparer, ContextProvider};
pub use crate::pipeline::coordination::EngagementTracker;
pub use crate::pipeline::stages::{PostProcessor, SimpleContextBuilder};
pub use crate::pipeline::stages::{SttProvider, SttStage, TtsProvider, TtsStage};
pub use crate::pipeline::{Pipeline, PipelineContext, PipelineStage, StreamingStage};

// Runtime
pub use crate::runtime::{
    MessageContext, Routine, RoutineContext, Runtime, RuntimeBuilder, TransportSend,
    TransportSender,
};

// Transport
pub use crate::transport::Transport;

// -- Feature-gated re-exports -------------------------------------------------

// Corpus
#[cfg(feature = "corpus")]
pub use crate::corpus::{CorpusClient, CorpusContextProvider};

// Identity
#[cfg(feature = "identity")]
pub use crate::identity::{CanonicalUserId, IdentityResolutionStage, IdentityResolver};

// Persona
#[cfg(feature = "persona")]
pub use crate::persona::{MagickmindPersonaClient, PersonaContextBuilder, PersonaProvider};

// LLM client
#[cfg(feature = "llm-client")]
pub use crate::pipeline::stages::gate::{AndGate, CoordinationGate, Gate, OrGate, RelevanceGate};
#[cfg(feature = "llm-client")]
pub use crate::pipeline::stages::{CorpusGateDecision, CorpusGateStage};
#[cfg(all(feature = "corpus", feature = "llm-client"))]
pub use crate::pipeline::stages::CorpusDistillStage;
#[cfg(feature = "llm-client")]
pub use crate::pipeline::stages::{
    GenericLlmProcessor, OpenAiStt, OpenAiSttConfig, OpenAiTts, OpenAiTtsConfig, collect_stream,
};

// Speech
#[cfg(feature = "speech")]
pub use crate::pipeline::stages::{DeepgramStt, DeepgramSttConfig, DeepgramTts, DeepgramTtsConfig};

// Audio transport
#[cfg(feature = "transport-audio")]
pub use crate::pipeline::extensions::{AudioInput, AudioOutput, TextInput};
#[cfg(feature = "transport-audio")]
pub use crate::pipeline::stages::AudioOutputStage;
#[cfg(feature = "transport-audio")]
pub use crate::pipeline::stages::StreamingTtsTransformer;
#[cfg(feature = "transport-audio")]
pub use crate::transport::audio::{AudioTransport, AudioTransportConfig};
