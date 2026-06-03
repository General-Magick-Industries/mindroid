//! Convenience re-exports for common mindroid types and traits.
//!
//! ```ignore
//! use mindroid::prelude::*;
//! ```

// Core types
pub use crate::config::{AgentConfig, MindroidConfig};
pub use crate::error::{MindroidError, Result};
pub use crate::models::{LlmMessage, Message, Response, Role, StreamEvent};

// Core traits
pub use crate::auth::Auth;
pub use crate::core::context::Context;
pub use crate::memory::{Memory, NoMemory};
pub use crate::observer::{NoObserver, Observer};
pub use crate::pipeline::context::{
    ContextPreparer, ContextProvider, PrepareOutcome, ProviderWarning,
};
pub use crate::pipeline::{Pipeline, PipelineContext, PipelineStage, StreamingStage};
pub use crate::tools::{Tool, ToolRegistry};
pub use crate::transport::Transport;

// Runtime
pub use crate::runtime::{
    MessageContext, Routine, RoutineContext, Runtime, RuntimeBuilder, TransportSend,
    TransportSender,
};

// Session coordination
pub use crate::core::coordinator::{CoordinatorPermit, PerKey, SessionCoordinator};
pub use crate::core::strategy::RunStrategy;

// Pipeline combinators
pub use crate::pipeline::combinators::{ApprovalStage, BranchStage, RetryStage, RouteFn, RouterStage};

// Tools
pub use crate::tools::DelegationTool;

// Pipeline presets
#[cfg(all(feature = "speech", feature = "llm-client"))]
pub use crate::pipeline::presets::voice::voice_pipeline;
#[cfg(feature = "llm-client")]
pub use crate::pipeline::presets::vision::vision_pipeline;
