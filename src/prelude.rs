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
