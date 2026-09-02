#[cfg(all(feature = "magickmind", feature = "llm-hosted"))]
pub mod corpus;
pub mod delegation;
#[cfg(feature = "artifacts")]
pub mod get_artifact;
#[cfg(feature = "magickmind")]
pub mod magickmind;
pub mod open;
mod registry;
pub mod reminder;
pub mod remote;
pub mod shell;
pub mod untrusted;

#[cfg(all(feature = "magickmind", feature = "llm-hosted"))]
pub use corpus::{CorpusCatalog, CorpusTool};
pub use delegation::DelegationTool;
#[cfg(feature = "artifacts")]
pub use get_artifact::{GET_ARTIFACT_TOOL, GetArtifactTool};
#[cfg(feature = "magickmind")]
pub use magickmind::{
    AgentCredentials, AgentCredentialsStage, EpisodicMemoryTool, RecallTimeWindowTool,
};
pub use open::OpenTool;
pub(crate) use registry::MAX_REMOTE_TOOL_PROMPT_BYTES;
pub use registry::{DynamicRegistry, ToolRegistry};
pub use reminder::{ReminderRoutine, ReminderStore, SetReminderTool, new_reminder_store};
pub use remote::{
    ManifestStage, ManifestTool, PerTurnTools, PerTurnToolsStage, RemoteTool, ToolsManifest,
};
pub use shell::ShellTool;
pub use untrusted::wrap_untrusted;

use async_trait::async_trait;
use std::collections::HashMap;

use serde_json::Value;
#[cfg(feature = "artifacts")]
use std::sync::Arc;

use crate::error::Result;

/// Per-invocation context passed to a tool: the message's channel/sender plus a
/// typed extension map. Backend-specific data (credentials, agent id) rides in
/// the map, set by a stage, so tools stay transport-agnostic.
///
/// Cloning shares the ext map (it's `Arc`-backed), so a value `set` on one clone
/// is visible through the others — this is how a stage hands data to the executor.
#[derive(Default, Clone)]
pub struct ToolContext {
    /// Channel the message arrived on (`"stdio"` for the stdio transport; a
    /// workspace id on Centrifugo). Never empty in practice — an empty value
    /// makes artifact scoping reject the message.
    pub channel_id: String,
    pub sender_id: String,
    ext: std::sync::Arc<std::sync::RwLock<HashMap<std::any::TypeId, ExtValue>>>,
}

type ExtValue = Box<dyn std::any::Any + Send + Sync>;

impl ToolContext {
    /// Attach a typed value for tools to read (replaces any existing of this type).
    pub fn set<T: Clone + Send + Sync + 'static>(&self, value: T) {
        self.ext
            .write()
            .unwrap()
            .insert(std::any::TypeId::of::<T>(), Box::new(value));
    }

    /// Read a previously attached value (cloned out).
    pub fn get<T: Clone + Send + Sync + 'static>(&self) -> Option<T> {
        self.ext
            .read()
            .unwrap()
            .get(&std::any::TypeId::of::<T>())
            .and_then(|v| v.downcast_ref::<T>())
            .cloned()
    }

    /// Drop all attached values.
    pub fn clear(&self) {
        self.ext.write().unwrap().clear();
    }
}

/// A capability the agent can invoke on the local computer.
///
/// Implement this trait to add any new tool — the agent will automatically
/// receive its description and be able to call it.
///
/// # Example
///
/// ```ignore
/// use mindroid::tools::Tool;
/// use serde_json::{json, Value};
///
/// struct EchoTool;
///
/// #[async_trait::async_trait]
/// impl Tool for EchoTool {
///     fn name(&self) -> &str { "echo" }
///     fn description(&self) -> &str { "Echo back the input text." }
///     fn parameters_schema(&self) -> Value {
///         json!({ "type": "object", "properties": { "text": { "type": "string" } }, "required": ["text"] })
///     }
///     async fn execute(&self, args: Value, _ctx: &mindroid::tools::ToolContext) -> mindroid::Result<String> {
///         Ok(args["text"].as_str().unwrap_or("").to_string())
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique name used in tool calls (e.g. `"shell"`, `"open"`).
    fn name(&self) -> &str;

    /// Human-readable description injected into the LLM's system prompt.
    fn description(&self) -> &str;

    /// JSON Schema object describing the tool's arguments.
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with parsed arguments and per-invocation context.
    /// Returns output as a plain string.
    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<String>;

    /// Whether this tool is executed by the client rather than the runtime.
    ///
    /// A remote tool is only *declared* to the LLM (name/description/schema);
    /// when the LLM calls it, [`XmlToolExecutorStage`] emits the call as the
    /// pipeline's response instead of running `execute` and looping. The client
    /// performs it and returns the result as a new inbound message.
    ///
    /// [`XmlToolExecutorStage`]: crate::pipeline::stages::XmlToolExecutorStage
    fn is_remote(&self) -> bool {
        false
    }

    /// Authenticated identity expected to execute this remote tool.
    ///
    /// Manifest-backed tools set this to their publisher. Static remote tools
    /// return `None` and are executed by the authenticated caller.
    fn remote_executor_id(&self) -> Option<&str> {
        None
    }

    /// If this tool is backed by an [`ArtifactStore`](crate::artifacts::ArtifactStore),
    /// expose it so the executor can re-inject loaded bytes without a separate
    /// store injection. Defaults to `None` (most tools have no store).
    #[cfg(feature = "artifacts")]
    fn artifact_store(&self) -> Option<Arc<dyn crate::artifacts::ArtifactStore>> {
        None
    }
}
