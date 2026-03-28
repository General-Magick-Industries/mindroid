mod registry;
pub mod shell;
pub mod open;
pub mod reminder;

pub use registry::ToolRegistry;
pub use shell::ShellTool;
pub use open::OpenTool;
pub use reminder::{SetReminderTool, ReminderRoutine, ReminderStore, new_reminder_store};

use async_trait::async_trait;

use serde_json::Value;

use crate::error::Result;

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
///     async fn execute(&self, args: Value) -> mindroid::Result<String> {
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

    /// Execute the tool with parsed arguments. Returns output as a plain string.
    async fn execute(&self, args: Value) -> Result<String>;
}
