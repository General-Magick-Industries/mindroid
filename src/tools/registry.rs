use std::fmt;
use std::sync::Arc;

use super::Tool;

/// Registry of tools available to the agent.
///
/// Pass to [`ToolExecutorStage`](crate::pipeline::stages::ToolExecutorStage) after building:
///
/// ```ignore
/// let registry = ToolRegistry::new()
///     .register(ShellTool::default())
///     .register(OpenTool);
/// let stage = ToolExecutorStage::new(client, Arc::new(registry));
/// ```
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Register a tool. Consumes and returns `self` for chaining.
    pub fn register(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Build the system prompt fragment that describes all registered tools to the LLM.
    pub fn system_prompt(&self) -> String {
        if self.tools.is_empty() {
            return String::new();
        }

        let mut out = String::from(
            "You can control this computer using tools.\n\
             To use a tool, output EXACTLY this on its own line:\n\n\
             <tool_call>{\"name\": \"tool_name\", \"args\": {\"param\": \"value\"}}</tool_call>\n\n\
             You will receive a <tool_result> back, then can call another tool or give your final answer.\n\
             Keep shell commands short and direct — avoid complex arithmetic or long pipelines in one command; use separate tool calls instead.\n\n\
             Available tools:\n",
        );

        for tool in &self.tools {
            let schema = tool.parameters_schema();
            let params = schema
                .get("properties")
                .and_then(|p| p.as_object())
                .map(|props| {
                    props
                        .iter()
                        .map(|(k, v)| {
                            let desc = v
                                .get("description")
                                .and_then(|d| d.as_str())
                                .unwrap_or("");
                            format!("  - {k}: {desc}")
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();

            out.push_str(&format!(
                "\n**{}** — {}\n{}\n",
                tool.name(),
                tool.description(),
                params
            ));
        }

        out
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &format!("{} tools", self.tools.len()))
            .finish()
    }
}

/// Build a registry from config, enabling only the tools that are configured on.
impl ToolRegistry {
    pub fn from_config(config: &crate::config::ToolsConfig) -> Self {
        let mut registry = Self::new();
        if config.shell.enabled {
            registry = registry.register(super::ShellTool::with_config(
                config.shell.timeout_secs,
                config.shell.instructions.clone(),
                config.shell.clone(),
            ));
        }
        if config.open.enabled {
            registry = registry.register(super::OpenTool::with_config(config.open.clone()));
        }
        registry
    }
}
