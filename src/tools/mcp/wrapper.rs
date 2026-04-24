//! Wrapper that adapts an MCP remote tool to the local `Tool` trait.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;
use crate::tools::Tool;

use super::client::McpClient;
use super::protocol::McpTool;

/// Wraps a remote MCP tool so it implements the local [`Tool`] trait.
///
/// When `execute()` is called, the wrapper sends a `tools/call` JSON-RPC
/// request to the MCP server and returns the text result.
pub struct McpToolWrapper {
    /// Prefixed name: "{server}_{tool}" (e.g. "context7_query-docs").
    prefixed_name: String,
    /// The original MCP tool metadata.
    tool: McpTool,
    /// Shared client for the server this tool belongs to.
    client: Arc<McpClient>,
}

impl McpToolWrapper {
    pub fn new(prefixed_name: String, tool: McpTool, client: Arc<McpClient>) -> Self {
        Self {
            prefixed_name,
            tool,
            client,
        }
    }

    /// The original (un-prefixed) tool name on the MCP server.
    pub fn remote_name(&self) -> &str {
        &self.tool.name
    }
}

#[async_trait]
impl Tool for McpToolWrapper {
    fn name(&self) -> &str {
        &self.prefixed_name
    }

    fn description(&self) -> &str {
        self.tool
            .description
            .as_deref()
            .unwrap_or("MCP remote tool")
    }

    fn parameters_schema(&self) -> Value {
        self.tool
            .input_schema
            .as_ref()
            .map(|s| s.to_json())
            .unwrap_or_else(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": {}
                })
            })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        self.client.call_tool(&self.tool.name, args).await
    }
}

impl std::fmt::Debug for McpToolWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpToolWrapper")
            .field("name", &self.prefixed_name)
            .field("remote_name", &self.tool.name)
            .field("server", &self.client.server_name())
            .finish()
    }
}
