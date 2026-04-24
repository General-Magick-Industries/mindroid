//! MCP (Model Context Protocol) client for connecting to external tool servers.
//!
//! Allows Mindroid agents to use tools provided by any MCP-compatible server
//! (e.g. Context7, Notion, GitHub) via JSON-RPC over HTTP with API key auth.
//!
//! # Example
//!
//! ```toml
//! [[tools.mcp_servers]]
//! name = "context7"
//! url = "https://mcp.context7.com/mcp"
//! api_key_env = "CONTEXT7_API_KEY"
//! ```

mod client;
mod protocol;
mod wrapper;

pub use client::McpClient;
pub use protocol::{McpTool, McpToolSchema};
pub use wrapper::McpToolWrapper;

use crate::error::Result;

/// Connect to an MCP server and return tool wrappers ready for registration.
pub async fn load_mcp_tools(
    server_name: &str,
    url: &str,
    api_key: Option<&str>,
) -> Result<Vec<McpToolWrapper>> {
    let client = std::sync::Arc::new(McpClient::new(server_name, url, api_key));
    client.initialize().await?;
    let tools = client.list_tools().await?;

    Ok(tools
        .into_iter()
        .map(|tool| {
            let prefixed_name = format!("{}_{}", server_name, tool.name);
            McpToolWrapper::new(prefixed_name, tool, client.clone())
        })
        .collect())
}
