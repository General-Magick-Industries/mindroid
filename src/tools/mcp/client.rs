//! HTTP-based MCP client that speaks JSON-RPC to external MCP servers.

use std::sync::atomic::{AtomicU64, Ordering};

use reqwest::Client;
use serde_json::{Value, json};
use tracing::{debug, warn};

use crate::error::{MindroidError, Result};

use super::protocol::*;

/// MCP client that communicates with an external MCP server over HTTP.
///
/// Supports API key authentication via Bearer token or custom header.
pub struct McpClient {
    server_name: String,
    url: String,
    api_key: Option<String>,
    http: Client,
    request_id: AtomicU64,
    /// Session ID returned by the server (if any).
    session_id: tokio::sync::RwLock<Option<String>>,
}

impl McpClient {
    /// Create a new MCP client.
    ///
    /// - `server_name`: prefix for tool names (e.g. "context7" → "context7_resolve-library-id")
    /// - `url`: the MCP server endpoint URL
    /// - `api_key`: optional API key for Bearer auth
    pub fn new(server_name: &str, url: &str, api_key: Option<&str>) -> Self {
        Self {
            server_name: server_name.to_string(),
            url: url.to_string(),
            api_key: api_key.map(String::from),
            http: Client::new(),
            request_id: AtomicU64::new(1),
            session_id: tokio::sync::RwLock::new(None),
        }
    }

    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Send the `initialize` handshake to the MCP server.
    pub async fn initialize(&self) -> Result<()> {
        let params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            capabilities: ClientCapabilities {},
            client_info: ClientInfo {
                name: "mindroid".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        };

        let resp = self
            .rpc("initialize", Some(serde_json::to_value(&params).unwrap()))
            .await?;

        if let Some(result) = &resp.result {
            let server_info = result
                .get("serverInfo")
                .cloned()
                .unwrap_or(Value::Null);
            debug!(
                server = %self.server_name,
                server_info = %server_info,
                "MCP server initialized"
            );
        }

        // Send initialized notification (no id, no response expected)
        self.notify("notifications/initialized", None).await?;

        Ok(())
    }

    /// Fetch the list of tools from the server.
    pub async fn list_tools(&self) -> Result<Vec<McpTool>> {
        let resp = self.rpc("tools/list", Some(json!({}))).await?;

        let result = resp.result.ok_or_else(|| {
            MindroidError::Api {
                message: format!("MCP server '{}' returned no result for tools/list", self.server_name),
                status_code: None,
            }
        })?;

        let list: ListToolsResult = serde_json::from_value(result).map_err(|e| {
            MindroidError::Api {
                message: format!("Failed to parse tools/list response: {e}"),
                status_code: None,
            }
        })?;

        debug!(
            server = %self.server_name,
            tool_count = list.tools.len(),
            "Discovered MCP tools"
        );

        Ok(list.tools)
    }

    /// Call a tool on the MCP server.
    pub async fn call_tool(&self, tool_name: &str, arguments: Value) -> Result<String> {
        let params = CallToolParams {
            name: tool_name.to_string(),
            arguments,
        };

        let resp = self
            .rpc(
                "tools/call",
                Some(serde_json::to_value(&params).unwrap()),
            )
            .await?;

        let result = resp.result.ok_or_else(|| {
            MindroidError::Api {
                message: format!(
                    "MCP server '{}' returned no result for tools/call '{tool_name}'",
                    self.server_name
                ),
                status_code: None,
            }
        })?;

        let call_result: CallToolResult = serde_json::from_value(result).map_err(|e| {
            MindroidError::Api {
                message: format!("Failed to parse tools/call response: {e}"),
                status_code: None,
            }
        })?;

        if call_result.is_error {
            let text = extract_text(&call_result.content);
            return Err(MindroidError::Api {
                message: format!("MCP tool '{tool_name}' returned error: {text}"),
                status_code: None,
            });
        }

        Ok(extract_text(&call_result.content))
    }

    // -- Internal helpers -----------------------------------------------------

    fn next_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::Relaxed)
    }

    async fn rpc(&self, method: &str, params: Option<Value>) -> Result<JsonRpcResponse> {
        let id = self.next_id();
        let request = JsonRpcRequest::new(id, method, params);

        let mut req_builder = self
            .http
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");

        if let Some(ref key) = self.api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        // Attach session ID if we have one.
        if let Some(ref sid) = *self.session_id.read().await {
            req_builder = req_builder.header("Mcp-Session-Id", sid.as_str());
        }

        let http_resp = req_builder
            .json(&request)
            .send()
            .await
            .map_err(|e| MindroidError::Api {
                message: format!("MCP HTTP request to '{}' failed: {e}", self.server_name),
                status_code: e.status().map(|s| s.as_u16()),
            })?;

        // Capture session ID from response headers.
        if let Some(sid) = http_resp.headers().get("mcp-session-id") {
            if let Ok(val) = sid.to_str() {
                *self.session_id.write().await = Some(val.to_string());
            }
        }

        let status = http_resp.status();
        if !status.is_success() {
            let body = http_resp.text().await.unwrap_or_default();
            return Err(MindroidError::Api {
                message: format!(
                    "MCP server '{}' returned HTTP {status}: {body}",
                    self.server_name
                ),
                status_code: Some(status.as_u16()),
            });
        }

        let body = http_resp.text().await.map_err(|e| MindroidError::Api {
            message: format!("Failed to read MCP response body: {e}"),
            status_code: None,
        })?;

        // Handle SSE-wrapped responses: some MCP servers return event-stream
        // with "event: message\ndata: {...}\n\n" format.
        let json_str = parse_sse_or_json(&body);

        serde_json::from_str(json_str).map_err(|e| MindroidError::Api {
            message: format!("Failed to parse MCP JSON-RPC response: {e}\nBody: {json_str}"),
            status_code: None,
        })
    }

    /// Send a notification (no id field, no response expected).
    async fn notify(&self, method: &str, params: Option<Value>) -> Result<()> {
        let body = if let Some(p) = params {
            json!({ "jsonrpc": "2.0", "method": method, "params": p })
        } else {
            json!({ "jsonrpc": "2.0", "method": method })
        };

        let mut req_builder = self
            .http
            .post(&self.url)
            .header("Content-Type", "application/json");

        if let Some(ref key) = self.api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        if let Some(ref sid) = *self.session_id.read().await {
            req_builder = req_builder.header("Mcp-Session-Id", sid.as_str());
        }

        let resp = req_builder.json(&body).send().await;
        if let Err(e) = resp {
            warn!(server = %self.server_name, "MCP notification '{method}' failed: {e}");
        }

        Ok(())
    }
}

/// Extract text content from MCP tool result content blocks.
fn extract_text(content: &[ToolContent]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            ToolContent::Text { text } => Some(text.as_str()),
            ToolContent::Other => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Some MCP servers wrap JSON-RPC in SSE. Extract the last `data:` line
/// that contains a complete JSON object, or return the original string.
fn parse_sse_or_json(body: &str) -> &str {
    let trimmed = body.trim();
    // If it starts with '{' or '[', it's plain JSON.
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return trimmed;
    }

    // SSE format: look for the last "data: {...}" line.
    for line in trimmed.lines().rev() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if data.starts_with('{') {
                return data;
            }
        }
    }

    trimmed
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpClient")
            .field("server_name", &self.server_name)
            .field("url", &self.url)
            .field("has_api_key", &self.api_key.is_some())
            .finish()
    }
}
