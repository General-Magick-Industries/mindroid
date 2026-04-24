//! MCP JSON-RPC protocol types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC 2.0 request envelope.
#[derive(Debug, Serialize)]
pub(crate) struct JsonRpcRequest {
    pub jsonrpc: &'static str,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC 2.0 response envelope.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct JsonRpcResponse {
    pub id: Option<u64>,
    pub result: Option<Value>,
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error object.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

// -- MCP-specific types -------------------------------------------------------

/// `initialize` request params.
#[derive(Debug, Serialize)]
pub(crate) struct InitializeParams {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ClientCapabilities,
    #[serde(rename = "clientInfo")]
    pub client_info: ClientInfo,
}

#[derive(Debug, Serialize)]
pub(crate) struct ClientCapabilities {}

#[derive(Debug, Serialize)]
pub(crate) struct ClientInfo {
    pub name: String,
    pub version: String,
}

/// A tool advertised by the MCP server.
#[derive(Debug, Clone, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "inputSchema", default)]
    pub input_schema: Option<McpToolSchema>,
}

/// JSON Schema for a tool's input parameters.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpToolSchema {
    #[serde(rename = "type", default)]
    pub schema_type: Option<String>,
    #[serde(default)]
    pub properties: Option<Value>,
    #[serde(default)]
    pub required: Option<Vec<String>>,
    /// Catch-all for additional schema fields (additionalProperties, etc.)
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, Value>,
}

impl McpToolSchema {
    /// Convert to a JSON Value matching the format expected by `Tool::parameters_schema()`.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Object(Default::default()))
    }
}

/// Response from `tools/list`.
#[derive(Debug, Deserialize)]
pub(crate) struct ListToolsResult {
    pub tools: Vec<McpTool>,
}

/// `tools/call` request params.
#[derive(Debug, Serialize)]
pub(crate) struct CallToolParams {
    pub name: String,
    pub arguments: Value,
}

/// Response from `tools/call`.
#[derive(Debug, Deserialize)]
pub(crate) struct CallToolResult {
    pub content: Vec<ToolContent>,
    #[serde(rename = "isError", default)]
    pub is_error: bool,
}

/// A single content block in a tool result.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum ToolContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Other,
}
