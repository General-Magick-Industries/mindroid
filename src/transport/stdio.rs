// Stdio transport for Mindroid

use async_trait::async_trait;

use std::collections::HashMap;

use crate::core::models::{CONTEXT_METADATA_KEY, TOOLS_METADATA_KEY};
use crate::runtime::TransportSend;
use crate::tools::remote;
use crate::{ChannelType, Message, MessageType, Response, Result, SenderType, Transport};
use chrono::Utc;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use uuid::Uuid;

/// One stdin line resolved into the same shape a metadata-stamping transport
/// delivers, so the stages downstream never learn which transport they are on.
struct Declared {
    message_type: MessageType,
    content: String,
    metadata: HashMap<String, Value>,
}

/// Resolve a stdin line into its declared type, body and metadata.
///
/// Sniffing the body is sound HERE and nowhere else: stdin has no out-of-band
/// channel to carry a declaration, and its only writer is the local operator —
/// not a third party on a shared channel whose quoted JSON could be mistaken for
/// control traffic. A line that is not a recognized envelope is plain text.
fn declare(line: String) -> Declared {
    let plain = |line: String| Declared {
        message_type: MessageType::Text,
        content: line,
        metadata: HashMap::new(),
    };

    let Ok(envelope) = serde_json::from_str::<Value>(&line) else {
        return plain(line);
    };
    let declared = envelope
        .get("type")
        .and_then(Value::as_str)
        .and_then(MessageType::from_wire);

    let mut metadata = HashMap::new();
    // A manifest envelope nests its tools under `payload`; a chat line carries
    // per-turn tools and context at the top level. Both land on the same keys
    // the Centrifugo fan-out uses, which is what lets the stages read one place.
    let tools = envelope
        .get("payload")
        .and_then(|p| p.get(TOOLS_METADATA_KEY))
        .or_else(|| envelope.get(TOOLS_METADATA_KEY));
    if let Some(tools) = tools.filter(|v| !v.is_null()) {
        metadata.insert(TOOLS_METADATA_KEY.to_string(), tools.clone());
    }
    if let Some(context) = envelope.get(CONTEXT_METADATA_KEY).filter(|v| !v.is_null()) {
        metadata.insert(CONTEXT_METADATA_KEY.to_string(), context.clone());
    }

    let Some(message_type) = declared else {
        return Declared {
            message_type: MessageType::Text,
            content: envelope
                .get("content")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or(line),
            metadata,
        };
    };

    let content = if message_type == MessageType::ToolResult {
        remote::normalize_tool_result(&line).unwrap_or(line)
    } else {
        line
    };
    Declared {
        message_type,
        content,
        metadata,
    }
}

pub struct StdioTransport {
    connected: bool,
}

impl StdioTransport {
    pub fn new() -> Self {
        Self { connected: false }
    }
}

impl Default for StdioTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Transport for StdioTransport {
    fn name(&self) -> &str {
        "stdio"
    }

    async fn connect(&mut self) -> Result<()> {
        self.connected = true;
        tracing::info!("StdioTransport connected");
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        self.connected = false;
        tracing::info!("StdioTransport disconnected");
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    async fn listen(&self, tx: mpsc::Sender<Message>) -> Result<()> {
        tracing::info!("StdioTransport listening on stdin");
        tokio::spawn(async move {
            let stdin = tokio::io::stdin();
            let reader = BufReader::new(stdin);
            let mut lines = reader.lines();

            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!("StdioTransport received line: {}", line);
                let Declared {
                    message_type,
                    content,
                    metadata,
                } = declare(line);
                let message = Message {
                    id: Uuid::new_v4().to_string(),
                    content,
                    sender_id: "stdin".to_string(),
                    sender_type: SenderType::User,
                    channel_id: "stdio".to_string(),
                    channel_type: ChannelType::Direct,
                    message_type,
                    timestamp: Utc::now(),
                    metadata,
                    platform: Some("stdio".into()),
                };
                if tx.send(message).await.is_err() {
                    tracing::warn!("StdioTransport: receiver dropped, stopping stdin listener");
                    break;
                }
            }
            tracing::info!("StdioTransport stdin closed (EOF)");
        });
        Ok(())
    }

    async fn send(&self, response: &Response) -> Result<Option<String>> {
        println!("{}", response.content);
        Ok(None)
    }

    async fn send_typing(&self, _channel_id: &str) -> Result<()> {
        Ok(())
    }
}

/// A `TransportSend` implementation that writes to stdout.
///
/// Use this with `RuntimeBuilder::transport_sender()` to enable routines
/// and other components to send messages through stdio.
pub struct StdioSender;

#[async_trait]
impl TransportSend for StdioSender {
    async fn send(&self, response: &Response) -> Result<Option<String>> {
        println!("{}", response.content);
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_plain_line_is_a_text_turn() {
        let d = declare("hello there".to_string());
        assert_eq!(d.message_type, MessageType::Text);
        assert_eq!(d.content, "hello there");
        assert!(d.metadata.is_empty());
    }

    /// Malformed JSON is text, not an error: stdin is a human-facing channel.
    #[test]
    fn a_broken_envelope_is_text() {
        let d = declare(r#"{"broken"#.to_string());
        assert_eq!(d.message_type, MessageType::Text);
        assert_eq!(d.content, r#"{"broken"#);
    }

    /// Stdio has no out-of-band channel, so its envelope IS the declaration —
    /// but it lands on the same metadata keys the Centrifugo fan-out uses, which
    /// is what lets the stages read one place regardless of transport.
    #[test]
    fn a_manifest_envelope_becomes_declared_type_plus_metadata() {
        let d = declare(
            r#"{"type":"tools_manifest","payload":{"tools":[{"name":"peek"}]}}"#.to_string(),
        );
        assert_eq!(d.message_type, MessageType::ToolManifest);
        assert_eq!(d.metadata[TOOLS_METADATA_KEY][0]["name"], "peek");
    }

    /// The flat shape matches what the Centrifugo fan-out sends, so a hand-typed
    /// manifest works whether or not the operator nests it under `payload`.
    #[test]
    fn a_manifest_is_accepted_flat_or_nested() {
        for line in [
            r#"{"type":"tools_manifest","payload":{"tools":[{"name":"peek"}]}}"#,
            r#"{"type":"tool_manifest","tools":[{"name":"peek"}]}"#,
        ] {
            let d = declare(line.to_string());
            assert_eq!(d.message_type, MessageType::ToolManifest);
            assert_eq!(
                d.metadata[TOOLS_METADATA_KEY][0]["name"], "peek",
                "for {line}"
            );
        }
    }

    #[test]
    fn a_chat_line_carries_per_turn_tools_and_context() {
        let d = declare(
            json!({
                "content": "whats the time",
                "tools": [{ "name": "peek" }],
                "context": { "page": "/spaces/1" },
            })
            .to_string(),
        );
        assert_eq!(d.message_type, MessageType::Text);
        assert_eq!(d.content, "whats the time");
        assert_eq!(d.metadata[TOOLS_METADATA_KEY][0]["name"], "peek");
        assert_eq!(d.metadata[CONTEXT_METADATA_KEY]["page"], "/spaces/1");
    }

    #[test]
    fn a_tool_result_envelope_is_framed_for_history() {
        let d = declare(
            r#"{"type":"tool_result","payload":{"name":"get_time","content":"3pm"}}"#.to_string(),
        );
        assert_eq!(d.message_type, MessageType::ToolResult);
        assert_eq!(
            d.content,
            "<tool_result name=\"get_time\">3pm</tool_result>"
        );
    }
}
