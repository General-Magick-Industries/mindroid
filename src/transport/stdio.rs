// Stdio transport for Mindroid

use async_trait::async_trait;

use crate::runtime::TransportSend;
use crate::tools::remote;
use crate::{ChannelType, Message, MessageType, Response, Result, SenderType, Transport};
use chrono::Utc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use uuid::Uuid;

/// The control-traffic type a stdin line declares, if any.
///
/// The stages dispatch on a `message_type` the sender declares, which a chat
/// backend stamps out of band. Stdin has no such channel, so the envelope's own
/// `type` field is the declaration — sound here because the only writer is the
/// local operator, not a third party whose quoted JSON could be mistaken for
/// control traffic.
fn declared_message_type(line: &str) -> Option<&'static str> {
    let envelope: serde_json::Value = serde_json::from_str(line).ok()?;
    match envelope.get("type")?.as_str()? {
        "tools_manifest" => Some(remote::TOOL_MANIFEST_MESSAGE_TYPE),
        "tool_result" => Some(remote::TOOL_RESULT_MESSAGE_TYPE),
        _ => None,
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
                let declared = declared_message_type(&line);
                let content = if declared == Some(remote::TOOL_RESULT_MESSAGE_TYPE) {
                    remote::normalize_tool_result(&line).unwrap_or(line)
                } else {
                    line
                };
                let mut metadata = std::collections::HashMap::new();
                if let Some(declared) = declared {
                    metadata.insert(
                        "message_type".to_string(),
                        serde_json::Value::String(declared.to_string()),
                    );
                }
                let message = Message {
                    id: Uuid::new_v4().to_string(),
                    content,
                    sender_id: "stdin".to_string(),
                    sender_type: SenderType::User,
                    channel_id: "stdio".to_string(),
                    channel_type: ChannelType::Direct,
                    message_type: MessageType::Text,
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

    #[test]
    fn envelopes_declare_their_control_type() {
        assert_eq!(
            declared_message_type(r#"{"type":"tools_manifest","payload":{"tools":[]}}"#),
            Some(remote::TOOL_MANIFEST_MESSAGE_TYPE)
        );
        assert_eq!(
            declared_message_type(r#"{"type":"tool_result","payload":{"name":"peek"}}"#),
            Some(remote::TOOL_RESULT_MESSAGE_TYPE)
        );
    }

    #[test]
    fn ordinary_lines_declare_nothing() {
        for line in [
            "what's the time",
            r#"{"type":"chat","content":"hi"}"#,
            r#"{"content":"hi"}"#,
            "{not json",
            "",
        ] {
            assert_eq!(declared_message_type(line), None, "{line}");
        }
    }
}
