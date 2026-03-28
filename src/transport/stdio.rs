// Stdio transport for Mindroid

use async_trait::async_trait;

use crate::runtime::TransportSend;
use crate::{ChannelType, Message, MessageType, Response, Result, SenderType, Transport};
use chrono::Utc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use uuid::Uuid;

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
                let message = Message {
                    id: Uuid::new_v4().to_string(),
                    content: line,
                    sender_id: "stdin".to_string(),
                    sender_type: SenderType::User,
                    channel_id: "stdio".to_string(),
                    channel_type: ChannelType::Direct,
                    message_type: MessageType::Text,
                    timestamp: Utc::now(),
                    metadata: Default::default(),
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
