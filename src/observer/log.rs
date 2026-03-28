use async_trait::async_trait;

use crate::{MindroidError, Message, Observer, StreamEvent};

pub struct LogObserver;

impl LogObserver {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LogObserver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Observer for LogObserver {
    async fn on_start(&self) {
        tracing::info!("Agent started");
    }

    async fn on_shutdown(&self) {
        tracing::info!("Agent shutting down");
    }

    async fn on_message_received(&self, msg: &Message) {
        tracing::info!(
            id = %msg.id,
            sender = %msg.sender_id,
            channel = %msg.channel_id,
            "Message received"
        );
    }

    async fn on_response_sent(&self, channel: &str, content: &str) {
        tracing::info!(channel = %channel, len = content.len(), "Response sent");
    }

    async fn on_stream_event(&self, event: &StreamEvent) {
        tracing::debug!(?event, "Stream event");
    }

    async fn on_error(&self, error: &MindroidError) {
        tracing::error!(%error, "Error occurred");
    }
}
