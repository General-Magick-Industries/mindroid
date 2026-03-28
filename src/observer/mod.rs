pub mod log;

use async_trait::async_trait;

use crate::error::MindroidError;
use crate::models::{Message, StreamEvent};

#[async_trait]
pub trait Observer: Send + Sync + 'static {
    async fn on_start(&self) {}
    async fn on_shutdown(&self) {}
    async fn on_message_received(&self, _msg: &Message) {}
    async fn on_response_sent(&self, _channel: &str, _content: &str) {}
    async fn on_stream_event(&self, _event: &StreamEvent) {}
    async fn on_error(&self, _error: &MindroidError) {}
}

/// No-op observer implementation.
pub struct NoObserver;

#[async_trait]
impl Observer for NoObserver {
    // All default implementations are no-ops
}
