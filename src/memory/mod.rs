#[cfg(feature = "persistence")]
pub mod magickmind;
pub mod sqlite;
pub mod markdown;

use async_trait::async_trait;

use crate::error::Result;
use crate::models::Message;

#[async_trait]
pub trait Memory: Send + Sync + 'static {
    async fn save_message(
        &self,
        channel_id: &str,
        sender_id: &str,
        content: &str,
        reply_to_id: Option<&str>,
    ) -> Result<Option<String>>;

    async fn get_history(&self, channel_id: &str, limit: usize) -> Result<Vec<Message>>;

    async fn clear_history(&self, channel_id: &str) -> Result<()>;
}

/// No-op memory implementation. Does not store anything.
pub struct NoMemory;

#[async_trait]
impl Memory for NoMemory {
    async fn save_message(
        &self,
        _channel_id: &str,
        _sender_id: &str,
        _content: &str,
        _reply_to_id: Option<&str>,
    ) -> Result<Option<String>> {
        Ok(None)
    }

    async fn get_history(&self, _channel_id: &str, _limit: usize) -> Result<Vec<Message>> {
        Ok(Vec::new())
    }

    async fn clear_history(&self, _channel_id: &str) -> Result<()> {
        Ok(())
    }
}
