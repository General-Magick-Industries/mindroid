#[cfg(feature = "transport-audio")]
pub mod audio;
#[cfg(feature = "transport-ws")]
pub mod centrifugo;
pub mod stdio;

use async_trait::async_trait;

use tokio::sync::mpsc;

use crate::error::Result;
use crate::models::{Message, Response};

#[async_trait]
pub trait Transport: Send + Sync + 'static {
    fn name(&self) -> &str;

    async fn connect(&mut self) -> Result<()>;

    async fn disconnect(&mut self) -> Result<()>;

    async fn listen(&self, tx: mpsc::Sender<Message>) -> Result<()>;

    async fn send(&self, response: &Response) -> Result<Option<String>>;

    fn is_connected(&self) -> bool;

    async fn send_typing(&self, _channel_id: &str) -> Result<()> {
        Ok(())
    }

    async fn health_check(&self) -> bool {
        self.is_connected()
    }
}
