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

    /// Report connection transitions to `reporter` for the life of the transport.
    ///
    /// Called by the runtime before `connect`. A transport that reconnects in
    /// the background must publish [`Health::Reconnecting`] while it retries and
    /// [`Health::Ready`] once it is serving again — otherwise a supervisor
    /// cannot tell a live-but-deaf agent from a working one.
    ///
    /// Defaulted to a no-op so existing implementations keep compiling; the
    /// runtime then reports `Ready` on a successful `connect` and leaves it
    /// there, which is the old behaviour.
    ///
    /// [`Health::Reconnecting`]: crate::Health::Reconnecting
    /// [`Health::Ready`]: crate::Health::Ready
    fn set_health_reporter(&mut self, _reporter: crate::core::health::HealthReporter) {}

    /// Whether this transport publishes its own [`Health`] transitions.
    ///
    /// The runtime reports `Ready` itself once `connect` succeeds, which is only
    /// true for a transport that finishes connecting there. One that connects
    /// lazily — Centrifugo opens its socket in `listen`, not `connect` — must
    /// return `true`, or the runtime announces `Ready` for a transport that has
    /// not reached its endpoint and a supervisor waits on a green light that
    /// never meant anything.
    ///
    /// [`Health`]: crate::Health
    fn reports_own_health(&self) -> bool {
        false
    }
}
