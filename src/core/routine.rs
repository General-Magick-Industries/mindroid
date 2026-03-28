use async_trait::async_trait;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::config::AgentConfig;
use crate::error::Result;
use crate::models::Response;
use crate::pipeline::Pipeline;
use super::message::TransportSender;

/// Context passed to a routine's `act()` call each tick.
pub struct RoutineContext {
    pub tick_count: u64,
    pub agent_config: Arc<AgentConfig>,
    pub transport_sender: Arc<TransportSender>,
    pub pipeline: Arc<Pipeline>,
}

impl RoutineContext {
    pub async fn send(&self, response: &Response) -> Result<Option<String>> {
        self.transport_sender.send(response).await
    }
}

/// A proactive timer-driven behaviour attached to a `Runtime`.
///
/// Each tick the runtime calls `poll()` and, if it returns `Some(data)`,
/// calls `act()` with that data. Returning `None` from `poll()` skips `act()`
/// for that tick (useful for cost-saving when there is nothing to act on).
#[async_trait]
pub trait Routine: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn interval(&self) -> std::time::Duration;

    /// Fetch the resource this routine monitors.
    /// Return `None` to skip `act()` this tick.
    async fn poll(&self) -> Result<Option<String>>;

    /// Act on the polled data.
    async fn act(&self, ctx: RoutineContext, data: String) -> Result<()>;
}

pub(crate) type PollFn = Box<dyn Fn() -> Pin<Box<dyn Future<Output = Result<Option<String>>> + Send>> + Send + Sync>;
pub(crate) type ActFn = Box<dyn Fn(RoutineContext, String) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

pub(crate) struct FnRoutine {
    pub(crate) name: String,
    pub(crate) interval: std::time::Duration,
    pub(crate) poll_fn: PollFn,
    pub(crate) act_fn: ActFn,
}

impl FnRoutine {
    pub(crate) fn new<PF, PFut, AF, AFut>(
        name: impl Into<String>,
        interval: std::time::Duration,
        poll_fn: PF,
        act_fn: AF,
    ) -> Self
    where
        PF: Fn() -> PFut + Send + Sync + 'static,
        PFut: Future<Output = Result<Option<String>>> + Send + 'static,
        AF: Fn(RoutineContext, String) -> AFut + Send + Sync + 'static,
        AFut: Future<Output = ()> + Send + 'static,
    {
        Self {
            name: name.into(),
            interval,
            poll_fn: Box::new(move || Box::pin(poll_fn())),
            act_fn: Box::new(move |ctx, data| Box::pin(act_fn(ctx, data))),
        }
    }
}

#[async_trait]
impl Routine for FnRoutine {
    fn name(&self) -> &str { &self.name }
    fn interval(&self) -> std::time::Duration { self.interval }
    async fn poll(&self) -> Result<Option<String>> {
        (self.poll_fn)().await
    }
    async fn act(&self, ctx: RoutineContext, data: String) -> Result<()> {
        (self.act_fn)(ctx, data).await;
        Ok(())
    }
}
