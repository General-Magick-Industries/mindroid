use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::config::AgentConfig;
use crate::error::Result;
use crate::models::Message;
use crate::observer::Observer;
use crate::pipeline::Pipeline;
use crate::transport::Transport;

// Re-export from sibling modules so lib.rs needn't change
pub use super::builder::RuntimeBuilder;
pub use super::message::{MessageContext, TransportSend, TransportSender};
pub use super::routine::{Routine, RoutineContext};

use super::message::MessageHandler;

/// The Mindroid agent runtime. Connects transport, pipeline, identity,
/// memory, and observers into a running agent.
pub struct Runtime {
    pub(crate) transport: Box<dyn Transport>,
    pub(crate) pipeline: Arc<Pipeline>,
    pub(crate) observers: Arc<Vec<Box<dyn Observer>>>,
    pub(crate) agent_config: Arc<AgentConfig>,
    pub(crate) handler: MessageHandler,
    pub(crate) transport_sender: Arc<TransportSender>,
    pub(crate) channel_buffer: usize,
    pub(crate) routines: Vec<Box<dyn Routine>>,
    pub(crate) routine_handles: Vec<(CancellationToken, tokio::task::JoinHandle<()>)>,
    pub(crate) coordinator: Arc<crate::core::coordinator::PerKey<String>>,
}

impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::new()
    }

    /// Start the runtime main loop. Blocks until the transport closes.
    pub async fn run(&mut self) -> Result<()> {
        self.run_inner(None).await
    }

    /// Like [`run`](Self::run), but also stops when `cancel` is cancelled —
    /// breaking the message loop and running [`shutdown`](Self::shutdown) for a
    /// clean exit (transport disconnected, routines cancelled).
    ///
    /// For a control plane that activates/deactivates agents: spawn the runtime
    /// with a per-agent [`CancellationToken`], and cancel it to stop that agent
    /// gracefully (vs. `handle.abort()`, which drops mid-await with no clean
    /// disconnect).
    pub async fn run_until_cancelled(&mut self, cancel: CancellationToken) -> Result<()> {
        self.run_inner(Some(cancel)).await
    }

    async fn run_inner(&mut self, cancel: Option<CancellationToken>) -> Result<()> {
        // Connect transport
        self.transport.connect().await?;
        tracing::info!("Transport '{}' connected", self.transport.name());

        // Notify observers
        for obs in self.observers.iter() {
            obs.on_start().await;
        }

        // Create message channel
        let (tx, mut rx) = mpsc::channel::<Message>(self.channel_buffer);

        // Start transport listening — feeds messages into tx
        self.transport.listen(tx).await?;
        tracing::info!("Listening for messages...");

        // Spawn routine tasks
        let routines = std::mem::take(&mut self.routines);
        for routine in routines {
            let interval = routine.interval();
            let agent_config = self.agent_config.clone();
            let transport_sender = self.transport_sender.clone();
            let pipeline = self.pipeline.clone();

            let token = CancellationToken::new();
            let child_token = token.child_token();

            let handle = tokio::spawn(async move {
                let mut ticker = tokio::time::interval(interval);
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                let mut tick_count: u64 = 0;

                loop {
                    tokio::select! {
                        _ = child_token.cancelled() => break,
                        _ = ticker.tick() => {}
                    }
                    tick_count += 1;

                    // Skip first immediate tick
                    if tick_count == 1 {
                        continue;
                    }

                    let name = routine.name();
                    tracing::debug!(routine = name, tick = tick_count, "Routine tick");

                    match routine.poll().await {
                        Ok(Some(data)) => {
                            let ctx = RoutineContext {
                                tick_count,
                                agent_config: agent_config.clone(),
                                transport_sender: transport_sender.clone(),
                                pipeline: pipeline.clone(),
                            };
                            if let Err(e) = routine.act(ctx, data).await {
                                tracing::error!(routine = name, tick = tick_count, error = %e, "Routine act error");
                            }
                        }
                        Ok(None) => {
                            tracing::debug!(
                                routine = name,
                                tick = tick_count,
                                "Routine poll returned None, skipping"
                            );
                        }
                        Err(e) => {
                            tracing::error!(routine = name, tick = tick_count, error = %e, "Routine poll error");
                        }
                    }
                }
            });

            self.routine_handles.push((token, handle));
        }

        // Process messages
        let handler_cancel = CancellationToken::new();
        let mut handler_tasks = JoinSet::new();
        let mut seen_ids = std::collections::HashSet::<String>::new();
        const MAX_SEEN_IDS: usize = 10_000;
        // A future that never resolves when there's no cancel token, so the
        // select! reduces to plain rx.recv() in the uncancellable `run()` case.
        let cancelled = async {
            match &cancel {
                Some(c) => c.cancelled().await,
                None => std::future::pending().await,
            }
        };
        tokio::pin!(cancelled);
        loop {
            let msg = tokio::select! {
                biased;
                _ = &mut cancelled => {
                    tracing::info!("Runtime cancelled — shutting down");
                    break;
                }
                Some(result) = handler_tasks.join_next(), if !handler_tasks.is_empty() => {
                    if let Err(error) = result {
                        tracing::warn!(%error, "Message handler task failed");
                    }
                    continue;
                }
                msg = rx.recv() => match msg {
                    Some(m) => m,
                    None => break, // transport closed
                },
            };
            tracing::debug!("Received message: {} from {}", msg.id, msg.sender_id);

            // Skip own messages to prevent infinite loops
            if msg.sender_id == self.agent_config.agent_id {
                tracing::debug!("Skipping own message from agent '{}'", msg.sender_id);
                continue;
            }

            // Skip duplicate messages
            if seen_ids.contains(&msg.id) {
                tracing::debug!("Skipping duplicate message: {}", msg.id);
                continue;
            }
            if seen_ids.len() >= MAX_SEEN_IDS {
                seen_ids.clear();
            }
            seen_ids.insert(msg.id.clone());

            for obs in self.observers.iter() {
                obs.on_message_received(&msg).await;
            }

            // Acquire a coordinator permit (strategy-aware)
            let permit = match self.coordinator.acquire(&msg.sender_id).await {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("Skipping message {}: {}", msg.id, e);
                    continue;
                }
            };

            let message_cancel = CancellationToken::new();
            let ctx = MessageContext {
                message: Arc::new(msg),
                agent_config: self.agent_config.clone(),
                pipeline: self.pipeline.clone(),
                transport: self.transport_sender.clone(),
                observers: self.observers.clone(),
                cancel: Some(message_cancel.child_token()),
            };

            let mut handler_fut = (self.handler)(ctx);
            let permit_cancel = permit.token().clone();
            let runtime_cancel = handler_cancel.child_token();
            handler_tasks.spawn(async move {
                let _permit = permit; // drop guard signals completion for Sequential
                tokio::select! {
                    () = &mut handler_fut => return,
                    () = permit_cancel.cancelled() => message_cancel.cancel(),
                    () = runtime_cancel.cancelled() => message_cancel.cancel(),
                }
                handler_fut.await;
            });
        }

        handler_cancel.cancel();
        self.coordinator.close_all();
        if tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(result) = handler_tasks.join_next().await {
                if let Err(error) = result {
                    tracing::warn!(%error, "Message handler task failed during shutdown");
                }
            }
        })
        .await
        .is_err()
        {
            tracing::warn!("Message handlers did not stop within timeout; aborting");
            handler_tasks.abort_all();
            while handler_tasks.join_next().await.is_some() {}
        }

        self.shutdown().await
    }

    /// Gracefully shut down the runtime.
    pub async fn shutdown(&mut self) -> Result<()> {
        for (token, mut handle) in self.routine_handles.drain(..) {
            token.cancel();
            let timeout = tokio::time::timeout(std::time::Duration::from_secs(5), &mut handle);
            match timeout.await {
                Ok(_) => {}
                Err(_) => {
                    tracing::warn!("Routine did not stop within timeout; aborting");
                    handle.abort();
                    let _ = handle.await;
                }
            }
        }

        // Close coordinator before transport disconnect
        self.coordinator.close_all();

        tracing::info!("Shutting down...");
        for obs in self.observers.iter() {
            obs.on_shutdown().await;
        }
        self.transport.disconnect().await?;
        tracing::info!("Shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Response;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Transport that parks the sender instead of dropping it, so `rx.recv()`
    /// genuinely pends and the loop can only exit via cancellation. Dropping the
    /// transport (or clearing `held`) is what closes the channel.
    struct SilentTransport {
        hold_sender: bool,
        held: Mutex<Option<mpsc::Sender<Message>>>,
        disconnected: Arc<AtomicBool>,
    }

    struct OneMessageTransport {
        disconnected: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl Transport for OneMessageTransport {
        fn name(&self) -> &str {
            "one-message"
        }
        async fn connect(&mut self) -> Result<()> {
            Ok(())
        }
        async fn disconnect(&mut self) -> Result<()> {
            self.disconnected.store(true, Ordering::SeqCst);
            Ok(())
        }
        async fn listen(&self, tx: mpsc::Sender<Message>) -> Result<()> {
            tx.send(Message::new("hello", "user", "channel"))
                .await
                .unwrap();
            Ok(())
        }
        async fn send(&self, _response: &Response) -> Result<Option<String>> {
            Ok(None)
        }
        fn is_connected(&self) -> bool {
            true
        }
    }

    impl SilentTransport {
        fn new(hold_sender: bool, disconnected: Arc<AtomicBool>) -> Self {
            Self {
                hold_sender,
                held: Mutex::new(None),
                disconnected,
            }
        }
    }

    #[async_trait::async_trait]
    impl Transport for SilentTransport {
        fn name(&self) -> &str {
            "silent"
        }
        async fn connect(&mut self) -> Result<()> {
            Ok(())
        }
        async fn disconnect(&mut self) -> Result<()> {
            self.disconnected.store(true, Ordering::SeqCst);
            Ok(())
        }
        async fn listen(&self, tx: mpsc::Sender<Message>) -> Result<()> {
            if self.hold_sender {
                *self.held.lock().unwrap() = Some(tx);
            }
            Ok(())
        }
        async fn send(&self, _response: &Response) -> Result<Option<String>> {
            Ok(None)
        }
        fn is_connected(&self) -> bool {
            true
        }
    }

    fn runtime_with(hold_sender: bool, disconnected: Arc<AtomicBool>) -> Runtime {
        Runtime::builder()
            .transport(SilentTransport::new(hold_sender, disconnected))
            .build()
            .expect("builder should produce a runtime")
    }

    #[tokio::test]
    async fn cancelling_breaks_the_message_loop_and_shuts_down() {
        let disconnected = Arc::new(AtomicBool::new(false));
        let mut rt = runtime_with(true, disconnected.clone());
        let cancel = CancellationToken::new();

        let token = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            token.cancel();
        });

        let result = tokio::time::timeout(Duration::from_secs(5), rt.run_until_cancelled(cancel))
            .await
            .expect("cancellation must break the loop, not hang");

        assert!(result.is_ok());
        assert!(
            disconnected.load(Ordering::SeqCst),
            "shutdown must disconnect the transport"
        );
    }

    #[tokio::test]
    async fn an_already_cancelled_token_exits_without_processing() {
        let disconnected = Arc::new(AtomicBool::new(false));
        let mut rt = runtime_with(true, disconnected.clone());
        let cancel = CancellationToken::new();
        cancel.cancel();

        tokio::time::timeout(Duration::from_secs(5), rt.run_until_cancelled(cancel))
            .await
            .expect("a pre-cancelled token must exit promptly")
            .expect("clean exit");

        assert!(disconnected.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn cancellation_waits_for_in_flight_handler_to_finish() {
        let disconnected = Arc::new(AtomicBool::new(false));
        let started = Arc::new(tokio::sync::Notify::new());
        let completed = Arc::new(AtomicBool::new(false));
        let handler_started = started.clone();
        let handler_completed = completed.clone();
        let mut rt = Runtime::builder()
            .transport(OneMessageTransport {
                disconnected: disconnected.clone(),
            })
            .on_message(move |ctx| {
                let started = handler_started.clone();
                let completed = handler_completed.clone();
                async move {
                    started.notify_one();
                    ctx.cancel.as_ref().unwrap().cancelled().await;
                    completed.store(true, Ordering::SeqCst);
                }
            })
            .build()
            .unwrap();
        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            started.notified().await;
            trigger.cancel();
        });

        rt.run_until_cancelled(cancel).await.unwrap();

        assert!(completed.load(Ordering::SeqCst));
        assert!(disconnected.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn completed_handlers_are_reaped_while_the_runtime_is_active() {
        let disconnected = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(tokio::sync::Notify::new());
        let handler_completed = completed.clone();
        let mut rt = Runtime::builder()
            .transport(OneMessageTransport {
                disconnected: disconnected.clone(),
            })
            .on_message(move |_| {
                let completed = handler_completed.clone();
                async move { completed.notify_one() }
            })
            .build()
            .unwrap();
        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            completed.notified().await;
            tokio::task::yield_now().await;
            trigger.cancel();
        });

        rt.run_until_cancelled(cancel).await.unwrap();

        assert!(disconnected.load(Ordering::SeqCst));
    }

    /// The `run()` path selects on `std::future::pending()`; it must still
    /// terminate when the transport drops its sender.
    #[tokio::test]
    async fn run_without_cancellation_still_exits_when_transport_closes() {
        let disconnected = Arc::new(AtomicBool::new(false));
        let mut rt = runtime_with(false, disconnected.clone());

        tokio::time::timeout(Duration::from_secs(5), rt.run())
            .await
            .expect("the no-cancel branch must not block shutdown on transport close")
            .expect("clean exit");

        assert!(disconnected.load(Ordering::SeqCst));
    }
}
