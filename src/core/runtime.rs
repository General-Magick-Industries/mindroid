use std::sync::Arc;
use tokio::sync::{Semaphore, mpsc};
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
}

impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::new()
    }

    /// Start the runtime main loop. Blocks until shutdown.
    pub async fn run(&mut self) -> Result<()> {
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
        const MAX_CONCURRENT_HANDLERS: usize = 50;
        let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_HANDLERS));

        let mut seen_ids = std::collections::HashSet::<String>::new();
        const MAX_SEEN_IDS: usize = 10_000;
        while let Some(msg) = rx.recv().await {
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

            let ctx = MessageContext {
                message: Arc::new(msg),
                agent_config: self.agent_config.clone(),
                pipeline: self.pipeline.clone(),
                transport: self.transport_sender.clone(),
                observers: self.observers.clone(),
            };

            let handler_fut = (self.handler)(ctx);
            let permit = match semaphore.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    tracing::error!("Handler semaphore closed, dropping message");
                    continue;
                }
            };
            tokio::spawn(async move {
                let _permit = permit; // dropped when handler completes
                handler_fut.await;
            });
        }

        self.shutdown().await
    }

    /// Gracefully shut down the runtime.
    pub async fn shutdown(&mut self) -> Result<()> {
        for (token, handle) in self.routine_handles.drain(..) {
            token.cancel();
            let timeout = tokio::time::timeout(std::time::Duration::from_secs(5), handle);
            match timeout.await {
                Ok(_) => {}
                Err(_) => {
                    tracing::warn!("Routine did not stop within timeout; aborting");
                }
            }
        }
        tracing::info!("Shutting down...");
        for obs in self.observers.iter() {
            obs.on_shutdown().await;
        }
        self.transport.disconnect().await?;
        tracing::info!("Shutdown complete");
        Ok(())
    }
}
