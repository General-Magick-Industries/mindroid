use async_trait::async_trait;
use futures::future::join_all;
use tracing::{debug, info, warn};

use crate::error::Result;
use crate::models::{LlmMessage, Message};

/// A source of context messages for LLM conversations.
///
/// Implement this trait to fetch context from any source (databases,
/// APIs, vector stores, etc.) and convert it into LLM messages.
///
/// ```ignore
/// use mindroid::{ContextProvider, LlmMessage, Message, Result};
///
/// struct MyVectorStore { /* ... */ }
///
/// impl ContextProvider for MyVectorStore {
///     fn name(&self) -> &str { "MyVectorStore" }
///
///     fn fetch(&self, message: &Message) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<LlmMessage>>> + Send + '_>> {
///         Box::pin(async move {
///             let docs = self.search(&message.content).await?;
///             Ok(vec![LlmMessage::system(format!("Relevant docs:\n{docs}"))])
///         })
///     }
/// }
/// ```
#[async_trait]
pub trait ContextProvider: Send + Sync {
    /// Provider name for logging and debugging.
    fn name(&self) -> &str;

    /// Fetch context for the given message. Returns LLM messages
    /// (typically system messages) to be included in the conversation.
    async fn fetch(&self, message: &Message) -> Result<Vec<LlmMessage>>;
}

/// Standalone context builder that runs multiple [`ContextProvider`]s
/// and merges their results.
///
/// Unlike pipeline stages, `ContextPreparer` runs independently before
/// any pipeline. Its results can be shared across multiple pipeline
/// runs via [`PipelineContext::context`].
///
/// ```ignore
/// use mindroid::{ContextPreparer, MagickmindContext};
///
/// let preparer = ContextPreparer::new()
///     .add_provider(MagickmindContext::new(magickmind.clone()));
///
/// // Fetch once, share across pipelines
/// let context = preparer.prepare(&message).await?;
///
/// let mut pctx = PipelineContext::new(msg, agent_config);
/// pctx.context = context;
///
/// // All pipelines see the same context
/// ctx.run_with_context(&classify, &mut pctx).await?;
/// pctx.reset_output();
/// ctx.run_with_context(&respond, &mut pctx).await?;
/// ```
pub struct ContextPreparer {
    providers: Vec<Box<dyn ContextProvider>>,
}

impl ContextPreparer {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    pub fn add_provider(mut self, provider: impl ContextProvider + 'static) -> Self {
        self.providers.push(Box::new(provider));
        self
    }

    /// Run all providers in parallel and return the merged context messages.
    pub async fn prepare(&self, message: &Message) -> Result<Vec<LlmMessage>> {
        if self.providers.is_empty() {
            return Ok(Vec::new());
        }

        debug!(
            "ContextPreparer::prepare running {} provider(s)",
            self.providers.len()
        );

        let futures: Vec<_> = self.providers.iter().map(|p| p.fetch(message)).collect();
        let results = join_all(futures).await;

        let mut had_error = false;

        let mut messages = Vec::new();
        for (i, result) in results.into_iter().enumerate() {
            let name = self.providers[i].name();
            match result {
                Ok(msgs) => {
                    info!(
                        "ContextProvider '{}' returned {} message(s)",
                        name,
                        msgs.len()
                    );
                    messages.extend(msgs);
                }
                Err(e) => {
                    warn!("ContextProvider '{}' failed: {}", name, e);
                    had_error = true;
                    continue;
                }
            }
        }

        if messages.is_empty() && had_error {
            return Err(crate::error::MindroidError::Memory {
                message: "All context providers failed".into(),
                source: None,
            });
        }

        info!(
            "ContextPreparer::prepare completed with {} total message(s)",
            messages.len()
        );
        Ok(messages)
    }
}

impl Default for ContextPreparer {
    fn default() -> Self {
        Self::new()
    }
}
