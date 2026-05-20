use async_trait::async_trait;
use futures::future::join_all;
use tracing::{debug, info, warn};

use crate::error::{MindroidError, Result};
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
/// #[async_trait::async_trait]
/// impl ContextProvider for MyVectorStore {
///     fn name(&self) -> &str { "MyVectorStore" }
///
///     async fn fetch(&self, message: &Message) -> Result<Vec<LlmMessage>> {
///         let docs = self.search(&message.content).await?;
///         Ok(vec![LlmMessage::system(format!("Relevant docs:\n{docs}"))])
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

// ── Outcome types ───────────────────────────────────────────────────────────

/// A provider that failed during context preparation.
#[derive(Debug)]
pub struct ProviderWarning {
    /// Name of the provider that failed.
    pub provider: String,
    /// The error produced by the provider.
    pub error: MindroidError,
}

/// Outcome of [`ContextPreparer::prepare`] — an algebraic data type (ADT)
/// encoding all three possible states of a fan-out + merge operation.
///
/// Unlike `Result` which only has two states (Ok/Err), context preparation
/// has three: all providers succeeded, some failed, or all failed. This
/// enum makes each state explicit so the compiler enforces handling them.
///
/// Use [`into_messages`](Self::into_messages) for the simple path that
/// auto-logs warnings and returns whatever messages were collected.
///
/// # Examples
///
/// Explicit pattern matching (compiler-enforced):
/// ```ignore
/// match preparer.prepare(&message).await {
///     PrepareOutcome::Complete(msgs) => {
///         pctx.context = msgs;
///     }
///     PrepareOutcome::Degraded { messages, warnings } => {
///         for w in &warnings {
///             tracing::warn!("Provider '{}' failed: {}", w.provider, w.error);
///         }
///         pctx.context = messages;
///     }
///     PrepareOutcome::Failed(warnings) => {
///         tracing::error!("All context providers failed");
///         return;
///     }
/// }
/// ```
///
/// Simple path (auto-logs warnings, returns messages or empty):
/// ```ignore
/// pctx.context = preparer.prepare(&message).await.into_messages();
/// ```
#[derive(Debug)]
pub enum PrepareOutcome {
    /// All registered providers succeeded.
    Complete(Vec<LlmMessage>),

    /// Some providers succeeded, some failed. Messages from successful
    /// providers are available; failures are captured as warnings.
    Degraded {
        messages: Vec<LlmMessage>,
        warnings: Vec<ProviderWarning>,
    },

    /// All providers failed. No context is available.
    Failed(Vec<ProviderWarning>),
}

impl PrepareOutcome {
    /// Extract messages regardless of outcome state.
    ///
    /// - `Complete` → returns all messages.
    /// - `Degraded` → logs warnings, returns partial messages.
    /// - `Failed` → logs warnings, returns empty vec.
    ///
    /// This is the easy path when you just need messages and don't
    /// need custom failure handling.
    pub fn into_messages(self) -> Vec<LlmMessage> {
        match self {
            Self::Complete(msgs) => msgs,
            Self::Degraded { messages, warnings } => {
                for w in &warnings {
                    warn!(
                        "Context provider '{}' failed (continuing with partial context): {}",
                        w.provider, w.error
                    );
                }
                messages
            }
            Self::Failed(warnings) => {
                for w in &warnings {
                    warn!("Context provider '{}' failed: {}", w.provider, w.error);
                }
                Vec::new()
            }
        }
    }

    /// `true` when every registered provider succeeded.
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete(_))
    }

    /// `true` when at least one provider failed but some succeeded.
    pub fn is_degraded(&self) -> bool {
        matches!(self, Self::Degraded { .. })
    }

    /// `true` when all providers failed and no context is available.
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }

    /// Reference to collected messages (empty for `Failed`).
    pub fn messages(&self) -> &[LlmMessage] {
        match self {
            Self::Complete(msgs) | Self::Degraded { messages: msgs, .. } => msgs,
            Self::Failed(_) => &[],
        }
    }

    /// Reference to provider warnings (empty for `Complete`).
    pub fn warnings(&self) -> &[ProviderWarning] {
        match self {
            Self::Complete(_) => &[],
            Self::Degraded { warnings, .. } | Self::Failed(warnings) => warnings,
        }
    }
}

// ── ContextPreparer ─────────────────────────────────────────────────────────

/// Standalone context builder that runs multiple [`ContextProvider`]s
/// in parallel and merges their results.
///
/// Unlike pipeline stages, `ContextPreparer` runs independently before
/// any pipeline. Its results can be shared across multiple pipeline
/// runs via [`PipelineContext::context`].
///
/// ```ignore
/// use mindroid::{ContextPreparer, MagickmindContext, PrepareOutcome};
///
/// let preparer = ContextPreparer::new()
///     .add_provider(MagickmindContext::new(magickmind.clone()));
///
/// // Simple: auto-logs warnings, returns messages
/// let context = preparer.prepare(&message).await.into_messages();
///
/// // Or explicit: handle each state
/// let context = match preparer.prepare(&message).await {
///     PrepareOutcome::Complete(msgs) => msgs,
///     PrepareOutcome::Degraded { messages, .. } => messages,
///     PrepareOutcome::Failed(_) => return,
/// };
///
/// let mut pctx = PipelineContext::new(msg, agent_config);
/// pctx.context = context;
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

    /// Run all providers in parallel and return the outcome.
    ///
    /// This method never fails — provider errors are captured inside
    /// [`PrepareOutcome`] as warnings, not propagated as `Result::Err`.
    pub async fn prepare(&self, message: &Message) -> PrepareOutcome {
        if self.providers.is_empty() {
            return PrepareOutcome::Complete(Vec::new());
        }

        debug!(
            "ContextPreparer::prepare running {} provider(s)",
            self.providers.len()
        );

        let futures: Vec<_> = self.providers.iter().map(|p| p.fetch(message)).collect();
        let results = join_all(futures).await;

        let mut messages = Vec::new();
        let mut warnings = Vec::new();

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
                    warnings.push(ProviderWarning {
                        provider: name.to_string(),
                        error: e,
                    });
                }
            }
        }

        if warnings.is_empty() {
            info!(
                "ContextPreparer::prepare completed with {} message(s) — all providers succeeded",
                messages.len()
            );
            PrepareOutcome::Complete(messages)
        } else if messages.is_empty() {
            warn!(
                "ContextPreparer::prepare — all {} provider(s) failed",
                warnings.len()
            );
            PrepareOutcome::Failed(warnings)
        } else {
            info!(
                "ContextPreparer::prepare completed with {} message(s), {} provider(s) failed",
                messages.len(),
                warnings.len()
            );
            PrepareOutcome::Degraded { messages, warnings }
        }
    }
}

impl Default for ContextPreparer {
    fn default() -> Self {
        Self::new()
    }
}
