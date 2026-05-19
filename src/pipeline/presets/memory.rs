//! Memory-backed preset: client, context provider, and persistence stage.
//!
//! Provides [`MemoryClient`] (a data-access layer over any [`Memory`] backend),
//! [`MemoryContext`] (a [`ContextProvider`] that fetches chat history), and
//! [`MemoryPersistence`] (a [`PipelineStage`] that saves messages).
//!
//! Unlike [`MagickmindClient`](super::magickmind::MagickmindClient) which
//! calls a remote API with rich features (corpus, pelican, auth), these types
//! operate on the [`Memory`] trait directly — making them compatible with any
//! backend: SQLite, Redis, in-memory stores, or any future implementation.
//!
//! # Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use mindroid::{ContextPreparer, Pipeline, PostProcessor};
//! use mindroid::memory::sqlite::SqliteMemory;
//! use mindroid::pipeline::presets::memory::{MemoryClient, MemoryContext, MemoryPersistence};
//!
//! let memory = Arc::new(SqliteMemory::new("chat.db")?);
//! let client = Arc::new(MemoryClient::new(memory));
//!
//! let preparer = ContextPreparer::new()
//!     .add_provider(MemoryContext::new(client.clone()).with_agent_id("bot-1"));
//!
//! let pipeline = Pipeline::new()
//!     .add_streaming_stage(llm)
//!     .add_stage(PostProcessor)
//!     .add_stage(MemoryPersistence::new(client));
//! ```

use async_trait::async_trait;
use std::sync::Arc;
use tracing::debug;

use crate::memory::Memory;
use crate::{
    ContextProvider, LlmMessage, Message, MindroidError, PipelineContext, PipelineStage, Result,
};

// ── MemoryClient ────────────────────────────────────────────────────────────

/// Data-access layer over any [`Memory`] backend.
///
/// Encapsulates shared concerns (channel ID normalization, history fetching
/// with role mapping, message saving) so that [`MemoryContext`] and
/// [`MemoryPersistence`] stay focused on their pipeline roles.
///
/// ```ignore
/// use mindroid::pipeline::presets::memory::MemoryClient;
///
/// let client = Arc::new(MemoryClient::new(memory));
/// ```
pub struct MemoryClient {
    memory: Arc<dyn Memory>,
}

impl MemoryClient {
    pub fn new(memory: Arc<dyn Memory>) -> Self {
        Self { memory }
    }

    /// Resolve empty channel IDs to a default value.
    fn resolve_channel_id<'a>(&self, channel_id: &'a str) -> &'a str {
        if channel_id.is_empty() {
            "stdio"
        } else {
            channel_id
        }
    }

    /// Fetch chat history and map messages to LLM roles.
    ///
    /// Messages where `sender_id` matches `agent_id` (or is empty) become
    /// `assistant`-role messages. All others become `user`-role messages.
    pub async fn prepare_context(
        &self,
        channel_id: &str,
        agent_id: &str,
        limit: usize,
    ) -> Result<Vec<LlmMessage>> {
        let channel_id = self.resolve_channel_id(channel_id);

        let history = self.memory.get_history(channel_id, limit).await?;

        debug!(
            "MemoryClient: fetched {} message(s) from channel '{}'",
            history.len(),
            channel_id
        );

        Ok(history
            .into_iter()
            .map(|msg| {
                if msg.sender_id == agent_id || msg.sender_id.is_empty() {
                    LlmMessage::assistant(msg.content)
                } else {
                    LlmMessage::user(msg.content)
                }
            })
            .collect())
    }

    /// Save a message to the memory backend.
    pub async fn save_message(
        &self,
        channel_id: &str,
        sender_id: &str,
        content: &str,
        reply_to_message_id: Option<&str>,
    ) -> Result<()> {
        let channel_id = self.resolve_channel_id(channel_id);

        self.memory
            .save_message(channel_id, sender_id, content, reply_to_message_id)
            .await?;

        Ok(())
    }
}

// ── MemoryContext ────────────────────────────────────────────────────────────

/// [`ContextProvider`] that fetches chat history via [`MemoryClient`]
/// and maps messages to LLM roles (user / assistant).
///
/// ```ignore
/// use mindroid::pipeline::presets::memory::{MemoryClient, MemoryContext};
///
/// let client = Arc::new(MemoryClient::new(memory));
/// let context = MemoryContext::new(client)
///     .with_agent_id("bot-1")
///     .with_limit(30);
/// ```
pub struct MemoryContext {
    client: Arc<MemoryClient>,
    agent_id: String,
    limit: usize,
}

impl MemoryContext {
    pub fn new(client: Arc<MemoryClient>) -> Self {
        Self {
            client,
            agent_id: String::new(),
            limit: 20,
        }
    }

    /// Set the agent's sender ID so its messages get the `assistant` role.
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = agent_id.into();
        self
    }

    /// Set the maximum number of history messages to fetch (default: 20).
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

#[async_trait]
impl ContextProvider for MemoryContext {
    fn name(&self) -> &str {
        "MemoryContext"
    }

    async fn fetch(&self, message: &Message) -> Result<Vec<LlmMessage>> {
        self.client
            .prepare_context(&message.channel_id, &self.agent_id, self.limit)
            .await
    }
}

// ── MemoryPersistence ───────────────────────────────────────────────────────

/// [`PipelineStage`] that persists user and assistant messages via
/// [`MemoryClient`] after pipeline processing.
///
/// Saves the incoming user message and, if a non-empty response was generated,
/// saves the assistant response with a reply-to link to the user message.
///
/// ```ignore
/// use mindroid::pipeline::presets::memory::{MemoryClient, MemoryPersistence};
///
/// let client = Arc::new(MemoryClient::new(memory));
/// let pipeline = Pipeline::new()
///     .add_streaming_stage(llm)
///     .add_stage(PostProcessor)
///     .add_stage(MemoryPersistence::new(client));
/// ```
pub struct MemoryPersistence {
    client: Arc<MemoryClient>,
}

impl MemoryPersistence {
    pub fn new(client: Arc<MemoryClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl PipelineStage for MemoryPersistence {
    fn name(&self) -> &str {
        "MemoryPersistence"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let channel_id = &ctx.message.channel_id;

        // Save user message
        self.client
            .save_message(channel_id, &ctx.message.sender_id, &ctx.message.content, None)
            .await
            .map_err(|e| MindroidError::Pipeline {
                stage: "MemoryPersistence".into(),
                message: format!("Failed to save user message: {e}"),
                source: None,
            })?;

        // Save assistant response (skip if empty or absent)
        if let Some(response) = ctx.response.as_deref().filter(|r| !r.is_empty()) {
            self.client
                .save_message(
                    channel_id,
                    &ctx.agent_config.agent_id,
                    response,
                    Some(&ctx.message.id),
                )
                .await
                .map_err(|e| MindroidError::Pipeline {
                    stage: "MemoryPersistence".into(),
                    message: format!("Failed to save assistant response: {e}"),
                    source: None,
                })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::sqlite::SqliteMemory;
    use crate::models::{ChannelType, MessageType, SenderType};

    fn test_message(channel_id: &str, sender_id: &str, content: &str) -> Message {
        Message {
            id: "msg-1".to_string(),
            content: content.to_string(),
            sender_id: sender_id.to_string(),
            sender_type: SenderType::default(),
            channel_id: channel_id.to_string(),
            channel_type: ChannelType::default(),
            message_type: MessageType::default(),
            timestamp: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
            platform: None,
        }
    }

    fn test_client() -> Arc<MemoryClient> {
        let mem = Arc::new(SqliteMemory::new(":memory:").unwrap());
        Arc::new(MemoryClient::new(mem))
    }

    #[tokio::test]
    async fn client_maps_roles_correctly() {
        let mem = Arc::new(SqliteMemory::new(":memory:").unwrap());

        // Save messages from different senders
        mem.save_message("chan1", "user-1", "hello from user", None)
            .await
            .unwrap();
        mem.save_message("chan1", "bot-1", "hello from bot", None)
            .await
            .unwrap();
        mem.save_message("chan1", "user-2", "hello from user 2", None)
            .await
            .unwrap();

        let client = Arc::new(MemoryClient::new(mem));
        let result = client.prepare_context("chan1", "bot-1", 10).await.unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].role, crate::models::Role::User);
        assert_eq!(result[1].role, crate::models::Role::Assistant);
        assert_eq!(result[2].role, crate::models::Role::User);
    }

    #[tokio::test]
    async fn client_empty_sender_becomes_assistant() {
        let mem = Arc::new(SqliteMemory::new(":memory:").unwrap());
        mem.save_message("chan1", "", "empty sender message", None)
            .await
            .unwrap();

        let client = Arc::new(MemoryClient::new(mem));
        let result = client.prepare_context("chan1", "bot-1", 10).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, crate::models::Role::Assistant);
    }

    #[tokio::test]
    async fn client_empty_channel_resolves_to_stdio() {
        let mem = Arc::new(SqliteMemory::new(":memory:").unwrap());
        mem.save_message("stdio", "user-1", "stdio message", None)
            .await
            .unwrap();

        let client = Arc::new(MemoryClient::new(mem));
        // Empty channel_id should resolve to "stdio"
        let result = client.prepare_context("", "bot-1", 10).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "stdio message");
    }

    #[tokio::test]
    async fn context_provider_delegates_to_client() {
        let mem = Arc::new(SqliteMemory::new(":memory:").unwrap());
        mem.save_message("chan1", "user-1", "hello", None)
            .await
            .unwrap();

        let client = Arc::new(MemoryClient::new(mem));
        let ctx = MemoryContext::new(client).with_agent_id("bot-1").with_limit(10);

        let message = test_message("chan1", "user-1", "test");
        let result = ctx.fetch(&message).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, crate::models::Role::User);
    }

    #[tokio::test]
    async fn persistence_skips_empty_response() {
        let mem = Arc::new(SqliteMemory::new(":memory:").unwrap());
        let client = Arc::new(MemoryClient::new(mem.clone()));
        let persistence = MemoryPersistence::new(client);

        let message = test_message("chan1", "user-1", "hello");
        let agent_config = crate::config::AgentConfig {
            agent_id: "bot-1".to_string(),
            ..Default::default()
        };

        let mut pctx = PipelineContext::new(message, agent_config);
        // No response set — ctx.response is None

        persistence.process(&mut pctx).await.unwrap();

        // Only user message saved (not empty assistant response)
        let history = mem.get_history("chan1", 10).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "hello");
        assert_eq!(history[0].sender_id, "user-1");
    }

    #[tokio::test]
    async fn persistence_saves_both_messages() {
        let mem = Arc::new(SqliteMemory::new(":memory:").unwrap());
        let client = Arc::new(MemoryClient::new(mem.clone()));
        let persistence = MemoryPersistence::new(client);

        let message = test_message("chan1", "user-1", "what is 2+2?");
        let agent_config = crate::config::AgentConfig {
            agent_id: "bot-1".to_string(),
            ..Default::default()
        };

        let mut pctx = PipelineContext::new(message, agent_config);
        pctx.response = Some("4".to_string());

        persistence.process(&mut pctx).await.unwrap();

        let history = mem.get_history("chan1", 10).await.unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].content, "what is 2+2?");
        assert_eq!(history[0].sender_id, "user-1");
        assert_eq!(history[1].content, "4");
        assert_eq!(history[1].sender_id, "bot-1");
    }
}
