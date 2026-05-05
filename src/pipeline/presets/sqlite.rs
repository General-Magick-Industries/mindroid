use async_trait::async_trait;
use std::sync::Arc;

use crate::memory::Memory;
use crate::{
    ContextProvider, LlmMessage, Message, MindroidError, PipelineContext, PipelineStage, Result,
};

// ── SqliteClient ─────────────────────────────────────────────────────────────

pub struct SqliteClient {
    memory: Arc<dyn Memory>,
}

impl SqliteClient {
    pub fn new(memory: Arc<dyn Memory>) -> Self {
        Self { memory }
    }

    fn resolve_channel_id<'a>(&self, channel_id: &'a str) -> &'a str {
        if channel_id.is_empty() {
            "stdio"
        } else {
            channel_id
        }
    }

    pub async fn get_context(
        &self,
        channel_id: &str,
        limit: usize,
        agent_id: &str,
    ) -> Result<Vec<LlmMessage>> {
        let channel_id = self.resolve_channel_id(channel_id);

        let history = self.memory.get_history(channel_id, limit).await?;

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

    pub async fn save_exchange(
        &self,
        channel_id: &str,
        user_id: &str,
        user_msg: &str,
        agent_id: &str,
        response: &str,
        reply_to: Option<&str>,
    ) -> Result<()> {
        let channel_id = self.resolve_channel_id(channel_id);

        // Save user message
        self.memory
            .save_message(channel_id, user_id, user_msg, None)
            .await?;

        // Save assistant response
        self.memory
            .save_message(channel_id, agent_id, response, reply_to)
            .await?;

        Ok(())
    }
}

// ── SqliteContext ────────────────────────────────────────────────────────────

/// Loads recent chat history from SQLite and converts it to LlmMessages.
pub struct SqliteContext {
    client: Arc<SqliteClient>,
    agent_id: String,
    limit: usize,
}

impl SqliteContext {
    pub fn new(client: Arc<SqliteClient>, agent_id: impl Into<String>) -> Self {
        Self {
            client,
            agent_id: agent_id.into(),
            limit: 20,
        }
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

#[async_trait]
impl ContextProvider for SqliteContext {
    fn name(&self) -> &str {
        "SqliteContext"
    }

    async fn fetch(&self, message: &Message) -> Result<Vec<LlmMessage>> {
        self.client
            .get_context(&message.channel_id, self.limit, &self.agent_id)
            .await
    }
}

// ── SqlitePersistence ────────────────────────────────────────────────────────

pub struct SqlitePersistence {
    client: Arc<SqliteClient>,
}

impl SqlitePersistence {
    pub fn new(client: Arc<SqliteClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl PipelineStage for SqlitePersistence {
    fn name(&self) -> &str {
        "SqlitePersistence"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let response = ctx.response.as_deref().unwrap_or("");

        self.client
            .save_exchange(
                &ctx.message.channel_id,
                &ctx.message.sender_id,
                &ctx.message.content,
                &ctx.agent_config.agent_id,
                response,
                Some(&ctx.message.id),
            )
            .await
            .map_err(|e| MindroidError::Pipeline {
                stage: "SqlitePersistence".into(),
                message: e.to_string(),
                source: None,
            })?;

        Ok(())
    }
}