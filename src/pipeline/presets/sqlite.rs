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

    pub async fn prepare_context(
        &self,
        channel_id: &str,
        agent_id: &str,
        limit: usize,
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

// ── SqliteContext ────────────────────────────────────────────────────────────

pub struct SqliteContext {
    client: Arc<SqliteClient>,
    agent_id: String,
    limit: usize,
}

impl SqliteContext {
    pub fn new(client: Arc<SqliteClient>) -> Self {
        Self {
            client,
            agent_id: String::new(),
            limit: 20,
        }
    }

    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = agent_id.into();
        self
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
            .prepare_context(&message.channel_id, &self.agent_id, self.limit)
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
        let channel_id = &ctx.message.channel_id;

        // Save user message
        self.client
            .save_message(
                channel_id,
                &ctx.message.sender_id,
                &ctx.message.content,
                None,
            )
            .await?;

        // Save assistant response
        let response = ctx.response.as_deref().unwrap_or("");

        self.client
            .save_message(
                channel_id,
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