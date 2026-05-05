use async_trait::async_trait;
use std::sync::Arc;

use crate::memory::Memory;
use crate::memory::sqlite::SqliteMemory;
use crate::{PipelineContext, PipelineStage, Result, ContextProvider, LlmMessage, Message};

/// Loads recent chat history from SQLite and converts it to LlmMessages.
/// Messages from `agent_id` are mapped to the `assistant` role; all others to `user`.
pub struct SqliteContextProvider {
    pub memory: Arc<SqliteMemory>,
    pub agent_id: String,
    pub limit: usize,
}

#[async_trait]
impl ContextProvider for SqliteContextProvider {
    fn name(&self) -> &str {
        "SqliteContextProvider"
    }

    async fn fetch(&self, message: &Message) -> Result<Vec<LlmMessage>> {
        let channel_id = if message.channel_id.is_empty() {
            "stdio".to_string()
        } else {
            message.channel_id.clone()
        };

        let history = self
            .memory
            .get_history(&channel_id, self.limit)
            .await?;

        let llm_messages = history
            .into_iter()
            .map(|msg| {
                if msg.sender_id == self.agent_id || msg.sender_id.is_empty() {
                    LlmMessage::assistant(msg.content)
                } else {
                    LlmMessage::user(msg.content)
                }
            })
            .collect();

        Ok(llm_messages)
    }
}


// ── SqlitePersistence ─────────────────────────────────────────────────────

pub struct SqlitePersistence {
    memory: Arc<dyn Memory>,
    agent_id: String,
}

impl SqlitePersistence {
    pub fn new(memory: Arc<dyn Memory>, agent_id: String) -> Self {
        Self { memory, agent_id }
    }
}

#[async_trait]
impl PipelineStage for SqlitePersistence {
    fn name(&self) -> &str {
        "SqlitePersistence"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let channel_id = if ctx.message.channel_id.is_empty() {
            "stdio".to_string()
        } else {
            ctx.message.channel_id.clone()
        };
        let user_id = &ctx.message.sender_id;

        // Save user message
        self.memory
            .save_message(&channel_id, user_id, &ctx.message.content, None)
            .await?;

        // Save agent response
        if let Some(response) = ctx.response.as_deref().filter(|r| !r.is_empty()) {
            self.memory
                .save_message(&channel_id, &self.agent_id, response, None)
                .await?;
        }

        Ok(())
    }
}
