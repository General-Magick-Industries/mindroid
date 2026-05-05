use async_trait::async_trait;
use std::sync::Arc;

use crate::memory::Memory;
use crate::{PipelineContext, PipelineStage, Result};

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
