use async_trait::async_trait;
use std::sync::Arc;

use tracing::debug;

#[cfg(feature = "transport-audio")]
use crate::pipeline::extensions::TextInput;
use crate::skills::skillset::SkillSet;
use crate::{LlmMessage, PipelineContext, PipelineStage, Result};

/// A simple context builder that creates LLM messages from an optional
/// system prompt, pre-fetched conversation history, and incoming message content.
///
/// # Examples
///
/// ```ignore
/// // No system prompt, no history
/// SimpleContextBuilder::new()
///
/// // With a system prompt
/// SimpleContextBuilder::with_prompt("You are a helpful assistant.")
///
/// // With pre-fetched conversation history
/// SimpleContextBuilder::with_history(history.clone())
///
/// // With both
/// SimpleContextBuilder::with_prompt_and_history("You are helpful.", history.clone())
/// ```
pub struct SimpleContextBuilder {
    system_prompt: Option<String>,
    history: Arc<Vec<LlmMessage>>,
}

impl SimpleContextBuilder {
    pub fn new() -> Self {
        Self {
            system_prompt: None,
            history: Arc::new(Vec::new()),
        }
    }

    pub fn with_prompt(prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: Some(prompt.into()),
            history: Arc::new(Vec::new()),
        }
    }

    pub fn with_history(history: Arc<Vec<LlmMessage>>) -> Self {
        Self {
            system_prompt: None,
            history,
        }
    }

    pub fn with_prompt_and_history(
        prompt: impl Into<String>,
        history: Arc<Vec<LlmMessage>>,
    ) -> Self {
        Self {
            system_prompt: Some(prompt.into()),
            history,
        }
    }

    /// Append a skill index to the system prompt.
    ///
    /// When skills are present, the index is appended after the base prompt.
    /// No-op if the skill set is empty.
    pub fn with_skills(mut self, skills: &SkillSet) -> Self {
        if !skills.is_empty() {
            let base = self.system_prompt.unwrap_or_default();
            self.system_prompt = Some(format!("{}\n\n{}", base, skills.index()));
        }
        self
    }
}

impl Default for SimpleContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for SimpleContextBuilder {
    fn name(&self) -> &str {
        "SimpleContextBuilder"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let mut messages = Vec::new();

        if let Some(ref prompt) = self.system_prompt {
            messages.push(LlmMessage::system(prompt));
        }

        messages.extend(self.history.as_ref().clone());

        #[cfg(feature = "transport-audio")]
        let user_text = ctx
            .get_ext::<TextInput>()
            .map(|t| t.0.as_str())
            .unwrap_or(&ctx.message.content);
        #[cfg(not(feature = "transport-audio"))]
        let user_text = &ctx.message.content;

        messages.push(LlmMessage::user(user_text));

        debug!(
            "SimpleContextBuilder: {} history messages, {} total llm_messages",
            self.history.len(),
            messages.len(),
        );
        for (i, msg) in messages.iter().enumerate() {
            debug!(
                "  llm_messages[{}] role={} len={}",
                i,
                msg.role.as_str(),
                msg.content.len()
            );
        }

        ctx.llm_messages = messages;

        Ok(())
    }
}
