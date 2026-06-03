use async_trait::async_trait;
use std::sync::Arc;

use tracing::debug;

use crate::core::context::Context;
use crate::error::Result;
use crate::models::{LlmMessage, SenderType};
#[cfg(feature = "transport-audio")]
use crate::pipeline::extensions::TextInput;
use crate::pipeline::PipelineStage;

use super::cache::PersonaCache;
use super::models::{EffectivePersonalityResponse, EffectiveTrait, PersonaSchema};
use super::provider::PersonaProvider;

/// A pipeline stage that fetches the effective personality from a persona
/// provider and builds a structured system prompt.
///
/// Replaces `SimpleContextBuilder` when a persona is configured.
/// Fetches the effective personality per-request using the sender's user_id
/// for dyadic (per-user) adaptation.
pub struct PersonaContextBuilder {
    provider: Arc<dyn PersonaProvider>,
    cache: Arc<PersonaCache>,
    persona_id: String,
    /// Static persona info (name, role, tones, background_story) fetched once.
    persona_info: PersonaSchema,
    /// Pre-fetched conversation history injected at construction time.
    history: Arc<Vec<LlmMessage>>,
}

impl PersonaContextBuilder {
    /// Create a new `PersonaContextBuilder`.
    ///
    /// Fetches the persona schema once at construction time for static fields
    /// (name, role, background_story, tones). The effective personality is
    /// fetched per-request in `process()`.
    pub async fn new(provider: Arc<dyn PersonaProvider>, persona_id: &str) -> Result<Self> {
        let persona_info = provider.get_persona(persona_id).await?;
        Ok(Self {
            provider,
            cache: Arc::new(PersonaCache::new()),
            persona_id: persona_id.to_string(),
            persona_info,
            history: Arc::new(Vec::new()),
        })
    }

    /// Provide conversation history for inclusion in the LLM prompt.
    pub fn with_history(mut self, history: Arc<Vec<LlmMessage>>) -> Self {
        self.history = history;
        self
    }

    /// Build a structured system prompt from the persona info and effective traits.
    fn build_system_prompt(&self, effective: &EffectivePersonalityResponse) -> String {
        let mut parts = Vec::new();

        // Identity line
        parts.push(format!(
            "You are {}, a {}.",
            self.persona_info.name, self.persona_info.role
        ));

        // Background story
        if !self.persona_info.background_story.is_empty() {
            parts.push(String::new());
            parts.push(self.persona_info.background_story.clone());
        }

        // Personality traits
        if !effective.traits.is_empty() {
            parts.push(String::new());
            parts.push("Your personality traits:".to_string());
            for t in &effective.traits {
                parts.push(format_trait(t));
            }
        }

        // Communication tones
        if !self.persona_info.tones.is_empty() {
            parts.push(String::new());
            parts.push(format!(
                "Communication tones: {}",
                self.persona_info.tones.join(", ")
            ));
        }

        parts.join("\n")
    }
}

#[async_trait]
impl PipelineStage for PersonaContextBuilder {
    fn name(&self) -> &str {
        "PersonaContextBuilder"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        // Determine user_id for dyadic blending (only for user-sent messages)
        // Prefer canonical user ID from identity resolution if available
        let canonical_id: Option<String>;
        #[cfg(feature = "identity")]
        {
            canonical_id = ctx
                .get_ext::<crate::identity::CanonicalUserId>()
                .map(|c| c.0.clone());
        }
        #[cfg(not(feature = "identity"))]
        {
            canonical_id = None;
        }

        let user_id = if ctx.message.sender_type == SenderType::User {
            canonical_id
                .as_deref()
                .or(Some(ctx.message.sender_id.as_str()))
        } else {
            None
        };

        // Check cache first
        let effective = if let Some(cached) = self.cache.get(&self.persona_id, user_id).await {
            debug!(
                "PersonaContextBuilder: cache hit for persona={} user={:?}",
                self.persona_id, user_id
            );
            cached
        } else {
            debug!(
                "PersonaContextBuilder: cache miss, fetching effective personality for persona={} user={:?}",
                self.persona_id, user_id
            );
            let resp = self
                .provider
                .get_effective_personality(&self.persona_id, user_id)
                .await?;
            self.cache.set(resp.clone()).await;
            resp
        };

        // Build the system prompt from effective personality
        let system_prompt = self.build_system_prompt(&effective);

        // Assemble LLM messages: system prompt first, then history, then user message
        let mut messages = vec![LlmMessage::system(&system_prompt)];
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
            "PersonaContextBuilder: {} history messages, {} total llm_messages, {} effective traits",
            self.history.len(),
            messages.len(),
            effective.traits.len(),
        );

        ctx.llm_messages = messages;

        Ok(())
    }
}

/// Format a single effective trait for the system prompt.
fn format_trait(t: &EffectiveTrait) -> String {
    let value = t.value.display();
    if let Some(ref lock) = t.sources.lock {
        format!("- {}: {} [lock: {}]", t.trait_ref, value, lock)
    } else {
        format!("- {}: {}", t.trait_ref, value)
    }
}
