mod cache;
mod client;
pub mod local;
mod magickmind_agent_stage;
mod magickmind_stage;
pub mod models;
mod provider;
mod runtime_state;
mod stage;

pub use client::MagickmindPersonaClient;
pub use local::LocalPersonaProvider;
pub use magickmind_agent_stage::{MagickmindAgentPersonaStage, PersonaCaller};
pub use magickmind_stage::{MagickmindPersonaStage, PersonaId};
pub use models::{
    EffectivePersonalityResponse, EffectiveSources, EffectiveTrait, PersonaSchema, TraitValue,
};
pub use provider::PersonaProvider;
pub use runtime_state::{RuntimeAffectSnapshot, RuntimeAffectState, RuntimeStateEnvelope};
pub use stage::PersonaContextBuilder;

use crate::core::context::Context;
use crate::models::{LlmMessage, SenderType};

/// Per-request conversation history stored in the pipeline [`Context`].
///
/// When present, persona stages ([`PersonaContextBuilder`],
/// [`MagickmindPersonaStage`]) splice these messages between the system prompt
/// and the user message instead of their construction-time history. Set it
/// per-request (e.g. from a context-preparation step) so a single shared stage
/// serves every message without rebuilding the pipeline. Mirrors [`PersonaId`].
#[derive(Debug, Clone, Default)]
pub struct ConversationHistory(pub Vec<LlmMessage>);

/// Resolve the user id used for dyadic (per-user) persona adaptation.
///
/// Only user-sent messages get a user id. Prefers the canonical user ID from
/// identity resolution when available, falling back to the raw sender id.
pub(crate) fn resolve_user_id(ctx: &Context) -> Option<String> {
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

    if ctx.message.sender_type == SenderType::User {
        canonical_id.or_else(|| Some(ctx.message.sender_id.clone()))
    } else {
        None
    }
}

/// The user-visible text for this message (the STT transcript when present).
pub(crate) fn user_text(ctx: &Context) -> &str {
    #[cfg(feature = "transport-audio")]
    if let Some(t) = ctx.get_ext::<crate::pipeline::extensions::TextInput>() {
        return &t.0;
    }
    &ctx.message.content
}

/// Assemble the LLM message list: system prompt, then history, then user text.
///
/// History comes from a per-request [`ConversationHistory`] extension when
/// present, else from the stage's construction-time `fallback_history`.
pub(crate) fn assemble_llm_messages(
    ctx: &Context,
    system_prompt: &str,
    fallback_history: &[LlmMessage],
) -> Vec<LlmMessage> {
    let history: &[LlmMessage] = ctx
        .get_ext::<ConversationHistory>()
        .map(|h| h.0.as_slice())
        .unwrap_or(fallback_history);

    let runtime_prompt = ctx
        .get::<RuntimeAffectSnapshot>()
        .map(|affect| format!("{system_prompt}\n\n{}", affect.prompt_instruction()));

    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(LlmMessage::system(
        runtime_prompt.as_deref().unwrap_or(system_prompt),
    ));
    messages.extend_from_slice(history);
    messages.push(LlmMessage::user(user_text(ctx)));
    messages
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::config::AgentConfig;
    use crate::models::Message;

    #[test]
    fn runtime_affect_is_appended_to_the_stable_persona_prompt() {
        let message = Arc::new(Message::new("hello", "user-1", "channel-1"));
        let mut ctx = Context::new(message, Arc::new(AgentConfig::default()));
        ctx.set_ext(RuntimeAffectSnapshot {
            pleasure: 0.25,
            arousal: -0.5,
            dominance: 0.75,
            state_version: 8,
        });

        let messages = assemble_llm_messages(&ctx, "You are Aria.", &[]);
        let system = messages[0].text();
        assert!(system.starts_with("You are Aria."));
        assert!(system.contains("Current temporary affect (PAD)"));
        assert!(system.contains("pleasure=+0.250"));
        assert!(system.contains("arousal=-0.500"));
        assert!(system.contains("dominance=+0.750"));
    }

    #[test]
    fn persona_prompt_is_unchanged_without_runtime_affect() {
        let message = Arc::new(Message::new("hello", "user-1", "channel-1"));
        let ctx = Context::new(message, Arc::new(AgentConfig::default()));

        let messages = assemble_llm_messages(&ctx, "You are Aria.", &[]);
        assert_eq!(messages[0].text(), "You are Aria.");
    }
}
