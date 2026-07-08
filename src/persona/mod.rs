mod cache;
mod client;
pub mod local;
mod magickmind_stage;
pub mod models;
mod provider;
mod stage;

pub use client::MagickmindPersonaClient;
pub use local::LocalPersonaProvider;
pub use magickmind_stage::{MagickmindPersonaStage, PersonaId};
pub use models::{
    EffectivePersonalityResponse, EffectiveSources, EffectiveTrait, PersonaSchema, TraitValue,
};
pub use provider::PersonaProvider;
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

    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(LlmMessage::system(system_prompt));
    messages.extend_from_slice(history);
    messages.push(LlmMessage::user(user_text(ctx)));
    messages
}
