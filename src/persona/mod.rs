mod cache;
mod client;
pub mod local;
mod magickmind_agent_stage;
mod magickmind_stage;
pub mod models;
mod provider;
mod stage;

pub use client::MagickmindPersonaClient;
pub use local::LocalPersonaProvider;
pub use magickmind_agent_stage::MagickmindAgentPersonaStage;
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

    // Attribute the live turn the same way rendered history is attributed
    // (`[Name]: text`), so the model knows who it is replying to. Tool-result
    // frames are machine output, not speech — they stay unprefixed.
    let text = user_text(ctx);
    let attributed = ctx
        .message
        .metadata
        .get("sent_by_user_name")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty() && !text.trim_start().starts_with("<tool_result"))
        .map(|name| format!("[{name}]: {text}"));
    if let Some(block) = turn_context_block(ctx) {
        messages.push(LlmMessage::system(block));
    }
    messages.push(LlmMessage::user(attributed.as_deref().unwrap_or(text)));
    messages
}

/// Truncation budget for per-turn context. The backend already bounds each
/// entry, but the product of its limits still exceeds what belongs in a prompt.
const MAX_TURN_CONTEXT_BYTES: usize = 8192;

/// Render the sender's per-turn context as a system block, or `None` when they
/// supplied none. Keys are sorted so the same context yields the same prompt.
fn turn_context_block(ctx: &Context) -> Option<String> {
    let entries = ctx.message.metadata.get("context")?.as_object()?;
    let mut keys: Vec<&String> = entries.keys().collect();
    keys.sort();

    let mut rendered = Vec::new();
    let mut budget = MAX_TURN_CONTEXT_BYTES;
    for key in keys {
        let Some(value) = entries[key]
            .as_str()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        else {
            continue;
        };
        let line = format!("- {key}: {value}");
        if line.len() > budget {
            tracing::warn!("Dropping per-turn context entries past the size budget");
            break;
        }
        budget -= line.len();
        rendered.push(line);
    }

    (!rendered.is_empty()).then(|| {
        format!(
            "Context the sender supplied with this message. It describes their \
             current situation, not something they said:\n{}",
            rendered.join("\n")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentConfig;
    use crate::models::Message;
    use std::sync::Arc;

    fn ctx_with(content: &str, name: Option<&str>) -> Context {
        let mut msg = Message::new(content, "u1", "chan");
        if let Some(n) = name {
            msg.metadata.insert(
                "sent_by_user_name".into(),
                serde_json::Value::String(n.into()),
            );
        }
        Context::new(Arc::new(msg), Arc::new(AgentConfig::default()))
    }

    #[test]
    fn live_turn_is_attributed_when_the_fanout_names_the_sender() {
        let messages = assemble_llm_messages(&ctx_with("hello", Some(" Turtle ")), "sys", &[]);
        assert_eq!(messages[1].text(), "[Turtle]: hello");
    }

    #[test]
    fn unnamed_turns_stay_raw() {
        let messages = assemble_llm_messages(&ctx_with("hello", None), "sys", &[]);
        assert_eq!(messages[1].text(), "hello");
    }

    /// Machine frames are not speech; attributing one would corrupt the
    /// envelope the executor's history expects.
    #[test]
    fn tool_result_frames_stay_unprefixed() {
        let framed = "<tool_result id=\"1\">ok</tool_result>";
        let messages = assemble_llm_messages(&ctx_with(framed, Some("Turtle")), "sys", &[]);
        assert_eq!(messages[1].text(), framed);
    }

    fn ctx_with_context(entries: serde_json::Value) -> Context {
        let mut msg = Message::new("whats the time", "u1", "chan");
        msg.metadata.insert("context".into(), entries);
        Context::new(Arc::new(msg), Arc::new(AgentConfig::default()))
    }

    #[test]
    fn per_turn_context_rides_a_system_block_before_the_user_turn() {
        let ctx = ctx_with_context(serde_json::json!({"page": "/spaces/1", "device": "laptop"}));
        let messages = assemble_llm_messages(&ctx, "sys", &[]);

        assert_eq!(messages.len(), 3);
        let block = messages[1].text().to_string();
        assert!(
            block.find("- device: laptop") < block.find("- page: /spaces/1"),
            "entries render in sorted order: {block}"
        );
        assert_eq!(messages[2].text(), "whats the time");
    }

    #[test]
    fn empty_and_absent_context_add_no_block() {
        for entries in [
            serde_json::json!({}),
            serde_json::json!({"page": "   "}),
            serde_json::json!("not an object"),
        ] {
            let messages = assemble_llm_messages(&ctx_with_context(entries), "sys", &[]);
            assert_eq!(messages.len(), 2, "no system block belongs here");
        }
    }

    #[test]
    fn oversized_context_is_truncated_not_dropped() {
        let ctx = ctx_with_context(serde_json::json!({
            "a_small": "keep",
            "b_huge": "x".repeat(MAX_TURN_CONTEXT_BYTES),
        }));
        let block = assemble_llm_messages(&ctx, "sys", &[])[1]
            .text()
            .to_string();

        assert!(block.contains("- a_small: keep"));
        assert!(!block.contains("b_huge"));
        assert!(block.len() <= MAX_TURN_CONTEXT_BYTES + 200);
    }
}
