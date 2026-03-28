//! Deterministic coordination gate for multi-agent conversations.
//!
//! Checks the conversation history for the agent's own previous responses
//! (identified by `role = "assistant"`) and decides whether re-engaging is
//! appropriate based on how many new messages have appeared since.

use async_trait::async_trait;
use std::sync::Arc;

use crate::{LlmMessage, PipelineContext, PipelineStage, Result, Role};

/// A deterministic gate that prevents agents from re-engaging in conversations
/// where they've already responded and no significant new input has arrived.
///
/// Works by scanning the conversation history for `assistant`-role messages
/// (which represent the agent's own previous responses). If the agent has
/// already responded and only 0–1 new `user`-role messages followed (likely
/// another agent reacting), the pipeline halts. If 2+ new user messages have
/// appeared, the conversation has progressed and the agent is allowed to
/// re-engage.
///
/// This gate does **not** use an LLM — it's a fast role-based check.
/// Use it alongside [`RelevanceGate`] for a two-layer approach:
///
/// 1. `CoordinationGate` — "should I speak now?" (deterministic)
/// 2. `RelevanceGate` — "is this my topic?" (LLM-based)
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use mindroid::{Pipeline, CoordinationGate, RelevanceGate};
///
/// let pipeline = Pipeline::new()
///     .add_stage(CoordinationGate::new(history.clone(), 2))
///     .add_stage(gate)
///     // ...
/// ```
pub struct CoordinationGate {
    history: Arc<Vec<LlmMessage>>,
    /// Minimum number of new user messages after the agent's last response
    /// before re-engagement is allowed.
    min_new_messages: usize,
}

impl CoordinationGate {
    /// Create a new coordination gate.
    ///
    /// - `history`: the conversation history (injected at construction time).
    /// - `min_new_messages` — how many new user messages must appear after the
    ///   agent's last response before it will re-engage. A value of 2 means:
    ///   "don't respond to the immediate reaction, but allow re-engagement after
    ///   the conversation progresses."
    pub fn new(history: Arc<Vec<LlmMessage>>, min_new_messages: usize) -> Self {
        Self { history, min_new_messages }
    }
}

#[async_trait]
impl PipelineStage for CoordinationGate {
    fn name(&self) -> &str {
        "CoordinationGate"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let agent_id = &ctx.agent_config.agent_id;

        // Find the index of the last assistant-role message (agent's own response).
        // History messages use:
        //   - role "assistant" for the agent's own previous responses
        //   - role "user" for other participants' messages
        //   - role "system" for knowledge/corpus context
        let last_assistant_idx = self.history.iter().rposition(|m| m.role == Role::Assistant);

        let Some(last_idx) = last_assistant_idx else {
            // Agent hasn't responded yet — first engagement, always allow.
            tracing::debug!("CoordinationGate: first engagement for '{agent_id}'");
            return Ok(());
        };

        // Count user-role messages after the agent's last response.
        let new_user_messages = self.history[last_idx + 1..]
            .iter()
            .filter(|m| m.role == Role::User)
            .count();

        if new_user_messages < self.min_new_messages {
            tracing::info!(
                "CoordinationGate: '{}' already responded, only {} new message(s) (need {}) — halting",
                agent_id,
                new_user_messages,
                self.min_new_messages,
            );
            ctx.halted = true;
        } else {
            tracing::debug!(
                "CoordinationGate: '{}' re-engaging ({} new messages since last response)",
                agent_id,
                new_user_messages,
            );
        }

        Ok(())
    }
}
