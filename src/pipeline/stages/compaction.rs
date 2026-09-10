//! Transcript compaction — a body stage, not a policy baked into an executor.
//!
//! An agent turn that calls tools grows its own transcript: every round appends
//! an assistant turn and one result per call, and tool output is the largest
//! thing in the conversation. Left alone the turn ends by overrunning the
//! context window rather than by finishing.
//!
//! Nothing about that is specific to a tool executor, which is why this is an
//! ordinary [`PipelineStage`] placed at the head of an
//! [`AgentLoop`](crate::AgentLoop) body: it runs before each model call, sees
//! the whole transcript, and is swappable for a different policy without
//! touching the round stages.
//!
//! # What it must not break
//!
//! An assistant turn carrying `tool_calls` and the `role: tool` results
//! answering them are one unit. Dropping either half leaves a call with no
//! result, or a result with no call, and the provider rejects the request. So
//! the transcript is grouped into blocks and whole blocks are dropped —
//! never individual messages.
//!
//! This is a size policy, not a summarizing one: it drops the oldest rounds
//! rather than compressing them. A summarizing stage implements the same trait
//! and goes in the same slot.

use async_openai::types::chat::ChatCompletionRequestMessage;
use async_trait::async_trait;
use tracing::debug;

use super::round::Transcript;
use crate::core::context::Context;
use crate::error::Result;
use crate::pipeline::PipelineStage;

/// Drops the oldest complete rounds once the transcript exceeds a budget.
///
/// Leading system messages and the most recent block always survive, so the
/// agent keeps its instructions and the round it is working on however tight
/// the budget is.
pub struct TranscriptCompaction {
    max_chars: usize,
}

impl TranscriptCompaction {
    /// Budget in serialized characters — a rough proxy for tokens (~4 chars
    /// each), deliberately cheap: an exact count needs the model's tokenizer.
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }

    /// Budget expressed in approximate tokens.
    pub fn from_tokens(max_tokens: usize) -> Self {
        Self::new(max_tokens.saturating_mul(4))
    }
}

#[async_trait]
impl PipelineStage for TranscriptCompaction {
    fn name(&self) -> &str {
        "TranscriptCompaction"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let Some(Transcript(messages)) = ctx.take::<Transcript>() else {
            return Ok(());
        };

        let before = messages.len();
        let kept = compact(messages, self.max_chars);
        if kept.len() < before {
            debug!(
                "TranscriptCompaction: dropped {} of {before} messages",
                before - kept.len()
            );
        }

        ctx.set(Transcript(kept));
        Ok(())
    }
}

/// Serialized size of one message, used as the token proxy.
fn size_of(msg: &ChatCompletionRequestMessage) -> usize {
    serde_json::to_string(msg).map(|s| s.len()).unwrap_or(0)
}

/// Whether this message opens a new block rather than continuing one.
///
/// `role: tool` results belong to the assistant turn before them; everything
/// else starts a block.
fn opens_block(msg: &ChatCompletionRequestMessage) -> bool {
    !matches!(msg, ChatCompletionRequestMessage::Tool(_))
}

/// Drop whole blocks from the oldest until the transcript fits.
fn compact(
    messages: Vec<ChatCompletionRequestMessage>,
    max_chars: usize,
) -> Vec<ChatCompletionRequestMessage> {
    let total: usize = messages.iter().map(size_of).sum();
    if total <= max_chars {
        return messages;
    }

    // Leading system messages are the agent's instructions: they are kept
    // whatever the budget, and are not a droppable block.
    let head_len = messages
        .iter()
        .take_while(|m| matches!(m, ChatCompletionRequestMessage::System(_)))
        .count();
    let mut head = messages;
    let rest = head.split_off(head_len);

    let mut blocks: Vec<Vec<ChatCompletionRequestMessage>> = Vec::new();
    for msg in rest {
        if blocks.is_empty() || opens_block(&msg) {
            blocks.push(vec![msg]);
        } else {
            blocks
                .last_mut()
                .expect("a block exists once one has been pushed")
                .push(msg);
        }
    }

    let head_size: usize = head.iter().map(size_of).sum();
    let mut budget = max_chars.saturating_sub(head_size);

    // Newest first, so what survives is the tail of the conversation. The last
    // block is kept unconditionally: a budget too small for one round degrades
    // to "the current round only" rather than to an empty transcript, which the
    // provider rejects outright.
    let mut keep_from = blocks.len();
    for (i, block) in blocks.iter().enumerate().rev() {
        let cost: usize = block.iter().map(size_of).sum();
        let newest = i + 1 == blocks.len();
        if !newest && cost > budget {
            break;
        }
        budget = budget.saturating_sub(cost);
        keep_from = i;
    }

    blocks.drain(..keep_from);
    head.extend(blocks.into_iter().flatten());
    head
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::chat::{
        ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs, FunctionCall,
    };

    fn system(text: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestSystemMessageArgs::default()
            .content(text)
            .build()
            .unwrap()
            .into()
    }

    fn user(text: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestUserMessageArgs::default()
            .content(text)
            .build()
            .unwrap()
            .into()
    }

    fn assistant_calling(id: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestAssistantMessageArgs::default()
            .tool_calls(vec![ChatCompletionMessageToolCalls::Function(
                ChatCompletionMessageToolCall {
                    id: id.into(),
                    function: FunctionCall {
                        name: "echo".into(),
                        arguments: "{}".into(),
                    },
                },
            )])
            .build()
            .unwrap()
            .into()
    }

    fn tool_result(id: &str, text: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestToolMessageArgs::default()
            .content(text)
            .tool_call_id(id)
            .build()
            .unwrap()
            .into()
    }

    /// A round is an assistant turn plus the results answering it.
    fn round(id: &str, payload: &str) -> Vec<ChatCompletionRequestMessage> {
        vec![assistant_calling(id), tool_result(id, payload)]
    }

    fn transcript(rounds: usize, payload_len: usize) -> Vec<ChatCompletionRequestMessage> {
        let mut msgs = vec![system("be helpful"), user("do the thing")];
        for i in 0..rounds {
            msgs.extend(round(&format!("c{i}"), &"x".repeat(payload_len)));
        }
        msgs
    }

    fn calls_and_results(msgs: &[ChatCompletionRequestMessage]) -> (Vec<String>, Vec<String>) {
        let mut calls = Vec::new();
        let mut results = Vec::new();
        for msg in msgs {
            match msg {
                ChatCompletionRequestMessage::Assistant(a) => {
                    for c in a.tool_calls.iter().flatten() {
                        let ChatCompletionMessageToolCalls::Function(f) = c else {
                            continue;
                        };
                        calls.push(f.id.clone());
                    }
                }
                ChatCompletionRequestMessage::Tool(t) => results.push(t.tool_call_id.clone()),
                _ => {}
            }
        }
        (calls, results)
    }

    #[test]
    fn a_transcript_inside_the_budget_is_untouched() {
        let msgs = transcript(3, 10);
        let before = msgs.len();
        assert_eq!(compact(msgs, 1_000_000).len(), before);
    }

    #[test]
    fn the_oldest_rounds_go_first() {
        let msgs = transcript(10, 200);
        let kept = compact(msgs, 2_000);

        assert!(kept.len() < 22, "something must have been dropped");
        let (calls, _) = calls_and_results(&kept);
        assert!(
            calls.last().is_some_and(|id| id == "c9"),
            "the newest round survives"
        );
        assert!(
            !calls.iter().any(|id| id == "c0"),
            "the oldest round is the first to go"
        );
    }

    /// The property the provider enforces: no call without its result, and no
    /// result without its call.
    #[test]
    fn compaction_never_splits_a_call_from_its_result() {
        for budget in [50, 200, 500, 1_500, 4_000] {
            let kept = compact(transcript(12, 150), budget);
            let (calls, results) = calls_and_results(&kept);
            assert_eq!(
                calls, results,
                "budget {budget} left a call and result mismatched"
            );
        }
    }

    #[test]
    fn the_system_prompt_always_survives() {
        let kept = compact(transcript(12, 400), 100);
        assert!(
            matches!(kept.first(), Some(ChatCompletionRequestMessage::System(_))),
            "the agent must keep its instructions at any budget"
        );
    }

    /// A budget too small for even one round must not produce an empty
    /// transcript — the provider rejects that outright.
    #[test]
    fn an_impossible_budget_still_leaves_the_current_round() {
        let kept = compact(transcript(6, 5_000), 1);
        assert!(kept.len() >= 2, "system prompt plus the newest block");
        let (calls, results) = calls_and_results(&kept);
        assert_eq!(calls, results);
    }

    #[tokio::test]
    async fn the_stage_is_a_no_op_without_a_transcript() {
        use crate::config::AgentConfig;
        use crate::models::Message;
        use std::sync::Arc;

        let mut ctx = Context::new(
            Arc::new(Message::new("hi", "u1", "c1")),
            Arc::new(AgentConfig::default()),
        );
        TranscriptCompaction::new(10)
            .process(&mut ctx)
            .await
            .unwrap();
        assert!(ctx.get_run::<Transcript>().is_none());
    }

    #[tokio::test]
    async fn the_stage_rewrites_the_transcript_in_run_scope() {
        use crate::config::AgentConfig;
        use crate::models::Message;
        use std::sync::Arc;

        let mut ctx = Context::new(
            Arc::new(Message::new("hi", "u1", "c1")),
            Arc::new(AgentConfig::default()),
        );
        ctx.set(Transcript(transcript(10, 300)));

        TranscriptCompaction::from_tokens(200)
            .process(&mut ctx)
            .await
            .unwrap();

        let kept = &ctx.get_run::<Transcript>().unwrap().0;
        assert!(kept.len() < 22);
        let (calls, results) = calls_and_results(kept);
        assert_eq!(calls, results);
    }
}
