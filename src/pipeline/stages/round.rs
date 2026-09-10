//! The native tool round, split into two stages for [`AgentLoop`].
//!
//! [`ToolExecutorStage`](super::ToolExecutorStage) is one stage that owns its
//! own loop: it calls the model, runs the tools, and calls the model again,
//! all inside a single `process`. Nothing can be composed between those steps,
//! which is where compaction, approval and routing all want to live.
//!
//! These two stages are the same round with the loop taken out. [`LlmRound`]
//! makes one call and leaves any tool calls in run scope; [`ToolRound`] runs
//! them and asks [`AgentLoop`] for another pass. Anything placed between them
//! in the body pipeline sits *inside* the agent's reasoning loop:
//!
//! ```rust,ignore
//! Pipeline::new()
//!     .add_stage(TranscriptCompaction::new(80_000))   // before every call
//!     .add_stage(LlmRound::new(client, registry.clone()))
//!     .add_stage(ApprovalStage::<UserApproval>::new("confirm"))  // before every tool
//!     .add_stage(ToolRound::new(registry))
//! ```
//!
//! # Loop state
//!
//! The transcript is a [`Transcript`] in run scope, not `ctx.llm_messages`:
//! [`LlmMessage`](crate::LlmMessage) cannot represent an assistant turn holding
//! `tool_calls`, nor a `role: tool` result keyed by `tool_call_id`, which is the
//! same reason [`LlmClient::chat_with_tools`] takes async-openai types directly.
//! [`LlmRound`] seeds it from `ctx.llm_messages` on the first pass, so a setup
//! pipeline still builds context the ordinary way.
//!
//! # Scope
//!
//! Local tools only. Remote tools, the correlation gate and artifact
//! re-attachment stay in [`ToolExecutorStage`](super::ToolExecutorStage);
//! a pipeline needing those should keep using it.

use async_openai::types::chat::ChatCompletionRequestMessage;
use async_trait::async_trait;
use std::sync::Arc;
use tracing::debug;

use super::tool_executor::{assistant_turn, tool_turn};
use super::tool_executor_xml::{registry_for_turn, tool_context_for, truncate_str};
use crate::core::agent_loop::Continue;
use crate::core::context::Context;
use crate::error::Result;
use crate::llm_client::{LlmClient, NativeToolCall};
use crate::pipeline::PipelineStage;
use crate::tools::{DynamicRegistry, ToolRegistry};

/// The loop's transcript, in run scope.
///
/// Holds the shapes `LlmMessage` cannot: an assistant turn carrying
/// `tool_calls`, and `role: tool` results keyed by `tool_call_id`.
#[derive(Debug, Clone, Default)]
pub struct Transcript(pub Vec<ChatCompletionRequestMessage>);

/// Calls [`LlmRound`] made and [`ToolRound`] has yet to run.
#[derive(Debug, Clone)]
pub struct PendingCalls(pub Vec<NativeToolCall>);

/// One native-tool-calling round: call the model, record what it asked for.
///
/// Sets `ctx.response` to the round's prose and, when the model called tools,
/// leaves [`PendingCalls`] in run scope for [`ToolRound`].
pub struct LlmRound {
    client: LlmClient,
    registry: DynamicRegistry,
    model: Option<String>,
}

impl LlmRound {
    pub fn new(client: LlmClient, registry: Arc<ToolRegistry>) -> Self {
        Self::with_dynamic_registry(client, DynamicRegistry::new((*registry).clone()))
    }

    /// Build with a [`DynamicRegistry`] whose tools can be swapped at runtime.
    pub fn with_dynamic_registry(client: LlmClient, registry: DynamicRegistry) -> Self {
        Self {
            client,
            registry,
            model: None,
        }
    }

    /// Override the model for this stage (default: the client's).
    ///
    /// Two `LlmRound`s with different models behind a
    /// [`RouterStage`](crate::pipeline::combinators::RouterStage) is per-round
    /// model routing — a cheap model that escalates mid-turn.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

#[async_trait]
impl PipelineStage for LlmRound {
    fn name(&self) -> &str {
        "LlmRound"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let registry = registry_for_turn(ctx, &self.registry);
        let specs = LlmClient::tool_specs(&registry);

        // First pass: the setup pipeline's context is the transcript's seed.
        let mut transcript = ctx
            .take::<Transcript>()
            .unwrap_or_else(|| Transcript(LlmClient::convert_messages(&ctx.llm_messages)));

        let outcome = self
            .client
            .chat_with_tools(transcript.0.clone(), &specs, self.model.as_deref())
            .await?;

        debug!(
            "LlmRound: {} call(s) {:?}, prose {:?}",
            outcome.tool_calls.len(),
            outcome
                .tool_calls
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            truncate_str(&outcome.content, 200)
        );

        if !outcome.content.trim().is_empty() {
            ctx.response = Some(outcome.content.clone());
        }

        if !outcome.tool_calls.is_empty() {
            transcript
                .0
                .push(assistant_turn(&outcome.content, &outcome.tool_calls)?);
            ctx.set(PendingCalls(outcome.tool_calls));
        }

        ctx.set(transcript);
        Ok(())
    }
}

/// Runs the calls [`LlmRound`] recorded and asks for another pass.
///
/// A pass with no [`PendingCalls`] does nothing and requests nothing, which is
/// how a turn ends: the model answered without reaching for a tool.
pub struct ToolRound {
    registry: DynamicRegistry,
}

impl ToolRound {
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self::with_dynamic_registry(DynamicRegistry::new((*registry).clone()))
    }

    pub fn with_dynamic_registry(registry: DynamicRegistry) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl PipelineStage for ToolRound {
    fn name(&self) -> &str {
        "ToolRound"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let Some(PendingCalls(calls)) = ctx.take::<PendingCalls>() else {
            return Ok(());
        };

        let registry = registry_for_turn(ctx, &self.registry);
        let tool_ctx = tool_context_for(ctx);
        let mut transcript = ctx.take::<Transcript>().unwrap_or_default();
        let mut ends_turn = false;

        for call in &calls {
            // Every declared id must be answered or the provider rejects the
            // next request, so a failure is a result, never a skipped turn.
            let result = match registry.get(&call.name) {
                None => format!("Error: no tool named '{}'", call.name),
                Some(tool) => match parse_args(&call.arguments) {
                    Err(e) => format!("Error: arguments are not valid JSON: {e}"),
                    Ok(args) => match tool.execute(args, &tool_ctx).await {
                        Ok(output) => {
                            ends_turn |= tool.ends_turn();
                            output
                        }
                        Err(e) => format!("Error: {e}"),
                    },
                },
            };
            transcript.0.push(tool_turn(&call.id, result)?);
        }

        ctx.set(transcript);

        // A tool that delivered the turn itself has nothing to report back.
        if !ends_turn {
            ctx.set(Continue);
        }
        Ok(())
    }
}

/// Empty arguments are `{}`; anything else must parse.
fn parse_args(raw: &str) -> serde_json::Result<serde_json::Value> {
    if raw.trim().is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use crate::models::Message;
    use crate::tools::{Tool, ToolContext};
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn ctx() -> Context {
        Context::new(
            Arc::new(Message::new("hi", "u1", "c1")),
            Arc::new(AgentConfig::default()),
        )
    }

    struct Echo {
        hits: Arc<AtomicUsize>,
        ends_turn: bool,
    }

    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echoes"
        }
        fn parameters_schema(&self) -> Value {
            json!({"type": "object", "properties": {"text": {"type": "string"}}})
        }
        fn ends_turn(&self) -> bool {
            self.ends_turn
        }
        async fn execute(&self, args: Value, _ctx: &ToolContext) -> Result<String> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            Ok(args
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string())
        }
    }

    fn registry(ends_turn: bool) -> (Arc<ToolRegistry>, Arc<AtomicUsize>) {
        let hits = Arc::new(AtomicUsize::new(0));
        let registry = ToolRegistry::new().register(Echo {
            hits: hits.clone(),
            ends_turn,
        });
        (Arc::new(registry), hits)
    }

    fn call(id: &str, name: &str, arguments: &str) -> NativeToolCall {
        NativeToolCall {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    #[tokio::test]
    async fn a_pass_with_no_calls_asks_for_nothing() {
        let (reg, hits) = registry(false);
        let mut ctx = ctx();

        ToolRound::new(reg).process(&mut ctx).await.unwrap();

        assert_eq!(hits.load(Ordering::SeqCst), 0);
        assert!(
            ctx.get_run::<Continue>().is_none(),
            "a turn the model finished must not loop"
        );
    }

    #[tokio::test]
    async fn running_a_call_appends_its_result_and_asks_for_another_pass() {
        let (reg, hits) = registry(false);
        let mut ctx = ctx();
        ctx.set(PendingCalls(vec![call("c1", "echo", r#"{"text":"ping"}"#)]));

        ToolRound::new(reg).process(&mut ctx).await.unwrap();

        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert!(ctx.get_run::<Continue>().is_some());

        let transcript = ctx.get_run::<Transcript>().unwrap();
        assert_eq!(transcript.0.len(), 1);
        let ChatCompletionRequestMessage::Tool(t) = &transcript.0[0] else {
            panic!("expected a tool message");
        };
        assert_eq!(t.tool_call_id, "c1");
    }

    /// The provider rejects a round that leaves a declared id unanswered, so an
    /// unknown tool has to come back as a result rather than a gap.
    #[tokio::test]
    async fn every_declared_call_is_answered_even_when_it_fails() {
        let (reg, _) = registry(false);
        let mut ctx = ctx();
        ctx.set(PendingCalls(vec![
            call("c1", "echo", r#"{"text":"ok"}"#),
            call("c2", "nonexistent", "{}"),
            call("c3", "echo", "{not json"),
        ]));

        ToolRound::new(reg).process(&mut ctx).await.unwrap();

        let transcript = ctx.get_run::<Transcript>().unwrap();
        assert_eq!(transcript.0.len(), 3, "one result per declared id");
    }

    /// `ends_turn` is a property of the stage that runs the tool, not a second
    /// copy of the executor's loop — the round simply stops asking.
    #[tokio::test]
    async fn a_turn_ending_tool_stops_the_loop() {
        let (reg, hits) = registry(true);
        let mut ctx = ctx();
        ctx.set(PendingCalls(vec![call("c1", "echo", r#"{"text":"sent"}"#)]));

        ToolRound::new(reg).process(&mut ctx).await.unwrap();

        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert!(ctx.get_run::<Continue>().is_none());
    }

    #[test]
    fn empty_arguments_parse_as_an_empty_object() {
        assert_eq!(parse_args("").unwrap(), json!({}));
        assert_eq!(parse_args("   ").unwrap(), json!({}));
        assert!(parse_args("{oops").is_err());
    }
}
