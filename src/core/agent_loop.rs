//! `AgentLoop` — the iterative execution model.
//!
//! A [`Pipeline`] runs its stages once per inbound message. That is the right
//! shape for a turn that answers directly, and the wrong shape for a turn that
//! reasons, calls a tool, reads the result and reasons again: the iteration has
//! to happen *somewhere*, and in a single-pass pipeline the only place left is
//! inside one stage. An executor that owns its own loop also owns everything
//! that wants to happen between rounds — compaction, approval, retry, routing —
//! and none of it can be expressed with the combinators in
//! [`combinators`](crate::pipeline::combinators).
//!
//! `AgentLoop` moves the iteration up one level. It is a separate execution
//! model composed *of* pipelines rather than an extended `Pipeline`, for the
//! same reason [`OmniSession`](crate::omni) is (ADR-0003): `Pipeline`'s
//! invariants — admission control before any stage, at most one
//! [`StreamingStage`](crate::pipeline::StreamingStage) — are stated per run, and
//! a `Pipeline` that looped would have to restate every one of them.
//! See `docs/adr/0009-agent-loop.md`.
//!
//! ```text
//! setup   ──▶ once   : persona, memory fetch, context building
//! body    ──▶ n times: compact → llm round → approve → tool round
//! finish  ──▶ once   : post-processing, persistence
//! ```
//!
//! The three phases are separate pipelines rather than a per-stage flag, which
//! is what keeps a context builder from rebuilding `llm_messages` from history
//! on every pass and discarding the rounds accumulated so far.
//!
//! # Termination
//!
//! A body stage asks for another pass by setting [`Continue`] in run scope; the
//! loop clears it before each pass, so the request is per-iteration and cannot
//! latch. Nothing asking is the normal end of a turn — a round that produced no
//! tool calls has nothing to come back for — so termination falls out of the
//! composition instead of being a case the loop has to recognise.
//!
//! `ctx.halted` keeps the meaning it has everywhere else: stop, and do not
//! resume. It is deliberately not cleared between passes.

use std::time::Instant;

use futures::StreamExt;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::core::context::Context;
use crate::core::events::PipelineEvent;
use crate::error::Result;
use crate::models::{StreamEvent, TokenUsage};
use crate::pipeline::Pipeline;

/// Fold one pass's usage into the turn's running total.
fn add_usage(total: Option<TokenUsage>, pass: Option<TokenUsage>) -> Option<TokenUsage> {
    match (total, pass) {
        (Some(a), Some(b)) => Some(TokenUsage {
            prompt_tokens: a.prompt_tokens + b.prompt_tokens,
            completion_tokens: a.completion_tokens + b.completion_tokens,
            total_tokens: a.total_tokens + b.total_tokens,
        }),
        (total, pass) => total.or(pass),
    }
}

/// Default cap on body iterations.
pub const DEFAULT_LOOP_ITERATIONS: usize = 20;

/// Set in run scope by a body stage to request another pass.
///
/// The loop clears it before each pass, so a stage that stops asking stops the
/// loop. Setting it in `setup` has no effect — the first pass is unconditional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Continue;

/// Marks `ctx.response` as control traffic for the client — a framed remote
/// call — rather than prose, so [`AgentLoop::run_streaming`] never speaks it as
/// a `Chunk`.
///
/// `ToolRound`, `ToolExecutorStage` and `XmlToolExecutorStage` mark every framed
/// remote call. A stage of your own that leaves control traffic on
/// `ctx.response` marks it the same way, after setting it:
/// [`ControlResponse::mark`].
///
/// The mark names the response it was made for, so a later stage that replaces
/// `ctx.response` — prose after a nested loop's framed call, say — is not
/// muted by it. A new pass clears it; `finish` never does, so a loop nested as a
/// stage hands a framed response back to its parent's body still marked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlResponse(String);

impl ControlResponse {
    /// Mark the current `ctx.response` as control traffic. Does nothing when
    /// there is no response yet, so set the response first.
    pub fn mark(ctx: &mut Context) {
        if let Some(text) = ctx.response.clone() {
            ctx.set(ControlResponse(text));
        }
    }

    /// Whether a mark in run scope covers `text`.
    fn covers(ctx: &Context, text: Option<&str>) -> bool {
        ctx.get_run::<ControlResponse>()
            .is_some_and(|mark| text == Some(mark.0.as_str()))
    }
}

/// Why the loop stopped.
///
/// Reported three ways: on [`LoopOutcome`], on
/// [`PipelineEvent::LoopCompleted`] (the only place a streaming caller sees
/// it), and in run scope while `finish` runs, so a stage there can act on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// No stage asked for another pass.
    Settled,
    /// A stage set `ctx.halted`.
    Halted,
    /// The iteration cap was reached with a pass still pending.
    MaxIterations,
    /// The context's cancellation token fired.
    Cancelled,
}

/// The result of running an [`AgentLoop`].
#[derive(Debug, Clone)]
pub struct LoopOutcome {
    /// The response left by the last pass that produced one, after `finish`.
    pub response: Option<String>,
    pub reason: StopReason,
    /// Body passes actually executed.
    pub iterations: usize,
}

/// An iterative agent turn composed of three pipelines.
///
/// # Example
///
/// ```rust,ignore
/// let llm = LlmRound::new(client, registry);
/// let tools = llm.tool_round();
/// let agent = AgentLoop::new(
///     Pipeline::new()
///         .add_stage(TranscriptCompaction::from_tokens(60_000))
///         .add_stage(llm)
///         .add_stage(tools),
/// )
/// .with_setup(Pipeline::new().add_stage(SimpleContextBuilder::with_prompt(SYSTEM)))
/// .with_finish(Pipeline::new().add_stage(PostProcessor));
///
/// let outcome = agent.run(&mut ctx).await?;
/// ```
pub struct AgentLoop {
    setup: Pipeline,
    body: Pipeline,
    finish: Pipeline,
    max_iterations: usize,
    name: String,
}

impl AgentLoop {
    /// Build a loop around the pipeline that runs once per iteration.
    pub fn new(body: Pipeline) -> Self {
        Self {
            setup: Pipeline::new(),
            body,
            finish: Pipeline::new(),
            max_iterations: DEFAULT_LOOP_ITERATIONS,
            name: "AgentLoop".to_string(),
        }
    }

    /// Name this loop, so nested loops are distinguishable in logs and events.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Stages that run once before the first pass.
    pub fn with_setup(mut self, setup: Pipeline) -> Self {
        self.setup = setup;
        self
    }

    /// Stages that run once after the last pass — including when the loop
    /// halted or hit the cap, so a stage here must tolerate a turn that
    /// produced nothing: an unclaimed `tool_result` refused by `LlmRound`
    /// halts with no response, and `finish` still runs in full over it.
    ///
    /// While these stages run, the loop's [`StopReason`] is in run scope
    /// (`ctx.get_run::<StopReason>()`), so a stage can act on why the turn
    /// ended. It is taken back afterwards, and an enclosing loop's reason that
    /// was already there — this loop nested in the parent's `finish` — is put
    /// back, so neither loop reads the other's as its own.
    ///
    /// A loop that hit the cap stopped with a round still unanswered — the
    /// model asked for tools, got their results, and never replied — and
    /// [`LlmRound::cap_summary`](crate::pipeline::stages::LlmRound::cap_summary)
    /// is the stage that closes it, making the same closing request
    /// `ToolExecutorStage` makes for itself.
    ///
    /// Two exits do not reach these stages. A cancelled loop skips `finish`,
    /// because cancellation stops every pipeline at its next stage boundary
    /// and a phase that ran one stage of three is worse than one that ran
    /// none; a body error ends the turn the same way `run` does by
    /// propagating it. And a message refused by admission control (ADR-0008)
    /// runs *zero* finish stages however the loop exited — `Pipeline::run`
    /// re-checks admission at the head of every phase, so the refusal repeats
    /// there. Control traffic no stage could consume persists nothing, which
    /// is the point of refusing it.
    pub fn with_finish(mut self, finish: Pipeline) -> Self {
        self.finish = finish;
        self
    }

    /// Override the iteration cap (default [`DEFAULT_LOOP_ITERATIONS`]).
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Run setup, then the body until it settles, then finish.
    ///
    /// One [`Context`] spans every phase: run scope is the loop's state, so the
    /// transcript a round appends survives into the next pass.
    pub async fn run(&self, ctx: &mut Context) -> Result<LoopOutcome> {
        Ok(self.run_taking_mark(ctx).await?.0)
    }

    /// Run, then take the loop's [`ControlResponse`] out of run scope, reporting
    /// whether it covered the response the loop returns. Only the stage impl,
    /// which hands that response on to its parent, has a use for the answer.
    async fn run_taking_mark(&self, ctx: &mut Context) -> Result<(LoopOutcome, bool)> {
        let outcome = self.drive(ctx).await?;
        let control = ControlResponse::covers(ctx, outcome.response.as_deref());
        ctx.take::<ControlResponse>();
        Ok((outcome, control))
    }

    async fn drive(&self, ctx: &mut Context) -> Result<LoopOutcome> {
        let started = Instant::now();

        let carried = self.setup.run(ctx).await?;
        if ctx.halted {
            return self
                .wrap_up(ctx, carried, StopReason::Halted, 0, started)
                .await;
        }

        let mut carried = carried;
        let mut iterations = 0;
        let reason = loop {
            if ctx.cancel.is_cancelled() {
                break StopReason::Cancelled;
            }
            if iterations >= self.max_iterations {
                info!(
                    "AgentLoop: reached max iterations ({})",
                    self.max_iterations
                );
                break StopReason::MaxIterations;
            }

            ctx.take::<Continue>();
            ctx.take::<ControlResponse>();
            ctx.emit_event(PipelineEvent::LoopIterationStarted {
                loop_name: self.name.clone(),
                iteration: iterations,
            });

            // `Pipeline::run` takes the response, so a pass that produces none
            // must not erase the one before it.
            if let Some(text) = self.body.run(ctx).await? {
                carried = Some(text);
            }
            iterations += 1;

            // A token cancelled mid-pass stops the pipeline at its next stage
            // boundary and returns `Ok` with nothing set — indistinguishable
            // from a settled pass unless the token is checked here too.
            if ctx.cancel.is_cancelled() {
                break StopReason::Cancelled;
            }
            if ctx.halted {
                break StopReason::Halted;
            }
            if ctx.take::<Continue>().is_none() {
                break StopReason::Settled;
            }
        };

        self.wrap_up(ctx, carried, reason, iterations, started)
            .await
    }

    /// Run the loop, yielding every body pass's events in order.
    ///
    /// Each pass streams through [`Pipeline::run_streaming`], so one turn yields
    /// one segment per iteration — prose, then a tool round, then prose — which
    /// is the shape an agent turn has. `Pipeline`'s one-streaming-stage rule is
    /// untouched: it is a rule about a single pass.
    ///
    /// A pass's own `Complete` is swallowed and its usage folded into the
    /// turn's; the turn emits one `Complete` at the end, reporting what the
    /// whole loop spent rather than only its last round.
    ///
    /// **A body with no streaming stage is spoken one pass at a time.**
    /// `LlmRound` and `ToolRound` are ordinary stages, so a pass through them
    /// yields no `Chunk` of its own, and TTS would hear nothing for the whole
    /// turn. The loop yields each such pass's prose as one `Chunk` when the pass
    /// ends — after its tools ran, since the pass is the unit. It skips a
    /// response a stage marked as control traffic (a framed remote call), empty
    /// prose, and a pass the cancellation token cut short. A body that does have
    /// a streaming stage is left to it: that stage already chose what to speak.
    /// Like `Pipeline::run_streaming`'s chunks, these carry the text before
    /// `finish`; the final `Complete` carries it after. The one exception is the
    /// cap: `finish` text that the last pass did not already say — a
    /// `CapSummary` answer — is spoken as a last `Chunk`, since no pass will.
    ///
    /// **An `Error` from a pass ends the turn**, which is stricter than
    /// [`Pipeline::run_streaming`]: there a streaming stage's `Error` is
    /// forwarded and the post-streaming stages still run, so the pipeline
    /// completes with whatever was collected. Here it is the end — no further
    /// pass, no `finish`, no `Complete` — matching [`run`](Self::run), which
    /// propagates the `Err` and reaches none of them. A `StreamingStage` that
    /// treats its own `Error` as recoverable does not get that latitude inside
    /// a loop body.
    pub fn run_streaming<'a>(&'a self, ctx: &'a mut Context) -> BoxStream<'a, StreamEvent> {
        Box::pin(async_stream::stream! {
            let started = Instant::now();
            let speaks_passes = !self.body.has_streaming_stage();

            let mut carried = match self.setup.run(ctx).await {
                Ok(text) => text,
                Err(e) => {
                    yield StreamEvent::Error { message: e.to_string() };
                    return;
                }
            };

            let mut iterations = 0;
            let mut spent: Option<TokenUsage> = None;
            let reason = if ctx.halted {
                StopReason::Halted
            } else {
                loop {
                    if ctx.cancel.is_cancelled() {
                        break StopReason::Cancelled;
                    }
                    if iterations >= self.max_iterations {
                        info!("AgentLoop: reached max iterations ({})", self.max_iterations);
                        break StopReason::MaxIterations;
                    }

                    ctx.take::<Continue>();
                    ctx.take::<ControlResponse>();
                    ctx.emit_event(PipelineEvent::LoopIterationStarted {
                        loop_name: self.name.clone(),
                        iteration: iterations,
                    });

                    // `Pipeline::run_streaming` ends its stream on an error; the
                    // turn ends with it, as `run` does by propagating the `Err`,
                    // rather than going on to `finish` and a success-shaped
                    // `Complete`.
                    let mut failed = false;
                    {
                        let mut pass = self.body.run_streaming(ctx);
                        while let Some(event) = pass.next().await {
                            match event {
                                StreamEvent::Complete { usage, .. } => spent = add_usage(spent, usage),
                                StreamEvent::Error { message } => {
                                    failed = true;
                                    yield StreamEvent::Error { message };
                                }
                                other => yield other,
                            }
                        }
                    }
                    if failed {
                        ctx.take::<Continue>();
                        return;
                    }
                    iterations += 1;

                    // The streaming pipeline leaves the pass's text on the context
                    // rather than returning it; carry it the way `run` does.
                    let control = ControlResponse::covers(ctx, ctx.response.as_deref());
                    ctx.take::<ControlResponse>();
                    if let Some(text) = ctx.response.take() {
                        if speaks_passes
                            && !control
                            && !ctx.cancel.is_cancelled()
                            && !text.trim().is_empty()
                        {
                            yield StreamEvent::Chunk { content: text.clone() };
                        }
                        carried = Some(text);
                    }

                    if ctx.cancel.is_cancelled() {
                        break StopReason::Cancelled;
                    }
                    if ctx.halted {
                        break StopReason::Halted;
                    }
                    if ctx.take::<Continue>().is_none() {
                        break StopReason::Settled;
                    }
                }
            };

            let last_pass = carried.clone();
            let content = match self.run_finish(ctx, carried, reason).await {
                Ok(text) => text.unwrap_or_default(),
                Err(e) => {
                    yield StreamEvent::Error { message: e.to_string() };
                    return;
                }
            };
            // At the cap the last pass only promised an answer, and `finish`
            // (`CapSummary`) may supply it. That is new speech, and nothing has
            // spoken it — `finish` is not a pass. On any other exit `finish`
            // only rewrites what was already said, so it stays unspoken.
            let control = ControlResponse::covers(ctx, Some(&content));
            ctx.take::<ControlResponse>();
            if reason == StopReason::MaxIterations
                && !control
                && !content.trim().is_empty()
                && last_pass.as_deref() != Some(content.as_str())
            {
                yield StreamEvent::Chunk { content: content.clone() };
            }
            ctx.emit_event(PipelineEvent::LoopCompleted {
                loop_name: self.name.clone(),
                iterations,
                elapsed: started.elapsed(),
                reason,
            });
            debug!("AgentLoop::run_streaming {reason:?} after {iterations} iteration(s)");

            yield StreamEvent::Complete { content, usage: spent };
        })
    }

    /// Restore the carried response, run `finish` over it, and take the turn's
    /// text.
    async fn run_finish(
        &self,
        ctx: &mut Context,
        carried: Option<String>,
        reason: StopReason,
    ) -> Result<Option<String>> {
        // A pass that halted or hit the cap can leave its request behind. Left
        // there it is a stale instruction to whoever reads run scope next —
        // an enclosing loop would take it as its own and run again.
        ctx.take::<Continue>();

        // `finish` post-processes the turn's response, so it has to be back on
        // the context before those stages run.
        ctx.response = carried;

        if reason == StopReason::Cancelled {
            return Ok(ctx.response.take());
        }

        // `halted` is sticky and `Pipeline::run` stops after any stage that
        // sees it, so `finish` would run exactly one stage. Lift it for the
        // phase and put it back — the halt still means what it meant.
        let halted = std::mem::replace(&mut ctx.halted, false);
        // A loop nested in its parent's `finish` sets the parent's reason aside
        // and hands it back, so the parent's later finish stages still see it.
        let enclosing = ctx.take::<StopReason>();
        ctx.set(reason);
        let finished = self.finish.run(ctx).await;
        ctx.take::<StopReason>();
        if let Some(reason) = enclosing {
            ctx.set(reason);
        }
        ctx.halted |= halted;
        Ok(finished?.or_else(|| ctx.response.take()))
    }

    /// Run `finish` and report.
    async fn wrap_up(
        &self,
        ctx: &mut Context,
        carried: Option<String>,
        reason: StopReason,
        iterations: usize,
        started: Instant,
    ) -> Result<LoopOutcome> {
        let response = self.run_finish(ctx, carried, reason).await?;

        let elapsed = started.elapsed();
        ctx.emit_event(PipelineEvent::LoopCompleted {
            loop_name: self.name.clone(),
            iterations,
            elapsed,
            reason,
        });
        info!("AgentLoop settled: {reason:?} after {iterations} iteration(s) in {elapsed:.2?}");

        Ok(LoopOutcome {
            response,
            reason,
            iterations,
        })
    }
}

/// A loop is also a stage, so one can nest inside another's body — a planning
/// loop that hands off to an executing loop, both inside one turn.
///
/// # Two things it does that `run` does not
///
/// `run` ends with `finish` taking the response off the context; as a stage the
/// turn is not over, so the response is written back for the stages after it.
/// And the enclosing loop's [`Continue`] is held aside for the duration, so the
/// inner loop neither consumes its parent's request nor leaves its own behind.
///
/// # What is still shared
///
/// Run scope. Both loops see one [`Context`], which is the point when composing
/// phases of a single turn — the transcript carries from the planning loop into
/// the executing one — and a hazard if the two are meant to be independent
/// agents, since they would write the same `Transcript`. A genuine sub-agent
/// wants its own context: nest it through
/// [`DelegationTool`](crate::tools::DelegationTool), which builds a fresh one.
///
/// `ctx.halted` propagates outward unchanged: an inner loop that halts stops
/// the enclosing pipeline too, which is what halting means everywhere else.
#[async_trait::async_trait]
impl crate::pipeline::PipelineStage for AgentLoop {
    fn name(&self) -> &str {
        &self.name
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let enclosing = ctx.take::<Continue>();
        let (outcome, control) = self.run_taking_mark(ctx).await?;
        ctx.response = outcome.response;
        if control {
            ControlResponse::mark(ctx);
        }
        if let Some(request) = enclosing {
            ctx.set(request);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use crate::models::Message;
    use crate::pipeline::PipelineStage;
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn ctx() -> Context {
        Context::new(
            Arc::new(Message::new("hello", "u1", "c1")),
            Arc::new(AgentConfig::default()),
        )
    }

    /// Counts its passes and asks for another until `until` are done.
    struct Rounds {
        seen: Arc<AtomicUsize>,
        until: usize,
    }

    #[async_trait]
    impl PipelineStage for Rounds {
        fn name(&self) -> &str {
            "rounds"
        }

        async fn process(&self, ctx: &mut Context) -> Result<()> {
            let n = self.seen.fetch_add(1, Ordering::SeqCst) + 1;
            ctx.response = Some(format!("round {n}"));
            if n < self.until {
                ctx.set(Continue);
            }
            Ok(())
        }
    }

    struct Marker(&'static str, Arc<AtomicUsize>);

    #[async_trait]
    impl PipelineStage for Marker {
        fn name(&self) -> &str {
            self.0
        }

        async fn process(&self, _ctx: &mut Context) -> Result<()> {
            self.1.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn rounds(until: usize) -> (Pipeline, Arc<AtomicUsize>) {
        let seen = Arc::new(AtomicUsize::new(0));
        let pipeline = Pipeline::new().add_stage(Rounds {
            seen: seen.clone(),
            until,
        });
        (pipeline, seen)
    }

    /// A body that never asks for another pass is exactly today's pipeline:
    /// one run. This is the property that makes the loop additive.
    #[tokio::test]
    async fn a_body_that_never_continues_runs_once() {
        let (body, seen) = rounds(1);
        let outcome = AgentLoop::new(body).run(&mut ctx()).await.unwrap();

        assert_eq!(seen.load(Ordering::SeqCst), 1);
        assert_eq!(outcome.iterations, 1);
        assert_eq!(outcome.reason, StopReason::Settled);
        assert_eq!(outcome.response.as_deref(), Some("round 1"));
    }

    #[tokio::test]
    async fn the_body_repeats_while_a_stage_asks_for_another_pass() {
        let (body, seen) = rounds(3);
        let outcome = AgentLoop::new(body).run(&mut ctx()).await.unwrap();

        assert_eq!(seen.load(Ordering::SeqCst), 3);
        assert_eq!(outcome.iterations, 3);
        assert_eq!(outcome.reason, StopReason::Settled);
    }

    /// The request is per-pass: a stage that set it once must not keep the loop
    /// alive on a later pass that stayed silent.
    #[tokio::test]
    async fn a_continue_request_does_not_latch() {
        struct AsksOnce;

        #[async_trait]
        impl PipelineStage for AsksOnce {
            fn name(&self) -> &str {
                "asks-once"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                if ctx.get_run::<Asked>().is_none() {
                    ctx.set(Asked);
                    ctx.set(Continue);
                }
                Ok(())
            }
        }
        struct Asked;

        let outcome = AgentLoop::new(Pipeline::new().add_stage(AsksOnce))
            .run(&mut ctx())
            .await
            .unwrap();

        assert_eq!(outcome.iterations, 2);
        assert_eq!(outcome.reason, StopReason::Settled);
    }

    #[tokio::test]
    async fn the_cap_stops_a_body_that_never_settles() {
        let (body, seen) = rounds(usize::MAX);
        let outcome = AgentLoop::new(body)
            .with_max_iterations(4)
            .run(&mut ctx())
            .await
            .unwrap();

        assert_eq!(seen.load(Ordering::SeqCst), 4);
        assert_eq!(outcome.reason, StopReason::MaxIterations);
    }

    #[tokio::test]
    async fn setup_and_finish_run_once_around_many_passes() {
        let setup_hits = Arc::new(AtomicUsize::new(0));
        let finish_hits = Arc::new(AtomicUsize::new(0));
        let (body, seen) = rounds(3);

        let outcome = AgentLoop::new(body)
            .with_setup(Pipeline::new().add_stage(Marker("setup", setup_hits.clone())))
            .with_finish(Pipeline::new().add_stage(Marker("finish", finish_hits.clone())))
            .run(&mut ctx())
            .await
            .unwrap();

        assert_eq!(seen.load(Ordering::SeqCst), 3);
        assert_eq!(setup_hits.load(Ordering::SeqCst), 1);
        assert_eq!(finish_hits.load(Ordering::SeqCst), 1);
        assert_eq!(outcome.iterations, 3);
    }

    /// `Pipeline::run` takes the response, so the last pass that produced one
    /// has to survive a later pass that produced none.
    #[tokio::test]
    async fn a_silent_final_pass_keeps_the_last_response() {
        struct SpeaksThenLoopsOnce {
            seen: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl PipelineStage for SpeaksThenLoopsOnce {
            fn name(&self) -> &str {
                "speaks-then-silent"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                if self.seen.fetch_add(1, Ordering::SeqCst) == 0 {
                    ctx.response = Some("the answer".into());
                    ctx.set(Continue);
                }
                Ok(())
            }
        }

        let outcome = AgentLoop::new(Pipeline::new().add_stage(SpeaksThenLoopsOnce {
            seen: Arc::new(AtomicUsize::new(0)),
        }))
        .run(&mut ctx())
        .await
        .unwrap();

        assert_eq!(outcome.iterations, 2);
        assert_eq!(outcome.response.as_deref(), Some("the answer"));
    }

    /// `finish` post-processes the turn, so the carried response must be back
    /// on the context by the time those stages run.
    #[tokio::test]
    async fn finish_sees_the_carried_response() {
        struct Shout;

        #[async_trait]
        impl PipelineStage for Shout {
            fn name(&self) -> &str {
                "shout"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                ctx.response = ctx.response.as_ref().map(|r| r.to_uppercase());
                Ok(())
            }
        }

        let (body, _) = rounds(2);
        let outcome = AgentLoop::new(body)
            .with_finish(Pipeline::new().add_stage(Shout))
            .run(&mut ctx())
            .await
            .unwrap();

        assert_eq!(outcome.response.as_deref(), Some("ROUND 2"));
    }

    #[tokio::test]
    async fn halting_stops_the_loop_but_still_finishes() {
        struct Halt;

        #[async_trait]
        impl PipelineStage for Halt {
            fn name(&self) -> &str {
                "halt"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                ctx.halted = true;
                Ok(())
            }
        }

        let finish_hits = Arc::new(AtomicUsize::new(0));
        let outcome = AgentLoop::new(Pipeline::new().add_stage(Halt))
            .with_finish(Pipeline::new().add_stage(Marker("finish", finish_hits.clone())))
            .run(&mut ctx())
            .await
            .unwrap();

        assert_eq!(outcome.reason, StopReason::Halted);
        assert_eq!(outcome.iterations, 1);
        assert_eq!(finish_hits.load(Ordering::SeqCst), 1);
    }

    /// `halted` is sticky and `Pipeline::run` stops after any stage that sees
    /// it, so a one-stage `finish` cannot tell whether the phase ran in full.
    #[tokio::test]
    async fn finish_runs_every_stage_after_a_halt() {
        struct Halt;

        #[async_trait]
        impl PipelineStage for Halt {
            fn name(&self) -> &str {
                "halt"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                ctx.halted = true;
                Ok(())
            }
        }

        let first = Arc::new(AtomicUsize::new(0));
        let second = Arc::new(AtomicUsize::new(0));
        let mut ctx = ctx();
        let outcome = AgentLoop::new(Pipeline::new().add_stage(Halt))
            .with_finish(
                Pipeline::new()
                    .add_stage(Marker("post-process", first.clone()))
                    .add_stage(Marker("persist", second.clone())),
            )
            .run(&mut ctx)
            .await
            .unwrap();

        assert_eq!(outcome.reason, StopReason::Halted);
        assert_eq!(first.load(Ordering::SeqCst), 1);
        assert_eq!(
            second.load(Ordering::SeqCst),
            1,
            "the second finish stage ran"
        );
        assert!(ctx.halted, "the halt still means what it meant");
    }

    /// A token cancelled during a pass stops the pipeline at the next stage
    /// boundary and returns `Ok` with nothing set — the same shape as a pass
    /// that settled. The loop must check the token, not infer from silence.
    #[tokio::test]
    async fn a_cancellation_mid_pass_is_reported_and_skips_finish() {
        struct CancelsItself;

        #[async_trait]
        impl PipelineStage for CancelsItself {
            fn name(&self) -> &str {
                "cancels-itself"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                ctx.response = Some("partial".into());
                ctx.cancel.cancel();
                Ok(())
            }
        }

        let finish_hits = Arc::new(AtomicUsize::new(0));
        let outcome = AgentLoop::new(Pipeline::new().add_stage(CancelsItself))
            .with_finish(Pipeline::new().add_stage(Marker("finish", finish_hits.clone())))
            .run(&mut ctx())
            .await
            .unwrap();

        assert_eq!(outcome.reason, StopReason::Cancelled);
        assert_eq!(outcome.iterations, 1);
        assert_eq!(
            finish_hits.load(Ordering::SeqCst),
            0,
            "a cancelled turn does not run a phase that would stop after one stage"
        );
    }

    /// Admission control (ADR-0008) is re-checked at the head of every phase,
    /// so a refused message reaches no stage of any of them — `finish`
    /// included. Refusing control traffic means persisting nothing for it.
    #[tokio::test]
    async fn an_admission_refusal_runs_no_stage_of_any_phase() {
        let mut msg = Message::new("<tool_call>whatever</tool_call>", "u1", "c1");
        msg.message_type = crate::MessageType::ToolCall;
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));

        let setup_hits = Arc::new(AtomicUsize::new(0));
        let finish_hits = Arc::new(AtomicUsize::new(0));
        let (body, seen) = rounds(3);

        let outcome = AgentLoop::new(body)
            .with_setup(Pipeline::new().add_stage(Marker("setup", setup_hits.clone())))
            .with_finish(Pipeline::new().add_stage(Marker("finish", finish_hits.clone())))
            .run(&mut ctx)
            .await
            .unwrap();

        assert_eq!(setup_hits.load(Ordering::SeqCst), 0);
        assert_eq!(seen.load(Ordering::SeqCst), 0);
        assert_eq!(finish_hits.load(Ordering::SeqCst), 0, "finish too");
        assert_eq!(outcome.reason, StopReason::Halted);
        assert_eq!(outcome.response, None);
    }

    /// A halt in setup must not run the body at all.
    #[tokio::test]
    async fn a_halt_in_setup_skips_the_body() {
        struct Halt;

        #[async_trait]
        impl PipelineStage for Halt {
            fn name(&self) -> &str {
                "halt"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                ctx.halted = true;
                Ok(())
            }
        }

        let (body, seen) = rounds(3);
        let outcome = AgentLoop::new(body)
            .with_setup(Pipeline::new().add_stage(Halt))
            .run(&mut ctx())
            .await
            .unwrap();

        assert_eq!(seen.load(Ordering::SeqCst), 0);
        assert_eq!(outcome.iterations, 0);
        assert_eq!(outcome.reason, StopReason::Halted);
    }

    #[tokio::test]
    async fn a_cancelled_context_stops_before_the_next_pass() {
        let (body, seen) = rounds(usize::MAX);
        let mut ctx = ctx();
        ctx.cancel.cancel();

        let outcome = AgentLoop::new(body).run(&mut ctx).await.unwrap();

        assert_eq!(seen.load(Ordering::SeqCst), 0);
        assert_eq!(outcome.reason, StopReason::Cancelled);
    }

    /// Run scope is the loop's state: what a pass writes is there for the next.
    #[tokio::test]
    async fn run_scope_carries_across_passes() {
        struct Tally(Arc<AtomicUsize>);
        struct Count(usize);

        #[async_trait]
        impl PipelineStage for Tally {
            fn name(&self) -> &str {
                "tally"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                let seen = ctx.get_run::<Count>().map(|c| c.0).unwrap_or(0) + 1;
                ctx.set(Count(seen));
                self.0.store(seen, Ordering::SeqCst);
                if seen < 3 {
                    ctx.set(Continue);
                }
                Ok(())
            }
        }

        let last = Arc::new(AtomicUsize::new(0));
        AgentLoop::new(Pipeline::new().add_stage(Tally(last.clone())))
            .run(&mut ctx())
            .await
            .unwrap();

        assert_eq!(last.load(Ordering::SeqCst), 3);
    }

    /// A pass that halted or hit the cap can still have asked for another one.
    /// Left in run scope that request is a stale instruction to the next reader.
    #[tokio::test]
    async fn the_loop_leaves_no_request_behind() {
        let mut capped = ctx();
        let (body, _) = rounds(usize::MAX);
        AgentLoop::new(body)
            .with_max_iterations(2)
            .run(&mut capped)
            .await
            .unwrap();
        assert!(capped.get_run::<Continue>().is_none(), "capped");

        struct AsksThenHalts;

        #[async_trait]
        impl PipelineStage for AsksThenHalts {
            fn name(&self) -> &str {
                "asks-then-halts"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                ctx.set(Continue);
                ctx.halted = true;
                Ok(())
            }
        }

        let mut halted = ctx();
        AgentLoop::new(Pipeline::new().add_stage(AsksThenHalts))
            .run(&mut halted)
            .await
            .unwrap();
        assert!(halted.get_run::<Continue>().is_none(), "halted");
    }

    /// Passes taken by the inner loop during the current outer pass. Its own
    /// `setup` resets it, which is what makes the inner budget per-invocation.
    struct InnerCount(usize);

    struct ResetInner;

    #[async_trait]
    impl PipelineStage for ResetInner {
        fn name(&self) -> &str {
            "reset-inner"
        }

        async fn process(&self, ctx: &mut Context) -> Result<()> {
            ctx.set(InnerCount(0));
            Ok(())
        }
    }

    /// Asks for another pass until `until` passes have run *this invocation*.
    struct InnerRounds {
        total: Arc<AtomicUsize>,
        until: usize,
    }

    #[async_trait]
    impl PipelineStage for InnerRounds {
        fn name(&self) -> &str {
            "inner-rounds"
        }

        async fn process(&self, ctx: &mut Context) -> Result<()> {
            let n = ctx.get_run::<InnerCount>().map(|c| c.0).unwrap_or(0) + 1;
            ctx.set(InnerCount(n));
            self.total.fetch_add(1, Ordering::SeqCst);
            if n < self.until {
                ctx.set(Continue);
            }
            Ok(())
        }
    }

    fn inner_loop(until: usize) -> (AgentLoop, Arc<AtomicUsize>) {
        let total = Arc::new(AtomicUsize::new(0));
        let agent = AgentLoop::new(Pipeline::new().add_stage(InnerRounds {
            total: total.clone(),
            until,
        }))
        .with_setup(Pipeline::new().add_stage(ResetInner))
        .with_name("inner");
        (agent, total)
    }

    /// A loop is a stage, so one nests in another's body.
    #[tokio::test]
    async fn a_loop_nests_inside_another_loops_body() {
        let (inner, inner_total) = inner_loop(2);

        // The outer body runs the inner loop to completion, then asks for one
        // more outer pass of its own.
        let outer_seen = Arc::new(AtomicUsize::new(0));
        let outcome = AgentLoop::new(Pipeline::new().add_stage(inner).add_stage(Rounds {
            seen: outer_seen.clone(),
            until: 2,
        }))
        .run(&mut ctx())
        .await
        .unwrap();

        assert_eq!(
            outer_seen.load(Ordering::SeqCst),
            2,
            "the outer loop ran twice"
        );
        assert_eq!(
            inner_total.load(Ordering::SeqCst),
            4,
            "the inner loop ran its two passes on each outer pass"
        );
        assert_eq!(outcome.iterations, 2);
    }

    /// The inner loop must neither swallow the enclosing loop's request nor
    /// leave its own behind — either one changes how many passes the parent runs.
    #[tokio::test]
    async fn a_nested_loop_does_not_disturb_its_parents_request() {
        struct AsksThenNests(AgentLoop);

        #[async_trait]
        impl PipelineStage for AsksThenNests {
            fn name(&self) -> &str {
                "asks-then-nests"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                // The parent's request is set BEFORE the inner loop runs.
                if ctx.get_run::<Asked>().is_none() {
                    ctx.set(Asked);
                    ctx.set(Continue);
                }
                self.0.process(ctx).await
            }
        }
        struct Asked;

        let (inner, inner_total) = inner_loop(3);

        let outcome = AgentLoop::new(Pipeline::new().add_stage(AsksThenNests(inner)))
            .run(&mut ctx())
            .await
            .unwrap();

        assert_eq!(
            outcome.iterations, 2,
            "the parent's request survived the nested loop"
        );
        assert_eq!(inner_total.load(Ordering::SeqCst), 6);
    }

    /// `run` ends with `finish` taking the response; as a stage the turn is not
    /// over, so it has to be readable by the stages after it.
    #[tokio::test]
    async fn as_a_stage_the_response_is_left_on_the_context() {
        let (body, _) = rounds(2);
        let mut ctx = ctx();

        crate::pipeline::PipelineStage::process(&AgentLoop::new(body), &mut ctx)
            .await
            .unwrap();

        assert_eq!(ctx.response.as_deref(), Some("round 2"));
    }

    #[tokio::test]
    async fn streaming_yields_one_segment_per_pass_and_one_complete() {
        struct Emit {
            seen: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl PipelineStage for Emit {
            fn name(&self) -> &str {
                "emit"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                let n = self.seen.fetch_add(1, Ordering::SeqCst) + 1;
                ctx.response = Some(format!("pass {n}"));
                if n < 3 {
                    ctx.set(Continue);
                }
                Ok(())
            }
        }

        let agent = AgentLoop::new(Pipeline::new().add_stage(Emit {
            seen: Arc::new(AtomicUsize::new(0)),
        }));

        let mut ctx = ctx();
        let events: Vec<_> = agent.run_streaming(&mut ctx).collect().await;

        let completes = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Complete { .. }))
            .count();
        assert_eq!(completes, 1, "the turn ends once, not once per pass");
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Complete { content, .. }) if content == "pass 3"
        ));
    }

    struct Fails;

    #[async_trait]
    impl PipelineStage for Fails {
        fn name(&self) -> &str {
            "fails"
        }

        async fn process(&self, _ctx: &mut Context) -> Result<()> {
            Err(crate::error::MindroidError::pipeline("boom"))
        }
    }

    /// `run` propagates a body error and runs nothing after it. Streaming must
    /// mean the same thing: the stream ends on the `Error`, with no further
    /// pass and no success-shaped `Complete` carrying an earlier pass's text.
    #[tokio::test]
    async fn a_body_error_ends_the_stream_without_a_complete() {
        struct AsksThenFails(Arc<AtomicUsize>);

        #[async_trait]
        impl PipelineStage for AsksThenFails {
            fn name(&self) -> &str {
                "asks-then-fails"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                self.0.fetch_add(1, Ordering::SeqCst);
                ctx.response = Some("stale".into());
                ctx.set(Continue);
                Ok(())
            }
        }

        let passes = Arc::new(AtomicUsize::new(0));
        let finish_hits = Arc::new(AtomicUsize::new(0));
        let agent = AgentLoop::new(
            Pipeline::new()
                .add_stage(AsksThenFails(passes.clone()))
                .add_stage(Fails),
        )
        .with_finish(Pipeline::new().add_stage(Marker("finish", finish_hits.clone())));

        let mut ctx = ctx();
        let events: Vec<_> = agent.run_streaming(&mut ctx).collect().await;

        assert_eq!(passes.load(Ordering::SeqCst), 1, "no pass after the error");
        assert!(matches!(events.last(), Some(StreamEvent::Error { .. })));
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StreamEvent::Complete { .. })),
            "{events:?}"
        );
        assert_eq!(finish_hits.load(Ordering::SeqCst), 0);
        assert!(
            ctx.get_run::<Continue>().is_none(),
            "no request left behind"
        );
    }

    /// An admission refusal is a setup halt, so this is a common path: the
    /// stream must still wrap up rather than end with nothing at all.
    #[tokio::test]
    async fn a_setup_halt_still_completes_the_stream() {
        struct Halt;

        #[async_trait]
        impl PipelineStage for Halt {
            fn name(&self) -> &str {
                "halt"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                ctx.set(Continue);
                ctx.halted = true;
                Ok(())
            }
        }

        let finish_hits = Arc::new(AtomicUsize::new(0));
        let (body, seen) = rounds(3);
        let agent = AgentLoop::new(body)
            .with_setup(Pipeline::new().add_stage(Halt))
            .with_finish(Pipeline::new().add_stage(Marker("finish", finish_hits.clone())));

        let mut ctx = ctx();
        let events: Vec<_> = agent.run_streaming(&mut ctx).collect().await;

        assert_eq!(seen.load(Ordering::SeqCst), 0);
        assert_eq!(finish_hits.load(Ordering::SeqCst), 1);
        assert!(matches!(
            events.as_slice(),
            [StreamEvent::Complete { content, .. }] if content.is_empty()
        ));
        assert!(ctx.get_run::<Continue>().is_none());
    }

    /// Streaming leaves each pass's text on the context rather than returning
    /// it; the carry has to be replicated or a silent final pass ends the turn
    /// with nothing.
    #[tokio::test]
    async fn streaming_carries_the_last_response_over_a_silent_pass() {
        struct SpeaksThenLoopsOnce(Arc<AtomicUsize>);

        #[async_trait]
        impl PipelineStage for SpeaksThenLoopsOnce {
            fn name(&self) -> &str {
                "speaks-then-silent"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                    ctx.response = Some("the answer".into());
                    ctx.set(Continue);
                } else {
                    ctx.response = None;
                }
                Ok(())
            }
        }

        let agent = AgentLoop::new(
            Pipeline::new().add_stage(SpeaksThenLoopsOnce(Arc::new(AtomicUsize::new(0)))),
        );
        let mut ctx = ctx();
        let events: Vec<_> = agent.run_streaming(&mut ctx).collect().await;

        assert!(matches!(
            events.last(),
            Some(StreamEvent::Complete { content, .. }) if content == "the answer"
        ));
    }

    fn chunks(events: &[StreamEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Chunk { content } => Some(content.as_str()),
                _ => None,
            })
            .collect()
    }

    /// `LlmRound` and `ToolRound` are ordinary stages, so a body built from
    /// them streams nothing of its own; without the loop speaking each pass,
    /// TTS hears nothing for the whole turn.
    #[tokio::test]
    async fn a_body_with_no_streaming_stage_is_spoken_one_pass_at_a_time() {
        let (body, _) = rounds(3);
        let mut ctx = ctx();
        let events: Vec<_> = AgentLoop::new(body).run_streaming(&mut ctx).collect().await;

        assert_eq!(chunks(&events), ["round 1", "round 2", "round 3"]);
        let complete = events
            .iter()
            .position(|e| matches!(e, StreamEvent::Complete { .. }))
            .expect("the turn completes");
        assert_eq!(complete, events.len() - 1, "every chunk precedes Complete");
    }

    /// A framed remote call is an envelope for the client, not prose.
    #[tokio::test]
    async fn a_control_response_is_never_spoken() {
        struct Frames;

        #[async_trait]
        impl PipelineStage for Frames {
            fn name(&self) -> &str {
                "frames"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                ctx.response = Some(r#"{"type":"tool_call"}"#.into());
                ControlResponse::mark(ctx);
                Ok(())
            }
        }

        let mut ctx = ctx();
        let events: Vec<_> = AgentLoop::new(Pipeline::new().add_stage(Frames))
            .run_streaming(&mut ctx)
            .collect()
            .await;

        assert!(chunks(&events).is_empty(), "{events:?}");
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Complete { content, .. }) if content.contains("tool_call")
        ));
        assert!(
            ctx.get_run::<ControlResponse>().is_none(),
            "the marker is per pass and must not outlive the loop"
        );
    }

    /// An inner loop that frames a remote call hands the envelope back to its
    /// parent's body as that pass's response; a streaming parent must not speak
    /// it just because the inner loop's `finish` has run.
    #[tokio::test]
    async fn a_nested_loops_control_response_is_not_spoken_by_its_parent() {
        struct Frames;

        #[async_trait]
        impl PipelineStage for Frames {
            fn name(&self) -> &str {
                "frames"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                ctx.response = Some(r#"{"type":"tool_call"}"#.into());
                ControlResponse::mark(ctx);
                Ok(())
            }
        }

        let inner = AgentLoop::new(Pipeline::new().add_stage(Frames)).with_name("inner");
        let mut ctx = ctx();
        let events: Vec<_> = AgentLoop::new(Pipeline::new().add_stage(inner))
            .run_streaming(&mut ctx)
            .collect()
            .await;

        assert!(chunks(&events).is_empty(), "{events:?}");
    }

    /// A pass with nothing to say says nothing — no empty utterance for TTS.
    #[tokio::test]
    async fn blank_prose_is_not_spoken() {
        struct Blank;

        #[async_trait]
        impl PipelineStage for Blank {
            fn name(&self) -> &str {
                "blank"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                ctx.response = Some("  ".into());
                Ok(())
            }
        }

        let mut ctx = ctx();
        let events: Vec<_> = AgentLoop::new(Pipeline::new().add_stage(Blank))
            .run_streaming(&mut ctx)
            .collect()
            .await;

        assert!(chunks(&events).is_empty(), "{events:?}");
    }

    /// A streaming stage already decided what to speak; the loop adding its
    /// own chunk would say everything twice.
    #[tokio::test]
    async fn a_streaming_body_is_not_spoken_twice() {
        struct Streams;

        #[async_trait]
        impl PipelineStage for Streams {
            fn name(&self) -> &str {
                "streams"
            }

            async fn process(&self, _ctx: &mut Context) -> Result<()> {
                Ok(())
            }
        }

        impl crate::pipeline::StreamingStage for Streams {
            fn stream<'a>(&'a self, _ctx: &'a mut Context) -> BoxStream<'a, StreamEvent> {
                Box::pin(async_stream::stream! {
                    yield StreamEvent::Chunk { content: "streamed".into() };
                    yield StreamEvent::Complete { content: "streamed".into(), usage: None };
                })
            }
        }

        let mut ctx = ctx();
        let events: Vec<_> = AgentLoop::new(Pipeline::new().add_streaming_stage(Streams))
            .run_streaming(&mut ctx)
            .collect()
            .await;

        assert_eq!(chunks(&events), ["streamed"]);
    }

    fn loop_completed(rx: &mut tokio::sync::mpsc::UnboundedReceiver<PipelineEvent>) -> StopReason {
        let mut reason = None;
        while let Ok(event) = rx.try_recv() {
            if let PipelineEvent::LoopCompleted { reason: r, .. } = event {
                assert!(reason.replace(r).is_none(), "one LoopCompleted per turn");
            }
        }
        reason.expect("the loop reports its completion")
    }

    /// `run_streaming` returns events, not a `LoopOutcome`, so the event is
    /// the only place a streaming caller learns why the loop stopped.
    #[tokio::test]
    async fn loop_completed_reports_why_the_loop_stopped() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let (body, _) = rounds(1);
        AgentLoop::new(body)
            .run(&mut ctx().with_events(tx))
            .await
            .unwrap();
        assert_eq!(loop_completed(&mut rx), StopReason::Settled);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let (body, _) = rounds(10);
        let mut ctx = ctx().with_events(tx);
        let _: Vec<_> = AgentLoop::new(body)
            .with_max_iterations(2)
            .run_streaming(&mut ctx)
            .collect()
            .await;
        assert_eq!(loop_completed(&mut rx), StopReason::MaxIterations);
    }

    /// `finish` can act on why the turn ended — that is how a summary runs
    /// only at the cap — but the reason must not outlive the phase, or an
    /// enclosing loop's `finish` would read an inner loop's as its own.
    #[tokio::test]
    async fn finish_sees_the_stop_reason_and_nothing_after_does() {
        struct Records(Arc<std::sync::Mutex<Option<StopReason>>>);

        #[async_trait]
        impl PipelineStage for Records {
            fn name(&self) -> &str {
                "records"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                *self.0.lock().unwrap() = ctx.get_run::<StopReason>().copied();
                Ok(())
            }
        }

        let seen = Arc::new(std::sync::Mutex::new(None));
        let (body, _) = rounds(10);
        let mut ctx = ctx();
        AgentLoop::new(body)
            .with_max_iterations(2)
            .with_finish(Pipeline::new().add_stage(Records(seen.clone())))
            .run(&mut ctx)
            .await
            .unwrap();

        assert_eq!(*seen.lock().unwrap(), Some(StopReason::MaxIterations));
        assert!(ctx.get_run::<StopReason>().is_none());
    }

    /// Frames a remote call and marks it, as `ToolRound` does.
    struct FramesACall;

    #[async_trait]
    impl PipelineStage for FramesACall {
        fn name(&self) -> &str {
            "frames-a-call"
        }

        async fn process(&self, ctx: &mut Context) -> Result<()> {
            ctx.response = Some(r#"{"type":"tool_call"}"#.into());
            ControlResponse::mark(ctx);
            Ok(())
        }
    }

    /// Leaves fixed prose as the response.
    struct Says(&'static str);

    #[async_trait]
    impl PipelineStage for Says {
        fn name(&self) -> &str {
            "says"
        }

        async fn process(&self, ctx: &mut Context) -> Result<()> {
            ctx.response = Some(self.0.into());
            Ok(())
        }
    }

    /// Stands in for `CapSummary`: answers only when the loop hit its cap.
    struct AnswersAtTheCap;

    #[async_trait]
    impl PipelineStage for AnswersAtTheCap {
        fn name(&self) -> &str {
            "answers-at-the-cap"
        }

        async fn process(&self, ctx: &mut Context) -> Result<()> {
            if ctx.get_run::<StopReason>() == Some(&StopReason::MaxIterations) {
                ctx.response = Some("the answer".into());
            }
            Ok(())
        }
    }

    /// A pass the token cut short may hold half a thought; it is not spoken.
    #[tokio::test]
    async fn a_cancelled_pass_is_not_spoken() {
        struct SpeaksThenCancels;

        #[async_trait]
        impl PipelineStage for SpeaksThenCancels {
            fn name(&self) -> &str {
                "speaks-then-cancels"
            }

            async fn process(&self, ctx: &mut Context) -> Result<()> {
                ctx.response = Some("half a thought".into());
                ctx.cancel.cancel();
                Ok(())
            }
        }

        let mut ctx = ctx();
        let events: Vec<_> = AgentLoop::new(Pipeline::new().add_stage(SpeaksThenCancels))
            .run_streaming(&mut ctx)
            .collect()
            .await;

        assert!(chunks(&events).is_empty(), "{events:?}");
    }

    /// At the cap the last pass only promised an answer. The answer `finish`
    /// supplies is spoken, or a voice listener never hears it.
    #[tokio::test]
    async fn at_the_cap_the_answer_finish_supplies_is_spoken() {
        let (body, _) = rounds(10);
        let mut ctx = ctx();
        let events: Vec<_> = AgentLoop::new(body)
            .with_max_iterations(2)
            .with_finish(Pipeline::new().add_stage(AnswersAtTheCap))
            .run_streaming(&mut ctx)
            .collect()
            .await;

        assert_eq!(chunks(&events), ["round 1", "round 2", "the answer"]);
    }

    /// On any other exit `finish` rewrites what a pass already said; speaking
    /// it again would say the answer twice.
    #[tokio::test]
    async fn a_settled_answer_rewritten_by_finish_is_not_spoken_again() {
        let (body, _) = rounds(1);
        let mut ctx = ctx();
        let events: Vec<_> = AgentLoop::new(body)
            .with_finish(Pipeline::new().add_stage(Says("ROUND 1")))
            .run_streaming(&mut ctx)
            .collect()
            .await;

        assert_eq!(chunks(&events), ["round 1"]);
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Complete { content, .. }) if content == "ROUND 1"
        ));
    }

    /// A loop nested in its parent's `finish` hands the parent's reason back,
    /// so a stage after it — a summary — still sees why the parent stopped.
    #[tokio::test]
    async fn a_loop_nested_in_finish_hands_back_the_parents_reason() {
        let (outer_body, _) = rounds(10);
        let (inner_body, _) = rounds(1);
        let mut ctx = ctx();
        let outcome = AgentLoop::new(outer_body)
            .with_max_iterations(2)
            .with_finish(
                Pipeline::new()
                    .add_stage(AgentLoop::new(inner_body).with_name("inner"))
                    .add_stage(AnswersAtTheCap),
            )
            .run(&mut ctx)
            .await
            .unwrap();

        assert_eq!(outcome.reason, StopReason::MaxIterations);
        assert_eq!(
            outcome.response.as_deref(),
            Some("the answer"),
            "the summary after the inner loop saw the parent's MaxIterations"
        );
    }

    /// The mark names its response: prose a later stage writes over a nested
    /// loop's framed call is not muted by it.
    #[tokio::test]
    async fn prose_after_a_nested_loops_framed_call_is_spoken() {
        let inner = AgentLoop::new(Pipeline::new().add_stage(FramesACall)).with_name("inner");
        let mut ctx = ctx();
        let events: Vec<_> = AgentLoop::new(
            Pipeline::new()
                .add_stage(inner)
                .add_stage(Says("here is what I found")),
        )
        .run_streaming(&mut ctx)
        .collect()
        .await;

        assert_eq!(chunks(&events), ["here is what I found"]);
    }

    /// `run` returns the response; the mark on it is the loop's business and
    /// must not be left in run scope for whoever reads it next.
    #[tokio::test]
    async fn run_leaves_no_mark_behind() {
        let mut ctx = ctx();
        let outcome = AgentLoop::new(Pipeline::new().add_stage(FramesACall))
            .run(&mut ctx)
            .await
            .unwrap();

        assert!(outcome.response.unwrap().contains("tool_call"));
        assert!(ctx.get_run::<ControlResponse>().is_none());
    }

    /// `with_name` promises nested loops are distinguishable in events.
    #[tokio::test]
    async fn loop_events_carry_the_loop_name() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let (body, _) = rounds(1);
        AgentLoop::new(body)
            .with_name("planner")
            .run(&mut ctx().with_events(tx))
            .await
            .unwrap();

        let mut names = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match event {
                PipelineEvent::LoopIterationStarted { loop_name, .. }
                | PipelineEvent::LoopCompleted { loop_name, .. } => names.push(loop_name),
                _ => {}
            }
        }
        assert_eq!(names, ["planner", "planner"]);
    }
}
