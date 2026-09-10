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

/// Why the loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
/// let agent = AgentLoop::new(
///     Pipeline::new()
///         .add_stage(TranscriptCompaction::from_tokens(60_000))
///         .add_stage(LlmRound::new(client, registry.clone()))
///         .add_stage(ToolRound::new(registry)),
/// )
/// .with_setup(Pipeline::new().add_stage(SimpleContextBuilder::with_prompt(SYSTEM)))
/// .with_finish(Pipeline::new().add_stage(PostProcessor::new()));
///
/// let outcome = agent.run(&mut ctx).await?;
/// ```
pub struct AgentLoop {
    setup: Pipeline,
    body: Pipeline,
    finish: Pipeline,
    max_iterations: usize,
}

impl AgentLoop {
    /// Build a loop around the pipeline that runs once per iteration.
    pub fn new(body: Pipeline) -> Self {
        Self {
            setup: Pipeline::new(),
            body,
            finish: Pipeline::new(),
            max_iterations: DEFAULT_LOOP_ITERATIONS,
        }
    }

    /// Stages that run once before the first pass.
    pub fn with_setup(mut self, setup: Pipeline) -> Self {
        self.setup = setup;
        self
    }

    /// Stages that run once after the last pass — including when the loop
    /// halted, hit the cap, or was cancelled.
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
            ctx.emit_event(PipelineEvent::LoopIterationStarted {
                iteration: iterations,
            });

            // `Pipeline::run` takes the response, so a pass that produces none
            // must not erase the one before it.
            if let Some(text) = self.body.run(ctx).await? {
                carried = Some(text);
            }
            iterations += 1;

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
    pub fn run_streaming<'a>(&'a self, ctx: &'a mut Context) -> BoxStream<'a, StreamEvent> {
        Box::pin(async_stream::stream! {
            let started = Instant::now();

            if let Err(e) = self.setup.run(ctx).await {
                yield StreamEvent::Error { message: e.to_string() };
                return;
            }
            if ctx.halted {
                return;
            }

            let mut iterations = 0;
            let mut spent: Option<TokenUsage> = None;
            while iterations < self.max_iterations && !ctx.cancel.is_cancelled() {
                ctx.take::<Continue>();
                ctx.emit_event(PipelineEvent::LoopIterationStarted { iteration: iterations });

                {
                    let mut pass = self.body.run_streaming(ctx);
                    while let Some(event) = pass.next().await {
                        match event {
                            StreamEvent::Complete { usage, .. } => spent = add_usage(spent, usage),
                            other => yield other,
                        }
                    }
                }
                iterations += 1;

                if ctx.halted || ctx.take::<Continue>().is_none() {
                    break;
                }
            }

            // `Pipeline::run` takes the response, so even an empty `finish`
            // drains it — the turn's text has to come back from the call.
            let content = match self.finish.run(ctx).await {
                Ok(finished) => finished.or_else(|| ctx.response.take()).unwrap_or_default(),
                Err(e) => {
                    yield StreamEvent::Error { message: e.to_string() };
                    return;
                }
            };
            ctx.emit_event(PipelineEvent::LoopCompleted {
                iterations,
                elapsed: started.elapsed(),
            });
            debug!("AgentLoop::run_streaming settled after {iterations} iteration(s)");

            yield StreamEvent::Complete { content, usage: spent };
        })
    }

    /// Restore the carried response, run `finish` over it, and report.
    async fn wrap_up(
        &self,
        ctx: &mut Context,
        carried: Option<String>,
        reason: StopReason,
        iterations: usize,
        started: Instant,
    ) -> Result<LoopOutcome> {
        // `finish` post-processes the turn's response, so it has to be back on
        // the context before those stages run.
        ctx.response = carried;
        let response = self.finish.run(ctx).await?.or_else(|| ctx.response.take());

        let elapsed = started.elapsed();
        ctx.emit_event(PipelineEvent::LoopCompleted {
            iterations,
            elapsed,
        });
        info!("AgentLoop settled: {reason:?} after {iterations} iteration(s) in {elapsed:.2?}");

        Ok(LoopOutcome {
            response,
            reason,
            iterations,
        })
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
}
