use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::core::context::Context;
use crate::error::{MindroidError, Result};

use super::{Pipeline, PipelineStage};

/// Gate-based binary branching combinator.
///
/// Runs a gate stage, then:
/// - Gate does NOT halt → run `pass` sub-pipeline
/// - Gate halts → reset halted, run `fail` sub-pipeline (if provided)
///
/// If the selected sub-pipeline sets `halted = true`, it propagates to the outer pipeline.
/// Sub-pipelines mutate the shared Context.
pub struct BranchStage {
    name: String,
    gate: Box<dyn PipelineStage>,
    pass: Arc<Pipeline>,
    fail: Option<Arc<Pipeline>>,
}

impl BranchStage {
    pub fn new(name: impl Into<String>, gate: impl PipelineStage + 'static) -> Self {
        Self {
            name: name.into(),
            gate: Box::new(gate),
            pass: Arc::new(Pipeline::new()),
            fail: None,
        }
    }

    pub fn on_pass(mut self, pipeline: Pipeline) -> Self {
        self.pass = Arc::new(pipeline);
        self
    }

    pub fn on_fail(mut self, pipeline: Pipeline) -> Self {
        self.fail = Some(Arc::new(pipeline));
        self
    }
}

#[async_trait]
impl PipelineStage for BranchStage {
    fn name(&self) -> &str {
        &self.name
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        self.gate.process(ctx).await?;

        if ctx.halted {
            ctx.halted = false;
            if let Some(ref fail_pipeline) = self.fail {
                fail_pipeline.run(ctx).await?;
            }
        } else {
            self.pass.run(ctx).await?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RouterStage — N-way named routing
// ---------------------------------------------------------------------------

/// Routing function type: inspects Context and returns a route name.
/// Returns `None` to use the default route (or skip if no default).
pub type RouteFn = Arc<dyn Fn(&Context) -> Option<String> + Send + Sync>;

/// N-way named routing combinator.
///
/// Inspects the [`Context`] via a routing function and dispatches to
/// the matching named sub-pipeline. Use for model selection, modality
/// routing, cost optimization, etc.
///
/// The selected sub-pipeline mutates the shared Context — any prior
/// `response` or run-scope extensions may be overwritten.
pub struct RouterStage {
    name: String,
    router: RouteFn,
    routes: HashMap<String, Arc<Pipeline>>,
    default_route: Option<String>,
}

impl RouterStage {
    pub fn new(
        name: impl Into<String>,
        router: impl Fn(&Context) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            router: Arc::new(router),
            routes: HashMap::new(),
            default_route: None,
        }
    }

    /// Register a named route with its sub-pipeline.
    pub fn route(mut self, name: impl Into<String>, pipeline: Pipeline) -> Self {
        self.routes.insert(name.into(), Arc::new(pipeline));
        self
    }

    /// Set the default route name (used when router returns None).
    pub fn default_route(mut self, name: impl Into<String>) -> Self {
        self.default_route = Some(name.into());
        self
    }
}

#[async_trait]
impl PipelineStage for RouterStage {
    fn name(&self) -> &str {
        &self.name
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let route_name = (self.router)(ctx).or_else(|| self.default_route.clone());

        let Some(name) = route_name else {
            // No route selected and no default — skip
            return Ok(());
        };

        let pipeline = self
            .routes
            .get(&name)
            .ok_or_else(|| MindroidError::pipeline(format!("unknown route: {name}")))?;

        pipeline.run(ctx).await?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RetryStage — configurable retry with exponential backoff
// ---------------------------------------------------------------------------

/// Wraps a [`PipelineStage`] with retry logic and exponential backoff.
///
/// On error, the stage is retried up to `max_retries` times with increasing
/// delays. Between retries, [`Context::reset_output`] is called to clear
/// run-scope state — **run-scope extensions from prior stages are lost**.
/// Session and global scopes are preserved.
///
/// If the wrapped stage depends on run-scope data from earlier stages,
/// wrap both in a sub-pipeline and retry the pipeline instead.
///
/// Retries are aborted immediately if [`Context::cancel`] is triggered.
pub struct RetryStage {
    name: String,
    inner: Box<dyn PipelineStage>,
    max_retries: u32,
    initial_delay: Duration,
    backoff_factor: f64,
}

impl RetryStage {
    /// Create a retry wrapper around `stage` with up to `max_retries` attempts.
    /// Default: 100ms initial delay, 2.0x exponential backoff.
    pub fn new(stage: impl PipelineStage + 'static, max_retries: u32) -> Self {
        Self {
            name: format!("retry({})", stage.name()),
            inner: Box::new(stage),
            max_retries,
            initial_delay: Duration::from_millis(100),
            backoff_factor: 2.0,
        }
    }

    /// Set the initial delay between retries.
    pub fn delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set the backoff multiplier (default 2.0 = exponential doubling).
    pub fn backoff(mut self, factor: f64) -> Self {
        self.backoff_factor = factor;
        self
    }
}

#[async_trait]
impl PipelineStage for RetryStage {
    fn name(&self) -> &str {
        &self.name
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let mut last_err = None;
        let mut delay = self.initial_delay;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                // Check cancellation before retrying
                if ctx.cancel.is_cancelled() {
                    return Err(last_err.unwrap_or_else(|| {
                        crate::error::MindroidError::pipeline("retry cancelled")
                    }));
                }
                ctx.reset_output();
                tokio::time::sleep(delay).await;
                delay = Duration::from_secs_f64(delay.as_secs_f64() * self.backoff_factor);
            }

            match self.inner.process(ctx).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!(
                        stage = self.inner.name(),
                        attempt = attempt + 1,
                        max = self.max_retries + 1,
                        error = %e,
                        "RetryStage: attempt failed"
                    );
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use async_trait::async_trait;

    use crate::config::AgentConfig;
    use crate::core::context::Context;
    use crate::models::Message;
    use crate::pipeline::{Pipeline, PipelineStage};

    use super::BranchStage;

    // ── Helpers ────────────────────────────────────────────────────────────────

    fn make_test_context() -> Context {
        let msg = Arc::new(Message::new("test", "user1", "ch1"));
        let config = Arc::new(AgentConfig::default());
        Context::new(msg, config)
    }

    /// Records whether it was called.
    struct RecorderStage {
        called: Arc<AtomicBool>,
    }

    #[async_trait]
    impl PipelineStage for RecorderStage {
        fn name(&self) -> &str {
            "recorder"
        }
        async fn process(&self, _ctx: &mut Context) -> crate::error::Result<()> {
            self.called.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    /// Gate that sets `ctx.halted = true`.
    struct HaltGate;

    #[async_trait]
    impl PipelineStage for HaltGate {
        fn name(&self) -> &str {
            "halt-gate"
        }
        async fn process(&self, ctx: &mut Context) -> crate::error::Result<()> {
            ctx.halted = true;
            Ok(())
        }
    }

    /// Gate that does nothing (pass-through).
    struct PassGate;

    #[async_trait]
    impl PipelineStage for PassGate {
        fn name(&self) -> &str {
            "pass-gate"
        }
        async fn process(&self, _ctx: &mut Context) -> crate::error::Result<()> {
            Ok(())
        }
    }

    // ── Tests ──────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_branch_pass() {
        let pass_called = Arc::new(AtomicBool::new(false));
        let fail_called = Arc::new(AtomicBool::new(false));

        let branch = BranchStage::new("branch", PassGate)
            .on_pass(
                Pipeline::new()
                    .add_stage(RecorderStage { called: pass_called.clone() }),
            )
            .on_fail(
                Pipeline::new()
                    .add_stage(RecorderStage { called: fail_called.clone() }),
            );

        let mut ctx = make_test_context();
        branch.process(&mut ctx).await.unwrap();

        assert!(pass_called.load(Ordering::SeqCst), "pass branch must run");
        assert!(!fail_called.load(Ordering::SeqCst), "fail branch must not run");
    }

    #[tokio::test]
    async fn test_branch_fail() {
        let pass_called = Arc::new(AtomicBool::new(false));
        let fail_called = Arc::new(AtomicBool::new(false));

        let branch = BranchStage::new("branch", HaltGate)
            .on_pass(
                Pipeline::new()
                    .add_stage(RecorderStage { called: pass_called.clone() }),
            )
            .on_fail(
                Pipeline::new()
                    .add_stage(RecorderStage { called: fail_called.clone() }),
            );

        let mut ctx = make_test_context();
        branch.process(&mut ctx).await.unwrap();

        assert!(!pass_called.load(Ordering::SeqCst), "pass branch must not run");
        assert!(fail_called.load(Ordering::SeqCst), "fail branch must run");
        // halted must be reset after BranchStage consumed it
        assert!(!ctx.halted, "halted must be cleared after branch");
    }

    #[tokio::test]
    async fn test_branch_no_fail() {
        // HaltGate fires but no fail pipeline is configured — should still be Ok
        let branch = BranchStage::new("branch", HaltGate);

        let mut ctx = make_test_context();
        let result = branch.process(&mut ctx).await;

        assert!(result.is_ok(), "branch without fail pipeline must return Ok");
        assert!(!ctx.halted, "halted must be cleared even when no fail pipeline");
    }

    #[tokio::test]
    async fn test_branch_cancellation() {
        let pass_called = Arc::new(AtomicBool::new(false));

        let branch = BranchStage::new("branch", PassGate).on_pass(
            Pipeline::new().add_stage(RecorderStage { called: pass_called.clone() }),
        );

        let mut ctx = make_test_context();
        // Cancel before running
        ctx.cancel.cancel();
        branch.process(&mut ctx).await.unwrap();

        // The sub-pipeline checks the cancel token and skips stages after the
        // first completed stage; since RecorderStage is the first (and only)
        // stage it will run once — but the pipeline respects cancellation for
        // subsequent stages.  What we really assert here is that the whole
        // thing returns Ok and does not panic.
        //
        // A pre-cancelled context still lets the first stage run (Pipeline::run
        // checks cancellation *after* each stage), so pass_called may be true.
        // The important invariant is: no panic, no error.
        let _ = pass_called.load(Ordering::SeqCst);
    }

    #[tokio::test]
    async fn test_branch_nested() {
        // BranchStage inside BranchStage
        let inner_pass_called = Arc::new(AtomicBool::new(false));
        let outer_pass_called = Arc::new(AtomicBool::new(false));

        let inner_branch = BranchStage::new("inner-branch", PassGate).on_pass(
            Pipeline::new()
                .add_stage(RecorderStage { called: inner_pass_called.clone() }),
        );

        let outer_branch = BranchStage::new("outer-branch", PassGate)
            .on_pass(
                Pipeline::new()
                    .add_stage(inner_branch)
                    .add_stage(RecorderStage { called: outer_pass_called.clone() }),
            );

        let mut ctx = make_test_context();
        outer_branch.process(&mut ctx).await.unwrap();

        assert!(
            inner_pass_called.load(Ordering::SeqCst),
            "inner branch pass must run"
        );
        assert!(
            outer_pass_called.load(Ordering::SeqCst),
            "outer pass recorder must run"
        );
    }

    use super::RouterStage;

    #[tokio::test]
    async fn test_router_dispatches() {
        let route_a = Arc::new(AtomicBool::new(false));
        let route_b = Arc::new(AtomicBool::new(false));

        let router = RouterStage::new("router", |_ctx| Some("route_a".into()))
            .route(
                "route_a",
                Pipeline::new().add_stage(RecorderStage {
                    called: route_a.clone(),
                }),
            )
            .route(
                "route_b",
                Pipeline::new().add_stage(RecorderStage {
                    called: route_b.clone(),
                }),
            );

        let mut ctx = make_test_context();
        router.process(&mut ctx).await.unwrap();

        assert!(route_a.load(Ordering::SeqCst), "route_a must run");
        assert!(!route_b.load(Ordering::SeqCst), "route_b must not run");
    }

    #[tokio::test]
    async fn test_router_default() {
        let default_called = Arc::new(AtomicBool::new(false));

        let router = RouterStage::new("router", |_ctx| None) // returns None
            .route(
                "fallback",
                Pipeline::new().add_stage(RecorderStage {
                    called: default_called.clone(),
                }),
            )
            .default_route("fallback");

        let mut ctx = make_test_context();
        router.process(&mut ctx).await.unwrap();

        assert!(
            default_called.load(Ordering::SeqCst),
            "default route must run"
        );
    }

    #[tokio::test]
    async fn test_router_unknown_error() {
        let router = RouterStage::new("router", |_ctx| Some("nonexistent".into()));

        let mut ctx = make_test_context();
        let result = router.process(&mut ctx).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("unknown route"),
            "error should mention unknown route: {err}"
        );
    }

    #[tokio::test]
    async fn test_router_cancellation() {
        let called = Arc::new(AtomicBool::new(false));

        let router = RouterStage::new("router", |_ctx| Some("main".into())).route(
            "main",
            Pipeline::new().add_stage(RecorderStage {
                called: called.clone(),
            }),
        );

        let mut ctx = make_test_context();
        ctx.cancel.cancel();
        router.process(&mut ctx).await.unwrap();
        // Same as BranchStage: cancellation is checked after stages, not before.
        // The important thing: no panic, returns Ok.
    }

    // ── RetryStage helpers & tests ─────────────────────────────────────────────

    use std::sync::atomic::AtomicU32;
    use std::time::Duration;
    use super::RetryStage;

    /// Stage that fails N times, then succeeds.
    struct FailNTimesStage {
        failures_remaining: Arc<AtomicU32>,
        called_count: Arc<AtomicU32>,
    }

    impl FailNTimesStage {
        fn new(fail_count: u32) -> Self {
            Self {
                failures_remaining: Arc::new(AtomicU32::new(fail_count)),
                called_count: Arc::new(AtomicU32::new(0)),
            }
        }
    }

    #[async_trait]
    impl PipelineStage for FailNTimesStage {
        fn name(&self) -> &str {
            "fail-n-times"
        }
        async fn process(&self, _ctx: &mut Context) -> crate::error::Result<()> {
            self.called_count.fetch_add(1, Ordering::SeqCst);
            let remaining = self.failures_remaining.fetch_sub(1, Ordering::SeqCst);
            if remaining > 0 {
                Err(crate::error::MindroidError::pipeline("intentional failure"))
            } else {
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn test_retry_succeeds_first_try() {
        let called = Arc::new(AtomicBool::new(false));
        let retry = RetryStage::new(RecorderStage { called: called.clone() }, 3)
            .delay(Duration::from_millis(1));

        let mut ctx = make_test_context();
        retry.process(&mut ctx).await.unwrap();
        assert!(called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_retry_succeeds_after_failures() {
        let stage = FailNTimesStage::new(2); // fail twice, succeed third
        let call_count = stage.called_count.clone();
        let retry = RetryStage::new(stage, 3).delay(Duration::from_millis(1));

        let mut ctx = make_test_context();
        retry.process(&mut ctx).await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 3); // 2 fails + 1 success
    }

    #[tokio::test]
    async fn test_retry_exhausted() {
        let stage = FailNTimesStage::new(10); // always fails
        let retry = RetryStage::new(stage, 2).delay(Duration::from_millis(1));

        let mut ctx = make_test_context();
        let result = retry.process(&mut ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retry_respects_cancellation() {
        let stage = FailNTimesStage::new(10); // always fails
        let call_count = stage.called_count.clone();
        let retry = RetryStage::new(stage, 5).delay(Duration::from_millis(1));

        let mut ctx = make_test_context();
        ctx.cancel.cancel(); // pre-cancel
        // First attempt runs (cancellation checked BEFORE retry, not before first attempt)
        let result = retry.process(&mut ctx).await;
        assert!(result.is_err());
        // Should have tried only once (first attempt) then bailed on retry
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_resets_output() {
        // Set a response, verify it's cleared between retries
        let stage = FailNTimesStage::new(1); // fail once, succeed second
        let retry = RetryStage::new(stage, 2).delay(Duration::from_millis(1));

        let mut ctx = make_test_context();
        ctx.response = Some("stale".into());
        retry.process(&mut ctx).await.unwrap();
        // After retry succeeds, the stale response should have been cleared by reset_output
        // (the FailNTimesStage doesn't set response, so it stays None after reset)
        assert!(ctx.response.is_none());
    }
}
