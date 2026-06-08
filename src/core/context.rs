use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::config::AgentConfig;
use crate::core::events::PipelineEvent;
use crate::core::extension_map::{ExtensionMap, SharedExtensionMap};
use crate::models::{LlmMessage, Message};

/// Unified pipeline context with scoped state (run → session → global).
///
/// Replaces the former `PipelineContext`. State is resolved inner-to-outer:
/// `get::<T>()` checks run scope first, then session, then global.
///
/// - **Run scope** (`ExtensionMap`): per-pipeline-run, ephemeral, owned.
/// - **Session scope** (`SharedExtensionMap`): per-connection, persistent, shared.
/// - **Global scope** (`SharedExtensionMap`): server-wide, shared.
pub struct Context {
    // -- Preserved from PipelineContext --
    /// The incoming message being processed.
    pub message: Arc<Message>,
    /// Agent configuration.
    pub agent_config: Arc<AgentConfig>,
    /// LLM conversation messages, built by ContextBuilder stage.
    pub llm_messages: Vec<LlmMessage>,
    /// Response text produced by the pipeline.
    pub response: Option<String>,
    /// When set to `true`, the pipeline stops executing further stages.
    pub halted: bool,

    // -- New: scoped state --
    /// Run-scoped extension map (per pipeline run, ephemeral).
    pub(crate) run: ExtensionMap,
    /// Session-scoped extension map (per connection, persistent, shared).
    session: Option<SharedExtensionMap>,
    /// Global-scoped extension map (server-wide, shared).
    global: Option<SharedExtensionMap>,

    // -- New: cancellation --
    /// Cancellation token for cooperative pipeline cancellation.
    pub cancel: CancellationToken,

    /// Optional pipeline event sender for system-level observability.
    events: Option<mpsc::UnboundedSender<PipelineEvent>>,
}

impl Context {
    /// Create a new context (matching old PipelineContext::new signature).
    pub fn new(message: Arc<Message>, agent_config: Arc<AgentConfig>) -> Self {
        Self {
            message,
            agent_config,
            llm_messages: Vec::new(),
            response: None,
            halted: false,
            run: ExtensionMap::default(),
            session: None,
            global: None,
            cancel: CancellationToken::new(),
            events: None,
        }
    }

    // -- Builders --

    /// Attach a session-scoped extension map.
    pub fn with_session(mut self, session: SharedExtensionMap) -> Self {
        self.session = Some(session);
        self
    }

    /// Attach a global-scoped extension map.
    pub fn with_global(mut self, global: SharedExtensionMap) -> Self {
        self.global = Some(global);
        self
    }

    /// Attach a cancellation token.
    pub fn with_cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Attach a pipeline event sender for system-level observability.
    pub fn with_events(mut self, tx: mpsc::UnboundedSender<PipelineEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    /// Emit a pipeline event. Fire-and-forget — never panics even if the
    /// receiver has been dropped or events are not configured.
    pub fn emit_event(&self, event: PipelineEvent) {
        if let Some(ref tx) = self.events {
            let _ = tx.send(event);
        }
    }

    /// Return a reference to the underlying event sender, if configured.
    ///
    /// Useful when `ctx` is mutably borrowed (e.g. inside a streaming borrow)
    /// and you still need to send events via a cloned sender.
    pub fn events_sender(&self) -> Option<&mpsc::UnboundedSender<PipelineEvent>> {
        self.events.as_ref()
    }

    // -- Scoped accessors (run → session → global) --

    /// Get a value by type, searching run → session → global.
    /// Run scope returns a borrow; session/global return clones.
    ///
    /// For T: Clone, checks run first (borrow + clone), then session, then global.
    pub fn get<T: Clone + Send + Sync + 'static>(&self) -> Option<T> {
        // Run scope: borrow → clone
        if let Some(val) = self.run.get::<T>() {
            return Some(val.clone());
        }
        // Session scope
        if let Some(ref session) = self.session
            && let Some(val) = session.get::<T>()
        {
            return Some(val);
        }
        // Global scope
        if let Some(ref global) = self.global
            && let Some(val) = global.get::<T>()
        {
            return Some(val);
        }
        None
    }

    /// Get a reference from run scope only (no clone needed).
    pub fn get_run<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.run.get::<T>()
    }

    /// Set a value in the run scope (default, ephemeral).
    pub fn set<T: Send + Sync + 'static>(&mut self, value: T) {
        self.run.set(value);
    }

    /// Set a value in the session scope (persistent, shared).
    pub fn set_session<T: Clone + Send + Sync + 'static>(&self, value: T) {
        if let Some(ref session) = self.session {
            session.set(value);
        }
    }

    /// Set a value in the global scope (server-wide, shared).
    pub fn set_global<T: Clone + Send + Sync + 'static>(&self, value: T) {
        if let Some(ref global) = self.global {
            global.set(value);
        }
    }

    /// Take (remove + return) a value from the run scope.
    pub fn take<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.run.take::<T>()
    }

    // -- Backward compat (delegate to run scope) --

    /// Backward-compatible: set a value in run scope.
    pub fn set_ext<T: Send + Sync + 'static>(&mut self, value: T) {
        self.run.set(value);
    }

    /// Backward-compatible: get a reference from run scope.
    pub fn get_ext<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.run.get::<T>()
    }

    /// Backward-compatible: take a value from run scope.
    pub fn take_ext<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.run.take::<T>()
    }

    /// Clear output fields for reuse across pipeline runs.
    /// Preserves message and agent_config.
    /// **Clears run scope only** — session and global state persists.
    pub fn reset_output(&mut self) {
        self.llm_messages.clear();
        self.response = None;
        self.halted = false;
        self.run.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_context() -> Context {
        let msg = Arc::new(Message::new("test", "user1", "channel1"));
        let config = Arc::new(AgentConfig::default());
        Context::new(msg, config)
    }

    #[test]
    fn test_context_scope_resolution() {
        let session = SharedExtensionMap::new();
        session.set(100u32); // session has 100
        let mut ctx = make_test_context().with_session(session);
        ctx.set(42u32); // run has 42

        // get::<T>() should find run scope first (42, not 100)
        assert_eq!(ctx.get::<u32>(), Some(42u32));
    }

    #[test]
    fn test_context_session_fallback() {
        let session = SharedExtensionMap::new();
        session.set(100u32);
        let ctx = make_test_context().with_session(session);

        // Nothing in run scope, should find session
        assert_eq!(ctx.get::<u32>(), Some(100u32));
    }

    #[test]
    fn test_context_backward_compat() {
        let mut ctx = make_test_context();
        ctx.set_ext(42u32);
        assert_eq!(ctx.get_ext::<u32>(), Some(&42u32));
        assert_eq!(ctx.take_ext::<u32>(), Some(42u32));
        assert_eq!(ctx.get_ext::<u32>(), None);
    }

    #[test]
    fn test_reset_output_clears_run_only() {
        let session = SharedExtensionMap::new();
        session.set(100u32);
        let mut ctx = make_test_context().with_session(session);
        ctx.set(42u32); // run scope
        ctx.llm_messages.push(LlmMessage::user("hello"));
        ctx.response = Some("world".into());

        ctx.reset_output();

        // Run scope cleared
        assert_eq!(ctx.get_run::<u32>(), None);
        assert!(ctx.llm_messages.is_empty());
        assert!(ctx.response.is_none());
        assert!(!ctx.halted);

        // Session scope survives
        assert_eq!(ctx.get::<u32>(), Some(100u32));
    }
}
