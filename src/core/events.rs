use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::core::agent_loop::StopReason;

/// System-level pipeline events for observability, metrics, and debugging.
///
/// Separate from the `Observer` trait (which is user-facing lifecycle hooks).
/// Events are sent via an opt-in `mpsc::UnboundedSender` on `Context` —
/// if nobody subscribes, zero cost.
///
/// `#[non_exhaustive]`: new execution models add events, so a `match` outside
/// this crate needs a wildcard arm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum PipelineEvent {
    PipelineStarted {
        stage_count: usize,
    },
    StageStarted {
        stage_name: String,
        stage_index: usize,
    },
    StageCompleted {
        stage_name: String,
        stage_index: usize,
        #[serde(with = "duration_millis")]
        elapsed: Duration,
    },
    StageError {
        stage_name: String,
        error: String,
    },
    Cancelled {
        stage_name: String,
    },
    PipelineCompleted {
        #[serde(with = "duration_millis")]
        elapsed: Duration,
    },
    /// One [`AgentLoop`](crate::core::agent_loop::AgentLoop) body pass is
    /// starting. Emitted around the pass, not by it, so a body with no
    /// loop-aware stage still reports its single iteration.
    LoopIterationStarted {
        /// The loop's [`with_name`](crate::core::agent_loop::AgentLoop::with_name),
        /// so nested loops' events can be told apart.
        loop_name: String,
        iteration: usize,
    },
    /// The loop finished. `finish` ran first unless `reason` is `Cancelled` or
    /// `Refused`, which skip it, so do not read this as "the turn was
    /// persisted". `reason` is why — the only way a
    /// [`run_streaming`](crate::core::agent_loop::AgentLoop::run_streaming)
    /// caller learns it, since that returns events, not a `LoopOutcome`.
    LoopCompleted {
        loop_name: String,
        iterations: usize,
        #[serde(with = "duration_millis")]
        elapsed: Duration,
        reason: StopReason,
    },
}

/// Serde helper: serialize Duration as milliseconds (u64).
mod duration_millis {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        d.as_millis().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let ms = u64::deserialize(d)?;
        Ok(Duration::from_millis(ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use crate::core::context::Context;
    use crate::models::Message;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    fn make_test_context() -> Context {
        let msg = Arc::new(Message::new("test", "user1", "channel1"));
        let config = Arc::new(AgentConfig::default());
        Context::new(msg, config)
    }

    #[tokio::test]
    async fn test_emit_event_with_receiver() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let ctx = make_test_context().with_events(tx);

        ctx.emit_event(PipelineEvent::PipelineStarted { stage_count: 3 });

        let event = rx.recv().await.unwrap();
        assert!(matches!(
            event,
            PipelineEvent::PipelineStarted { stage_count: 3 }
        ));
    }

    #[test]
    fn test_emit_event_no_receiver() {
        // No events configured — emit should not panic
        let ctx = make_test_context();
        ctx.emit_event(PipelineEvent::PipelineStarted { stage_count: 3 });
    }

    #[tokio::test]
    async fn test_emit_event_dropped_receiver() {
        let (tx, rx) = mpsc::unbounded_channel();
        let ctx = make_test_context().with_events(tx);

        // Drop the receiver
        drop(rx);

        // Should not panic
        ctx.emit_event(PipelineEvent::PipelineStarted { stage_count: 3 });
    }

    #[test]
    fn test_serde_roundtrip() {
        let events = vec![
            PipelineEvent::PipelineStarted { stage_count: 3 },
            PipelineEvent::StageStarted {
                stage_name: "ctx_builder".into(),
                stage_index: 0,
            },
            PipelineEvent::StageCompleted {
                stage_name: "ctx_builder".into(),
                stage_index: 0,
                elapsed: Duration::from_millis(42),
            },
            PipelineEvent::StageError {
                stage_name: "llm".into(),
                error: "timeout".into(),
            },
            PipelineEvent::Cancelled {
                stage_name: "post".into(),
            },
            PipelineEvent::PipelineCompleted {
                elapsed: Duration::from_millis(100),
            },
            PipelineEvent::LoopIterationStarted {
                loop_name: "planner".into(),
                iteration: 2,
            },
            PipelineEvent::LoopCompleted {
                loop_name: "planner".into(),
                iterations: 3,
                elapsed: Duration::from_millis(250),
                reason: StopReason::MaxIterations,
            },
        ];

        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            let deserialized: PipelineEvent = serde_json::from_str(&json).unwrap();
            // Just verify it roundtrips without panic
            let _ = format!("{:?}", deserialized);
        }
    }

    /// The stop reason is wire-visible, so its spelling is part of the event
    /// format a dashboard parses.
    #[test]
    fn loop_completed_serializes_its_reason_in_snake_case() {
        let json = serde_json::to_value(PipelineEvent::LoopCompleted {
            loop_name: "planner".into(),
            iterations: 3,
            elapsed: Duration::from_millis(250),
            reason: StopReason::MaxIterations,
        })
        .unwrap();

        assert_eq!(json["event"], "loop_completed");
        assert_eq!(json["loop_name"], "planner");
        assert_eq!(json["reason"], "max_iterations");
        assert_eq!(json["elapsed"], 250);
    }
}
