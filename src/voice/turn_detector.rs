//! TurnDetector trait + silence-timer default impl — endpointing (task 6).
//!
//! # Design
//!
//! [`TurnDetector`] is a thin trait over the [`VadDecision`] stream emitted by
//! [`crate::voice::vad::VadStateMachine`].  It exists so a future smart-turn v3
//! model (feature C) can slot in as an alternate impl without touching the
//! session machinery.
//!
//! Today's endpointing behaviour is reproduced by [`SilenceTimerTurnDetector`]:
//! it simply maps `VadDecision::SpeechEnded → TurnDecision::Complete` and every
//! other decision → `TurnDecision::Pending`.  The actual silence-duration timer
//! lives in `VadStateMachine`/`VadConfig`; this layer does not re-implement it.

use std::time::Instant;

use crate::voice::vad::VadDecision;

// ─── TurnDecision ─────────────────────────────────────────────────────────────

/// Outcome returned by [`TurnDetector::observe`] for each audio frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDecision {
    /// The user has not yet finished their utterance.
    Pending,
    /// The user's turn is complete; the session should endpoint.
    Complete,
}

// ─── TurnDetector trait ───────────────────────────────────────────────────────

/// Decides when a speaker's turn ends.
///
/// Implementors receive [`VadDecision`] events from the VAD state machine and
/// return a [`TurnDecision`].  The `now` timestamp is threaded through for
/// implementations that need it (e.g. min/max endpointing-delay modulation,
/// transcript-latency compensation, smart-turn v3).  Implementors **must not**
/// call `Instant::now()` internally; use the provided `now` value.
pub trait TurnDetector {
    /// Observe one VAD frame and decide whether the turn is complete.
    fn observe(&mut self, vad: VadDecision, now: Instant) -> TurnDecision;
}

// ─── SilenceTimerTurnDetector ─────────────────────────────────────────────────

/// Default [`TurnDetector`] that mirrors today's silence-based endpointing.
///
/// Converts `VadDecision::SpeechEnded` → `TurnDecision::Complete` and treats
/// every other VAD event as `TurnDecision::Pending`.  The silence threshold is
/// owned by `VadStateMachine`/`VadConfig`; this struct holds no timers itself.
///
/// # Future work
///
/// Smart-turn v3 (feature C) will be an alternate impl of [`TurnDetector`] that
/// inspects audio features and/or a partial transcript to end turns earlier or
/// later than pure silence detection.
#[derive(Debug, Default)]
pub struct SilenceTimerTurnDetector;

impl SilenceTimerTurnDetector {
    /// Create a new instance.
    pub fn new() -> Self {
        Self
    }
}

impl TurnDetector for SilenceTimerTurnDetector {
    /// Map `SpeechEnded` → `Complete`; everything else → `Pending`.
    fn observe(&mut self, vad: VadDecision, _now: Instant) -> TurnDecision {
        match vad {
            VadDecision::SpeechEnded => TurnDecision::Complete,
            VadDecision::SpeechStarted | VadDecision::SpeechContinues | VadDecision::Silence => {
                TurnDecision::Pending
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> Instant {
        Instant::now()
    }

    #[test]
    fn speech_started_and_continues_are_pending() {
        let mut td = SilenceTimerTurnDetector::new();
        assert_eq!(
            td.observe(VadDecision::SpeechStarted, now()),
            TurnDecision::Pending
        );
        assert_eq!(
            td.observe(VadDecision::SpeechContinues, now()),
            TurnDecision::Pending
        );
    }

    #[test]
    fn speech_ended_is_complete() {
        let mut td = SilenceTimerTurnDetector::new();
        assert_eq!(
            td.observe(VadDecision::SpeechEnded, now()),
            TurnDecision::Complete
        );
    }

    #[test]
    fn silence_in_idle_is_pending() {
        let mut td = SilenceTimerTurnDetector::new();
        assert_eq!(
            td.observe(VadDecision::Silence, now()),
            TurnDecision::Pending
        );
    }
}
