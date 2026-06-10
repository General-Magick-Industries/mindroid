//! EchoGuard — suppression gate for the agent's own playback (task 4).
//!
//! Decides when to suppress barge-in because the agent is hearing its own TTS
//! echo.  Mirrors the policy in `voice_session.rs::run_vad`: suppress while the
//! agent is speaking AND within `suppression_window` of the last TTS chunk.
//!
//! # Design
//!
//! Pure, `Send`, non-async state machine.  The caller passes a monotonic
//! `std::time::Instant` — no `Instant::now()` is called internally, so the
//! machine is fully deterministic and easy to unit-test.

use std::time::{Duration, Instant};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Default echo-suppression window (matches `ECHO_SUPPRESSION_MS` in
/// `voice_session.rs`).
pub const DEFAULT_SUPPRESSION_MS: u64 = 250;

// ─── EchoGuard ────────────────────────────────────────────────────────────────

/// Pure state machine that decides whether barge-in should be suppressed.
///
/// Suppression is active when **both** conditions hold:
/// 1. The agent is currently speaking (set via [`set_agent_speaking`] /
///    [`notify_agent_stopped`]).
/// 2. `now - last_tts_chunk < suppression_window`.
///
/// [`set_agent_speaking`]: EchoGuard::set_agent_speaking
/// [`notify_agent_stopped`]: EchoGuard::notify_agent_stopped
#[derive(Debug)]
pub struct EchoGuard {
    /// How long after the last TTS chunk to keep suppression active.
    suppression_window: Duration,
    /// Whether the agent is currently playing back TTS audio.
    agent_speaking: bool,
    /// Timestamp of the most recent TTS chunk, if any.
    last_tts_chunk: Option<Instant>,
}

impl EchoGuard {
    /// Create a new `EchoGuard` with a custom suppression window.
    pub fn new(suppression_window: Duration) -> Self {
        Self {
            suppression_window,
            agent_speaking: false,
            last_tts_chunk: None,
        }
    }

    /// Create a new `EchoGuard` using the default 250 ms suppression window.
    pub fn default_window() -> Self {
        Self::new(Duration::from_millis(DEFAULT_SUPPRESSION_MS))
    }

    /// Mark the agent as speaking or not speaking.
    ///
    /// Prefer [`notify_agent_stopped`](Self::notify_agent_stopped) to clear the
    /// speaking flag — it is more expressive at call sites.
    pub fn set_agent_speaking(&mut self, speaking: bool) {
        self.agent_speaking = speaking;
    }

    /// Record that a TTS audio chunk was just sent at `now`.
    ///
    /// Also implicitly sets the agent-speaking flag to `true`.
    pub fn notify_tts_chunk(&mut self, now: Instant) {
        self.last_tts_chunk = Some(now);
        self.agent_speaking = true;
    }

    /// Record that the agent has finished speaking.
    ///
    /// Clears the speaking flag so [`is_suppressed`](Self::is_suppressed)
    /// returns `false` regardless of timing.
    pub fn notify_agent_stopped(&mut self) {
        self.agent_speaking = false;
    }

    /// Returns `true` iff barge-in should be suppressed at timestamp `now`.
    ///
    /// Suppression requires:
    /// - `agent_speaking == true`, **and**
    /// - a TTS chunk was recorded, **and**
    /// - `now - last_tts_chunk < suppression_window`.
    pub fn is_suppressed(&self, now: Instant) -> bool {
        if !self.agent_speaking {
            return false;
        }
        match self.last_tts_chunk {
            None => false,
            Some(last) => now.duration_since(last) < self.suppression_window,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Helper: construct a base instant and produce offsets from it.
    fn base() -> Instant {
        Instant::now()
    }

    #[test]
    fn suppress_within_window() {
        // Agent speaking, last TTS chunk 200 ms ago (< 250 ms window) → suppressed.
        let t0 = base();
        let mut guard = EchoGuard::default_window();

        guard.notify_tts_chunk(t0);
        // Simulate 200 ms elapsing.
        let now = t0 + Duration::from_millis(200);
        assert!(
            guard.is_suppressed(now),
            "should be suppressed within window"
        );
    }

    #[test]
    fn allow_after_window() {
        // Last TTS chunk 300 ms ago (> 250 ms window) → not suppressed.
        let t0 = base();
        let mut guard = EchoGuard::default_window();

        guard.notify_tts_chunk(t0);
        let now = t0 + Duration::from_millis(300);
        assert!(
            !guard.is_suppressed(now),
            "should allow after window expires"
        );
    }

    #[test]
    fn allow_when_agent_idle() {
        // Agent not speaking → never suppressed, regardless of timing.
        let t0 = base();
        let mut guard = EchoGuard::default_window();

        // Record a TTS chunk but then stop the agent.
        guard.notify_tts_chunk(t0);
        guard.notify_agent_stopped();

        // Even immediately after the chunk, not suppressed.
        let now = t0 + Duration::from_millis(10);
        assert!(
            !guard.is_suppressed(now),
            "should not suppress when agent idle"
        );
    }

    #[test]
    fn notify_agent_stopped_clears_suppression() {
        let t0 = base();
        let mut guard = EchoGuard::default_window();

        guard.notify_tts_chunk(t0);
        let now = t0 + Duration::from_millis(50);
        // Confirm suppressed before stopping.
        assert!(guard.is_suppressed(now));

        guard.notify_agent_stopped();
        // After stop, no longer suppressed even though window hasn't elapsed.
        assert!(
            !guard.is_suppressed(now),
            "notify_agent_stopped should clear suppression"
        );
    }

    #[test]
    fn no_tts_chunk_not_suppressed() {
        // Speaking flag set manually but no TTS chunk ever recorded → not suppressed.
        let t0 = base();
        let mut guard = EchoGuard::default_window();

        guard.set_agent_speaking(true);
        let now = t0 + Duration::from_millis(10);
        assert!(
            !guard.is_suppressed(now),
            "no tts chunk means no suppression"
        );
    }

    #[test]
    fn exact_window_boundary_not_suppressed() {
        // Duration exactly equal to the window is NOT suppressed (strict <).
        let t0 = base();
        let mut guard = EchoGuard::default_window();

        guard.notify_tts_chunk(t0);
        let now = t0 + Duration::from_millis(250);
        assert!(
            !guard.is_suppressed(now),
            "at exactly window boundary should not suppress"
        );
    }
}
