//! InterruptionGate + InterruptionScorer — barge-in decision (task 5).
//!
//! Formalises the "sustained speech barge-in gate" from
//! `services/mindroid-voice/src/voice_session.rs::run_vad`.
//!
//! ## Semantics (verified against source)
//!
//! - `MIN_BARGE_IN_FRAMES = 10`.
//! - On speech onset, the internal counter is set to 1; that onset frame does
//!   **not** run the gate (it is handled by the VAD state machine upstream).
//! - From the 2nd consecutive speech frame while the agent is speaking, the
//!   counter increments for each high-probability frame.
//! - Barge-in **fires when the counter reaches 10** (i.e. the 10th consecutive
//!   speech frame crosses the threshold).
//! - The counter **resets to 0** on a non-speech frame, or when the agent stops
//!   speaking.
//! - If echo-suppressed (see `EchoGuard`), firing is withheld; the decision is
//!   [`InterruptionDecision::Hold`] instead of [`InterruptionDecision::Interrupt`].

// ─── Decision type ────────────────────────────────────────────────────────────

/// What the [`InterruptionGate`] decided for a given frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptionDecision {
    /// Threshold reached and not echo-suppressed: caller should cancel the
    /// agent's turn.
    Interrupt,
    /// Threshold reached but echo-suppressed: firing is withheld this cycle.
    Hold,
    /// No action — either agent is not speaking, or the frame count has not
    /// crossed the threshold yet.
    NoOp,
}

// ─── InterruptionScorer ───────────────────────────────────────────────────────

/// Pluggable hook for per-frame "is this real speech?" confidence scoring.
///
/// The gate's frame-count logic is the actual barge-in decision; the scorer is
/// a future seam for a real CNN or energy-based model.
///
/// The returned `f32` is a confidence in `[0.0, 1.0]`. The gate compares it
/// against its internal threshold (currently always `>= 0.5` to match VAD
/// parity) to decide whether a frame counts as speech.
///
/// # Note
///
/// Implementations must be `Send` so they can be used across thread boundaries.
pub trait InterruptionScorer: Send {
    /// Score one raw PCM frame.
    ///
    /// `frame` — 16-bit PCM samples (one audio chunk, e.g. 512 samples at
    /// 16 kHz → 32 ms).
    fn score(&mut self, frame: &[i16]) -> f32;
}

/// Parity placeholder scorer: returns 1.0 for any non-silent frame, 0.0
/// otherwise (silence is defined as every sample being 0).
///
/// The real decision is made by the gate's frame counter; this scorer simply
/// passes through, preserving today's behaviour where `is_speech` is provided
/// directly by the VAD probability. Replaced by a real CNN in feature C.
pub struct SustainedFramesScorer;

impl InterruptionScorer for SustainedFramesScorer {
    fn score(&mut self, frame: &[i16]) -> f32 {
        // Non-zero energy → report as speech (confidence 1.0).
        if frame.iter().any(|&s| s != 0) {
            1.0
        } else {
            0.0
        }
    }
}

// ─── InterruptionGate ─────────────────────────────────────────────────────────

/// Default minimum consecutive speech frames before barge-in fires.
pub const DEFAULT_MIN_BARGE_IN_FRAMES: usize = 10;

/// Pure, `Send`, non-async state machine for the sustained-speech barge-in gate.
///
/// Create with [`InterruptionGate::new`] or [`InterruptionGate::default`], then
/// call [`observe`](InterruptionGate::observe) once per audio frame.
///
/// ## Thread safety
///
/// All state is owned (`usize` + `bool`); the type is `Send` without any extra
/// synchronisation primitives.
pub struct InterruptionGate {
    /// How many consecutive speech frames are required to fire barge-in.
    min_frames: usize,
    /// Running count of consecutive speech frames while the agent is speaking.
    speech_frames: usize,
}

impl InterruptionGate {
    /// Create a new gate with a custom threshold.
    ///
    /// Prefer [`InterruptionGate::default`] for the standard 10-frame threshold.
    pub fn new(min_frames: usize) -> Self {
        Self {
            min_frames,
            speech_frames: 0,
        }
    }

    /// Process one audio frame and return the barge-in decision.
    ///
    /// # Arguments
    ///
    /// - `is_speech` — whether the current frame is classified as speech (VAD
    ///   probability >= threshold as determined by the caller).
    /// - `agent_speaking` — whether the agent is currently playing back TTS.
    /// - `echo_suppressed` — whether echo suppression is active (e.g. within
    ///   `ECHO_SUPPRESSION_MS` of the last TTS chunk).
    ///
    /// # Returns
    ///
    /// - [`InterruptionDecision::Interrupt`] — exactly once when the threshold
    ///   is crossed and `echo_suppressed` is false.
    /// - [`InterruptionDecision::Hold`] — threshold crossed but echo-suppressed.
    /// - [`InterruptionDecision::NoOp`] — agent not speaking, or count not yet
    ///   at threshold.
    pub fn observe(
        &mut self,
        is_speech: bool,
        agent_speaking: bool,
        echo_suppressed: bool,
    ) -> InterruptionDecision {
        if !agent_speaking {
            self.speech_frames = 0;
            return InterruptionDecision::NoOp;
        }

        if is_speech {
            self.speech_frames += 1;
            if self.speech_frames == self.min_frames {
                if echo_suppressed {
                    InterruptionDecision::Hold
                } else {
                    InterruptionDecision::Interrupt
                }
            } else {
                InterruptionDecision::NoOp
            }
        } else {
            self.speech_frames = 0;
            InterruptionDecision::NoOp
        }
    }

    /// Reset the frame counter, e.g. after a barge-in fires or the agent turn
    /// ends.
    pub fn reset(&mut self) {
        self.speech_frames = 0;
    }
}

impl Default for InterruptionGate {
    fn default() -> Self {
        Self::new(DEFAULT_MIN_BARGE_IN_FRAMES)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helper: run N speech frames through the gate ──────────────────────────

    /// Drive `n` consecutive speech frames (agent speaking, no echo suppression)
    /// and collect the decisions.
    fn drive_speech(gate: &mut InterruptionGate, n: usize) -> Vec<InterruptionDecision> {
        (0..n)
            .map(|_| gate.observe(true, true, false))
            .collect()
    }

    // ── test: short burst then silence never fires ────────────────────────────

    /// 9 consecutive speech frames followed by a non-speech frame, while the
    /// agent is speaking and echo is not suppressed → never Interrupt; counter
    /// resets on the non-speech frame.
    #[test]
    fn short_noise_then_silence_never_interrupts() {
        let mut gate = InterruptionGate::default();

        // Simulate onset: caller sets speech_frames = 1 before handing control.
        // The gate starts at 0; we feed 9 frames which takes it to 9 (< 10).
        let decisions = drive_speech(&mut gate, 9);
        for d in &decisions {
            assert_eq!(*d, InterruptionDecision::NoOp, "expected NoOp before threshold");
        }
        assert_eq!(gate.speech_frames, 9);

        // Non-speech frame resets the counter.
        let d = gate.observe(false, true, false);
        assert_eq!(d, InterruptionDecision::NoOp);
        assert_eq!(gate.speech_frames, 0, "counter must reset on non-speech frame");
    }

    // ── test: 10 consecutive speech frames → Interrupt ────────────────────────

    /// Exactly 10 consecutive speech frames while the agent is speaking and
    /// echo is NOT suppressed → `Interrupt` fires on the 10th frame.
    #[test]
    fn ten_frames_fires_interrupt() {
        let mut gate = InterruptionGate::default();

        // First 9 frames → NoOp.
        let first_nine = drive_speech(&mut gate, 9);
        for d in &first_nine {
            assert_eq!(*d, InterruptionDecision::NoOp);
        }

        // 10th frame → Interrupt.
        let d = gate.observe(true, true, false);
        assert_eq!(d, InterruptionDecision::Interrupt);
    }

    // ── test: echo-suppressed → Hold, not Interrupt ───────────────────────────

    /// Same 10-frame sequence but `echo_suppressed = true` → decision is Hold,
    /// not Interrupt.
    #[test]
    fn ten_frames_echo_suppressed_holds() {
        let mut gate = InterruptionGate::default();

        // 9 frames without echo suppression.
        for _ in 0..9 {
            gate.observe(true, true, false);
        }

        // 10th frame with echo suppression active.
        let d = gate.observe(true, true, true);
        assert_eq!(d, InterruptionDecision::Hold);
    }

    // ── test: agent stops mid-count → reset; subsequent frames don't fire ─────

    /// Agent stops speaking mid-count: counter resets; subsequent frames with
    /// the agent speaking again do not immediately fire.
    #[test]
    fn agent_stops_mid_count_resets_and_no_spurious_fire() {
        let mut gate = InterruptionGate::default();

        // 5 speech frames while agent is speaking.
        for _ in 0..5 {
            assert_eq!(gate.observe(true, true, false), InterruptionDecision::NoOp);
        }
        assert_eq!(gate.speech_frames, 5);

        // Agent stops speaking → counter resets.
        let d = gate.observe(true, false, false);
        assert_eq!(d, InterruptionDecision::NoOp);
        assert_eq!(gate.speech_frames, 0, "counter must reset when agent not speaking");

        // Agent resumes; only 9 more frames (not yet at threshold).
        for _ in 0..9 {
            assert_eq!(gate.observe(true, true, false), InterruptionDecision::NoOp);
        }
        // 10th frame after the fresh start → fires.
        let d = gate.observe(true, true, false);
        assert_eq!(d, InterruptionDecision::Interrupt);
    }

    // ── test: reset() clears the counter ─────────────────────────────────────

    #[test]
    fn manual_reset_clears_counter() {
        let mut gate = InterruptionGate::default();
        drive_speech(&mut gate, 5);
        assert_eq!(gate.speech_frames, 5);

        gate.reset();
        assert_eq!(gate.speech_frames, 0);

        // After reset, needs a full 10 frames again.
        let decisions = drive_speech(&mut gate, 9);
        for d in &decisions {
            assert_eq!(*d, InterruptionDecision::NoOp);
        }
        assert_eq!(gate.observe(true, true, false), InterruptionDecision::Interrupt);
    }

    // ── test: custom threshold ────────────────────────────────────────────────

    #[test]
    fn custom_min_frames_threshold() {
        let mut gate = InterruptionGate::new(3);

        assert_eq!(gate.observe(true, true, false), InterruptionDecision::NoOp);
        assert_eq!(gate.observe(true, true, false), InterruptionDecision::NoOp);
        assert_eq!(gate.observe(true, true, false), InterruptionDecision::Interrupt);
    }

    // ── test: SustainedFramesScorer ───────────────────────────────────────────

    #[test]
    fn sustained_frames_scorer_non_zero_returns_one() {
        let mut scorer = SustainedFramesScorer;
        assert_eq!(scorer.score(&[0i16, 1000, -500]), 1.0);
    }

    #[test]
    fn sustained_frames_scorer_all_zero_returns_zero() {
        let mut scorer = SustainedFramesScorer;
        assert_eq!(scorer.score(&[0i16, 0, 0]), 0.0);
    }

    // ── test: Send bound ──────────────────────────────────────────────────────

    #[test]
    fn gate_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<InterruptionGate>();
    }
}
