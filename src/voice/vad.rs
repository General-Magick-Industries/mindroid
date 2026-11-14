//! VAD (Voice Activity Detection) state machine.
//!
//! # Components
//!
//! - [`VadStateMachine`]: pure logic that processes voice probability floats and
//!   emits [`VadDecision`]s. Always available (no feature gate).

use std::time::Duration;

use crate::voice::types::VadConfig;

// ─── State & Decision types ───────────────────────────────────────────────────

/// Current state of the VAD state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    /// No speech detected; waiting for voice activity.
    Idle,
    /// Speech is in progress.
    Speaking,
}

/// Decision emitted by [`VadStateMachine::process`] for each audio frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadDecision {
    /// Speech just started (transition from Idle → Speaking).
    SpeechStarted,
    /// Speech is ongoing (already Speaking, not yet ended).
    SpeechContinues,
    /// Speech just ended (enough consecutive silence detected).
    SpeechEnded,
    /// Silence in Idle state — no change.
    Silence,
}

// ─── VadStateMachine ─────────────────────────────────────────────────────────

/// Pure VAD state machine.
///
/// Accepts voice-activity probabilities (one per audio chunk) and emits
/// [`VadDecision`]s. Contains no I/O — the caller is responsible for feeding
/// probabilities from any source (e.g. a Silero wrapper, a mock, a test vector).
///
/// ## Hysteresis
///
/// - Speech **starts** when `probability >= config.speech_threshold` (default 0.5).
/// - Speech **ends** when `probability < config.speech_end_threshold` (default 0.3)
///   for at least `config.silence_duration` of consecutive audio.
///
/// Silence is counted in time, not frames: each call says how much audio its
/// probability covers, so chunks of any length can be mixed.
pub struct VadStateMachine {
    config: VadConfig,
    state: VadState,
    silence: Duration,
    speech_frames: u64,
}

impl VadStateMachine {
    /// Create a new state machine.
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            state: VadState::Idle,
            silence: Duration::ZERO,
            speech_frames: 0,
        }
    }

    /// Current state.
    pub fn state(&self) -> VadState {
        self.state
    }

    /// Process one audio chunk's voice probability; `chunk` is how much audio
    /// that probability covers.
    ///
    /// Returns the [`VadDecision`] for this chunk.
    pub fn process(&mut self, probability: f32, chunk: Duration) -> VadDecision {
        match self.state {
            VadState::Idle => {
                if probability >= self.config.speech_threshold {
                    self.state = VadState::Speaking;
                    self.silence = Duration::ZERO;
                    self.speech_frames = 1;
                    VadDecision::SpeechStarted
                } else {
                    VadDecision::Silence
                }
            }
            VadState::Speaking => {
                self.speech_frames += 1;
                if probability < self.config.speech_end_threshold {
                    self.silence += chunk;
                    if self.silence >= self.config.silence_duration {
                        self.state = VadState::Idle;
                        VadDecision::SpeechEnded
                    } else {
                        VadDecision::SpeechContinues
                    }
                } else {
                    self.silence = Duration::ZERO;
                    VadDecision::SpeechContinues
                }
            }
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.state = VadState::Idle;
        self.silence = Duration::ZERO;
        self.speech_frames = 0;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::types::VadConfig;
    use std::time::Duration;

    const CHUNK: Duration = Duration::from_millis(30);

    fn default_config() -> VadConfig {
        VadConfig::default()
        // speech_threshold: 0.5, speech_end_threshold: 0.3,
        // silence_duration: 500ms, speech_pad: 300ms, max_utterance: 30s
    }

    /// A config with a 300ms silence window (10 frames at 30ms/frame).
    fn config_300ms_silence() -> VadConfig {
        VadConfig {
            silence_duration: Duration::from_millis(300),
            ..VadConfig::default()
        }
    }

    #[test]
    fn test_vad_idle_to_speaking() {
        let mut sm = VadStateMachine::new(default_config());
        assert_eq!(sm.state(), VadState::Idle);

        // Below threshold → Silence
        let d = sm.process(0.3, CHUNK);
        assert_eq!(d, VadDecision::Silence);
        assert_eq!(sm.state(), VadState::Idle);

        // At threshold → SpeechStarted
        let d = sm.process(0.5, CHUNK);
        assert_eq!(d, VadDecision::SpeechStarted);
        assert_eq!(sm.state(), VadState::Speaking);
    }

    #[test]
    fn test_vad_speaking_to_ended() {
        // silence_duration=300ms, chunk=30ms → 10 frames needed
        let mut sm = VadStateMachine::new(config_300ms_silence());

        // Start speaking
        assert_eq!(sm.process(0.8, CHUNK), VadDecision::SpeechStarted);

        // Feed 9 low-probability frames — still Speaking
        for _ in 0..9 {
            let d = sm.process(0.1, CHUNK);
            assert_eq!(d, VadDecision::SpeechContinues);
            assert_eq!(sm.state(), VadState::Speaking);
        }

        // 10th low-probability frame → SpeechEnded
        let d = sm.process(0.1, CHUNK);
        assert_eq!(d, VadDecision::SpeechEnded);
        assert_eq!(sm.state(), VadState::Idle);
    }

    #[test]
    fn test_vad_speech_continues() {
        // silence_duration=300ms, chunk=30ms → 10 frames needed
        let mut sm = VadStateMachine::new(config_300ms_silence());

        // Start speaking
        assert_eq!(sm.process(0.9, CHUNK), VadDecision::SpeechStarted);

        // 5 low-probability frames accumulate silence
        for _ in 0..5 {
            assert_eq!(sm.process(0.1, CHUNK), VadDecision::SpeechContinues);
        }

        // A high-probability frame resets silence counter
        assert_eq!(sm.process(0.7, CHUNK), VadDecision::SpeechContinues);
        assert_eq!(sm.state(), VadState::Speaking);

        // 9 more low frames → still Speaking (counter was reset)
        for _ in 0..9 {
            assert_eq!(sm.process(0.1, CHUNK), VadDecision::SpeechContinues);
            assert_eq!(sm.state(), VadState::Speaking);
        }

        // 10th → ended
        assert_eq!(sm.process(0.1, CHUNK), VadDecision::SpeechEnded);
    }

    #[test]
    fn test_vad_reset() {
        let mut sm = VadStateMachine::new(default_config());

        // Transition to Speaking
        sm.process(0.9, CHUNK);
        assert_eq!(sm.state(), VadState::Speaking);

        // Reset → back to Idle
        sm.reset();
        assert_eq!(sm.state(), VadState::Idle);

        // After reset, acts like a fresh machine
        assert_eq!(sm.process(0.1, CHUNK), VadDecision::Silence);
        assert_eq!(sm.process(0.9, CHUNK), VadDecision::SpeechStarted);
    }

    #[test]
    fn silence_ends_speech_once_it_adds_up_to_the_configured_duration() {
        // 500 ms of silence in 30 ms chunks: 480 ms after 16 chunks, 510 ms after 17.
        let mut sm = VadStateMachine::new(default_config());
        assert_eq!(sm.process(0.9, CHUNK), VadDecision::SpeechStarted);
        for _ in 0..16 {
            assert_eq!(sm.process(0.1, CHUNK), VadDecision::SpeechContinues);
        }
        assert_eq!(sm.process(0.1, CHUNK), VadDecision::SpeechEnded);
    }

    #[test]
    fn chunks_of_different_lengths_count_by_their_own_duration() {
        let mut sm = VadStateMachine::new(config_300ms_silence());
        assert_eq!(
            sm.process(0.9, Duration::from_millis(10)),
            VadDecision::SpeechStarted
        );
        for _ in 0..20 {
            assert_eq!(
                sm.process(0.1, Duration::from_millis(10)),
                VadDecision::SpeechContinues
            );
        }
        assert_eq!(
            sm.process(0.1, Duration::from_millis(100)),
            VadDecision::SpeechEnded,
            "200 ms in 10 ms chunks plus one 100 ms chunk is 300 ms"
        );
    }
}
