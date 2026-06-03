//! VAD (Voice Activity Detection) state machine and inference wrapper.
//!
//! # Components
//!
//! - [`VadStateMachine`]: pure logic that processes voice probability floats and
//!   emits [`VadDecision`]s. Always available (no feature gate).
//! - [`VadInference`]: wraps the `voice_activity_detector` (Silero) crate to
//!   run ONNX inference on PCM audio. Feature-gated behind `transport-audio`.

use crate::omni::types::VadConfig;

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
/// probabilities from any source (e.g. [`VadInference`], a mock, a test vector).
///
/// ## Hysteresis
///
/// - Speech **starts** when `probability >= config.speech_threshold` (default 0.5).
/// - Speech **ends** when `probability < config.speech_end_threshold` (default 0.3)
///   for at least `silence_threshold_frames` consecutive frames.
///
/// ## Frame-count semantics
///
/// `silence_threshold_frames` is computed as:
/// ```text
/// silence_threshold_frames = silence_duration_ms / chunk_duration_ms
/// ```
/// where `chunk_duration_ms` is the duration of a single audio chunk (e.g. 30 ms
/// for 512 samples @ 16 kHz). This avoids coupling the state machine to the
/// sample rate.
pub struct VadStateMachine {
    config: VadConfig,
    state: VadState,
    silence_frames: u64,
    speech_frames: u64,
    /// How many frames of sub-threshold probability are required to end speech.
    silence_threshold_frames: u64,
}

impl VadStateMachine {
    /// Create a new state machine.
    ///
    /// `chunk_duration_ms` is how long each audio chunk is in milliseconds
    /// (i.e. how often [`process`](Self::process) will be called per real-time
    /// second). For example, 512 samples at 16 kHz → 32 ms chunks.
    pub fn new(config: VadConfig, chunk_duration_ms: u64) -> Self {
        let silence_threshold_frames =
            config.silence_duration.as_millis() as u64 / chunk_duration_ms;
        Self {
            config,
            state: VadState::Idle,
            silence_frames: 0,
            speech_frames: 0,
            silence_threshold_frames,
        }
    }

    /// Current state.
    pub fn state(&self) -> VadState {
        self.state
    }

    /// Process one audio chunk's voice probability.
    ///
    /// Returns the [`VadDecision`] for this frame.
    pub fn process(&mut self, probability: f32) -> VadDecision {
        match self.state {
            VadState::Idle => {
                if probability >= self.config.speech_threshold {
                    self.state = VadState::Speaking;
                    self.silence_frames = 0;
                    self.speech_frames = 1;
                    VadDecision::SpeechStarted
                } else {
                    VadDecision::Silence
                }
            }
            VadState::Speaking => {
                self.speech_frames += 1;
                if probability < self.config.speech_end_threshold {
                    self.silence_frames += 1;
                    if self.silence_frames >= self.silence_threshold_frames {
                        self.state = VadState::Idle;
                        VadDecision::SpeechEnded
                    } else {
                        VadDecision::SpeechContinues
                    }
                } else {
                    self.silence_frames = 0;
                    VadDecision::SpeechContinues
                }
            }
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.state = VadState::Idle;
        self.silence_frames = 0;
        self.speech_frames = 0;
    }
}

// ─── VadInference ─────────────────────────────────────────────────────────────

/// Wrapper around the Silero ONNX model for voice activity detection.
///
/// Converts raw PCM i16 samples to f32, then runs the Silero model.
///
/// **Must be called from a blocking context** (e.g. inside
/// `tokio::task::spawn_blocking`) — ONNX inference is synchronous and
/// CPU-bound.
#[cfg(feature = "transport-audio")]
pub struct VadInference {
    detector: voice_activity_detector::VoiceActivityDetector,
}

#[cfg(feature = "transport-audio")]
impl VadInference {
    /// Build a new `VadInference` for the given `sample_rate`.
    ///
    /// Silero supports 8000 Hz and 16000 Hz.
    pub fn new(sample_rate: u32, chunk_size: usize) -> crate::error::Result<Self> {
        let detector = voice_activity_detector::VoiceActivityDetector::builder()
            .sample_rate(sample_rate)
            .chunk_size(chunk_size)
            .build()
            .map_err(|e| crate::MindroidError::Transport {
                message: format!("VadInference init failed: {e}"),
                source: None,
            })?;
        Ok(Self { detector })
    }

    /// Convert PCM i16 bytes to f32 samples and run Silero inference.
    ///
    /// Returns a voice probability in `[0.0, 1.0]`.
    ///
    /// **Must be called on a blocking thread** (ONNX inference is synchronous).
    pub fn predict(&mut self, pcm_i16: &[i16]) -> f32 {
        let samples: Vec<f32> = pcm_i16
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();
        self.detector.predict(samples)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::omni::types::VadConfig;
    use std::time::Duration;

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
        let mut sm = VadStateMachine::new(default_config(), 30);
        assert_eq!(sm.state(), VadState::Idle);

        // Below threshold → Silence
        let d = sm.process(0.3);
        assert_eq!(d, VadDecision::Silence);
        assert_eq!(sm.state(), VadState::Idle);

        // At threshold → SpeechStarted
        let d = sm.process(0.5);
        assert_eq!(d, VadDecision::SpeechStarted);
        assert_eq!(sm.state(), VadState::Speaking);
    }

    #[test]
    fn test_vad_speaking_to_ended() {
        // silence_duration=300ms, chunk=30ms → 10 frames needed
        let mut sm = VadStateMachine::new(config_300ms_silence(), 30);

        // Start speaking
        assert_eq!(sm.process(0.8), VadDecision::SpeechStarted);

        // Feed 9 low-probability frames — still Speaking
        for _ in 0..9 {
            let d = sm.process(0.1);
            assert_eq!(d, VadDecision::SpeechContinues);
            assert_eq!(sm.state(), VadState::Speaking);
        }

        // 10th low-probability frame → SpeechEnded
        let d = sm.process(0.1);
        assert_eq!(d, VadDecision::SpeechEnded);
        assert_eq!(sm.state(), VadState::Idle);
    }

    #[test]
    fn test_vad_speech_continues() {
        // silence_duration=300ms, chunk=30ms → 10 frames needed
        let mut sm = VadStateMachine::new(config_300ms_silence(), 30);

        // Start speaking
        assert_eq!(sm.process(0.9), VadDecision::SpeechStarted);

        // 5 low-probability frames accumulate silence
        for _ in 0..5 {
            assert_eq!(sm.process(0.1), VadDecision::SpeechContinues);
        }

        // A high-probability frame resets silence counter
        assert_eq!(sm.process(0.7), VadDecision::SpeechContinues);
        assert_eq!(sm.state(), VadState::Speaking);

        // 9 more low frames → still Speaking (counter was reset)
        for _ in 0..9 {
            assert_eq!(sm.process(0.1), VadDecision::SpeechContinues);
            assert_eq!(sm.state(), VadState::Speaking);
        }

        // 10th → ended
        assert_eq!(sm.process(0.1), VadDecision::SpeechEnded);
    }

    #[test]
    fn test_vad_reset() {
        let mut sm = VadStateMachine::new(default_config(), 30);

        // Transition to Speaking
        sm.process(0.9);
        assert_eq!(sm.state(), VadState::Speaking);

        // Reset → back to Idle
        sm.reset();
        assert_eq!(sm.state(), VadState::Idle);

        // After reset, acts like a fresh machine
        assert_eq!(sm.process(0.1), VadDecision::Silence);
        assert_eq!(sm.process(0.9), VadDecision::SpeechStarted);
    }

    #[test]
    fn test_vad_threshold_frames_calculation() {
        // 500ms silence / 30ms chunk = 16 frames (integer division)
        let cfg = VadConfig {
            silence_duration: Duration::from_millis(500),
            ..VadConfig::default()
        };
        let sm = VadStateMachine::new(cfg, 30);
        assert_eq!(sm.silence_threshold_frames, 16);

        // 1200ms / 30ms = 40 frames
        let cfg2 = VadConfig {
            silence_duration: Duration::from_millis(1200),
            ..VadConfig::default()
        };
        let sm2 = VadStateMachine::new(cfg2, 30);
        assert_eq!(sm2.silence_threshold_frames, 40);

        // 300ms / 30ms = 10 frames
        let cfg3 = VadConfig {
            silence_duration: Duration::from_millis(300),
            ..VadConfig::default()
        };
        let sm3 = VadStateMachine::new(cfg3, 30);
        assert_eq!(sm3.silence_threshold_frames, 10);
    }
}
