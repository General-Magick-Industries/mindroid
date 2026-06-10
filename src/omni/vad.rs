//! VAD (Voice Activity Detection) state machine and inference wrapper.
//!
//! # Components
//!
//! - [`VadStateMachine`]: pure logic that processes voice probability floats and
//!   emits [`VadDecision`]s. Always available (no feature gate). Defined in
//!   [`crate::voice::vad`] and re-exported here for backwards compatibility.
//! - [`VadInference`]: wraps the `voice_activity_detector` (Silero) crate to
//!   run ONNX inference on PCM audio. Feature-gated behind `transport-audio`.

// Re-export pure VAD primitives from the neutral `voice` module.
pub use crate::voice::vad::{VadDecision, VadState, VadStateMachine};

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
