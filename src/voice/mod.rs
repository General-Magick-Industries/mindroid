//! Pure voice primitives: VAD state machine and policy enums.
//!
//! These types are always available — no feature gate required.
//! The Silero ONNX inference wrapper ([`crate::omni::vad::VadInference`]) remains
//! in `omni/vad.rs` behind the `transport-audio` feature.
//!
//! The `wav` sub-module is gated behind `transport-audio` because it depends on
//! `hound`; all other `voice/` modules are feature-free.

pub mod conditioning;
pub mod echo_guard;
pub mod frontend;
pub mod interruption;
pub mod turn_detector;
pub mod turn_id;
pub mod types;
pub mod vad;
/// WAV encoding helper. Requires the `transport-audio` feature (needs `hound`).
#[cfg(feature = "transport-audio")]
pub mod wav;

pub use conditioning::{CaptureConditioner, PassthroughConditioner};
pub use echo_guard::EchoGuard;
pub use frontend::{AudioFrontend, FrontendEvent};
pub use interruption::{
    InterruptionDecision, InterruptionGate, InterruptionScorer, SustainedFramesScorer,
};
pub use turn_detector::{SilenceTimerTurnDetector, TurnDecision, TurnDetector};
pub use turn_id::{TurnCounter, TurnId};
pub use types::{BargeInMode, TurnDetection, VadConfig};
pub use vad::{VadDecision, VadState, VadStateMachine};
#[cfg(feature = "transport-audio")]
pub use wav::encode_wav;
