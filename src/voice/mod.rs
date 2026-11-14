//! Pure voice primitives: VAD state machine and policy enums.
//!
//! These types are always available — no feature gate required.
//! The Silero ONNX inference wrapper ([`crate::omni::vad::VadInference`]) remains
//! in `omni/vad.rs` behind the `transport-audio` feature. WAV encoding lives in
//! [`crate::omni::audio::wav`].

pub mod conditioning;
pub mod echo_guard;
pub mod frontend;
pub mod interruption;
pub mod turn_detector;
pub mod turn_id;
pub mod types;
pub mod vad;

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
