pub mod audio;
pub mod gate;
#[cfg(feature = "omni-gemini")]
pub mod gemini;
pub mod mock;
#[cfg(feature = "omni-openai")]
pub mod openai_realtime;
pub mod provider;
pub mod session;
pub mod types;
pub mod vad;

#[cfg(feature = "transport-audio")]
pub mod cpal_audio;

pub use audio::{AudioSink, AudioSource};
#[cfg(feature = "transport-audio")]
pub use gate::SileroDetector;
pub use gate::{SpeechDetector, VoiceGate, VoiceGateBuilder};
#[cfg(feature = "omni-gemini")]
pub use gemini::{GeminiLiveConfig, GeminiLiveProvider};
#[cfg(feature = "omni-openai")]
pub use openai_realtime::{OpenAiRealtimeConfig, OpenAiRealtimeProvider};
pub use provider::OmniProvider;
pub use session::{OmniSession, OmniSessionBuilder};
pub use types::*;
pub use vad::{VadDecision, VadState, VadStateMachine};

#[cfg(feature = "transport-audio")]
pub use cpal_audio::{CpalAudio, CpalAudioSink, CpalAudioSource};
