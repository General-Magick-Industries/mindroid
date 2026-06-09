use crate::core::error::MindroidError;
use serde_json::Value;
use std::sync::Arc;

// Re-export pure voice primitives from the neutral `voice` module.
pub use crate::voice::types::{BargeInMode, TurnDetection, VadConfig};

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub data: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
}

#[derive(Debug, Clone)]
pub enum OmniEvent {
    AudioChunk(AudioChunk),
    Transcript {
        text: String,
        is_final: bool,
    },
    ToolCall {
        id: String,
        name: String,
        args: Value,
    },
    Interrupted,
    TurnComplete,
    /// `MindroidError` is not `Clone`, so we wrap it in `Arc`.
    Error(Arc<MindroidError>),
}

#[derive(Debug, Clone)]
pub struct OmniConfig {
    pub turn_detection: TurnDetection,
    pub barge_in: BargeInMode,
    pub system_prompt: Option<String>,
    pub tools_schema: Option<Value>,
    pub voice: Option<String>,
}

impl Default for OmniConfig {
    fn default() -> Self {
        Self {
            turn_detection: TurnDetection::Server,
            barge_in: BargeInMode::LocalVad,
            system_prompt: None,
            tools_schema: None,
            voice: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Connecting,
    Listening,
    Speaking,
    ToolCall,
    Closed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omni_config_default() {
        let cfg = OmniConfig::default();
        assert!(matches!(cfg.turn_detection, TurnDetection::Server));
        assert!(matches!(cfg.barge_in, BargeInMode::LocalVad));
        assert!(cfg.system_prompt.is_none());
        assert!(cfg.tools_schema.is_none());
        assert!(cfg.voice.is_none());
    }

    #[test]
    fn audio_chunk_construction() {
        let chunk = AudioChunk {
            data: vec![0u8, 1, 2, 3],
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
        };
        assert_eq!(chunk.data.len(), 4);
        assert_eq!(chunk.sample_rate, 16_000);
        assert_eq!(chunk.channels, 1);
        assert_eq!(chunk.bits_per_sample, 16);
    }

    #[test]
    fn audio_chunk_clone() {
        let chunk = AudioChunk {
            data: vec![10, 20, 30],
            sample_rate: 44_100,
            channels: 2,
            bits_per_sample: 24,
        };
        let cloned = chunk.clone();
        assert_eq!(cloned.data, chunk.data);
        assert_eq!(cloned.sample_rate, chunk.sample_rate);
    }

    #[test]
    fn omni_event_clone() {
        let chunk = AudioChunk {
            data: vec![1, 2],
            sample_rate: 8_000,
            channels: 1,
            bits_per_sample: 8,
        };
        let event = OmniEvent::AudioChunk(chunk);
        let _cloned = event.clone();

        let transcript = OmniEvent::Transcript {
            text: "hello".into(),
            is_final: true,
        };
        let _cloned_t = transcript.clone();

        let interrupted = OmniEvent::Interrupted;
        let _cloned_i = interrupted.clone();

        let complete = OmniEvent::TurnComplete;
        let _cloned_c = complete.clone();
    }

    #[test]
    fn session_state_eq() {
        assert_eq!(SessionState::Listening, SessionState::Listening);
        assert_ne!(SessionState::Speaking, SessionState::Closed);
    }
}
