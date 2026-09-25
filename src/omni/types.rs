use crate::core::error::MindroidError;
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// Re-export pure voice primitives from the neutral `voice` module.
pub use crate::voice::types::{BargeInMode, TurnDetection, VadConfig};

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub data: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
}

/// Which side of the conversation a [`OmniEvent::Transcript`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptSource {
    Input,
    Output,
}

/// Token accounting for one turn, as the provider reports it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub input_audio_tokens: u64,
    pub output_audio_tokens: u64,
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, o: Self) {
        self.input_tokens += o.input_tokens;
        self.output_tokens += o.output_tokens;
        self.input_audio_tokens += o.input_audio_tokens;
        self.output_audio_tokens += o.output_audio_tokens;
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum OmniEvent {
    AudioChunk(AudioChunk),
    /// A transcript piece, or — when `is_final` — the complete text for that turn
    /// and source. Providers normalize to this contract so consumers can persist
    /// finals without re-assembling fragments.
    Transcript {
        text: String,
        is_final: bool,
        source: TranscriptSource,
    },
    ToolCall {
        id: String,
        name: String,
        args: Value,
    },
    Interrupted,
    /// The provider's turn detection heard the user stop speaking. The session
    /// closes its utterance capture here; the reply, if any, follows on its own.
    UserSpeechEnded,
    TurnComplete,
    /// Provider will close the session soon; realtime sessions are time-capped.
    SessionEnding {
        in_secs: Option<u64>,
    },
    /// Opaque handle for resuming this session after a disconnect.
    ResumptionHandle(String),
    /// Token counts for the turn that just finished; emitted before `TurnComplete`.
    Usage(Usage),
    /// `MindroidError` is not `Clone`, so we wrap it in `Arc`.
    Error(Arc<MindroidError>),
}

/// Who spoke a [`HistoryTurn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Model,
}

/// One prior turn, seeded as text at session start. Both providers accept text
/// history into an audio session; neither needs the original audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryTurn {
    pub role: Role,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct OmniConfig {
    pub turn_detection: TurnDetection,
    pub barge_in: BargeInMode,
    pub system_prompt: Option<String>,
    pub tools_schema: Option<Value>,
    pub voice: Option<String>,
    /// Prior turns, oldest first. Providers serialize these once, right after the
    /// setup handshake and before any audio.
    pub history: Vec<HistoryTurn>,
}

impl Default for OmniConfig {
    fn default() -> Self {
        Self {
            turn_detection: TurnDetection::Server,
            barge_in: BargeInMode::LocalVad,
            system_prompt: None,
            tools_schema: None,
            voice: None,
            history: Vec::new(),
        }
    }
}

/// A handle the session puts in every tool's [`ToolContext`](crate::tools::ToolContext)
/// so a tool can end the session — after the current turn, so a spoken goodbye
/// still plays out.
#[derive(Debug, Clone, Default)]
pub struct SessionControl {
    end: Arc<AtomicBool>,
}

impl SessionControl {
    pub fn end_after_turn(&self) {
        self.end.store(true, Ordering::Relaxed);
    }

    pub fn end_requested(&self) -> bool {
        self.end.load(Ordering::Relaxed)
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
            source: TranscriptSource::Output,
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
