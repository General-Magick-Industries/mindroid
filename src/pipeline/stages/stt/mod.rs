#[cfg(feature = "speech")]
mod deepgram;
#[cfg(feature = "llm-client")]
mod openai;

#[cfg(feature = "speech")]
pub use deepgram::{DeepgramStt, DeepgramSttConfig};
#[cfg(feature = "llm-client")]
pub use openai::{OpenAiStt, OpenAiSttConfig};

use async_trait::async_trait;
use std::sync::Arc;

use tracing::warn;

use crate::error::Result;
#[cfg(feature = "transport-audio")]
use crate::pipeline::extensions::{AudioInput, TextInput};
use crate::{MindroidError, PipelineContext, PipelineStage};

/// Converts raw audio bytes to a text transcript.
#[async_trait]
pub trait SttProvider: Send + Sync {
    async fn transcribe(&self, audio: &[u8]) -> Result<String>;
}

/// Resolves audio bytes for the current pipeline context.
///
/// Checks `ctx.audio_input` first (direct bytes, e.g. set programmatically).
/// When the `audio` feature is enabled, also checks
/// `ctx.message.metadata["audio_data"]` for a base64-encoded WAV written by
/// `AudioTransport`.
fn resolve_audio(_ctx: &PipelineContext) -> Result<Vec<u8>> {
    #[cfg(feature = "transport-audio")]
    if let Some(audio) = _ctx.get_ext::<AudioInput>() {
        return Ok(audio.0.clone());
    }

    #[cfg(feature = "transport-audio")]
    {
        use base64::{Engine, engine::general_purpose::STANDARD};
        if let Some(b64) = _ctx
            .message
            .metadata
            .get("audio_data")
            .and_then(|v| v.as_str())
        {
            return STANDARD.decode(b64).map_err(|e| MindroidError::Pipeline {
                stage: "SttStage".into(),
                message: format!("Failed to base64-decode audio_data: {e}"),
                source: None,
            });
        }
    }

    Err(MindroidError::Pipeline {
        stage: "SttStage".into(),
        message: "No audio available: set AudioInput extension or use AudioTransport".into(),
        source: None,
    })
}

/// Pipeline stage that transcribes audio input to text.
///
/// Reads raw audio bytes from `ctx.audio_input`, calls the configured
/// [`SttProvider`], and writes the transcript to `ctx.text_input` so
/// that downstream stages (e.g. `SimpleContextBuilder`) see the spoken text
/// rather than the placeholder in `message.content`.
///
/// Halts the pipeline if no audio is present or the transcript is empty.
pub struct SttStage {
    provider: Arc<dyn SttProvider>,
}

impl SttStage {
    pub fn new(provider: impl SttProvider + 'static) -> Self {
        Self {
            provider: Arc::new(provider),
        }
    }

    pub fn from_arc(provider: Arc<dyn SttProvider>) -> Self {
        Self { provider }
    }
}

#[async_trait]
impl PipelineStage for SttStage {
    fn name(&self) -> &str {
        "SttStage"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let audio = resolve_audio(ctx)?;
        let transcript = self.provider.transcribe(&audio).await?;

        if transcript.trim().is_empty() {
            warn!("SttStage: empty transcript, halting pipeline");
            ctx.halted = true;
            return Ok(());
        }

        #[cfg(feature = "transport-audio")]
        {
            ctx.set_ext(TextInput(transcript));
        }
        #[cfg(not(feature = "transport-audio"))]
        {
            let _ = transcript;
        }
        Ok(())
    }
}
