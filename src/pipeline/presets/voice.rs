use crate::Pipeline;
use crate::llm_client::LlmClient;
use crate::pipeline::stages::{GenericLlmProcessor, SimpleContextBuilder, SttStage, TtsStage};
use crate::pipeline::stages::{SttProvider, TtsProvider};

/// Builds a full voice pipeline: audio-in → LLM → audio-out.
///
/// The pipeline executes the following stages in order:
/// 1. [`SttStage`] — transcribes incoming audio bytes to text
/// 2. [`SimpleContextBuilder`] — assembles LLM messages from the transcript
/// 3. [`GenericLlmProcessor`] — streams the LLM response
/// 4. [`TtsStage`] — synthesizes the response text to audio bytes
///
/// Callers are responsible for constructing the providers and the
/// [`LlmClient`] before calling this function, which keeps the preset
/// backend-agnostic (works with [`OpenAiStt`] / [`DeepgramStt`],
/// any OpenAI-compatible LLM, and [`OpenAiTts`] / [`DeepgramTts`]).
///
/// # Example
///
/// ```ignore
/// use mindroid::llm_client::{AuthStyle, LlmClient, LlmClientConfig};
/// use mindroid::pipeline::presets::voice::voice_pipeline;
/// use mindroid::{OpenAiStt, OpenAiSttConfig, OpenAiTts, OpenAiTtsConfig};
///
/// let stt = OpenAiStt::new(OpenAiSttConfig::new("https://api.openai.com/v1", api_key));
/// let tts = OpenAiTts::new(OpenAiTtsConfig::new("https://api.openai.com/v1", api_key));
///
/// let mut cfg = LlmClientConfig::new("https://api.openai.com/v1");
/// cfg.default_model = Some("gpt-4o".to_string());
/// let client = LlmClient::new(cfg)?;
///
/// let pipeline = voice_pipeline(stt, client, tts)?;
/// ```
pub fn voice_pipeline(
    stt: impl SttProvider + 'static,
    llm: LlmClient,
    tts: impl TtsProvider + 'static,
) -> crate::Result<Pipeline> {
    Ok(Pipeline::new()
        .add_stage(SttStage::new(stt))
        .add_stage(SimpleContextBuilder::new())
        .add_streaming_stage(GenericLlmProcessor::new(llm))
        .add_stage(TtsStage::new(tts)))
}

/// Builds a voice pipeline backed by Cartesia for both ears and voice:
/// Cartesia Ink (STT) → LLM → Cartesia Sonic (TTS).
///
/// This is a thin convenience wrapper over [`voice_pipeline`] that constructs
/// [`CartesiaStt`](crate::CartesiaStt) and [`CartesiaTts`](crate::CartesiaTts)
/// from their configs. Set the voice tone on [`CartesiaTtsConfig`] before
/// calling (voice id, speed, volume, or the tone presets).
///
/// # Example
///
/// ```ignore
/// use mindroid::llm_client::{LlmClient, LlmClientConfig};
/// use mindroid::pipeline::presets::voice::cartesia_voice_pipeline;
/// use mindroid::{CartesiaSttConfig, CartesiaTtsConfig};
///
/// let stt = CartesiaSttConfig::new(api_key.clone());
/// let tts = CartesiaTtsConfig::new(api_key, voice_id).lively();
///
/// let mut cfg = LlmClientConfig::new("http://localhost:13305/api/v1");
/// cfg.default_model = Some("Qwen3-0.6B-GGUF".to_string());
/// let client = LlmClient::new(cfg)?;
///
/// let pipeline = cartesia_voice_pipeline(stt, client, tts)?;
/// ```
#[cfg(feature = "speech")]
pub fn cartesia_voice_pipeline(
    stt: crate::CartesiaSttConfig,
    llm: LlmClient,
    tts: crate::CartesiaTtsConfig,
) -> crate::Result<Pipeline> {
    voice_pipeline(
        crate::CartesiaStt::new(stt),
        llm,
        crate::CartesiaTts::new(tts),
    )
}
