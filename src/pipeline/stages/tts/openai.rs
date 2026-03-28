use async_openai::{
    config::OpenAIConfig,
    types::audio::{CreateSpeechRequestArgs, SpeechModel, Voice},
    Client,
};
use async_trait::async_trait;

use crate::error::Result;
use crate::MindroidError;

use super::TtsProvider;

pub struct OpenAiTtsConfig {
    pub api_key: String,
    /// TTS model. `"tts-1"` (lower latency) or `"tts-1-hd"` (higher quality).
    /// Defaults to `"tts-1"`.
    pub model: String,
    /// Voice name. One of: `alloy`, `ash`, `echo`, `fable`, `onyx`, `nova`, `shimmer`.
    /// Defaults to `"nova"`.
    pub voice: String,
    pub base_url: Option<String>,
}

impl Default for OpenAiTtsConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "tts-1".to_string(),
            voice: "nova".to_string(),
            base_url: None,
        }
    }
}

pub struct OpenAiTts {
    client: Client<OpenAIConfig>,
    model: String,
    voice: String,
}

impl OpenAiTts {
    pub fn new(config: OpenAiTtsConfig) -> Self {
        let mut openai_config = OpenAIConfig::new().with_api_key(&config.api_key);
        if let Some(base_url) = config.base_url {
            openai_config = openai_config.with_api_base(base_url);
        }
        Self {
            client: Client::with_config(openai_config),
            model: config.model,
            voice: config.voice,
        }
    }
}

#[async_trait]
impl TtsProvider for OpenAiTts {
    async fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        let model = match self.model.as_str() {
            "tts-1-hd" => SpeechModel::Tts1Hd,
            _ => SpeechModel::Tts1,
        };

        let voice = match self.voice.as_str() {
            "alloy" => Voice::Alloy,
            "ash" => Voice::Ash,
            "echo" => Voice::Echo,
            "fable" => Voice::Fable,
            "onyx" => Voice::Onyx,
            "shimmer" => Voice::Shimmer,
            _ => Voice::Nova,
        };

        let request = CreateSpeechRequestArgs::default()
            .input(text)
            .model(model)
            .voice(voice)
            .build()
            .map_err(|e| MindroidError::Pipeline {
                stage: "OpenAiTts".into(),
                message: format!("Failed to build speech request: {e}"),
                source: None,
            })?;

        let response = self
            .client
            .audio()
            .speech()
            .create(request)
            .await
            .map_err(|e| MindroidError::Pipeline {
                stage: "OpenAiTts".into(),
                message: format!("OpenAI TTS error: {e}"),
                source: None,
            })?;

        Ok(response.bytes.to_vec())
    }
}
