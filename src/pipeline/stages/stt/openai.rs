use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        InputSource,
        audio::{AudioInput, CreateTranscriptionRequestArgs},
    },
};
use async_trait::async_trait;

use crate::MindroidError;
use crate::error::Result;

use super::SttProvider;

pub struct OpenAiSttConfig {
    pub api_key: String,
    /// Whisper model to use. Defaults to `"whisper-1"`.
    pub model: String,
    pub base_url: Option<String>,
}

impl Default for OpenAiSttConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "whisper-1".to_string(),
            base_url: None,
        }
    }
}

pub struct OpenAiStt {
    client: Client<OpenAIConfig>,
    model: String,
}

impl OpenAiStt {
    pub fn new(config: OpenAiSttConfig) -> Self {
        let mut openai_config = OpenAIConfig::new().with_api_key(&config.api_key);
        if let Some(base_url) = config.base_url {
            openai_config = openai_config.with_api_base(base_url);
        }
        Self {
            client: Client::with_config(openai_config),
            model: config.model,
        }
    }
}

#[async_trait]
impl SttProvider for OpenAiStt {
    async fn transcribe(&self, audio: &[u8]) -> Result<String> {
        let audio = audio.to_vec();
        let request = CreateTranscriptionRequestArgs::default()
            .file(AudioInput {
                source: InputSource::VecU8 {
                    filename: "audio.wav".to_string(),
                    vec: audio,
                },
            })
            .model(self.model.clone())
            .build()
            .map_err(|e| MindroidError::Pipeline {
                stage: "OpenAiStt".into(),
                message: format!("Failed to build transcription request: {e}"),
                source: None,
            })?;

        let response = self
            .client
            .audio()
            .transcription()
            .create(request)
            .await
            .map_err(|e| MindroidError::Pipeline {
                stage: "OpenAiStt".into(),
                message: format!("Whisper API error: {e}"),
                source: None,
            })?;

        Ok(response.text)
    }
}
