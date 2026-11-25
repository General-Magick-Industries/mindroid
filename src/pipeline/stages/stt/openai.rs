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
    /// ISO-639-1 language of the speech (`"en"`). Without it the model guesses per
    /// clip, and short or noisy clips come back in random languages.
    pub language: Option<String>,
}

impl Default for OpenAiSttConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "whisper-1".to_string(),
            base_url: None,
            language: None,
        }
    }
}

pub struct OpenAiStt {
    client: Client<OpenAIConfig>,
    model: String,
    language: Option<String>,
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
            language: config.language,
        }
    }
}

#[async_trait]
impl SttProvider for OpenAiStt {
    async fn transcribe(&self, audio: &[u8]) -> Result<String> {
        let audio = audio.to_vec();
        let mut request = CreateTranscriptionRequestArgs::default();
        request
            .file(AudioInput {
                source: InputSource::VecU8 {
                    filename: "audio.wav".to_string(),
                    vec: audio,
                },
            })
            .model(self.model.clone());
        if let Some(language) = &self.language {
            request.language(language.clone());
        }
        let request = request.build().map_err(|e| MindroidError::Pipeline {
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
