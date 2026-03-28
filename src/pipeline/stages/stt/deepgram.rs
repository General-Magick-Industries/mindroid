use async_trait::async_trait;
use serde::Deserialize;

use crate::error::Result;
use crate::MindroidError;

use super::SttProvider;

pub struct DeepgramSttConfig {
    pub api_key: String,
    /// Deepgram model. Defaults to `"nova-2"`.
    pub model: String,
    /// BCP-47 language code. Defaults to `"en"`.
    pub language: String,
    pub base_url: String,
}

impl Default for DeepgramSttConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "nova-2".to_string(),
            language: "en".to_string(),
            base_url: "https://api.deepgram.com".to_string(),
        }
    }
}

pub struct DeepgramStt {
    client: reqwest::Client,
    config: DeepgramSttConfig,
}

impl DeepgramStt {
    pub fn new(config: DeepgramSttConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }
}

// Minimal deserialization for Deepgram's /v1/listen response.
#[derive(Deserialize)]
struct DgResponse {
    results: DgResults,
}

#[derive(Deserialize)]
struct DgResults {
    channels: Vec<DgChannel>,
}

#[derive(Deserialize)]
struct DgChannel {
    alternatives: Vec<DgAlternative>,
}

#[derive(Deserialize)]
struct DgAlternative {
    transcript: String,
}

#[async_trait]
impl SttProvider for DeepgramStt {
    async fn transcribe(&self, audio: &[u8]) -> Result<String> {
        let audio = audio.to_vec();
        let url = format!(
            "{}/v1/listen?model={}&language={}&smart_format=true",
            self.config.base_url, self.config.model, self.config.language,
        );

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Token {}", self.config.api_key))
            .header("Content-Type", "audio/wav")
            .body(audio)
            .send()
            .await
            .map_err(|e| MindroidError::Pipeline {
                stage: "DeepgramStt".into(),
                message: format!("HTTP error: {e}"),
                source: None,
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(MindroidError::Pipeline {
                stage: "DeepgramStt".into(),
                message: format!("Deepgram STT returned {status}: {body}"),
                source: None,
            });
        }

        let parsed: DgResponse = response.json().await.map_err(|e| MindroidError::Pipeline {
            stage: "DeepgramStt".into(),
            message: format!("Failed to parse Deepgram response: {e}"),
            source: None,
        })?;

        let transcript = parsed
            .results
            .channels
            .into_iter()
            .next()
            .and_then(|ch| ch.alternatives.into_iter().next())
            .map(|alt| alt.transcript)
            .unwrap_or_default();

        Ok(transcript)
    }
}
