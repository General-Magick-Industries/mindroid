use async_trait::async_trait;
use reqwest::multipart::{Form, Part};
use serde::Deserialize;

use crate::MindroidError;
use crate::error::Result;

use super::SttProvider;

/// Configuration for Cartesia's batch speech-to-text (Ink) endpoint.
///
/// This is the one-shot REST transcription API (`POST /stt`), which fits the
/// [`SttProvider`] trait shape (full audio bytes in, transcript out). For
/// low-latency streaming STT, Cartesia exposes a separate websocket API that
/// is not modeled by this provider.
pub struct CartesiaSttConfig {
    pub api_key: String,
    /// Batch STT model. Must be in the `ink-whisper` family. Defaults to
    /// `"ink-whisper"`.
    pub model: String,
    /// ISO-639-1 language code (e.g. `"en"`). Defaults to `"en"`.
    pub language: String,
    /// API base URL. Defaults to `"https://api.cartesia.ai"`.
    pub base_url: String,
    /// `Cartesia-Version` date header. Defaults to `"2026-03-01"`.
    pub version: String,
}

impl Default for CartesiaSttConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "ink-whisper".to_string(),
            language: "en".to_string(),
            base_url: "https://api.cartesia.ai".to_string(),
            version: "2026-03-01".to_string(),
        }
    }
}

impl CartesiaSttConfig {
    /// Construct a config with the given API key and library defaults.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Override the ISO-639-1 language code.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Override the batch STT model (must be in the `ink-whisper` family).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

pub struct CartesiaStt {
    client: reqwest::Client,
    config: CartesiaSttConfig,
}

impl CartesiaStt {
    pub fn new(config: CartesiaSttConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }
}

// Minimal deserialization for Cartesia's /stt response.
#[derive(Deserialize)]
struct CartesiaSttResponse {
    #[serde(default)]
    text: String,
}

#[async_trait]
impl SttProvider for CartesiaStt {
    async fn transcribe(&self, audio: &[u8]) -> Result<String> {
        let url = format!("{}/stt", self.config.base_url);

        let file_part = Part::bytes(audio.to_vec())
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| MindroidError::Pipeline {
                stage: "CartesiaStt".into(),
                message: format!("Failed to build multipart audio part: {e}"),
                source: None,
            })?;

        let form = Form::new()
            .part("file", file_part)
            .text("model", self.config.model.clone())
            .text("language", self.config.language.clone());

        let response = self
            .client
            .post(&url)
            .header("X-API-Key", &self.config.api_key)
            .header("Cartesia-Version", &self.config.version)
            .multipart(form)
            .send()
            .await
            .map_err(|e| MindroidError::Pipeline {
                stage: "CartesiaStt".into(),
                message: format!("HTTP error: {e}"),
                source: None,
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(MindroidError::Pipeline {
                stage: "CartesiaStt".into(),
                message: format!("Cartesia STT returned {status}: {body}"),
                source: None,
            });
        }

        let parsed: CartesiaSttResponse =
            response.json().await.map_err(|e| MindroidError::Pipeline {
                stage: "CartesiaStt".into(),
                message: format!("Failed to parse Cartesia STT response: {e}"),
                source: None,
            })?;

        Ok(parsed.text)
    }
}
