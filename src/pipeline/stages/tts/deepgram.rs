use async_trait::async_trait;
use serde::Serialize;

use crate::error::Result;
use crate::MindroidError;

use super::TtsProvider;

pub struct DeepgramTtsConfig {
    pub api_key: String,
    /// Deepgram Aura voice model. Defaults to `"aura-asteria-en"`.
    ///
    /// Other options: `"aura-luna-en"`, `"aura-stella-en"`, `"aura-athena-en"`,
    /// `"aura-hera-en"`, `"aura-orion-en"`, `"aura-arcas-en"`, `"aura-perseus-en"`,
    /// `"aura-angus-en"`, `"aura-orpheus-en"`, `"aura-helios-en"`, `"aura-zeus-en"`.
    pub model: String,
    pub base_url: String,
}

impl Default for DeepgramTtsConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "aura-asteria-en".to_string(),
            base_url: "https://api.deepgram.com".to_string(),
        }
    }
}

pub struct DeepgramTts {
    client: reqwest::Client,
    config: DeepgramTtsConfig,
}

impl DeepgramTts {
    pub fn new(config: DeepgramTtsConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }
}

#[derive(Serialize)]
struct DgSpeakRequest<'a> {
    text: &'a str,
}

const MAX_CHARS: usize = 2000;

#[async_trait]
impl TtsProvider for DeepgramTts {
    async fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        let chunks = chunk_text(text, MAX_CHARS);

        let mut wavs: Vec<Vec<u8>> = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            wavs.push(self.synthesize_chunk(chunk).await?);
        }

        if wavs.len() == 1 {
            return Ok(wavs.remove(0));
        }

        Ok(concat_wav(wavs))
    }
}

impl DeepgramTts {
    async fn synthesize_chunk(&self, text: &str) -> Result<Vec<u8>> {
        let url = format!(
            "{}/v1/speak?model={}&encoding=linear16&container=wav",
            self.config.base_url, self.config.model,
        );

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Token {}", self.config.api_key))
            .json(&DgSpeakRequest { text })
            .send()
            .await
            .map_err(|e| MindroidError::Pipeline {
                stage: "DeepgramTts".into(),
                message: format!("HTTP error: {e}"),
                source: None,
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(MindroidError::Pipeline {
                stage: "DeepgramTts".into(),
                message: format!("Deepgram TTS returned {status}: {body}"),
                source: None,
            });
        }

        Ok(response.bytes().await.map_err(|e| MindroidError::Pipeline {
            stage: "DeepgramTts".into(),
            message: format!("Failed to read audio bytes: {e}"),
            source: None,
        })?.to_vec())
    }
}

/// Split text into chunks no longer than `max_chars`, breaking at sentence
/// boundaries (`. `, `! `, `? `) where possible, otherwise at word boundaries.
fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.len() <= max_chars {
            chunks.push(remaining.trim().to_string());
            break;
        }

        let slice = &remaining[..max_chars];
        let split_at = slice.rfind(". ")
            .or_else(|| slice.rfind("! "))
            .or_else(|| slice.rfind("? "))
            .map(|i| i + 2)
            .or_else(|| slice.rfind(' ').map(|i| i + 1))
            .unwrap_or(max_chars);

        chunks.push(remaining[..split_at].trim().to_string());
        remaining = remaining[split_at..].trim_start();
    }

    chunks.retain(|c| !c.is_empty());
    chunks
}

/// Concatenate multiple WAV buffers into one by keeping the first file's
/// header and appending the PCM data from all chunks.
fn concat_wav(chunks: Vec<Vec<u8>>) -> Vec<u8> {
    // Locate the PCM data offset by finding the "data" sub-chunk marker.
    fn data_offset(wav: &[u8]) -> usize {
        wav.windows(4)
            .position(|w| w == b"data")
            .map(|i| i + 8) // skip "data" (4) + chunk size (4)
            .unwrap_or(44)  // fall back to standard 44-byte header
    }

    let first_offset = data_offset(&chunks[0]);
    let mut header = chunks[0][..first_offset].to_vec();
    let mut pcm: Vec<u8> = Vec::new();

    for chunk in &chunks {
        let offset = data_offset(chunk);
        if chunk.len() > offset {
            pcm.extend_from_slice(&chunk[offset..]);
        }
    }

    // Patch RIFF size (bytes 4-7) and data chunk size (offset-4 to offset).
    let riff_size = (header.len() as u32 - 8) + pcm.len() as u32;
    header[4..8].copy_from_slice(&riff_size.to_le_bytes());
    let data_size_pos = first_offset - 4;
    header[data_size_pos..first_offset].copy_from_slice(&(pcm.len() as u32).to_le_bytes());

    header.extend(pcm);
    header
}
