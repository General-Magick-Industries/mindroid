use async_trait::async_trait;
use serde::Serialize;

use crate::MindroidError;
use crate::error::Result;

use super::TtsProvider;

/// Cartesia Sonic TTS generation tuning ("voice tone").
///
/// Only fields that are `Some` are sent to the API, so the server defaults
/// apply for anything left unset. `speed`, `volume`, and `emotion` map directly
/// to Cartesia's `generation_config`.
#[derive(Clone, Default, Serialize)]
pub struct GenerationConfig {
    /// Speaking rate. Valid range `0.6`–`1.5` (1.0 is normal).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// Output volume. Valid range `0.5`–`2.0` (1.0 is normal).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<f32>,
    /// Emotion, as a single value. Cartesia (sonic-3 / sonic-3.5) accepts one
    /// of: `"neutral"`, `"calm"`, `"angry"`, `"content"`, `"sad"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
}

impl GenerationConfig {
    fn is_empty(&self) -> bool {
        self.speed.is_none() && self.volume.is_none() && self.emotion.is_none()
    }
}

/// Configuration for Cartesia's text-to-speech (Sonic) `POST /tts/bytes`
/// endpoint.
///
/// Returns a fully-formed audio container (`WAV` by default) so downstream
/// playback (`rodio` via `AudioOutputStage`) and transports can decode it
/// without extra header handling.
///
/// # Voice tones
///
/// A "voice tone" is the combination of a `voice_id` and a [`GenerationConfig`]
/// (speed / volume). Use the tone presets — [`CartesiaTtsConfig::calm`],
/// [`lively`](CartesiaTtsConfig::lively), [`narration`](CartesiaTtsConfig::narration) —
/// or set [`speed`](CartesiaTtsConfig::speed) / [`volume`](CartesiaTtsConfig::volume)
/// directly.
pub struct CartesiaTtsConfig {
    pub api_key: String,
    /// Sonic model id: `"sonic-3.5"`, `"sonic-3"`, or `"sonic-latest"`.
    /// Defaults to `"sonic-3.5"`.
    pub model_id: String,
    /// Cartesia voice id (UUID string).
    pub voice_id: String,
    /// ISO-639-1 language code. Defaults to `"en"`.
    pub language: String,
    /// Output sample rate in Hz (8000–48000). Defaults to `44100`.
    pub sample_rate: u32,
    /// Tone tuning (speed / volume).
    pub generation_config: GenerationConfig,
    /// API base URL. Defaults to `"https://api.cartesia.ai"`.
    pub base_url: String,
    /// `Cartesia-Version` date header. Defaults to `"2026-03-01"`.
    pub version: String,
}

impl Default for CartesiaTtsConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model_id: "sonic-3.5".to_string(),
            // Cartesia Sonic default voice (overridable per account/voice library).
            voice_id: "f786b574-daa5-4673-aa0c-cbe3e8534c02".to_string(),
            language: "en".to_string(),
            sample_rate: 44100,
            generation_config: GenerationConfig::default(),
            base_url: "https://api.cartesia.ai".to_string(),
            version: "2026-03-01".to_string(),
        }
    }
}

impl CartesiaTtsConfig {
    /// Construct a config with the given API key and the given voice id.
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            ..Default::default()
        }
    }

    /// Override the Sonic model id.
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Override the language code.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Override the output sample rate (Hz).
    pub fn sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the speaking rate (`0.6`–`1.5`).
    pub fn speed(mut self, speed: f32) -> Self {
        self.generation_config.speed = Some(speed);
        self
    }

    /// Set the output volume (`0.5`–`2.0`).
    pub fn volume(mut self, volume: f32) -> Self {
        self.generation_config.volume = Some(volume);
        self
    }

    /// Set the emotion (`"neutral"`, `"calm"`, `"angry"`, `"content"`, `"sad"`).
    pub fn emotion(mut self, emotion: impl Into<String>) -> Self {
        self.generation_config.emotion = Some(emotion.into());
        self
    }

    // ─── Voice tone presets ─────────────────────────────────────────────

    /// Calm, slower delivery — good for soothing or deliberate personas.
    pub fn calm(mut self) -> Self {
        self.generation_config.speed = Some(0.85);
        self.generation_config.emotion = Some("calm".to_string());
        self
    }

    /// Lively, faster delivery — good for energetic, upbeat personas.
    pub fn lively(mut self) -> Self {
        self.generation_config.speed = Some(1.15);
        self.generation_config.emotion = Some("content".to_string());
        self
    }

    /// Steady narration pace at normal volume — good for read-aloud content.
    pub fn narration(mut self) -> Self {
        self.generation_config.speed = Some(0.95);
        self.generation_config.volume = Some(1.0);
        self.generation_config.emotion = Some("neutral".to_string());
        self
    }
}

pub struct CartesiaTts {
    client: reqwest::Client,
    config: CartesiaTtsConfig,
}

impl CartesiaTts {
    pub fn new(config: CartesiaTtsConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }

    /// Synthesize `text` with a per-request voice tone, overriding the config's
    /// [`GenerationConfig`].
    ///
    /// Use this when the tone is decided dynamically (e.g. an LLM picks the
    /// emotion per turn) rather than fixed at construction time.
    pub async fn synthesize_with(&self, text: &str, tone: &GenerationConfig) -> Result<Vec<u8>> {
        self.request(text, tone).await
    }

    /// Build and send the `/tts/bytes` request with the given tone, returning
    /// the raw audio bytes.
    async fn request(&self, text: &str, tone: &GenerationConfig) -> Result<Vec<u8>> {
        let url = format!("{}/tts/bytes", self.config.base_url);

        let request = TtsRequest {
            model_id: &self.config.model_id,
            transcript: text,
            voice: VoiceSpec {
                mode: "id",
                id: &self.config.voice_id,
            },
            output_format: OutputFormat {
                // WAV container so downstream rodio/transport can decode directly.
                container: "wav",
                encoding: "pcm_s16le",
                sample_rate: self.config.sample_rate,
            },
            language: &self.config.language,
            generation_config: if tone.is_empty() { None } else { Some(tone) },
        };

        let response = self
            .client
            .post(&url)
            .header("X-API-Key", &self.config.api_key)
            .header("Cartesia-Version", &self.config.version)
            .json(&request)
            .send()
            .await
            .map_err(|e| MindroidError::Pipeline {
                stage: "CartesiaTts".into(),
                message: format!("HTTP error: {e}"),
                source: None,
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(MindroidError::Pipeline {
                stage: "CartesiaTts".into(),
                message: format!("Cartesia TTS returned {status}: {body}"),
                source: None,
            });
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| MindroidError::Pipeline {
                stage: "CartesiaTts".into(),
                message: format!("Failed to read Cartesia TTS audio bytes: {e}"),
                source: None,
            })?;

        Ok(bytes.to_vec())
    }
}

#[derive(Serialize)]
struct VoiceSpec<'a> {
    mode: &'static str,
    id: &'a str,
}

#[derive(Serialize)]
struct OutputFormat {
    container: &'static str,
    encoding: &'static str,
    sample_rate: u32,
}

#[derive(Serialize)]
struct TtsRequest<'a> {
    model_id: &'a str,
    transcript: &'a str,
    voice: VoiceSpec<'a>,
    output_format: OutputFormat,
    language: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<&'a GenerationConfig>,
}

#[async_trait]
impl TtsProvider for CartesiaTts {
    async fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        self.request(text, &self.config.generation_config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_request<'a>(cfg: &'a CartesiaTtsConfig, text: &'a str) -> serde_json::Value {
        let req = TtsRequest {
            model_id: &cfg.model_id,
            transcript: text,
            voice: VoiceSpec {
                mode: "id",
                id: &cfg.voice_id,
            },
            output_format: OutputFormat {
                container: "wav",
                encoding: "pcm_s16le",
                sample_rate: cfg.sample_rate,
            },
            language: &cfg.language,
            generation_config: if cfg.generation_config.is_empty() {
                None
            } else {
                Some(&cfg.generation_config)
            },
        };
        serde_json::to_value(&req).unwrap()
    }

    #[test]
    fn request_body_matches_cartesia_schema() {
        let cfg = CartesiaTtsConfig::new("key", "voice-123");
        let body = build_request(&cfg, "hello");

        assert_eq!(body["model_id"], "sonic-3.5");
        assert_eq!(body["transcript"], "hello");
        assert_eq!(body["voice"]["mode"], "id");
        assert_eq!(body["voice"]["id"], "voice-123");
        assert_eq!(body["output_format"]["container"], "wav");
        assert_eq!(body["output_format"]["encoding"], "pcm_s16le");
        assert_eq!(body["output_format"]["sample_rate"], 44100);
        assert_eq!(body["language"], "en");
        // No tone tuning set → generation_config omitted entirely.
        assert!(body.get("generation_config").is_none());
    }

    fn approx(value: &serde_json::Value, expected: f64) -> bool {
        value.as_f64().is_some_and(|v| (v - expected).abs() < 1e-4)
    }

    #[test]
    fn tone_presets_set_generation_config() {
        let calm = build_request(&CartesiaTtsConfig::new("k", "v").calm(), "hi");
        assert!(approx(&calm["generation_config"]["speed"], 0.85));
        assert_eq!(calm["generation_config"]["emotion"], "calm");
        assert!(calm["generation_config"].get("volume").is_none());

        let narration = build_request(&CartesiaTtsConfig::new("k", "v").narration(), "hi");
        assert!(approx(&narration["generation_config"]["speed"], 0.95));
        assert!(approx(&narration["generation_config"]["volume"], 1.0));
        assert_eq!(narration["generation_config"]["emotion"], "neutral");
    }

    #[test]
    fn emotion_only_tone_serializes() {
        let cfg = CartesiaTtsConfig::new("k", "v").emotion("sad");
        let body = build_request(&cfg, "hi");
        assert_eq!(body["generation_config"]["emotion"], "sad");
        assert!(body["generation_config"].get("speed").is_none());
    }

    #[test]
    fn builder_overrides_apply() {
        let cfg = CartesiaTtsConfig::new("k", "v")
            .model_id("sonic-3")
            .language("fr")
            .sample_rate(24000)
            .speed(1.2)
            .volume(0.9);
        let body = build_request(&cfg, "bonjour");

        assert_eq!(body["model_id"], "sonic-3");
        assert_eq!(body["language"], "fr");
        assert_eq!(body["output_format"]["sample_rate"], 24000);
        assert!(approx(&body["generation_config"]["speed"], 1.2));
        assert!(approx(&body["generation_config"]["volume"], 0.9));
    }
}
