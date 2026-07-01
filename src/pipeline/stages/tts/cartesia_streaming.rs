use base64::{Engine, engine::general_purpose::STANDARD};
use futures::stream::BoxStream;
use futures::{SinkExt, StreamExt};
use serde_json::json;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;

use crate::MindroidError;
use crate::error::Result;

use super::{CartesiaTtsConfig, GenerationConfig};

/// Streaming text-to-speech over Cartesia Sonic's websocket API.
///
/// Unlike [`CartesiaTts`](super::CartesiaTts) — which POSTs to `/tts/bytes` and
/// returns one finished WAV — this opens `wss://.../tts/websocket` and yields
/// **raw PCM audio chunks as they are generated**, so playback (or forwarding to
/// a downstream client) can start on the first chunk. Output is mono
/// `pcm_s16le` at [`sample_rate`](Self::sample_rate).
///
/// Reuses [`CartesiaTtsConfig`] for the voice, model, sample rate, language, and
/// tone. Each [`stream`](Self::stream) call opens its own short-lived websocket
/// connection, sends the full transcript as a single request, streams the audio
/// back, then closes.
///
/// # Example
///
/// ```ignore
/// use futures::StreamExt;
/// use mindroid::{CartesiaStreamingTts, CartesiaTtsConfig, CartesiaGenerationConfig};
///
/// let tts = CartesiaStreamingTts::new(CartesiaTtsConfig::new(api_key, voice_id));
/// let tone = CartesiaGenerationConfig { emotion: Some("content".into()), ..Default::default() };
/// let mut chunks = tts.stream("Hello there!", &tone);
/// while let Some(chunk) = chunks.next().await {
///     let pcm: Vec<u8> = chunk?; // mono pcm_s16le @ tts.sample_rate()
///     // play it, or forward it over your own websocket
/// }
/// ```
pub struct CartesiaStreamingTts {
    config: CartesiaTtsConfig,
}

impl CartesiaStreamingTts {
    pub fn new(config: CartesiaTtsConfig) -> Self {
        Self { config }
    }

    /// The sample rate (Hz) of the emitted PCM. Consumers need this to play or
    /// re-encode the raw `pcm_s16le` chunks.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Stream synthesized audio for `text` with the given voice `tone`.
    ///
    /// Returns a stream of raw mono `pcm_s16le` chunks (little-endian `i16`
    /// samples). The stream ends after Cartesia's `done` message; a transport or
    /// API error is surfaced as a single `Err` item.
    pub fn stream(
        &self,
        text: &str,
        tone: &GenerationConfig,
    ) -> BoxStream<'static, Result<Vec<u8>>> {
        let c = &self.config;

        // Derive the websocket URL from the (http) base_url.
        let ws_base = c
            .base_url
            .replacen("https://", "wss://", 1)
            .replacen("http://", "ws://", 1);
        let url = format!("{ws_base}/tts/websocket?cartesia_version={}", c.version);
        let api_key = c.api_key.clone();

        let tone_empty = tone.speed.is_none() && tone.volume.is_none() && tone.emotion.is_none();

        let mut request = json!({
            "model_id": c.model_id,
            "transcript": text,
            "voice": { "mode": "id", "id": c.voice_id },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": c.sample_rate,
            },
            "language": c.language,
            "context_id": uuid::Uuid::new_v4().to_string(),
            "continue": false,
        });
        if !tone_empty && let Ok(v) = serde_json::to_value(tone) {
            request["generation_config"] = v;
        }
        let body = request.to_string();

        Box::pin(async_stream::stream! {
            let mut req = match url.into_client_request() {
                Ok(r) => r,
                Err(e) => {
                    yield Err(ws_err(format!("invalid websocket url: {e}")));
                    return;
                }
            };
            match HeaderValue::from_str(&api_key) {
                Ok(v) => {
                    req.headers_mut().insert("X-API-Key", v);
                }
                Err(e) => {
                    yield Err(ws_err(format!("invalid api key header: {e}")));
                    return;
                }
            }

            let (ws, _resp) = match connect_async(req).await {
                Ok(v) => v,
                Err(e) => {
                    yield Err(ws_err(format!("connect failed: {e}")));
                    return;
                }
            };
            let (mut write, mut read) = ws.split();

            if let Err(e) = write.send(WsMessage::Text(body)).await {
                yield Err(ws_err(format!("send failed: {e}")));
                return;
            }

            while let Some(msg) = read.next().await {
                match msg {
                    Ok(WsMessage::Text(text)) => {
                        let v: serde_json::Value = match serde_json::from_str(&text) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        match v.get("type").and_then(|t| t.as_str()) {
                            Some("chunk") => {
                                if let Some(data) = v.get("data").and_then(|d| d.as_str()) {
                                    match STANDARD.decode(data) {
                                        Ok(bytes) => yield Ok(bytes),
                                        Err(e) => {
                                            yield Err(ws_err(format!("bad base64 chunk: {e}")));
                                        }
                                    }
                                }
                            }
                            Some("done") => break,
                            Some("error") => {
                                let m = v
                                    .get("error")
                                    .and_then(|e| e.as_str())
                                    .or_else(|| v.get("message").and_then(|e| e.as_str()))
                                    .unwrap_or("unknown error");
                                yield Err(ws_err(format!("Cartesia TTS error: {m}")));
                                break;
                            }
                            _ => {}
                        }
                    }
                    // Some deployments may frame audio as binary.
                    Ok(WsMessage::Binary(b)) => yield Ok(b),
                    Ok(WsMessage::Close(_)) => break,
                    Ok(_) => {}
                    Err(e) => {
                        yield Err(ws_err(format!("websocket error: {e}")));
                        break;
                    }
                }
            }

            let _ = write.send(WsMessage::Close(None)).await;
        })
    }
}

fn ws_err(message: String) -> MindroidError {
    MindroidError::Pipeline {
        stage: "CartesiaStreamingTts".into(),
        message,
        source: None,
    }
}
