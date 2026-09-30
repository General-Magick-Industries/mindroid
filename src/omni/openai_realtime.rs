//! OpenAI Realtime (GA `gpt-realtime`) implementation of [`OmniProvider`].
//!
//! Speaks the GA event names (`response.output_audio.delta`, not the beta
//! `response.audio.delta`). Works direct against OpenAI or through a LiteLLM
//! `/v1/realtime` passthrough — the wire format is identical.

use std::borrow::Cow;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use futures::{SinkExt, Stream, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_stream::wrappers::ReceiverStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};
use tracing::warn;

use crate::core::error::MindroidError;
use crate::omni::provider::OmniProvider;
use crate::omni::types::{
    AudioChunk, HistoryTurn, OmniConfig, OmniEvent, Role, TranscriptSource, TurnDetection, Usage,
};
use crate::tools::Tool;

/// OpenAI Realtime WebSocket endpoint; a LiteLLM passthrough works at `/v1/realtime` too.
pub const DEFAULT_ENDPOINT: &str = "wss://api.openai.com/v1/realtime";
/// GA realtime model.
pub const DEFAULT_MODEL: &str = "gpt-realtime";

/// OpenAI Realtime is symmetric: 24 kHz PCM16 in both directions.
pub const SAMPLE_RATE: u32 = 24_000;

/// The server rejects a manual commit under 100 ms of audio.
const MIN_COMMIT_BYTES: usize = (SAMPLE_RATE as usize / 10) * 2;

fn transport(message: impl Into<String>) -> MindroidError {
    MindroidError::Transport {
        message: message.into(),
        source: None,
    }
}

/// Connection settings for [`OpenAiRealtimeProvider`].
#[derive(Clone)]
pub struct OpenAiRealtimeConfig {
    pub api_key: String,
    pub model: String,
    pub endpoint: String,
    /// Model for user-speech transcription (`whisper-1`). `None` disables it —
    /// unlike Gemini, OpenAI does not transcribe input unless asked.
    pub input_transcription_model: Option<String>,
    /// Permit a plaintext `ws://` endpoint; the key rides the Authorization header.
    pub allow_insecure: bool,
    /// A proxy in front of the model creates each reply itself (some do, after
    /// input transcription). Server VAD then only segments and interrupts; with
    /// nothing in front that creates replies, the agent is silent.
    pub proxy_creates_responses: bool,
}

impl fmt::Debug for OpenAiRealtimeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiRealtimeConfig")
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("endpoint", &self.endpoint)
            .field("input_transcription_model", &self.input_transcription_model)
            .field("allow_insecure", &self.allow_insecure)
            .field("proxy_creates_responses", &self.proxy_creates_responses)
            .finish()
    }
}

impl OpenAiRealtimeConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            input_transcription_model: Some("whisper-1".to_string()),
            allow_insecure: false,
            proxy_creates_responses: false,
        }
    }

    /// Read `OPENAI_API_KEY` + optional `OPENAI_REALTIME_URL`, falling back to the
    /// LiteLLM pair (`LITELLM_APIKEY`, `LITELLM_URL` → `wss://…/v1/realtime`).
    ///
    /// # Errors
    ///
    /// Returns [`MindroidError::Config`] when neither key is set.
    pub fn from_env() -> Result<Self, MindroidError> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());

        if let Some(key) = var("OPENAI_API_KEY") {
            let mut cfg = Self::new(key);
            if let Some(url) = var("OPENAI_REALTIME_URL") {
                cfg.endpoint = url;
            }
            return Ok(cfg);
        }

        let key = var("LITELLM_APIKEY")
            .ok_or_else(|| MindroidError::config("set OPENAI_API_KEY or LITELLM_APIKEY"))?;
        let base = var("LITELLM_URL")
            .ok_or_else(|| MindroidError::config("LITELLM_APIKEY set but LITELLM_URL is not"))?;
        let ws_base = base
            .trim_end_matches('/')
            .replacen("https://", "wss://", 1)
            .replacen("http://", "ws://", 1);
        Ok(Self::new(key).with_endpoint(format!("{ws_base}/v1/realtime")))
    }

    /// Override the model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the WebSocket endpoint (a proxy, or a local fake in tests).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Allow a plaintext `ws://` endpoint. Never for a real key.
    pub fn with_allow_insecure(mut self, allow: bool) -> Self {
        self.allow_insecure = allow;
        self
    }

    /// Let the proxy create replies; see [`Self::proxy_creates_responses`].
    pub fn with_proxy_creates_responses(mut self, yes: bool) -> Self {
        self.proxy_creates_responses = yes;
        self
    }

    /// Skip user-speech transcription (no `Transcript{Input}` events).
    pub fn without_input_transcription(mut self) -> Self {
        self.input_transcription_model = None;
        self
    }

    fn url(&self) -> String {
        let sep = if self.endpoint.contains('?') {
            '&'
        } else {
            '?'
        };
        format!("{}{sep}model={}", self.endpoint, self.model)
    }
}

/// Render tools as OpenAI Realtime function tools, ready for [`OmniConfig::tools_schema`].
///
/// OpenAI accepts standard JSON Schema for `parameters`, so nothing is stripped.
pub fn tool_declarations(tools: &[Arc<dyn Tool>]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "name": t.name(),
                    "description": t.description(),
                    "parameters": t.parameters_schema(),
                })
            })
            .collect(),
    )
}

/// [`OmniProvider`] over the GA OpenAI Realtime WebSocket protocol.
pub struct OpenAiRealtimeProvider {
    config: OpenAiRealtimeConfig,
    outbound: Option<mpsc::Sender<String>>,
    events: Option<mpsc::Receiver<OmniEvent>>,
    manual_turns: bool,
    /// Bytes appended since the last commit. An empty commit is a server error.
    uncommitted: Arc<AtomicUsize>,
    tasks: JoinSet<()>,
}

impl OpenAiRealtimeProvider {
    pub fn new(config: OpenAiRealtimeConfig) -> Self {
        Self {
            config,
            outbound: None,
            events: None,
            manual_turns: false,
            uncommitted: Arc::new(AtomicUsize::new(0)),
            tasks: JoinSet::new(),
        }
    }

    fn session_update(&self, config: &OmniConfig) -> Value {
        let pcm = json!({ "type": "audio/pcm", "rate": SAMPLE_RATE });

        // Local VAD is hybrid here too: server VAD stays on, `commit` finalizes early.
        let turn_detection = match config.turn_detection {
            TurnDetection::Manual => Value::Null,
            // Server default is 200 ms of silence, which cuts in mid-sentence.
            _ => json!({
                "type": "server_vad",
                "threshold": 0.6,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 700,
                "create_response": !self.config.proxy_creates_responses,
                "interrupt_response": true,
            }),
        };

        // Explicit null: a proxy may have injected transcription at connect, and
        // omitting the field would leave that in place.
        let transcription = match &self.config.input_transcription_model {
            Some(model) => json!({ "model": model }),
            None => Value::Null,
        };
        let input = json!({
            "format": pcm,
            "turn_detection": turn_detection,
            "transcription": transcription,
        });

        let mut output = json!({ "format": pcm });
        if let Some(voice) = &config.voice {
            output["voice"] = json!(voice);
        }

        let mut session = json!({
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": { "input": input, "output": output },
        });
        if let Some(prompt) = &config.system_prompt {
            session["instructions"] = json!(prompt);
        }
        if let Some(tools) = &config.tools_schema {
            session["tools"] = tools.clone();
            session["tool_choice"] = json!("auto");
        }

        json!({ "type": "session.update", "session": session })
    }

    async fn send_json(&self, payload: Value) -> Result<(), MindroidError> {
        let tx = self
            .outbound
            .as_ref()
            .ok_or_else(|| transport("OpenAI Realtime provider is not connected"))?;
        tx.send(payload.to_string())
            .await
            .map_err(|_| transport("OpenAI Realtime writer task has stopped"))
    }
}

#[async_trait]
impl OmniProvider for OpenAiRealtimeProvider {
    async fn connect(&mut self, config: &OmniConfig) -> Result<(), MindroidError> {
        self.manual_turns = matches!(config.turn_detection, TurnDetection::Manual);

        if self.config.endpoint.starts_with("ws://") && !self.config.allow_insecure {
            return Err(transport(
                "refusing to send the API key over plaintext ws://; set allow_insecure for a local fake",
            ));
        }
        let mut request = self
            .config
            .url()
            .into_client_request()
            .map_err(|e| transport(format!("OpenAI Realtime bad endpoint: {e}")))?;
        let bearer = HeaderValue::from_str(&format!("Bearer {}", self.config.api_key))
            .map_err(|_| transport("OpenAI Realtime api key is not a valid header value"))?;
        request.headers_mut().insert("Authorization", bearer);

        let (ws, _) = connect_async(request)
            .await
            .map_err(|e| transport(format!("OpenAI Realtime connect failed: {e}")))?;
        let (mut sink, mut stream) = ws.split();

        wait_for(&mut stream, "session.created").await?;

        sink.send(WsMessage::Text(self.session_update(config).to_string()))
            .await
            .map_err(|e| transport(format!("OpenAI Realtime session.update failed: {e}")))?;

        wait_for(&mut stream, "session.updated").await?;

        for turn in &config.history {
            sink.send(WsMessage::Text(history_item(turn).to_string()))
                .await
                .map_err(|e| transport(format!("OpenAI Realtime history seed failed: {e}")))?;
        }

        let (out_tx, mut out_rx) = mpsc::channel::<String>(64);
        let (event_tx, event_rx) = mpsc::channel::<OmniEvent>(256);

        self.tasks.spawn(async move {
            while let Some(text) = out_rx.recv().await {
                if !text.contains("\"input_audio_buffer.append\"") {
                    let head: String = text.chars().take(160).collect();
                    tracing::debug!(frame = %head, "-> server");
                }
                if sink.send(WsMessage::Text(text)).await.is_err() {
                    break;
                }
            }
            let _ = sink.close().await;
        });

        let uncommitted = Arc::clone(&self.uncommitted);
        self.tasks.spawn(async move {
            while let Some(frame) = stream.next().await {
                let frame = match frame {
                    Ok(f) => f,
                    Err(e) => {
                        let _ = event_tx
                            .send(OmniEvent::Error(Arc::new(transport(format!(
                                "OpenAI Realtime stream error: {e}"
                            )))))
                            .await;
                        break;
                    }
                };
                let Some(value) = frame_to_json(frame) else {
                    continue;
                };
                let kind = value.get("type").and_then(Value::as_str).unwrap_or("?");
                if kind != "response.output_audio.delta" {
                    let id = |p: &str| value.pointer(p).and_then(Value::as_str).unwrap_or("");
                    tracing::debug!(
                        kind,
                        response = id("/response/id"),
                        item = id("/item_id"),
                        "<- server"
                    );
                }
                if matches!(
                    value.get("type").and_then(Value::as_str),
                    Some("input_audio_buffer.committed" | "input_audio_buffer.speech_stopped")
                ) {
                    uncommitted.store(0, Ordering::Relaxed);
                }
                for event in parse_server_event(&value) {
                    if event_tx.send(event).await.is_err() {
                        return;
                    }
                }
            }
        });

        self.outbound = Some(out_tx);
        self.events = Some(event_rx);
        Ok(())
    }

    async fn send_audio(&self, chunk: AudioChunk) -> Result<(), MindroidError> {
        let data = downsample_to(&chunk.data, chunk.sample_rate, SAMPLE_RATE)?;
        self.uncommitted.fetch_add(data.len(), Ordering::Relaxed);
        self.send_json(json!({
            "type": "input_audio_buffer.append",
            "audio": STANDARD.encode(&*data),
        }))
        .await
    }

    async fn send_text(&self, text: &str) -> Result<(), MindroidError> {
        self.send_json(json!({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": text }],
            }
        }))
        .await?;
        self.send_json(json!({ "type": "response.create" })).await
    }

    async fn send_tool_result(&self, call_id: &str, result: Value) -> Result<(), MindroidError> {
        // `output` must be a string; anything structured is serialized.
        let output = match result {
            Value::String(s) => s,
            other => other.to_string(),
        };
        self.send_json(json!({
            "type": "conversation.item.create",
            "item": { "type": "function_call_output", "call_id": call_id, "output": output },
        }))
        .await?;
        self.send_json(json!({ "type": "response.create" })).await
    }

    async fn end_audio_stream(&self) -> Result<(), MindroidError> {
        if self.uncommitted.swap(0, Ordering::Relaxed) < MIN_COMMIT_BYTES {
            return Ok(());
        }
        self.send_json(json!({ "type": "input_audio_buffer.commit" }))
            .await?;
        if self.manual_turns {
            self.send_json(json!({ "type": "response.create" })).await?;
        }
        Ok(())
    }

    fn events(&mut self) -> Pin<Box<dyn Stream<Item = OmniEvent> + Send>> {
        match self.events.take() {
            Some(rx) => Box::pin(ReceiverStream::new(rx)),
            None => Box::pin(futures::stream::empty()),
        }
    }

    async fn disconnect(&mut self) -> Result<(), MindroidError> {
        self.outbound = None;
        self.tasks.abort_all();
        self.events = None;
        Ok(())
    }
}

type WsStream = futures::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

/// Block until the server sends `expected`, surfacing an `error` event as a failure.
async fn wait_for(stream: &mut WsStream, expected: &str) -> Result<(), MindroidError> {
    loop {
        let frame = stream
            .next()
            .await
            .ok_or_else(|| transport(format!("OpenAI Realtime closed before {expected}")))?
            .map_err(|e| transport(format!("OpenAI Realtime read failed: {e}")))?;
        let Some(value) = frame_to_json(frame) else {
            continue;
        };
        match value.get("type").and_then(Value::as_str) {
            Some(t) if t == expected => return Ok(()),
            Some("error") => {
                let msg = value
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error");
                return Err(transport(format!(
                    "OpenAI Realtime rejected session: {msg}"
                )));
            }
            _ => {}
        }
    }
}

/// One prior turn as a conversation item. Assistant text uses the GA `output_text`
/// content type; user text uses `input_text`.
fn history_item(turn: &HistoryTurn) -> Value {
    let (role, kind) = match turn.role {
        Role::User => ("user", "input_text"),
        Role::Model => ("assistant", "output_text"),
    };
    json!({
        "type": "conversation.item.create",
        "item": { "type": "message", "role": role, "content": [{ "type": kind, "text": turn.text }] },
    })
}

/// Integer-ratio decimation by averaging each group of `from / to` samples.
///
/// Interim until the crate has a real resampler. A non-integer ratio is refused
/// loudly rather than sent at the wrong pitch — which the server accepts without
/// complaint and then fails to understand a word of.
fn downsample_to(pcm: &[u8], from: u32, to: u32) -> Result<Cow<'_, [u8]>, MindroidError> {
    if from == to {
        return Ok(Cow::Borrowed(pcm));
    }
    if from == 0 || to == 0 || !from.is_multiple_of(to) {
        return Err(transport(format!(
            "capture rate {from} Hz is not an integer multiple of {to} Hz; resample before send_audio"
        )));
    }
    let n = (from / to) as usize;
    let samples: Vec<i16> = pcm
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(i16::from_le_bytes)
        .collect();
    let out: Vec<u8> = samples
        .chunks_exact(n)
        .flat_map(|group| {
            let avg = group.iter().map(|&s| i32::from(s)).sum::<i32>() / n as i32;
            (avg as i16).to_le_bytes()
        })
        .collect();
    Ok(Cow::Owned(out))
}

fn frame_to_json(frame: WsMessage) -> Option<Value> {
    let bytes = match frame {
        WsMessage::Text(text) => text.into_bytes(),
        WsMessage::Binary(bytes) => bytes,
        _ => return None,
    };
    serde_json::from_slice(&bytes).ok()
}

/// Map one server event onto zero or more [`OmniEvent`]s.
fn parse_server_event(value: &Value) -> Vec<OmniEvent> {
    let Some(kind) = value.get("type").and_then(Value::as_str) else {
        return Vec::new();
    };
    let text = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);

    match kind {
        "response.output_audio.delta" => text("delta")
            .and_then(|d| STANDARD.decode(d).ok())
            .map(|data| {
                vec![OmniEvent::AudioChunk(AudioChunk {
                    data,
                    sample_rate: SAMPLE_RATE,
                    channels: 1,
                    bits_per_sample: 16,
                })]
            })
            .unwrap_or_default(),

        "response.output_audio_transcript.delta" => text("delta")
            .map(|t| vec![transcript(t, false, TranscriptSource::Output)])
            .unwrap_or_default(),
        "response.output_audio_transcript.done" => text("transcript")
            .map(|t| vec![transcript(t, true, TranscriptSource::Output)])
            .unwrap_or_default(),
        "conversation.item.input_audio_transcription.delta" => text("delta")
            .map(|t| vec![transcript(t, false, TranscriptSource::Input)])
            .unwrap_or_default(),
        "conversation.item.input_audio_transcription.completed" => text("transcript")
            .map(|t| vec![transcript(t, true, TranscriptSource::Input)])
            .unwrap_or_default(),

        // With `interrupt_response`, the server cancels its own response on speech
        // start, so no client-side cancel is sent; this is the barge-in signal.
        "input_audio_buffer.speech_started" => vec![OmniEvent::Interrupted],
        "input_audio_buffer.speech_stopped" => vec![OmniEvent::UserSpeechEnded],
        "response.done" => {
            let mut events = Vec::new();
            if let Some(u) = value.pointer("/response/usage") {
                let n = |p: &str| u.pointer(p).and_then(Value::as_u64).unwrap_or(0);
                events.push(OmniEvent::Usage(Usage {
                    input_tokens: n("/input_tokens"),
                    output_tokens: n("/output_tokens"),
                    input_audio_tokens: n("/input_token_details/audio_tokens"),
                    output_audio_tokens: n("/output_token_details/audio_tokens"),
                }));
            }
            // Tool calls are surfaced here, not at `function_call_arguments.done`:
            // the host answers with `response.create`, which the server rejects
            // while this response is still active.
            let calls = value
                .pointer("/response/output")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|item| item["type"] == "function_call")
                .filter_map(|item| {
                    let id = item["call_id"].as_str()?.to_string();
                    let name = item["name"].as_str()?.to_string();
                    let args = item["arguments"]
                        .as_str()
                        .and_then(|a| serde_json::from_str(a).ok())
                        .unwrap_or_else(|| json!({}));
                    Some(OmniEvent::ToolCall { id, name, args })
                });
            events.extend(calls);
            events.push(OmniEvent::TurnComplete);
            events
        }

        "error" => {
            let err = &value["error"];
            let kind = err.get("type").and_then(Value::as_str).unwrap_or("unknown");
            let code = err.get("code").and_then(Value::as_str).unwrap_or("");
            let msg = err
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            // Only `server_error` is fatal. `invalid_request_error` covers recoverable
            // timing complaints server-VAD produces on its own, e.g. a new turn
            // starting while the previous response is still streaming.
            // Only these are the server complaining about its own timing. Anything
            // else is a request of ours being rejected, and must surface.
            const RECOVERABLE: [&str; 3] = [
                "conversation_already_has_active_response",
                "input_audio_buffer_commit_empty",
                "response_cancel_not_active",
            ];
            if kind == "server_error" || !RECOVERABLE.contains(&code) {
                vec![OmniEvent::Error(Arc::new(transport(format!(
                    "OpenAI Realtime rejected a request ({kind}/{code}): {msg}"
                ))))]
            } else {
                warn!(kind, code, %msg, "OpenAI Realtime recoverable error");
                Vec::new()
            }
        }

        _ => Vec::new(),
    }
}

fn transcript(text: String, is_final: bool, source: TranscriptSource) -> OmniEvent {
    OmniEvent::Transcript {
        text,
        is_final,
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream::SplitSink;
    use std::sync::Mutex;
    use tokio::net::TcpStream;
    use tokio_tungstenite::WebSocketStream;
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

    type Sink = SplitSink<WebSocketStream<TcpStream>, WsMessage>;

    /// Minimal Realtime server: records the upgrade's Authorization header, completes
    /// the created → update → updated handshake, then replies per `script`.
    async fn spawn_fake_openai<F>(
        script: F,
    ) -> (String, mpsc::Receiver<Value>, Arc<Mutex<Option<String>>>)
    where
        F: Fn(&Value) -> Vec<Value> + Send + 'static,
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (seen_tx, seen_rx) = mpsc::channel::<Value>(64);
        let auth_header: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let auth_capture = Arc::clone(&auth_header);

        tokio::spawn(async move {
            let Ok((tcp, _)) = listener.accept().await else {
                return;
            };
            // Err type is fixed by tungstenite's `Callback` bound; nothing to shrink.
            #[allow(clippy::result_large_err)]
            let callback = move |req: &Request, resp: Response| {
                let header = req
                    .headers()
                    .get("Authorization")
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string);
                *auth_capture.lock().unwrap() = header;
                Ok(resp)
            };
            let Ok(ws) = tokio_tungstenite::accept_hdr_async(tcp, callback).await else {
                return;
            };
            let (mut sink, mut stream) = ws.split();

            let _ = send(
                &mut sink,
                json!({ "type": "session.created", "session": {} }),
            )
            .await;

            while let Some(Ok(frame)) = stream.next().await {
                let Some(value) = frame_to_json(frame) else {
                    continue;
                };
                let is_update = value["type"] == "session.update";
                let replies = script(&value);
                if seen_tx.send(value).await.is_err() {
                    return;
                }
                if is_update {
                    let _ = send(
                        &mut sink,
                        json!({ "type": "session.updated", "session": {} }),
                    )
                    .await;
                }
                for reply in replies {
                    if send(&mut sink, reply).await.is_err() {
                        return;
                    }
                }
            }
        });

        (format!("ws://{addr}/v1/realtime"), seen_rx, auth_header)
    }

    #[allow(clippy::result_large_err)]
    async fn send(
        sink: &mut Sink,
        value: Value,
    ) -> Result<(), tokio_tungstenite::tungstenite::Error> {
        sink.send(WsMessage::Text(value.to_string())).await
    }

    async fn connected(
        script: impl Fn(&Value) -> Vec<Value> + Send + 'static,
    ) -> (
        OpenAiRealtimeProvider,
        mpsc::Receiver<Value>,
        Arc<Mutex<Option<String>>>,
    ) {
        let (url, seen, auth) = spawn_fake_openai(script).await;
        let config = OpenAiRealtimeConfig::new("test-key")
            .with_endpoint(url)
            .with_allow_insecure(true);
        let mut provider = OpenAiRealtimeProvider::new(config);
        provider.connect(&OmniConfig::default()).await.unwrap();
        (provider, seen, auth)
    }

    async fn connected_with(
        config: OmniConfig,
        script: impl Fn(&Value) -> Vec<Value> + Send + 'static,
    ) -> (OpenAiRealtimeProvider, mpsc::Receiver<Value>) {
        let (url, seen, _) = spawn_fake_openai(script).await;
        let mut provider = OpenAiRealtimeProvider::new(
            OpenAiRealtimeConfig::new("k")
                .with_endpoint(url)
                .with_allow_insecure(true),
        );
        provider.connect(&config).await.unwrap();
        (provider, seen)
    }

    #[tokio::test]
    async fn history_is_seeded_after_session_update_in_order() {
        let config = OmniConfig {
            history: vec![
                HistoryTurn {
                    role: Role::User,
                    text: "hi".into(),
                },
                HistoryTurn {
                    role: Role::Model,
                    text: "hello".into(),
                },
            ],
            ..OmniConfig::default()
        };
        let (mut provider, mut seen) = connected_with(config, |_| vec![]).await;

        let update = seen.recv().await.expect("session.update");
        assert_eq!(update["type"], "session.update");

        let user = seen.recv().await.expect("user item");
        assert_eq!(user["type"], "conversation.item.create");
        assert_eq!(user["item"]["role"], "user");
        assert_eq!(user["item"]["content"][0]["type"], "input_text");
        assert_eq!(user["item"]["content"][0]["text"], "hi");

        let model = seen.recv().await.expect("assistant item");
        assert_eq!(model["item"]["role"], "assistant");
        assert_eq!(model["item"]["content"][0]["type"], "output_text");
        assert_eq!(model["item"]["content"][0]["text"], "hello");
        provider.disconnect().await.unwrap();
    }

    #[tokio::test]
    async fn connect_sends_bearer_header_and_session_update() {
        let (mut provider, mut seen, auth) = connected(|_| vec![]).await;

        assert_eq!(auth.lock().unwrap().as_deref(), Some("Bearer test-key"));

        let update = seen.recv().await.expect("session.update");
        assert_eq!(update["type"], "session.update");
        assert_eq!(update["session"]["type"], "realtime");
        assert_eq!(update["session"]["output_modalities"][0], "audio");
        assert_eq!(
            update["session"]["audio"]["input"]["format"]["rate"],
            24_000
        );
        assert_eq!(
            update["session"]["audio"]["input"]["turn_detection"]["type"],
            "server_vad"
        );
        assert_eq!(
            update["session"]["audio"]["input"]["transcription"]["model"],
            "whisper-1"
        );

        provider.disconnect().await.unwrap();
    }

    /// Off must be an explicit null, or a proxy's injected transcription survives.
    #[test]
    fn transcription_off_is_sent_as_explicit_null() {
        let provider = OpenAiRealtimeProvider::new(
            OpenAiRealtimeConfig::new("k").without_input_transcription(),
        );
        let input = &provider.session_update(&OmniConfig::default())["session"]["audio"]["input"];
        assert!(input.get("transcription").is_some(), "key must be present");
        assert!(input["transcription"].is_null());
    }

    #[test]
    fn proxy_owned_replies_keep_server_vad_but_not_auto_response() {
        let provider = OpenAiRealtimeProvider::new(
            OpenAiRealtimeConfig::new("k").with_proxy_creates_responses(true),
        );
        let td = &provider.session_update(&OmniConfig::default())["session"]["audio"]["input"]["turn_detection"];
        assert_eq!(td["type"], "server_vad");
        assert_eq!(td["create_response"], false);
        assert_eq!(td["interrupt_response"], true);
    }

    #[test]
    fn only_manual_turn_detection_disables_server_vad() {
        let provider = OpenAiRealtimeProvider::new(OpenAiRealtimeConfig::new("k"));
        let td = |t: TurnDetection| {
            let cfg = OmniConfig {
                turn_detection: t,
                ..OmniConfig::default()
            };
            provider.session_update(&cfg)["session"]["audio"]["input"]["turn_detection"].clone()
        };
        assert_eq!(td(TurnDetection::Server)["type"], "server_vad");
        assert_eq!(
            td(TurnDetection::Local(
                crate::omni::types::VadConfig::default()
            ))["type"],
            "server_vad"
        );
        assert!(td(TurnDetection::Manual).is_null());
    }

    #[tokio::test]
    async fn connect_fails_when_session_update_is_rejected() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
            let (mut sink, mut stream) = ws.split();
            let _ = send(&mut sink, json!({ "type": "session.created" })).await;
            let _ = stream.next().await;
            let _ = send(
                &mut sink,
                json!({ "type": "error", "error": { "message": "bad voice" } }),
            )
            .await;
        });

        let config = OpenAiRealtimeConfig::new("k")
            .with_endpoint(format!("ws://{addr}/v1/realtime"))
            .with_allow_insecure(true);
        let err = OpenAiRealtimeProvider::new(config)
            .connect(&OmniConfig::default())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("rejected session: bad voice"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn send_audio_appends_base64_and_commit_follows() {
        let (provider, mut seen, _) = connected(|_| vec![]).await;
        let _update = seen.recv().await;

        provider
            .send_audio(AudioChunk {
                data: vec![7u8; MIN_COMMIT_BYTES],
                sample_rate: SAMPLE_RATE,
                channels: 1,
                bits_per_sample: 16,
            })
            .await
            .unwrap();
        provider.end_audio_stream().await.unwrap();

        let append = seen.recv().await.expect("append");
        assert_eq!(append["type"], "input_audio_buffer.append");
        assert_eq!(
            append["audio"],
            STANDARD.encode(vec![7u8; MIN_COMMIT_BYTES])
        );

        let commit = seen.recv().await.expect("commit");
        assert_eq!(commit["type"], "input_audio_buffer.commit");
    }

    /// An empty commit is a server error, so end_audio_stream with nothing
    /// appended must send nothing.
    #[tokio::test]
    async fn end_audio_stream_with_no_audio_sends_nothing() {
        let (provider, mut seen, _) = connected(|_| vec![]).await;
        let _update = seen.recv().await;

        provider.end_audio_stream().await.unwrap();
        provider.send_text("ping").await.unwrap();

        let next = seen.recv().await.expect("frame");
        assert_eq!(
            next["type"], "conversation.item.create",
            "commit must not precede this"
        );
    }

    #[tokio::test]
    async fn tool_call_round_trip() {
        let (mut provider, mut seen, _) = connected(|frame| {
            if frame["type"] == "input_audio_buffer.commit" {
                vec![json!({
                    "type": "response.done",
                    "response": { "output": [{
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Oslo\"}",
                    }] },
                })]
            } else {
                vec![]
            }
        })
        .await;
        let _update = seen.recv().await;

        let mut events = provider.events();
        provider
            .send_audio(AudioChunk {
                data: vec![0; MIN_COMMIT_BYTES],
                sample_rate: SAMPLE_RATE,
                channels: 1,
                bits_per_sample: 16,
            })
            .await
            .unwrap();
        provider.end_audio_stream().await.unwrap();
        let _append = seen.recv().await;
        let _commit = seen.recv().await;

        let event = events.next().await.expect("tool call");
        let OmniEvent::ToolCall { id, name, args } = event else {
            panic!("expected ToolCall, got {event:?}");
        };
        assert_eq!(id, "call_1");
        assert_eq!(name, "get_weather");
        assert_eq!(args["city"], "Oslo");

        provider
            .send_tool_result(&id, json!({ "temp_c": 4 }))
            .await
            .unwrap();

        let output = seen.recv().await.expect("function_call_output");
        assert_eq!(output["type"], "conversation.item.create");
        assert_eq!(output["item"]["type"], "function_call_output");
        assert_eq!(output["item"]["call_id"], "call_1");
        assert_eq!(output["item"]["output"], "{\"temp_c\":4}");

        let create = seen.recv().await.expect("response.create");
        assert_eq!(create["type"], "response.create");
    }

    #[test]
    fn audio_delta_decodes_at_24k() {
        let events = parse_server_event(&json!({
            "type": "response.output_audio.delta",
            "delta": STANDARD.encode([9u8, 9]),
        }));
        assert!(matches!(
            events[0],
            OmniEvent::AudioChunk(AudioChunk {
                sample_rate: 24_000,
                ..
            })
        ));
    }

    #[test]
    fn transcripts_are_distinguishable_and_finality_tracked() {
        let out_delta = parse_server_event(
            &json!({ "type": "response.output_audio_transcript.delta", "delta": "hel" }),
        );
        assert!(matches!(
            out_delta[0],
            OmniEvent::Transcript {
                source: TranscriptSource::Output,
                is_final: false,
                ..
            }
        ));

        let out_done = parse_server_event(
            &json!({ "type": "response.output_audio_transcript.done", "transcript": "hello" }),
        );
        assert!(matches!(
            out_done[0],
            OmniEvent::Transcript {
                source: TranscriptSource::Output,
                is_final: true,
                ..
            }
        ));

        let in_done = parse_server_event(&json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "what time is it",
        }));
        assert!(matches!(
            in_done[0],
            OmniEvent::Transcript {
                source: TranscriptSource::Input,
                is_final: true,
                ..
            }
        ));
    }

    #[test]
    fn speech_started_is_barge_in_and_response_done_is_turn_complete() {
        assert!(matches!(
            parse_server_event(&json!({ "type": "input_audio_buffer.speech_started" }))[0],
            OmniEvent::Interrupted
        ));
        assert!(matches!(
            parse_server_event(&json!({ "type": "response.done", "response": {} }))[0],
            OmniEvent::TurnComplete
        ));
    }

    #[test]
    fn unknown_events_produce_nothing() {
        assert!(parse_server_event(&json!({ "type": "rate_limits.updated" })).is_empty());
        assert!(parse_server_event(&json!({ "nope": 1 })).is_empty());
    }

    #[test]
    fn tool_declarations_are_flat_function_tools() {
        struct T;
        #[async_trait]
        impl Tool for T {
            fn name(&self) -> &str {
                "t"
            }
            fn description(&self) -> &str {
                "d"
            }
            fn parameters_schema(&self) -> Value {
                json!({ "type": "object", "properties": { "a": { "type": "string" } } })
            }
            async fn execute(
                &self,
                _args: Value,
                _ctx: &crate::tools::ToolContext,
            ) -> crate::Result<String> {
                Ok(String::new())
            }
        }
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(T)];
        let decl = tool_declarations(&tools);
        assert_eq!(decl[0]["type"], "function");
        assert_eq!(decl[0]["name"], "t");
        assert_eq!(decl[0]["parameters"]["properties"]["a"]["type"], "string");
    }

    #[test]
    fn from_env_falls_back_to_litellm_and_rewrites_scheme() {
        // SAFETY: single-threaded test body; no other thread reads env concurrently.
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::set_var("LITELLM_APIKEY", "lk");
            std::env::set_var("LITELLM_URL", "https://proxy.example/");
        }
        let cfg = OpenAiRealtimeConfig::from_env().unwrap();
        assert_eq!(cfg.api_key, "lk");
        assert_eq!(cfg.endpoint, "wss://proxy.example/v1/realtime");
        assert_eq!(
            cfg.url(),
            "wss://proxy.example/v1/realtime?model=gpt-realtime"
        );
        unsafe {
            std::env::remove_var("LITELLM_APIKEY");
            std::env::remove_var("LITELLM_URL");
        }
    }

    /// Opens and closes one real session. The fake server cannot tell us whether the
    /// GA `session.update` shape — object-form audio format, transcription model, flat
    /// tools — is what the server actually accepts; only this can.
    #[tokio::test]
    #[ignore = "opens a real session; needs OPENAI_API_KEY or LITELLM_APIKEY + LITELLM_URL"]
    async fn live_handshake_against_real_server() {
        let cfg = match OpenAiRealtimeConfig::from_env() {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("skipping live handshake: {e}");
                return;
            }
        };
        eprintln!("endpoint: {}", cfg.url());

        struct SmokeTool;
        #[async_trait]
        impl Tool for SmokeTool {
            fn name(&self) -> &str {
                "smoke"
            }
            fn description(&self) -> &str {
                "never called"
            }
            fn parameters_schema(&self) -> Value {
                json!({ "type": "object", "properties": { "x": { "type": "string" } } })
            }
            async fn execute(
                &self,
                _args: Value,
                _ctx: &crate::tools::ToolContext,
            ) -> crate::Result<String> {
                Ok(String::new())
            }
        }
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(SmokeTool)];
        let config = OmniConfig {
            system_prompt: Some("Smoke test. Say nothing.".into()),
            tools_schema: Some(tool_declarations(&tools)),
            voice: Some("alloy".into()),
            history: vec![
                HistoryTurn {
                    role: Role::User,
                    text: "Hello there.".into(),
                },
                HistoryTurn {
                    role: Role::Model,
                    text: "Hi! How can I help?".into(),
                },
            ],
            ..OmniConfig::default()
        };

        let mut provider = OpenAiRealtimeProvider::new(cfg);
        provider
            .connect(&config)
            .await
            .expect("real server rejected our session.update");
        eprintln!("session.updated received — handshake accepted");

        // The seeded history is fire-and-forget; a rejected item would arrive as an
        // error event within a moment of the handshake.
        let mut events = provider.events();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        while let Ok(Some(event)) = tokio::time::timeout_at(deadline, events.next()).await {
            if let OmniEvent::Error(e) = event {
                panic!("history seed rejected by the real server: {e}");
            }
        }
        eprintln!("history seed accepted (no error within 2s)");
        provider.disconnect().await.unwrap();
    }

    #[test]
    fn downsample_passes_matching_rate_through_unchanged() {
        let pcm = [1u8, 0, 2, 0, 3, 0];
        let out = downsample_to(&pcm, 24_000, 24_000).unwrap();
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(&*out, &pcm);
    }

    #[test]
    fn downsample_48k_to_24k_averages_pairs() {
        let samples: [i16; 4] = [100, 300, 500, 700];
        let pcm: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let out = downsample_to(&pcm, 48_000, 24_000).unwrap();
        let got: Vec<i16> = out
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(i16::from_le_bytes)
            .collect();
        assert_eq!(got, vec![200, 600]);
    }

    #[test]
    fn downsample_refuses_non_integer_ratio() {
        let err = downsample_to(&[0u8; 8], 44_100, 24_000).unwrap_err();
        assert!(err.to_string().contains("integer multiple"), "got: {err}");
    }

    /// The server sends this on its own when server-VAD starts a new turn while the
    /// previous response is still streaming. It must not end the session.
    #[test]
    fn invalid_request_error_is_recoverable_not_fatal() {
        let events = parse_server_event(&json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "conversation_already_has_active_response",
                "message": "Conversation already has an active response in progress",
            }
        }));
        assert!(events.is_empty(), "got: {events:?}");
    }

    /// A rejected request of ours — bad tool result, bad history item — must not be
    /// mistaken for the server's own VAD timing noise.
    #[test]
    fn unknown_invalid_request_is_fatal() {
        let events = parse_server_event(&json!({
            "type": "error",
            "error": { "type": "invalid_request_error", "code": "invalid_value", "message": "bad item" }
        }));
        assert!(matches!(events[0], OmniEvent::Error(_)));
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let shown = format!("{:?}", OpenAiRealtimeConfig::new("sk-very-secret"));
        assert!(shown.contains("<redacted>"));
        assert!(!shown.contains("very-secret"));
    }

    #[tokio::test]
    async fn plaintext_ws_is_refused_without_opt_in() {
        let mut provider = OpenAiRealtimeProvider::new(
            OpenAiRealtimeConfig::new("k").with_endpoint("ws://127.0.0.1:1/x"),
        );
        let err = provider.connect(&OmniConfig::default()).await.unwrap_err();
        assert!(err.to_string().contains("plaintext ws://"), "got: {err}");
    }

    /// Usage first, then each function call, then TurnComplete, all from the one
    /// `response.done` frame.
    #[test]
    fn response_done_orders_usage_tool_calls_turn_complete() {
        let events = parse_server_event(&json!({
            "type": "response.done",
            "response": {
                "usage": { "input_tokens": 3, "output_tokens": 4 },
                "output": [
                    { "type": "message", "role": "assistant" },
                    { "type": "function_call", "call_id": "c1", "name": "t", "arguments": "{\"a\":1}" },
                ],
            },
        }));
        assert!(matches!(events[0], OmniEvent::Usage(_)));
        assert!(
            matches!(&events[1], OmniEvent::ToolCall { id, name, args } if id == "c1" && name == "t" && args["a"] == 1)
        );
        assert!(matches!(events[2], OmniEvent::TurnComplete));
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn speech_stopped_ends_the_user_turn() {
        let events = parse_server_event(&json!({ "type": "input_audio_buffer.speech_stopped" }));
        assert!(matches!(events[0], OmniEvent::UserSpeechEnded));
    }

    #[test]
    fn server_error_is_fatal() {
        let events = parse_server_event(&json!({
            "type": "error",
            "error": { "type": "server_error", "message": "internal" }
        }));
        assert!(matches!(events[0], OmniEvent::Error(_)));
    }

    #[test]
    fn response_done_yields_usage_then_turn_complete() {
        let events = parse_server_event(&json!({
            "type": "response.done",
            "response": { "usage": {
                "input_tokens": 120, "output_tokens": 45,
                "input_token_details": { "audio_tokens": 100, "text_tokens": 20 },
                "output_token_details": { "audio_tokens": 40, "text_tokens": 5 },
            }}
        }));
        let want = Usage {
            input_tokens: 120,
            output_tokens: 45,
            input_audio_tokens: 100,
            output_audio_tokens: 40,
        };
        assert!(
            matches!(events[0], OmniEvent::Usage(u) if u == want),
            "got: {:?}",
            events[0]
        );
        assert!(matches!(events[1], OmniEvent::TurnComplete));
    }

    /// Text in, tool call out, result back, spoken answer — the full tool loop on a
    /// real server with no microphone involved.
    #[tokio::test]
    #[ignore = "opens a real session; needs OPENAI_API_KEY or LITELLM_APIKEY + LITELLM_URL"]
    async fn live_tool_call_round_trip() {
        use tokio::time::{Duration, timeout};
        let cfg = match OpenAiRealtimeConfig::from_env() {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };

        struct TimeTool;
        #[async_trait]
        impl Tool for TimeTool {
            fn name(&self) -> &str {
                "current_time"
            }
            fn description(&self) -> &str {
                "Returns the current time"
            }
            fn parameters_schema(&self) -> Value {
                json!({ "type": "object", "properties": {} })
            }
            async fn execute(
                &self,
                _args: Value,
                _ctx: &crate::tools::ToolContext,
            ) -> crate::Result<String> {
                Ok("It is exactly 10:30 in the morning.".into())
            }
        }
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(TimeTool)];
        let config = OmniConfig {
            system_prompt: Some(
                "Answer in one short sentence. Always call current_time when asked the time."
                    .into(),
            ),
            tools_schema: Some(tool_declarations(&tools)),
            ..OmniConfig::default()
        };

        let mut provider = OpenAiRealtimeProvider::new(cfg);
        provider.connect(&config).await.expect("connect");
        let mut events = provider.events();
        provider
            .send_text("What time is it right now? Use the current_time tool.")
            .await
            .unwrap();

        let (id, name) = loop {
            match timeout(Duration::from_secs(25), events.next()).await {
                Ok(Some(OmniEvent::ToolCall { id, name, .. })) => break (id, name),
                Ok(Some(OmniEvent::Error(e))) => panic!("server error before tool call: {e}"),
                Ok(Some(_)) => {}
                Ok(None) => panic!("stream ended before tool call"),
                Err(_) => panic!("no tool call within 25s"),
            }
        };
        eprintln!("tool call: {name} ({id})");
        assert_eq!(name, "current_time");

        let output = tools[0]
            .execute(json!({}), &crate::tools::ToolContext::default())
            .await
            .unwrap();
        provider
            .send_tool_result(&id, Value::String(output))
            .await
            .unwrap();

        let mut finals: Vec<String> = Vec::new();
        let mut turns_done = 0;
        loop {
            match timeout(Duration::from_secs(25), events.next()).await {
                Ok(Some(OmniEvent::Transcript {
                    text,
                    is_final: true,
                    source: TranscriptSource::Output,
                })) => {
                    // A tool-calling response may carry its own speech ("let me
                    // check"); the answer that uses the result is the next turn.
                    eprintln!("model said: {text}");
                    let done = text.contains("10:30");
                    finals.push(text);
                    if done {
                        break;
                    }
                }
                Ok(Some(OmniEvent::TurnComplete)) => {
                    turns_done += 1;
                    if turns_done >= 2 {
                        break;
                    }
                }
                Ok(Some(OmniEvent::Error(e))) => panic!("server error after tool result: {e}"),
                Ok(Some(_)) => {}
                Ok(None) => panic!("stream ended before the answer"),
                Err(_) => panic!("no answer within 25s of the tool result"),
            }
        }
        assert!(
            finals
                .iter()
                .any(|t| t.contains("10:30") || t.to_lowercase().contains("ten thirty")),
            "no spoken answer used the tool result; heard: {finals:?}"
        );
        provider.disconnect().await.unwrap();
    }

    /// The same tool loop, but the question arrives as speech: a WAV is streamed
    /// through send_audio the way the mic would, trailing silence lets the server's
    /// own VAD end the turn, and the spoken answer must carry the tool's value.
    /// Point OMNI_TEST_WAV at a 24 kHz mono 16-bit WAV of "what time is it".
    #[tokio::test]
    #[ignore = "opens a real session; needs credentials and OMNI_TEST_WAV"]
    async fn live_voice_tool_call_round_trip() {
        use tokio::time::{Duration, timeout};
        let cfg = match OpenAiRealtimeConfig::from_env() {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let Some(wav_path) = std::env::var_os("OMNI_TEST_WAV") else {
            eprintln!("skipping: OMNI_TEST_WAV not set");
            return;
        };
        let (rate, pcm) = read_wav_pcm16_mono(&std::fs::read(wav_path).expect("read wav"));
        eprintln!("wav: {} Hz, {} bytes", rate, pcm.len());

        struct TimeTool;
        #[async_trait]
        impl Tool for TimeTool {
            fn name(&self) -> &str {
                "current_time"
            }
            fn description(&self) -> &str {
                "Returns the current time"
            }
            fn parameters_schema(&self) -> Value {
                json!({ "type": "object", "properties": {} })
            }
            async fn execute(
                &self,
                _args: Value,
                _ctx: &crate::tools::ToolContext,
            ) -> crate::Result<String> {
                Ok("It is exactly 10:30 in the morning.".into())
            }
        }
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(TimeTool)];
        let config = OmniConfig {
            system_prompt: Some(
                "Answer in one short sentence. Always call current_time when asked the time."
                    .into(),
            ),
            tools_schema: Some(tool_declarations(&tools)),
            ..OmniConfig::default()
        };

        let mut provider = OpenAiRealtimeProvider::new(cfg);
        provider.connect(&config).await.expect("connect");
        let mut events = provider.events();

        let frame = (rate as usize / 50) * 2; // 20 ms of PCM16 mono
        let chunk = |data: Vec<u8>| AudioChunk {
            data,
            sample_rate: rate,
            channels: 1,
            bits_per_sample: 16,
        };
        for piece in pcm.chunks(frame) {
            provider.send_audio(chunk(piece.to_vec())).await.unwrap();
        }
        for _ in 0..60 {
            provider.send_audio(chunk(vec![0u8; frame])).await.unwrap();
        }

        let (id, name) = loop {
            match timeout(Duration::from_secs(30), events.next()).await {
                Ok(Some(OmniEvent::ToolCall { id, name, .. })) => break (id, name),
                Ok(Some(OmniEvent::Transcript {
                    text,
                    is_final: true,
                    source: TranscriptSource::Input,
                })) => {
                    eprintln!("server heard: {text:?}");
                }
                Ok(Some(OmniEvent::Error(e))) => panic!("server error before tool call: {e}"),
                Ok(Some(_)) => {}
                Ok(None) => panic!("stream ended before tool call"),
                Err(_) => panic!("no tool call within 30s of the audio"),
            }
        };
        eprintln!("tool call: {name} ({id})");
        assert_eq!(name, "current_time");

        let output = tools[0]
            .execute(json!({}), &crate::tools::ToolContext::default())
            .await
            .unwrap();
        provider
            .send_tool_result(&id, Value::String(output))
            .await
            .unwrap();

        let mut finals: Vec<String> = Vec::new();
        let mut turns_done = 0;
        loop {
            match timeout(Duration::from_secs(25), events.next()).await {
                Ok(Some(OmniEvent::Transcript {
                    text,
                    is_final: true,
                    source: TranscriptSource::Output,
                })) => {
                    // A tool-calling response may carry its own speech ("let me
                    // check"); the answer that uses the result is the next turn.
                    eprintln!("model said: {text}");
                    let done = text.contains("10:30");
                    finals.push(text);
                    if done {
                        break;
                    }
                }
                Ok(Some(OmniEvent::TurnComplete)) => {
                    turns_done += 1;
                    if turns_done >= 2 {
                        break;
                    }
                }
                Ok(Some(OmniEvent::Error(e))) => panic!("server error after tool result: {e}"),
                Ok(Some(_)) => {}
                Ok(None) => panic!("stream ended before the answer"),
                Err(_) => panic!("no answer within 25s of the tool result"),
            }
        }
        assert!(
            finals
                .iter()
                .any(|t| t.contains("10:30") || t.to_lowercase().contains("ten thirty")),
            "no spoken answer used the tool result; heard: {finals:?}"
        );
        provider.disconnect().await.unwrap();
    }

    /// Minimal RIFF reader: returns (sample_rate, PCM bytes) from the `data` chunk.
    fn read_wav_pcm16_mono(bytes: &[u8]) -> (u32, Vec<u8>) {
        let rate = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let mut i = 12;
        while i + 8 <= bytes.len() {
            let id = &bytes[i..i + 4];
            let len = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
            if id == b"data" {
                return (rate, bytes[i + 8..(i + 8 + len).min(bytes.len())].to_vec());
            }
            i += 8 + len + (len & 1);
        }
        panic!("no data chunk in wav");
    }

    #[tokio::test]
    async fn sending_before_connect_is_an_error() {
        let provider = OpenAiRealtimeProvider::new(OpenAiRealtimeConfig::new("k"));
        let err = provider.send_text("hi").await.unwrap_err();
        assert!(err.to_string().contains("not connected"), "got: {err}");
    }
}
