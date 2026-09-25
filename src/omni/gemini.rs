//! Gemini Live (`BidiGenerateContent`) implementation of [`OmniProvider`].

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use dashmap::DashMap;
use futures::{SinkExt, Stream, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_stream::wrappers::ReceiverStream;
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};

use crate::core::error::MindroidError;
use crate::omni::provider::OmniProvider;
use crate::omni::types::{
    AudioChunk, HistoryTurn, OmniConfig, OmniEvent, Role, TranscriptSource, TurnDetection, Usage,
};
use crate::tools::Tool;

/// Google AI Studio `BidiGenerateContent` WebSocket endpoint.
pub const DEFAULT_ENDPOINT: &str = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
/// Current Live model; the 2.0/2.5 live models are deprecated.
pub const DEFAULT_MODEL: &str = "models/gemini-3.1-flash-live-preview";

/// Gemini Live speaks 16 kHz PCM16 in and 24 kHz PCM16 out.
pub const INPUT_SAMPLE_RATE: u32 = 16_000;
pub const OUTPUT_SAMPLE_RATE: u32 = 24_000;

fn transport(message: impl Into<String>) -> MindroidError {
    MindroidError::Transport {
        message: message.into(),
        source: None,
    }
}

/// Connection settings for [`GeminiLiveProvider`].
#[derive(Clone)]
pub struct GeminiLiveConfig {
    pub api_key: String,
    pub model: String,
    pub endpoint: String,
    pub input_sample_rate: u32,
    pub input_transcription: bool,
    pub output_transcription: bool,
    pub session_resumption: bool,
    /// Permit a plaintext `ws://` endpoint. The key rides the URL, so this is for
    /// local fakes only.
    pub allow_insecure: bool,
}

impl fmt::Debug for GeminiLiveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiLiveConfig")
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("endpoint", &self.endpoint)
            .field("input_sample_rate", &self.input_sample_rate)
            .field("input_transcription", &self.input_transcription)
            .field("output_transcription", &self.output_transcription)
            .field("session_resumption", &self.session_resumption)
            .field("allow_insecure", &self.allow_insecure)
            .finish()
    }
}

impl GeminiLiveConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            input_sample_rate: INPUT_SAMPLE_RATE,
            input_transcription: true,
            output_transcription: true,
            session_resumption: true,
            allow_insecure: false,
        }
    }

    /// Read the key from `GEMINI_API_KEY`, or `GOOGLE_API_KEY` as a fallback (the
    /// name Google's own SDKs use).
    ///
    /// # Errors
    ///
    /// Returns [`MindroidError::Config`] when neither variable is set or non-empty.
    pub fn from_env() -> Result<Self, MindroidError> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let key = var("GEMINI_API_KEY")
            .or_else(|| var("GOOGLE_API_KEY"))
            .ok_or_else(|| MindroidError::config("set GEMINI_API_KEY or GOOGLE_API_KEY"))?;
        Ok(Self::new(key))
    }

    /// Override the model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the WebSocket endpoint (tests point this at a local fake).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Allow a plaintext `ws://` endpoint. Never for a real key.
    pub fn with_allow_insecure(mut self, allow: bool) -> Self {
        self.allow_insecure = allow;
        self
    }

    fn url(&self) -> String {
        if self.api_key.is_empty() {
            self.endpoint.clone()
        } else {
            let sep = if self.endpoint.contains('?') {
                '&'
            } else {
                '?'
            };
            format!("{}{sep}key={}", self.endpoint, self.api_key)
        }
    }
}

/// Render tools as Gemini `functionDeclarations`, ready for [`OmniConfig::tools_schema`].
///
/// [`OmniSession`](crate::omni::OmniSession) keeps its tool list and its schema blob
/// separate, so the caller sets both from the same slice.
pub fn tool_declarations(tools: &[Arc<dyn Tool>]) -> Value {
    let declarations: Vec<Value> = tools
        .iter()
        .map(|t| {
            let mut params = t.parameters_schema();
            sanitize_schema(&mut params);
            json!({
                "name": t.name(),
                "description": t.description(),
                "parameters": params,
            })
        })
        .collect();
    json!([{ "functionDeclarations": declarations }])
}

/// Strip JSON Schema keywords Gemini's subset rejects.
fn sanitize_schema(schema: &mut Value) {
    const UNSUPPORTED: [&str; 6] = [
        "$schema",
        "additionalProperties",
        "default",
        "definitions",
        "$defs",
        "$ref",
    ];
    match schema {
        Value::Object(map) => {
            for key in UNSUPPORTED {
                map.remove(key);
            }
            for value in map.values_mut() {
                sanitize_schema(value);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(sanitize_schema),
        _ => {}
    }
}

/// [`OmniProvider`] over Gemini Live's `BidiGenerateContent` WebSocket.
pub struct GeminiLiveProvider {
    config: GeminiLiveConfig,
    outbound: Option<mpsc::Sender<String>>,
    events: Option<mpsc::Receiver<OmniEvent>>,
    /// Gemini's `functionResponse` needs the function *name*, but
    /// [`OmniProvider::send_tool_result`] is only handed the call id.
    pending_calls: Arc<DashMap<String, String>>,
    tasks: JoinSet<()>,
    /// `TurnDetection::Manual`: the client marks turn boundaries with
    /// `activityStart`/`activityEnd` instead of relying on the server VAD.
    manual: AtomicBool,
    /// An `activityStart` has been sent and its `activityEnd` has not.
    activity_open: AtomicBool,
}

impl GeminiLiveProvider {
    pub fn new(config: GeminiLiveConfig) -> Self {
        Self {
            config,
            outbound: None,
            events: None,
            pending_calls: Arc::new(DashMap::new()),
            tasks: JoinSet::new(),
            manual: AtomicBool::new(false),
            activity_open: AtomicBool::new(false),
        }
    }

    fn setup_payload(&self, config: &OmniConfig) -> Value {
        let mut setup = json!({
            "model": self.config.model,
            "generationConfig": { "responseModalities": ["AUDIO"] },
        });

        if let Some(voice) = &config.voice {
            setup["generationConfig"]["speechConfig"] = json!({
                "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": voice } }
            });
        }
        if let Some(prompt) = &config.system_prompt {
            setup["systemInstruction"] = json!({ "parts": [{ "text": prompt }] });
        }
        if let Some(tools) = &config.tools_schema {
            setup["tools"] = tools.clone();
        }
        if self.config.input_transcription {
            setup["inputAudioTranscription"] = json!({});
        }
        if self.config.output_transcription {
            setup["outputAudioTranscription"] = json!({});
        }
        if self.config.session_resumption {
            setup["sessionResumption"] = json!({});
        }
        if !config.history.is_empty() {
            setup["historyConfig"] = json!({ "initialHistoryInClientContent": true });
        }

        // Local VAD is hybrid: server detection stays ON and `audioStreamEnd`
        // finalizes early. Disabling it would need activityStart/activityEnd.
        let manual = matches!(config.turn_detection, TurnDetection::Manual);
        setup["realtimeInputConfig"] = json!({
            "automaticActivityDetection": { "disabled": manual }
        });

        json!({ "setup": setup })
    }

    async fn send_json(&self, payload: Value) -> Result<(), MindroidError> {
        let tx = self
            .outbound
            .as_ref()
            .ok_or_else(|| transport("Gemini Live provider is not connected"))?;
        tx.send(payload.to_string())
            .await
            .map_err(|_| transport("Gemini Live writer task has stopped"))
    }
}

#[async_trait]
impl OmniProvider for GeminiLiveProvider {
    async fn connect(&mut self, config: &OmniConfig) -> Result<(), MindroidError> {
        if self.config.endpoint.starts_with("ws://") && !self.config.allow_insecure {
            return Err(transport(
                "refusing to send the API key over plaintext ws://; set allow_insecure for a local fake",
            ));
        }
        self.manual.store(
            matches!(config.turn_detection, TurnDetection::Manual),
            Ordering::Relaxed,
        );
        self.activity_open.store(false, Ordering::Relaxed);
        let (ws, _) = connect_async(self.config.url())
            .await
            .map_err(|e| transport(format!("Gemini Live connect failed: {e}")))?;
        let (mut sink, mut stream) = ws.split();

        sink.send(WsMessage::Text(self.setup_payload(config).to_string()))
            .await
            .map_err(|e| transport(format!("Gemini Live setup send failed: {e}")))?;

        // The socket opening does not mean the session is usable — anything sent
        // before `setupComplete` is discarded by the server.
        loop {
            let frame = stream
                .next()
                .await
                .ok_or_else(|| transport("Gemini Live closed before setupComplete"))?
                .map_err(|e| transport(format!("Gemini Live read failed: {e}")))?;
            let Some(value) = frame_to_json(frame) else {
                continue;
            };
            if value.get("setupComplete").is_some() {
                break;
            }
            if let Some(err) = value.get("error") {
                return Err(transport(format!("Gemini Live setup rejected: {err}")));
            }
        }

        if !config.history.is_empty() {
            sink.send(WsMessage::Text(
                history_payload(&config.history).to_string(),
            ))
            .await
            .map_err(|e| transport(format!("Gemini Live history seed failed: {e}")))?;
        }

        let (out_tx, mut out_rx) = mpsc::channel::<String>(64);
        let (event_tx, event_rx) = mpsc::channel::<OmniEvent>(256);

        self.tasks.spawn(async move {
            while let Some(text) = out_rx.recv().await {
                if sink.send(WsMessage::Text(text)).await.is_err() {
                    break;
                }
            }
            let _ = sink.close().await;
        });

        let pending = Arc::clone(&self.pending_calls);
        self.tasks.spawn(async move {
            let mut acc = TranscriptAcc::default();
            while let Some(frame) = stream.next().await {
                let frame = match frame {
                    Ok(f) => f,
                    Err(e) => {
                        let _ = event_tx
                            .send(OmniEvent::Error(Arc::new(transport(format!(
                                "Gemini Live stream error: {e}"
                            )))))
                            .await;
                        break;
                    }
                };
                let Some(value) = frame_to_json(frame) else {
                    continue;
                };
                if tracing::enabled!(tracing::Level::TRACE) {
                    tracing::trace!(frame = %elide_audio(&value), "gemini frame");
                }
                for event in parse_server_message(&value, &pending, &mut acc) {
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
        if self.manual.load(Ordering::Relaxed) && !self.activity_open.swap(true, Ordering::Relaxed)
        {
            // Clear the flag if the start never reached the wire, or the retry
            // skips it and `end_audio_stream` sends an unmatched `activityEnd`.
            if let Err(e) = self
                .send_json(json!({ "realtimeInput": { "activityStart": {} } }))
                .await
            {
                self.activity_open.store(false, Ordering::Relaxed);
                return Err(e);
            }
        }
        self.send_json(json!({
            "realtimeInput": {
                "audio": {
                    "mimeType": format!("audio/pcm;rate={}", chunk.sample_rate),
                    "data": STANDARD.encode(&chunk.data),
                }
            }
        }))
        .await
    }

    async fn send_text(&self, text: &str) -> Result<(), MindroidError> {
        self.send_json(json!({
            "clientContent": {
                "turns": [{ "role": "user", "parts": [{ "text": text }] }],
                "turnComplete": true,
            }
        }))
        .await
    }

    async fn send_tool_result(&self, call_id: &str, result: Value) -> Result<(), MindroidError> {
        let name = self
            .pending_calls
            .remove(call_id)
            .map(|(_, name)| name)
            .ok_or_else(|| transport(format!("unknown Gemini tool call id: {call_id}")))?;

        // Gemini requires `response` to be an object; tools return plain strings.
        let response = match result {
            Value::Object(map) => Value::Object(map),
            other => json!({ "result": other }),
        };

        self.send_json(json!({
            "toolResponse": {
                "functionResponses": [{ "id": call_id, "name": name, "response": response }]
            }
        }))
        .await
    }

    /// Ends the user's turn: `activityEnd` under manual detection (the server VAD
    /// is off, so this is what makes the model answer), `audioStreamEnd` otherwise.
    async fn end_audio_stream(&self) -> Result<(), MindroidError> {
        if self.manual.load(Ordering::Relaxed) {
            if self.activity_open.swap(false, Ordering::Relaxed) {
                self.send_json(json!({ "realtimeInput": { "activityEnd": {} } }))
                    .await?;
            }
            return Ok(());
        }
        self.send_json(json!({ "realtimeInput": { "audioStreamEnd": true } }))
            .await
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
        self.pending_calls.clear();
        Ok(())
    }
}

/// Seed prior turns as text. Sent once, right after `setupComplete` and before any
/// audio: `clientContent` interrupts generation and is seeding-only on 3.1.
fn history_payload(history: &[HistoryTurn]) -> Value {
    let turns: Vec<Value> = history
        .iter()
        .map(|t| {
            json!({
                "role": match t.role { Role::User => "user", Role::Model => "model" },
                "parts": [{ "text": t.text }],
            })
        })
        .collect();
    json!({ "clientContent": { "turns": turns, "turnComplete": false } })
}

/// Per-source transcript accumulator for one connection.
#[derive(Default)]
struct TranscriptAcc {
    input: String,
    output: String,
}

impl TranscriptAcc {
    fn slot(&mut self, source: TranscriptSource) -> &mut String {
        match source {
            TranscriptSource::Input => &mut self.input,
            TranscriptSource::Output => &mut self.output,
        }
    }
}

fn frame_to_json(frame: WsMessage) -> Option<Value> {
    let bytes = match frame {
        WsMessage::Text(text) => text.into_bytes(),
        WsMessage::Binary(bytes) => bytes,
        _ => return None,
    };
    serde_json::from_slice(&bytes).ok()
}

/// The frame with every `inlineData.data` replaced by its byte length, for logs.
fn elide_audio(value: &Value) -> Value {
    let mut v = value.clone();
    if let Some(parts) = v
        .pointer_mut("/serverContent/modelTurn/parts")
        .and_then(Value::as_array_mut)
    {
        for part in parts {
            if let Some(data) = part.pointer_mut("/inlineData/data") {
                *data = json!(format!("<{} b64 chars>", data.as_str().map_or(0, str::len)));
            }
        }
    }
    v
}

/// Map one Gemini server frame onto zero or more [`OmniEvent`]s.
///
/// A single `serverContent` can carry audio, a transcript and `turnComplete`
/// together, so this returns a list rather than one event.
fn parse_server_message(
    value: &Value,
    pending: &DashMap<String, String>,
    acc: &mut TranscriptAcc,
) -> Vec<OmniEvent> {
    let mut events = Vec::new();

    if let Some(u) = value.get("usageMetadata") {
        let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
        let audio = |k: &str| {
            u.get(k)
                .and_then(Value::as_array)
                .map(|details| {
                    details
                        .iter()
                        .filter(|d| d.get("modality").and_then(Value::as_str) == Some("AUDIO"))
                        .filter_map(|d| d.get("tokenCount").and_then(Value::as_u64))
                        .sum()
                })
                .unwrap_or(0)
        };
        events.push(OmniEvent::Usage(Usage {
            input_tokens: n("promptTokenCount"),
            output_tokens: n("responseTokenCount"),
            input_audio_tokens: audio("promptTokensDetails"),
            output_audio_tokens: audio("responseTokensDetails"),
        }));
    }

    if let Some(content) = value.get("serverContent") {
        if content.get("interrupted").and_then(Value::as_bool) == Some(true) {
            // What was generated before the cut is this turn's whole output; it
            // must not bleed into the next turn's final.
            let whole = std::mem::take(&mut acc.output);
            if !whole.is_empty() {
                events.push(OmniEvent::Transcript {
                    text: whole,
                    is_final: true,
                    source: TranscriptSource::Output,
                });
            }
            events.push(OmniEvent::Interrupted);
        }

        let parts = content
            .pointer("/modelTurn/parts")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        for part in parts {
            let Some(data) = part.pointer("/inlineData/data").and_then(Value::as_str) else {
                continue;
            };
            let Ok(decoded) = STANDARD.decode(data) else {
                continue;
            };
            let sample_rate = part
                .pointer("/inlineData/mimeType")
                .and_then(Value::as_str)
                .and_then(parse_rate)
                .unwrap_or(OUTPUT_SAMPLE_RATE);
            events.push(OmniEvent::AudioChunk(AudioChunk {
                data: decoded,
                sample_rate,
                channels: 1,
                bits_per_sample: 16,
            }));
        }

        let turn_complete = content.get("turnComplete").and_then(Value::as_bool) == Some(true);

        for (key, source) in [
            ("inputTranscription", TranscriptSource::Input),
            ("outputTranscription", TranscriptSource::Output),
        ] {
            if let Some(text) = content
                .pointer(&format!("/{key}/text"))
                .and_then(Value::as_str)
            {
                acc.slot(source).push_str(text);
                events.push(OmniEvent::Transcript {
                    text: text.to_string(),
                    is_final: false,
                    source,
                });
            }
        }

        // Gemini streams pieces; the final carries the whole turn, matching OpenAI's
        // `.done` events, so consumers persist finals without re-assembling.
        if turn_complete {
            for source in [TranscriptSource::Input, TranscriptSource::Output] {
                let whole = std::mem::take(acc.slot(source));
                if !whole.is_empty() {
                    events.push(OmniEvent::Transcript {
                        text: whole,
                        is_final: true,
                        source,
                    });
                }
            }
            events.push(OmniEvent::TurnComplete);
        }
    }

    if let Some(calls) = value
        .pointer("/toolCall/functionCalls")
        .and_then(Value::as_array)
    {
        for call in calls {
            let Some(name) = call.get("name").and_then(Value::as_str) else {
                continue;
            };
            // Gemini omits `id` for non-parallel calls; the name is then unique.
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or(name)
                .to_string();
            pending.insert(id.clone(), name.to_string());
            events.push(OmniEvent::ToolCall {
                id,
                name: name.to_string(),
                args: call.get("args").cloned().unwrap_or_else(|| json!({})),
            });
        }
    }

    if let Some(go_away) = value.get("goAway") {
        events.push(OmniEvent::SessionEnding {
            in_secs: go_away
                .get("timeLeft")
                .and_then(Value::as_str)
                .and_then(|s| s.trim_end_matches('s').parse().ok()),
        });
    }

    if let Some(handle) = value
        .pointer("/sessionResumptionUpdate/newHandle")
        .and_then(Value::as_str)
    {
        events.push(OmniEvent::ResumptionHandle(handle.to_string()));
    }

    events
}

fn parse_rate(mime: &str) -> Option<u32> {
    mime.split(';')
        .find_map(|p| p.trim().strip_prefix("rate="))
        .and_then(|r| r.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::omni::types::VadConfig;
    use futures::stream::SplitSink;
    use tokio::net::TcpStream;
    use tokio_tungstenite::WebSocketStream;

    type Sink = SplitSink<WebSocketStream<TcpStream>, WsMessage>;

    /// Minimal Gemini Live: completes setup, then replies to each client frame
    /// with whatever `script` produces. Binding port 0 keeps tests parallel-safe.
    async fn spawn_fake_gemini<F>(script: F) -> (String, mpsc::Receiver<Value>)
    where
        F: Fn(&Value) -> Vec<Value> + Send + 'static,
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (seen_tx, seen_rx) = mpsc::channel::<Value>(64);

        tokio::spawn(async move {
            let Ok((tcp, _)) = listener.accept().await else {
                return;
            };
            let Ok(ws) = tokio_tungstenite::accept_async(tcp).await else {
                return;
            };
            let (mut sink, mut stream) = ws.split();

            let setup = stream.next().await;
            if let Some(Ok(frame)) = setup
                && let Some(value) = frame_to_json(frame)
            {
                let _ = seen_tx.send(value).await;
            }
            let _ = send(&mut sink, json!({ "setupComplete": {} })).await;

            while let Some(Ok(frame)) = stream.next().await {
                let Some(value) = frame_to_json(frame) else {
                    continue;
                };
                let replies = script(&value);
                if seen_tx.send(value).await.is_err() {
                    return;
                }
                for reply in replies {
                    if send(&mut sink, reply).await.is_err() {
                        return;
                    }
                }
            }
        });

        (format!("ws://{addr}/live"), seen_rx)
    }

    async fn send(
        sink: &mut Sink,
        value: Value,
    ) -> Result<(), tokio_tungstenite::tungstenite::Error> {
        sink.send(WsMessage::Text(value.to_string())).await
    }

    async fn connected(
        script: impl Fn(&Value) -> Vec<Value> + Send + 'static,
    ) -> (GeminiLiveProvider, mpsc::Receiver<Value>) {
        let (url, seen) = spawn_fake_gemini(script).await;
        let config = GeminiLiveConfig::new("test-key")
            .with_endpoint(url)
            .with_allow_insecure(true);
        let mut provider = GeminiLiveProvider::new(config);
        provider.connect(&OmniConfig::default()).await.unwrap();
        (provider, seen)
    }

    #[tokio::test]
    async fn connect_sends_setup_and_waits_for_setup_complete() {
        let (mut provider, mut seen) = connected(|_| vec![]).await;

        let setup = seen.recv().await.expect("setup frame");
        assert_eq!(setup["setup"]["model"], DEFAULT_MODEL);
        assert_eq!(
            setup["setup"]["generationConfig"]["responseModalities"][0],
            "AUDIO"
        );
        // Server turn detection is the OmniConfig default, so server VAD stays on.
        assert_eq!(
            setup["setup"]["realtimeInputConfig"]["automaticActivityDetection"]["disabled"],
            false
        );

        provider.disconnect().await.unwrap();
    }

    /// Local VAD must keep server detection ON (hybrid). Disabling it while the
    /// session still signals turn end via `audioStreamEnd` means turns never
    /// finalize — `audioStreamEnd` is only honored when server VAD is enabled.
    #[test]
    fn only_manual_turn_detection_disables_server_vad() {
        let provider = GeminiLiveProvider::new(GeminiLiveConfig::new("k"));
        let disabled = |td: TurnDetection| {
            let cfg = OmniConfig {
                turn_detection: td,
                ..OmniConfig::default()
            };
            provider.setup_payload(&cfg)["setup"]["realtimeInputConfig"]
                ["automaticActivityDetection"]["disabled"]
                .as_bool()
                .expect("disabled flag")
        };

        assert!(!disabled(TurnDetection::Server));
        assert!(!disabled(TurnDetection::Local(VadConfig::default())));
        assert!(disabled(TurnDetection::Manual));
    }

    #[tokio::test]
    async fn connect_fails_when_setup_is_rejected() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
            let (mut sink, mut stream) = ws.split();
            let _ = stream.next().await;
            let _ = send(&mut sink, json!({ "error": { "message": "bad key" } })).await;
        });

        let config = GeminiLiveConfig::new("k")
            .with_endpoint(format!("ws://{addr}/live"))
            .with_allow_insecure(true);
        let err = GeminiLiveProvider::new(config)
            .connect(&OmniConfig::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("setup rejected"), "got: {err}");
    }

    #[tokio::test]
    async fn send_audio_is_base64_pcm_at_the_chunk_rate() {
        let (provider, mut seen) = connected(|_| vec![]).await;
        let _setup = seen.recv().await;

        provider
            .send_audio(AudioChunk {
                data: vec![1, 2, 3, 4],
                sample_rate: INPUT_SAMPLE_RATE,
                channels: 1,
                bits_per_sample: 16,
            })
            .await
            .unwrap();

        let frame = seen.recv().await.expect("audio frame");
        assert_eq!(
            frame["realtimeInput"]["audio"]["mimeType"],
            "audio/pcm;rate=16000"
        );
        assert_eq!(
            frame["realtimeInput"]["audio"]["data"],
            STANDARD.encode([1u8, 2, 3, 4])
        );
    }

    #[tokio::test]
    async fn end_audio_stream_signals_turn_end() {
        let (provider, mut seen) = connected(|_| vec![]).await;
        let _setup = seen.recv().await;

        provider.end_audio_stream().await.unwrap();

        let frame = seen.recv().await.expect("audioStreamEnd frame");
        assert_eq!(frame["realtimeInput"]["audioStreamEnd"], true);
    }

    #[tokio::test]
    async fn manual_detection_brackets_the_audio_with_activity_markers() {
        let (url, mut seen) = spawn_fake_gemini(|_| vec![]).await;
        let config = GeminiLiveConfig::new("test-key")
            .with_endpoint(url)
            .with_allow_insecure(true);
        let mut provider = GeminiLiveProvider::new(config);
        provider
            .connect(&OmniConfig {
                turn_detection: TurnDetection::Manual,
                ..OmniConfig::default()
            })
            .await
            .unwrap();
        let _setup = seen.recv().await;
        let chunk = AudioChunk {
            data: vec![0, 0],
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
        };

        provider.send_audio(chunk.clone()).await.unwrap();
        provider.send_audio(chunk).await.unwrap();
        provider.end_audio_stream().await.unwrap();

        let frames: Vec<Value> = [
            seen.recv().await.unwrap(),
            seen.recv().await.unwrap(),
            seen.recv().await.unwrap(),
            seen.recv().await.unwrap(),
        ]
        .into();
        assert!(frames[0]["realtimeInput"]["activityStart"].is_object());
        assert!(frames[1]["realtimeInput"]["audio"].is_object());
        assert!(frames[2]["realtimeInput"]["audio"].is_object());
        assert!(frames[3]["realtimeInput"]["activityEnd"].is_object());
        assert!(frames[3]["realtimeInput"].get("audioStreamEnd").is_none());
    }

    /// The round trip the trait signature makes easy to get wrong: `send_tool_result`
    /// only receives the call id, but Gemini's `functionResponse` also needs the name.
    #[tokio::test]
    async fn tool_call_round_trip_carries_the_function_name_back() {
        let (mut provider, mut seen) = connected(|frame| {
            if frame.get("realtimeInput").is_some() {
                vec![json!({
                    "toolCall": {
                        "functionCalls": [{ "id": "fc-1", "name": "get_weather", "args": { "city": "Oslo" } }]
                    }
                })]
            } else {
                vec![]
            }
        })
        .await;
        let _setup = seen.recv().await;

        let mut events = provider.events();
        provider.end_audio_stream().await.unwrap();
        let _echo = seen.recv().await;

        let event = events.next().await.expect("tool call");
        let OmniEvent::ToolCall { id, name, args } = event else {
            panic!("expected ToolCall, got {event:?}");
        };
        assert_eq!(id, "fc-1");
        assert_eq!(name, "get_weather");
        assert_eq!(args["city"], "Oslo");

        provider
            .send_tool_result(&id, json!({ "temp_c": 4 }))
            .await
            .unwrap();

        let frame = seen.recv().await.expect("toolResponse frame");
        let response = &frame["toolResponse"]["functionResponses"][0];
        assert_eq!(response["id"], "fc-1");
        assert_eq!(
            response["name"], "get_weather",
            "name must survive the round trip"
        );
        assert_eq!(response["response"]["temp_c"], 4);
    }

    #[tokio::test]
    async fn tool_result_for_an_unknown_id_is_rejected() {
        let (provider, mut seen) = connected(|_| vec![]).await;
        let _setup = seen.recv().await;

        let err = provider
            .send_tool_result("never-seen", json!({}))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("unknown Gemini tool call id"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn string_tool_results_are_wrapped_into_an_object() {
        let (mut provider, mut seen) = connected(|frame| {
            if frame.get("realtimeInput").is_some() {
                vec![json!({ "toolCall": { "functionCalls": [{ "id": "c1", "name": "echo" }] } })]
            } else {
                vec![]
            }
        })
        .await;
        let _setup = seen.recv().await;

        let mut events = provider.events();
        provider.end_audio_stream().await.unwrap();
        let _echo = seen.recv().await;
        let _call = events.next().await;

        provider
            .send_tool_result("c1", Value::String("sunny".into()))
            .await
            .unwrap();

        let frame = seen.recv().await.expect("toolResponse frame");
        assert_eq!(
            frame["toolResponse"]["functionResponses"][0]["response"]["result"],
            "sunny"
        );
    }

    #[test]
    fn server_content_yields_audio_transcript_and_turn_complete() {
        let pending = DashMap::new();
        let mut acc = TranscriptAcc::default();
        let events = parse_server_message(
            &json!({
                "serverContent": {
                    "modelTurn": { "parts": [{ "inlineData": {
                        "mimeType": "audio/pcm;rate=24000",
                        "data": STANDARD.encode([9u8, 9]),
                    }}]},
                    "outputTranscription": { "text": "hello" },
                    "turnComplete": true,
                }
            }),
            &pending,
            &mut acc,
        );

        assert!(matches!(
            events[0],
            OmniEvent::AudioChunk(AudioChunk {
                sample_rate: 24_000,
                ..
            })
        ));
        assert!(matches!(
            &events[1],
            OmniEvent::Transcript { source: TranscriptSource::Output, is_final: false, text } if text == "hello"
        ));
        assert!(matches!(
            &events[2],
            OmniEvent::Transcript { source: TranscriptSource::Output, is_final: true, text } if text == "hello"
        ));
        assert!(matches!(events[3], OmniEvent::TurnComplete));
    }

    /// Gemini streams transcript pieces; the final must carry the whole turn.
    #[test]
    fn transcript_pieces_accumulate_into_one_final() {
        let pending = DashMap::new();
        let mut acc = TranscriptAcc::default();
        let piece = |t: &str, done: bool| {
            let mut v = json!({ "serverContent": { "outputTranscription": { "text": t } } });
            if done {
                v["serverContent"]["turnComplete"] = json!(true);
            }
            v
        };
        let a = parse_server_message(&piece("hel", false), &pending, &mut acc);
        let b = parse_server_message(&piece("lo ", false), &pending, &mut acc);
        let c = parse_server_message(&piece("there", true), &pending, &mut acc);

        assert!(
            matches!(&a[0], OmniEvent::Transcript { is_final: false, text, .. } if text == "hel")
        );
        assert!(
            matches!(&b[0], OmniEvent::Transcript { is_final: false, text, .. } if text == "lo ")
        );
        assert!(
            matches!(&c[1], OmniEvent::Transcript { is_final: true, text, .. } if text == "hello there")
        );
        assert!(matches!(c[2], OmniEvent::TurnComplete));
        assert!(
            acc.output.is_empty(),
            "accumulator must reset after the final"
        );
    }

    #[tokio::test]
    async fn history_seeds_setup_flag_and_client_content_after_setup_complete() {
        let (url, mut seen) = spawn_fake_gemini(|_| vec![]).await;
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
        let mut provider = GeminiLiveProvider::new(
            GeminiLiveConfig::new("k")
                .with_endpoint(url)
                .with_allow_insecure(true),
        );
        provider.connect(&config).await.unwrap();

        let setup = seen.recv().await.expect("setup");
        assert_eq!(
            setup["setup"]["historyConfig"]["initialHistoryInClientContent"],
            true
        );

        let seed = seen.recv().await.expect("clientContent");
        assert_eq!(seed["clientContent"]["turnComplete"], false);
        assert_eq!(seed["clientContent"]["turns"][0]["role"], "user");
        assert_eq!(seed["clientContent"]["turns"][0]["parts"][0]["text"], "hi");
        assert_eq!(seed["clientContent"]["turns"][1]["role"], "model");
        assert_eq!(
            seed["clientContent"]["turns"][1]["parts"][0]["text"],
            "hello"
        );
        provider.disconnect().await.unwrap();
    }

    #[test]
    fn input_and_output_transcripts_are_distinguishable() {
        let pending = DashMap::new();
        let events = parse_server_message(
            &json!({ "serverContent": { "inputTranscription": { "text": "what time is it" } } }),
            &pending,
            &mut TranscriptAcc::default(),
        );
        assert!(matches!(
            events[0],
            OmniEvent::Transcript {
                source: TranscriptSource::Input,
                ..
            }
        ));
    }

    #[test]
    fn interrupted_precedes_any_audio_in_the_same_frame() {
        let pending = DashMap::new();
        let events = parse_server_message(
            &json!({ "serverContent": { "interrupted": true } }),
            &pending,
            &mut TranscriptAcc::default(),
        );
        assert!(matches!(events[0], OmniEvent::Interrupted));
    }

    #[test]
    fn go_away_and_resumption_handle_are_surfaced() {
        let pending = DashMap::new();
        let events = parse_server_message(
            &json!({ "goAway": { "timeLeft": "12s" } }),
            &pending,
            &mut TranscriptAcc::default(),
        );
        assert!(matches!(
            events[0],
            OmniEvent::SessionEnding { in_secs: Some(12) }
        ));

        let events = parse_server_message(
            &json!({ "sessionResumptionUpdate": { "newHandle": "h-1", "resumable": true } }),
            &pending,
            &mut TranscriptAcc::default(),
        );
        assert!(matches!(&events[0], OmniEvent::ResumptionHandle(h) if h == "h-1"));
    }

    #[test]
    fn usage_metadata_is_surfaced_with_audio_split() {
        let pending = DashMap::new();
        let events = parse_server_message(
            &json!({ "usageMetadata": {
                "promptTokenCount": 300, "responseTokenCount": 80,
                "promptTokensDetails": [
                    { "modality": "AUDIO", "tokenCount": 250 },
                    { "modality": "TEXT", "tokenCount": 50 },
                ],
                "responseTokensDetails": [{ "modality": "AUDIO", "tokenCount": 80 }],
            }}),
            &pending,
            &mut TranscriptAcc::default(),
        );
        let want = Usage {
            input_tokens: 300,
            output_tokens: 80,
            input_audio_tokens: 250,
            output_audio_tokens: 80,
        };
        assert!(
            matches!(events[0], OmniEvent::Usage(u) if u == want),
            "got: {:?}",
            events[0]
        );
    }

    /// A barged-in turn's partial output is finalized at the cut, not merged into
    /// the next turn.
    #[test]
    fn interrupted_flushes_partial_output_before_the_next_turn() {
        let pending = DashMap::new();
        let mut acc = TranscriptAcc::default();
        let piece = |t: &str| json!({ "serverContent": { "outputTranscription": { "text": t } } });
        let _ = parse_server_message(&piece("I was say"), &pending, &mut acc);
        let cut = parse_server_message(
            &json!({ "serverContent": { "interrupted": true } }),
            &pending,
            &mut acc,
        );
        assert!(
            matches!(&cut[0], OmniEvent::Transcript { is_final: true, text, .. } if text == "I was say")
        );
        assert!(matches!(cut[1], OmniEvent::Interrupted));

        let _ = parse_server_message(&piece("Sure."), &pending, &mut acc);
        let next = parse_server_message(
            &json!({ "serverContent": { "outputTranscription": { "text": "" }, "turnComplete": true } }),
            &pending,
            &mut acc,
        );
        let finals: Vec<&String> = next
            .iter()
            .filter_map(|e| match e {
                OmniEvent::Transcript {
                    is_final: true,
                    text,
                    ..
                } => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(
            finals,
            vec!["Sure."],
            "next turn must not carry the cut text"
        );
    }

    #[test]
    fn usage_precedes_turn_complete_on_a_combined_frame() {
        let pending = DashMap::new();
        let events = parse_server_message(
            &json!({
                "serverContent": { "turnComplete": true },
                "usageMetadata": { "promptTokenCount": 1, "responseTokenCount": 1 },
            }),
            &pending,
            &mut TranscriptAcc::default(),
        );
        assert!(matches!(events[0], OmniEvent::Usage(_)));
        assert!(matches!(events[1], OmniEvent::TurnComplete));
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let shown = format!("{:?}", GeminiLiveConfig::new("sk-very-secret"));
        assert!(shown.contains("<redacted>"));
        assert!(!shown.contains("very-secret"));
    }

    #[tokio::test]
    async fn plaintext_ws_is_refused_without_opt_in() {
        let mut provider =
            GeminiLiveProvider::new(GeminiLiveConfig::new("k").with_endpoint("ws://127.0.0.1:1/x"));
        let err = provider.connect(&OmniConfig::default()).await.unwrap_err();
        assert!(err.to_string().contains("plaintext ws://"), "got: {err}");
    }

    #[test]
    fn unknown_frames_produce_no_events() {
        let pending = DashMap::new();
        assert!(
            parse_server_message(
                &json!({ "somethingElse": {} }),
                &pending,
                &mut TranscriptAcc::default()
            )
            .is_empty()
        );
    }

    #[test]
    fn tool_declarations_strip_unsupported_schema_keywords() {
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
                json!({
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "additionalProperties": false,
                    "properties": { "a": { "type": "string", "default": "x" } },
                })
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
        let declarations = tool_declarations(&tools);
        let params = &declarations[0]["functionDeclarations"][0]["parameters"];

        assert!(params.get("$schema").is_none());
        assert!(params.get("additionalProperties").is_none());
        assert!(params["properties"]["a"].get("default").is_none());
        assert_eq!(params["properties"]["a"]["type"], "string");
    }

    #[test]
    fn url_appends_the_key_as_a_query_parameter() {
        let config = GeminiLiveConfig::new("secret")
            .with_endpoint("wss://host/live")
            .with_allow_insecure(true);
        assert_eq!(config.url(), "wss://host/live?key=secret");

        let config = GeminiLiveConfig::new("secret")
            .with_endpoint("wss://host/live?alt=1")
            .with_allow_insecure(true);
        assert_eq!(config.url(), "wss://host/live?alt=1&key=secret");
    }

    #[tokio::test]
    async fn sending_before_connect_is_an_error() {
        let provider = GeminiLiveProvider::new(GeminiLiveConfig::new("k"));
        let err = provider.send_text("hi").await.unwrap_err();
        assert!(err.to_string().contains("not connected"), "got: {err}");
    }
}
