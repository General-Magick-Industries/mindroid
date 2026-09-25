//! A realtime provider in the LLM slot of a turn-based pipeline.
//!
//! [`OmniStage`] takes the utterance the transport delivered (`metadata["audio_data"]`,
//! a WAV — or plain text for a text transport), opens one provider session for the
//! turn, seeds it with the system prompt and history the persona stage assembled
//! into `ctx.llm_messages`, streams the audio, runs the tool loop against the same
//! registry and [`ToolContext`] the text executor uses, and writes back:
//!
//! - `ctx.response` — the reply's spoken transcript, so post-processing, reply
//!   ingest and persistence run unchanged;
//! - [`TextInput`] — the provider's transcript of the user's utterance;
//! - [`AudioOutput`] — the reply audio as a WAV, for an output stage or transport.
//!
//! One session per turn keeps the stage stateless. The connect and the history seed
//! are paid on every utterance; a session cache keyed by conversation is the
//! obvious next step once latency matters.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use futures::{Stream, StreamExt};
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::{
    Result,
    core::{
        context::Context,
        error::MindroidError,
        models::{LlmMessage, Role},
    },
    omni::{
        AudioChunk, AudioSink, HistoryTurn, OmniConfig, OmniEvent, OmniProvider, TranscriptSource,
        Usage, types::Role as OmniRole,
    },
    pipeline::{
        PipelineStage,
        extensions::{AudioOutput, CurrentUserMessage, TextInput},
        stages::{registry_for_turn, tool_context_for},
    },
    tools::{DynamicRegistry, Tool, ToolContext, ToolRegistry},
};

/// What [`OmniStage`] needs from a provider family: a fresh connection per turn, the
/// provider's own rendering of the tool schema, and the input rate it expects.
pub trait OmniBackend: Send + Sync + 'static {
    fn provider(&self) -> Box<dyn OmniProvider>;
    fn tool_declarations(&self, tools: &[Arc<dyn Tool>]) -> Value;
    fn input_sample_rate(&self) -> u32;
}

#[cfg(feature = "omni-gemini")]
impl OmniBackend for crate::omni::gemini::GeminiLiveConfig {
    fn provider(&self) -> Box<dyn OmniProvider> {
        Box::new(crate::omni::gemini::GeminiLiveProvider::new(self.clone()))
    }

    fn tool_declarations(&self, tools: &[Arc<dyn Tool>]) -> Value {
        crate::omni::gemini::tool_declarations(tools)
    }

    fn input_sample_rate(&self) -> u32 {
        self.input_sample_rate
    }
}

/// A run-scoped sink the stage plays reply audio into as it arrives, so a host
/// can start playback before the turn (and the stages after it) finish. The
/// full reply still lands in [`AudioOutput`].
pub struct LiveAudioSink(pub Arc<dyn AudioSink>);

/// The turn-invariant inputs the collection loop reads.
struct TurnDeps<'a> {
    registry: &'a ToolRegistry,
    tool_ctx: &'a ToolContext,
    live: Option<&'a dyn AudioSink>,
}

/// Token accounting for the turn, left in the run scope for whoever bills.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OmniTurnUsage(pub Usage);

/// Appended to the system prompt so a persona written for text answers as
/// speech: a realtime model given a "diffable output" instruction will
/// otherwise drop the audio modality entirely.
/// Ceiling on a base64 `audio_data` payload, decoded. An utterance is orders of
/// magnitude smaller; this only stops a malformed or hostile message from being
/// expanded into memory.
const MAX_AUDIO_BYTES: usize = 32 * 1024 * 1024;

pub const VOICE_INSTRUCTION: &str = "VOICE MODE. You are talking, not writing: this conversation is carried as audio over a voice channel and the user only hears you. Every reply must be spoken audio — never a text-only reply. Speak in natural sentences, without markdown, lists, ids, code or headings, and keep it short enough to say out loud. Where the instructions below ask for a written format or for verbatim ids and payloads, say the substance in speech instead.";

pub struct OmniStage {
    backend: Arc<dyn OmniBackend>,
    registry: DynamicRegistry,
    voice: Option<String>,
    voice_instruction: Option<String>,
    turn_timeout: Duration,
    /// Grace after a `TurnComplete` that produced nothing: a provider may end the
    /// user's turn before it emits the tool call that answers it.
    tool_call_grace: Duration,
    max_tool_rounds: usize,
}

impl OmniStage {
    pub fn new(backend: impl OmniBackend, registry: ToolRegistry) -> Self {
        Self::with_dynamic_registry(backend, DynamicRegistry::new(registry))
    }

    pub fn with_dynamic_registry(backend: impl OmniBackend, registry: DynamicRegistry) -> Self {
        Self {
            backend: Arc::new(backend),
            registry,
            voice: None,
            voice_instruction: Some(VOICE_INSTRUCTION.to_string()),
            turn_timeout: Duration::from_secs(60),
            tool_call_grace: Duration::from_millis(1500),
            max_tool_rounds: 8,
        }
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Replace the appended voice-mode instruction; `None` sends the persona as-is.
    pub fn with_voice_instruction(mut self, text: Option<String>) -> Self {
        self.voice_instruction = text;
        self
    }

    pub fn with_turn_timeout(mut self, timeout: Duration) -> Self {
        self.turn_timeout = timeout;
        self
    }

    pub fn with_max_tool_rounds(mut self, rounds: usize) -> Self {
        self.max_tool_rounds = rounds;
        self
    }

    fn config(&self, ctx: &Context, tools: &[Arc<dyn Tool>]) -> OmniConfig {
        let (mut system_prompt, history) = split_llm_messages(ctx);
        if let Some(voice) = &self.voice_instruction {
            system_prompt = Some(match system_prompt {
                Some(p) => format!(
                    "{voice}

{p}"
                ),
                None => voice.clone(),
            });
        }
        OmniConfig {
            system_prompt,
            tools_schema: (!tools.is_empty()).then(|| self.backend.tool_declarations(tools)),
            voice: self.voice.clone(),
            history,
            // The transport already cut the utterance; marking its ends is exact
            // where a second VAD on the provider is a guess (and can never fire).
            turn_detection: crate::omni::TurnDetection::Manual,
            ..OmniConfig::default()
        }
    }
}

#[async_trait]
impl PipelineStage for OmniStage {
    fn name(&self) -> &str {
        "OmniStage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let registry = registry_for_turn(ctx, &self.registry);
        let tool_ctx = tool_context_for(ctx);
        let live = ctx.get_run::<LiveAudioSink>().map(|l| Arc::clone(&l.0));
        let deps = TurnDeps {
            registry: &registry,
            tool_ctx: &tool_ctx,
            live: live.as_deref(),
        };
        let input = TurnInput::from_context(ctx, self.backend.input_sample_rate())?;
        let config = self.config(ctx, registry.tools());

        let mut provider = self.backend.provider();
        provider.connect(&config).await?;
        let mut events = provider.events();
        let sent = match &input {
            TurnInput::Audio(chunk) => {
                debug!(
                    bytes = chunk.data.len(),
                    rate = chunk.sample_rate,
                    "OmniStage: sending utterance"
                );
                match provider.send_audio(chunk.clone()).await {
                    Ok(()) => provider.end_audio_stream().await,
                    Err(e) => Err(e),
                }
            }
            TurnInput::Text(text) => provider.send_text(text).await,
        };
        let turn = match sent {
            Ok(()) => self.collect_turn(&*provider, &mut events, &deps).await,
            Err(e) => Err(e),
        };
        let turn = match turn {
            Ok(turn) if turn.audio.is_empty() && !turn.reply.trim().is_empty() => {
                self.speak_fallback(&*provider, &mut events, &deps, turn)
                    .await
            }
            other => other,
        };
        if let Err(e) = provider.disconnect().await {
            warn!("OmniStage: disconnect failed: {e}");
        }
        if let Some(sink) = deps.live
            && let Err(e) = sink.flush().await
        {
            warn!("OmniStage: live sink flush failed: {e}");
        }
        let turn = turn?;

        info!(
            reply_chars = turn.reply.len(),
            audio_bytes = turn.audio.len(),
            input = turn.usage.input_tokens,
            output = turn.usage.output_tokens,
            "OmniStage: turn complete"
        );
        if let Some(user) = turn.user_text.filter(|t| !t.trim().is_empty()) {
            ctx.set_ext(TextInput(user));
        }
        if !turn.audio.is_empty() {
            ctx.set_ext(AudioOutput(encode_wav(
                &turn.audio,
                turn.sample_rate,
                turn.channels,
            )?));
        }
        ctx.set(OmniTurnUsage(turn.usage));
        ctx.response = Some(turn.reply);
        Ok(())
    }
}

enum TurnInput {
    Audio(AudioChunk),
    Text(String),
}

impl TurnInput {
    fn from_context(ctx: &Context, target_rate: u32) -> Result<Self> {
        if let Some(encoded) = ctx
            .message
            .metadata
            .get("audio_data")
            .and_then(Value::as_str)
        {
            let wav = STANDARD
                .decode(encoded)
                .map_err(|e| stage_err(format!("audio_data is not base64: {e}")))?;
            if wav.len() > MAX_AUDIO_BYTES {
                return Err(stage_err(format!(
                    "audio_data is {} bytes, over the {MAX_AUDIO_BYTES} limit",
                    wav.len()
                )));
            }
            return Ok(Self::Audio(wav_to_pcm(&wav, target_rate)?));
        }
        let live = ctx
            .get::<CurrentUserMessage>()
            .and_then(|c| ctx.llm_messages.get(c.0))
            .map(LlmMessage::text)
            .filter(|t| !t.trim().is_empty());
        let text = live.unwrap_or_else(|| ctx.message.content.clone());
        if text.trim().is_empty() {
            return Err(stage_err("message carries neither audio_data nor text"));
        }
        Ok(Self::Text(text))
    }
}

struct Turn {
    reply: String,
    user_text: Option<String>,
    audio: Vec<u8>,
    sample_rate: u32,
    channels: u16,
    usage: Usage,
}

impl OmniStage {
    async fn collect_turn(
        &self,
        provider: &dyn OmniProvider,
        events: &mut (impl Stream<Item = OmniEvent> + Unpin),
        deps: &TurnDeps<'_>,
    ) -> Result<Turn> {
        let TurnDeps {
            registry,
            tool_ctx,
            live,
        } = *deps;
        let mut turn = Turn {
            reply: String::new(),
            user_text: None,
            audio: Vec::new(),
            sample_rate: 0,
            channels: 1,
            usage: Usage::default(),
        };
        let mut rounds = 0usize;
        let mut output_since_tool = true;
        let mut wait = self.turn_timeout;

        loop {
            let event = match tokio::time::timeout(wait, events.next()).await {
                Ok(Some(event)) => event,
                Ok(None) => return Err(stage_err("provider closed the event stream mid-turn")),
                Err(_) if wait == self.tool_call_grace => break,
                Err(_) => return Err(stage_err("timed out waiting for the provider")),
            };
            wait = self.turn_timeout;
            match event {
                OmniEvent::AudioChunk(chunk) => {
                    output_since_tool = true;
                    turn.sample_rate = chunk.sample_rate;
                    turn.channels = chunk.channels;
                    turn.audio.extend_from_slice(&chunk.data);
                    if let Some(sink) = live
                        && let Err(e) = sink.play(chunk).await
                    {
                        warn!("OmniStage: live sink rejected a chunk: {e}");
                    }
                }
                OmniEvent::Transcript {
                    text,
                    is_final: true,
                    source,
                } => match source {
                    TranscriptSource::Input => {
                        append(turn.user_text.get_or_insert_default(), &text)
                    }
                    TranscriptSource::Output => {
                        output_since_tool = true;
                        append(&mut turn.reply, &text);
                    }
                },
                OmniEvent::Transcript { .. } => {}
                OmniEvent::ToolCall { id, name, args } => {
                    rounds += 1;
                    let over_limit = rounds > self.max_tool_rounds;
                    let result = if over_limit {
                        Value::String(format!(
                            "Error: tool round limit ({}) reached; answer with what you have",
                            self.max_tool_rounds
                        ))
                    } else {
                        execute_tool(registry, tool_ctx, &name, args).await
                    };
                    info!(%id, %name, round = rounds, "OmniStage: tool call");
                    provider.send_tool_result(&id, result).await?;
                    output_since_tool = false;
                    // Leave once the limit is past. Sending the error alone does not
                    // bound the loop: a model that calls another tool anyway gets
                    // another round, and `wait` below resets on every event, so
                    // `turn_timeout` bounds the gap between events, not the turn.
                    if over_limit {
                        warn!(
                            limit = self.max_tool_rounds,
                            "OmniStage: tool round limit reached, ending the turn"
                        );
                        break;
                    }
                }
                OmniEvent::Usage(usage) => turn.usage += usage,
                OmniEvent::TurnComplete => {
                    if !output_since_tool {
                        continue;
                    }
                    if turn.reply.is_empty() && turn.audio.is_empty() {
                        wait = self.tool_call_grace;
                    } else {
                        break;
                    }
                }
                OmniEvent::Error(e) => return Err(stage_err(format!("provider error: {e}"))),
                _ => {}
            }
        }
        Ok(turn)
    }
}

impl OmniStage {
    /// A realtime model sometimes answers in text alone. The reply is what the
    /// model meant; this asks the same session to say it, keeps that audio, and
    /// leaves the reply text as it was.
    async fn speak_fallback(
        &self,
        provider: &dyn OmniProvider,
        events: &mut (impl Stream<Item = OmniEvent> + Unpin),
        deps: &TurnDeps<'_>,
        mut turn: Turn,
    ) -> Result<Turn> {
        warn!(
            reply_chars = turn.reply.len(),
            "OmniStage: reply had no audio; asking the model to say it"
        );
        let prompt = format!(
            "Say the following aloud, word for word, and say nothing else:

{}",
            turn.reply
        );
        // Best effort: the turn already has its answer, so a failure here is
        // logged, not raised.
        let spoken = match provider.send_text(&prompt).await {
            Ok(()) => self.collect_turn(provider, events, deps).await,
            Err(e) => Err(e),
        };
        match spoken {
            Ok(spoken) if !spoken.audio.is_empty() => {
                turn.audio = spoken.audio;
                turn.sample_rate = spoken.sample_rate;
                turn.channels = spoken.channels;
                turn.usage += spoken.usage;
            }
            Ok(spoken) => {
                warn!("OmniStage: the model produced no audio on the second attempt either");
                turn.usage += spoken.usage;
            }
            Err(e) => warn!("OmniStage: speak fallback failed: {e}"),
        }
        Ok(turn)
    }
}

async fn execute_tool(
    registry: &ToolRegistry,
    tool_ctx: &ToolContext,
    name: &str,
    args: Value,
) -> Value {
    let Some(tool) = registry.get(name) else {
        return serde_json::json!({ "error": format!("unknown tool '{name}'") });
    };
    match tool.execute(args, tool_ctx).await {
        Ok(text) => Value::String(text),
        Err(e) => serde_json::json!({ "error": e.to_string() }),
    }
}

fn append(buf: &mut String, text: &str) {
    if !buf.is_empty()
        && !buf.ends_with(char::is_whitespace)
        && !text.starts_with(char::is_whitespace)
    {
        buf.push(' ');
    }
    buf.push_str(text);
}

/// The persona stage's `llm_messages`, split the way a realtime session takes them:
/// every system message joined into one instruction, user/assistant turns as
/// history, and the live turn left out — it goes in as audio (or `send_text`).
fn split_llm_messages(ctx: &Context) -> (Option<String>, Vec<HistoryTurn>) {
    let live = ctx.get::<CurrentUserMessage>().map(|c| c.0);
    let mut system = Vec::new();
    let mut history = Vec::new();
    for (i, m) in ctx.llm_messages.iter().enumerate() {
        if Some(i) == live {
            continue;
        }
        match m.role {
            Role::System => system.push(m.text()),
            Role::User => history.push(turn(OmniRole::User, m)),
            Role::Assistant => history.push(turn(OmniRole::Model, m)),
            Role::Tool | Role::Unknown => {}
        }
    }
    let system = (!system.is_empty()).then(|| system.join("\n\n"));
    (system, history.into_iter().flatten().collect())
}

fn turn(role: OmniRole, m: &LlmMessage) -> Option<HistoryTurn> {
    let text = m.text();
    (!text.trim().is_empty()).then_some(HistoryTurn { role, text })
}

/// Decode a WAV to mono 16-bit PCM at `target_rate`, the shape every realtime
/// provider takes on input.
fn wav_to_pcm(wav: &[u8], target_rate: u32) -> Result<AudioChunk> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(wav))
        .map_err(|e| stage_err(format!("audio_data is not a WAV: {e}")))?;
    let spec = reader.spec();
    // hound accepts a header declaring a zero sample rate; resampling one would
    // saturate the output length and abort the process on the allocation.
    if spec.sample_rate == 0 {
        return Err(stage_err("audio_data declares a zero sample rate"));
    }
    let channels = spec.channels.max(1) as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let scale = ((1u64 << (spec.bits_per_sample - 1)) as f32).max(1.0);
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / scale)
                .collect()
        }
    };
    let mono: Vec<f32> = samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    let resampled = resample(&mono, spec.sample_rate, target_rate);
    let data: Vec<u8> = resampled
        .iter()
        .flat_map(|s| ((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes())
        .collect();
    Ok(AudioChunk {
        data,
        sample_rate: target_rate,
        channels: 1,
        bits_per_sample: 16,
    })
}

fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    // A zero rate would make `ratio` 0.0 and `out_len` saturate to `usize::MAX`,
    // aborting the process on the allocation. `wav_to_pcm` rejects it first;
    // this keeps the arithmetic sound for any other caller.
    if from == to || from == 0 || to == 0 || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let out_len = ((samples.len() as f64) / ratio).floor() as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;
            let a = samples[idx];
            let b = samples.get(idx + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

fn encode_wav(pcm: &[u8], sample_rate: u32, channels: u16) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: channels.max(1),
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Vec::with_capacity(pcm.len() + 44);
    let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut buf), spec)
        .map_err(|e| stage_err(format!("WAV writer: {e}")))?;
    let (pairs, _) = pcm.as_chunks::<2>();
    for pair in pairs {
        writer
            .write_sample(i16::from_le_bytes([pair[0], pair[1]]))
            .map_err(|e| stage_err(format!("WAV write: {e}")))?;
    }
    writer
        .finalize()
        .map_err(|e| stage_err(format!("WAV finalize: {e}")))?;
    Ok(buf)
}

fn stage_err(message: impl Into<String>) -> MindroidError {
    MindroidError::Pipeline {
        stage: "OmniStage".into(),
        message: message.into(),
        source: None,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        pin::Pin,
        sync::{Arc, Mutex},
    };

    use serde_json::json;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    use super::*;
    use crate::{
        AgentConfig, Message, MessageType, core::content::ContentPart, tools::ToolRegistry,
    };

    #[derive(Default)]
    struct Seen {
        config: Option<OmniConfig>,
        audio: Vec<AudioChunk>,
        text: Vec<String>,
        tool_results: Vec<(String, Value)>,
        ended: bool,
        disconnected: bool,
    }

    struct ScriptedProvider {
        seen: Arc<Mutex<Seen>>,
        events: Arc<Mutex<Option<mpsc::Receiver<OmniEvent>>>>,
    }

    #[async_trait]
    impl OmniProvider for ScriptedProvider {
        async fn connect(&mut self, config: &OmniConfig) -> std::result::Result<(), MindroidError> {
            self.seen.lock().unwrap().config = Some(config.clone());
            Ok(())
        }
        async fn send_audio(&self, chunk: AudioChunk) -> std::result::Result<(), MindroidError> {
            self.seen.lock().unwrap().audio.push(chunk);
            Ok(())
        }
        async fn send_text(&self, text: &str) -> std::result::Result<(), MindroidError> {
            self.seen.lock().unwrap().text.push(text.into());
            Ok(())
        }
        async fn send_tool_result(
            &self,
            id: &str,
            result: Value,
        ) -> std::result::Result<(), MindroidError> {
            self.seen
                .lock()
                .unwrap()
                .tool_results
                .push((id.into(), result));
            Ok(())
        }
        async fn end_audio_stream(&self) -> std::result::Result<(), MindroidError> {
            self.seen.lock().unwrap().ended = true;
            Ok(())
        }
        fn events(&mut self) -> Pin<Box<dyn Stream<Item = OmniEvent> + Send>> {
            let rx = self
                .events
                .lock()
                .unwrap()
                .take()
                .expect("events taken once");
            Box::pin(ReceiverStream::new(rx))
        }
        async fn disconnect(&mut self) -> std::result::Result<(), MindroidError> {
            self.seen.lock().unwrap().disconnected = true;
            Ok(())
        }
    }

    struct ScriptedBackend {
        seen: Arc<Mutex<Seen>>,
        script: Mutex<Vec<OmniEvent>>,
    }

    impl OmniBackend for ScriptedBackend {
        fn provider(&self) -> Box<dyn OmniProvider> {
            let (tx, rx) = mpsc::channel(64);
            let script = std::mem::take(&mut *self.script.lock().unwrap());
            tokio::spawn(async move {
                for event in script {
                    if tx.send(event).await.is_err() {
                        break;
                    }
                }
            });
            Box::new(ScriptedProvider {
                seen: Arc::clone(&self.seen),
                events: Arc::new(Mutex::new(Some(rx))),
            })
        }
        fn tool_declarations(&self, tools: &[Arc<dyn Tool>]) -> Value {
            json!(
                tools
                    .iter()
                    .map(|t| t.name().to_string())
                    .collect::<Vec<_>>()
            )
        }
        fn input_sample_rate(&self) -> u32 {
            16_000
        }
    }

    struct Echo;

    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echoes"
        }
        fn parameters_schema(&self) -> Value {
            json!({ "type": "object" })
        }
        async fn execute(&self, args: Value, _: &ToolContext) -> Result<String> {
            Ok(format!("echo:{}", args["v"]))
        }
    }

    fn chunk(data: Vec<u8>) -> OmniEvent {
        OmniEvent::AudioChunk(AudioChunk {
            data,
            sample_rate: 24_000,
            channels: 1,
            bits_per_sample: 16,
        })
    }

    fn transcript(source: TranscriptSource, text: &str) -> OmniEvent {
        OmniEvent::Transcript {
            text: text.into(),
            is_final: true,
            source,
        }
    }

    fn stage(script: Vec<OmniEvent>) -> (OmniStage, Arc<Mutex<Seen>>) {
        let seen = Arc::new(Mutex::new(Seen::default()));
        let backend = ScriptedBackend {
            seen: Arc::clone(&seen),
            script: Mutex::new(script),
        };
        let stage = OmniStage::new(backend, ToolRegistry::new().register(Echo))
            .with_turn_timeout(Duration::from_secs(2));
        (stage, seen)
    }

    fn audio_context() -> Context {
        let wav = encode_wav(&[0u8; 24], 48_000, 1).unwrap();
        let mut message = Message::new("", "microphone", "space-1");
        message.message_type = MessageType::Audio;
        message
            .metadata
            .insert("audio_data".into(), Value::String(STANDARD.encode(wav)));
        let mut ctx = Context::new(Arc::new(message), Arc::new(AgentConfig::default()));
        ctx.llm_messages = vec![
            LlmMessage::system("be brief"),
            LlmMessage {
                role: Role::User,
                content: vec![ContentPart::text("earlier question")],
            },
            LlmMessage {
                role: Role::Assistant,
                content: vec![ContentPart::text("earlier answer")],
            },
            LlmMessage::system("today is Monday"),
            LlmMessage {
                role: Role::User,
                content: vec![ContentPart::text("")],
            },
        ];
        ctx.set(CurrentUserMessage(4));
        ctx
    }

    #[tokio::test]
    async fn audio_turn_seeds_history_streams_audio_and_collects_the_reply() {
        let (stage, seen) = stage(vec![
            transcript(TranscriptSource::Input, "what time is it"),
            chunk(vec![1, 0, 2, 0]),
            transcript(TranscriptSource::Output, "It is"),
            chunk(vec![3, 0]),
            transcript(TranscriptSource::Output, "noon."),
            OmniEvent::Usage(Usage {
                input_tokens: 5,
                output_tokens: 7,
                ..Usage::default()
            }),
            OmniEvent::TurnComplete,
        ]);
        let mut ctx = audio_context();

        stage.process(&mut ctx).await.unwrap();

        assert_eq!(ctx.response.as_deref(), Some("It is noon."));
        assert_eq!(ctx.get_ext::<TextInput>().unwrap().0, "what time is it");
        assert_eq!(ctx.get::<OmniTurnUsage>().unwrap().0.output_tokens, 7);
        let wav = ctx.take_ext::<AudioOutput>().unwrap().0;
        let reader = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
        assert_eq!(reader.spec().sample_rate, 24_000);
        assert_eq!(reader.len(), 3);

        let seen = seen.lock().unwrap();
        let config = seen.config.as_ref().unwrap();
        assert_eq!(
            config.system_prompt.as_deref(),
            Some(
                format!(
                    "{VOICE_INSTRUCTION}

be brief

today is Monday"
                )
                .as_str()
            )
        );
        assert_eq!(
            config.history,
            vec![
                HistoryTurn {
                    role: OmniRole::User,
                    text: "earlier question".into()
                },
                HistoryTurn {
                    role: OmniRole::Model,
                    text: "earlier answer".into()
                },
            ]
        );
        assert_eq!(config.tools_schema, Some(json!(["echo"])));
        assert_eq!(seen.audio.len(), 1);
        assert_eq!(seen.audio[0].sample_rate, 16_000);
        assert_eq!(
            seen.audio[0].data.len(),
            4 * 2,
            "12 samples at 48 kHz, 4 at 16 kHz"
        );
        assert!(matches!(
            config.turn_detection,
            crate::omni::TurnDetection::Manual
        ));
        assert!(seen.ended && seen.disconnected);
        assert!(seen.text.is_empty());
    }

    #[tokio::test]
    async fn tool_call_after_an_empty_turn_complete_is_answered_and_the_reply_follows() {
        let (stage, seen) = stage(vec![
            OmniEvent::TurnComplete,
            OmniEvent::ToolCall {
                id: "c1".into(),
                name: "echo".into(),
                args: json!({ "v": 42 }),
            },
            transcript(TranscriptSource::Output, "forty-two"),
            OmniEvent::TurnComplete,
        ]);
        let mut ctx = audio_context();

        stage.process(&mut ctx).await.unwrap();

        assert_eq!(ctx.response.as_deref(), Some("forty-two"));
        let seen = seen.lock().unwrap();
        assert_eq!(
            seen.tool_results,
            vec![("c1".to_string(), Value::String("echo:42".into()))]
        );
    }

    #[tokio::test]
    async fn text_turn_goes_in_as_text() {
        let (stage, seen) = stage(vec![
            transcript(TranscriptSource::Output, "hi there"),
            OmniEvent::TurnComplete,
        ]);
        let message = Message::new("hello", "user", "space-1");
        let mut ctx = Context::new(Arc::new(message), Arc::new(AgentConfig::default()));
        ctx.llm_messages = vec![
            LlmMessage::system("sys"),
            LlmMessage {
                role: Role::User,
                content: vec![ContentPart::text("[Linn]: hello")],
            },
        ];
        ctx.set(CurrentUserMessage(1));

        stage.process(&mut ctx).await.unwrap();

        assert_eq!(ctx.response.as_deref(), Some("hi there"));
        assert!(ctx.get_ext::<AudioOutput>().is_none());
        let seen = seen.lock().unwrap();
        assert_eq!(seen.text[0], "[Linn]: hello");
        assert!(
            seen.text[1].contains("hi there"),
            "no audio came back, so the reply is re-asked aloud"
        );
        assert!(seen.config.as_ref().unwrap().history.is_empty());
    }

    #[tokio::test]
    async fn provider_error_fails_the_turn_and_still_disconnects() {
        let (stage, seen) = stage(vec![OmniEvent::Error(Arc::new(MindroidError::config(
            "boom",
        )))]);
        let mut ctx = audio_context();

        let err = stage.process(&mut ctx).await.unwrap_err();

        assert!(err.to_string().contains("boom"));
        assert!(seen.lock().unwrap().disconnected);
    }
}
