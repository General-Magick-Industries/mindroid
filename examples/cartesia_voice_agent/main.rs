//! Persona-driven agentic voice pipeline with Cartesia + AI-selected tone.
//!
//! ```text
//!   input ─▶ (Cartesia Ink STT if --audio)  transcript
//!                                               │
//!                       BifrostPersonaStage ────┤  persona prompt from Bifrost prepare
//!                                               │
//!                         ToolExecutorStage ────┤  agent loop; may call web_search
//!                                               │
//!                          ToneSelectorStage ───┤  a 2nd LLM picks the Cartesia tone
//!                                               │  that best fits the persona + reply
//!                         StreamingTtsStage ────┘  Cartesia Sonic TTS over a websocket,
//!                                                  streamed to your speakers 🔊
//! ```
//!
//! What this shows:
//!   1. **Persona from Bifrost.** Nothing but a `persona_id` (plus login creds):
//!      `BifrostPersonaStage` calls `POST /v1/persona/{persona_id}/prepare` and uses
//!      the returned system prompt verbatim.
//!   2. **A tone-deciding AI.** After the reply is generated, a dedicated LLM call
//!      (see `TONE_SELECTOR_SYSTEM_PROMPT`) reads the persona and the reply and
//!      chooses the Cartesia emotion + speed that best suit them.
//!   3. **It's an agent.** `ToolExecutorStage` can call the `web_search` tool.
//!   4. **Type or talk, always hear.** Text REPL by default (nice at a desk);
//!      `--audio file.wav` transcribes speech via Cartesia Ink instead. Either way
//!      the reply is streamed to your speakers as it's synthesized.
//!
//! ## Setup
//!
//! Copy the template and fill in your keys (the real file is git-ignored):
//!
//! ```bash
//! cp examples/cartesia_voice_agent/cartesia.example.toml \
//!    examples/cartesia_voice_agent/cartesia.toml
//! ```
//!
//! ## Run
//!
//! ```bash
//! cargo run --example cartesia_voice_agent               # text REPL (default)
//! cargo run --example cartesia_voice_agent -- --prompt "How are you?"
//! cargo run --example cartesia_voice_agent -- --audio input.wav
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use clap::Parser;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

use mindroid::auth::Auth;
use mindroid::auth::apikey::ApiKeyAuth;
use mindroid::llm_client::{ChatRequest, LlmClient, LlmClientConfig};
use mindroid::tools::{Tool, ToolRegistry};
use mindroid::{
    AgentConfig, BifrostPersonaStage, CartesiaGenerationConfig as GenerationConfig,
    CartesiaStreamingTts, CartesiaStt, CartesiaSttConfig, CartesiaTtsConfig, Context, LlmMessage,
    Message, MindroidError, Pipeline, PipelineStage, Result, Role, SttProvider, ToolExecutorStage,
};

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct AppConfig {
    bifrost: BifrostSection,
    llm: LlmSection,
    cartesia: CartesiaSection,
    #[serde(default)]
    web_search: WebSearchSection,
}

/// Bifrost persona: the stage calls `POST /v1/persona/{persona_id}/prepare`.
/// Auth is an email/password login against `base_url`.
#[derive(Deserialize)]
struct BifrostSection {
    base_url: String,
    email: String,
    password: String,
    persona_id: String,
}

/// LLM brain — any OpenAI-compatible endpoint (here LiteLLM → OpenRouter).
#[derive(Deserialize)]
struct LlmSection {
    base_url: String,
    api_key: Option<String>,
    model: String,
}

#[derive(Deserialize)]
struct CartesiaSection {
    api_key: String,
    voice_id: Option<String>,
    model_id: Option<String>,
    language: Option<String>,
    sample_rate: Option<u32>,
}

#[derive(Deserialize, Default)]
struct WebSearchSection {
    tavily_api_key: Option<String>,
}

// ─── Tone selection (the "AI that decides the tone") ─────────────────────────

/// Cartesia's valid `generation_config.emotion` values (sonic-3 / sonic-3.5).
const VALID_EMOTIONS: [&str; 5] = ["neutral", "calm", "angry", "content", "sad"];

/// System prompt for the tone-selection LLM call. Deliberately strict: the model
/// must return only JSON so parsing is reliable.
const TONE_SELECTOR_SYSTEM_PROMPT: &str = "\
You are a voice director for a text-to-speech engine. Given a character's persona \
and the exact line they are about to say, choose the delivery that best fits BOTH \
the character and the emotional content of the line.

Return ONLY compact JSON, no prose, in this shape:
{\"emotion\": <one of: neutral, calm, angry, content, sad>, \"speed\": <number 0.6-1.5>}

How to decide:
- emotion: match the feeling of the line as this character would express it. Use \
\"content\" for warm/happy/pleased lines, \"calm\" for reassuring or gentle lines, \
\"sad\" for sympathetic or somber lines, \"angry\" for firm/frustrated lines, and \
\"neutral\" for plain informational lines.
- speed: 1.0 is normal. Nudge faster (up to ~1.2) for excitement or urgency, slower \
(down to ~0.85) for calm, serious, or somber lines. Keep within 0.6-1.5.
- Stay in character: let the persona's temperament bias your choice (a bubbly \
persona leans warmer/faster; a stoic persona leans neutral/steadier).";

/// The tone chosen for the current turn, passed from [`ToneSelectorStage`] to
/// [`StreamingTtsStage`] via the pipeline context.
struct SelectedTone(GenerationConfig);

/// Clamp a model-proposed emotion/speed into Cartesia's accepted ranges.
fn normalize_tone(emotion: &str, speed: Option<f32>) -> GenerationConfig {
    let emotion = emotion.trim().to_lowercase();
    let emotion = if VALID_EMOTIONS.contains(&emotion.as_str()) {
        emotion
    } else {
        "neutral".to_string()
    };
    GenerationConfig {
        speed: speed.map(|s| s.clamp(0.6, 1.5)),
        volume: None,
        emotion: Some(emotion),
    }
}

/// Extract `{ "emotion": ..., "speed": ... }` from the model's reply.
fn parse_tone_json(s: &str) -> GenerationConfig {
    if let (Some(a), Some(b)) = (s.find('{'), s.rfind('}'))
        && a <= b
        && let Ok(v) = serde_json::from_str::<Value>(&s[a..=b])
    {
        let emotion = v
            .get("emotion")
            .and_then(|x| x.as_str())
            .unwrap_or("neutral");
        let speed = v.get("speed").and_then(|x| x.as_f64()).map(|f| f as f32);
        return normalize_tone(emotion, speed);
    }
    normalize_tone("neutral", None)
}

/// A pipeline stage that asks a second LLM to pick the voice tone for this turn,
/// based on the persona (the system prompt) and the generated reply.
struct ToneSelectorStage {
    client: LlmClient,
}

#[async_trait]
impl PipelineStage for ToneSelectorStage {
    fn name(&self) -> &str {
        "ToneSelectorStage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let reply = ctx.response.clone().unwrap_or_default();
        if reply.trim().is_empty() {
            return Ok(());
        }

        // The persona lives in the system prompt built by the persona stage.
        let persona = ctx
            .llm_messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.text())
            .unwrap_or_default();

        let user_msg = ctx
            .llm_messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.text())
            .unwrap_or_default();

        let prompt = format!(
            "PERSONA:\n{persona}\n\nThe user said:\n\"{user_msg}\"\n\n\
             The character is about to say this line out loud:\n\"{reply}\"\n\n\
             Choose the delivery. Return only the JSON."
        );

        let messages = vec![
            LlmMessage::system(TONE_SELECTOR_SYSTEM_PROMPT),
            LlmMessage::user(prompt),
        ];

        let tone = match self
            .client
            .chat(ChatRequest {
                messages: &messages,
                model: None,
                temperature: Some(0.0),
                max_tokens: Some(60),
                stream: false,
                response_format: None,
            })
            .await
        {
            Ok((text, _)) => parse_tone_json(&text),
            Err(e) => {
                tracing::warn!("ToneSelectorStage: tone LLM failed ({e}); defaulting to neutral");
                normalize_tone("neutral", None)
            }
        };

        tracing::info!(
            "Selected tone: emotion={:?} speed={:?}",
            tone.emotion,
            tone.speed
        );
        ctx.set_ext(SelectedTone(tone));
        Ok(())
    }
}

// ─── StreamingTtsStage ──────────────────────────────────────────────────────

/// Streams `ctx.response` from Cartesia Sonic's websocket using the tone chosen
/// by [`ToneSelectorStage`], and plays each raw-PCM chunk through the speakers as
/// it arrives — so audio starts before the full reply has finished synthesizing.
///
/// A dedicated thread owns the (non-`Send`) rodio output device and appends each
/// chunk to a sink; the async side pumps websocket chunks into it over a channel.
/// On a server you'd forward the same chunks over your own websocket instead of
/// playing them locally.
struct StreamingTtsStage {
    tts: Arc<CartesiaStreamingTts>,
}

#[async_trait]
impl PipelineStage for StreamingTtsStage {
    fn name(&self) -> &str {
        "StreamingTtsStage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let text = ctx.response.clone().unwrap_or_default();
        if text.trim().is_empty() {
            tracing::warn!("StreamingTtsStage: empty response, skipping synthesis");
            return Ok(());
        }

        let tone = ctx
            .take_ext::<SelectedTone>()
            .map(|s| s.0)
            .unwrap_or_default();

        let tone_label = tone.emotion.as_deref().unwrap_or("neutral").to_string();
        println!("\n🗣️  {text}");
        println!(
            "🔊 speaking [{}{}]",
            tone_label,
            tone.speed
                .map(|s| format!(" @ {s:.2}x"))
                .unwrap_or_default(),
        );

        let sample_rate = self.tts.sample_rate();

        // Playback thread: owns the rodio device, plays PCM chunks as they arrive.
        let (tx, rx) = std::sync::mpsc::channel::<Vec<i16>>();
        let player = std::thread::spawn(move || play_pcm_stream(sample_rate, rx));

        // Pump websocket audio chunks into the player.
        let mut stream = self.tts.stream(&text, &tone);
        let mut stream_err = None;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    if tx.send(pcm_bytes_to_i16(&bytes)).is_err() {
                        break; // playback thread ended
                    }
                }
                Err(e) => {
                    stream_err = Some(e);
                    break;
                }
            }
        }
        drop(tx); // signal end-of-stream so the player drains and stops

        if let Err(e) = player.join() {
            tracing::error!("StreamingTtsStage: playback thread panicked: {e:?}");
        }
        if let Some(e) = stream_err {
            return Err(e);
        }
        Ok(())
    }
}

/// Convert little-endian `pcm_s16le` bytes to `i16` samples.
fn pcm_bytes_to_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// Play a stream of mono PCM sample-blocks on the default output device, blocking
/// until every queued block has finished.
fn play_pcm_stream(sample_rate: u32, rx: std::sync::mpsc::Receiver<Vec<i16>>) {
    let (_stream, handle) = match rodio::OutputStream::try_default() {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("StreamingTtsStage: no audio output device: {e}");
            return;
        }
    };
    let sink = match rodio::Sink::try_new(&handle) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("StreamingTtsStage: sink error: {e}");
            return;
        }
    };
    while let Ok(samples) = rx.recv() {
        if !samples.is_empty() {
            sink.append(rodio::buffer::SamplesBuffer::new(1, sample_rate, samples));
        }
    }
    sink.sleep_until_end();
}

// ─── WebSearchTool ──────────────────────────────────────────────────────────

/// A web-search tool the agent can call mid-turn. Uses Tavily when a key is
/// configured (clean, LLM-friendly results); otherwise falls back to the keyless
/// DuckDuckGo Instant Answer API (short abstracts only).
struct WebSearchTool {
    client: reqwest::Client,
    tavily_key: Option<String>,
}

impl WebSearchTool {
    fn new(tavily_key: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            tavily_key: tavily_key.filter(|k| !k.is_empty()),
        }
    }

    async fn search_tavily(&self, key: &str, query: &str) -> Result<String> {
        let resp = self
            .client
            .post("https://api.tavily.com/search")
            .json(&json!({
                "api_key": key,
                "query": query,
                "max_results": 5,
                "include_answer": true,
            }))
            .send()
            .await
            .map_err(tool_err)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Ok(format!("Search failed ({status}): {body}"));
        }

        let v: Value = resp.json().await.map_err(tool_err)?;
        let mut out = String::new();
        if let Some(answer) = v.get("answer").and_then(|a| a.as_str())
            && !answer.is_empty()
        {
            out.push_str(&format!("Answer: {answer}\n\n"));
        }
        if let Some(results) = v.get("results").and_then(|r| r.as_array()) {
            for (i, r) in results.iter().enumerate() {
                let title = r.get("title").and_then(|t| t.as_str()).unwrap_or("");
                let url = r.get("url").and_then(|u| u.as_str()).unwrap_or("");
                let content = r.get("content").and_then(|c| c.as_str()).unwrap_or("");
                out.push_str(&format!("{}. {title} — {url}\n{content}\n\n", i + 1));
            }
        }
        if out.is_empty() {
            out.push_str("No results found.");
        }
        Ok(out)
    }

    async fn search_duckduckgo(&self, query: &str) -> Result<String> {
        let resp = self
            .client
            .get("https://api.duckduckgo.com/")
            .query(&[
                ("q", query),
                ("format", "json"),
                ("no_html", "1"),
                ("skip_disambig", "1"),
            ])
            .send()
            .await
            .map_err(tool_err)?;

        let v: Value = resp.json().await.map_err(tool_err)?;
        let mut out = String::new();

        if let Some(abstract_text) = v.get("AbstractText").and_then(|a| a.as_str())
            && !abstract_text.is_empty()
        {
            let src = v
                .get("AbstractSource")
                .and_then(|s| s.as_str())
                .unwrap_or("");
            let url = v.get("AbstractURL").and_then(|u| u.as_str()).unwrap_or("");
            out.push_str(&format!("{abstract_text}\n(source: {src} {url})\n\n"));
        }

        if let Some(topics) = v.get("RelatedTopics").and_then(|t| t.as_array()) {
            for t in topics.iter().take(5) {
                if let Some(text) = t.get("Text").and_then(|x| x.as_str()) {
                    out.push_str(&format!("- {text}\n"));
                }
            }
        }

        if out.trim().is_empty() {
            out.push_str(
                "No instant answer found (DuckDuckGo IA is limited). \
                 Set a Tavily key in [web_search] for full web search.",
            );
        }
        Ok(out)
    }
}

fn tool_err(e: reqwest::Error) -> MindroidError {
    MindroidError::Pipeline {
        stage: "WebSearchTool".into(),
        message: format!("HTTP error: {e}"),
        source: None,
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for current or factual information you don't already know \
         (news, recent events, specifics). Returns a short summary and top results."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "The search query." }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let query = args
            .get("query")
            .and_then(|q| q.as_str())
            .unwrap_or("")
            .trim();
        if query.is_empty() {
            return Ok("Error: no query provided".into());
        }
        match &self.tavily_key {
            Some(key) => self.search_tavily(key, query).await,
            None => self.search_duckduckgo(query).await,
        }
    }
}

// ─── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "cartesia-voice-agent",
    about = "Persona voice agent with AI-selected Cartesia tone"
)]
struct Cli {
    /// Path to the TOML config (copy from cartesia.example.toml).
    #[arg(
        short,
        long,
        default_value = "examples/cartesia_voice_agent/cartesia.toml"
    )]
    config: String,

    /// One-shot text prompt instead of the interactive REPL.
    #[arg(short, long)]
    prompt: Option<String>,

    /// Transcribe this WAV with Cartesia Ink STT and answer it (one-shot).
    #[arg(short, long)]
    audio: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,mindroid=info")
        .init();

    let cli = Cli::parse();

    let raw = std::fs::read_to_string(&cli.config).map_err(|e| {
        anyhow::anyhow!(
            "Could not read {} ({e}). Copy cartesia.example.toml → cartesia.toml and fill it in.",
            cli.config
        )
    })?;
    let cfg: AppConfig = toml::from_str(&raw)?;

    if cfg.cartesia.api_key.trim().is_empty() || cfg.cartesia.api_key.starts_with("sk_car_...") {
        anyhow::bail!("[cartesia].api_key is not set in {}", cli.config);
    }

    // Persona: Bifrost prepare endpoint, authenticated by email/password login.
    let auth: Arc<dyn Auth> = Arc::new(ApiKeyAuth::new(
        &cfg.bifrost.base_url,
        &cfg.bifrost.email,
        &cfg.bifrost.password,
    ));
    let persona = BifrostPersonaStage::new(&cfg.bifrost.base_url, &cfg.bifrost.persona_id, auth);
    tracing::info!(
        "Persona: Bifrost prepare for persona_id={}",
        cfg.bifrost.persona_id
    );

    // LLM brain + tone selector share one config.
    let mut llm_cfg = LlmClientConfig::new(&cfg.llm.base_url);
    llm_cfg.default_model = Some(cfg.llm.model.clone());
    llm_cfg.default_temperature = Some(0.7);
    llm_cfg.api_key = cfg.llm.api_key.clone();
    tracing::info!("LLM model: {}", cfg.llm.model);

    // Cartesia Sonic streaming TTS (websocket).
    let mut tts_cfg = CartesiaTtsConfig::new(
        cfg.cartesia.api_key.clone(),
        cfg.cartesia
            .voice_id
            .clone()
            .unwrap_or_else(|| "f786b574-daa5-4673-aa0c-cbe3e8534c02".to_string()),
    );
    if let Some(m) = cfg.cartesia.model_id.clone() {
        tts_cfg.model_id = m;
    }
    if let Some(l) = cfg.cartesia.language.clone() {
        tts_cfg.language = l;
    }
    if let Some(sr) = cfg.cartesia.sample_rate {
        tts_cfg.sample_rate = sr;
    }
    let tts = Arc::new(CartesiaStreamingTts::new(tts_cfg));

    // Agent tools.
    let registry = Arc::new(
        ToolRegistry::new().register(WebSearchTool::new(cfg.web_search.tavily_api_key.clone())),
    );

    // Build the pipeline once; reuse it for every turn.
    let pipeline = Pipeline::new()
        .add_stage(persona)
        .add_streaming_stage(ToolExecutorStage::new(
            LlmClient::new(llm_cfg.clone())?,
            registry,
        ))
        .add_stage(ToneSelectorStage {
            client: LlmClient::new(llm_cfg)?,
        })
        .add_stage(StreamingTtsStage { tts });

    // Resolve input mode.
    if let Some(audio_path) = &cli.audio {
        let bytes = std::fs::read(audio_path)?;
        let stt = CartesiaStt::new(cartesia_stt_config(&cfg.cartesia));
        let transcript = stt.transcribe(&bytes).await?;
        println!("📝 You (from audio): {transcript}");
        run_turn(&pipeline, &transcript).await?;
    } else if let Some(prompt) = &cli.prompt {
        run_turn(&pipeline, prompt).await?;
    } else {
        repl(&pipeline).await?;
    }

    Ok(())
}

/// Build a Cartesia STT config from the `[cartesia]` section.
fn cartesia_stt_config(c: &CartesiaSection) -> CartesiaSttConfig {
    let mut cfg = CartesiaSttConfig::new(c.api_key.clone());
    if let Some(l) = &c.language {
        cfg.language = l.clone();
    }
    cfg
}

/// Run a single turn: feed `text` through the pipeline (synthesis happens inside).
async fn run_turn(pipeline: &Pipeline, text: &str) -> anyhow::Result<()> {
    let message = Arc::new(Message::new(text, "user", "cli"));
    let mut ctx = Context::new(message, Arc::new(AgentConfig::default()));
    pipeline.run(&mut ctx).await?;
    Ok(())
}

/// Interactive text REPL — type instead of talking.
async fn repl(pipeline: &Pipeline) -> anyhow::Result<()> {
    use std::io::{BufRead, Write};

    println!("💬 Text chat ready. Type a message (empty line or 'quit' to exit).");
    let stdin = std::io::stdin();
    loop {
        print!("\nYou: ");
        std::io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF (Ctrl-D / Ctrl-Z)
        }
        let line = line.trim();
        if line.is_empty() || line.eq_ignore_ascii_case("quit") || line.eq_ignore_ascii_case("exit")
        {
            break;
        }

        if let Err(e) = run_turn(pipeline, line).await {
            tracing::error!("Turn failed: {e}");
        }
    }
    println!("👋 Bye.");
    Ok(())
}
