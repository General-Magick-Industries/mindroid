//! Voice agent: microphone → STT → LLM (with tools) → TTS.
//!
//! Demonstrates the full audio pipeline: AudioTransport captures speech from
//! the microphone using Silero VAD, SttStage transcribes it, ToolExecutorStage
//! generates a response and can control the local computer (run commands, open
//! apps and URLs), and TtsStage synthesizes the final spoken reply.
//!
//! Example: say "drop the needle" and the agent will check if Spotify is
//! running, open it if not, and start playback — all via shell commands.
//!
//! Run with:
//!   cargo run -p mindroid-example-voice-agent --bin voice_agent
//!
//! Text mode (keyboard input):
//!   cargo run -p mindroid-example-voice-agent --bin voice_agent -- --text
//!
//! Custom config:
//!   cargo run -p mindroid-example-voice-agent --bin voice_agent -- --config examples/voice_agent/config.toml
//!
//! See `examples/voice_agent/config.toml` for all configuration options.
//! API keys can be set in the config or via environment variables:
//!   OPENAI_API_KEY, DEEPGRAM_API_KEY

use std::io::{BufRead, Write as _};
use std::sync::Arc;

use clap::Parser;

use mindroid::TransportSender;
use mindroid::auth::static_id::StaticAuth;
use mindroid::llm_client::LlmClient;
use mindroid::pipeline::stages::{PostProcessor, SimpleContextBuilder, ToolExecutorStage};
use mindroid::runtime::TransportSend;
use mindroid::skills::SkillSet;
use mindroid::tools::ToolRegistry;
use mindroid::tools::reminder::{ReminderRoutine, SetReminderTool, new_reminder_store};
use mindroid::transport::audio::{AudioTransport, AudioTransportConfig};
#[cfg(feature = "transport-audio")]
use mindroid::{
    ChannelType, Message, MessageType, MindroidConfig, Pipeline, Response,
    Result as MindroidResult, Runtime, SenderType, StreamEvent, Transport, TtsProvider,
};

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "voice-agent",
    about = "Mindroid voice agent — talk to your computer",
    long_about = "A voice-controlled AI assistant that captures speech from the microphone,\n\
                  transcribes it, generates responses with tool use, and speaks back.\n\n\
                  In text mode (--text), type messages instead of speaking."
)]
struct Cli {
    /// Path to the TOML config file
    #[arg(short, long, default_value = "examples/voice_agent/config.toml")]
    config: String,

    /// Use keyboard input instead of microphone
    #[arg(short, long)]
    text: bool,

    /// Override the TTS voice (e.g. aura-asteria-en, nova, alloy)
    #[arg(long)]
    voice: Option<String>,

    /// Override the LLM model for responses
    #[arg(long)]
    model: Option<String>,

    /// Override the STT provider (openai, deepgram)
    #[arg(long)]
    stt: Option<String>,

    /// Override the TTS provider (openai, deepgram)
    #[arg(long)]
    tts: Option<String>,

    /// Skills directory path
    #[arg(long, default_value = "examples/voice_agent/skills")]
    skills_dir: String,

    /// Log verbosity: -v info, -vv debug, -vvv trace
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Suppress all output except errors
    #[arg(short, long)]
    quiet: bool,
}

// ─── chat transport ──────────────────────────────────────────────────────────

/// Signal shared between the listen loop and on_message callback so the
/// prompt is only printed after the agent has finished its full response.
static PROMPT_READY: std::sync::LazyLock<tokio::sync::Notify> =
    std::sync::LazyLock::new(tokio::sync::Notify::new);

/// A stdio transport that shows `You: ` prompts and formats output for chat.
struct ChatTransport {
    connected: bool,
}

impl ChatTransport {
    fn new() -> Self {
        Self { connected: false }
    }

    fn print_prompt() {
        let mut stdout = std::io::stdout().lock();
        let _ = write!(stdout, "\x1b[1;36mYou:\x1b[0m ");
        let _ = stdout.flush();
    }
}

#[async_trait::async_trait]
impl Transport for ChatTransport {
    fn name(&self) -> &str {
        "chat"
    }

    async fn connect(&mut self) -> MindroidResult<()> {
        self.connected = true;
        Ok(())
    }

    async fn disconnect(&mut self) -> MindroidResult<()> {
        self.connected = false;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<Message>) -> MindroidResult<()> {
        tokio::spawn(async move {
            Self::print_prompt();

            loop {
                // Read a line on a blocking thread so we don't hold the runtime.
                let line = tokio::task::spawn_blocking(|| {
                    let mut buf = String::new();
                    match std::io::stdin().lock().read_line(&mut buf) {
                        Ok(0) => None, // EOF
                        Ok(_) => Some(buf),
                        Err(_) => None,
                    }
                })
                .await;

                let line = match line {
                    Ok(Some(l)) => l.trim().to_string(),
                    _ => break,
                };
                if line.is_empty() {
                    Self::print_prompt();
                    continue;
                }

                let message = Message {
                    id: uuid::Uuid::new_v4().to_string(),
                    content: line,
                    sender_id: "stdin".to_string(),
                    sender_type: SenderType::User,
                    channel_id: "stdio".to_string(),
                    channel_type: ChannelType::Direct,
                    message_type: MessageType::Text,
                    timestamp: chrono::Utc::now(),
                    metadata: Default::default(),
                    platform: None,
                };
                if tx.send(message).await.is_err() {
                    break;
                }

                // Wait for on_message to signal that the response is done
                // before printing the next prompt.
                PROMPT_READY.notified().await;
                Self::print_prompt();
            }
        });
        Ok(())
    }

    async fn send(&self, _response: &Response) -> MindroidResult<Option<String>> {
        // Output is handled by the on_message callback, not here.
        Ok(None)
    }
}

/// A no-op sender — chat output goes through the on_message streaming callback.
struct ChatSender;

#[async_trait::async_trait]
impl TransportSend for ChatSender {
    async fn send(&self, _response: &Response) -> MindroidResult<Option<String>> {
        Ok(None)
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Resolve an API key from the config provider, falling back to an env var.
fn resolve_api_key(config: &MindroidConfig, provider_name: &str, env_var: &str) -> String {
    config
        .providers
        .get(provider_name)
        .and_then(|p| p.api_key.as_deref())
        .filter(|k| !k.is_empty())
        .map(String::from)
        .or_else(|| std::env::var(env_var).ok())
        .unwrap_or_default()
}

/// Build an STT provider from `[models.stt]` + `[providers.*]` config.
fn build_stt(config: &MindroidConfig) -> Box<dyn mindroid::SttProvider> {
    let model_cfg = config
        .models
        .get("stt")
        .expect("[models.stt] is required in config");
    let provider_name = &model_cfg.provider;
    let model = model_cfg.model.as_deref().unwrap_or("whisper-1");

    match provider_name.as_str() {
        #[cfg(feature = "speech")]
        "deepgram" => {
            let api_key = resolve_api_key(config, "deepgram", "DEEPGRAM_API_KEY");
            let base_url = config
                .providers
                .get("deepgram")
                .and_then(|p| p.base_url.clone())
                .unwrap_or_else(|| "https://api.deepgram.com".into());
            Box::new(mindroid::DeepgramStt::new(mindroid::DeepgramSttConfig {
                api_key,
                model: model.into(),
                base_url,
                ..Default::default()
            }))
        }
        _ => {
            let api_key = resolve_api_key(config, "openai", "OPENAI_API_KEY");
            let base_url = config
                .providers
                .get("openai")
                .and_then(|p| p.base_url.clone());
            Box::new(mindroid::OpenAiStt::new(mindroid::OpenAiSttConfig {
                api_key,
                model: model.into(),
                base_url,
            }))
        }
    }
}

/// Build a TTS provider from `[models.tts]` + `[providers.*]` config.
fn build_tts(config: &MindroidConfig) -> Box<dyn mindroid::TtsProvider> {
    let model_cfg = config
        .models
        .get("tts")
        .expect("[models.tts] is required in config");
    let provider_name = &model_cfg.provider;
    let model = model_cfg.model.as_deref().unwrap_or("tts-1");
    let voice = model_cfg
        .options
        .get("voice")
        .and_then(|v| v.as_str())
        .unwrap_or("nova");

    match provider_name.as_str() {
        #[cfg(feature = "speech")]
        "deepgram" => {
            let api_key = resolve_api_key(config, "deepgram", "DEEPGRAM_API_KEY");
            let base_url = config
                .providers
                .get("deepgram")
                .and_then(|p| p.base_url.clone())
                .unwrap_or_else(|| "https://api.deepgram.com".into());
            Box::new(mindroid::DeepgramTts::new(mindroid::DeepgramTtsConfig {
                api_key,
                model: model.into(),
                base_url,
            }))
        }
        _ => {
            let api_key = resolve_api_key(config, "openai", "OPENAI_API_KEY");
            let base_url = config
                .providers
                .get("openai")
                .and_then(|p| p.base_url.clone());
            Box::new(mindroid::OpenAiTts::new(mindroid::OpenAiTtsConfig {
                api_key,
                model: model.into(),
                voice: voice.into(),
                base_url,
            }))
        }
    }
}

// ─── audio playback ──────────────────────────────────────────────────────────

#[cfg(feature = "transport-audio")]
fn play_audio(wav_bytes: Vec<u8>) {
    use std::io::Cursor;
    let (_stream, handle) = match rodio::OutputStream::try_default() {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("Audio output error: {e}");
            return;
        }
    };
    let sink = match rodio::Sink::try_new(&handle) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("Audio sink error: {e}");
            return;
        }
    };
    let source = match rodio::Decoder::new(Cursor::new(wav_bytes)) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("Audio decode error: {e}");
            return;
        }
    };
    sink.append(source);
    sink.sleep_until_end();
    std::thread::sleep(std::time::Duration::from_millis(80));
}

async fn speak_sentence(tts: &Arc<dyn TtsProvider>, text: &str) {
    let clean: String = text
        .chars()
        .filter(|c| !matches!(c, '*' | '_' | '`' | '#'))
        .collect();
    let clean = clean.trim().to_string();
    if clean.is_empty() {
        return;
    }

    match tts.synthesize(&clean).await {
        Ok(wav) => {
            #[cfg(feature = "transport-audio")]
            tokio::task::spawn_blocking(move || play_audio(wav))
                .await
                .ok();
        }
        Err(e) => tracing::error!("TTS error: {e}"),
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Set up logging based on verbosity flags.
    let filter = if cli.quiet {
        "error"
    } else {
        match cli.verbose {
            0 => "warn,mindroid=warn",
            1 => "info,mindroid=info",
            2 => "debug,mindroid=debug",
            _ => "trace",
        }
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let mut config = MindroidConfig::resolve(Some(&cli.config))?;

    // Apply CLI overrides to config.
    if let Some(ref model) = cli.model
        && let Some(respond) = config.models.get_mut("respond")
    {
        respond.model = Some(model.clone());
    }
    if let Some(ref stt_provider) = cli.stt
        && let Some(stt) = config.models.get_mut("stt")
    {
        stt.provider = stt_provider.clone();
    }
    if let Some(ref tts_provider) = cli.tts
        && let Some(tts) = config.models.get_mut("tts")
    {
        tts.provider = tts_provider.clone();
    }
    if let Some(ref voice) = cli.voice
        && let Some(tts) = config.models.get_mut("tts")
    {
        tts.options
            .insert("voice".into(), serde_json::Value::String(voice.clone()));
    }

    let tts = Arc::from(build_tts(&config));
    let llm = LlmClient::new(config.llm("respond")?)?;

    // Discover skills from the skills directory.
    let skills = SkillSet::from_workspace(&cli.skills_dir).await;

    // Create shared reminder store for the reminder tool + routine.
    let reminder_store = new_reminder_store();

    // Build tool registry from config (shell + open enabled by default),
    // plus the read_skill tool for on-demand skill loading and reminders.
    let registry = skills
        .extend_tools(ToolRegistry::from_config(&config.tools))
        .register(SetReminderTool::new(reminder_store.clone()));
    let registry = Arc::new(registry);

    let respond_cfg = config.models.get("respond").unwrap();

    // ─── startup banner ──────────────────────────────────────────────────
    let text_mode = cli.text;
    if !cli.quiet {
        println!();
        println!("  \x1b[1;35m╭─────────────────────────────────────╮\x1b[0m");
        if text_mode {
            println!(
                "  \x1b[1;35m│\x1b[0m  Mindroid Voice Agent  \x1b[2m(text mode)\x1b[0m  \x1b[1;35m│\x1b[0m"
            );
        } else {
            println!(
                "  \x1b[1;35m│\x1b[0m  Mindroid Voice Agent  \x1b[2m(mic mode)\x1b[0m   \x1b[1;35m│\x1b[0m"
            );
        }
        println!("  \x1b[1;35m╰─────────────────────────────────────╯\x1b[0m");
        println!();

        if !text_mode {
            let stt_cfg = config.models.get("stt").unwrap();
            let tts_cfg = config.models.get("tts").unwrap();
            println!(
                "    \x1b[2mSTT\x1b[0m   {} ({})",
                stt_cfg.provider,
                stt_cfg.model.as_deref().unwrap_or("default")
            );
            println!(
                "    \x1b[2mTTS\x1b[0m   {} ({})",
                tts_cfg.provider,
                tts_cfg.model.as_deref().unwrap_or("default")
            );
        }
        println!(
            "    \x1b[2mLLM\x1b[0m   {} ({})",
            respond_cfg.provider,
            respond_cfg.model.as_deref().unwrap_or("default")
        );

        if !skills.is_empty() {
            println!(
                "    \x1b[2mSkills\x1b[0m {}",
                skills.skill_names().join(", ")
            );
        }

        let tool_names: Vec<_> = registry.tools().iter().map(|t| t.name()).collect();
        println!("    \x1b[2mTools\x1b[0m  {}", tool_names.join(", "));

        println!();
        if text_mode {
            println!("  Type a message and press Enter. Ctrl+C to quit.\n");
        } else {
            println!("  Speak into your microphone. Ctrl+C to quit.\n");
        }
    }

    let jarvis_prompt = "You are Genseeks, the sophisticated AI assistant developed and designed by Adrian. \
        You are calm, precise, subtly witty, and unfailingly helpful. \
        You can control the computer you run on using tools — use them \
        whenever the user asks you to play music, open apps, check system \
        state, or interact with anything on this machine. \
        Speak in concise, well-formed sentences. \
        Never use markdown, bullet points, or lists — only natural spoken prose. \
        Keep responses under 80 words unless detail is essential. \
        IMPORTANT — your responses are read aloud by a text-to-speech engine, so: \
        always express times in spoken form (say 'four nineteen PM' not '16:19:59'); \
        express dates as 'March 4th' not '2026-03-04'; \
        never include timezone codes like UTC+07, just say 'local time' if relevant; \
        translate any raw command output into a natural spoken sentence.";

    let tts: Arc<dyn TtsProvider> = tts;

    let builder = if text_mode {
        let pipeline = Pipeline::new()
            .add_stage(SimpleContextBuilder::with_prompt(jarvis_prompt).with_skills(&skills))
            .add_streaming_stage(ToolExecutorStage::new(llm, registry))
            .add_stage(PostProcessor);
        Runtime::builder()
            .config(config)
            .transport(ChatTransport::new())
            .transport_sender(TransportSender::new(ChatSender))
            .pipeline(pipeline)
            .auth(StaticAuth::new("voice-agent"))
            .add_routine(ReminderRoutine::new(reminder_store))
    } else {
        let stt = build_stt(&config);
        let pipeline = Pipeline::new()
            .add_stage(mindroid::SttStage::from_arc(Arc::from(stt)))
            .add_stage(SimpleContextBuilder::with_prompt(jarvis_prompt).with_skills(&skills))
            .add_streaming_stage(ToolExecutorStage::new(llm, registry))
            .add_stage(PostProcessor);
        let audio =
            AudioTransport::with_config(&config.agent.agent_id, AudioTransportConfig::default());
        Runtime::builder()
            .config(config)
            .transport(audio)
            .transport_sender(TransportSender::new(ChatSender))
            .pipeline(pipeline)
            .auth(StaticAuth::new("voice-agent"))
            .add_routine(ReminderRoutine::new(reminder_store))
    };

    let quiet = cli.quiet;
    let mut runtime = builder
        .on_message(move |ctx| {
            let tts = tts.clone();
            async move {
                use futures::StreamExt;

                let mut stream = ctx.process_streaming();
                let mut response_text = String::new();
                let mut printed_prefix = false;
                let mut stdout = std::io::stdout();

                while let Some(event) = stream.next().await {
                    match event {
                        StreamEvent::Chunk { content } => {
                            if !quiet {
                                if !printed_prefix {
                                    let _ = write!(stdout, "\x1b[1;32mAgent:\x1b[0m ");
                                    printed_prefix = true;
                                }
                                let _ = write!(stdout, "{content}");
                                let _ = stdout.flush();
                            }
                            response_text.push_str(&content);
                        }
                        StreamEvent::ToolCall { name, arguments } if !quiet => {
                            if printed_prefix {
                                println!();
                                printed_prefix = false;
                            }
                            println!(
                                "       \x1b[2;33m[tool: {name}]\x1b[0m \x1b[2m{arguments}\x1b[0m"
                            );
                        }
                        StreamEvent::ToolResult { name, result } if !quiet => {
                            let preview: String = result
                                .lines()
                                .next()
                                .unwrap_or("")
                                .trim()
                                .chars()
                                .take(80)
                                .collect();
                            println!(
                                "       \x1b[2;32m[result: {name}]\x1b[0m \x1b[2m{preview}\x1b[0m"
                            );
                        }
                        StreamEvent::Error { message } => {
                            if printed_prefix {
                                println!();
                            }
                            eprintln!("       \x1b[1;31m[error]\x1b[0m {message}");
                            break;
                        }
                        _ => {}
                    }
                }

                if printed_prefix {
                    println!();
                }

                if !quiet {
                    println!();
                }

                // Signal the listen loop to print the next prompt BEFORE TTS,
                // so the user can type while audio plays.
                PROMPT_READY.notify_one();

                // Speak the response (TTS). In text mode the prompt is already
                // visible so the user can type while audio plays.
                let speakable = response_text.trim().to_string();
                if !speakable.is_empty() {
                    speak_sentence(&tts, &speakable).await;
                }
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
