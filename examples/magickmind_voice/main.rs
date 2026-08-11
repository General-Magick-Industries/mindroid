//! Magickmind Voice Agent: Centrifugo websocket input → LLM with tools → TTS output.
//!
//! Messages arrive via Centrifugo (Magickmind websocket). The magickmind WsMessage
//! envelope carries a ChatHistoryItem payload whose `magickspace_id` field is
//! extracted by the transport and placed in `message.channel_id` — that is the
//! correct MagickMind magickspace ID to use for context and persistence calls.
//!
//! The agent can control the local computer using shell and open tools.
//! Responses are spoken aloud via TTS and persisted back to MagickMind.
//!
//! Run with:
//!   cargo run -p mindroid-example-magickmind-voice --bin magickmind_voice -- \
//!     --config examples/magickmind_voice/config.toml

use std::sync::Arc;

use mindroid::auth::apikey::ApiKeyAuth;
use mindroid::llm_client::LlmClient;
use mindroid::memory::magickmind::MagickmindMemory;
use mindroid::observer::log::LogObserver;
use mindroid::pipeline::presets::magickmind::{MagickmindClient, MagickmindContextConfig};
use mindroid::pipeline::stages::{PostProcessor, SimpleContextBuilder, ToolExecutorStage};
use mindroid::tools::ToolRegistry;
use mindroid::transport::centrifugo::CentrifugoTransport;
#[cfg(feature = "transport-audio")]
use mindroid::{MindroidConfig, Pipeline, PipelineContext, Runtime, StreamEvent, TtsProvider};

// ─── TTS ──────────────────────────────────────────────────────────────────────

fn resolve_api_key(config: &MindroidConfig, provider: &str, env_var: &str) -> String {
    config
        .providers
        .get(provider)
        .and_then(|p| p.api_key.as_deref())
        .filter(|k| !k.is_empty())
        .map(String::from)
        .or_else(|| std::env::var(env_var).ok())
        .unwrap_or_default()
}

fn build_tts(config: &MindroidConfig) -> Box<dyn TtsProvider> {
    let m = config.models.get("tts").expect("[models.tts] required");
    let voice = m
        .options
        .get("voice")
        .and_then(|v| v.as_str())
        .unwrap_or("nova");
    match m.provider.as_str() {
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
                model: m.model.as_deref().unwrap_or("tts-1").into(),
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
                model: m.model.as_deref().unwrap_or("tts-1").into(),
                voice: voice.into(),
                base_url,
            }))
        }
    }
}

// ─── audio playback ───────────────────────────────────────────────────────────

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

async fn speak(tts: &Arc<dyn TtsProvider>, text: &str) {
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

// ─── main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,mindroid=debug")
        .init();

    let config = MindroidConfig::resolve_from_args()?;

    let respond_llm = config.llm("respond")?;
    let tts = build_tts(&config);
    let tools_cfg = config.tools.clone();

    let magickmind_url = config
        .auth
        .base_url
        .as_deref()
        .unwrap_or("https://dev-magickmind.magickmind.ai")
        .to_string();
    let email = config.auth.email.as_deref().unwrap_or("").to_string();
    let password = config.auth.password.as_deref().unwrap_or("").to_string();
    let api_key = config.auth.api_key.clone();
    let ws_url = config
        .transport
        .url
        .as_deref()
        .unwrap_or("wss://dev-centrifugo.magickmind.ai/connection/websocket")
        .to_string();
    let agent_id = config.agent.agent_id.clone();

    let respond_cfg = config.models.get("respond").unwrap();
    let tts_cfg = config.models.get("tts").unwrap();
    println!("Magickmind voice agent — agent '{agent_id}'");
    println!(
        "  LLM : {} ({})",
        respond_cfg.provider,
        respond_cfg.model.as_deref().unwrap_or("default")
    );
    println!(
        "  TTS : {} ({})",
        tts_cfg.provider,
        tts_cfg.model.as_deref().unwrap_or("default")
    );
    println!("  MagickMind: {magickmind_url}");
    println!();

    let identity: Arc<dyn mindroid::Auth> =
        Arc::new(ApiKeyAuth::new(&magickmind_url, &email, &password));

    let transport = CentrifugoTransport::new(&ws_url, &agent_id, identity.clone());
    let memory = MagickmindMemory::new(&magickmind_url, identity.clone());

    let mut magickmind_client = MagickmindClient::new(&magickmind_url, identity.clone());
    if let Some(ref key) = api_key {
        magickmind_client = magickmind_client.with_api_key(key);
    }
    let magickmind = Arc::new(magickmind_client);

    let llm_config = Arc::new(respond_llm);
    let registry = Arc::new(ToolRegistry::from_config(&tools_cfg));
    let tts: Arc<dyn TtsProvider> = Arc::from(tts);

    let prompt: Arc<str> = Arc::from(
        "You are Genseeks, the sophisticated AI assistant developed and designed by Adrian. \
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
        translate any raw command output into a natural spoken sentence.",
    );

    let mut runtime = Runtime::builder()
        .config(config)
        .transport(transport)
        .auth_shared(identity)
        .memory(memory)
        .observer(LogObserver::new())
        .on_message(move |ctx| {
            let tts = tts.clone();
            let magickmind = Arc::clone(&magickmind);
            let llm_config = Arc::clone(&llm_config);
            let registry = Arc::clone(&registry);
            let prompt = Arc::clone(&prompt);

            async move {
                use futures::StreamExt;

                // message.channel_id is the MagickMind magickspace_id — extracted by
                // parse_push from the ChatHistoryItem.magickspace_id field in the
                // magickmind WsMessage envelope.
                let magickspace_id = ctx.message.channel_id.clone();
                let sender_id = ctx.message.sender_id.clone();
                let content = ctx.message.content.clone();
                let msg_id = ctx.message.id.clone();

                tracing::info!(
                    "Message from {sender_id} in magickspace {magickspace_id}: {content:?}"
                );

                // Fetch conversation history and knowledge from MagickMind.
                let context = magickmind
                    .prepare_context(
                        &magickspace_id,
                        &sender_id,
                        &content,
                        &MagickmindContextConfig::default(),
                        Some(&ctx.agent_config.agent_id),
                    )
                    .await
                    .unwrap_or_else(|e| {
                        tracing::warn!("MagickMind context load failed: {e}");
                        Vec::new()
                    });

                let history = Arc::new(context);

                // Build pipeline per-request so history can be injected at construction time.
                let pipeline = Pipeline::new()
                    .add_stage(SimpleContextBuilder::with_prompt_and_history(
                        prompt.as_ref(),
                        history,
                    ))
                    .add_streaming_stage(match LlmClient::new((*llm_config).clone()) {
                        Ok(c) => ToolExecutorStage::new(c, Arc::clone(&registry)),
                        Err(e) => {
                            tracing::error!("LlmClient init failed: {e}");
                            return;
                        }
                    })
                    .add_stage(PostProcessor);

                let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

                let mut response_text = String::new();
                {
                    let mut stream = ctx.run_streaming_with_context(&pipeline, &mut pctx);
                    while let Some(event) = stream.next().await {
                        match event {
                            StreamEvent::Chunk { content } => {
                                response_text.push_str(&content);
                            }
                            StreamEvent::ToolCall { name, arguments } => {
                                println!("  [using tool: {name}] {arguments}");
                            }
                            StreamEvent::ToolResult { name, result } => {
                                let preview = result.lines().next().unwrap_or("").trim();
                                println!("  [tool result: {name}] {preview}");
                            }
                            StreamEvent::Error { message } => {
                                tracing::error!("LLM error: {message}");
                                break;
                            }
                            _ => {}
                        }
                    }
                }

                let speakable = response_text.trim().to_string();
                if speakable.is_empty() {
                    return;
                }

                println!("\nAgent: {speakable}\n");

                // Persist the response back to the same magickspace.
                if let Err(e) = magickmind
                    .save_message(
                        &magickspace_id,
                        &ctx.agent_config.agent_id,
                        &speakable,
                        Some(&msg_id),
                    )
                    .await
                {
                    tracing::warn!("MagickMind persistence failed: {e}");
                }

                speak(&tts, &speakable).await;
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
