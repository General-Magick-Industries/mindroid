//! Talk to OpenAI Realtime through the default microphone and speaker.
//!
//! Run direct:
//!   OPENAI_API_KEY=... cargo run -p mindroid-example-openai-realtime --bin openai_realtime
//!
//! Or through the LiteLLM passthrough using the root .env:
//!   set -a; . ../.env; set +a
//!   cargo run -p mindroid-example-openai-realtime --bin openai_realtime
//!
//! Nothing connects until you speak: a local VAD opens a session on the first
//! words and closes it after OMNI_IDLE_SECONDS (default 15) of silence, which is
//! when the user turns are transcribed and written to memory. The next words open
//! a fresh session that seeds that history back in.
//!
//! Optional: OPENAI_REALTIME_MODEL, OPENAI_VOICE, OMNI_MAX_SECONDS (auto-stop),
//! OMNI_IDLE_SECONDS, OMNI_STT_MODEL, OMNI_STT_LANGUAGE (default en),
//! OPENAI_INPUT_TRANSCRIPTION=on, OMNI_VAD_THRESHOLD (default 0.5).
//! Ctrl-C ends the run.
//!
//! Turns persist to ./omni-voice.db.

use std::sync::Arc;

use async_trait::async_trait;
use mindroid::memory::sqlite::SqliteMemory;
use mindroid::omni::cpal_audio::CpalAudio;
use mindroid::omni::openai_realtime::{
    OpenAiRealtimeConfig, OpenAiRealtimeProvider, SAMPLE_RATE, tool_declarations,
};
use mindroid::omni::types::OmniConfig;
use mindroid::omni::{AudioSink, AudioSource, OmniProvider, SileroDetector, VoiceGate};
use mindroid::pipeline::stages::stt::SttProvider;
use mindroid::pipeline::stages::{OpenAiStt, OpenAiSttConfig};
use mindroid::tools::{Tool, ToolContext};
use mindroid::voice::types::VadConfig;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

struct TimeTool;

#[async_trait]
impl Tool for TimeTool {
    fn name(&self) -> &str {
        "current_time"
    }

    fn description(&self) -> &str {
        "Returns the current local time as an ISO-8601 string"
    }

    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn execute(&self, _args: Value, _ctx: &ToolContext) -> mindroid::Result<String> {
        Ok(chrono::Local::now().to_rfc3339())
    }
}

/// Speech-to-text over the same credentials as the realtime session:
/// `OPENAI_API_KEY` direct, else the LiteLLM pair. Model from `OMNI_STT_MODEL`.
fn own_transcriber() -> anyhow::Result<Arc<dyn SttProvider>> {
    let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    let (api_key, base_url) = match var("OPENAI_API_KEY") {
        Some(key) => (key, None),
        None => (
            var("LITELLM_APIKEY")
                .ok_or_else(|| anyhow::anyhow!("set OPENAI_API_KEY or LITELLM_APIKEY"))?,
            Some(format!(
                "{}/v1",
                var("LITELLM_URL")
                    .ok_or_else(|| anyhow::anyhow!("LITELLM_URL is not set"))?
                    .trim_end_matches('/')
            )),
        ),
    };
    let model = var("OMNI_STT_MODEL").unwrap_or_else(|| "gpt-4o-transcribe".to_string());
    let language = var("OMNI_STT_LANGUAGE").unwrap_or_else(|| "en".to_string());
    tracing::info!(
        %model,
        %language,
        via = base_url.as_deref().unwrap_or("api.openai.com"),
        "user transcriber"
    );
    Ok(Arc::new(OpenAiStt::new(OpenAiSttConfig {
        api_key,
        model,
        base_url,
        language: Some(language),
    })))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info,ort=warn".into()))
        .init();

    let mut openai = OpenAiRealtimeConfig::from_env()?;
    if let Ok(model) = std::env::var("OPENAI_REALTIME_MODEL") {
        openai = openai.with_model(model);
    }
    // Default: the provider does not transcribe input (some proxies react to the
    // input transcript and reply on their own); we transcribe user utterances ourselves.
    let provider_transcripts = std::env::var("OPENAI_INPUT_TRANSCRIPTION").is_ok_and(|v| v == "on");
    let transcriber = if provider_transcripts {
        None
    } else {
        openai = openai.without_input_transcription();
        Some(own_transcriber()?)
    };
    if std::env::var("OPENAI_PROXY_CREATES_RESPONSES").is_ok_and(|v| v == "1") {
        openai = openai.with_proxy_creates_responses(true);
        tracing::info!("proxy owns reply creation: replies follow input transcription");
    }
    tracing::info!(endpoint = %openai.endpoint, model = %openai.model, "realtime target");

    let audio = CpalAudio::new(SAMPLE_RATE)?;
    tracing::info!(
        capture_hz = audio.source().sample_rate(),
        playback_hz = audio.sink().sample_rate(),
        "audio devices open"
    );

    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(TimeTool)];
    let config = OmniConfig {
        system_prompt: Some(
            concat!(
                "You are a voice assistant. Always reply in English, whatever language or ",
                "accent you hear. Keep every reply to one or two short sentences. Never ",
                "list what you can do. If the user only says a filler word or thanks, ",
                "answer in a few words."
            )
            .into(),
        ),
        tools_schema: Some(tool_declarations(&tools)),
        voice: std::env::var("OPENAI_VOICE").ok(),
        ..OmniConfig::default()
    };

    let cancel = CancellationToken::new();
    let shutdown = cancel.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutting down");
        shutdown.cancel();
    });
    if let Some(secs) = std::env::var("OMNI_MAX_SECONDS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        let stop = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
            tracing::info!(secs, "max session length reached; shutting down");
            stop.cancel();
        });
    }

    let (source, sink) = audio.into_parts();
    let detector = SileroDetector::new(source.sample_rate())?;
    let idle = std::env::var("OMNI_IDLE_SECONDS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(15);
    let speech_threshold = std::env::var("OMNI_VAD_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5);

    let mut gate = VoiceGate::builder()
        .provider(move || {
            Box::new(OpenAiRealtimeProvider::new(openai.clone())) as Box<dyn OmniProvider>
        })
        .detector(detector)
        .audio_source(source)
        .audio_sink(sink)
        .tools(tools)
        .config(config)
        .memory(Arc::new(SqliteMemory::new("./omni-voice.db")?))
        .conversation("local-voice", "user", "agent")
        .vad(VadConfig {
            speech_threshold,
            ..VadConfig::default()
        })
        .idle_timeout(std::time::Duration::from_secs(idle))
        .cancel_token(cancel);
    if let Some(stt) = transcriber {
        gate = gate.transcriber(stt);
    }

    tracing::info!(idle_seconds = idle, "ready — speak to open a session");
    gate.build()?.run().await?;
    Ok(())
}
