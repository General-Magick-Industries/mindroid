//! Talk to Gemini Live through the default microphone and speaker.
//!
//! Run with:
//!   GOOGLE_API_KEY=... cargo run -p mindroid-example-gemini-live --bin gemini_live
//!   (GEMINI_API_KEY works too.)
//!
//! Gemini Live connects directly to Google and transcribes both sides itself, so
//! both halves of the conversation land in memory with no separate transcriber. A
//! local VAD opens a session on the first words and closes it after
//! OMNI_IDLE_SECONDS (default 15) of silence, which is when the turns flush to
//! memory; the next words open a fresh session that seeds that history back in.
//!
//! Optional: GEMINI_LIVE_MODEL, GEMINI_VOICE, OMNI_MAX_SECONDS (auto-stop),
//! OMNI_IDLE_SECONDS, OMNI_VAD_THRESHOLD (0-1, default 0.5; lower opens more
//! readily). Ctrl-C ends the run.
//!
//! Turns persist to ./omni-gemini.db.

use std::sync::Arc;

use async_trait::async_trait;
use mindroid::memory::sqlite::SqliteMemory;
use mindroid::omni::cpal_audio::CpalAudio;
use mindroid::omni::gemini::{
    GeminiLiveConfig, GeminiLiveProvider, INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, tool_declarations,
};
use mindroid::omni::types::OmniConfig;
use mindroid::omni::{AudioSink, AudioSource, OmniProvider, SileroDetector, VoiceGate};
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info,ort=warn".into()))
        .init();

    let mut gemini = GeminiLiveConfig::from_env()?;
    if let Ok(model) = std::env::var("GEMINI_LIVE_MODEL") {
        gemini = gemini.with_model(model);
    }
    tracing::info!(model = %gemini.model, "gemini live target");

    // Capture at Gemini's input rate and play back at its output rate; the two
    // differ, so a single-rate CpalAudio::new would resample-by-accident.
    let audio = CpalAudio::new_split(INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)?;
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
        voice: std::env::var("GEMINI_VOICE").ok(),
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

    let gate = VoiceGate::builder()
        .provider(move || Box::new(GeminiLiveProvider::new(gemini.clone())) as Box<dyn OmniProvider>)
        .detector(detector)
        .audio_source(source)
        .audio_sink(sink)
        .tools(tools)
        .config(config)
        .memory(Arc::new(SqliteMemory::new("./omni-gemini.db")?))
        .conversation("local-voice", "user", "agent")
        .vad(VadConfig {
            speech_threshold,
            ..VadConfig::default()
        })
        .idle_timeout(std::time::Duration::from_secs(idle))
        .cancel_token(cancel)
        .build()?;

    tracing::info!(idle_seconds = idle, "ready — speak to open a session");
    gate.run().await?;
    Ok(())
}
