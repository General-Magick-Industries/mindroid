//! Full Magick Mind integration: Centrifugo transport + MagickMind pipeline + MagickMind memory.
//!
//! Run with: `cargo run --example magickmind --features full`
//!
//! Requires environment variables:
//!   MINDROID_EMAIL     — MagickMind login email
//!   MINDROID_PASSWORD  — MagickMind login password
//!   MINDROID_BASE_URL  — Base URL for MagickMind/Cortex (e.g. https://api.magickmind.io)
//!   MINDROID_API_KEY   — Cortex API key
//!   MINDROID_AGENT_ID  — Agent ID for channel subscription

use std::sync::Arc;

use mindroid::auth::apikey::ApiKeyAuth;
use mindroid::memory::magickmind::MagickmindMemory;
use mindroid::observer::log::LogObserver;
use mindroid::pipeline::presets::magickmind::magickmind_pipeline;
use mindroid::transport::centrifugo::CentrifugoTransport;
use mindroid::{MindroidConfig, Runtime};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,mindroid=debug")
        .init();

    let config = MindroidConfig::resolve_from_args()?;

    let base_url = config
        .pipeline
        .base_url
        .as_deref()
        .unwrap_or("https://dev-magickmind.magickmind.ai");

    let api_key = config.auth.api_key.as_deref().unwrap_or("");

    let email = config.auth.email.as_deref().unwrap_or("");
    let password = config.auth.password.as_deref().unwrap_or("");

    let identity = Arc::new(ApiKeyAuth::new(base_url, email, password));

    let ws_url = config
        .transport
        .url
        .as_deref()
        .unwrap_or("wss://dev-centrifugo.magickmind.ai/connection/websocket");

    let agent_id = &config.agent.agent_id;

    let transport = CentrifugoTransport::new(ws_url, agent_id, identity.clone());
    let pipeline = magickmind_pipeline(
        identity.clone(),
        base_url,
        api_key,
        config.agent.compute_power,
    )?;
    let memory = MagickmindMemory::new(base_url, identity.clone());

    let mut runtime = Runtime::builder()
        .config(config)
        .transport(transport)
        .pipeline(pipeline)
        .auth_shared(identity)
        .memory(memory)
        .observer(LogObserver::new())
        .on_message(|ctx| async move {
            if let Err(e) = ctx.process_and_respond().await {
                tracing::error!("Error processing message: {e}");
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
