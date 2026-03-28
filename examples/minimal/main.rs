//! Minimal Mindroid agent: stdio transport + Ollama local LLM.
//!
//! Run with: `cargo run --example minimal --features ollama,stdio,static-auth`
//! Requires a running Ollama instance at http://localhost:11434

use mindroid::Runtime;
use mindroid::auth::static_id::StaticAuth;
use mindroid::pipeline::presets::ollama::ollama_pipeline;
use mindroid::transport::stdio::StdioTransport;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let mut runtime = Runtime::builder()
        .transport(StdioTransport::new())
        .pipeline(ollama_pipeline("http://localhost:11434", "llama3.2")?)
        .auth(StaticAuth::new("dev"))
        .on_message(|ctx| async move {
            if let Err(e) = ctx.process_and_respond().await {
                tracing::error!("Error: {e}");
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
