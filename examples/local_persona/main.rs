//! Local persona agent: load a persona from a local markdown file and run
//! a stdio-based agent with Ollama as the LLM backend.
//!
//! This example demonstrates the simplest path to a persona-driven agent
//! that works entirely offline — no MagickMind account required.
//!
//! # How it works
//!
//! 1. Config is loaded from `examples/local_persona/config.toml`.
//! 2. `Runtime::from_config` builds auth, transport (stdio), and memory from
//!    the config and also resolves the `[persona]` section into a
//!    `LocalPersonaProvider`.
//! 3. `builder.build_persona_stage()` reads the persona.md file from disk and
//!    returns a `PersonaContextBuilder` ready to use as a pipeline stage.
//! 4. A simple pipeline is assembled:
//!    `PersonaContextBuilder` → `GenericLlmProcessor` (Ollama) → `PostProcessor`
//! 5. The runtime reads messages from stdin and writes responses to stdout.
//!
//! # Running
//!
//! Make sure Ollama is running locally, then:
//!
//! ```sh
//! cargo run -p mindroid-example-local-persona --bin local_persona -- \
//!     --config examples/local_persona/config.toml
//! ```
//!
//! Type a message and press Enter. The agent responds as "Guide".

use std::sync::Arc;

use mindroid::llm_client::{AuthStyle, LlmClient, LlmClientConfig};
use mindroid::{
    GenericLlmProcessor, MindroidConfig, Pipeline, PipelineContext, PostProcessor, Runtime,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,mindroid=debug")
        .init();

    let config = MindroidConfig::resolve_from_args()?;

    // Resolve the LLM client from config's [models.respond] + [providers.*].
    // Falls back to a local Ollama endpoint if no model config is present.
    let llm_config = match config.llm("respond") {
        Ok(c) => c,
        Err(_) => {
            tracing::warn!("No [models.respond] found in config — falling back to Ollama llama3.2");
            let mut c = LlmClientConfig::new("http://localhost:11434/v1".to_string());
            c.auth_style = AuthStyle::None;
            c.default_model = Some("llama3.2".to_string());
            c
        }
    };
    let llm_client = LlmClient::new(llm_config)?;

    // Build runtime components from config (auth, transport, memory, persona provider).
    let builder = Runtime::from_config(config)?;

    // Build the persona stage — reads the persona.md file from disk.
    let persona_stage = builder
        .build_persona_stage()
        .await?
        .expect("local_persona requires a [persona] section with type = \"local\" in config");

    // Assemble the pipeline once: persona system prompt → LLM → post-process output.
    // The pipeline is shared across all incoming messages via Arc.
    let pipeline = Arc::new(
        Pipeline::new()
            .add_stage(persona_stage)
            .add_streaming_stage(GenericLlmProcessor::new(llm_client))
            .add_stage(PostProcessor),
    );

    let mut runtime = builder
        .on_message(move |ctx| {
            let pipeline = Arc::clone(&pipeline);

            async move {
                let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

                match ctx.run_with_context(&pipeline, &mut pctx).await {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        tracing::info!("No response generated");
                        return;
                    }
                    Err(e) => {
                        tracing::error!("Pipeline error: {e}");
                        return;
                    }
                }

                let response = pctx.response.as_deref().unwrap_or("").trim().to_string();
                if !response.is_empty()
                    && let Err(e) = ctx.respond(&response).await
                {
                    tracing::error!("Failed to send response: {e}");
                }
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
