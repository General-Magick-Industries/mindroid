//! Trip planning agent: multi-pipeline workflow on the MagickMind stack.
//!
//! Demonstrates running multiple pipelines per message to compose a multi-step
//! LLM workflow using named providers and models from the config file:
//!
//!   1. Context preparation — fetch MagickMind context once, share across pipelines
//!   2. Engagement tracking — prevent feedback loops with other agents
//!   3. Relevance gate — lightweight Ollama call decides if the agent should engage
//!      - If not relevant → pipeline halts, agent stays silent.
//!      - Otherwise → continue to full response.
//!   4. Full response — streaming Cortex call with persistence
//!
//! Run with:
//!   cargo run --example cot_trip_agent --features full -- --config examples/cot_trip_agent/config.toml

use std::sync::Arc;

use futures::StreamExt;
use mindroid::llm_client::LlmClient;
use mindroid::pipeline::presets::magickmind::{
    MagickmindClient, MagickmindContext, MagickmindPersistence,
};
use mindroid::{
    ContextPreparer, GenericLlmProcessor, MindroidConfig, Pipeline, PipelineContext, PostProcessor,
    PrepareOutcome, RelevanceGate, Runtime, SimpleContextBuilder, StreamEvent,
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,mindroid=debug")
        .init();

    let config = MindroidConfig::resolve_from_args()?;

    // Extract LLM configs before from_config consumes the config
    let gate_llm = config.llm("gate")?;
    let _ack_llm = config.llm("ack")?;
    let respond_llm = config.llm("respond")?;

    // Auto-build identity, transport, memory, observer from config
    let builder = Runtime::from_config(config)?;

    // MagickmindClient needs the shared identity for persistence and context
    let identity = builder.auth_arc().unwrap();
    let config = builder.config_ref().unwrap();
    let magickmind_url = config
        .auth
        .base_url
        .as_deref()
        .unwrap_or("https://dev-magickmind.magickmind.ai");
    let mut magickmind_client = MagickmindClient::new(magickmind_url, identity);
    if let Some(api_key) = &config.auth.api_key {
        magickmind_client = magickmind_client.with_api_key(api_key);
    }
    let magickmind = Arc::new(magickmind_client);

    // -- Context preparer: fetch once, share across pipelines ----------------

    let agent_id = config.agent.agent_id.clone();
    let context_preparer = Arc::new(
        ContextPreparer::new()
            .add_provider(MagickmindContext::new(magickmind.clone()).with_self_id(agent_id)),
    );

    let respond_llm = Arc::new(respond_llm);

    // -- Wire up the runtime with a custom handler -------------------------

    let mut runtime = builder
        .on_message(move |ctx| {
            let preparer = Arc::clone(&context_preparer);
            let magickmind = Arc::clone(&magickmind);
            let gate_llm = gate_llm.clone();
            let respond_llm = Arc::clone(&respond_llm);

            async move {
                // Fetch context once from MagickMind
                let history = match preparer.prepare(&ctx.message).await {
                    PrepareOutcome::Complete(msgs) => Arc::new(msgs),
                    PrepareOutcome::Degraded { messages, warnings } => {
                        for w in &warnings {
                            tracing::warn!("Context provider '{}' failed: {}", w.provider, w.error);
                        }
                        Arc::new(messages)
                    }
                    PrepareOutcome::Failed(warnings) => {
                        for w in &warnings {
                            tracing::error!(
                                "Context provider '{}' failed: {}",
                                w.provider,
                                w.error
                            );
                        }
                        tracing::error!("All context providers failed — aborting");
                        return;
                    }
                };

                // Build pipelines with history injected at construction time
                let gate = match RelevanceGate::from_config(
                    "trip planning — destinations, itineraries, visas, passports, \
                     hotels, flights, activities, transport, or travel logistics",
                    gate_llm,
                ) {
                    Ok(g) => g,
                    Err(e) => {
                        tracing::error!("RelevanceGate init failed: {e}");
                        return;
                    }
                };
                let gate = gate
                    .instructions(
                        "You handle trip planning — destinations, itineraries, visas, transport, \
                     accommodation, activities, and travel preferences (like 'affordable', \
                     'mid-range', 'luxury'). \
                     \n\nIMPORTANT: Look at the conversation context. If another assistant is \
                     currently handling the conversation (e.g. asking the user follow-up questions \
                     about budgets, costs, or calculations), do NOT jump in — respond false. \
                     Only respond true if the latest USER message is directed at trip logistics \
                     and no other assistant is actively handling it.",
                    )
                    .with_history(history.clone());

                let classify_pipeline = Pipeline::new().add_stage(gate);

                let respond_pipeline = Pipeline::new()
                    .add_stage(SimpleContextBuilder::with_history(history.clone()))
                    .add_streaming_stage(match LlmClient::new((*respond_llm).clone()) {
                        Ok(c) => GenericLlmProcessor::new(c),
                        Err(e) => {
                            tracing::error!("LlmClient init failed: {e}");
                            return;
                        }
                    })
                    .add_stage(PostProcessor)
                    .add_stage(MagickmindPersistence::new(magickmind));

                let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

                // Relevance gate (invisible — halts if not trip-related)
                match ctx.run_with_context(&classify_pipeline, &mut pctx).await {
                    Ok(Some(_)) => {
                        tracing::info!("Gate passed — message is relevant");
                    }
                    Ok(None) => {
                        tracing::info!("Message not relevant — staying silent");
                        return;
                    }
                    Err(e) => {
                        tracing::error!("Gate pipeline failed: {e}");
                        return;
                    }
                };
                pctx.reset_output();

                // Full response (streamed, with persistence)
                tracing::info!("Generating full response...");
                let mut stream = ctx.run_streaming_with_context(&respond_pipeline, &mut pctx);
                let mut full_response = String::new();
                while let Some(event) = stream.next().await {
                    match &event {
                        StreamEvent::Chunk { content } => {
                            full_response.push_str(content);
                        }
                        StreamEvent::Complete { content, .. } if !content.is_empty() => {
                            full_response = content.clone();
                        }
                        StreamEvent::Error { message } => {
                            tracing::error!("Stream error: {message}");
                        }
                        _ => {}
                    }
                }

                let response = full_response.trim().to_string();
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
