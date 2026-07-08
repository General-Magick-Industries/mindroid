//! MagickMind CLI: Centrifugo transport + LiteLLM inference + MagickMind memory.
//!
//! A lightweight agent that connects to the MagickMind WebSocket channel,
//! routes inference through a LiteLLM proxy, and uses MagickMind for
//! conversation context and persistence.
//!
//! Flow per message:
//!   1. ContextPreparer       — fetch chat history + knowledge from MagickMind
//!   2. SimpleContextBuilder  — assemble LLM messages (history + user)
//!   3. GenericLlmProcessor   — streaming inference via LiteLLM
//!   4. PostProcessor         — clean up response text
//!   5. MagickmindPersistence — save response back to MagickMind
//!
//! Run with:
//!   cargo run --example magickmind_cli --features full -- --config examples/magickmind_cli/config.toml

use std::sync::Arc;

use mindroid::llm_client::LlmClient;
use mindroid::pipeline::presets::magickmind::{
    MagickmindClient, MagickmindContext, MagickmindPersistence,
};
use mindroid::{
    ContextPreparer, GenericLlmProcessor, MindroidConfig, Pipeline, PipelineContext, PostProcessor,
    Runtime, SimpleContextBuilder,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("warn,mindroid=info")
        .init();

    let config = MindroidConfig::resolve_from_args()?;

    // Resolve LiteLLM client config from [providers.litellm] + [models.main]
    let llm_config = config.llm("main")?;

    // Auto-build transport (centrifugo), auth (apikey), memory (magickmind), observer from config
    let builder = Runtime::from_config(config)?;

    let identity = builder.auth_arc().unwrap();
    let config = builder.config_ref().unwrap();

    let magickmind_url = config
        .auth
        .base_url
        .as_deref()
        .unwrap_or("https://dev-magickmind.magickmind.ai");

    let mut magickmind_client = MagickmindClient::new(magickmind_url, identity.clone());
    if let Some(key) = &config.auth.api_key {
        magickmind_client = magickmind_client.with_api_key(key);
    }
    let magickmind = Arc::new(magickmind_client);

    let agent_id = config.agent.agent_id.clone();

    // Context preparer: fetch chat history and knowledge from MagickMind
    let context_preparer = Arc::new(
        ContextPreparer::new()
            .add_provider(MagickmindContext::new(magickmind.clone()).with_self_id(agent_id)),
    );

    let llm_config = Arc::new(llm_config);

    let mut runtime = builder
        .on_message(move |ctx| {
            let preparer = Arc::clone(&context_preparer);
            let magickmind = Arc::clone(&magickmind);
            let llm_config = Arc::clone(&llm_config);

            async move {
                // Step 1: fetch conversation context from MagickMind
                let context = preparer.prepare(&ctx.message).await.into_messages();

                // Step 2: build LLM client for this request
                let llm_client = match LlmClient::new((*llm_config).clone()) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!("LLM client error: {e}");
                        return;
                    }
                };

                // Step 3: assemble pipeline with system prompt + MagickMind context
                let system_prompt = "\
You are the MagickMind Support Agent — a friendly, knowledgeable assistant \
for the MagickMind platform.\n\n\
Your responsibilities:\n\
- Help users integrate MagickMind APIs (MagickMind, Cortex, Centrifugo, Pelican, Corpus).\n\
- Explain MagickMind features: magickspaces, agents, personas, memory, skills, and tools.\n\
- Answer questions about configuration, authentication, deployment, and troubleshooting.\n\
- Guide developers through SDK setup, pipeline construction, and transport wiring.\n\
- Provide clear code examples and step-by-step instructions when relevant.\n\n\
Guidelines:\n\
- Be concise and direct. Lead with the answer, then explain if needed.\n\
- If you don't know something, say so honestly rather than guessing.\n\
- When a question is ambiguous, ask a clarifying question before answering.\n\
- Reference official MagickMind documentation and endpoints where applicable.";

                let pipeline = Pipeline::new()
                    .add_stage(SimpleContextBuilder::with_prompt_and_history(
                        system_prompt,
                        Arc::new(context),
                    ))
                    .add_streaming_stage(GenericLlmProcessor::new(llm_client))
                    .add_stage(PostProcessor)
                    .add_stage(MagickmindPersistence::new(Arc::clone(&magickmind)));

                let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

                match ctx.run_with_context(&pipeline, &mut pctx).await {
                    Ok(None) => {
                        tracing::info!("No response generated");
                        return;
                    }
                    Ok(Some(_)) => {}
                    Err(e) => {
                        tracing::error!("Pipeline error: {e}");
                        return;
                    }
                }

                let response = pctx.response.as_deref().unwrap_or("").trim().to_string();
                if !response.is_empty()
                    && let Err(e) = ctx.respond(&response).await
                {
                    tracing::error!("Send error: {e}");
                }
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
