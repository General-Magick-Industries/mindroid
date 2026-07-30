//! End-user platform agent — acts on its own end-user identity, not a service
//! user. With `auth.type = "enduser"` the whole magickmind surface (Centrifugo
//! connect + subscribe, context/history, send-with-fan-out) routes to the
//! end-user API automatically: `MagickmindClient` derives it from the `Auth`
//! credential, so the same code runs a service-user (`apikey`) config against
//! `/v1/magickspaces/...` instead.
//!
//! Flow per message: fetch context → LLM → persist (fans out to participants).
//!
//! Run:
//!   cargo run --example enduser_agent --features full -- --config ./enduser-agent.toml

use std::sync::Arc;

use mindroid::llm_client::LlmClient;
use mindroid::pipeline::presets::magickmind::{
    MagickmindClient, MagickmindContext, MagickmindPersistence,
};
use mindroid::{
    ContextPreparer, GenericLlmProcessor, MindroidConfig, Pipeline, PipelineContext, PostProcessor,
    PrepareOutcome, Runtime,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,mindroid=debug")
        .init();

    let config = MindroidConfig::resolve_from_args()?;
    let respond_llm = config.llm("respond")?;

    // Auth, transport, memory, observer come from config. auth.type = "enduser"
    // makes the transport connect + subscribe as the agent's own identity.
    let builder = Runtime::from_config(config)?;
    let identity = builder.auth_arc().unwrap();
    let config = builder.config_ref().unwrap();

    let base_url = config
        .auth
        .base_url
        .as_deref()
        .unwrap_or("https://magickmind.example.com");

    // Route surface follows the configured credential.
    let kind = mindroid::credential_kind_from_config(config);
    let magickmind = Arc::new(MagickmindClient::new(base_url, identity).with_caller(kind));

    let agent_id = config.agent.agent_id.clone();
    let context_preparer = Arc::new(
        ContextPreparer::new()
            .add_provider(MagickmindContext::new(magickmind.clone()).with_self_id(agent_id)),
    );

    // Respond: LLM → post-process → persist (persist fans the reply out to every
    // other participant on the end-user route).
    let respond_pipeline = Arc::new(
        Pipeline::new()
            .add_streaming_stage(GenericLlmProcessor::new(LlmClient::new(respond_llm)?))
            .add_stage(PostProcessor)
            .add_stage(MagickmindPersistence::new(Arc::clone(&magickmind))),
    );

    let mut runtime = builder
        .on_message(move |ctx| {
            let preparer = Arc::clone(&context_preparer);
            let respond = Arc::clone(&respond_pipeline);

            async move {
                tracing::info!(
                    "Message from {}: {:?}",
                    ctx.message.sender_id,
                    ctx.message.content
                );

                let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

                // Fetch conversation context (chat history + knowledge) for the space.
                let history = match preparer.prepare(&ctx.message).await {
                    PrepareOutcome::Complete(msgs) => msgs,
                    PrepareOutcome::Degraded { messages, warnings } => {
                        for w in &warnings {
                            tracing::warn!("Context provider '{}' failed: {}", w.provider, w.error);
                        }
                        messages
                    }
                    PrepareOutcome::Failed(warnings) => {
                        for w in &warnings {
                            tracing::error!(
                                "Context provider '{}' failed: {}",
                                w.provider,
                                w.error
                            );
                        }
                        return;
                    }
                };

                pctx.reset_output();
                pctx.set_ext(mindroid::ConversationHistory(history));

                match ctx.run_with_context(&respond, &mut pctx).await {
                    Ok(_) => {}
                    Err(e) => {
                        tracing::error!("Pipeline failed: {e}");
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
