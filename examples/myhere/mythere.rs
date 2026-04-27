//! MyThere — Layer 2 architecture with MyHere as an internal stage.
//!
//! This demonstrates MyHere operating as a PipelineStage within MyThere's
//! larger pipeline. MyThere enriches context and orchestrates the flow.
//!
//! Run with:
//!   cargo run -p myhere --bin mythere -- --config examples/myhere/myhere.toml

#[allow(dead_code)]
mod myhere;

use std::io::Write;
use std::sync::Arc;

use async_trait::async_trait;
use mindroid::pipeline::presets::magickmind::{
    MagickmindClient, MagickmindContext, MagickmindPersistence,
};
use mindroid::{
    ContextPreparer, MindroidConfig, MessageContext, Pipeline, PipelineContext, PipelineStage,
    Result, Runtime, ShellTool
};

use myhere::{create_tool_registry, MyHereStage, PersistenceBackend};

// ── MyThere context enrichment stage ─────────────────────────────────────

/// Enriches the context before passing to MyHere.
/// In a real system, this might fetch data from external APIs, databases, etc.
pub struct MyThereContextEnricher;

#[async_trait]
impl PipelineStage for MyThereContextEnricher {
    fn name(&self) -> &str {
        "MyThereContextEnricher"
    }

    async fn process(&self, _ctx: &mut PipelineContext) -> Result<()> {
        tracing::info!("MyThere: enriching context for MyHere");
        // In a real implementation, enrich the context here.
        // For now, just log it as an example.
        Ok(())
    }
}

// ── MyThere post-processing stage ────────────────────────────────────────

/// Processes MyHere's response before returning to user.
pub struct MyTherePostProcessor;

#[async_trait]
impl PipelineStage for MyTherePostProcessor {
    fn name(&self) -> &str {
        "MyTherePostProcessor"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        tracing::info!(
            "MyThere: processing response from MyHere: {:?}",
            ctx.response.as_deref().unwrap_or("(none)")
        );
        // MyHere's response is not final — MyThere can further process it here.
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = MindroidConfig::resolve_from_args()?;

    let log_level = config.observer.level.as_deref().unwrap_or("info");

    tracing_subscriber::fmt().with_env_filter(log_level).init();

    let fast_llm = config.llm("fast")?;
    let smart_llm = config.llm("smart")?;

    let fast_persona: Arc<str> = config
        .models
        .get("fast")
        .and_then(|m| m.options.get("persona"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("fast model persona is required in config"))?
        .into();

    let smart_persona: Arc<str> = config
        .models
        .get("smart")
        .and_then(|m| m.options.get("persona"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("smart model persona is required in config"))?
        .into();

    let agent_id = if config.agent.agent_id.trim().is_empty() {
        config.agent.name.clone()
    } else {
        config.agent.agent_id.clone()
    };

    let builder = Runtime::from_config(config.clone())?;
    let builder_auth = builder.auth_arc();
    let magickmind_url = config
        .auth
        .base_url
        .as_deref()
        .unwrap_or("https://dev-magickmind.magickmind.ai");

    let mut magickmind_client = MagickmindClient::new(
        magickmind_url,
        builder_auth.ok_or_else(|| anyhow::anyhow!("auth not configured"))?,
    );
    if let Some(key) = &config.auth.api_key {
        magickmind_client = magickmind_client.with_api_key(key);
    }
    let magickmind = Arc::new(magickmind_client);

    let context_preparer = Arc::new(
        ContextPreparer::new()
            .add_provider(MagickmindContext::new(magickmind.clone()).with_self_id(agent_id.clone()))
    );

    let tool_registry = Arc::new(
        create_tool_registry()
            .register(ShellTool::default())
    );

    println!("MyThere is running with MyHere as an internal stage.");
    println!("\x1b[90mType your messages below:\x1b[0m");

    // Build MyThere's pipeline with MyHere as a stage
    let message_handler = move |ctx: MessageContext| {
        let context_preparer = Arc::clone(&context_preparer);
        let memory = Arc::clone(&magickmind);
        let fast_llm = fast_llm.clone();
        let smart_llm = smart_llm.clone();
        let fast_persona = Arc::clone(&fast_persona);
        let smart_persona = Arc::clone(&smart_persona);
        let tool_registry = Arc::clone(&tool_registry);
        let agent_id = agent_id.clone();

        Box::pin(async move {
            let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

            // Build MyThere's pipeline:
            // 1. Enrich context (MyThere-specific)
            // 2. Run MyHere's dual-brain logic
            // 3. Post-process the result
            // 4. Persist response to MagickMind
            let pipeline = Pipeline::new()
                .add_stage(MyThereContextEnricher)
                .add_stage(MyHereStage::new(
                    context_preparer,
                    PersistenceBackend::magickmind(memory.clone()),
                    fast_llm,
                    smart_llm,
                    fast_persona,
                    smart_persona,
                    tool_registry,
                    agent_id,
                ))
                .add_stage(MyTherePostProcessor)
                .add_stage(MagickmindPersistence::new(memory));

            if let Err(e) = pipeline.run(&mut pctx).await {
                tracing::error!("MyThere pipeline failed: {e}");
                return;
            }

            if let Some(response) = pctx.response {
                print!("\nMyThere: {response}\n\n");
                std::io::stdout().flush().ok();
                if let Err(e) = ctx.respond(&response).await {
                    tracing::error!("Failed to send response: {e}");
                }
            }

            println!("\x1b[90mType your messages below:\x1b[0m");
        })
    };

    let mut runtime = builder.on_message(message_handler).build()?;

    runtime.run().await?;
    Ok(())
}
