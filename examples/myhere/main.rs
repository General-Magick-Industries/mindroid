//! MyHere — Layer 1 of the MyThere architecture.
//!
//! The immediate-execution mind with a Fast/Smart brain duality:
//!
//!   Per-message flow:
//!   1. Fetch MagickMind context (chat history + knowledge)
//!   2. Fast brain pipeline (litellm):
//!      SimpleContextBuilder(fast prompt + history)
//!        → GenericLlmProcessor(fast)   [streaming]
//!        → IsFinalExtractor            [parse JSON, set IsFinal ext]
//!        → BrainRouterGate             [halt if escalation needed]
//!        → PostProcessor + Persistence [only on fast-brain final answers]
//!   3. If halted (smart brain needed):
//!      Smart brain pipeline (BiFrost):
//!        SimpleContextBuilder(smart prompt + history)
//!        → GenericLlmProcessor(smart)  [streaming]
//!        → PostProcessor + Persistence
//!
//! Run with:
//!   cargo run -p myhere -- --config examples/myhere/myhere.toml

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;

use mindroid::llm_client::LlmClient;
use mindroid::pipeline::presets::magickmind::{
    MagickmindClient, MagickmindContext, MagickmindPersistence,
};
use mindroid::{
    ContextPreparer, GenericLlmProcessor, MindroidConfig, Pipeline, PipelineContext, PipelineStage,
    PostProcessor, Result, Runtime, SimpleContextBuilder, StreamEvent,
};

// ── IsFinal extension ────────────────────────────────────────────────────────

/// `true`  = fast brain answered sufficiently, skip smart brain.
/// `false` = question needs deep reasoning, escalate to smart brain.
struct IsFinal(bool);

#[derive(Deserialize)]
struct FastBrainOutput {
    is_final: bool,
    response: String,
}

// ── IsFinalExtractor ─────────────────────────────────────────────────────────

struct IsFinalExtractor;

#[async_trait]
impl PipelineStage for IsFinalExtractor {
    fn name(&self) -> &str {
        "IsFinalExtractor"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let raw = ctx.response.as_deref().unwrap_or("{}");
        match serde_json::from_str::<FastBrainOutput>(raw) {
            Ok(parsed) => {
                ctx.set_ext(IsFinal(parsed.is_final));
                ctx.response = Some(parsed.response);
            }
            Err(_) => {
                ctx.set_ext(IsFinal(true));
            }
        }
        Ok(())
    }
}

// ── BrainRouterGate ──────────────────────────────────────────────────────────

struct BrainRouterGate;

#[async_trait]
impl PipelineStage for BrainRouterGate {
    fn name(&self) -> &str {
        "BrainRouterGate"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let is_final = ctx.get_ext::<IsFinal>().map(|f| f.0).unwrap_or(true);
        if !is_final {
            tracing::info!("BrainRouterGate: escalating to smart brain");
            ctx.halted = true;
        }
        Ok(())
    }
}

// ── Prompts ──────────────────────────────────────────────────────────────────

const FAST_BRAIN_PROMPT_DEFAULT: &str =
    "If the question needs deep thinking or reasoning, set is_final to false, \
     otherwise set it to true. \
     Always respond with valid JSON: {\"is_final\": bool, \"response\": string}.";

const SMART_BRAIN_PROMPT_DEFAULT: &str =
    "You are a deep reasoning assistant. Think carefully and thoroughly before responding. \
     Provide complete, well-considered answers.";

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("warn,myhere=info,mindroid::pipeline::context=debug")
        .init();

    let config = MindroidConfig::resolve_from_args()?;

    let fast_llm = config.llm("fast")?;
    let smart_llm = config.llm("smart")?;

    let fast_persona: Arc<str> = config
        .models
        .get("fast")
        .and_then(|m| m.options.get("persona"))
        .and_then(|v| v.as_str())
        .unwrap_or(FAST_BRAIN_PROMPT_DEFAULT)
        .into();

    let smart_persona: Arc<str> = config
        .models
        .get("smart")
        .and_then(|m| m.options.get("persona"))
        .and_then(|v| v.as_str())
        .unwrap_or(SMART_BRAIN_PROMPT_DEFAULT)
        .into();

    let builder = Runtime::from_config(config)?;

    let identity = builder.auth_arc().unwrap();
    let config = builder.config_ref().unwrap();

    let magickmind_url = config
        .auth
        .base_url
        .as_deref()
        .unwrap_or("https://dev-magickmind.magickmind.ai");

    let mut magickmind_client = MagickmindClient::new(magickmind_url, identity);
    if let Some(key) = &config.auth.api_key {
        magickmind_client = magickmind_client.with_api_key(key);
    }
    let magickmind = Arc::new(magickmind_client);

    let agent_id = config.agent.agent_id.clone();
    let context_preparer = Arc::new(
        ContextPreparer::new()
            .add_provider(MagickmindContext::new(magickmind.clone()).with_self_id(agent_id)),
    );

    // TODO: persona stage belongs to MyThere (Layer 2), not MyHere.
    // let persona_stage = builder.build_persona_stage().await?;

    let fast_llm = Arc::new(fast_llm);
    let smart_llm = Arc::new(smart_llm);

    let mut runtime = builder
        .on_message(move |ctx| {
            let preparer = Arc::clone(&context_preparer);
            let magickmind = Arc::clone(&magickmind);
            let fast_llm = Arc::clone(&fast_llm);
            let smart_llm = Arc::clone(&smart_llm);
            let fast_persona = Arc::clone(&fast_persona);
            let smart_persona = Arc::clone(&smart_persona);

            async move {
                // ── 1. Fetch MagickMind context ───────────────────────────────
                let context = match preparer.prepare(&ctx.message).await {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("Context fetch failed, continuing without history: {e}");
                        Vec::new()
                    }
                };
                let history = Arc::new(context);
                let persist = !ctx.message.channel_id.is_empty() && ctx.message.channel_id != "stdio";

                // ── 2. Fast brain pipeline ────────────────────────────────────
                let fast_client = match LlmClient::new((*fast_llm).clone()) {
                    Ok(c) => c,
                    Err(e) => { tracing::error!("Fast brain LLM init failed: {e}"); return; }
                };

                let mut fast_pipeline = Pipeline::new()
                    .add_stage(SimpleContextBuilder::with_prompt_and_history(
                        fast_persona.as_ref(),
                        history.clone(),
                    ))
                    .add_streaming_stage(GenericLlmProcessor::new(fast_client))
                    .add_stage(IsFinalExtractor)
                    .add_stage(BrainRouterGate)
                    .add_stage(PostProcessor);
                if persist {
                    fast_pipeline = fast_pipeline.add_stage(MagickmindPersistence::new(Arc::clone(&magickmind)));
                }

                let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

                let (needs_smart, fast_response) = match ctx.run_with_context(&fast_pipeline, &mut pctx).await {
                    Ok(resp) => {
                        let needs_smart = pctx.get_ext::<IsFinal>().map(|f| !f.0).unwrap_or(false);
                        (needs_smart, resp.unwrap_or_default())
                    }
                    Err(e) => { tracing::error!("Fast brain pipeline failed: {e}"); return; }
                };

                if !needs_smart {
                    let response = fast_response.trim().to_string();
                    if !response.is_empty() {
                        println!("\nMyHere [Fast]: {response}\n");
                        if let Err(e) = ctx.respond(&response).await {
                            tracing::error!("Failed to send fast brain response: {e}");
                        }
                    }
                    return;
                }

                // ── 3. Smart brain pipeline (BiFrost) ─────────────────────────
                tracing::info!("Fast brain escalated — running smart brain");

                let smart_client = match LlmClient::new((*smart_llm).clone()) {
                    Ok(c) => c,
                    Err(e) => { tracing::error!("Smart brain LLM init failed: {e}"); return; }
                };

                let mut smart_pipeline = Pipeline::new()
                    .add_stage(SimpleContextBuilder::with_prompt_and_history(
                        smart_persona.as_ref(),
                        history,
                    ))
                    .add_streaming_stage(GenericLlmProcessor::new(smart_client))
                    .add_stage(PostProcessor);
                if persist {
                    smart_pipeline = smart_pipeline.add_stage(MagickmindPersistence::new(Arc::clone(&magickmind)));
                }

                pctx.reset_output();

                let mut stream = ctx.run_streaming_with_context(&smart_pipeline, &mut pctx);
                let mut full_response = String::new();

                print!("\nMyHere [Smart]: ");
                while let Some(event) = stream.next().await {
                    match &event {
                        StreamEvent::Chunk { content } => {
                            print!("{content}");
                            let _ = std::io::Write::flush(&mut std::io::stdout());
                            full_response.push_str(content);
                        }
                        StreamEvent::Complete { content, .. } => {
                            if !content.is_empty() { full_response = content.clone(); }
                        }
                        StreamEvent::Error { message } => {
                            tracing::error!("Smart brain stream error: {message}");
                        }
                        _ => {}
                    }
                }
                println!();

                let response = full_response.trim().to_string();
                if !response.is_empty() {
                    if let Err(e) = ctx.respond(&response).await {
                        tracing::error!("Failed to send smart brain response: {e}");
                    }
                }
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
