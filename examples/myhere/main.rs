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
//!        PersonaWithHistory(MagickMind persona + history)
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
    ContextPreparer, GenericLlmProcessor, LlmMessage, MindroidConfig, PersonaContextBuilder,
    Pipeline, PipelineContext, PipelineStage, PostProcessor, Result, Runtime,
    SimpleContextBuilder, StreamEvent,
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

// ── PersonaWithHistory ───────────────────────────────────────────────────────

/// Runs the shared PersonaContextBuilder then injects per-request history
/// after the system messages, before the user message.
struct PersonaWithHistory {
    persona: Arc<PersonaContextBuilder>,
    history: Arc<Vec<LlmMessage>>,
}

#[async_trait]
impl PipelineStage for PersonaWithHistory {
    fn name(&self) -> &str {
        "PersonaWithHistory"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        self.persona.process(ctx).await?;

        if !self.history.is_empty() {
            let system_msgs: Vec<_> = ctx.llm_messages.drain(..).collect();
            let mut new_messages = Vec::with_capacity(system_msgs.len() + self.history.len());
            let mut rest_start = 0;
            for (i, msg) in system_msgs.iter().enumerate() {
                if msg.role == "system".into() {
                    new_messages.push(msg.clone());
                    rest_start = i + 1;
                } else {
                    break;
                }
            }
            new_messages.extend(self.history.iter().cloned());
            new_messages.extend(system_msgs[rest_start..].iter().cloned());
            ctx.llm_messages = new_messages;
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
        .with_env_filter("info,mindroid=debug,myhere=debug")
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

    // Build persona stage once — fetches persona schema from MagickMind at init.
    // Falls back to SimpleContextBuilder with the smart persona prompt if not configured.
    let persona_stage = builder.build_persona_stage().await?;

    let fast_llm = Arc::new(fast_llm);
    let smart_llm = Arc::new(smart_llm);
    let persona_stage = persona_stage.map(Arc::new);

    // Capture smart_persona fallback only if persona stage is absent
    let smart_persona_fallback: Option<Arc<str>> = if persona_stage.is_none() {
        Some(
            config
                .models
                .get("smart")
                .and_then(|m| m.options.get("persona"))
                .and_then(|v| v.as_str())
                .unwrap_or(SMART_BRAIN_PROMPT_DEFAULT)
                .into(),
        )
    } else {
        None
    };

    let mut runtime = builder
        .on_message(move |ctx| {
            let preparer = Arc::clone(&context_preparer);
            let magickmind = Arc::clone(&magickmind);
            let fast_llm = Arc::clone(&fast_llm);
            let smart_llm = Arc::clone(&smart_llm);
            let fast_persona = Arc::clone(&fast_persona);
            let persona_stage = persona_stage.clone();
            let smart_persona_fallback = smart_persona_fallback.clone();

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

                // ── 2. Fast brain pipeline ────────────────────────────────────
                let fast_client = match LlmClient::new((*fast_llm).clone()) {
                    Ok(c) => c,
                    Err(e) => { tracing::error!("Fast brain LLM init failed: {e}"); return; }
                };

                let fast_pipeline = Pipeline::new()
                    .add_stage(SimpleContextBuilder::with_prompt_and_history(
                        fast_persona.as_ref(),
                        history.clone(),
                    ))
                    .add_streaming_stage(GenericLlmProcessor::new(fast_client))
                    .add_stage(IsFinalExtractor)
                    .add_stage(BrainRouterGate)
                    .add_stage(PostProcessor)
                    .add_stage(MagickmindPersistence::new(Arc::clone(&magickmind)));

                let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

                let needs_smart = match ctx.run_with_context(&fast_pipeline, &mut pctx).await {
                    Ok(_) => pctx.get_ext::<IsFinal>().map(|f| !f.0).unwrap_or(false),
                    Err(e) => { tracing::error!("Fast brain pipeline failed: {e}"); return; }
                };

                if !needs_smart {
                    let response = pctx.response.as_deref().unwrap_or("").trim().to_string();
                    if !response.is_empty() {
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

                let smart_pipeline = if let Some(ref persona) = persona_stage {
                    Pipeline::new()
                        .add_stage(PersonaWithHistory {
                            persona: Arc::clone(persona),
                            history,
                        })
                        .add_streaming_stage(GenericLlmProcessor::new(smart_client))
                        .add_stage(PostProcessor)
                        .add_stage(MagickmindPersistence::new(Arc::clone(&magickmind)))
                } else {
                    let prompt = smart_persona_fallback.as_deref().unwrap_or(SMART_BRAIN_PROMPT_DEFAULT);
                    Pipeline::new()
                        .add_stage(SimpleContextBuilder::with_prompt_and_history(prompt, history))
                        .add_streaming_stage(GenericLlmProcessor::new(smart_client))
                        .add_stage(PostProcessor)
                        .add_stage(MagickmindPersistence::new(Arc::clone(&magickmind)))
                };

                pctx.reset_output();

                let mut stream = ctx.run_streaming_with_context(&smart_pipeline, &mut pctx);
                let mut full_response = String::new();

                while let Some(event) = stream.next().await {
                    match &event {
                        StreamEvent::Chunk { content } => full_response.push_str(content),
                        StreamEvent::Complete { content, .. } => {
                            if !content.is_empty() { full_response = content.clone(); }
                        }
                        StreamEvent::Error { message } => {
                            tracing::error!("Smart brain stream error: {message}");
                        }
                        _ => {}
                    }
                }

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
