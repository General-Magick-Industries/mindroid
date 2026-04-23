//! MyHere — Layer 1 of the MyThere architecture.
//!
//! The immediate-execution mind with a Fast/Smart brain duality:
//!
//!   Per-message flow:
//!   1. Fetch context from local SQLite DB (chat history)
//!   2. Fast brain pipeline (litellm):
//!      SimpleContextBuilder(fast prompt + history)
//!        → GenericLlmProcessor(fast)   [streaming]
//!        → IsFinalExtractor            [parse JSON, set IsFinal ext]
//!        → BrainRouterGate             [halt if escalation needed]
//!        → PostProcessor + SqlitePersistence [only on fast-brain final answers]
//!   3. If halted (smart brain needed):
//!      Smart brain pipeline (BiFrost):
//!        SimpleContextBuilder(smart prompt + history)
//!        → GenericLlmProcessor(smart)  [streaming]
//!        → PostProcessor + SqlitePersistence
//!
//! Run with:
//!   cargo run -p myhere -- --config examples/myhere/myhere.toml

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;

use mindroid::llm_client::LlmClient;
use mindroid::memory::sqlite::SqliteMemory;
use mindroid::{
    ContextPreparer, ContextProvider, LlmMessage, Memory, Message, MindroidConfig, Pipeline,
    PipelineContext, PipelineStage, PostProcessor, Result, Runtime, ShellTool, OpenTool,
    SimpleContextBuilder, StreamEvent, ToolExecutorStage, ToolRegistry,
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

// ── SqliteContextProvider ────────────────────────────────────────────────────

/// Loads recent chat history from SQLite and converts it to LlmMessages.
/// Messages from `agent_id` are mapped to the `assistant` role; all others to `user`.
struct SqliteContextProvider {
    memory: Arc<SqliteMemory>,
    agent_id: String,
    limit: usize,
}

#[async_trait]
impl ContextProvider for SqliteContextProvider {
    fn name(&self) -> &str {
        "SqliteContextProvider"
    }

    async fn fetch(&self, message: &Message) -> Result<Vec<LlmMessage>> {
        let history = self
            .memory
            .get_history(&message.channel_id, self.limit)
            .await?;

        let llm_messages = history
            .into_iter()
            .map(|msg| {
                if msg.sender_id == self.agent_id {
                    LlmMessage::assistant(msg.content)
                } else {
                    LlmMessage::user(msg.content)
                }
            })
            .collect();

        Ok(llm_messages)
    }
}

// ── SqlitePersistence ────────────────────────────────────────────────────────

/// Saves both the incoming user message and the agent's response to SQLite.
struct SqlitePersistence {
    memory: Arc<SqliteMemory>,
    agent_id: String,
}

#[async_trait]
impl PipelineStage for SqlitePersistence {
    fn name(&self) -> &str {
        "SqlitePersistence"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let channel_id = &ctx.message.channel_id;
        let user_id = &ctx.message.sender_id;

        // Save user message
        self.memory
            .save_message(channel_id, user_id, &ctx.message.content, None)
            .await?;

        // Save agent response
        if let Some(response) = ctx.response.as_deref().filter(|r| !r.is_empty()) {
            self.memory
                .save_message(channel_id, &self.agent_id, response, None)
                .await?;
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
        .with_env_filter("warn")
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

    let db_path = config
        .memory
        .path
        .as_deref()
        .unwrap_or("./myhere.db");

    let max_memory_items = config
        .memory
        .options
        .get("max_memory_items")
        .and_then(|v| v.as_u64())
        .unwrap_or(20);
    // 0 means unlimited; otherwise use the configured value
    let history_limit = if max_memory_items == 0 { usize::MAX } else { max_memory_items as usize };

    let memory = Arc::new(SqliteMemory::new(db_path)?);

    let agent_id = config.agent.agent_id.clone();

    let context_preparer = Arc::new(
        ContextPreparer::new().add_provider(SqliteContextProvider {
            memory: Arc::clone(&memory),
            agent_id: agent_id.clone(),
            limit: history_limit,
        }),
    );

    let tool_registry = Arc::new(
        ToolRegistry::new()
            .register(OpenTool::default()),
            .register(ShellTool::default()),
    );

    let fast_llm = Arc::new(fast_llm);
    let smart_llm = Arc::new(smart_llm);

    let builder = Runtime::from_config(config)?;

    let mut runtime = builder
        .on_message(move |ctx| {
            let preparer = Arc::clone(&context_preparer);
            let memory = Arc::clone(&memory);
            let fast_llm = Arc::clone(&fast_llm);
            let smart_llm = Arc::clone(&smart_llm);
            let fast_persona = Arc::clone(&fast_persona);
            let smart_persona = Arc::clone(&smart_persona);
            let tool_registry = Arc::clone(&tool_registry);
            let agent_id = agent_id.clone();

            async move {
                // ── 1. Fetch local SQLite context ─────────────────────────────
                let context = match preparer.prepare(&ctx.message).await {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("Context fetch failed, continuing without history: {e}");
                        Vec::new()
                    }
                };
                let history = Arc::new(context);
                let persist = !ctx.message.channel_id.is_empty();

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
                    .add_streaming_stage(ToolExecutorStage::new(fast_client, Arc::clone(&tool_registry)))
                    .add_stage(IsFinalExtractor)
                    .add_stage(BrainRouterGate)
                    .add_stage(PostProcessor);
                if persist {
                    fast_pipeline = fast_pipeline.add_stage(SqlitePersistence {
                        memory: Arc::clone(&memory),
                        agent_id: agent_id.clone(),
                    });
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

                // ── 3. Smart brain pipeline ───────────────────────────────────
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
                    .add_streaming_stage(ToolExecutorStage::new(smart_client, Arc::clone(&tool_registry)))
                    .add_stage(PostProcessor);
                if persist {
                    smart_pipeline = smart_pipeline.add_stage(SqlitePersistence {
                        memory: Arc::clone(&memory),
                        agent_id: agent_id.clone(),
                    });
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

                let response = full_response.trim().to_string();
                if !response.is_empty() {
                    if let Err(e) = ctx.respond(&response).await {
                        tracing::error!("Failed to send smart brain response: {e}");
                    }
                }
                println!("\n");
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
