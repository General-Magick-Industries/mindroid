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

use std::io::Write;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;

use mindroid::llm_client::LlmClient;
use mindroid::memory::sqlite::SqliteMemory;
use mindroid::{
    ContextPreparer, ContextProvider, LlmMessage, Memory, Message, MindroidConfig, OpenTool,
    Pipeline, PipelineContext, PipelineStage, PostProcessor, Result, Runtime, ShellTool,
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
                let trimmed = raw.trim();
                if trimmed.eq_ignore_ascii_case("false") {
                    ctx.set_ext(IsFinal(false));
                    ctx.response = Some("Let me hand this to my smart brain for a minute.".into());
                } else if trimmed.eq_ignore_ascii_case("true") {
                    ctx.set_ext(IsFinal(true));
                    ctx.response = None;
                } else {
                    ctx.set_ext(IsFinal(true));
                }
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
        let needs_smart = ctx.get_ext::<IsFinal>().map(|f| !f.0).unwrap_or(true);
        if needs_smart {
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
        let channel_id = if message.channel_id.is_empty() {
            "stdio".to_string()
        } else {
            message.channel_id.clone()
        };

        let history = self
            .memory
            .get_history(&channel_id, self.limit)
            .await?;

        let llm_messages = history
            .into_iter()
            .map(|msg| {
                if msg.sender_id == self.agent_id || msg.sender_id.is_empty() {
                    LlmMessage::assistant(msg.content)
                } else {
                    LlmMessage::user(msg.content)
                }
            })
            .collect();

        Ok(llm_messages)
    }
}

// ── Animation helper ────────────────────────────────────────────────────────

/// Spawns an animated loading indicator that cycles through `.`, `..`, `...`
fn spawn_loading_animation(initial_label: &str) -> tokio::task::JoinHandle<()> {
    let label = initial_label.to_string();
    tokio::spawn(async move {
        let mut stage = 0;
        loop {
            let dots = match stage {
                0 => "|",
                1 => "/",
                2 => "—",
                3 => "\\",
                _ => "*",
            };
            print!("\r{}\x1b[90m{}\x1b[0m\x1b[K", label, dots);
            let _ = std::io::stdout().flush();

            stage = (stage + 1) % 4;
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }
    })
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
        let channel_id = if ctx.message.channel_id.is_empty() {
            "stdio".to_string()
        } else {
            ctx.message.channel_id.clone()
        };
        let user_id = &ctx.message.sender_id;

        // Save user message
        self.memory
            .save_message(&channel_id, user_id, &ctx.message.content, None)
            .await?;

        // Save agent response
        if let Some(response) = ctx.response.as_deref().filter(|r| !r.is_empty()) {
            self.memory
                .save_message(&channel_id, &self.agent_id, response, None)
                .await?;
        }

        Ok(())
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

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

    let db_path = config.memory.path.as_deref().unwrap_or("./myhere.db");

    let max_memory_items = config
        .memory
        .options
        .get("max_memory_items")
        .and_then(|v| v.as_u64())
        .unwrap_or(20);
    // 0 means unlimited; otherwise use the configured value
    let history_limit = if max_memory_items == 0 {
        usize::MAX
    } else {
        max_memory_items as usize
    };

    let memory = Arc::new(SqliteMemory::new(db_path)?);

    let agent_id = if config.agent.agent_id.trim().is_empty() {
        config.agent.name.clone()
    } else {
        config.agent.agent_id.clone()
    };

    let context_preparer = Arc::new(ContextPreparer::new().add_provider(SqliteContextProvider {
        memory: Arc::clone(&memory),
        agent_id: agent_id.clone(),
        limit: history_limit,
    }));

    let tool_registry = Arc::new(
        ToolRegistry::new()
            .register(OpenTool::default())
            .register(ShellTool::default()),
    );

    let fast_llm = Arc::new(fast_llm);
    let smart_llm = Arc::new(smart_llm);

    let builder = Runtime::from_config(config)?;

    println!("MyHere is running! This agent has a Fast brain for quick answers and a Smart brain for complex questions.");
    println!("\x1b[90mType your messages below:\x1b[0m");

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
                let persist = true;

                // ── 2. Fast brain pipeline ────────────────────────────────────
                let fast_client = match LlmClient::new((*fast_llm).clone()) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!("Fast brain LLM init failed: {e}");
                        return;
                    }
                };

                let mut fast_pipeline = Pipeline::new()
                    .add_stage(SimpleContextBuilder::with_prompt_and_history(
                        fast_persona.as_ref(),
                        history.clone(),
                    ))
                    .add_streaming_stage(ToolExecutorStage::new(
                        fast_client,
                        Arc::clone(&tool_registry),
                    ))
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

                print!("\n\x1b[90mMyHere [Fast]: \x1b[0m");
                std::io::stdout().flush().ok();
                let animation = spawn_loading_animation("\x1b[90mMyHere [Fast]: \x1b[0m");

                let (needs_smart, fast_response) =
                    match ctx.run_with_context(&fast_pipeline, &mut pctx).await {
                        Ok(resp) => {
                            let needs_smart =
                                pctx.get_ext::<IsFinal>().map(|f| !f.0).unwrap_or(false);
                            (needs_smart, resp.unwrap_or_default())
                        }
                        Err(e) => {
                            tracing::error!("Fast brain pipeline failed: {e}");
                            return;
                        }
                    };

                animation.abort();
                let response = fast_response.trim().to_string();
                if !response.is_empty() {
                    if let Err(e) = ctx.respond(&response).await {
                        tracing::error!("Failed to send fast brain response: {e}");
                    } else {
                        print!("\rMyHere [Fast]: {response}\n\n");
                        std::io::stdout().flush().ok();
                    }
                }

                if !needs_smart {
                    println!("\x1b[90mType your messages below:\x1b[0m");
                    return;
                }

                // ── 3. Smart brain pipeline ───────────────────────────────────
                tracing::info!("Fast brain escalated — running smart brain");

                let smart_client = match LlmClient::new((*smart_llm).clone()) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!("Smart brain LLM init failed: {e}");
                        return;
                    }
                };

                // Include fast brain's response in context for smart brain
                let mut smart_history = (*history).clone();
                if !response.is_empty() {
                    smart_history.push(LlmMessage::assistant(response.clone()));
                }
                let smart_history = Arc::new(smart_history);

                let mut smart_pipeline = Pipeline::new()
                    .add_stage(SimpleContextBuilder::with_prompt_and_history(
                        smart_persona.as_ref(),
                        smart_history,
                    ))
                    .add_streaming_stage(ToolExecutorStage::new(
                        smart_client,
                        Arc::clone(&tool_registry),
                    ))
                    .add_stage(PostProcessor);
                if persist {
                    smart_pipeline = smart_pipeline.add_stage(SqlitePersistence {
                        memory: Arc::clone(&memory),
                        agent_id: agent_id.clone(),
                    });
                }

                pctx.reset_output();

                print!("\x1b[90mMyHere [Smart]: \x1b[0m");
                std::io::stdout().flush().ok();
                let animation = spawn_loading_animation("\x1b[90mMyHere [Smart]: \x1b[0m");

                let mut stream = ctx.run_streaming_with_context(&smart_pipeline, &mut pctx);
                let mut full_response = String::new();

                let mut first_chunk = true;
                while let Some(event) = stream.next().await {
                    match &event {
                        StreamEvent::Chunk { content } => {
                            if first_chunk {
                                animation.abort();
                                print!("\rMyHere [Smart]: ");
                                first_chunk = false;
                            }
                            print!("{content}");
                            let _ = std::io::Write::flush(&mut std::io::stdout());
                            full_response.push_str(content);
                        }
                        StreamEvent::Complete { content, .. } => {
                            if !content.is_empty() {
                                full_response = content.clone();
                            }
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
                println!("\x1b[90mType your messages below:\x1b[0m");
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
