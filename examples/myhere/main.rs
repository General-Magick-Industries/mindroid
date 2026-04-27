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

mod myhere;

use std::sync::Arc;

use mindroid::memory::sqlite::SqliteMemory;
use mindroid::{ContextPreparer, MindroidConfig, Runtime};

use myhere::{create_tool_registry, build_myhere_pipeline, SqliteContextProvider};

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

    // Build tool registry — add custom tools before wrapping in Arc
    let mut tool_registry = create_tool_registry();
    // Example: tool_registry = tool_registry.register(MyCustomTool::default());

    let tool_registry = Arc::new(tool_registry);

    let builder = Runtime::from_config(config)?;

    println!("MyHere is running! This agent has a Fast brain for quick answers and a Smart brain for complex questions.");
    println!("\x1b[90mType your messages below:\x1b[0m");

    let mut runtime = builder
        .on_message(build_myhere_pipeline(
            context_preparer,
            memory,
            fast_llm,
            smart_llm,
            fast_persona,
            smart_persona,
            tool_registry,
            agent_id,
        ))
        .build()?;

    runtime.run().await?;
    Ok(())
}
