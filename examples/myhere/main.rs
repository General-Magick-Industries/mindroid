#[allow(dead_code)]
mod myhere;

use std::io::Write;
use std::sync::Arc;

use futures::StreamExt;

use mindroid::memory::sqlite::SqliteMemory;
use mindroid::pipeline::presets::sqlite::{SqliteClient, SqliteContext};
use mindroid::{
    ContextPreparer, MessageContext, MindroidConfig, Pipeline, PipelineContext, PostProcessor,
    Runtime, StreamEvent,
};

use myhere::{
    add_persistence_stage, create_tool_registry, DualBrainEventMode, DualBrainStage,
    PersistenceBackend, PrepareHistoryStage,
};

// ── Animation helper ─────────────────────────────────────────────────────

pub fn spawn_loading_animation(initial_label: &str) -> tokio::task::JoinHandle<()> {
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
        .ok_or_else(|| anyhow::anyhow!("fast model persona is required"))?
        .into();

    let smart_persona: Arc<str> = config
        .models
        .get("smart")
        .and_then(|m| m.options.get("persona"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("smart model persona is required"))?
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

    // ✅ NEW: use SqliteClient (not raw memory everywhere)
    let memory = Arc::new(SqliteMemory::new(db_path)?);
    let client = Arc::new(SqliteClient::new(memory));

    let agent_id = if config.agent.agent_id.trim().is_empty() {
        config.agent.name.clone()
    } else {
        config.agent.agent_id.clone()
    };

    // use SqliteContext (preset)
    let context_preparer = Arc::new(
        ContextPreparer::new().add_provider(
            SqliteContext::new(client.clone())
                .with_agent_id(agent_id.clone())
                .with_limit(history_limit),
        ),
    );

    // Tool registry
    let tool_registry = create_tool_registry();
    let tool_registry = Arc::new(tool_registry);

    let builder = Runtime::from_config(config)?;

    println!("MyHere is running!");
    println!("\x1b[90mType your messages below:\x1b[0m");

    let message_handler = move |ctx: MessageContext| {
        let context_preparer = Arc::clone(&context_preparer);
        let client = Arc::clone(&client);
        let fast_llm = fast_llm.clone();
        let smart_llm = smart_llm.clone();
        let fast_persona = Arc::clone(&fast_persona);
        let smart_persona = Arc::clone(&smart_persona);
        let tool_registry = Arc::clone(&tool_registry);

        Box::pin(async move {
            let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

            let persistence = PersistenceBackend::Sqlite(client.clone());

            let pipeline = add_persistence_stage(
                Pipeline::new()
                    .add_stage(PrepareHistoryStage::new(context_preparer))
                    .add_streaming_stage(
                        DualBrainStage::new(
                            fast_llm,
                            smart_llm,
                            fast_persona,
                            smart_persona,
                            tool_registry,
                        )
                        .with_event_mode(DualBrainEventMode::SpinnerFriendly),
                    )
                    .add_stage(PostProcessor),
                Some(persistence),
            );

            let is_stdio =
                ctx.message.channel_id.is_empty() || ctx.message.channel_id == "stdio";

            if is_stdio {
                print!("\n");
            }

            let mut animation: Option<tokio::task::JoinHandle<()>> = None;
            let mut current_label: Option<String> = None;
            let mut started_line = false;
            let mut printed_any = false;

            {
                let mut stream = ctx.run_streaming_with_context(&pipeline, &mut pctx);

                while let Some(event) = stream.next().await {
                    match event {
                        StreamEvent::Thinking { content } => {
                            if is_stdio {
                                if let Some(anim) = animation.take() {
                                    anim.abort();
                                }

                                if printed_any {
                                    print!("\n");
                                }

                                let label = format!("\x1b[90m{content}: \x1b[0m");
                                current_label = Some(label.clone());
                                animation = Some(spawn_loading_animation(&label));
                                started_line = false;
                            }
                        }
                        StreamEvent::Chunk { content } => {
                            if is_stdio {
                                if let Some(anim) = animation.take() {
                                    anim.abort();
                                }

                                if !started_line {
                                    let label = current_label
                                        .as_deref()
                                        .unwrap_or("\x1b[90mMyHere: \x1b[0m");
                                    print!("\r{label}\x1b[K");
                                    started_line = true;
                                }

                                print!("{content}");
                                std::io::stdout().flush().ok();
                                printed_any = true;
                            }
                        }
                        StreamEvent::Complete { content, .. } => {
                            if is_stdio {
                                if let Some(anim) = animation.take() {
                                    anim.abort();
                                }

                                if !printed_any && !content.is_empty() {
                                    if !started_line {
                                        let label = current_label
                                            .as_deref()
                                            .unwrap_or("\x1b[90mMyHere: \x1b[0m");
                                        print!("\r{label}\x1b[K");
                                        started_line = true;
                                    }

                                    print!("{content}");
                                    std::io::stdout().flush().ok();
                                }
                            }
                        }
                        StreamEvent::Error { message } => {
                            tracing::error!("Stream error: {message}");
                        }
                        _ => {}
                    }
                }

            }
            if let Some(anim) = animation.take() {
                anim.abort();
            }

            let response = pctx.response.as_deref().unwrap_or("").trim().to_string();

            if is_stdio {
                print!("\n\n");
                std::io::stdout().flush().ok();
            } else if !response.is_empty() {
                if let Err(e) = ctx.respond(&response).await {
                    tracing::error!("Failed to send response: {e}");
                }
            }
        })
    };

    let mut runtime: Runtime = builder.on_message(message_handler).build()?;
    runtime.run().await?;

    Ok(())
}