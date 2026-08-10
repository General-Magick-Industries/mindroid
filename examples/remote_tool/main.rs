//! Remote (client-executed) tool round trip, backed by a local SQLite db.
//!
//! A `RemoteTool` is only *declared* to the LLM. When the LLM calls it, the
//! ToolExecutor emits the call as the pipeline response instead of running it —
//! the "client" performs the work and feeds the result back as a new message.
//! SQLite is the local db that carries conversation state between the two runs.
//!
//! Flow:
//!   run 1:  user asks -> LLM emits <tool_call get_time> -> executor frames it
//!           as the response  (saved to sqlite)
//!   client: performs get_time, writes the <tool_result> back  (saved to sqlite)
//!   run 2:  LLM sees the history (incl. the result) -> final natural answer
//!
//! Configure the LLM via env before running. Ollama (default):
//!   REMOTE_TOOL_BASE_URL=http://localhost:11434  REMOTE_TOOL_MODEL=llama3.2 \
//!     cargo run -p mindroid-example-remote-tool
//!
//! LiteLLM proxy (OpenAI-compatible; needs a proxy key):
//!   REMOTE_TOOL_BASE_URL=http://localhost:4000  REMOTE_TOOL_MODEL=gpt-4o-mini \
//!   REMOTE_TOOL_API_KEY=sk-... \
//!     cargo run -p mindroid-example-remote-tool

use std::sync::Arc;

use mindroid::core::config::AgentConfig;
use mindroid::llm_client::{LlmClient, LlmClientConfig};
use mindroid::memory::Memory;
use mindroid::memory::sqlite::SqliteMemory;
use mindroid::models::{LlmMessage, Message};
use mindroid::pipeline::stages::{SimpleContextBuilder, ToolExecutorStage};
use mindroid::{Pipeline, PipelineContext, RemoteTool, Result, ToolRegistry};

const CHANNEL: &str = "demo-channel";
const SENDER: &str = "demo-user";
const SYSTEM_PROMPT: &str = "You are a helpful assistant. You have a `get_time` tool. \
     When the user asks for the time, call it. Do not guess the time yourself.";

fn llm_client() -> Result<LlmClient> {
    // Defaults suit a local Ollama; override for a LiteLLM proxy, e.g.:
    //   REMOTE_TOOL_BASE_URL=http://localhost:4000
    //   REMOTE_TOOL_MODEL=gpt-4o-mini
    //   REMOTE_TOOL_API_KEY=sk-...        (LiteLLM proxy key)
    let base_url = std::env::var("REMOTE_TOOL_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    let model = std::env::var("REMOTE_TOOL_MODEL").unwrap_or_else(|_| "llama3.2".to_string());
    let mut cfg = LlmClientConfig::new(format!("{base_url}/v1"));
    cfg.default_model = Some(model);
    cfg.api_key = std::env::var("REMOTE_TOOL_API_KEY").ok();
    LlmClient::new(cfg)
}

/// Load conversation history from the local db as LLM messages.
async fn load_history(mem: &SqliteMemory) -> Result<Arc<Vec<LlmMessage>>> {
    let msgs = mem.get_history(CHANNEL, 50).await?;
    let llm = msgs
        .into_iter()
        .map(|m| {
            if m.sender_id == SENDER {
                LlmMessage::user(m.content)
            } else {
                LlmMessage::assistant(m.content)
            }
        })
        .collect();
    Ok(Arc::new(llm))
}

/// Run one message through a freshly-built pipeline (history reloaded from db).
/// History excludes the message being run, so ContextBuilder appends it as the
/// current user turn.
async fn run_turn(
    mem: &SqliteMemory,
    executor: ToolExecutorStage,
    content: &str,
) -> Result<String> {
    let history = load_history(mem).await?;
    // ToolExecutorStage IS the LLM caller (it adds the tool prompt and parses
    // <tool_call>), so it's the streaming stage — no separate LLM stage.
    let gate = executor.result_gate();
    let pipeline = Pipeline::new()
        .add_stage(gate)
        .add_stage(SimpleContextBuilder::with_prompt_and_history(
            SYSTEM_PROMPT,
            history,
        ))
        .add_streaming_stage(executor);

    let msg = Arc::new(Message::new(content, SENDER, CHANNEL));
    let mut ctx = PipelineContext::new(msg, Arc::new(AgentConfig::default()));
    // run() returns the response (and takes it out of ctx), so use the return value.
    let response = pipeline.run(&mut ctx).await?;
    Ok(response.unwrap_or_default())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    // Local db (fresh temp file each run).
    let db_path = std::env::temp_dir()
        .join("mindroid_remote_tool_demo.sqlite")
        .to_string_lossy()
        .into_owned();
    let _ = std::fs::remove_file(&db_path);
    let mem = SqliteMemory::new(&db_path)?;
    println!("local db: {db_path}\n");

    // Declare the remote tool — runtime never executes it.
    let registry = Arc::new(ToolRegistry::new().register(
        RemoteTool::new("get_time", "Get the current wall-clock time.").schema(serde_json::json!({
            "type": "object", "properties": {}
        })),
    ));
    let executor = ToolExecutorStage::new(llm_client()?, registry);

    // ── run 1: user asks; LLM should emit the remote tool call ───────────────
    let question = "What time is it?";
    println!("[user]      {question}");
    let emitted = run_turn(&mem, executor.clone(), question).await?;
    println!("[runtime →] {emitted}");

    // Persist the turn: user question + the emitted call (assistant turn) so
    // run 2's history reflects that the call was made.
    mem.save_message(CHANNEL, SENDER, question, None).await?;
    mem.save_message(CHANNEL, "assistant", &emitted, None)
        .await?;

    // ── client performs the tool and feeds the result back ───────────────────
    let call: serde_json::Value = serde_json::from_str(&emitted).unwrap_or(serde_json::Value::Null);
    if call["type"] != "tool_call" {
        println!("\n(LLM did not emit the remote tool — got a chat reply instead)");
        println!("Ensure your configured model emits <tool_call> XML for tools.");
        return Ok(());
    }
    println!(
        "[client]    got tool_call id={} name={}",
        call["payload"]["tool_call_id"], call["payload"]["name"]
    );
    if let Some(ack) = call["payload"]["ack"].as_str().filter(|s| !s.is_empty()) {
        println!("[npc says]  {ack}"); // spoken while the action runs
    }

    // "Execute" get_time on the client side and feed the result back as a new
    // inbound message (stored in the local db, seen by run 2's history).
    let now = "2026-08-03T15:04:05Z";
    let call_id = call["payload"]["tool_call_id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("tool call omitted its correlation id"))?;
    let tool_result =
        format!("<tool_result name=\"get_time\" call=\"{call_id}\">{now}</tool_result>");
    println!("[client]    performed get_time -> {now}");

    // Run 2: the result is the current turn (ContextBuilder appends it), so run
    // first, then persist both the result and the agent's answer — matching the
    // run-1 ordering and avoiding a double-append.
    let answer = run_turn(&mem, executor, &tool_result).await?;
    mem.save_message(CHANNEL, SENDER, &tool_result, None)
        .await?;
    mem.save_message(CHANNEL, "assistant", &answer, None)
        .await?;
    println!("[assistant] {answer}");

    Ok(())
}
