//! An agentic turn built from stages instead of from an executor's private loop.
//!
//! The body pipeline *is* the agent's reasoning loop, so every step of it is an
//! ordinary stage that can be reordered, replaced or wrapped in a combinator:
//!
//! ```text
//! setup : SimpleContextBuilder          — once
//! body  : TranscriptCompaction          — before every model call
//!         LlmRound                      — one call, records the tool calls
//!         ToolRound                     — runs them, asks for another pass
//! finish: PostProcessor                 — once
//! ```
//!
//! Run against any OpenAI-compatible endpoint:
//!
//! ```bash
//! cargo run -p mindroid-example-agent-loop --bin agent_loop -- \
//!     --base-url "$LITELLM_URL" --model gpt-4o-mini \
//!     "how much disk space is free on this machine?"
//! ```
//!
//! `MINDROID_API_KEY` supplies the key. The agent has one tool — `shell` — so a
//! question it cannot answer from memory forces a real multi-pass turn.

use std::sync::Arc;

use clap::Parser;
use mindroid::config::AgentConfig;
use mindroid::llm_client::{LlmClient, LlmClientConfig};
use mindroid::{
    AgentLoop, Context, LlmRound, Message, Pipeline, PostProcessor, ShellTool,
    SimpleContextBuilder, ToolRegistry, ToolRound, TranscriptCompaction,
};

const SYSTEM: &str = "You are a terse assistant with shell access on this machine. \
     Use the shell tool to find things out rather than guessing, then answer in one sentence.";

#[derive(Parser)]
struct Args {
    /// The question to answer.
    prompt: Vec<String>,
    /// OpenAI-compatible base URL, including `/v1`.
    #[arg(long, default_value = "http://localhost:11434/v1")]
    base_url: String,
    #[arg(long, default_value = "gpt-4o-mini")]
    model: String,
    /// Approximate token budget for the transcript before old rounds are dropped.
    #[arg(long, default_value_t = 60_000)]
    context_budget: usize,
    /// Cap on body passes.
    #[arg(long, default_value_t = 10)]
    max_iterations: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("warn,mindroid=debug")
        .init();

    let args = Args::parse();
    let prompt = args.prompt.join(" ");
    if prompt.is_empty() {
        anyhow::bail!("give the agent something to do");
    }

    let mut llm = LlmClientConfig::new(&args.base_url);
    llm.api_key = std::env::var("MINDROID_API_KEY").ok();
    llm.default_model = Some(args.model.clone());
    let client = LlmClient::new(llm)?;

    let registry = Arc::new(ToolRegistry::new().register(ShellTool::new(30)));

    let agent = AgentLoop::new(
        Pipeline::new()
            .add_stage(TranscriptCompaction::from_tokens(args.context_budget))
            .add_stage(LlmRound::new(client, registry.clone()))
            .add_stage(ToolRound::new(registry)),
    )
    .with_setup(Pipeline::new().add_stage(SimpleContextBuilder::with_prompt(SYSTEM)))
    .with_finish(Pipeline::new().add_stage(PostProcessor))
    .with_max_iterations(args.max_iterations);

    let mut ctx = Context::new(
        Arc::new(Message::new(prompt, "operator", "cli")),
        Arc::new(AgentConfig::default()),
    );

    let outcome = agent.run(&mut ctx).await?;

    println!("\n{}", outcome.response.unwrap_or_default());
    eprintln!(
        "\n[{} pass(es), stopped: {:?}]",
        outcome.iterations, outcome.reason
    );
    Ok(())
}
