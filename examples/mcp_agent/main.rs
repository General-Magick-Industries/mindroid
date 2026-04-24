//! MCP-powered agent: Ollama + stdio + MCP tool servers.
//!
//! Demonstrates connecting to external MCP servers (e.g. Context7) so that
//! tools from those servers appear automatically in the agent's toolbox.
//!
//! Run with:
//!   CONTEXT7_API_KEY=your-key cargo run -p mindroid-example-mcp-agent -- --config examples/mcp_agent/config.toml
//!
//! Or without config (pure code setup):
//!   CONTEXT7_API_KEY=your-key cargo run -p mindroid-example-mcp-agent

use std::sync::Arc;

use futures::StreamExt;
use mindroid::auth::static_id::StaticAuth;
use mindroid::transport::stdio::StdioTransport;
use mindroid::{
    GenericLlmProcessor, Pipeline, PipelineContext, PostProcessor, Runtime, SimpleContextBuilder,
    StreamEvent, Tool, ToolExecutorStage, ToolRegistry, XmlToolCallParser,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,mindroid=debug")
        .init();

    // -- Build tool registry with MCP servers ---------------------------------

    let mut registry = ToolRegistry::new().register(mindroid::ShellTool::default());

    // Load MCP tools from Context7 (or any MCP server).
    // In production you'd use config-driven loading via ToolRegistry::load_mcp_servers().
    let api_key = std::env::var("CONTEXT7_API_KEY").ok();
    let mcp_tools = mindroid::load_mcp_tools(
        "context7",
        "https://mcp.context7.com/mcp",
        api_key.as_deref(),
    )
    .await?;

    tracing::info!("Loaded {} MCP tools from Context7", mcp_tools.len());
    for tool in &mcp_tools {
        tracing::info!("  - {} : {}", tool.name(), tool.description());
    }

    for tool in mcp_tools {
        registry = registry.register(tool);
    }

    let registry = Arc::new(registry);

    // -- Build LLM clients (Ollama) -------------------------------------------

    let llm_config =
        mindroid::llm_client::LlmClientConfig::new("http://localhost:11434/v1");
    let llm_stream = mindroid::llm_client::LlmClient::new(llm_config.clone())?;
    let llm_tools = mindroid::llm_client::LlmClient::new(llm_config)?;

    // -- Build pipeline: context → LLM (streaming) → tool executor → post -----

    let tool_stage = ToolExecutorStage::new(llm_tools, registry.clone())
        .with_parser(XmlToolCallParser);

    let pipeline = Arc::new(
        Pipeline::new()
            .add_stage(SimpleContextBuilder::new())
            .add_streaming_stage(GenericLlmProcessor::new(llm_stream))
            .add_stage(tool_stage)
            .add_stage(PostProcessor),
    );

    // -- Wire up runtime ------------------------------------------------------

    let mut runtime = Runtime::builder()
        .transport(StdioTransport::new())
        .auth(StaticAuth::new("dev"))
        .on_message(move |ctx| {
            let registry = Arc::clone(&registry);
            let pipeline = Arc::clone(&pipeline);

            async move {
                let mut pctx =
                    PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

                // Inject tool descriptions into the system prompt
                let tool_prompt = registry.system_prompt();
                pctx.llm_messages.push(mindroid::LlmMessage::system(format!(
                    "You are a helpful coding assistant with access to external documentation tools.\n\
                     When the user asks about a library or framework, use the context7 tools to \
                     look up current documentation before answering.\n\n{tool_prompt}"
                )));

                // Run the pipeline (streaming)
                let mut full_response = String::new();
                {
                    let mut stream = ctx.run_streaming_with_context(&pipeline, &mut pctx);
                    while let Some(event) = stream.next().await {
                        match &event {
                            StreamEvent::Chunk { content } => {
                                full_response.push_str(content);
                            }
                            StreamEvent::Complete { content, .. } => {
                                if !content.is_empty() {
                                    full_response = content.clone();
                                }
                            }
                            StreamEvent::Error { message } => {
                                tracing::error!("Stream error: {message}");
                            }
                            _ => {}
                        }
                    }
                }

                // Check post-processing output
                if let Some(ref response) = pctx.response {
                    full_response = response.clone();
                }

                let response = full_response.trim().to_string();
                if !response.is_empty() {
                    if let Err(e) = ctx.respond(&response).await {
                        tracing::error!("Failed to send response: {e}");
                    }
                }
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
