//! Demonstrates building a custom pipeline stage.
//!
//! This example creates a simple echo pipeline that uppercases the input,
//! showing how to implement `PipelineStage` for custom processing.
//!
//! Run with: `cargo run --example custom_pipeline --features stdio,static-auth`

use async_trait::async_trait;
use mindroid::{Pipeline, PipelineContext, PipelineStage, Result, Runtime};
use mindroid::auth::static_id::StaticAuth;
use mindroid::transport::stdio::StdioTransport;

/// A custom stage that echoes the message content in uppercase.
struct UppercaseEcho;

#[async_trait]
impl PipelineStage for UppercaseEcho {
    fn name(&self) -> &str {
        "UppercaseEcho"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        ctx.response = Some(ctx.message.content.to_uppercase());
        Ok(())
    }
}

/// A stage that wraps the response with a prefix and suffix.
struct Wrapper {
    prefix: String,
    suffix: String,
}

#[async_trait]
impl PipelineStage for Wrapper {
    fn name(&self) -> &str {
        "Wrapper"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let raw = ctx.response.as_deref().unwrap_or("");
        ctx.response = Some(format!("{}{}{}", self.prefix, raw, self.suffix));
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let pipeline = Pipeline::new()
        .add_stage(UppercaseEcho)
        .add_stage(Wrapper {
            prefix: ">>> ".into(),
            suffix: " <<<".into(),
        });

    let mut runtime = Runtime::builder()
        .transport(StdioTransport::new())
        .pipeline(pipeline)
        .auth(StaticAuth::new("dev"))
        .on_message(|ctx| async move {
            if let Err(e) = ctx.process_and_respond().await {
                tracing::error!("Error: {e}");
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
