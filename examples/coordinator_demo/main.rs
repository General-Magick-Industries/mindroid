//! Demonstrates the LatestWins run strategy via the coordinator.
//!
//! When a new message arrives while a slow pipeline is still running,
//! the coordinator cancels the old run so only the latest message is processed.
//!
//! Run with: `cargo run -p coordinator_demo`
//! Type a message and press Enter. While the slow pipeline runs (2 seconds),
//! send another message to see LatestWins cancel the first.

use async_trait::async_trait;
use mindroid::Pipeline;
use mindroid::core::context::Context;
use mindroid::core::strategy::RunStrategy;
use mindroid::error::Result;
use mindroid::pipeline::PipelineStage;
use mindroid::transport::stdio::StdioTransport;
use mindroid::Runtime;
use std::time::Duration;

/// A slow stage that simulates work (2 seconds).
/// Checks cancellation periodically so it can be interrupted by LatestWins.
struct SlowStage;

#[async_trait]
impl PipelineStage for SlowStage {
    fn name(&self) -> &str {
        "slow_stage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        tracing::info!("SlowStage: starting work for message: {:?}", ctx.message.content);
        for i in 0..20 {
            if ctx.cancel.is_cancelled() {
                tracing::warn!("SlowStage: CANCELLED at step {}/20 — a newer message took over", i);
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        tracing::info!("SlowStage: completed all 20 steps");
        ctx.response = Some(format!("Done processing: {}", ctx.message.content));
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    tracing::info!("=== Coordinator Demo: LatestWins Strategy ===");
    tracing::info!("Type a message and press Enter. While the slow pipeline runs,");
    tracing::info!("send another message to see LatestWins cancel the first.");
    tracing::info!("");

    let pipeline = Pipeline::new().add_stage(SlowStage);

    let mut runtime = Runtime::builder()
        .transport(StdioTransport::new())
        .pipeline(pipeline)
        .strategy(RunStrategy::LatestWins)
        .build()?;

    runtime.run().await?;
    Ok(())
}
