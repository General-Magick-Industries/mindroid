use async_trait::async_trait;

use crate::core::context::Context;
use crate::{PipelineStage, Result};

/// Trims whitespace from the LLM response and writes it back to `ctx.response`.
///
/// This is typically the last stage in a pipeline.
pub struct PostProcessor;

#[async_trait]
impl PipelineStage for PostProcessor {
    fn name(&self) -> &str {
        "PostProcessor"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        ctx.response = Some(ctx.response.as_deref().unwrap_or("").trim().to_string());
        Ok(())
    }
}
