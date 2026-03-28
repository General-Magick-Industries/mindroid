use async_trait::async_trait;
use std::sync::Arc;

use crate::error::Result;
use crate::models::SenderType;
use crate::pipeline::{PipelineContext, PipelineStage};

use super::resolver::IdentityResolver;

/// Typed extension stored in PipelineContext for downstream stages.
pub struct CanonicalUserId(pub String);

pub struct IdentityResolutionStage {
    resolver: Arc<IdentityResolver>,
}

impl IdentityResolutionStage {
    pub fn new(resolver: Arc<IdentityResolver>) -> Self {
        Self { resolver }
    }
}

#[async_trait]
impl PipelineStage for IdentityResolutionStage {
    fn name(&self) -> &str {
        "IdentityResolution"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        if ctx.message.sender_type != SenderType::User {
            return Ok(());
        }

        let platform = ctx.message.platform.as_deref().unwrap_or("unknown");
        let canonical = self
            .resolver
            .resolve(platform, &ctx.message.sender_id)
            .await;

        tracing::debug!(
            "IdentityResolution: {}:{} → canonical {}",
            platform,
            ctx.message.sender_id,
            canonical
        );

        ctx.set_ext(CanonicalUserId(canonical));
        Ok(())
    }
}
