//! Artifact offload stage.
//!
//! Walks the last user message and replaces each inline media part
//! (`Image`/`Audio`/`Video`/`File` carrying `ContentSource::Inline` bytes) with a
//! compact `File` reference whose source is the bare artifact id returned by the
//! [`ArtifactStore`]. The heavy bytes live durably in the store; the conversation
//! carries only the cheap reference.
//!
//! This stage **stores bytes out-of-band**; it does NOT persist anything to chat
//! history — that is [`MemoryPersistence`](crate::pipeline::presets::memory::MemoryPersistence)'s
//! job, kept fully separate. Compose this wherever you want the offload to happen:
//!
//! - **Early** (before the LLM) — the model never sees inline bytes; it gets the
//!   reference and uses `get_artifact` to view. Good for "store the image as soon
//!   as it arrives" flows.
//! - **Late** (after the LLM, before `MemoryPersistence`) — the model saw the
//!   inline image this turn; only history keeps the reference.
//!
//! Opt-in by adding this stage. Omit it to keep inline behavior.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::debug;

use crate::artifacts::{ArtifactManager, ArtifactStore};
use crate::core::context::Context;
use crate::models::Role;
use crate::{PipelineStage, Result};

/// Replaces inline media in the last user message with bare-id artifact
/// references. Delegates the actual work to an [`ArtifactManager`]; decoupled from
/// chat-history persistence — place it anywhere in the pipeline.
pub struct ArtifactOffload {
    manager: ArtifactManager,
}

impl ArtifactOffload {
    pub fn new(store: Arc<dyn ArtifactStore>) -> Self {
        Self::from_manager(ArtifactManager::new(store))
    }

    /// Build from a shared [`ArtifactManager`] (so the stage and the load tool
    /// operate on the same store).
    pub fn from_manager(manager: ArtifactManager) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl PipelineStage for ArtifactOffload {
    fn name(&self) -> &str {
        "ArtifactOffload"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let scope = ctx.message.channel_id.clone();

        let Some(user_msg) = ctx
            .llm_messages
            .iter_mut()
            .rev()
            .find(|m| m.role == Role::User)
        else {
            debug!("ArtifactOffload: no user message; pass-through");
            return Ok(());
        };

        self.manager.offload(&scope, &mut user_msg.content).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::LocalArtifactStore;
    use crate::config::AgentConfig;
    use crate::core::content::{ContentPart, ContentSource};
    use crate::models::{LlmMessage, Message};
    use std::sync::Arc;

    #[tokio::test]
    async fn replaces_inline_image_with_bare_id_reference() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalArtifactStore::new(tmp.path()));

        let msg = Arc::new(Message::new("look", "user", "chan1"));
        let mut ctx = Context::new(msg, Arc::new(AgentConfig::default()));
        ctx.llm_messages.push(LlmMessage::with_parts(
            Role::User,
            vec![
                ContentPart::text("look"),
                ContentPart::image(
                    ContentSource::Inline {
                        data: vec![1, 2, 3],
                    },
                    "image/png",
                ),
            ],
        ));

        ArtifactOffload::new(store.clone())
            .process(&mut ctx)
            .await
            .unwrap();

        let last = ctx.llm_messages.last().unwrap();
        match last.content.last().unwrap() {
            ContentPart::File {
                source: ContentSource::Uri { uri },
                mime_type,
                ..
            } => {
                assert_eq!(mime_type, "image/png");
                // The reference holds a bare id, not an artifact:// URI.
                assert!(!uri.contains("://"));
                // And the id round-trips back to the original bytes.
                let art = futures::executor::block_on(store.load("chan1", uri)).unwrap();
                assert_eq!(art.data, vec![1, 2, 3]);
            }
            other => panic!("expected File reference, got {other:?}"),
        }
    }

    /// `LocalArtifactStore` rejects an empty scope component, so a transport that
    /// leaves `channel_id` blank makes offload fail rather than silently writing
    /// every tenant's media into one shared namespace. Pinning the failure keeps
    /// that a deliberate guard rather than an accident.
    #[tokio::test]
    async fn an_empty_channel_id_fails_offload_instead_of_sharing_a_namespace() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalArtifactStore::new(tmp.path()));

        let msg = Arc::new(Message::new("look", "user", ""));
        let mut ctx = Context::new(msg, Arc::new(AgentConfig::default()));
        ctx.llm_messages.push(LlmMessage::with_parts(
            Role::User,
            vec![ContentPart::image(
                ContentSource::Inline {
                    data: vec![1, 2, 3],
                },
                "image/png",
            )],
        ));

        let err = ArtifactOffload::new(store)
            .process(&mut ctx)
            .await
            .expect_err("an empty scope must not be accepted");
        assert!(
            err.to_string().contains("empty"),
            "error should name the empty component: {err}"
        );
    }
}
