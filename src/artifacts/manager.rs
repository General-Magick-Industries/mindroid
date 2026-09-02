//! [`ArtifactManager`] — the store-agnostic orchestration layer.
//!
//! Division of responsibility:
//! - [`ArtifactStore`] (the trait) is **pure CRUD** (`save`/`load`/`delete`).
//!   Implementing a new backend (disk, S3, Magickmind) means writing only those
//!   three methods.
//! - `ArtifactManager` wraps any store and owns the **backend-agnostic logic**
//!   that both the offload stage and the load tool need: walking message content
//!   to offload inline media, and validating + describing an id for the tool. It
//!   is written once and works with every store because it only calls the CRUD
//!   primitives. Framework code; not something a store author touches.

use std::sync::Arc;

use crate::artifacts::ArtifactStore;
use crate::core::content::{ContentPart, ContentSource};
use crate::error::Result;

/// Wraps an [`ArtifactStore`] and provides the offload + load-tool orchestration
/// that the [`ArtifactOffload`](crate::pipeline::stages::ArtifactOffload) stage
/// and [`GetArtifactTool`](crate::tools::GetArtifactTool) delegate to.
///
/// Cheaply cloneable (holds an `Arc`). Build one and share it across the stage
/// and tool so they operate on the same store.
#[derive(Clone)]
pub struct ArtifactManager {
    store: Arc<dyn ArtifactStore>,
}

impl ArtifactManager {
    pub fn new(store: Arc<dyn ArtifactStore>) -> Self {
        Self { store }
    }

    /// The underlying store (for direct CRUD or re-injection by the executor).
    pub fn store(&self) -> &Arc<dyn ArtifactStore> {
        &self.store
    }

    /// Mint the matched `(offload stage, get_artifact tool)` pair from this manager
    /// — the one call a dev needs to wire artifacts into a pipeline. Both share this
    /// manager's store, so what the stage offloads, the tool can always load back.
    ///
    /// ```ignore
    /// let (offload, tool) = ArtifactManager::new(store).into_stage_and_tool(channel_id);
    /// let registry = ToolRegistry::new().register(tool);
    /// let pipeline = Pipeline::new()
    ///     .add_stage(IngestStage::default_media())
    ///     .add_streaming_stage(XmlToolExecutorStage::new(client, Arc::new(registry)))
    ///     .add_stage(offload);
    /// ```
    #[cfg(feature = "llm-client")]
    pub fn into_stage_and_tool(
        self,
        scope: impl Into<String>,
    ) -> (
        crate::pipeline::stages::ArtifactOffload,
        crate::tools::GetArtifactTool,
    ) {
        let offload = crate::pipeline::stages::ArtifactOffload::from_manager(self.clone());
        let tool = crate::tools::GetArtifactTool::from_manager(self, scope);
        (offload, tool)
    }

    /// Offload every inline media part in `content` to the store under `scope`,
    /// replacing each with a bare-id `File` reference. Text / already-referenced
    /// parts are untouched. Returns the number of parts offloaded.
    pub async fn offload(&self, scope: &str, content: &mut [ContentPart]) -> Result<usize> {
        let mut count = 0;
        for part in content.iter_mut() {
            let Some((data, mime)) = inline_media(part) else {
                continue;
            };
            let mime = mime.to_string();
            // The store returns the id plus any metadata it chose to attach (a
            // caption, backend facts…); that metadata rides along on the reference.
            let stored = self.store.save(scope, data, &mime).await?;
            *part = ContentPart::File {
                source: ContentSource::Uri { uri: stored.id },
                mime_type: mime,
                filename: None,
                metadata: stored.metadata,
            };
            count += 1;
        }
        Ok(count)
    }

    /// Validate an artifact id and return a human-readable confirmation for the
    /// load tool (the actual byte re-injection is done by the executor). Returns
    /// an error string as a normal `Ok` so the model can recover.
    pub async fn load_described(&self, scope: &str, id: &str) -> String {
        if id.is_empty() {
            return "Error: no artifact id provided".to_string();
        }
        match self.store.load(scope, id).await {
            Ok(_) => format!("Loaded artifact {id}"),
            Err(e) => format!("Error: could not load artifact {id}: {e}"),
        }
    }
}

/// Extract `(bytes, mime)` from an inline media part. Returns `None` for text
/// parts, already-referenced (`Uri`) parts, or non-media.
fn inline_media(part: &ContentPart) -> Option<(&[u8], &str)> {
    match part {
        ContentPart::Image {
            source: ContentSource::Inline { data },
            mime_type,
            ..
        }
        | ContentPart::Audio {
            source: ContentSource::Inline { data },
            mime_type,
            ..
        }
        | ContentPart::Video {
            source: ContentSource::Inline { data },
            mime_type,
            ..
        }
        | ContentPart::File {
            source: ContentSource::Inline { data },
            mime_type,
            ..
        } => Some((data, mime_type)),
        _ => None,
    }
}

#[cfg(all(test, feature = "llm-client"))]
mod tests {
    use super::*;
    use crate::artifacts::LocalArtifactStore;
    use crate::config::AgentConfig;
    use crate::models::{LlmMessage, Message, Role};
    use crate::pipeline::PipelineStage;
    use crate::tools::{Tool, ToolContext};

    /// The minted pair must share the manager's store: what the stage offloads,
    /// the tool resolves. Driven end-to-end through the returned stage and tool
    /// (the stage holds its manager privately, so this is the real assertion).
    #[tokio::test]
    async fn into_stage_and_tool_shares_one_store() {
        let tmp = tempfile::tempdir().unwrap();
        let store: Arc<dyn ArtifactStore> = Arc::new(LocalArtifactStore::new(tmp.path()));
        let (offload, tool) = ArtifactManager::new(store).into_stage_and_tool("chan1");

        let message = Message::new("look", "user-1", "chan1");
        let mut ctx = crate::Context::new(message.into(), AgentConfig::default().into());
        ctx.llm_messages.push(LlmMessage::with_parts(
            Role::User,
            vec![
                ContentPart::text("look"),
                ContentPart::image(
                    ContentSource::Inline {
                        data: vec![9, 9, 9],
                    },
                    "image/png",
                ),
            ],
        ));

        offload.process(&mut ctx).await.unwrap();

        // The inline image is now a bare-id File reference.
        let id = match &ctx.llm_messages.last().unwrap().content[1] {
            ContentPart::File {
                source: ContentSource::Uri { uri },
                ..
            } => uri.clone(),
            other => panic!("expected a File reference, got {other:?}"),
        };

        // The tool from the SAME manager resolves that id. Assert the success
        // string, not just the id — the failure string also contains the id.
        let out = tool
            .execute(serde_json::json!({ "id": id }), &ToolContext::default())
            .await
            .unwrap();
        assert!(
            out.starts_with("Loaded artifact"),
            "tool must load from the shared store, got: {out}"
        );

        // And the store the tool carries returns the original bytes.
        let bytes = tool
            .artifact_store()
            .expect("tool carries the store")
            .load("chan1", &id)
            .await
            .unwrap()
            .data;
        assert_eq!(bytes, vec![9, 9, 9]);
    }
}
