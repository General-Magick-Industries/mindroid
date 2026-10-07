//! Inline the images a sender attached by reference, so the model sees them on
//! the turn they arrive instead of calling `get_artifact`.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, warn};

use crate::artifacts::{ArtifactManager, ArtifactStore};
use crate::core::content::{
    ARTIFACT_DATA_METADATA_KEY, ArtifactReference, ContentPart, ContentSource,
    MAX_ARTIFACT_REFERENCES,
};
use crate::core::context::Context;
use crate::models::Role;
use crate::pipeline::extensions::CurrentUserMessage;
use crate::pipeline::stages::tool_executor_xml::MAX_REINJECTED_ARTIFACTS;
use crate::{PipelineStage, Result};

/// Default cap on bytes inlined per turn: base64 grows it by a third, and a 1 MiB
/// request cap must also fit the prompt and history.
pub const DEFAULT_MAX_INLINE_BYTES: usize = 512 * 1024;

/// Loads the images in the inbound message's `artifact_data` and attaches them
/// inline to the current user turn, up to [`MAX_REINJECTED_ARTIFACTS`] and
/// [`with_max_bytes`](Self::with_max_bytes). Anything else stays a reference for
/// `get_artifact`. Run it after the user turn is built; it replaces the references
/// [`AttachMedia`](super::AttachMedia) added for the same ids.
pub struct InlineArtifacts {
    store: Arc<dyn ArtifactStore>,
    max_bytes: usize,
}

impl InlineArtifacts {
    pub fn new(store: Arc<dyn ArtifactStore>) -> Self {
        Self {
            store,
            max_bytes: DEFAULT_MAX_INLINE_BYTES,
        }
    }

    /// Use the store behind an [`ArtifactManager`] and its `get_artifact` tool.
    pub fn from_manager(manager: &ArtifactManager) -> Self {
        Self::new(manager.store().clone())
    }

    /// Cap the total bytes inlined in one turn.
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    async fn resolve(&self, scope: &str, references: Vec<ArtifactReference>) -> Vec<ContentPart> {
        let mut budget = self.max_bytes;
        let mut inlined = 0;
        let mut parts = Vec::with_capacity(references.len());
        for reference in references {
            let declared_image =
                reference.mime_type.is_empty() || reference.mime_type.starts_with("image/");
            if inlined < MAX_REINJECTED_ARTIFACTS && declared_image {
                match self.store.load(scope, &reference.id).await {
                    Ok(art) if art.mime_type.starts_with("image/") && art.data.len() <= budget => {
                        budget -= art.data.len();
                        inlined += 1;
                        parts.push(ContentPart::image(
                            ContentSource::Inline { data: art.data },
                            art.mime_type,
                        ));
                        continue;
                    }
                    Ok(art) => debug!(
                        "InlineArtifacts: '{}' is {} ({} bytes, {budget} left); keeping the reference",
                        reference.id,
                        art.mime_type,
                        art.data.len()
                    ),
                    Err(e) => warn!("InlineArtifacts: loading '{}' failed: {e}", reference.id),
                }
            }
            parts.extend(reference.into_part());
        }
        parts
    }
}

#[async_trait]
impl PipelineStage for InlineArtifacts {
    fn name(&self) -> &str {
        "InlineArtifacts"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let references: Vec<ArtifactReference> = ctx
            .message
            .metadata
            .get(ARTIFACT_DATA_METADATA_KEY)
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|v| serde_json::from_value::<ArtifactReference>(v.clone()).ok())
            .filter(|r| crate::core::content::is_artifact_id(&r.id))
            .take(MAX_ARTIFACT_REFERENCES)
            .collect();
        if references.is_empty() {
            return Ok(());
        }

        let current = ctx.get_run::<CurrentUserMessage>().map(|c| c.0);
        let Some(index) = current
            .filter(|&i| {
                ctx.llm_messages
                    .get(i)
                    .is_some_and(|m| m.role == Role::User)
            })
            .or_else(|| ctx.llm_messages.iter().rposition(|m| m.role == Role::User))
        else {
            debug!("InlineArtifacts: no user turn to attach to; pass-through");
            return Ok(());
        };

        let scope = self.store.scope_for(&ctx.message);
        let ids: Vec<String> = references.iter().map(|r| r.id.clone()).collect();
        let parts = self.resolve(&scope, references).await;

        let turn = &mut ctx.llm_messages[index];
        turn.content.retain(|p| {
            !matches!(p, ContentPart::File { source: ContentSource::Uri { uri }, .. } if ids.contains(uri))
        });
        turn.content.extend(parts);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::LocalArtifactStore;
    use crate::config::AgentConfig;
    use crate::models::{LlmMessage, Message};

    async fn setup(
        files: &[(&[u8], &str)],
    ) -> (tempfile::TempDir, Arc<LocalArtifactStore>, Vec<String>) {
        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalArtifactStore::new(tmp.path()));
        let mut ids = Vec::new();
        for (data, mime) in files {
            ids.push(store.save("space1", data, mime).await.unwrap().id);
        }
        (tmp, store, ids)
    }

    fn ctx_with(artifact_data: serde_json::Value) -> Context {
        let mut msg = Message::new("look", "u1", "space1");
        msg.metadata
            .insert(ARTIFACT_DATA_METADATA_KEY.into(), artifact_data);
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));
        ctx.llm_messages = vec![LlmMessage::system("sys"), LlmMessage::user("look")];
        ctx
    }

    fn inline_images(ctx: &Context, index: usize) -> Vec<Vec<u8>> {
        ctx.llm_messages[index]
            .content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Image {
                    source: ContentSource::Inline { data },
                    ..
                } => Some(data.clone()),
                _ => None,
            })
            .collect()
    }

    fn references(ctx: &Context, index: usize) -> Vec<String> {
        ctx.llm_messages[index]
            .content
            .iter()
            .filter_map(|p| match p {
                ContentPart::File {
                    source: ContentSource::Uri { uri },
                    ..
                } => Some(uri.clone()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn an_attached_image_reaches_the_model_inline() {
        let (_tmp, store, ids) = setup(&[(&[1, 2, 3], "image/jpeg")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0], "mime_type": "image/jpeg"}]));

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(inline_images(&ctx, 1), vec![vec![1, 2, 3]]);
        assert!(references(&ctx, 1).is_empty());
    }

    #[tokio::test]
    async fn it_replaces_the_reference_attach_media_added() {
        let (_tmp, store, ids) = setup(&[(&[7], "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0], "mime_type": "image/png"}]));
        super::super::AttachMedia.process(&mut ctx).await.unwrap();
        assert_eq!(references(&ctx, 1), vec![ids[0].clone()]);

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(inline_images(&ctx, 1), vec![vec![7]]);
        assert!(references(&ctx, 1).is_empty());
    }

    #[tokio::test]
    async fn what_cannot_be_inlined_stays_a_reference() {
        let (_tmp, store, ids) = setup(&[(&[1], "application/pdf"), (&[2; 10], "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([
            {"id": ids[0], "mime_type": "application/pdf"},
            {"id": ids[1], "mime_type": "image/png"},
            {"id": "missing", "mime_type": "image/png"},
        ]));

        InlineArtifacts::new(store)
            .with_max_bytes(4)
            .process(&mut ctx)
            .await
            .unwrap();

        assert!(inline_images(&ctx, 1).is_empty());
        assert_eq!(
            references(&ctx, 1),
            vec![ids[0].clone(), ids[1].clone(), "missing".to_string()]
        );
    }

    #[tokio::test]
    async fn the_byte_budget_is_shared_across_the_turn() {
        let (_tmp, store, ids) = setup(&[(&[1; 3], "image/png"), (&[2; 3], "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0]}, {"id": ids[1]}]));

        InlineArtifacts::new(store)
            .with_max_bytes(5)
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(inline_images(&ctx, 1), vec![vec![1; 3]]);
        assert_eq!(references(&ctx, 1), vec![ids[1].clone()]);
    }

    #[tokio::test]
    async fn at_most_the_reinjection_cap_is_inlined() {
        let files: Vec<(Vec<u8>, &str)> = (0..MAX_REINJECTED_ARTIFACTS as u8 + 2)
            .map(|i| (vec![i], "image/png"))
            .collect();
        let borrowed: Vec<(&[u8], &str)> = files.iter().map(|(d, m)| (d.as_slice(), *m)).collect();
        let (_tmp, store, ids) = setup(&borrowed).await;
        let data: Vec<_> = ids.iter().map(|id| serde_json::json!({"id": id})).collect();
        let mut ctx = ctx_with(serde_json::Value::Array(data));

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(inline_images(&ctx, 1).len(), MAX_REINJECTED_ARTIFACTS);
        assert_eq!(references(&ctx, 1).len(), 2);
    }

    #[tokio::test]
    async fn it_attaches_to_the_current_user_turn_not_the_last() {
        let (_tmp, store, ids) = setup(&[(&[9], "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0]}]));
        ctx.llm_messages
            .push(LlmMessage::user("a later injected note"));
        ctx.set(CurrentUserMessage(1));

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(inline_images(&ctx, 1), vec![vec![9]]);
        assert!(inline_images(&ctx, 2).is_empty());
    }

    #[tokio::test]
    async fn a_message_without_artifacts_is_untouched() {
        let (_tmp, store, _) = setup(&[]).await;
        let mut msg = Message::new("hi", "u1", "space1");
        msg.metadata.clear();
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));
        ctx.llm_messages = vec![LlmMessage::user("hi")];

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(ctx.llm_messages[0].content.len(), 1);
    }
}
