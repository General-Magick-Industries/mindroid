//! Inline the images a sender attached by reference, so the model sees them on
//! the turn they arrive instead of calling `get_artifact`.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, warn};

use crate::artifacts::{ArtifactManager, ArtifactStore};
use crate::core::content::{
    ARTIFACT_DATA_METADATA_KEY, ArtifactReference, ContentPart, ContentSource,
    MAX_ARTIFACT_REFERENCES, is_artifact_id,
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
/// inline to the current user turn; with
/// [`with_history_messages`](Self::with_history_messages) it also inlines the
/// references in that many earlier messages, newest first. The current turn has
/// first claim on [`with_max_images`](Self::with_max_images) and
/// [`with_max_bytes`](Self::with_max_bytes); anything else stays a reference for
/// `get_artifact`. Run it after the user turn is built; it replaces the references
/// [`AttachMedia`](super::AttachMedia) added for the same ids.
pub struct InlineArtifacts {
    store: Arc<dyn ArtifactStore>,
    max_bytes: usize,
    max_images: usize,
    history_messages: usize,
}

struct Budget {
    bytes: usize,
    images: usize,
    inlined: HashSet<String>,
}

impl InlineArtifacts {
    pub fn new(store: Arc<dyn ArtifactStore>) -> Self {
        Self {
            store,
            max_bytes: DEFAULT_MAX_INLINE_BYTES,
            max_images: MAX_REINJECTED_ARTIFACTS,
            history_messages: 0,
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

    /// Cap the images inlined in one turn.
    pub fn with_max_images(mut self, max_images: usize) -> Self {
        self.max_images = max_images;
        self
    }

    /// Also inline references in the `count` messages before the current turn,
    /// system prompts not counted. Defaults to 0, the current turn only.
    pub fn with_history_messages(mut self, count: usize) -> Self {
        self.history_messages = count;
        self
    }

    async fn try_inline(
        &self,
        scope: &str,
        id: &str,
        mime_type: &str,
        budget: &mut Budget,
    ) -> Option<ContentPart> {
        let maybe_image = mime_type.is_empty()
            || mime_type == "application/octet-stream"
            || mime_type.starts_with("image/");
        if budget.images == 0 || !maybe_image || budget.inlined.contains(id) {
            return None;
        }
        match self.store.load(scope, id).await {
            Ok(art) => match sniff_image(&art.data) {
                Some(sniffed) if art.data.len() <= budget.bytes => {
                    budget.bytes -= art.data.len();
                    budget.images -= 1;
                    budget.inlined.insert(id.to_string());
                    Some(ContentPart::image(
                        ContentSource::Inline { data: art.data },
                        sniffed,
                    ))
                }
                _ => {
                    debug!(
                        "InlineArtifacts: '{id}' ({}, {} bytes, {} left) cannot be inlined; keeping the reference",
                        art.mime_type,
                        art.data.len(),
                        budget.bytes
                    );
                    None
                }
            },
            Err(e) => {
                warn!("InlineArtifacts: loading '{id}' failed: {e}");
                None
            }
        }
    }
}

/// The image formats vision endpoints accept, by signature rather than declared
/// type: one HEIC, SVG or corrupt file inlined fails the whole request.
fn sniff_image(data: &[u8]) -> Option<&'static str> {
    match data {
        [
            0x89,
            b'P',
            b'N',
            b'G',
            0x0D,
            0x0A,
            0x1A,
            0x0A,
            _,
            _,
            _,
            _,
            b'I',
            b'H',
            b'D',
            b'R',
            ..,
        ] => Some("image/png"),
        [0xFF, 0xD8, 0xFF, ..] => Some("image/jpeg"),
        [b'G', b'I', b'F', b'8', b'7' | b'9', b'a', ..] => Some("image/gif"),
        [
            b'R',
            b'I',
            b'F',
            b'F',
            _,
            _,
            _,
            _,
            b'W',
            b'E',
            b'B',
            b'P',
            ..,
        ] => Some("image/webp"),
        _ => None,
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
            .filter(|r| is_artifact_id(&r.id))
            .take(MAX_ARTIFACT_REFERENCES)
            .collect();
        if references.is_empty() && self.history_messages == 0 {
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
        let mut budget = Budget {
            bytes: self.max_bytes,
            images: self.max_images,
            inlined: HashSet::new(),
        };

        if !references.is_empty() {
            let ids: Vec<String> = references.iter().map(|r| r.id.clone()).collect();
            let mut parts = Vec::with_capacity(references.len());
            for reference in references {
                match self
                    .try_inline(&scope, &reference.id, &reference.mime_type, &mut budget)
                    .await
                {
                    Some(image) => parts.push(image),
                    None => parts.extend(reference.into_part()),
                }
            }
            let turn = &mut ctx.llm_messages[index];
            turn.content.retain(|p| {
                !matches!(p, ContentPart::File { source: ContentSource::Uri { uri }, .. } if ids.contains(uri))
            });
            turn.content.extend(parts);
        }

        let earlier: Vec<(usize, usize, String, String)> = ctx.llm_messages[..index]
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, m)| m.role != Role::System)
            .take(self.history_messages)
            .filter(|(_, m)| m.role == Role::User)
            .flat_map(|(i, m)| {
                m.content
                    .iter()
                    .enumerate()
                    .filter_map(move |(j, part)| match part {
                        ContentPart::File {
                            source: ContentSource::Uri { uri },
                            mime_type,
                            ..
                        } if is_artifact_id(uri) => Some((i, j, uri.clone(), mime_type.clone())),
                        _ => None,
                    })
            })
            .collect();
        for (i, j, id, mime_type) in earlier {
            if let Some(image) = self.try_inline(&scope, &id, &mime_type, &mut budget).await {
                ctx.llm_messages[i].content[j] = image;
            }
        }

        debug!(
            "InlineArtifacts: inlined {} images, {} bytes left",
            budget.inlined.len(),
            budget.bytes
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::LocalArtifactStore;
    use crate::config::AgentConfig;
    use crate::models::{LlmMessage, Message};

    fn jpeg(tag: &[u8]) -> Vec<u8> {
        [&[0xFF, 0xD8, 0xFF][..], tag].concat()
    }

    async fn setup(
        files: &[(Vec<u8>, &str)],
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

    fn history_ctx(history: Vec<LlmMessage>, artifact_data: serde_json::Value) -> Context {
        let mut msg = Message::new("look", "u1", "space1");
        msg.metadata
            .insert(ARTIFACT_DATA_METADATA_KEY.into(), artifact_data);
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));
        ctx.llm_messages = std::iter::once(LlmMessage::system("sys"))
            .chain(history)
            .chain([LlmMessage::system("context"), LlmMessage::user("look")])
            .collect();
        ctx
    }

    fn with_ref(role: Role, text: &str, id: &str) -> LlmMessage {
        LlmMessage::with_parts(
            role,
            vec![
                ContentPart::text(text),
                ContentPart::file(ContentSource::Uri { uri: id.into() }, "image/png", None),
            ],
        )
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
        let (_tmp, store, ids) = setup(&[(jpeg(&[1, 2, 3]), "image/jpeg")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0], "mime_type": "image/jpeg"}]));

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(inline_images(&ctx, 1), vec![jpeg(&[1, 2, 3])]);
        assert!(references(&ctx, 1).is_empty());
    }

    #[tokio::test]
    async fn it_replaces_the_reference_attach_media_added() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[7]), "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0], "mime_type": "image/png"}]));
        super::super::AttachMedia.process(&mut ctx).await.unwrap();
        assert_eq!(references(&ctx, 1), vec![ids[0].clone()]);

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(inline_images(&ctx, 1), vec![jpeg(&[7])]);
        assert!(references(&ctx, 1).is_empty());
    }

    #[tokio::test]
    async fn what_cannot_be_inlined_stays_a_reference() {
        let (_tmp, store, ids) =
            setup(&[(vec![1], "application/pdf"), (jpeg(&[2; 10]), "image/png")]).await;
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
        let (_tmp, store, ids) =
            setup(&[(jpeg(&[1; 3]), "image/png"), (jpeg(&[2; 3]), "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0]}, {"id": ids[1]}]));

        InlineArtifacts::new(store)
            .with_max_bytes(11)
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(inline_images(&ctx, 1), vec![jpeg(&[1; 3])]);
        assert_eq!(references(&ctx, 1), vec![ids[1].clone()]);
    }

    #[tokio::test]
    async fn at_most_max_images_are_inlined() {
        let files: Vec<(Vec<u8>, &str)> = (0..MAX_REINJECTED_ARTIFACTS as u8 + 2)
            .map(|i| (jpeg(&[i]), "image/png"))
            .collect();
        let (_tmp, store, ids) = setup(&files).await;
        let data: Vec<_> = ids.iter().map(|id| serde_json::json!({"id": id})).collect();

        let mut ctx = ctx_with(serde_json::Value::Array(data.clone()));
        InlineArtifacts::new(store.clone())
            .process(&mut ctx)
            .await
            .unwrap();
        assert_eq!(inline_images(&ctx, 1).len(), MAX_REINJECTED_ARTIFACTS);
        assert_eq!(references(&ctx, 1).len(), 2);

        let mut ctx = ctx_with(serde_json::Value::Array(data));
        InlineArtifacts::new(store)
            .with_max_images(3)
            .process(&mut ctx)
            .await
            .unwrap();
        assert_eq!(inline_images(&ctx, 1).len(), 3);
    }

    #[tokio::test]
    async fn it_attaches_to_the_current_user_turn_not_the_last() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[9]), "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0]}]));
        ctx.llm_messages
            .push(LlmMessage::user("a later injected note"));
        ctx.set(CurrentUserMessage(1));

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(inline_images(&ctx, 1), vec![jpeg(&[9])]);
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

    #[tokio::test]
    async fn earlier_messages_are_left_alone_by_default() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[1]), "image/png")]).await;
        let mut ctx = history_ctx(
            vec![with_ref(Role::User, "old", &ids[0])],
            serde_json::json!([]),
        );

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(references(&ctx, 1), vec![ids[0].clone()]);
    }

    #[tokio::test]
    async fn references_inside_the_history_window_are_inlined_in_place() {
        let (_tmp, store, ids) =
            setup(&[(jpeg(&[1]), "image/png"), (jpeg(&[2]), "image/png")]).await;
        let mut ctx = history_ctx(
            vec![
                with_ref(Role::User, "too old", &ids[0]),
                LlmMessage::assistant("ok"),
                with_ref(Role::User, "recent", &ids[1]),
            ],
            serde_json::json!([]),
        );

        InlineArtifacts::new(store)
            .with_history_messages(2)
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(references(&ctx, 1), vec![ids[0].clone()]);
        assert_eq!(inline_images(&ctx, 3), vec![jpeg(&[2])]);
        assert_eq!(ctx.llm_messages[3].text(), "recent");
    }

    #[tokio::test]
    async fn the_current_turn_has_first_claim_on_the_budget() {
        let (_tmp, store, ids) =
            setup(&[(jpeg(&[1]), "image/png"), (jpeg(&[2]), "image/png")]).await;
        let mut ctx = history_ctx(
            vec![with_ref(Role::User, "earlier", &ids[0])],
            serde_json::json!([{"id": ids[1]}]),
        );

        InlineArtifacts::new(store)
            .with_history_messages(5)
            .with_max_images(1)
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(inline_images(&ctx, 3), vec![jpeg(&[2])]);
        assert_eq!(references(&ctx, 1), vec![ids[0].clone()]);
    }

    #[tokio::test]
    async fn an_image_referenced_twice_is_inlined_once() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[4]), "image/png")]).await;
        let mut ctx = history_ctx(
            vec![
                with_ref(Role::User, "first", &ids[0]),
                with_ref(Role::User, "again", &ids[0]),
            ],
            serde_json::json!([]),
        );

        InlineArtifacts::new(store)
            .with_history_messages(2)
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(inline_images(&ctx, 2), vec![jpeg(&[4])]);
        assert_eq!(references(&ctx, 1), vec![ids[0].clone()]);
    }

    #[tokio::test]
    async fn assistant_turns_keep_their_references() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[5]), "image/png")]).await;
        let mut ctx = history_ctx(
            vec![with_ref(Role::Assistant, "mine", &ids[0])],
            serde_json::json!([]),
        );

        InlineArtifacts::new(store)
            .with_history_messages(3)
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(references(&ctx, 1), vec![ids[0].clone()]);
    }

    #[test]
    fn only_formats_vision_endpoints_accept_are_sniffed_as_images() {
        let png = [
            0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13, b'I', b'H', b'D', b'R',
        ];
        assert_eq!(sniff_image(&png), Some("image/png"));
        assert_eq!(sniff_image(&jpeg(&[])), Some("image/jpeg"));
        assert_eq!(sniff_image(b"GIF89a..."), Some("image/gif"));
        assert_eq!(sniff_image(b"RIFF\0\0\0\0WEBPVP8 "), Some("image/webp"));
        assert_eq!(sniff_image(b"\x89PNG\r\n\x1a\nnot really a png"), None);
        assert_eq!(sniff_image(b"\0\0\0\x18ftypheic"), None);
        assert_eq!(
            sniff_image(b"<svg xmlns=\"http://www.w3.org/2000/svg\"/>"),
            None
        );
    }

    #[tokio::test]
    async fn a_declared_image_the_endpoint_would_reject_stays_a_reference() {
        let (_tmp, store, ids) = setup(&[
            (b"\0\0\0\x18ftypheic".to_vec(), "image/heic"),
            (b"\x89PNG\r\n\x1a\nnot really a png".to_vec(), "image/png"),
        ])
        .await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0]}, {"id": ids[1]}]));

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert!(inline_images(&ctx, 1).is_empty());
        assert_eq!(references(&ctx, 1), ids);
    }
}
