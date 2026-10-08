//! Inline the images a sender attached by reference, so the model sees them on
//! the turn they arrive instead of calling `get_artifact`.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use tracing::debug;

use crate::artifacts::{ArtifactManager, ArtifactStore, INLINED_ARTIFACT_KEY, inline_image_type};
use crate::core::content::{
    ARTIFACT_DATA_METADATA_KEY, ArtifactReference, ContentPart, ContentSource, is_artifact_id,
};
use crate::core::context::Context;
use crate::llm_client::{render_llm_metadata, sanitize_llm_visible};
use crate::models::Role;
use crate::pipeline::extensions::CurrentUserMessage;
use crate::pipeline::stages::tool_executor_xml::MAX_REINJECTED_ARTIFACTS;
use crate::{PipelineStage, Result};

/// Default cap on the image bytes a turn's requests carry: base64 grows it by a
/// third, and a 1 MiB request cap must also fit the prompt and history. The
/// executors' `get_artifact` re-attachment spends what this stage leaves.
pub const DEFAULT_MAX_INLINE_BYTES: usize = 512 * 1024;

/// The image bytes this turn's requests may carry, set by [`InlineArtifacts`]
/// from [`with_max_bytes`](InlineArtifacts::with_max_bytes). Without it, the
/// executors re-attach with no byte bound.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ImageAllowance(pub(crate) usize);

/// Loads the images in the inbound message's `artifact_data` and attaches them
/// inline to the current user turn; with
/// [`with_history_messages`](Self::with_history_messages) it also inlines the
/// references in that many earlier messages, newest first. The current turn has
/// first claim on [`with_max_images`](Self::with_max_images) and
/// [`with_max_bytes`](Self::with_max_bytes); anything else stays a reference for
/// `get_artifact`.
///
/// Run it after the user turn is built; it replaces the references
/// [`AttachMedia`](super::AttachMedia) added for the same ids. An
/// [`ArtifactOffload`](super::ArtifactOffload) placed after it turns the images
/// back into references to the senders' artifacts rather than storing them again.
pub struct InlineArtifacts {
    store: Arc<dyn ArtifactStore>,
    max_bytes: usize,
    max_images: usize,
    history_messages: usize,
}

struct Budget {
    bytes: usize,
    loads: usize,
    attempted: HashSet<String>,
    inlined: usize,
}

impl InlineArtifacts {
    /// Inline from `store`, with the default budgets and no history window.
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

    /// Cap the image bytes a turn's requests carry: what this stage inlines, plus
    /// what both tool executors re-attach for `get_artifact` (anything they load
    /// counts, image or not). Defaults to [`DEFAULT_MAX_INLINE_BYTES`].
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Cap the artifacts loaded in one turn, and so the images inlined; a load
    /// that cannot be inlined still counts. Defaults to 8.
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

    async fn resolve(
        &self,
        scope: &str,
        part: ContentPart,
        budget: &mut Budget,
    ) -> Vec<ContentPart> {
        let ContentPart::File {
            source: ContentSource::Uri { uri },
            mime_type,
            filename,
            metadata,
        } = &part
        else {
            return vec![part];
        };
        let maybe_image =
            mime_type == "application/octet-stream" || mime_type.starts_with("image/");
        if !maybe_image
            || !is_artifact_id(uri)
            || budget.loads == 0
            || budget.bytes == 0
            || !budget.attempted.insert(uri.clone())
        {
            return vec![part];
        }
        budget.loads -= 1;
        let artifact = match self.store.load_bounded(scope, uri, budget.bytes).await {
            Ok(artifact) => artifact,
            Err(e) => {
                debug!("InlineArtifacts: keeping '{uri}' as a reference: {e}");
                return vec![part];
            }
        };
        let Some(image_type) = inline_image_type(&artifact.data) else {
            debug!(
                "InlineArtifacts: '{uri}' ({}) is not an image the model accepts; keeping the reference",
                artifact.mime_type
            );
            return vec![part];
        };
        let Some(left) = budget.bytes.checked_sub(artifact.data.len()) else {
            return vec![part];
        };
        budget.bytes = left;
        budget.inlined += 1;

        let name = filename
            .as_deref()
            .map(sanitize_llm_visible)
            .filter(|f| !f.is_empty())
            .map(|f| format!(" \"{f}\""))
            .unwrap_or_default();
        let label = format!(
            "[{image_type} artifact {}{name}{}]",
            sanitize_llm_visible(uri),
            render_llm_metadata(metadata)
        );
        let mut image = ContentPart::image(
            ContentSource::Inline {
                data: artifact.data,
            },
            image_type,
        );
        if let Some(image_metadata) = image.metadata_mut() {
            image_metadata.insert(INLINED_ARTIFACT_KEY.into(), uri.clone().into());
        }
        vec![ContentPart::text(label), image]
    }
}

fn references_one_of(part: &ContentPart, ids: &HashSet<String>) -> bool {
    matches!(part, ContentPart::File { source: ContentSource::Uri { uri }, .. } if ids.contains(uri))
}

#[async_trait]
impl PipelineStage for InlineArtifacts {
    fn name(&self) -> &str {
        "InlineArtifacts"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        ctx.set(ImageAllowance(self.max_bytes));
        let mut ids = HashSet::new();
        let references: Vec<ContentPart> = ctx
            .message
            .metadata
            .get(ARTIFACT_DATA_METADATA_KEY)
            .map(ArtifactReference::parts_from_value)
            .unwrap_or_default()
            .into_iter()
            .filter(|part| match part {
                ContentPart::File {
                    source: ContentSource::Uri { uri },
                    ..
                } => ids.insert(uri.clone()),
                _ => false,
            })
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
            loads: self.max_images,
            attempted: HashSet::new(),
            inlined: 0,
        };

        if !references.is_empty() {
            let mut parts = Vec::with_capacity(references.len());
            for reference in references {
                parts.extend(self.resolve(&scope, reference, &mut budget).await);
            }
            for message in ctx.llm_messages[index..]
                .iter_mut()
                .filter(|m| m.role == Role::User)
            {
                message.content.retain(|p| !references_one_of(p, &ids));
            }
            ctx.llm_messages[index].content.extend(parts);
        }

        let earlier: Vec<usize> = ctx.llm_messages[..index]
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, m)| m.role != Role::System)
            .take(self.history_messages)
            .filter(|(_, m)| m.role == Role::User)
            .map(|(i, _)| i)
            .collect();
        for i in earlier {
            let content = std::mem::take(&mut ctx.llm_messages[i].content);
            let mut resolved = Vec::with_capacity(content.len());
            for part in content {
                resolved.extend(self.resolve(&scope, part, &mut budget).await);
            }
            ctx.llm_messages[i].content = resolved;
        }

        debug!(
            "InlineArtifacts: inlined {} images, {} bytes left",
            budget.inlined, budget.bytes
        );
        if budget.inlined > 0 {
            match ctx.llm_messages.iter_mut().find(|m| m.role == Role::System) {
                Some(system) => system.append_text(&format!("\n\n{CAN_SEE_IMAGES}")),
                None => {
                    ctx.llm_messages
                        .insert(0, crate::LlmMessage::system(CAN_SEE_IMAGES));
                    if current.is_some() {
                        ctx.set(CurrentUserMessage(index + 1));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Persona prompts lead some vision models (gpt-4o-mini among them) to answer
/// "I can't see images" about a photo in the same request.
const CAN_SEE_IMAGES: &str = "You can see images. When a message has an image attached, it is \
    shown to you in that message: look at it and answer from it directly.";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::{Artifact, LocalArtifactStore, StoredArtifact};
    use crate::config::AgentConfig;
    use crate::models::{LlmMessage, Message};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn jpeg(tag: &[u8]) -> Vec<u8> {
        [&[0xFF, 0xD8, 0xFF][..], tag].concat()
    }

    struct Counting {
        inner: LocalArtifactStore,
        loads: AtomicUsize,
        saves: AtomicUsize,
    }

    #[async_trait]
    impl ArtifactStore for Counting {
        async fn save(&self, scope: &str, data: &[u8], mime_type: &str) -> Result<StoredArtifact> {
            self.saves.fetch_add(1, Ordering::SeqCst);
            self.inner.save(scope, data, mime_type).await
        }
        async fn load(&self, scope: &str, id: &str) -> Result<Artifact> {
            self.loads.fetch_add(1, Ordering::SeqCst);
            self.inner.load(scope, id).await
        }
        async fn delete(&self, scope: &str, id: &str) -> Result<()> {
            self.inner.delete(scope, id).await
        }
    }

    async fn setup(files: &[(Vec<u8>, &str)]) -> (tempfile::TempDir, Arc<Counting>, Vec<String>) {
        let tmp = tempfile::tempdir().unwrap();
        let inner = LocalArtifactStore::new(tmp.path());
        let mut ids = Vec::new();
        for (data, mime) in files {
            ids.push(inner.save("space1", data, mime).await.unwrap().id);
        }
        let store = Arc::new(Counting {
            inner,
            loads: AtomicUsize::new(0),
            saves: AtomicUsize::new(0),
        });
        (tmp, store, ids)
    }

    fn ctx_with(artifact_data: serde_json::Value) -> Context {
        history_ctx(vec![], artifact_data)
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

    fn current(ctx: &Context) -> usize {
        ctx.llm_messages.len() - 1
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
    async fn an_attached_image_reaches_the_model_inline_with_its_label() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[1, 2, 3]), "image/jpeg")]).await;
        let mut ctx = ctx_with(serde_json::json!([{
            "id": ids[0], "mime_type": "image/jpeg", "file_name": "desk.jpg",
            "metadata": {"caption": "my desk"}
        }]));

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        let turn = current(&ctx);
        assert_eq!(inline_images(&ctx, turn), vec![jpeg(&[1, 2, 3])]);
        assert!(references(&ctx, turn).is_empty());
        let label = ctx.llm_messages[turn].text();
        assert!(label.contains(&ids[0]), "{label}");
        assert!(
            label.contains("desk.jpg") && label.contains("my desk"),
            "{label}"
        );
        assert!(!label.contains("get_artifact"), "{label}");
    }

    #[tokio::test]
    async fn the_inserted_system_note_keeps_the_current_turn_pointed_at() {
        use crate::pipeline::extensions::PersistedUserTurn;

        let (_tmp, store, ids) = setup(&[(jpeg(&[1]), "image/jpeg")]).await;
        let mut msg = Message::new("mine", "u1", "space1");
        msg.metadata.insert(
            ARTIFACT_DATA_METADATA_KEY.into(),
            serde_json::json!([{"id": ids[0]}]),
        );
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));
        ctx.llm_messages = vec![LlmMessage::user("someone else's"), LlmMessage::user("mine")];
        ctx.set(CurrentUserMessage(1));

        InlineArtifacts::new(store.clone())
            .process(&mut ctx)
            .await
            .unwrap();
        crate::pipeline::stages::ArtifactOffload::new(store)
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(ctx.get_run::<CurrentUserMessage>().map(|c| c.0), Some(2));
        let persisted = &ctx.get_run::<PersistedUserTurn>().unwrap().0;
        assert!(
            persisted.contains("mine") && persisted.contains(&ids[0]),
            "{persisted}"
        );
        assert!(!persisted.contains("someone else"), "{persisted}");
    }

    #[tokio::test]
    async fn without_a_system_prompt_the_note_gets_its_own_not_the_users_turn() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[1]), "image/jpeg")]).await;
        let mut msg = Message::new("look", "u1", "space1");
        msg.metadata.insert(
            ARTIFACT_DATA_METADATA_KEY.into(),
            serde_json::json!([{"id": ids[0]}]),
        );
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));
        ctx.llm_messages = vec![LlmMessage::user("look")];

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(ctx.llm_messages[0].role, Role::System);
        assert_eq!(ctx.llm_messages[0].text(), CAN_SEE_IMAGES);
        assert!(!ctx.llm_messages[1].text().contains(CAN_SEE_IMAGES));
        assert_eq!(inline_images(&ctx, 1), vec![jpeg(&[1])]);
    }

    #[tokio::test]
    async fn the_model_is_told_it_can_see_only_when_an_image_was_inlined() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[1]), "image/jpeg")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0]}]));
        InlineArtifacts::new(store.clone())
            .process(&mut ctx)
            .await
            .unwrap();
        assert!(ctx.llm_messages[0].text().ends_with(CAN_SEE_IMAGES));
        assert!(!ctx.llm_messages[1].text().contains(CAN_SEE_IMAGES));

        let mut ctx = ctx_with(serde_json::json!([{"id": "missing"}]));
        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();
        assert!(
            ctx.llm_messages
                .iter()
                .all(|m| !m.text().contains(CAN_SEE_IMAGES))
        );
    }

    #[tokio::test]
    async fn it_replaces_the_reference_attach_media_added() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[7]), "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0], "mime_type": "image/png"}]));
        super::super::AttachMedia.process(&mut ctx).await.unwrap();
        assert_eq!(references(&ctx, current(&ctx)), vec![ids[0].clone()]);

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(inline_images(&ctx, current(&ctx)), vec![jpeg(&[7])]);
        assert!(references(&ctx, current(&ctx)).is_empty());
    }

    #[tokio::test]
    async fn attach_media_references_on_a_later_turn_are_removed_too() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[8]), "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0]}]));
        let turn = current(&ctx);
        ctx.llm_messages
            .push(LlmMessage::user("a later injected note"));
        ctx.set(CurrentUserMessage(turn));
        super::super::AttachMedia.process(&mut ctx).await.unwrap();

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(inline_images(&ctx, turn), vec![jpeg(&[8])]);
        assert!(references(&ctx, turn + 1).is_empty());
    }

    #[tokio::test]
    async fn what_cannot_be_inlined_stays_a_reference_and_non_images_are_not_loaded() {
        let (_tmp, store, ids) =
            setup(&[(vec![1], "application/pdf"), (jpeg(&[2; 10]), "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([
            {"id": ids[0], "mime_type": "application/pdf"},
            {"id": ids[1], "mime_type": "image/png"},
            {"id": "missing", "mime_type": "image/png"},
        ]));

        InlineArtifacts::new(store.clone())
            .with_max_bytes(4)
            .process(&mut ctx)
            .await
            .unwrap();

        assert!(inline_images(&ctx, current(&ctx)).is_empty());
        assert_eq!(
            references(&ctx, current(&ctx)),
            vec![ids[0].clone(), ids[1].clone(), "missing".to_string()]
        );
        assert_eq!(
            store.loads.load(Ordering::SeqCst),
            2,
            "the PDF is never loaded"
        );
    }

    #[tokio::test]
    async fn an_oversized_image_does_not_stop_a_later_one_that_fits() {
        let (_tmp, store, ids) =
            setup(&[(jpeg(&[1; 20]), "image/png"), (jpeg(&[2]), "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0]}, {"id": ids[1]}]));

        InlineArtifacts::new(store)
            .with_max_bytes(10)
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(inline_images(&ctx, current(&ctx)), vec![jpeg(&[2])]);
        assert_eq!(references(&ctx, current(&ctx)), vec![ids[0].clone()]);
    }

    #[tokio::test]
    async fn every_load_counts_against_the_cap_even_when_nothing_inlines() {
        let heic = b"\0\0\0\x18ftypheic".to_vec();
        let files: Vec<(Vec<u8>, &str)> = (0..10).map(|_| (heic.clone(), "image/heic")).collect();
        let (_tmp, store, ids) = setup(&files).await;
        let data: Vec<_> = ids.iter().map(|id| serde_json::json!({"id": id})).collect();
        let mut ctx = ctx_with(serde_json::Value::Array(data));

        InlineArtifacts::new(store.clone())
            .with_max_images(3)
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(store.loads.load(Ordering::SeqCst), 3);
        assert_eq!(references(&ctx, current(&ctx)), ids);
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
        assert_eq!(
            inline_images(&ctx, current(&ctx)).len(),
            MAX_REINJECTED_ARTIFACTS
        );
        assert_eq!(references(&ctx, current(&ctx)).len(), 2);

        let mut ctx = ctx_with(serde_json::Value::Array(data));
        InlineArtifacts::new(store)
            .with_max_images(3)
            .process(&mut ctx)
            .await
            .unwrap();
        assert_eq!(inline_images(&ctx, current(&ctx)).len(), 3);
    }

    #[tokio::test]
    async fn a_repeated_reference_is_loaded_and_shown_once() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[3]), "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0]}, {"id": ids[0]}, {"id": ids[0]}]));

        InlineArtifacts::new(store.clone())
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(inline_images(&ctx, current(&ctx)), vec![jpeg(&[3])]);
        assert!(references(&ctx, current(&ctx)).is_empty());
        assert_eq!(store.loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn it_attaches_to_the_current_user_turn_not_the_last() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[9]), "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0]}]));
        let turn = current(&ctx);
        ctx.llm_messages
            .push(LlmMessage::user("a later injected note"));
        ctx.set(CurrentUserMessage(turn));

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(inline_images(&ctx, turn), vec![jpeg(&[9])]);
        assert!(inline_images(&ctx, turn + 1).is_empty());
    }

    #[tokio::test]
    async fn a_current_marker_on_a_non_user_turn_falls_back_to_the_last_user_turn() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[6]), "image/png")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0]}]));
        ctx.set(CurrentUserMessage(0));

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert_eq!(inline_images(&ctx, current(&ctx)), vec![jpeg(&[6])]);
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

        InlineArtifacts::new(store.clone())
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(references(&ctx, 1), vec![ids[0].clone()]);
        assert_eq!(store.loads.load(Ordering::SeqCst), 0);
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
        assert!(ctx.llm_messages[3].text().starts_with("recent"));
    }

    #[tokio::test]
    async fn a_missing_artifact_in_history_stays_a_reference() {
        let (_tmp, store, _) = setup(&[]).await;
        let mut ctx = history_ctx(
            vec![with_ref(Role::User, "gone", "missing")],
            serde_json::json!([]),
        );

        InlineArtifacts::new(store)
            .with_history_messages(1)
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(references(&ctx, 1), vec!["missing".to_string()]);
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

        assert_eq!(inline_images(&ctx, current(&ctx)), vec![jpeg(&[2])]);
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

        InlineArtifacts::new(store.clone())
            .with_history_messages(2)
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(inline_images(&ctx, 2), vec![jpeg(&[4])]);
        assert_eq!(references(&ctx, 1), vec![ids[0].clone()]);
        assert_eq!(store.loads.load(Ordering::SeqCst), 1);
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

    #[tokio::test]
    async fn a_declared_image_the_endpoint_would_reject_stays_a_reference() {
        let (_tmp, store, ids) = setup(&[
            (b"\0\0\0\x18ftypheic".to_vec(), "image/heic"),
            (b"\x89PNG\r\n\x1a\nnot really a png".to_vec(), "image/png"),
        ])
        .await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0]}, {"id": ids[1]}]));

        InlineArtifacts::new(store).process(&mut ctx).await.unwrap();

        assert!(inline_images(&ctx, current(&ctx)).is_empty());
        assert_eq!(references(&ctx, current(&ctx)), ids);
    }

    #[tokio::test]
    async fn a_later_offload_restores_the_senders_reference_instead_of_storing_again() {
        let (_tmp, store, ids) = setup(&[(jpeg(&[1]), "image/jpeg")]).await;
        let mut ctx = ctx_with(serde_json::json!([{"id": ids[0], "mime_type": "image/jpeg"}]));

        InlineArtifacts::new(store.clone())
            .process(&mut ctx)
            .await
            .unwrap();
        crate::pipeline::stages::ArtifactOffload::new(store.clone())
            .process(&mut ctx)
            .await
            .unwrap();

        assert_eq!(store.saves.load(Ordering::SeqCst), 0);
        assert_eq!(references(&ctx, current(&ctx)), vec![ids[0].clone()]);
        assert!(inline_images(&ctx, current(&ctx)).is_empty());
    }
}
