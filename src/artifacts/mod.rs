//! Pluggable out-of-band artifact storage.
//!
//! When media is offloaded from the conversation, its bytes are saved here and
//! the chat history keeps only a compact reference (an id). The model re-fetches
//! the bytes on demand via the `get_artifact` tool.
//!
//! The store is a swappable trait (mirroring `Memory`/`Auth`): [`NoArtifactStore`]
//! (no-op default) and [`LocalArtifactStore`] (on-disk) ship here; a remote
//! backend slots in later behind the same trait. The factory returns
//! `Arc<dyn ArtifactStore>` because the store is shared into BOTH the persistence
//! stage AND the load path.

mod local;
#[cfg(feature = "magickmind")]
mod magickmind;
mod manager;

pub use local::LocalArtifactStore;
#[cfg(feature = "magickmind")]
pub use magickmind::MagickmindArtifactStore;
pub use manager::ArtifactManager;

use async_trait::async_trait;
use std::sync::Arc;

use crate::error::{MindroidError, Result};
use crate::models::Message;

/// Metadata key an inlined image carries its artifact id under, so offload can
/// restore the reference instead of storing the bytes again.
pub(crate) const INLINED_ARTIFACT_KEY: &str = "_artifact_id";

pub use crate::core::content::inline_image_type;

/// Bytes + metadata returned by [`ArtifactStore::load`].
#[derive(Debug, Clone)]
pub struct Artifact {
    pub data: Vec<u8>,
    pub mime_type: String,
}

/// The result of [`ArtifactStore::save`]: the reference id plus any metadata the
/// store chose to attach (a caption, backend facts like an S3 etag/region, a
/// content hash, extracted entities…).
///
/// A store that doesn't care about enrichment returns [`StoredArtifact::new(id)`],
/// which leaves `metadata` empty — and because empty metadata is omitted from
/// serialization downstream, a minimal store costs nothing extra in storage or
/// tokens. To surface a caption to the model, put it in `metadata` (visible by
/// default); prefix a key with `_` to keep it code-only.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredArtifact {
    pub id: String,
    pub metadata: crate::core::content::ContentMetadata,
}

impl StoredArtifact {
    /// The zero-cost path: just an id, no metadata.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            metadata: Default::default(),
        }
    }

    /// Attach a metadata map (caption, backend facts, entities, hashes…).
    pub fn with_metadata(mut self, metadata: crate::core::content::ContentMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl From<String> for StoredArtifact {
    fn from(id: String) -> Self {
        Self::new(id)
    }
}

/// A pluggable store for binary artifacts, keyed by `(scope, id)`.
///
/// **Security:** `scope` MUST be supplied from trusted session context (e.g.
/// `ctx.message.channel_id`), never from model- or user-supplied input. The store
/// enforces path containment but NOT tenant authorization.
#[async_trait]
pub trait ArtifactStore: Send + Sync + 'static {
    /// Save bytes under `scope`, returning a [`StoredArtifact`] — at minimum a
    /// stable opaque id (`StoredArtifact::new(id)`), optionally with metadata the
    /// store chose to attach (a caption, backend facts, entities…).
    async fn save(&self, scope: &str, data: &[u8], mime_type: &str) -> Result<StoredArtifact>;

    /// Load raw bytes + metadata back by `(scope, id)`.
    async fn load(&self, scope: &str, id: &str) -> Result<Artifact>;

    /// Like [`load`](Self::load), but fails for an artifact over `max_bytes`. The
    /// default loads it whole and then checks: a store that can stop reading early
    /// should override it, and a store wrapping another should forward it.
    async fn load_bounded(&self, scope: &str, id: &str, max_bytes: usize) -> Result<Artifact> {
        let artifact = self.load(scope, id).await?;
        if artifact.data.len() > max_bytes {
            return Err(MindroidError::artifact(format!(
                "artifact '{id}' exceeds {max_bytes} bytes"
            )));
        }
        Ok(artifact)
    }

    /// Delete an artifact. Idempotent — deleting a missing id is `Ok`.
    async fn delete(&self, scope: &str, id: &str) -> Result<()>;

    /// The scope `message`'s artifacts live under: the trusted delivery channel,
    /// unless the backend re-authorizes every call (see ADR-0009).
    fn scope_for(&self, message: &Message) -> String {
        message.channel_id.clone()
    }
}

/// No-op store: both `save` and `load` error. The default when artifacts are not
/// configured.
///
/// `save` deliberately fails rather than returning a placeholder id: offloading
/// replaces inline bytes with a reference, so a fake-success `save` would drop the
/// bytes on the floor and hand back an id that can never load. Erroring surfaces
/// the misconfiguration at the point it happens.
pub struct NoArtifactStore;

#[async_trait]
impl ArtifactStore for NoArtifactStore {
    async fn save(&self, _scope: &str, _data: &[u8], _mime_type: &str) -> Result<StoredArtifact> {
        Err(crate::error::MindroidError::artifact(
            "NoArtifactStore: cannot save artifact (no store configured) — set \
             [artifacts] in config or omit the ArtifactOffload stage",
        ))
    }

    async fn load(&self, _scope: &str, id: &str) -> Result<Artifact> {
        Err(crate::error::MindroidError::artifact(format!(
            "NoArtifactStore: cannot load artifact '{id}' (no store configured)"
        )))
    }

    async fn delete(&self, _scope: &str, _id: &str) -> Result<()> {
        Ok(())
    }
}

// Arc blanket impl — share one store into the stage and the tool (mirror `Auth`).
#[async_trait]
impl<T: ArtifactStore> ArtifactStore for Arc<T> {
    async fn save(&self, scope: &str, data: &[u8], mime_type: &str) -> Result<StoredArtifact> {
        (**self).save(scope, data, mime_type).await
    }

    async fn load(&self, scope: &str, id: &str) -> Result<Artifact> {
        (**self).load(scope, id).await
    }

    async fn load_bounded(&self, scope: &str, id: &str, max_bytes: usize) -> Result<Artifact> {
        (**self).load_bounded(scope, id, max_bytes).await
    }

    async fn delete(&self, scope: &str, id: &str) -> Result<()> {
        (**self).delete(scope, id).await
    }

    fn scope_for(&self, message: &Message) -> String {
        (**self).scope_for(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn only_formats_vision_endpoints_accept_count_as_images() {
        let png = [
            0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13, b'I', b'H', b'D', b'R',
        ];
        assert_eq!(inline_image_type(&png), Some("image/png"));
        assert_eq!(
            inline_image_type(&[0xFF, 0xD8, 0xFF, 0xE0]),
            Some("image/jpeg")
        );
        assert_eq!(inline_image_type(b"GIF89a..."), Some("image/gif"));
        assert_eq!(
            inline_image_type(b"RIFF\0\0\0\0WEBPVP8 "),
            Some("image/webp")
        );
        assert_eq!(
            inline_image_type(b"\x89PNG\r\n\x1a\nnot really a png"),
            None
        );
        assert_eq!(inline_image_type(b"\0\0\0\x18ftypheic"), None);
        assert_eq!(
            inline_image_type(b"<svg xmlns=\"http://www.w3.org/2000/svg\"/>"),
            None
        );
        assert_eq!(inline_image_type(b""), None);
    }

    #[tokio::test]
    async fn no_store_save_errors_rather_than_fabricating_an_id() {
        // A placeholder id would discard the bytes and never load; the error has
        // to name the fix.
        let err = NoArtifactStore
            .save("chan", b"bytes", "image/png")
            .await
            .expect_err("save must not succeed without a store");
        let msg = err.to_string();
        assert!(msg.contains("no store configured"), "{msg}");
        assert!(
            msg.contains("[artifacts]"),
            "error should name the fix: {msg}"
        );
    }

    #[tokio::test]
    async fn no_store_load_errors_and_names_the_id() {
        let err = NoArtifactStore
            .load("chan", "abc123")
            .await
            .expect_err("load must fail without a store");
        assert!(err.to_string().contains("abc123"));
    }

    /// Delete is idempotent even with nothing behind it, so cleanup paths need
    /// no special-casing for the unconfigured default.
    #[tokio::test]
    async fn no_store_delete_is_ok() {
        assert!(NoArtifactStore.delete("chan", "missing").await.is_ok());
    }

    #[derive(Default)]
    struct CountingStore {
        saves: AtomicUsize,
        loads: AtomicUsize,
        deletes: AtomicUsize,
        last_scope: Mutex<String>,
    }

    #[async_trait]
    impl ArtifactStore for CountingStore {
        async fn save(&self, scope: &str, _d: &[u8], _m: &str) -> Result<StoredArtifact> {
            self.saves.fetch_add(1, Ordering::SeqCst);
            *self.last_scope.lock().unwrap() = scope.to_string();
            Ok(StoredArtifact::new("id-1"))
        }
        async fn load(&self, scope: &str, _id: &str) -> Result<Artifact> {
            self.loads.fetch_add(1, Ordering::SeqCst);
            *self.last_scope.lock().unwrap() = scope.to_string();
            Ok(Artifact {
                data: b"loaded".to_vec(),
                mime_type: "image/png".into(),
            })
        }
        async fn delete(&self, _scope: &str, _id: &str) -> Result<()> {
            self.deletes.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// The blanket impl is what lets one store be shared into both the offload
    /// stage and the load tool. It must forward, not re-wrap.
    #[tokio::test]
    async fn arc_blanket_impl_forwards_every_method_to_the_inner_store() {
        let inner = Arc::new(CountingStore::default());
        let store: Arc<dyn ArtifactStore> = Arc::new(Arc::clone(&inner));

        let saved = store.save("chan-a", b"x", "image/png").await.unwrap();
        assert_eq!(saved.id, "id-1");
        assert_eq!(*inner.last_scope.lock().unwrap(), "chan-a");

        let loaded = store.load("chan-b", "id-1").await.unwrap();
        assert_eq!(loaded.data, b"loaded");
        assert_eq!(*inner.last_scope.lock().unwrap(), "chan-b");

        store.delete("chan-c", "id-1").await.unwrap();

        assert_eq!(inner.saves.load(Ordering::SeqCst), 1);
        assert_eq!(inner.loads.load(Ordering::SeqCst), 1);
        assert_eq!(inner.deletes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stored_artifact_starts_empty_and_takes_metadata() {
        let bare = StoredArtifact::new("id");
        assert!(
            bare.metadata.is_empty(),
            "empty metadata is the zero-cost path"
        );

        let mut m = crate::core::content::ContentMetadata::new();
        m.insert("caption".into(), "a cat".into());
        let enriched = StoredArtifact::new("id").with_metadata(m);
        assert_eq!(enriched.metadata["caption"], "a cat");
        assert_eq!(enriched.id, "id");

        // The From<String> shortcut must agree with the explicit constructor.
        assert_eq!(StoredArtifact::from("id".to_string()), bare);
    }
}
