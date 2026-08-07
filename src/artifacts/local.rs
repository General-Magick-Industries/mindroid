//! On-disk [`ArtifactStore`]: `<base>/<scope>/<id>` bytes + `<id>.json` sidecar.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{Artifact, ArtifactStore, StoredArtifact};
use crate::error::{MindroidError, Result};

#[derive(Serialize, Deserialize)]
struct Sidecar {
    mime_type: String,
}

/// Stores artifacts on the local filesystem under a base directory.
///
/// Layout: `<base>/<scope>/<id>` holds the raw bytes, `<base>/<scope>/<id>.json`
/// holds the mime sidecar. Ids are opaque UUIDs.
pub struct LocalArtifactStore {
    base: PathBuf,
}

impl LocalArtifactStore {
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    /// Validate a single path component (scope or id): no separators, no `..`, no
    /// absolute markers, no null bytes, non-empty. This is the path-jail guard —
    /// `canonicalize` alone fails for not-yet-existing files.
    fn safe_component(label: &str, s: &str) -> Result<()> {
        if s.is_empty() {
            return Err(MindroidError::artifact(format!("empty {label}")));
        }
        if s.contains('\0') {
            return Err(MindroidError::artifact(format!(
                "{label} contains null byte"
            )));
        }
        if s == "." || s == ".." {
            return Err(MindroidError::artifact(format!(
                "{label} is a path-relative component"
            )));
        }
        // Reject any path separator or parent-dir traversal.
        if s.contains('/') || s.contains('\\') {
            return Err(MindroidError::artifact(format!(
                "{label} contains a path separator"
            )));
        }
        if Path::new(s).is_absolute() {
            return Err(MindroidError::artifact(format!(
                "{label} is an absolute path"
            )));
        }
        Ok(())
    }

    fn scope_dir(&self, scope: &str) -> Result<PathBuf> {
        Self::safe_component("scope", scope)?;
        Ok(self.base.join(scope))
    }

    fn artifact_path(&self, scope: &str, id: &str) -> Result<PathBuf> {
        Self::safe_component("id", id)?;
        Ok(self.scope_dir(scope)?.join(id))
    }
}

#[async_trait]
impl ArtifactStore for LocalArtifactStore {
    async fn save(&self, scope: &str, data: &[u8], mime_type: &str) -> Result<StoredArtifact> {
        let id = Uuid::new_v4().to_string();
        let dir = self.scope_dir(scope)?;
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| MindroidError::artifact(format!("create_dir_all failed: {e}")))?;

        let bytes_path = dir.join(&id);
        let sidecar_path = dir.join(format!("{id}.json"));

        tokio::fs::write(&bytes_path, data)
            .await
            .map_err(|e| MindroidError::artifact(format!("write bytes failed: {e}")))?;

        let sidecar = Sidecar {
            mime_type: mime_type.to_string(),
        };
        let json = serde_json::to_vec(&sidecar)
            .map_err(|e| MindroidError::artifact(format!("serialize sidecar failed: {e}")))?;
        tokio::fs::write(&sidecar_path, json)
            .await
            .map_err(|e| MindroidError::artifact(format!("write sidecar failed: {e}")))?;

        // LocalArtifactStore is a plain store: just the id, no metadata of its own.
        Ok(StoredArtifact::new(id))
    }

    async fn load(&self, scope: &str, id: &str) -> Result<Artifact> {
        let bytes_path = self.artifact_path(scope, id)?;
        let sidecar_path = self.scope_dir(scope)?.join(format!("{id}.json"));

        let data = tokio::fs::read(&bytes_path)
            .await
            .map_err(|e| MindroidError::artifact(format!("artifact '{id}' not found: {e}")))?;
        let json = tokio::fs::read(&sidecar_path)
            .await
            .map_err(|e| MindroidError::artifact(format!("sidecar for '{id}' not found: {e}")))?;
        let sidecar: Sidecar = serde_json::from_slice(&json)
            .map_err(|e| MindroidError::artifact(format!("parse sidecar failed: {e}")))?;

        Ok(Artifact {
            data,
            mime_type: sidecar.mime_type,
        })
    }

    async fn delete(&self, scope: &str, id: &str) -> Result<()> {
        let bytes_path = self.artifact_path(scope, id)?;
        let sidecar_path = self.scope_dir(scope)?.join(format!("{id}.json"));
        // Idempotent: ignore not-found.
        let _ = tokio::fs::remove_file(&bytes_path).await;
        let _ = tokio::fs::remove_file(&sidecar_path).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn save_load_delete_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LocalArtifactStore::new(tmp.path());

        let stored = store
            .save("chan1", &[1, 2, 3, 4], "image/png")
            .await
            .unwrap();
        // The plain store returns just the id, no metadata.
        assert!(stored.metadata.is_empty());
        let id = stored.id;

        let art = store.load("chan1", &id).await.unwrap();
        assert_eq!(art.data, vec![1, 2, 3, 4]);
        assert_eq!(art.mime_type, "image/png");

        store.delete("chan1", &id).await.unwrap();
        assert!(store.load("chan1", &id).await.is_err());
        // Delete is idempotent.
        store.delete("chan1", &id).await.unwrap();
    }

    #[tokio::test]
    async fn scopes_isolate() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LocalArtifactStore::new(tmp.path());
        let id = store.save("a", &[9], "image/png").await.unwrap().id;
        // Same id under a different scope must not resolve.
        assert!(store.load("b", &id).await.is_err());
        assert!(store.load("a", &id).await.is_ok());
    }

    #[tokio::test]
    async fn path_jail_rejects_traversal() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LocalArtifactStore::new(tmp.path());
        assert!(store.load("..", "x").await.is_err());
        assert!(store.load("chan", "../escape").await.is_err());
        assert!(store.load("chan", "a/b").await.is_err());
        assert!(store.save("..", &[1], "image/png").await.is_err());
        assert!(store.load("chan", "with\0null").await.is_err());
    }
}
