//! On-disk [`ArtifactStore`]: `<base>/<scope>/<id>` bytes + `<id>.json` sidecar.

use std::path::{Component, Path, PathBuf};

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
        // Must be exactly one ordinary component. Subsumes absolute paths and,
        // on Windows, a drive prefix like `C:evil` — which is not `is_absolute`
        // yet makes `Path::join` discard the base it is joined onto.
        if Path::new(s).components().next() != Some(Component::Normal(s.as_ref())) {
            return Err(MindroidError::artifact(format!(
                "{label} is not a plain path component"
            )));
        }
        Ok(())
    }

    fn scope_dir(&self, scope: &str) -> Result<PathBuf> {
        Self::safe_component("scope", scope)?;
        Ok(self.base.join(scope))
    }

    /// Resolve the bytes + sidecar paths for `<base>/<scope>/<id>`, proving the
    /// scope directory really resolves under the canonical base (a symlinked
    /// scope resolves to its target and is rejected) and that neither target is
    /// itself a symlink. `create` makes the base and scope directories first.
    async fn resolve_paths(
        &self,
        scope: &str,
        id: &str,
        create: bool,
    ) -> Result<(PathBuf, PathBuf)> {
        Self::safe_component("id", id)?;
        let dir = self.scope_dir(scope)?;

        if create {
            // Recursive, so this makes the base too.
            tokio::fs::create_dir_all(&dir)
                .await
                .map_err(|e| MindroidError::artifact(format!("create_dir_all failed: {e}")))?;
        }
        let base = tokio::fs::canonicalize(&self.base)
            .await
            .map_err(|e| MindroidError::artifact(format!("canonicalize base failed: {e}")))?;
        let dir = tokio::fs::canonicalize(&dir)
            .await
            .map_err(|e| MindroidError::artifact(format!("scope '{scope}' not found: {e}")))?;
        if !dir.starts_with(&base) {
            return Err(MindroidError::artifact(format!(
                "scope '{scope}' resolves outside the artifact base"
            )));
        }

        let bytes_path = dir.join(id);
        let sidecar_path = dir.join(format!("{id}.json"));
        if !bytes_path.starts_with(&dir) || !sidecar_path.starts_with(&dir) {
            return Err(MindroidError::artifact(format!(
                "artifact '{id}' resolves outside its scope"
            )));
        }
        Self::reject_symlink(&bytes_path).await?;
        Self::reject_symlink(&sidecar_path).await?;
        Ok((bytes_path, sidecar_path))
    }

    /// Best-effort: this stats before the caller opens the path, so it does not
    /// close the check-then-use race against an attacker who can already write
    /// into the scope dir. Closing that needs `O_NOFOLLOW` at open time.
    async fn reject_symlink(path: &Path) -> Result<()> {
        let is_symlink = tokio::fs::symlink_metadata(path)
            .await
            .is_ok_and(|m| m.file_type().is_symlink());
        if is_symlink {
            return Err(MindroidError::artifact(format!(
                "refusing to follow symlink at '{}'",
                path.display()
            )));
        }
        Ok(())
    }
}

#[async_trait]
impl ArtifactStore for LocalArtifactStore {
    async fn save(&self, scope: &str, data: &[u8], mime_type: &str) -> Result<StoredArtifact> {
        let id = Uuid::new_v4().to_string();
        let (bytes_path, sidecar_path) = self.resolve_paths(scope, &id, true).await?;

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
        let (bytes_path, sidecar_path) = self.resolve_paths(scope, id, false).await?;

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
        // Validate before the early return, so the contract does not depend on
        // whether the scope happens to exist.
        Self::safe_component("id", id)?;
        // Idempotent: a scope that was never created has nothing to delete.
        if tokio::fs::symlink_metadata(self.scope_dir(scope)?)
            .await
            .is_err()
        {
            return Ok(());
        }
        let (bytes_path, sidecar_path) = self.resolve_paths(scope, id, false).await?;
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

    /// `C:evil` is not `is_absolute`, but joining it discards the base entirely,
    /// so an unvalidated id would escape the jail to the drive's working dir.
    #[cfg(windows)]
    #[tokio::test]
    async fn drive_relative_component_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LocalArtifactStore::new(tmp.path());
        store.save("chan", &[1], "image/png").await.unwrap();

        // Assert the guard fired, not merely that something downstream failed:
        // an escaped path errors anyway, which would make this vacuous.
        for e in [
            store.load("chan", "C:evil").await.unwrap_err(),
            store.delete("chan", "C:evil").await.unwrap_err(),
            store.delete("never-created", "C:evil").await.unwrap_err(),
            store.save("C:evil", &[1], "image/png").await.unwrap_err(),
        ] {
            assert!(
                e.to_string().contains("not a plain path component"),
                "expected the component guard to reject it, got: {e}"
            );
        }
    }

    #[cfg(unix)]
    use std::os::unix::fs::{symlink as symlink_dir, symlink as symlink_file};
    #[cfg(windows)]
    use std::os::windows::fs::{symlink_dir, symlink_file};

    /// `(base_dir, outside_dir, store)` where `<base>/evil` symlinks to `outside`,
    /// which holds `secret` + `secret.json`.
    fn symlinked_scope() -> (tempfile::TempDir, tempfile::TempDir, LocalArtifactStore) {
        let base = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret"), b"top secret").unwrap();
        std::fs::write(outside.path().join("secret.json"), br#"{"mime_type":"x"}"#).unwrap();
        symlink_dir(outside.path(), base.path().join("evil")).unwrap();

        let store = LocalArtifactStore::new(base.path());
        (base, outside, store)
    }

    /// Ignored rather than compiled out where symlink creation needs privileges,
    /// so the gap shows up as `ignored` instead of silently vanishing.
    #[cfg_attr(not(unix), ignore = "creating symlinks needs privileges")]
    #[tokio::test]
    async fn symlinked_scope_load_is_rejected() {
        let (_base, _outside, store) = symlinked_scope();
        let res = store.load("evil", "secret").await;
        assert!(res.is_err(), "load through a symlinked scope must fail");
    }

    #[cfg_attr(not(unix), ignore = "creating symlinks needs privileges")]
    #[tokio::test]
    async fn symlinked_scope_save_is_rejected() {
        let (_base, outside, store) = symlinked_scope();
        let before = std::fs::read_dir(outside.path()).unwrap().count();

        assert!(store.save("evil", &[1, 2, 3], "image/png").await.is_err());

        let after = std::fs::read_dir(outside.path()).unwrap().count();
        assert_eq!(before, after, "save must not write outside the base");
    }

    #[cfg_attr(not(unix), ignore = "creating symlinks needs privileges")]
    #[tokio::test]
    async fn symlinked_scope_delete_does_not_touch_outside() {
        let (_base, outside, store) = symlinked_scope();
        assert!(store.delete("evil", "secret").await.is_err());
        assert!(outside.path().join("secret").exists());
        assert!(outside.path().join("secret.json").exists());
    }

    #[cfg_attr(not(unix), ignore = "creating symlinks needs privileges")]
    #[tokio::test]
    async fn symlinked_artifact_file_load_is_rejected() {
        let base = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("secret");
        std::fs::write(&target, b"top secret").unwrap();

        let dir = base.path().join("chan");
        std::fs::create_dir_all(&dir).unwrap();
        symlink_file(&target, dir.join("art")).unwrap();
        std::fs::write(dir.join("art.json"), br#"{"mime_type":"x"}"#).unwrap();

        let store = LocalArtifactStore::new(base.path());
        let res = store.load("chan", "art").await;
        assert!(res.is_err(), "load of a symlinked artifact file must fail");
    }
}
