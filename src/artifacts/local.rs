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

/// Ceiling on a single artifact read. Anyone who can write into the store's
/// directories chooses this size otherwise.
const MAX_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;

/// The sidecar holds one mime type, so it gets a far tighter bound than the
/// bytes it describes — it is fed to a JSON parser.
const MAX_SIDECAR_BYTES: u64 = 64 * 1024;

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
        // Windows resolves these to devices, not files, and opening one can
        // block. Verbatim `\\?\` paths happen to bypass that today, so this
        // does not depend on the canonicalize step staying where it is.
        #[cfg(windows)]
        {
            const EXACT: [&str; 6] = ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"];
            // Devices only when numbered 1-9; bare `COM`/`LPT` are ordinary names.
            const NUMBERED: [&str; 2] = ["COM", "LPT"];
            // Win32 resolves the name before the first `.` or `:` and strips
            // trailing dots and spaces. Compared as bytes so a multi-byte id
            // cannot land mid-codepoint.
            let stem = s
                .split(['.', ':'])
                .next()
                .unwrap_or(s)
                .trim_end_matches([' ', '.'])
                .as_bytes();
            let is_reserved = EXACT
                .iter()
                .any(|r| stem.eq_ignore_ascii_case(r.as_bytes()))
                || (stem.len() == 4
                    && matches!(stem[3], b'1'..=b'9')
                    && NUMBERED
                        .iter()
                        .any(|r| stem[..3].eq_ignore_ascii_case(r.as_bytes())));
            if is_reserved {
                return Err(MindroidError::artifact(format!(
                    "{label} is a reserved device name"
                )));
            }
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
        Self::reject_symlink(&bytes_path, "artifact", id).await?;
        Self::reject_symlink(&sidecar_path, "sidecar", id).await?;
        Ok((bytes_path, sidecar_path))
    }

    /// A fast, clear rejection for the common case. It is *not* what closes the
    /// check-then-use race — [`open_no_follow`](Self::open_no_follow) and the
    /// `create_new` write path do that at open time, so a symlink swapped in
    /// after this stat still cannot be followed.
    async fn reject_symlink(path: &Path, kind: &str, id: &str) -> Result<()> {
        let is_symlink = tokio::fs::symlink_metadata(path)
            .await
            .is_ok_and(|m| m.file_type().is_symlink());
        if is_symlink {
            // Only the absolute path is withheld — it reaches the model and the
            // wire and would disclose the host's layout. The id is the caller's.
            tracing::debug!(path = %path.display(), "refusing to follow symlink");
            return Err(MindroidError::artifact(format!(
                "refusing to follow a symlinked {kind} for '{id}'"
            )));
        }
        Ok(())
    }

    /// Open for reading, refusing a final-component symlink, so the path that
    /// was validated is the path that is read.
    ///
    /// `O_NOFOLLOW` fails the open outright. Windows has no equivalent:
    /// `FILE_FLAG_OPEN_REPARSE_POINT` opens the link itself rather than
    /// failing, so the refusal is asserted here against the opened handle
    /// instead of being inferred from what a later read happens to do.
    ///
    /// The Windows check is tag-agnostic by design — see ADR-0006.
    async fn open_no_follow(path: &Path) -> std::io::Result<tokio::fs::File> {
        let mut opts = tokio::fs::OpenOptions::new();
        opts.read(true);
        #[cfg(unix)]
        opts.custom_flags(libc::O_NOFOLLOW);
        #[cfg(windows)]
        {
            const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
            opts.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
        }

        let file = opts.open(path).await?;

        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
            if file.metadata().await?.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "refusing to follow a reparse point",
                ));
            }
        }
        Ok(file)
    }

    /// Capped with `take` rather than a `metadata()` pre-size: the length is
    /// attacker-controlled, and a sparse file costs them nothing while an
    /// up-front reservation of it aborts the process under `panic = "abort"`.
    async fn read_no_follow(path: &Path, max: u64) -> std::io::Result<Vec<u8>> {
        use tokio::io::AsyncReadExt;

        let mut buf = Vec::new();
        Self::open_no_follow(path)
            .await?
            .take(max + 1)
            .read_to_end(&mut buf)
            .await?;
        if buf.len() as u64 > max {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "artifact exceeds the size limit",
            ));
        }
        Ok(buf)
    }

    /// `create_new` is `O_EXCL`: it fails if anything already occupies the path,
    /// symlink included, so a write can never land on an attacker's target. Ids
    /// are fresh UUIDs, so a collision is a bug, not a case to overwrite.
    ///
    /// The `flush` is load-bearing, not hygiene: `tokio::fs::File` hands the
    /// write to a blocking task and reports its error only on a later poll, so
    /// dropping the handle after `write_all` discards a failed write entirely.
    ///
    /// Bounded here rather than at the caller so both writes are covered: a
    /// sidecar past the read cap would store fine and never load again.
    async fn write_new(path: &Path, data: &[u8], max: u64) -> std::io::Result<()> {
        use tokio::io::AsyncWriteExt;

        if data.len() as u64 > max {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "artifact exceeds the size limit",
            ));
        }
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .await?;
        file.write_all(data).await?;
        file.flush().await
    }
}

#[async_trait]
impl ArtifactStore for LocalArtifactStore {
    async fn save(&self, scope: &str, data: &[u8], mime_type: &str) -> Result<StoredArtifact> {
        let id = Uuid::new_v4().to_string();
        let (bytes_path, sidecar_path) = self.resolve_paths(scope, &id, true).await?;

        // Both writes are validated before either happens: failing the sidecar
        // after the bytes are down would orphan a file no id can ever reach.
        let sidecar = Sidecar {
            mime_type: mime_type.to_string(),
        };
        let json = serde_json::to_vec(&sidecar)
            .map_err(|e| MindroidError::artifact(format!("serialize sidecar failed: {e}")))?;
        if data.len() as u64 > MAX_ARTIFACT_BYTES || json.len() as u64 > MAX_SIDECAR_BYTES {
            return Err(MindroidError::artifact("artifact exceeds the size limit"));
        }

        Self::write_new(&bytes_path, data, MAX_ARTIFACT_BYTES)
            .await
            .map_err(|e| MindroidError::artifact(format!("write bytes failed: {e}")))?;
        Self::write_new(&sidecar_path, &json, MAX_SIDECAR_BYTES)
            .await
            .map_err(|e| MindroidError::artifact(format!("write sidecar failed: {e}")))?;

        // LocalArtifactStore is a plain store: just the id, no metadata of its own.
        Ok(StoredArtifact::new(id))
    }

    async fn load(&self, scope: &str, id: &str) -> Result<Artifact> {
        let (bytes_path, sidecar_path) = self.resolve_paths(scope, id, false).await?;

        let data = Self::read_no_follow(&bytes_path, MAX_ARTIFACT_BYTES)
            .await
            .map_err(|e| MindroidError::artifact(format!("read artifact '{id}' failed: {e}")))?;
        let json = Self::read_no_follow(&sidecar_path, MAX_SIDECAR_BYTES)
            .await
            .map_err(|e| MindroidError::artifact(format!("read sidecar for '{id}' failed: {e}")))?;
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

    /// Opening `NUL`/`COM1` reaches a device rather than a file. The verbatim
    /// path from `canonicalize` sidesteps that today; the guard means the jail
    /// does not silently depend on that.
    #[cfg(windows)]
    #[tokio::test]
    async fn reserved_device_names_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LocalArtifactStore::new(tmp.path());
        store.save("chan", &[1], "image/png").await.unwrap();

        for id in [
            "NUL",
            "nul",
            "CON",
            "COM1",
            "LPT9",
            "NUL.txt",
            "CONIN$",
            "NUL ",
            "NUL.",
            "NUL:$DATA",
        ] {
            let e = store.load("chan", id).await.unwrap_err();
            assert!(
                e.to_string().contains("reserved device name"),
                "{id} must be rejected by the guard, got: {e}"
            );
        }
        // Near-misses stay usable. The multi-byte ids are the regression: the
        // guard used to slice by byte index and panicked on them.
        for id in [
            "COMET",
            "NULL",
            "CONSOLE",
            "COM",
            "LPT",
            "COM0",
            "LPT0",
            "CON1",
            "COM10",
            "😀",
            "éé",
            "日本語",
        ] {
            assert!(
                LocalArtifactStore::safe_component("id", id).is_ok(),
                "{id} must be accepted"
            );
        }
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

    /// Symlink creation needs a privilege many Windows machines lack; skip
    /// there, fail on anything else. `MINDROID_REQUIRE_SYMLINKS` forbids the
    /// skip — CI sets it, so a runner without the privilege fails loudly
    /// instead of reporting green on tests that asserted nothing.
    fn made_symlink(r: std::io::Result<()>) -> bool {
        #[cfg(windows)]
        const ERROR_PRIVILEGE_NOT_HELD: i32 = 1314;

        match r {
            Ok(()) => true,
            #[cfg(windows)]
            Err(e) if e.raw_os_error() == Some(ERROR_PRIVILEGE_NOT_HELD) => {
                assert!(
                    std::env::var_os("MINDROID_REQUIRE_SYMLINKS").is_none(),
                    "symlink privilege is required here but absent: {e}"
                );
                eprintln!("skipping symlink test: {e}");
                false
            }
            Err(e) => panic!("symlink creation failed unexpectedly: {e}"),
        }
    }

    /// The race-closing guarantee, exercised directly rather than through the
    /// `reject_symlink` pre-check: even handed a symlink, the open must refuse
    /// to follow it. This is what holds when a swap wins the race.
    #[tokio::test]
    async fn read_no_follow_refuses_a_symlink() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("secret");
        std::fs::write(&target, b"top secret").unwrap();
        let link = tmp.path().join("link");
        if !made_symlink(symlink_file(&target, &link)) {
            return;
        }

        assert!(
            LocalArtifactStore::read_no_follow(&link, MAX_ARTIFACT_BYTES)
                .await
                .is_err()
        );
        assert_eq!(
            LocalArtifactStore::read_no_follow(&target, MAX_ARTIFACT_BYTES)
                .await
                .unwrap(),
            b"top secret"
        );
    }

    /// `create_new` is what stops a write landing on a path an attacker placed.
    #[tokio::test]
    async fn write_new_refuses_an_existing_path() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("taken");
        std::fs::write(&path, b"original").unwrap();

        assert!(
            LocalArtifactStore::write_new(&path, b"overwrite", MAX_ARTIFACT_BYTES)
                .await
                .is_err()
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
    }

    /// Smoke test only. It does not pin the flush: dropping the handle without
    /// flushing still lands the bytes almost every time, so this passes either
    /// way. `flush_surfaces_a_deferred_write_error` covers what the flush is
    /// actually for.
    #[tokio::test]
    async fn write_new_persists_the_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("fresh");

        LocalArtifactStore::write_new(&path, b"payload", MAX_ARTIFACT_BYTES)
            .await
            .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"payload");
    }

    /// The cap is exercised through the `max` parameter rather than a 64 MiB
    /// fixture, so the boundary is pinned without the suite paying for it.
    #[tokio::test]
    async fn write_new_enforces_the_cap() {
        let tmp = tempfile::tempdir().unwrap();

        let at_limit = tmp.path().join("at-limit");
        LocalArtifactStore::write_new(&at_limit, b"1234", 4)
            .await
            .unwrap();
        assert_eq!(std::fs::read(&at_limit).unwrap(), b"1234");

        let over = tmp.path().join("over");
        let e = LocalArtifactStore::write_new(&over, b"12345", 4)
            .await
            .unwrap_err();
        assert!(e.to_string().contains("exceeds the size limit"), "got: {e}");
        assert!(!over.exists(), "nothing may be left behind");
    }

    /// The sidecar is written from a caller-supplied mime type, so it needs the
    /// same bound as the bytes — otherwise it stores and never loads again.
    #[tokio::test]
    async fn save_rejects_an_oversized_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LocalArtifactStore::new(tmp.path());

        let huge_mime = "x".repeat(MAX_SIDECAR_BYTES as usize + 1);
        let e = store
            .save("chan", &[1, 2, 3], &huge_mime)
            .await
            .unwrap_err();

        assert!(e.to_string().contains("exceeds the size limit"), "got: {e}");
        // The id never escapes a failed save, so anything left behind is
        // unreachable by `load` or `delete` — it must write nothing at all.
        assert!(
            std::fs::read_dir(tmp.path().join("chan"))
                .unwrap()
                .next()
                .is_none(),
            "a rejected save left a file behind"
        );
    }

    /// Pins tokio's behaviour, not `write_new`'s use of it — deleting the
    /// `flush` fails no test, because a deferred write failure cannot be
    /// induced through `write_new` portably. This is the canary: `write_all`
    /// reports success for a write that cannot succeed, and only `flush`
    /// surfaces the error.
    #[tokio::test]
    async fn flush_surfaces_a_deferred_write_error() {
        use tokio::io::AsyncWriteExt;

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("read-only-handle");
        std::fs::write(&path, b"existing").unwrap();

        let mut file = tokio::fs::OpenOptions::new()
            .read(true)
            .open(&path)
            .await
            .unwrap();

        assert!(
            file.write_all(b"denied").await.is_ok(),
            "tokio defers the write, so write_all reports success"
        );
        assert!(
            file.flush().await.is_err(),
            "flush must surface the deferred failure"
        );
    }

    /// `(base_dir, outside_dir, store)` where `<base>/evil` symlinks to `outside`,
    /// which holds `secret` + `secret.json`. `None` when symlinks are unavailable.
    fn symlinked_scope() -> Option<(tempfile::TempDir, tempfile::TempDir, LocalArtifactStore)> {
        let base = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret"), b"top secret").unwrap();
        std::fs::write(outside.path().join("secret.json"), br#"{"mime_type":"x"}"#).unwrap();
        if !made_symlink(symlink_dir(outside.path(), base.path().join("evil"))) {
            return None;
        }

        let store = LocalArtifactStore::new(base.path());
        Some((base, outside, store))
    }

    #[tokio::test]
    async fn symlinked_scope_load_is_rejected() {
        let Some((_base, _outside, store)) = symlinked_scope() else {
            return;
        };
        let res = store.load("evil", "secret").await;
        assert!(res.is_err(), "load through a symlinked scope must fail");
    }

    #[tokio::test]
    async fn symlinked_scope_save_is_rejected() {
        let Some((_base, outside, store)) = symlinked_scope() else {
            return;
        };
        let before = std::fs::read_dir(outside.path()).unwrap().count();

        assert!(store.save("evil", &[1, 2, 3], "image/png").await.is_err());

        let after = std::fs::read_dir(outside.path()).unwrap().count();
        assert_eq!(before, after, "save must not write outside the base");
    }

    #[tokio::test]
    async fn symlinked_scope_delete_does_not_touch_outside() {
        let Some((_base, outside, store)) = symlinked_scope() else {
            return;
        };
        assert!(store.delete("evil", "secret").await.is_err());
        assert!(outside.path().join("secret").exists());
        assert!(outside.path().join("secret.json").exists());
    }

    #[tokio::test]
    async fn symlinked_artifact_file_load_is_rejected() {
        let base = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("secret");
        std::fs::write(&target, b"top secret").unwrap();

        let dir = base.path().join("chan");
        std::fs::create_dir_all(&dir).unwrap();
        if !made_symlink(symlink_file(&target, dir.join("art"))) {
            return;
        }
        std::fs::write(dir.join("art.json"), br#"{"mime_type":"x"}"#).unwrap();

        let store = LocalArtifactStore::new(base.path());
        let res = store.load("chan", "art").await;
        assert!(res.is_err(), "load of a symlinked artifact file must fail");
    }
}
