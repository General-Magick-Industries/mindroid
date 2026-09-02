//! Artifact builders — mint a matched `(offload stage, load tool, store)` set
//! from a single store, so the three never mismatch.
//!
//! The full artifact flow needs the same [`ArtifactStore`] in two places: the
//! [`ArtifactOffload`] stage (write references) and the [`GetArtifactTool`] (the
//! model's fetch tool). The store lives ON the tool, so the
//! [`XmlToolExecutorStage`](crate::pipeline::stages::XmlToolExecutorStage) finds it
//! automatically when the tool is registered — no separate store injection.
//! These builders hand back the matched `(offload, tool)` pair sharing one store;
//! you place each where you want:
//!
//! ```ignore
//! let (offload, tool) = artifacts_local("./artifacts", "my-channel");
//!
//! let registry = ToolRegistry::new().register(tool); // + your other tools
//! let pipeline = Pipeline::new()
//!     .add_stage(IngestStage::default_media())
//!     .add_streaming_stage(XmlToolExecutorStage::new(client, Arc::new(registry)))
//!     .add_stage(offload)          // place the offload wherever you like
//!     .add_stage(PostProcessor);
//! ```
//!
//! Swapping backends means implementing [`ArtifactStore`] for the new backend
//! and passing it to [`artifacts_from_store`] — no change to the offload stage
//! or the tool.

use std::sync::Arc;

use crate::artifacts::{ArtifactManager, ArtifactStore, LocalArtifactStore};
use crate::pipeline::stages::ArtifactOffload;
use crate::tools::GetArtifactTool;

/// The matched artifact pieces sharing one [`ArtifactManager`] (hence one store):
/// the offload stage and the load tool. The store rides on the tool, so the
/// executor needs no separate handle.
pub type ArtifactSet = (ArtifactOffload, GetArtifactTool);

/// Build the artifact set backed by an on-disk [`LocalArtifactStore`] at `base`.
///
/// `scope` pins the tool's *validation message* only. The bytes the model sees
/// are re-injected by the executor under the live per-message scope
/// (`ctx.message.channel_id`), which is authoritative — see [`GetArtifactTool`].
/// It MUST still come from trusted context, never model/user input.
pub fn artifacts_local(
    base: impl Into<std::path::PathBuf>,
    scope: impl Into<String>,
) -> ArtifactSet {
    let store: Arc<dyn ArtifactStore> = Arc::new(LocalArtifactStore::new(base));
    artifacts_from_store(store, scope)
}

/// Build the artifact set from an already-constructed store. Use this to plug in
/// a custom [`ArtifactStore`] impl while still getting the matched, consistent
/// `(offload, tool)` pair — both share one [`ArtifactManager`].
pub fn artifacts_from_store(
    store: Arc<dyn ArtifactStore>,
    scope: impl Into<String>,
) -> ArtifactSet {
    ArtifactManager::new(store).into_stage_and_tool(scope)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Tool;

    #[tokio::test]
    async fn local_builder_shares_one_store() {
        let tmp = tempfile::tempdir().unwrap();
        let (_offload, tool) = artifacts_local(tmp.path(), "chan1");

        // The tool exposes its store; save through it...
        let store = tool.artifact_store().expect("load tool has a store");
        let id = store
            .save("chan1", &[7, 7, 7], "image/png")
            .await
            .unwrap()
            .id;
        // ...and the same tool resolves it.
        let out = tool
            .execute(
                serde_json::json!({ "id": id }),
                &crate::tools::ToolContext::default(),
            )
            .await
            .unwrap();
        assert!(out.contains("Loaded artifact"));
        assert!(out.contains(&id));
    }

    /// The documented contract: `artifacts_local`'s `scope` pins only the tool's
    /// confirmation string. The bytes the model sees are re-injected by the
    /// executor under the live per-message scope, which is authoritative.
    ///
    /// In a multi-channel process the two can disagree — the tool reports
    /// not-found for an artifact whose bytes the executor still attaches
    /// correctly. This pins that split so the strings and the bytes cannot
    /// silently swap roles.
    #[tokio::test]
    async fn pinned_tool_scope_can_disagree_with_the_authoritative_load_scope() {
        let tmp = tempfile::tempdir().unwrap();
        // Tool pinned to chan1; the artifact actually lives under chan2.
        let (_offload, tool) = artifacts_local(tmp.path(), "chan1");
        let store = tool.artifact_store().expect("load tool has a store");
        let id = store.save("chan2", &[9, 9], "image/png").await.unwrap().id;

        // The confirmation string is computed against the PINNED scope, so it
        // fails to find an artifact that exists.
        let confirmation = tool
            .execute(
                serde_json::json!({ "id": &id }),
                &crate::tools::ToolContext::default(),
            )
            .await
            .unwrap();
        assert!(
            !confirmation.contains("Loaded artifact"),
            "pinned scope must not claim success for another channel's artifact: {confirmation}"
        );

        // The executor's authoritative scope still resolves the real bytes.
        let loaded = store
            .load("chan2", &id)
            .await
            .expect("bytes are retrievable");
        assert_eq!(loaded.data, vec![9, 9]);
    }

    /// `unscoped` opts out of the scope-dependent validation entirely, so the
    /// confirmation never contradicts the attached bytes.
    #[tokio::test]
    async fn an_unscoped_tool_reports_neutrally_instead_of_contradicting() {
        let tmp = tempfile::tempdir().unwrap();
        let (_offload, tool) = artifacts_local(tmp.path(), "chan1");
        let store = tool.artifact_store().expect("load tool has a store");
        let id = store.save("chan2", &[4], "image/png").await.unwrap().id;

        let unscoped = crate::tools::GetArtifactTool::unscoped(
            crate::artifacts::ArtifactManager::new(store.clone()),
        );
        let out = unscoped
            .execute(
                serde_json::json!({ "id": &id }),
                &crate::tools::ToolContext::default(),
            )
            .await
            .unwrap();

        assert!(
            out.contains(&id),
            "should reference the id it is loading: {out}"
        );
        assert!(
            !out.contains("not found"),
            "an unscoped tool must not claim not-found: {out}"
        );
    }
}
