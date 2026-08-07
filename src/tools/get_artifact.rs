//! Tool that lets the agent re-fetch an offloaded artifact by id.
//!
//! Because [`Tool::execute`] has no access to the live pipeline `Context`, this
//! tool cannot itself attach image bytes to the conversation. It validates the id
//! and returns a confirmation string; the actual byte re-injection is done by
//! [`ToolExecutorStage`](crate::pipeline::stages::ToolExecutorStage), which holds
//! `&mut ctx` and recognizes a `get_artifact` call by name.

use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::artifacts::{ArtifactManager, ArtifactStore};
use crate::error::Result;

use super::Tool;

/// The canonical tool name. The executor keys re-injection off this.
pub const GET_ARTIFACT_TOOL: &str = "get_artifact";

/// Re-fetches an artifact's bytes into the conversation by its id. Delegates id
/// validation + description to a shared [`ArtifactManager`].
///
/// **Scope:** the `scope` held here is used ONLY for the validation message
/// returned by `execute`. The bytes the model actually sees are re-injected by
/// [`ToolExecutorStage`](crate::pipeline::stages::ToolExecutorStage) using the
/// live per-message scope (`ctx.message.channel_id`), which is authoritative. In
/// a multi-channel process this tool's pinned scope can therefore disagree with
/// the executor's, making `execute`'s confirmation string wrong while the
/// attached bytes stay correct. Prefer one tool instance per channel, or use
/// [`Self::unscoped`] to skip the scope-dependent validation entirely.
pub struct GetArtifactTool {
    manager: ArtifactManager,
    /// `None` = skip scope-dependent validation; the executor still attaches the
    /// bytes using the authoritative per-message scope.
    scope: Option<String>,
    description: String,
}

impl GetArtifactTool {
    /// `scope` MUST come from trusted session context (e.g. the channel id),
    /// never from model/user input.
    pub fn new(store: Arc<dyn ArtifactStore>, scope: impl Into<String>) -> Self {
        Self::from_manager(ArtifactManager::new(store), scope)
    }

    /// Build from a shared [`ArtifactManager`] (so the tool and the offload stage
    /// operate on the same store).
    pub fn from_manager(manager: ArtifactManager, scope: impl Into<String>) -> Self {
        Self::with_scope(manager, Some(scope.into()))
    }

    /// Build without a pinned scope — for multi-channel processes where one tool
    /// instance serves many channels. `execute` skips scope-dependent validation
    /// and confirms the id was requested; the executor attaches the real bytes
    /// using the live per-message scope.
    pub fn unscoped(manager: ArtifactManager) -> Self {
        Self::with_scope(manager, None)
    }

    fn with_scope(manager: ArtifactManager, scope: Option<String>) -> Self {
        Self {
            manager,
            scope,
            description:
                "Re-attach a previously offloaded artifact (image/file) to the conversation \
                 so you can view it again. Pass the artifact id exactly as shown in the \
                 reference, e.g. get_artifact(\"<id>\")."
                    .to_string(),
        }
    }
}

#[async_trait]
impl Tool for GetArtifactTool {
    fn name(&self) -> &str {
        GET_ARTIFACT_TOOL
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The artifact id to load (as shown in the reference)."
                }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, args: Value, _ctx: &crate::tools::ToolContext) -> Result<String> {
        let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("");
        // The actual bytes are attached by ToolExecutorStage (which holds ctx and
        // the authoritative scope). Here we only produce the confirmation string.
        match &self.scope {
            Some(scope) => Ok(self.manager.load_described(scope, id).await),
            None if id.is_empty() => Ok("Error: no artifact id provided".to_string()),
            None => Ok(format!("Loading artifact {id}")),
        }
    }

    /// Expose the backing store so `ToolExecutorStage` can re-inject loaded bytes
    /// without a separate store injection.
    fn artifact_store(&self) -> Option<Arc<dyn ArtifactStore>> {
        Some(self.manager.store().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::LocalArtifactStore;

    #[tokio::test]
    async fn validates_and_confirms() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalArtifactStore::new(tmp.path()));
        let id = store
            .save("chan1", &[1, 2, 3], "image/png")
            .await
            .unwrap()
            .id;

        let tool = GetArtifactTool::new(store, "chan1");
        let out = tool
            .execute(json!({ "id": id }), &crate::tools::ToolContext::default())
            .await
            .unwrap();
        assert!(out.contains("Loaded artifact"));
        assert!(out.contains(&id));
    }

    #[tokio::test]
    async fn missing_id_errors_gracefully() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalArtifactStore::new(tmp.path()));
        let tool = GetArtifactTool::new(store, "chan1");
        let out = tool
            .execute(json!({}), &crate::tools::ToolContext::default())
            .await
            .unwrap();
        assert!(out.starts_with("Error"));
    }

    #[tokio::test]
    async fn unscoped_skips_scope_validation() {
        let tmp = tempfile::tempdir().unwrap();
        let store: Arc<dyn ArtifactStore> = Arc::new(LocalArtifactStore::new(tmp.path()));
        let tool = GetArtifactTool::unscoped(ArtifactManager::new(store));

        // An id from a channel this tool was never pinned to must not be rejected —
        // the executor resolves it against the live scope.
        let out = tool
            .execute(
                json!({ "id": "some-id" }),
                &crate::tools::ToolContext::default(),
            )
            .await
            .unwrap();
        assert!(!out.starts_with("Error"), "got: {out}");
        assert!(out.contains("some-id"));

        // A missing id is still caught.
        let out = tool
            .execute(json!({}), &crate::tools::ToolContext::default())
            .await
            .unwrap();
        assert!(out.starts_with("Error"));
    }
}
