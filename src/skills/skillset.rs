//! `SkillSet` — one-line skill setup for mindroid agents.
//!
//! Wraps `SkillRegistry`, `build_skill_index`, and `ReadSkillTool` into a single
//! composable unit that plugs into the pipeline without touching RuntimeBuilder.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::skills::index::build_skill_index;
use crate::skills::read_tool::ReadSkillTool;
use crate::skills::registry::SkillRegistry;
use crate::tools::ToolRegistry;

/// A ready-to-use skill set that owns the registry and provides pipeline integration.
///
/// # Usage
///
/// ```ignore
/// let skills = SkillSet::from_workspace("./skills").await;
///
/// let registry = skills.extend_tools(ToolRegistry::from_config(&config.tools));
/// let pipeline = Pipeline::new()
///     .add_stage(SimpleContextBuilder::with_prompt("You are...").with_skills(&skills))
///     .add_streaming_stage(XmlToolExecutorStage::new(llm, Arc::new(registry)));
/// ```
pub struct SkillSet {
    registry: Arc<RwLock<SkillRegistry>>,
    index: String,
    skill_names: Vec<String>,
}

impl SkillSet {
    /// Discover skills from workspace and user directories.
    ///
    /// Discovery order (earlier wins on name collision):
    /// 1. Workspace directory
    /// 2. User directory
    pub async fn discover(workspace_dir: impl AsRef<Path>, user_dir: impl AsRef<Path>) -> Self {
        let ws = workspace_dir.as_ref();
        let mut registry = SkillRegistry::new(user_dir.as_ref().to_path_buf());
        if ws.exists() {
            registry = registry.with_workspace_dir(ws.to_path_buf());
        }
        let skill_names = registry.discover_all().await;
        let index = build_skill_index(registry.skills());

        Self {
            registry: Arc::new(RwLock::new(registry)),
            index,
            skill_names,
        }
    }

    /// Discover skills from just a workspace directory.
    ///
    /// Uses `~/.mindroid/skills/` as the user directory (falls back to
    /// `./.mindroid/skills/` if the HOME environment variable is not set).
    pub async fn from_workspace(workspace_dir: impl AsRef<Path>) -> Self {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        let user_dir = PathBuf::from(home).join(".mindroid/skills");
        Self::discover(workspace_dir, user_dir).await
    }

    /// Create a SkillSet from an existing registry.
    ///
    /// Useful when you've already configured the registry manually
    /// (e.g., registered built-in skills).
    pub fn from_registry(registry: SkillRegistry) -> Self {
        let index = build_skill_index(registry.skills());
        let skill_names = registry
            .skills()
            .iter()
            .map(|s| s.name().to_string())
            .collect();
        Self {
            registry: Arc::new(RwLock::new(registry)),
            index,
            skill_names,
        }
    }

    /// Create an empty SkillSet with no skills.
    pub fn empty() -> Self {
        let user_dir = PathBuf::from(".mindroid/skills");
        Self {
            registry: Arc::new(RwLock::new(SkillRegistry::new(user_dir))),
            index: String::new(),
            skill_names: Vec::new(),
        }
    }

    /// Get the compact XML skill index for system prompt injection.
    pub fn index(&self) -> &str {
        &self.index
    }

    /// Whether no skills were discovered.
    pub fn is_empty(&self) -> bool {
        self.skill_names.is_empty()
    }

    /// Get the names of all discovered skills.
    pub fn skill_names(&self) -> &[String] {
        &self.skill_names
    }

    /// Get a shared reference to the underlying registry.
    pub fn registry(&self) -> &Arc<RwLock<SkillRegistry>> {
        &self.registry
    }

    /// Create a `ReadSkillTool` for this skill set.
    pub fn read_tool(&self) -> ReadSkillTool {
        ReadSkillTool::new(self.registry.clone())
    }

    /// Extend a `ToolRegistry` with the `read_skill` tool.
    ///
    /// Returns the extended registry. Only adds the tool if skills exist.
    pub fn extend_tools(&self, registry: ToolRegistry) -> ToolRegistry {
        if self.is_empty() {
            registry
        } else {
            registry.register(self.read_tool())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_empty_skillset() {
        let skills = SkillSet::empty();
        assert!(skills.is_empty());
        assert!(skills.index().is_empty());
        assert_eq!(skills.skill_names().len(), 0);
    }

    #[tokio::test]
    async fn test_from_registry() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("my-skill");
        fs::create_dir(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: my-skill\ndescription: Test\n---\n\nTest prompt.\n",
        )
        .unwrap();

        let mut registry = SkillRegistry::new(dir.path().to_path_buf());
        registry.discover_all().await;

        let skills = SkillSet::from_registry(registry);
        assert!(!skills.is_empty());
        assert_eq!(skills.skill_names(), &["my-skill".to_string()]);
        assert!(skills.index().contains("my-skill"));
    }

    #[tokio::test]
    async fn test_extend_tools_adds_read_skill() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("test-skill");
        fs::create_dir(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: test-skill\n---\n\nPrompt.\n",
        )
        .unwrap();

        let mut registry = SkillRegistry::new(dir.path().to_path_buf());
        registry.discover_all().await;

        let skills = SkillSet::from_registry(registry);
        let tool_registry = ToolRegistry::new();
        let extended = skills.extend_tools(tool_registry);

        let tool_names: Vec<&str> = extended.tools().iter().map(|t| t.name()).collect();
        assert!(tool_names.contains(&"read_skill"));
    }

    #[test]
    fn test_extend_tools_noop_when_empty() {
        let skills = SkillSet::empty();
        let tool_registry = ToolRegistry::new();
        let extended = skills.extend_tools(tool_registry);

        let tool_names: Vec<&str> = extended.tools().iter().map(|t| t.name()).collect();
        assert!(!tool_names.contains(&"read_skill"));
    }
}
