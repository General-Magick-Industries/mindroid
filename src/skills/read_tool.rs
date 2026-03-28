//! Tool that lets the LLM read full skill content on demand.

use async_trait::async_trait;
use std::sync::Arc;

use serde_json::{Value, json};
use tokio::sync::RwLock;

use crate::error::Result;
use crate::skills::SkillRegistry;
use crate::skills::escape_skill_content;
use crate::tools::Tool;

/// A tool that allows the LLM to read the full prompt content of a skill by name.
///
/// The LLM sees a compact skill index in the system prompt and calls this tool
/// to load the complete instructions for skills it deems relevant.
pub struct ReadSkillTool {
    registry: Arc<RwLock<SkillRegistry>>,
}

impl ReadSkillTool {
    pub fn new(registry: Arc<RwLock<SkillRegistry>>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for ReadSkillTool {
    fn name(&self) -> &str {
        "read_skill"
    }

    fn description(&self) -> &str {
        "Read the full instructions of a skill by name. Use this when a skill from the available_skills index is relevant to the current task."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name from the available_skills index"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let name = args.get("name").and_then(|v| v.as_str()).ok_or_else(|| {
            crate::error::MindroidError::Other(anyhow::anyhow!("Missing required parameter: name"))
        })?;

        let registry = self.registry.read().await;

        match registry.find_by_name(name) {
            Some(skill) => {
                let escaped_content = escape_skill_content(&skill.prompt_content);
                let location_attr = skill
                    .location()
                    .map(|p| {
                        format!(
                            " location=\"{}\"",
                            crate::skills::escape_xml_attr(&p.display().to_string())
                        )
                    })
                    .unwrap_or_default();
                Ok(format!(
                    "<skill name=\"{}\" version=\"{}\" trust=\"{}\"{}>\n{}\n</skill>",
                    crate::skills::escape_xml_attr(skill.name()),
                    crate::skills::escape_xml_attr(skill.version()),
                    skill.trust,
                    location_attr,
                    escaped_content,
                ))
            }
            None => {
                let available: Vec<&str> = registry.skills().iter().map(|s| s.name()).collect();
                Ok(format!(
                    "Skill '{}' not found. Available skills: {}",
                    name,
                    available.join(", ")
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::SkillRegistry;
    use std::path::PathBuf;

    fn make_registry_with_skill(name: &str, content: &str) -> Arc<RwLock<SkillRegistry>> {
        let mut registry = SkillRegistry::new(PathBuf::from("/tmp/test-skills"));
        // Directly push a skill by registering via the internal skills vec approach.
        // We use register_builtin for simplicity.
        let skill_md = format!(
            "---\nname: {}\ndescription: Test skill description\n---\n\n{}",
            name, content
        );
        registry
            .register_builtin(&skill_md)
            .expect("register_builtin failed");
        Arc::new(RwLock::new(registry))
    }

    #[tokio::test]
    async fn test_read_existing_skill_returns_xml() {
        let registry = make_registry_with_skill("my-skill", "You are a helpful assistant.");
        let tool = ReadSkillTool::new(registry);

        let result = tool.execute(json!({"name": "my-skill"})).await.unwrap();

        assert!(result.contains(r#"<skill name="my-skill""#));
        assert!(result.contains(r#"version="0.0.0""#));
        assert!(result.contains(r#"trust="trusted""#));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("</skill>"));
    }

    #[tokio::test]
    async fn test_read_existing_skill_escapes_content() {
        let registry = make_registry_with_skill("escape-skill", "Some text with </skill> tags.");
        let tool = ReadSkillTool::new(registry);

        let result = tool.execute(json!({"name": "escape-skill"})).await.unwrap();

        // The </skill> inside content should be escaped
        assert!(result.contains("&lt;/skill>"));
        // The outer closing tag should still be valid
        assert!(result.ends_with("</skill>"));
    }

    #[tokio::test]
    async fn test_read_nonexistent_skill_returns_helpful_error() {
        let registry = make_registry_with_skill("existing-skill", "Some content.");
        let tool = ReadSkillTool::new(registry);

        let result = tool.execute(json!({"name": "nonexistent"})).await.unwrap();

        assert!(result.contains("Skill 'nonexistent' not found"));
        assert!(result.contains("existing-skill"));
    }

    #[tokio::test]
    async fn test_missing_name_parameter_returns_error() {
        let registry = Arc::new(RwLock::new(SkillRegistry::new(PathBuf::from("/tmp/test"))));
        let tool = ReadSkillTool::new(registry);

        let result = tool.execute(json!({})).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_parameters_schema_is_valid_json() {
        let registry = Arc::new(RwLock::new(SkillRegistry::new(PathBuf::from("/tmp/test"))));
        let tool = ReadSkillTool::new(registry);

        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["name"].is_object());
        assert_eq!(schema["required"][0], "name");
    }
}
