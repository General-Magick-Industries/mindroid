//! Compact skill index for system prompt injection.
//!
//! Generates a lightweight `<available_skills>` XML block containing only
//! skill names and descriptions. The LLM uses this index to decide which
//! skills to load via the `read_skill` tool.

use crate::skills::{LoadedSkill, escape_xml_attr};

/// Build a compact XML index of available skills for system prompt injection.
///
/// Output format:
/// ```xml
/// ## Available Skills
///
/// <available_skills>
///   <skill name="m01-ownership" description="Ownership/borrow/lifetime issues..." />
///   <skill name="domain-web" description="Web services: axum, actix, REST..." />
/// </available_skills>
///
/// To use a skill, call the `read_skill` tool with the skill name.
/// Only load skills that are clearly relevant to the current task.
/// ```
pub fn build_skill_index(skills: &[LoadedSkill]) -> String {
    if skills.is_empty() {
        return String::new();
    }

    let mut out = String::from("## Available Skills\n\n<available_skills>\n");

    for skill in skills {
        let name = escape_xml_attr(skill.name());
        let desc = escape_xml_attr(&skill.manifest.description);
        let location_attr = skill
            .location()
            .map(|p| {
                format!(
                    " location=\"{}\"",
                    escape_xml_attr(&p.display().to_string())
                )
            })
            .unwrap_or_default();
        out.push_str(&format!(
            "  <skill name=\"{name}\" description=\"{desc}\"{location_attr} />\n"
        ));
    }

    out.push_str("</available_skills>\n\n");
    out.push_str("To use a skill, call the `read_skill` tool with the skill name.\n");
    out.push_str("Only load skills that are clearly relevant to the current task.\n");
    out.push_str(
        "Skills may have supporting files (scripts, references) in their location directory.\n",
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::{ActivationCriteria, LoadedSkill, SkillManifest, SkillSource, SkillTrust};
    use std::path::PathBuf;

    fn make_skill(name: &str, description: &str) -> LoadedSkill {
        LoadedSkill {
            manifest: SkillManifest {
                name: name.to_string(),
                version: "1.0.0".to_string(),
                description: description.to_string(),
                activation: ActivationCriteria::default(),
                requires: None,
                globs: vec![],
                source: None,
                user_invocable: false,
            },
            prompt_content: "Test prompt content".to_string(),
            trust: SkillTrust::Trusted,
            source: SkillSource::User(PathBuf::from("/tmp/test")),
            content_hash: "sha256:000".to_string(),
            compiled_patterns: vec![],
            lowercased_keywords: vec![],
            lowercased_tags: vec![],
        }
    }

    #[test]
    fn test_empty_skills_returns_empty_string() {
        assert_eq!(build_skill_index(&[]), "");
    }

    #[test]
    fn test_single_skill_generates_correct_xml() {
        let skills = vec![make_skill(
            "m01-ownership",
            "Ownership/borrow/lifetime issues",
        )];
        let result = build_skill_index(&skills);
        assert!(result.contains("## Available Skills"));
        assert!(result.contains("<available_skills>"));
        assert!(result.contains("</available_skills>"));
        assert!(result.contains(r#"name="m01-ownership""#));
        assert!(result.contains(r#"description="Ownership/borrow/lifetime issues""#));
        assert!(result.contains(r#"location="/tmp/test""#));
        assert!(result.contains("To use a skill, call the `read_skill` tool"));
        assert!(result.contains("Only load skills that are clearly relevant"));
        assert!(result.contains("supporting files"));
    }

    #[test]
    fn test_multiple_skills_all_included() {
        let skills = vec![
            make_skill("m01-ownership", "Ownership issues"),
            make_skill("domain-web", "Web services"),
            make_skill("async-patterns", "Async patterns"),
        ];
        let result = build_skill_index(&skills);
        assert!(result.contains(r#"name="m01-ownership""#));
        assert!(result.contains(r#"name="domain-web""#));
        assert!(result.contains(r#"name="async-patterns""#));
    }

    #[test]
    fn test_special_chars_in_name_and_description_escaped() {
        let skills = vec![make_skill(
            "test-skill",
            r#"Description with "quotes" & <angle> brackets"#,
        )];
        let result = build_skill_index(&skills);
        assert!(result.contains("&quot;quotes&quot;"));
        assert!(result.contains("&amp;"));
        assert!(result.contains("&lt;angle&gt;"));
        // No raw unescaped special chars in attribute positions
        assert!(!result.contains(r#"description=""#.to_string().as_str()
            .chars()
            .chain(r#"Description with ""#.chars())
            .collect::<String>()
            .as_str()));
    }
}
