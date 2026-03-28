use std::collections::HashMap;

use serde::Deserialize;

use crate::error::{MindroidError, Result};
use crate::persona::models::PersonaSchema;

#[derive(Debug, Clone, Deserialize)]
pub struct LocalPersonaFrontmatter {
    pub name: String,
    pub role: String,
    #[serde(default)]
    pub tones: Vec<String>,
    #[serde(default)]
    pub traits: HashMap<String, LocalTraitDef>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LocalTraitDef {
    pub value: f64,
    pub lock: Option<String>,
}

/// Parse a `persona.md` file with TOML frontmatter delimited by `+++`.
///
/// Returns `(frontmatter, markdown_body)`.
pub fn parse_persona_file(content: &str) -> Result<(LocalPersonaFrontmatter, String)> {
    // Split on `+++` delimiters. Expected format:
    //   +++
    //   <toml frontmatter>
    //   +++
    //   <markdown body>
    let parts: Vec<&str> = content.splitn(3, "+++").collect();
    if parts.len() < 3 {
        return Err(MindroidError::config(
            "persona.md must have TOML frontmatter delimited by +++",
        ));
    }
    // parts[0] is empty (before first +++), parts[1] is TOML, parts[2] is body
    let toml_str = parts[1].trim();
    let body = parts[2].trim().to_string();

    let frontmatter: LocalPersonaFrontmatter = toml::from_str(toml_str)
        .map_err(|e| MindroidError::config(format!("persona.md frontmatter parse error: {e}")))?;

    Ok((frontmatter, body))
}

/// Convert parsed frontmatter + body into a `PersonaSchema`.
pub fn to_persona_schema(id: &str, frontmatter: &LocalPersonaFrontmatter, body: &str) -> PersonaSchema {
    PersonaSchema {
        id: id.to_string(),
        artifact_id: None,
        name: frontmatter.name.clone(),
        role: frontmatter.role.clone(),
        traits: frontmatter.traits.keys().cloned().collect(),
        tones: frontmatter.tones.clone(),
        background_story: body.to_string(),
        created_by: String::new(),
        updated_by: String::new(),
        active_version: None,
    }
}
