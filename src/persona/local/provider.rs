use async_trait::async_trait;
use std::path::PathBuf;

use chrono::Utc;

use crate::error::{MindroidError, Result};
use crate::persona::models::{EffectivePersonalityResponse, PersonaSchema};
use crate::persona::provider::PersonaProvider;

use super::blender::{blend_traits, DyadicLearnedTraits};
use super::parser::{parse_persona_file, to_persona_schema, LocalPersonaFrontmatter};

pub struct LocalPersonaProvider {
    persona_id: String,
    data_dir: PathBuf,
    frontmatter: LocalPersonaFrontmatter,
    background_story: String,
}

impl LocalPersonaProvider {
    /// Load a persona from `{data_dir}/{persona_id}/persona.md`.
    ///
    /// Reads synchronously since this only runs once at startup.
    pub fn load(data_dir: &str, persona_id: &str) -> Result<Self> {
        let data_dir = PathBuf::from(data_dir);
        let persona_file = data_dir.join(persona_id).join("persona.md");

        let content = std::fs::read_to_string(&persona_file).map_err(|e| {
            MindroidError::config(format!(
                "Failed to read persona file '{}': {e}",
                persona_file.display()
            ))
        })?;

        let (frontmatter, background_story) = parse_persona_file(&content)?;

        Ok(Self {
            persona_id: persona_id.to_string(),
            data_dir,
            frontmatter,
            background_story,
        })
    }
}

#[async_trait]
impl PersonaProvider for LocalPersonaProvider {
    fn name(&self) -> &str {
        "local"
    }

    async fn get_persona(&self, _persona_id: &str) -> Result<PersonaSchema> {
        Ok(to_persona_schema(
            &self.persona_id,
            &self.frontmatter,
            &self.background_story,
        ))
    }

    async fn get_effective_personality(
        &self,
        _persona_id: &str,
        user_id: Option<&str>,
    ) -> Result<EffectivePersonalityResponse> {
        let dyadic = if let Some(uid) = user_id {
            let dyadic_file = self
                .data_dir
                .join(&self.persona_id)
                .join("dyadic")
                .join(format!("{uid}.json"));

            if dyadic_file.exists() {
                let raw = std::fs::read_to_string(&dyadic_file).map_err(|e| {
                    MindroidError::config(format!(
                        "Failed to read dyadic file '{}': {e}",
                        dyadic_file.display()
                    ))
                })?;
                let parsed: DyadicLearnedTraits =
                    serde_json::from_str(&raw).map_err(|e| {
                        MindroidError::config(format!(
                            "Failed to parse dyadic file '{}': {e}",
                            dyadic_file.display()
                        ))
                    })?;
                Some(parsed)
            } else {
                None
            }
        } else {
            None
        };

        let traits = blend_traits(&self.frontmatter.traits, dyadic.as_ref());

        Ok(EffectivePersonalityResponse {
            persona_id: self.persona_id.clone(),
            user_id: user_id.map(|s| s.to_string()),
            traits,
            computed_at: Utc::now().to_rfc3339(),
            ttl_seconds: 0,
        })
    }
}
