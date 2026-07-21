use super::models::{EffectiveTrait, PersonaSchema};

/// Assemble a system prompt from a persona schema and its blended traits.
///
/// This is the client-side path used by providers that return raw persona data
/// (the legacy magickmind client, the local file provider). Providers that
/// return a server-assembled prompt bypass this entirely — see
/// [`super::PersonaProvider::prepared_prompt`].
pub fn build_system_prompt(persona: &PersonaSchema, traits: &[EffectiveTrait]) -> String {
    let mut parts = Vec::new();

    parts.push(format!("You are {}, a {}.", persona.name, persona.role));

    if !persona.background_story.is_empty() {
        parts.push(String::new());
        parts.push(persona.background_story.clone());
    }

    if !traits.is_empty() {
        parts.push(String::new());
        parts.push("Your personality traits:".to_string());
        for t in traits {
            parts.push(format_trait(t));
        }
    }

    if !persona.tones.is_empty() {
        parts.push(String::new());
        parts.push(format!("Communication tones: {}", persona.tones.join(", ")));
    }

    parts.join("\n")
}

/// Format a single effective trait for the system prompt.
fn format_trait(t: &EffectiveTrait) -> String {
    let value = t.value.display();
    if let Some(ref lock) = t.sources.lock {
        format!("- {}: {} [lock: {}]", t.trait_ref, value, lock)
    } else {
        format!("- {}: {}", t.trait_ref, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persona::models::{EffectiveSources, TraitValue};

    fn schema() -> PersonaSchema {
        PersonaSchema {
            id: "p1".into(),
            artifact_id: None,
            name: "Aria".into(),
            role: "research assistant".into(),
            traits: Vec::new(),
            tones: vec!["warm".into(), "concise".into()],
            background_story: "You were built to help.".into(),
            created_by: String::new(),
            updated_by: String::new(),
            active_version: None,
        }
    }

    fn numeric_trait(name: &str, value: f64, lock: Option<&str>) -> EffectiveTrait {
        EffectiveTrait {
            trait_ref: name.into(),
            value: TraitValue {
                numeric_value: Some(value),
                string_value: None,
                string_list_value: None,
            },
            sources: EffectiveSources {
                lock: lock.map(String::from),
                ..Default::default()
            },
        }
    }

    #[test]
    fn builds_full_prompt() {
        let prompt = build_system_prompt(&schema(), &[numeric_trait("warmth", 0.8, None)]);
        assert!(prompt.starts_with("You are Aria, a research assistant."));
        assert!(prompt.contains("You were built to help."));
        assert!(prompt.contains("- warmth: 0.8"));
        assert!(prompt.contains("Communication tones: warm, concise"));
    }

    #[test]
    fn renders_lock_annotation() {
        let prompt = build_system_prompt(&schema(), &[numeric_trait("candor", 0.5, Some("HARD"))]);
        assert!(prompt.contains("- candor: 0.5 [lock: HARD]"));
    }

    #[test]
    fn omits_empty_sections() {
        let mut s = schema();
        s.background_story = String::new();
        s.tones = Vec::new();
        let prompt = build_system_prompt(&s, &[]);
        assert_eq!(prompt, "You are Aria, a research assistant.");
    }
}
