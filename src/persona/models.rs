use serde::{Deserialize, Deserializer, Serialize};

/// Deserialize a `null` JSON value as `T::default()` instead of failing.
fn deserialize_null_as_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// Flexible trait value — exactly one field should be set per instance.
///
/// Maps to the magickmind-api `TraitValue` REST type where the value kind is
/// determined by which of the three JSON fields is present.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraitValue {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numeric_value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub string_value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub string_list_value: Option<Vec<String>>,
}

impl TraitValue {
    /// Format the value for display in a system prompt.
    pub fn display(&self) -> String {
        if let Some(v) = self.numeric_value {
            format!("{v:.1}")
        } else if let Some(ref v) = self.string_value {
            v.clone()
        } else if let Some(ref v) = self.string_list_value {
            v.join(", ")
        } else {
            String::new()
        }
    }
}

// -- Effective personality (from runtime service) -----------------------------

/// The blended personality snapshot returned by the runtime service.
///
/// Contains all traits with their computed values and provenance information,
/// along with caching metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectivePersonalityResponse {
    pub persona_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    pub traits: Vec<EffectiveTrait>,
    /// RFC 3339 timestamp of when this personality was computed.
    pub computed_at: String,
    /// Recommended cache TTL in seconds from the server.
    /// A value of 0 means "do not cache". Negative server values deserialize as 0.
    #[serde(default)]
    pub ttl_seconds: u64,
}

/// A single trait with its blended value and source provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveTrait {
    pub trait_ref: String,
    pub value: TraitValue,
    pub sources: EffectiveSources,
}

/// Provenance layer showing how a trait value was composed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EffectiveSources {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authored: Option<TraitValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_learned: Option<TraitValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dyadic_learned: Option<TraitValue>,
    /// Lock level: `"HARD"` or `"SOFT"`, or `None` if unconstrained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock: Option<String>,
    #[serde(default)]
    pub was_clamped: bool,
}

// -- Prepared persona (from end-user prepare endpoint) ------------------------

/// A server-assembled persona snapshot from `POST /v1/end-users/{agent_id}/persona/prepare`.
///
/// `system_prompt` is complete — identity, background, traits and tones are all
/// blended server-side, so no client-side assembly is required.
///
/// Note the identifier asymmetry: the request is keyed by *agent* id, while
/// `persona_id` in the response is the persona that agent resolved to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedPersonaResponse {
    pub agent_id: String,
    pub persona_id: String,
    pub active_persona_version_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    pub system_prompt: String,
    /// RFC 3339 timestamp of when this prompt was computed.
    pub computed_at: String,
    /// Recommended cache TTL in seconds. `0` means "do not cache".
    #[serde(default)]
    pub ttl_seconds: u64,
}

// -- Persona schema (from persona service) ------------------------------------

/// A persona definition as returned by the persona service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaSchema {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
    pub name: String,
    pub role: String,
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    pub traits: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    pub tones: Vec<String>,
    #[serde(default)]
    pub background_story: String,
    #[serde(default)]
    pub created_by: String,
    #[serde(default)]
    pub updated_by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_version: Option<String>,
}
