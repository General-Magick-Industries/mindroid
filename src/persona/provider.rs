use async_trait::async_trait;

use crate::error::Result;

use super::models::{EffectivePersonalityResponse, PersonaSchema};

/// Abstraction over persona data sources (remote API, local files, etc.).
///
/// Implementations must be thread-safe (`Send + Sync`) for use behind `Arc`.
#[async_trait]
pub trait PersonaProvider: Send + Sync {
    /// Human-readable name for logging (e.g. "magickmind", "local").
    fn name(&self) -> &str;

    /// Fetch the static persona definition (name, role, tones, background story).
    async fn get_persona(&self, persona_id: &str) -> Result<PersonaSchema>;

    /// Fetch the effective (blended) personality, optionally scoped to a user
    /// for dyadic adaptation.
    async fn get_effective_personality(
        &self,
        persona_id: &str,
        user_id: Option<&str>,
    ) -> Result<EffectivePersonalityResponse>;
}
