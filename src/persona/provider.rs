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

    /// Whether this provider returns a server-assembled prompt.
    ///
    /// When `true`, `PersonaContextBuilder` skips the persona-schema fetch at
    /// construction and calls [`Self::prepared_prompt`] per request. Providers
    /// overriding `prepared_prompt` must override this too.
    fn is_prepared(&self) -> bool {
        false
    }

    /// Return a fully assembled system prompt, if this provider can produce one.
    ///
    /// Providers backed by a server that blends and formats the prompt itself
    /// override this. The default returns `None`, meaning "I only supply raw
    /// persona data — assemble the prompt client-side from `get_persona` and
    /// `get_effective_personality`."
    ///
    /// When this returns `Some`, `PersonaContextBuilder` uses the prompt
    /// verbatim and skips both of the above calls entirely.
    async fn prepared_prompt(
        &self,
        _id: &str,
        _user_id: Option<&str>,
    ) -> Result<Option<PreparedPrompt>> {
        Ok(None)
    }
}

/// A server-assembled system prompt plus its cache metadata.
#[derive(Debug, Clone)]
pub struct PreparedPrompt {
    pub system_prompt: String,
    /// Cache TTL in seconds. `0` means "do not cache".
    pub ttl_seconds: u64,
}
