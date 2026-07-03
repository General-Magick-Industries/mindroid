use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::auth::Auth;
use crate::core::context::Context;
use crate::error::{MindroidError, Result};
use crate::models::{LlmMessage, SenderType};
use crate::pipeline::PipelineStage;
#[cfg(feature = "transport-audio")]
use crate::pipeline::extensions::TextInput;

/// A pipeline stage that delegates persona prompt construction to MagickMind's
/// `PreparePersona` endpoint and uses the returned system prompt verbatim.
///
/// Unlike [`PersonaContextBuilder`](super::PersonaContextBuilder), which fetches
/// structured persona + effective-personality data and formats the prompt
/// in-process, this stage offloads *all* formatting to MagickMind. MagickMind fans out
/// to the persona and runtime services over gRPC, runs its own
/// `buildSystemPrompt` / `formatEffectiveTrait` (trait banding, structured
/// trait-ref parsing), and returns a finished `system_prompt` string.
///
/// Use this when MagickMind is the single source of truth for prompt rendering.
///
/// `POST {base_url}/v1/persona/{persona_id}/prepare`
///
/// ## Persona selection (constant vs. per-message)
///
/// The configured `persona_id` is a **default**. If an inbound message carries a
/// [`PersonaId`] extension in the pipeline [`Context`], that id is used instead —
/// letting a single stage serve many personas (e.g. a server whose mobile clients
/// each send their own persona id). The application sets the extension; the SDK
/// only reads it.
///
/// ## Caching
///
/// Prepared prompts are cached per `(persona_id, user_id)` for [`with_ttl`] (default
/// 10 minutes) so voice/chat turns don't hit MagickMind on every message.
///
/// [`with_ttl`]: MagickmindPersonaStage::with_ttl
pub struct MagickmindPersonaStage {
    http: reqwest::Client,
    base_url: String,
    /// Default persona id, used when no [`PersonaId`] extension is in context.
    persona_id: String,
    identity: Arc<dyn Auth>,
    /// Pre-fetched conversation history injected at construction time.
    history: Arc<Vec<LlmMessage>>,
    /// How long a prepared prompt stays valid in [`Self::cache`].
    ttl: Duration,
    /// Cache of prepared system prompts keyed by `(persona_id, user_id)`.
    cache: Mutex<HashMap<(String, Option<String>), CacheEntry>>,
}

/// A cached prepared prompt plus the instant it was fetched.
struct CacheEntry {
    prompt: String,
    fetched_at: Instant,
}

/// Per-message persona selector stored in the pipeline [`Context`].
///
/// When present, [`MagickmindPersonaStage`] uses this persona id instead of its
/// configured default — enabling one stage to serve many personas. The
/// application is responsible for setting it (e.g. extracting it from inbound
/// message metadata); the SDK only reads it. Mirrors `CanonicalUserId` from the
/// identity module.
#[derive(Debug, Clone)]
pub struct PersonaId(pub String);

impl MagickmindPersonaStage {
    /// Create a new `MagickmindPersonaStage`.
    ///
    /// No network call is made at construction time — MagickMind computes the
    /// entire prompt per-request in `process()`.
    pub fn new(base_url: &str, persona_id: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            persona_id: persona_id.to_string(),
            identity,
            history: Arc::new(Vec::new()),
            ttl: Duration::from_secs(Self::DEFAULT_TTL_SECS),
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Default prepared-prompt cache TTL in seconds (10 minutes).
    pub const DEFAULT_TTL_SECS: u64 = 600;

    /// Provide conversation history for inclusion in the LLM prompt.
    pub fn with_history(mut self, history: Arc<Vec<LlmMessage>>) -> Self {
        self.history = history;
        self
    }

    /// Override the prepared-prompt cache TTL (default 10 minutes).
    ///
    /// A TTL of zero disables caching (every message re-fetches from MagickMind).
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Resolve the system prompt for `persona_id`/`user_id`, serving a fresh
    /// cache entry when available and otherwise fetching from MagickMind.
    async fn resolve_prompt(&self, persona_id: &str, user_id: Option<&str>) -> Result<String> {
        let key = (persona_id.to_string(), user_id.map(str::to_string));

        // Fast path: fresh cache hit. Never hold the lock across the await below.
        {
            let cache = self.cache.lock().expect("persona cache mutex poisoned");
            if let Some(entry) = cache.get(&key)
                && entry.fetched_at.elapsed() < self.ttl
            {
                debug!("MagickmindPersonaStage: cache hit for {key:?}");
                return Ok(entry.prompt.clone());
            }
        }

        // Miss or stale: fetch from MagickMind, then populate the cache.
        let prompt = self.prepare(persona_id, user_id).await?;
        {
            let mut cache = self.cache.lock().expect("persona cache mutex poisoned");
            cache.insert(
                key,
                CacheEntry {
                    prompt: prompt.clone(),
                    fetched_at: Instant::now(),
                },
            );
        }
        Ok(prompt)
    }

    /// Call MagickMind's prepare endpoint and return the finished system prompt.
    async fn prepare(&self, persona_id: &str, user_id: Option<&str>) -> Result<String> {
        let url = {
            let mut u = reqwest::Url::parse(&self.base_url).map_err(|e| MindroidError::Api {
                message: format!("invalid base_url: {e}"),
                status_code: None,
            })?;
            u.path_segments_mut()
                .map_err(|_| MindroidError::Api {
                    message: "base_url cannot be a base URL".to_string(),
                    status_code: None,
                })?
                .extend(&["v1", "persona", persona_id, "prepare"]);
            u
        };
        let headers = crate::auth::build_auth_header_map(self.identity.as_ref()).await?;
        let body = PreparePersonaRequest { user_id };

        let resp = self
            .http
            .post(url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| MindroidError::Api {
                message: e.to_string(),
                status_code: None,
            })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(MindroidError::Api {
                message: format!("Failed to prepare persona {persona_id}: {text}"),
                status_code: Some(status.as_u16()),
            });
        }

        let prepared: PreparePersonaResponse =
            resp.json().await.map_err(|e| MindroidError::Api {
                message: format!("error decoding prepare-persona response: {e}"),
                status_code: None,
            })?;
        Ok(prepared.system_prompt)
    }
}

#[async_trait]
impl PipelineStage for MagickmindPersonaStage {
    fn name(&self) -> &str {
        "MagickmindPersonaStage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        // Determine user_id for dyadic blending (only for user-sent messages).
        // Prefer canonical user ID from identity resolution if available.
        let canonical_id: Option<String>;
        #[cfg(feature = "identity")]
        {
            canonical_id = ctx
                .get_ext::<crate::identity::CanonicalUserId>()
                .map(|c| c.0.clone());
        }
        #[cfg(not(feature = "identity"))]
        {
            canonical_id = None;
        }

        let user_id = if ctx.message.sender_type == SenderType::User {
            canonical_id
                .as_deref()
                .or(Some(ctx.message.sender_id.as_str()))
        } else {
            None
        };

        // Resolve persona id: prefer a per-message `PersonaId` extension (set by
        // the application), else fall back to the configured default.
        let persona_id = ctx
            .get_ext::<PersonaId>()
            .map(|p| p.0.clone())
            .unwrap_or_else(|| self.persona_id.clone());

        debug!("MagickmindPersonaStage: preparing persona={persona_id} user={user_id:?}");
        let system_prompt = self.resolve_prompt(&persona_id, user_id).await?;

        // Assemble LLM messages: system prompt first, then history, then user message.
        let mut messages = vec![LlmMessage::system(&system_prompt)];
        messages.extend(self.history.as_ref().clone());

        #[cfg(feature = "transport-audio")]
        let user_text = ctx
            .get_ext::<TextInput>()
            .map(|t| t.0.as_str())
            .unwrap_or(&ctx.message.content);
        #[cfg(not(feature = "transport-audio"))]
        let user_text = &ctx.message.content;

        messages.push(LlmMessage::user(user_text));

        debug!(
            "MagickmindPersonaStage: {} history messages, {} total llm_messages",
            self.history.len(),
            messages.len(),
        );

        ctx.llm_messages = messages;

        Ok(())
    }
}

/// Request body for `POST /v1/persona/{id}/prepare`.
///
/// Mirrors MagickMind's `PreparePersonaRequest` (the `id` is a path segment).
#[derive(Serialize)]
struct PreparePersonaRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<&'a str>,
}

/// Response body for `POST /v1/persona/{id}/prepare`.
///
/// Mirrors MagickMind's `PreparePersonaResponse`.
#[derive(Deserialize)]
struct PreparePersonaResponse {
    system_prompt: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_omits_user_id_when_absent() {
        let body = PreparePersonaRequest { user_id: None };
        assert_eq!(serde_json::to_string(&body).unwrap(), "{}");
    }

    #[test]
    fn request_includes_user_id_when_present() {
        let body = PreparePersonaRequest {
            user_id: Some("user-123"),
        };
        assert_eq!(
            serde_json::to_string(&body).unwrap(),
            r#"{"user_id":"user-123"}"#
        );
    }

    #[test]
    fn response_decodes_system_prompt() {
        let resp: PreparePersonaResponse =
            serde_json::from_str(r#"{"system_prompt":"You are Aria."}"#).unwrap();
        assert_eq!(resp.system_prompt, "You are Aria.");
    }

    #[test]
    fn default_ttl_is_ten_minutes() {
        let stage = MagickmindPersonaStage::new(
            "https://x",
            "p1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        );
        assert_eq!(stage.ttl, Duration::from_secs(600));
    }

    #[test]
    fn with_ttl_overrides_default() {
        let stage = MagickmindPersonaStage::new(
            "https://x",
            "p1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        )
        .with_ttl(Duration::from_secs(30));
        assert_eq!(stage.ttl, Duration::from_secs(30));
    }
}
