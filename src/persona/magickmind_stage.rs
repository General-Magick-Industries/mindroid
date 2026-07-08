use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::auth::Auth;
use crate::core::context::Context;
use crate::error::{MindroidError, Result};
use crate::models::LlmMessage;
use crate::pipeline::PipelineStage;

/// A pipeline stage that delegates persona prompt construction to MagickMind's
/// prepare endpoint and uses the returned system prompt verbatim.
///
/// Unlike [`PersonaContextBuilder`](super::PersonaContextBuilder), which fetches
/// structured persona + effective-personality data and formats the prompt
/// in-process, this stage offloads *all* formatting to MagickMind. The server
/// computes the effective personality — including trait banding (e.g. "very
/// high", "moderate") — and returns a finished `system_prompt` string.
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
/// ## Caching and degradation
///
/// Prepared prompts are cached per `(persona_id, user_id)` for [`with_ttl`]
/// (default 10 minutes) so voice/chat turns don't hit MagickMind on every
/// message. If a re-fetch fails, the last-good (stale) prompt is served so a
/// persona-service outage degrades gracefully instead of dropping messages.
/// With a TTL of zero, caching is disabled entirely: every message fetches
/// fresh, and a fetch failure fails the message (there is no stale entry to
/// fall back to).
///
/// ## Transport security
///
/// Auth headers accompany every prepare request, so plaintext `http://` base
/// URLs are refused unless [`with_allow_insecure`] (or
/// `persona.allow_insecure = true` in config) explicitly opts in for local
/// development.
///
/// [`with_ttl`]: MagickmindPersonaStage::with_ttl
/// [`with_allow_insecure`]: MagickmindPersonaStage::with_allow_insecure
pub struct MagickmindPersonaStage {
    http: reqwest::Client,
    base_url: String,
    /// Default persona id, used when no [`PersonaId`] extension is in context.
    persona_id: String,
    identity: Arc<dyn Auth>,
    /// Fallback conversation history injected at construction time. A
    /// per-request [`ConversationHistory`](super::ConversationHistory)
    /// extension takes precedence.
    history: Arc<Vec<LlmMessage>>,
    /// How long a prepared prompt stays valid in [`Self::cache`].
    ttl: Duration,
    /// Cache of prepared system prompts keyed by `(persona_id, user_id)`.
    cache: Mutex<HashMap<(String, Option<String>), CacheEntry>>,
    /// Permit sending auth headers over plaintext `http://` (local dev only).
    allow_insecure: bool,
}

/// A cached prepared prompt plus the instant it was fetched.
struct CacheEntry {
    prompt: String,
    fetched_at: Instant,
}

type CacheKey = (String, Option<String>);

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
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(Self::HTTP_TIMEOUT_SECS))
                .build()
                .expect("failed to build HTTP client"),
            base_url: base_url.trim_end_matches('/').to_string(),
            persona_id: persona_id.to_string(),
            identity,
            history: Arc::new(Vec::new()),
            ttl: Duration::from_secs(Self::DEFAULT_TTL_SECS),
            cache: Mutex::new(HashMap::new()),
            allow_insecure: false,
        }
    }

    /// Default prepared-prompt cache TTL in seconds (10 minutes).
    pub const DEFAULT_TTL_SECS: u64 = 600;

    /// Total HTTP request timeout in seconds. The prepare call gates message
    /// processing, so a hung server must not stall pipelines indefinitely.
    pub const HTTP_TIMEOUT_SECS: u64 = 10;

    /// Evict expired cache entries once the cache grows beyond this size.
    const CACHE_SWEEP_THRESHOLD: usize = 200;

    /// Provide fallback conversation history for inclusion in the LLM prompt.
    ///
    /// A per-request [`ConversationHistory`](super::ConversationHistory)
    /// extension in the pipeline [`Context`] takes precedence over this.
    pub fn with_history(mut self, history: Arc<Vec<LlmMessage>>) -> Self {
        self.history = history;
        self
    }

    /// Override the prepared-prompt cache TTL (default 10 minutes).
    ///
    /// A TTL of zero disables caching: every message re-fetches from
    /// MagickMind, nothing is stored, and there is no stale fallback.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Permit sending auth headers over plaintext `http://` (local development only).
    pub fn with_allow_insecure(mut self, allow_insecure: bool) -> Self {
        self.allow_insecure = allow_insecure;
        self
    }

    /// Resolve the system prompt for `persona_id`/`user_id`: fresh cache entry
    /// when available, otherwise fetch from MagickMind, and on fetch failure
    /// degrade to the last-good (stale) entry.
    async fn resolve_prompt(&self, persona_id: &str, user_id: Option<&str>) -> Result<String> {
        let key: CacheKey = (persona_id.to_string(), user_id.map(str::to_string));

        if let Some(prompt) = self.cache_get_fresh(&key) {
            debug!("MagickmindPersonaStage: cache hit for {key:?}");
            return Ok(prompt);
        }

        match self.prepare(persona_id, user_id).await {
            Ok(prompt) => {
                self.cache_insert(key, prompt.clone());
                Ok(prompt)
            }
            Err(e) => {
                // Persona service unavailable: serve the last-good prompt
                // rather than dropping the message.
                if let Some(stale) = self.cache_get_any(&key) {
                    warn!(
                        "MagickmindPersonaStage: prepare failed for {key:?}, \
                         serving stale cached prompt: {e}"
                    );
                    Ok(stale)
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Get a cache entry that is still within TTL. Never evicts — expired
    /// entries are kept as stale fallbacks until the write-path sweep.
    fn cache_get_fresh(&self, key: &CacheKey) -> Option<String> {
        let cache = self.cache.lock().expect("persona cache mutex poisoned");
        cache
            .get(key)
            .filter(|e| e.fetched_at.elapsed() < self.ttl)
            .map(|e| e.prompt.clone())
    }

    /// Get a cache entry regardless of age (stale-serve fallback).
    fn cache_get_any(&self, key: &CacheKey) -> Option<String> {
        let cache = self.cache.lock().expect("persona cache mutex poisoned");
        cache.get(key).map(|e| e.prompt.clone())
    }

    /// Insert a prepared prompt. Skips insertion when caching is disabled
    /// (zero TTL) and sweeps expired entries once the cache grows beyond
    /// [`Self::CACHE_SWEEP_THRESHOLD`].
    fn cache_insert(&self, key: CacheKey, prompt: String) {
        if self.ttl.is_zero() {
            return;
        }

        let mut cache = self.cache.lock().expect("persona cache mutex poisoned");
        if cache.len() > Self::CACHE_SWEEP_THRESHOLD {
            cache.retain(|_, e| e.fetched_at.elapsed() < self.ttl);
        }
        cache.insert(
            key,
            CacheEntry {
                prompt,
                fetched_at: Instant::now(),
            },
        );
    }

    /// Call MagickMind's prepare endpoint and return the finished system prompt.
    async fn prepare(&self, persona_id: &str, user_id: Option<&str>) -> Result<String> {
        // Auth headers ride on every prepare request — refuse to send them
        // over an unencrypted connection unless explicitly opted in.
        if self.base_url.starts_with("http://") && !self.allow_insecure {
            return Err(MindroidError::Api {
                message: format!(
                    "refusing to send auth headers over plaintext {}: use https://, \
                     or set persona.allow_insecure = true for local development",
                    self.base_url
                ),
                status_code: None,
            });
        }

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
        // User id for dyadic blending (only for user-sent messages).
        let user_id = super::resolve_user_id(ctx);

        // Resolve persona id: prefer a per-message `PersonaId` extension (set by
        // the application), else fall back to the configured default.
        let persona_id = ctx
            .get_ext::<PersonaId>()
            .map(|p| p.0.clone())
            .unwrap_or_else(|| self.persona_id.clone());

        debug!("MagickmindPersonaStage: preparing persona={persona_id} user={user_id:?}");
        let system_prompt = self.resolve_prompt(&persona_id, user_id.as_deref()).await?;

        let messages = super::assemble_llm_messages(ctx, &system_prompt, &self.history);

        debug!(
            "MagickmindPersonaStage: {} total llm_messages",
            messages.len(),
        );

        ctx.llm_messages = messages;

        Ok(())
    }
}

/// Request body for `POST /v1/persona/{id}/prepare`.
#[derive(Serialize)]
struct PreparePersonaRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<&'a str>,
}

/// Response body for `POST /v1/persona/{id}/prepare`.
#[derive(Deserialize)]
struct PreparePersonaResponse {
    system_prompt: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(ttl: Duration) -> MagickmindPersonaStage {
        MagickmindPersonaStage::new(
            "https://x",
            "p1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        )
        .with_ttl(ttl)
    }

    fn key(persona: &str, user: Option<&str>) -> CacheKey {
        (persona.to_string(), user.map(str::to_string))
    }

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
        let s = MagickmindPersonaStage::new(
            "https://x",
            "p1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        );
        assert_eq!(s.ttl, Duration::from_secs(600));
    }

    #[test]
    fn with_ttl_overrides_default() {
        assert_eq!(stage(Duration::from_secs(30)).ttl, Duration::from_secs(30));
    }

    #[test]
    fn zero_ttl_never_inserts() {
        let s = stage(Duration::ZERO);
        s.cache_insert(key("p1", Some("u1")), "prompt".into());
        assert!(s.cache.lock().unwrap().is_empty());
    }

    #[test]
    fn fresh_entry_is_served() {
        let s = stage(Duration::from_secs(60));
        s.cache_insert(key("p1", Some("u1")), "prompt".into());
        assert_eq!(
            s.cache_get_fresh(&key("p1", Some("u1"))).as_deref(),
            Some("prompt")
        );
    }

    #[test]
    fn expired_entry_is_not_fresh_but_serves_stale() {
        let s = stage(Duration::from_millis(1));
        s.cache_insert(key("p1", None), "prompt".into());
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(s.cache_get_fresh(&key("p1", None)), None);
        assert_eq!(s.cache_get_any(&key("p1", None)).as_deref(), Some("prompt"));
    }

    #[test]
    fn insert_sweeps_expired_entries_beyond_threshold() {
        let s = stage(Duration::from_millis(1));
        for i in 0..=MagickmindPersonaStage::CACHE_SWEEP_THRESHOLD {
            s.cache_insert(key(&format!("p{i}"), None), "prompt".into());
        }
        std::thread::sleep(Duration::from_millis(5));
        s.cache_insert(key("fresh", None), "prompt".into());
        // All expired entries were swept; only the newest insert survives.
        assert_eq!(s.cache.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn prepare_refuses_plaintext_http() {
        let s = MagickmindPersonaStage::new(
            "http://persona.internal",
            "p1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        );
        let err = s.prepare("p1", None).await.unwrap_err();
        assert!(err.to_string().contains("plaintext"), "got: {err}");
    }

    #[tokio::test]
    async fn prepare_allows_plaintext_http_with_explicit_flag() {
        // With the flag set, the scheme check passes; the request then fails
        // at the network layer (no server), which proves we got past the guard.
        let s = MagickmindPersonaStage::new(
            "http://127.0.0.1:1",
            "p1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        )
        .with_allow_insecure(true);
        let err = s.prepare("p1", None).await.unwrap_err();
        assert!(!err.to_string().contains("plaintext"), "got: {err}");
    }
}
