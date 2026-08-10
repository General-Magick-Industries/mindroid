use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::auth::Auth;
use crate::core::context::Context;
use crate::core::net::{PreparedPromptCache, PromptCacheKey};
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
    /// Prepared prompts keyed by `(persona_id, user_id)`.
    cache: PreparedPromptCache,
    /// Permit sending auth headers over plaintext `http://` (local dev only).
    allow_insecure: bool,
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
            http: crate::core::net::secure_json_client(Duration::from_secs(
                Self::HTTP_TIMEOUT_SECS,
            )),
            base_url: base_url.trim_end_matches('/').to_string(),
            persona_id: persona_id.to_string(),
            identity,
            history: Arc::new(Vec::new()),
            cache: PreparedPromptCache::new(Duration::from_secs(Self::DEFAULT_TTL_SECS)),
            allow_insecure: false,
        }
    }

    /// Default prepared-prompt cache TTL in seconds (10 minutes).
    pub const DEFAULT_TTL_SECS: u64 = 600;

    /// Total HTTP request timeout in seconds. The prepare call gates message
    /// processing, so a hung server must not stall pipelines indefinitely.
    pub const HTTP_TIMEOUT_SECS: u64 = 10;

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
        self.cache.set_ttl(ttl);
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
        let key: PromptCacheKey = (persona_id.to_string(), user_id.map(str::to_string));

        // Log the persona component only — the key's other half is an end-user id.
        let scope = key.0.clone();

        if let Some(prompt) = self.cache.get_fresh(&key) {
            debug!("MagickmindPersonaStage: cache hit for persona={scope}");
            return Ok(prompt);
        }

        match self.prepare(persona_id, user_id).await {
            Ok(prompt) => {
                self.cache.insert(key, prompt.clone());
                Ok(prompt)
            }
            Err(e) => {
                // Persona service unavailable: serve the last-good prompt
                // rather than dropping the message.
                if let Some(stale) = self.cache.get_any(&key) {
                    warn!(
                        "MagickmindPersonaStage: prepare failed for persona={scope}, \
                         serving stale cached prompt: {e}"
                    );
                    Ok(stale)
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Call MagickMind's prepare endpoint and return the finished system prompt.
    async fn prepare(&self, persona_id: &str, user_id: Option<&str>) -> Result<String> {
        // Auth headers ride on every prepare request — refuse to send them
        // over an unencrypted connection unless explicitly opted in.
        // Defense in depth: the builder already refuses a non-TLS base_url at
        // startup, but a directly-constructed stage has not been through it.
        crate::core::net::require_secure_url(
            &self.base_url,
            self.allow_insecure,
            "persona.allow_insecure",
        )?;

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
        crate::core::net::note_auth_status(self.identity.as_ref(), status);
        if !status.is_success() {
            let text = crate::core::net::error_excerpt(&resp.text().await.unwrap_or_default());
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

        // `user_id` is deliberately not logged — see resolve_prompt. Record
        // only whether the prompt is personalized.
        debug!(
            "MagickmindPersonaStage: preparing persona={persona_id} personalized={}",
            user_id.is_some()
        );
        let system_prompt = self.resolve_prompt(&persona_id, user_id.as_deref()).await?;

        let messages = super::assemble_llm_messages(ctx, &system_prompt, &self.history);

        debug!(
            "MagickmindPersonaStage: {} total llm_messages",
            messages.len(),
        );

        let current_user = messages.len() - 1;
        ctx.llm_messages = messages;
        ctx.set(crate::pipeline::extensions::CurrentUserMessage(
            current_user,
        ));

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

    fn key(persona: &str, user: Option<&str>) -> PromptCacheKey {
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
        assert_eq!(s.cache.ttl(), Duration::from_secs(600));
    }

    #[test]
    fn with_ttl_overrides_default() {
        assert_eq!(
            stage(Duration::from_secs(30)).cache.ttl(),
            Duration::from_secs(30)
        );
    }

    // TTL, staleness and sweep behaviour live in core::net, shared with
    // MagickmindAgentPersonaStage. What follows is specific to this stage:
    // how resolve_prompt drives that cache.

    /// Graceful degradation: a prepare failure serves the last-good prompt
    /// rather than dropping the message.
    #[tokio::test]
    async fn stale_prompt_is_served_when_prepare_fails() {
        // Port 1 is unreachable, so prepare() always errors.
        let s = MagickmindPersonaStage::new(
            "https://127.0.0.1:1",
            "p1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        )
        .with_ttl(Duration::from_millis(1));

        s.cache.insert(key("p1", Some("u1")), "last good".into());
        std::thread::sleep(Duration::from_millis(5));

        let prompt = s.resolve_prompt("p1", Some("u1")).await.unwrap();
        assert_eq!(prompt, "last good");
    }

    /// With nothing cached, the failure must propagate.
    #[tokio::test]
    async fn prepare_failure_without_cache_propagates() {
        let s = MagickmindPersonaStage::new(
            "https://127.0.0.1:1",
            "p1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        );
        assert!(s.resolve_prompt("p1", Some("u1")).await.is_err());
    }

    #[tokio::test]
    async fn prepare_refuses_plaintext_http() {
        let s = MagickmindPersonaStage::new(
            "http://persona.internal",
            "p1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        );
        let err = s.prepare("p1", None).await.unwrap_err().to_string();
        assert!(err.contains("http://persona.internal"), "got: {err}");
        assert!(err.contains("persona.allow_insecure"), "got: {err}");
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
        // With the flag set the scheme check passes; the request then fails at
        // the network layer, which proves we got past the guard.
        let err = s.prepare("p1", None).await.unwrap_err().to_string();
        assert!(!err.contains("allow_insecure"), "got: {err}");
    }
}
