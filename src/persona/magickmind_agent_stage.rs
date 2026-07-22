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

/// Which credential the caller holds, and therefore which prepare route the
/// stage uses.
///
/// Bifrost split one preparation behind two routes so no handler has to ask
/// which caller it is serving. A service user names the agent in the path; an
/// agent reaches the same logic as itself, with no id to supply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersonaCaller {
    /// Service-user credential — `POST /v1/end-users/{agent_id}/persona/prepare`.
    /// The path segment is an **agent id**, not a persona id.
    ServiceUser,
    /// The agent's own end-user JWT — `POST /v1/end-user/persona/prepare`. No id
    /// is sent; the agent is the token subject.
    EndUser,
}

/// A pipeline stage that delegates persona prompt construction to MagickMind's
/// **agent-scoped** prepare endpoint and uses the returned prompt verbatim.
///
/// Like [`MagickmindPersonaStage`](super::MagickmindPersonaStage), the server
/// computes a finished `system_prompt`. The difference is the identifier: this
/// stage is keyed by **agent id**, not persona id, and follows the credential:
///
/// - [`PersonaCaller::ServiceUser`] — `POST /v1/end-users/{agent_id}/persona/prepare`.
///   Passing a persona id here yields a 404 ("Agent not found").
/// - [`PersonaCaller::EndUser`] — `POST /v1/end-user/persona/prepare`. The agent
///   is the token subject, so `agent_id` is neither required nor sent.
///
/// The service-user route also accepts an end-user token, but then pins the path
/// id to the token subject (403 on mismatch). Hold an end-user JWT and use
/// [`PersonaCaller::EndUser`] — it has no id to mismatch.
///
/// ## Caching and degradation
///
/// Identical policy to [`MagickmindPersonaStage`](super::MagickmindPersonaStage):
/// prepared prompts are cached per `(agent_key, user_id)` for [`with_ttl`]
/// (default 10 minutes), a failed re-fetch degrades to the last-good prompt, and
/// a zero TTL disables caching. On the end-user route the agent id is fixed by
/// the token, so the cache key uses a constant marker for it.
///
/// [`with_ttl`]: MagickmindAgentPersonaStage::with_ttl
pub struct MagickmindAgentPersonaStage {
    http: reqwest::Client,
    base_url: String,
    /// Default agent id for the service-user route. Ignored on the end-user
    /// route, where the agent is the token subject.
    agent_id: String,
    caller: PersonaCaller,
    identity: Arc<dyn Auth>,
    /// Fallback conversation history injected at construction time. A
    /// per-request [`ConversationHistory`](super::ConversationHistory)
    /// extension takes precedence.
    history: Arc<Vec<LlmMessage>>,
    ttl: Duration,
    cache: Mutex<HashMap<CacheKey, CacheEntry>>,
    allow_insecure: bool,
}

struct CacheEntry {
    prompt: String,
    fetched_at: Instant,
}

type CacheKey = (String, Option<String>);

/// The cache-key agent component on the end-user route, where no agent id is
/// sent and the subject is fixed by the token.
const END_USER_AGENT_KEY: &str = "\0end-user";

impl MagickmindAgentPersonaStage {
    /// Create a new stage, defaulting to the [`PersonaCaller::ServiceUser`] route.
    ///
    /// No network call is made at construction time.
    pub fn new(base_url: &str, agent_id: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(Self::HTTP_TIMEOUT_SECS))
                // A prepare endpoint has no reason to redirect, and reqwest's
                // cross-host header strip compares host and port but not scheme
                // — so a same-host https->http redirect would forward the
                // bearer token in cleartext. Refuse to follow at all.
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("failed to build HTTP client"),
            base_url: base_url.trim_end_matches('/').to_string(),
            agent_id: agent_id.to_string(),
            caller: PersonaCaller::ServiceUser,
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

    const CACHE_SWEEP_THRESHOLD: usize = 200;

    /// Select which route the stage uses. Defaults to [`PersonaCaller::ServiceUser`].
    pub fn with_caller(mut self, caller: PersonaCaller) -> Self {
        self.caller = caller;
        self
    }

    /// Provide fallback conversation history for inclusion in the LLM prompt.
    pub fn with_history(mut self, history: Arc<Vec<LlmMessage>>) -> Self {
        self.history = history;
        self
    }

    /// Override the prepared-prompt cache TTL (default 10 minutes).
    ///
    /// A TTL of zero disables caching: every message re-fetches, nothing is
    /// stored, and there is no stale fallback.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Permit sending auth headers over plaintext `http://` (local dev only).
    pub fn with_allow_insecure(mut self, allow_insecure: bool) -> Self {
        self.allow_insecure = allow_insecure;
        self
    }

    /// The cache-key agent component for the active route.
    fn agent_cache_key(&self) -> String {
        match self.caller {
            PersonaCaller::ServiceUser => self.agent_id.clone(),
            PersonaCaller::EndUser => END_USER_AGENT_KEY.to_string(),
        }
    }

    /// Resolve the system prompt: fresh cache entry, else fetch, else the
    /// last-good (stale) entry on failure.
    async fn resolve_prompt(&self, user_id: Option<&str>) -> Result<String> {
        let key: CacheKey = (self.agent_cache_key(), user_id.map(str::to_string));

        if let Some(prompt) = self.cache_get_fresh(&key) {
            debug!("MagickmindAgentPersonaStage: cache hit for {key:?}");
            return Ok(prompt);
        }

        match self.prepare(user_id).await {
            Ok(prompt) => {
                self.cache_insert(key, prompt.clone());
                Ok(prompt)
            }
            Err(e) => {
                if let Some(stale) = self.cache_get_any(&key) {
                    warn!(
                        "MagickmindAgentPersonaStage: prepare failed for {key:?}, \
                         serving stale cached prompt: {e}"
                    );
                    Ok(stale)
                } else {
                    Err(e)
                }
            }
        }
    }

    fn cache_get_fresh(&self, key: &CacheKey) -> Option<String> {
        let cache = self.cache.lock().expect("persona cache mutex poisoned");
        cache
            .get(key)
            .filter(|e| e.fetched_at.elapsed() < self.ttl)
            .map(|e| e.prompt.clone())
    }

    fn cache_get_any(&self, key: &CacheKey) -> Option<String> {
        let cache = self.cache.lock().expect("persona cache mutex poisoned");
        cache.get(key).map(|e| e.prompt.clone())
    }

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

    /// Build the prepare URL for the active route.
    fn prepare_url(&self) -> Result<reqwest::Url> {
        let mut u = reqwest::Url::parse(&self.base_url).map_err(|e| MindroidError::Api {
            message: format!("invalid base_url: {e}"),
            status_code: None,
        })?;
        {
            let mut segments = u.path_segments_mut().map_err(|_| MindroidError::Api {
                message: "base_url cannot be a base URL".to_string(),
                status_code: None,
            })?;
            match self.caller {
                PersonaCaller::ServiceUser => {
                    segments.extend(&["v1", "end-users", &self.agent_id, "persona", "prepare"]);
                }
                PersonaCaller::EndUser => {
                    segments.extend(&["v1", "end-user", "persona", "prepare"]);
                }
            }
        }
        Ok(u)
    }

    /// Call the prepare endpoint and return the finished system prompt.
    async fn prepare(&self, user_id: Option<&str>) -> Result<String> {
        // Defense in depth: the builder already refuses a non-TLS base_url at
        // startup, but a directly-constructed stage has not been through it.
        crate::core::net::require_secure_url(
            &self.base_url,
            self.allow_insecure,
            "persona.allow_insecure",
        )?;

        let url = self.prepare_url()?;
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
            let text = crate::core::net::error_excerpt(&resp.text().await.unwrap_or_default());
            let hint = match (self.caller, status.as_u16()) {
                (PersonaCaller::ServiceUser, 404) => {
                    " (is this an agent id? this route is keyed by agent, not persona)"
                }
                (PersonaCaller::ServiceUser, 403) => {
                    " (agent id does not match the token subject; holding an end-user JWT, \
                      configure auth.type = \"enduser\")"
                }
                (PersonaCaller::EndUser, 401) => {
                    " (this route needs an end-user JWT; with a service-user credential the \
                      agent id is named in the path instead)"
                }
                (PersonaCaller::EndUser, 403) => " (end-user token revoked or not permitted)",
                _ => "",
            };
            let subject = match self.caller {
                PersonaCaller::ServiceUser => format!("agent {}", self.agent_id),
                PersonaCaller::EndUser => "the calling agent".to_string(),
            };
            return Err(MindroidError::Api {
                message: format!("Failed to prepare persona for {subject}{hint}: {text}"),
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
impl PipelineStage for MagickmindAgentPersonaStage {
    fn name(&self) -> &str {
        "MagickmindAgentPersonaStage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let user_id = super::resolve_user_id(ctx);

        debug!(
            "MagickmindAgentPersonaStage: preparing agent={} caller={:?} user={user_id:?}",
            self.agent_id, self.caller
        );
        let system_prompt = self.resolve_prompt(user_id.as_deref()).await?;

        let messages = super::assemble_llm_messages(ctx, &system_prompt, &self.history);

        debug!(
            "MagickmindAgentPersonaStage: {} total llm_messages",
            messages.len(),
        );

        ctx.llm_messages = messages;

        Ok(())
    }
}

/// Request body for both prepare routes.
#[derive(Serialize)]
struct PreparePersonaRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<&'a str>,
}

/// Response body. The prepare endpoint returns richer metadata, but only the
/// finished prompt is used here.
#[derive(Deserialize)]
struct PreparePersonaResponse {
    system_prompt: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(caller: PersonaCaller) -> MagickmindAgentPersonaStage {
        MagickmindAgentPersonaStage::new(
            "https://x",
            "agent-1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        )
        .with_caller(caller)
    }

    #[test]
    fn service_user_route_carries_the_agent_id() {
        let url = stage(PersonaCaller::ServiceUser).prepare_url().unwrap();
        assert_eq!(url.path(), "/v1/end-users/agent-1/persona/prepare");
    }

    #[test]
    fn end_user_route_omits_the_agent_id() {
        let url = stage(PersonaCaller::EndUser).prepare_url().unwrap();
        assert_eq!(url.path(), "/v1/end-user/persona/prepare");
        assert!(!url.path().contains("agent-1"));
    }

    #[test]
    fn service_user_is_the_default() {
        let s = MagickmindAgentPersonaStage::new(
            "https://x",
            "agent-1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        );
        assert_eq!(s.caller, PersonaCaller::ServiceUser);
    }

    #[test]
    fn cache_key_is_route_scoped() {
        // Service-user keys by agent id; end-user by the fixed marker, so the two
        // routes never share a cache entry even at the same base_url.
        assert_eq!(
            stage(PersonaCaller::ServiceUser).agent_cache_key(),
            "agent-1"
        );
        assert_eq!(
            stage(PersonaCaller::EndUser).agent_cache_key(),
            END_USER_AGENT_KEY
        );
    }

    #[test]
    fn request_omits_user_id_when_absent() {
        let body = PreparePersonaRequest { user_id: None };
        assert_eq!(serde_json::to_string(&body).unwrap(), "{}");
    }

    #[test]
    fn response_decodes_system_prompt() {
        let resp: PreparePersonaResponse =
            serde_json::from_str(r#"{"system_prompt":"You are Aria.","agent_id":"a"}"#).unwrap();
        assert_eq!(resp.system_prompt, "You are Aria.");
    }

    #[tokio::test]
    async fn prepare_refuses_plaintext_http() {
        let s = MagickmindAgentPersonaStage::new(
            "http://persona.internal",
            "agent-1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        );
        let err = s.prepare(None).await.unwrap_err().to_string();
        assert!(err.contains("http://persona.internal"), "got: {err}");
        assert!(err.contains("persona.allow_insecure"), "got: {err}");
    }

    #[test]
    fn zero_ttl_never_inserts() {
        let s = stage(PersonaCaller::ServiceUser).with_ttl(Duration::ZERO);
        s.cache_insert(("agent-1".into(), None), "p".into());
        assert!(s.cache.lock().unwrap().is_empty());
    }
}
