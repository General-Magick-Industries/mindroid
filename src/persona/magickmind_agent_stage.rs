use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::auth::Auth;
use crate::core::context::Context;
use crate::core::net::{PreparedPromptCache, PromptCacheKey};
use crate::error::{MindroidError, Result};
use crate::models::CredentialKind;
use crate::models::LlmMessage;
use crate::pipeline::PipelineStage;

/// A pipeline stage that delegates persona prompt construction to MagickMind's
/// **agent-scoped** prepare endpoint and uses the returned prompt verbatim.
///
/// Like [`MagickmindPersonaStage`](super::MagickmindPersonaStage), the server
/// computes a finished `system_prompt`. The difference is the identifier: this
/// stage is keyed by **agent id**, not persona id, and follows the credential:
///
/// - [`CredentialKind::ServiceUser`] — `POST /v1/end-users/{agent_id}/persona/prepare`.
///   Passing a persona id here yields a 404 ("Agent not found").
/// - [`CredentialKind::EndUser`] — `POST /v1/end-user/persona/prepare`. The agent
///   is the token subject, so `agent_id` is neither required nor sent.
///
/// The service-user route also accepts an end-user token, but then pins the path
/// id to the token subject (403 on mismatch). Hold an end-user JWT and use
/// [`CredentialKind::EndUser`] — it has no id to mismatch.
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
    credential_kind: CredentialKind,
    identity: Arc<dyn Auth>,
    /// Fallback conversation history injected at construction time. A
    /// per-request [`ConversationHistory`](super::ConversationHistory)
    /// extension takes precedence.
    history: Arc<Vec<LlmMessage>>,
    cache: PreparedPromptCache,
    allow_insecure: bool,
}

/// The cache-key agent component on the end-user route, where no agent id is
/// sent and the subject is fixed by the token.
const END_USER_AGENT_KEY: &str = "\0end-user";

impl MagickmindAgentPersonaStage {
    /// Create a new stage, defaulting to the [`CredentialKind::ServiceUser`] route.
    ///
    /// No network call is made at construction time.
    pub fn new(base_url: &str, agent_id: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            http: crate::core::net::secure_json_client(Duration::from_secs(
                Self::HTTP_TIMEOUT_SECS,
            )),
            base_url: base_url.trim_end_matches('/').to_string(),
            agent_id: agent_id.to_string(),
            credential_kind: CredentialKind::ServiceUser,
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

    /// Select which route the stage uses. Defaults to [`CredentialKind::ServiceUser`].
    pub fn with_credential_kind(mut self, credential_kind: CredentialKind) -> Self {
        self.credential_kind = credential_kind;
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
        self.cache.set_ttl(ttl);
        self
    }

    /// Permit sending auth headers over plaintext `http://` (local dev only).
    pub fn with_allow_insecure(mut self, allow_insecure: bool) -> Self {
        self.allow_insecure = allow_insecure;
        self
    }

    /// The cache-key agent component for the active route.
    ///
    /// The key's user component is `Option<String>`: `resolve_user_id` returns
    /// `None` only for non-user senders, so the `None` slot holds the agent's
    /// generic, non-personalized prompt and two real users never share it. That
    /// invariant depends on the server not personalizing a request that carries
    /// no `user_id` — if it ever does, this slot must be split per credential_kind.
    fn agent_cache_key(&self) -> String {
        match self.credential_kind {
            CredentialKind::ServiceUser => self.agent_id.clone(),
            CredentialKind::EndUser => END_USER_AGENT_KEY.to_string(),
        }
    }

    /// Resolve the system prompt: fresh cache entry, else fetch, else the
    /// last-good (stale) entry on failure.
    async fn resolve_prompt(&self, user_id: Option<&str>) -> Result<String> {
        let key: PromptCacheKey = (self.agent_cache_key(), user_id.map(str::to_string));

        // Log the agent component only. The key's other half is the end-user
        // id — operator-owned data is fine to log, but a user's identity is
        // not, and these lines can leave the device.
        let scope = key.0.clone();

        if let Some(prompt) = self.cache.get_fresh(&key) {
            debug!("MagickmindAgentPersonaStage: cache hit for agent={scope}");
            return Ok(prompt);
        }

        match self.prepare(user_id).await {
            Ok(prompt) => {
                self.cache.insert(key, prompt.clone());
                Ok(prompt)
            }
            Err(e) => {
                if let Some(stale) = self.cache.get_any(&key) {
                    warn!(
                        "MagickmindAgentPersonaStage: prepare failed for agent={scope}, \
                         serving stale cached prompt: {e}"
                    );
                    Ok(stale)
                } else {
                    Err(e)
                }
            }
        }
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
            match self.credential_kind {
                CredentialKind::ServiceUser => {
                    segments.extend(&["v1", "end-users", &self.agent_id, "persona", "prepare"]);
                }
                CredentialKind::EndUser => {
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
        crate::core::net::note_auth_status(self.identity.as_ref(), status);
        if !status.is_success() {
            let text = crate::core::net::error_excerpt(&resp.text().await.unwrap_or_default());
            let hint = match (self.credential_kind, status.as_u16()) {
                (CredentialKind::ServiceUser, 404) => {
                    " (is this an agent id? this route is keyed by agent, not persona)"
                }
                (CredentialKind::ServiceUser, 403) => {
                    " (agent id does not match the token subject; holding an end-user JWT, \
                      configure auth.type = \"enduser\")"
                }
                (CredentialKind::EndUser, 401) => {
                    " (this route needs an end-user JWT; with a service-user credential the \
                      agent id is named in the path instead)"
                }
                (CredentialKind::EndUser, 403) => " (end-user token revoked or not permitted)",
                _ => "",
            };
            let subject = match self.credential_kind {
                CredentialKind::ServiceUser => format!("agent {}", self.agent_id),
                CredentialKind::EndUser => "the calling agent".to_string(),
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

        // On the end-user route `agent_id` is empty by construction (the token
        // carries the subject), so name the route instead of logging a blank
        // field. `user_id` is deliberately omitted — see resolve_prompt.
        debug!(
            "MagickmindAgentPersonaStage: preparing agent={} credential_kind={:?} personalized={}",
            self.agent_cache_key(),
            self.credential_kind,
            user_id.is_some()
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

    fn stage(credential_kind: CredentialKind) -> MagickmindAgentPersonaStage {
        MagickmindAgentPersonaStage::new(
            "https://x",
            "agent-1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        )
        .with_credential_kind(credential_kind)
    }

    #[test]
    fn service_user_route_carries_the_agent_id() {
        let url = stage(CredentialKind::ServiceUser).prepare_url().unwrap();
        assert_eq!(url.path(), "/v1/end-users/agent-1/persona/prepare");
    }

    #[test]
    fn end_user_route_omits_the_agent_id() {
        let url = stage(CredentialKind::EndUser).prepare_url().unwrap();
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
        assert_eq!(s.credential_kind, CredentialKind::ServiceUser);
    }

    #[test]
    fn cache_key_is_route_scoped() {
        // Service-user keys by agent id; end-user by the fixed marker, so the two
        // routes never share a cache entry even at the same base_url.
        assert_eq!(
            stage(CredentialKind::ServiceUser).agent_cache_key(),
            "agent-1"
        );
        assert_eq!(
            stage(CredentialKind::EndUser).agent_cache_key(),
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

    fn key(agent: &str, user: Option<&str>) -> PromptCacheKey {
        (agent.to_string(), user.map(str::to_string))
    }

    // TTL, staleness and sweep behaviour are covered once in core::net, since
    // both prepare stages share PreparedPromptCache. The tests below cover what
    // is specific to this stage: how resolve_prompt drives that cache.

    /// The graceful-degradation guarantee: when prepare() fails, a previously
    /// cached prompt is served rather than failing the message.
    #[tokio::test]
    async fn stale_prompt_is_served_when_prepare_fails() {
        // Port 1 is unreachable, so prepare() always errors.
        let s = MagickmindAgentPersonaStage::new(
            "https://127.0.0.1:1",
            "agent-1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        )
        .with_ttl(Duration::from_millis(1));

        s.cache
            .insert(key("agent-1", Some("u1")), "last good".into());
        std::thread::sleep(Duration::from_millis(5));

        let prompt = s.resolve_prompt(Some("u1")).await.unwrap();
        assert_eq!(prompt, "last good");
    }

    /// With nothing cached, a prepare failure must propagate rather than
    /// silently yielding an empty prompt.
    #[tokio::test]
    async fn prepare_failure_without_cache_propagates() {
        let s = MagickmindAgentPersonaStage::new(
            "https://127.0.0.1:1",
            "agent-1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        );
        assert!(s.resolve_prompt(Some("u1")).await.is_err());
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

    /// With caching disabled there is no stale entry to fall back on, so a
    /// prepare failure must surface rather than serving a phantom prompt.
    #[tokio::test]
    async fn zero_ttl_disables_stale_serve() {
        let s = MagickmindAgentPersonaStage::new(
            "https://127.0.0.1:1",
            "agent-1",
            Arc::new(crate::auth::static_id::StaticAuth::new("t")),
        )
        .with_ttl(Duration::ZERO);

        s.cache.insert(key("agent-1", Some("u1")), "ignored".into());
        assert!(s.resolve_prompt(Some("u1")).await.is_err());
    }
}
