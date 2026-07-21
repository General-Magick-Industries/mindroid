use async_trait::async_trait;
use std::sync::Arc;

use serde::Serialize;

use crate::auth::Auth;
use crate::error::{MindroidError, Result};

use super::models::{EffectivePersonalityResponse, PersonaSchema, PreparedPersonaResponse};
use super::provider::{PersonaProvider, PreparedPrompt};

/// Which credential the caller holds, and therefore which door it uses into
/// the same persona preparation.
///
/// Bifrost exposes one preparation behind two routes: a service user names the
/// agent in the path, while an agent reaches it as itself with no id to supply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersonaCaller {
    /// Service-user credential — `POST /v1/end-users/{agent_id}/persona/prepare`.
    ServiceUser,
    /// The agent's own end-user JWT — `POST /v1/end-user/persona/prepare`.
    EndUser,
}

/// HTTP client for the magickmind persona prepare endpoint.
///
/// Unlike [`super::MagickmindPersonaClient`], this returns a fully assembled
/// `system_prompt` — no client-side blending or formatting is needed.
///
/// The route follows the credential, set via [`Self::with_caller`]:
///
/// - [`PersonaCaller::ServiceUser`] (default) — `POST /v1/end-users/{agent_id}/persona/prepare`.
///   The path segment is an **agent id**, not a persona id; passing a persona
///   id yields a 404 ("Agent not found").
/// - [`PersonaCaller::EndUser`] — `POST /v1/end-user/persona/prepare`. The
///   agent is the token subject, so no id is sent and `agent_id` is ignored.
///
/// Each route takes one credential: bifrost split them so no handler has to ask
/// which kind of caller it is serving. An end-user token reaching the
/// service-user route is still pinned to its own subject (403 on mismatch), but
/// that is a backstop, not a supported path — hold an end-user JWT and use
/// [`PersonaCaller::EndUser`].
pub struct MagickmindAgentPersonaClient {
    http: reqwest::Client,
    base_url: String,
    identity: Arc<dyn Auth>,
    caller: PersonaCaller,
}

impl MagickmindAgentPersonaClient {
    pub fn new(base_url: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            identity,
            caller: PersonaCaller::ServiceUser,
        }
    }

    /// Select which route to use. Defaults to [`PersonaCaller::ServiceUser`].
    pub fn with_caller(mut self, caller: PersonaCaller) -> Self {
        self.caller = caller;
        self
    }

    /// Prepare the persona, optionally scoped to a user for dyadic adaptation.
    ///
    /// `agent_id` names the agent on the service-user route and is ignored on
    /// the end-user route, where the agent is the token subject.
    pub async fn prepare(
        &self,
        agent_id: &str,
        user_id: Option<&str>,
    ) -> Result<PreparedPersonaResponse> {
        let url = {
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
                        segments.extend(&["v1", "end-users", agent_id, "persona", "prepare"]);
                    }
                    PersonaCaller::EndUser => {
                        segments.extend(&["v1", "end-user", "persona", "prepare"]);
                    }
                }
            }
            u
        };
        let headers = crate::auth::build_auth_header_map(self.identity.as_ref()).await?;

        let resp = self
            .http
            .post(url)
            .headers(headers)
            .json(&PrepareRequest { user_id })
            .send()
            .await
            .map_err(|e| MindroidError::Api {
                message: e.to_string(),
                status_code: None,
            })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let hint = match (self.caller, status.as_u16()) {
                (PersonaCaller::ServiceUser, 404) => {
                    " (is this an agent id? the prepare route is keyed by agent, not persona)"
                }
                (PersonaCaller::ServiceUser, 403) => {
                    " (agent id does not match the token subject, or is not visible to these \
                      credentials; holding an end-user JWT, use PersonaCaller::EndUser)"
                }
                (PersonaCaller::EndUser, 401) => {
                    " (this route needs an end-user JWT; with a service-user credential use \
                      PersonaCaller::ServiceUser)"
                }
                (PersonaCaller::EndUser, 403) => " (end-user token revoked or not permitted)",
                _ => "",
            };
            let subject = match self.caller {
                PersonaCaller::ServiceUser => format!("agent {agent_id}"),
                PersonaCaller::EndUser => "the calling agent".to_string(),
            };
            return Err(MindroidError::Api {
                message: format!("Failed to prepare persona for {subject}{hint}: {text}"),
                status_code: Some(status.as_u16()),
            });
        }

        let text = resp.text().await.map_err(|e| MindroidError::Api {
            message: format!("Failed to read prepare response body: {e}"),
            status_code: None,
        })?;
        tracing::debug!("Prepare persona raw response: {text}");
        serde_json::from_str(&text).map_err(|e| MindroidError::Api {
            message: format!("error decoding prepare response: {e}\nraw: {text}"),
            status_code: None,
        })
    }
}

#[async_trait]
impl PersonaProvider for MagickmindAgentPersonaClient {
    fn name(&self) -> &str {
        "magickmind-prepared"
    }

    fn is_prepared(&self) -> bool {
        true
    }

    /// Not available on the prepare endpoint — it returns an assembled prompt,
    /// not a persona definition. Reached only if a caller bypasses
    /// `prepared_prompt`, which for this provider always returns `Some`.
    async fn get_persona(&self, _persona_id: &str) -> Result<PersonaSchema> {
        Err(MindroidError::Api {
            message: "the prepare endpoint returns no persona schema; \
                      use prepared_prompt(), or MagickmindPersonaClient for raw persona data"
                .to_string(),
            status_code: None,
        })
    }

    /// Not available on the prepare endpoint — traits are blended server-side
    /// and never returned individually.
    async fn get_effective_personality(
        &self,
        _persona_id: &str,
        _user_id: Option<&str>,
    ) -> Result<EffectivePersonalityResponse> {
        Err(MindroidError::Api {
            message: "the prepare endpoint returns no trait list; \
                      use prepared_prompt(), or MagickmindPersonaClient for raw traits"
                .to_string(),
            status_code: None,
        })
    }

    async fn prepared_prompt(
        &self,
        agent_id: &str,
        user_id: Option<&str>,
    ) -> Result<Option<PreparedPrompt>> {
        let resp = self.prepare(agent_id, user_id).await?;
        Ok(Some(PreparedPrompt {
            system_prompt: resp.system_prompt,
            ttl_seconds: resp.ttl_seconds,
        }))
    }
}

#[derive(Serialize)]
struct PrepareRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<&'a str>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::static_id::StaticAuth;

    fn client(caller: PersonaCaller) -> MagickmindAgentPersonaClient {
        MagickmindAgentPersonaClient::new("https://example.test", Arc::new(StaticAuth::new("t")))
            .with_caller(caller)
    }

    /// Mirrors the path construction in `prepare` so route selection is covered
    /// without a live server.
    fn route(c: &MagickmindAgentPersonaClient, agent_id: &str) -> String {
        let mut u = reqwest::Url::parse(&c.base_url).unwrap();
        {
            let mut s = u.path_segments_mut().unwrap();
            match c.caller {
                PersonaCaller::ServiceUser => {
                    s.extend(&["v1", "end-users", agent_id, "persona", "prepare"]);
                }
                PersonaCaller::EndUser => {
                    s.extend(&["v1", "end-user", "persona", "prepare"]);
                }
            }
        }
        u.path().to_string()
    }

    #[test]
    fn service_user_route_carries_the_agent_id() {
        let c = client(PersonaCaller::ServiceUser);
        assert_eq!(
            route(&c, "agent-1"),
            "/v1/end-users/agent-1/persona/prepare"
        );
    }

    #[test]
    fn end_user_route_omits_the_agent_id() {
        let c = client(PersonaCaller::EndUser);
        let path = route(&c, "agent-1");
        assert_eq!(path, "/v1/end-user/persona/prepare");
        assert!(
            !path.contains("agent-1"),
            "end-user route must not leak an id"
        );
    }

    #[test]
    fn service_user_is_the_default() {
        let c = MagickmindAgentPersonaClient::new(
            "https://example.test",
            Arc::new(StaticAuth::new("t")),
        );
        assert_eq!(c.caller, PersonaCaller::ServiceUser);
    }
}
