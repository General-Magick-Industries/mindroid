use async_trait::async_trait;
use std::sync::Arc;

use serde::Serialize;

use crate::auth::Auth;
use crate::error::{MindroidError, Result};

use super::models::{EffectivePersonalityResponse, PersonaSchema, PreparedPersonaResponse};
use super::provider::{PersonaProvider, PreparedPrompt};

/// HTTP client for the magickmind end-user persona prepare endpoint.
///
/// `POST /v1/end-users/{agent_id}/persona/prepare`
///
/// Unlike [`super::MagickmindPersonaClientOld`], this returns a fully assembled
/// `system_prompt` — no client-side blending or formatting is needed.
///
/// The path segment is an **agent id**, not a persona id. Passing a persona id
/// yields a 404 ("Agent not found"), or a 403 if it collides with an end-user
/// in another tenant.
pub struct MagickmindPersonaClient {
    http: reqwest::Client,
    base_url: String,
    identity: Arc<dyn Auth>,
}

impl MagickmindPersonaClient {
    pub fn new(base_url: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            identity,
        }
    }

    /// Prepare the persona for `agent_id`, optionally scoped to a user for
    /// dyadic adaptation.
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
            u.path_segments_mut()
                .map_err(|_| MindroidError::Api {
                    message: "base_url cannot be a base URL".to_string(),
                    status_code: None,
                })?
                .extend(&["v1", "end-users", agent_id, "persona", "prepare"]);
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
            let hint = match status.as_u16() {
                404 => " (is this an agent id? the prepare route is keyed by agent, not persona)",
                403 => " (agent id not visible to these credentials — possible cross-tenant id)",
                _ => "",
            };
            return Err(MindroidError::Api {
                message: format!("Failed to prepare persona for agent {agent_id}{hint}: {text}"),
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
impl PersonaProvider for MagickmindPersonaClient {
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
                      use prepared_prompt(), or MagickmindPersonaClientOld for raw persona data"
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
                      use prepared_prompt(), or MagickmindPersonaClientOld for raw traits"
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
