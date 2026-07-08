use async_trait::async_trait;
use std::sync::Arc;

use serde::Serialize;

use crate::auth::Auth;
use crate::error::{MindroidError, Result};

use super::models::{EffectivePersonalityResponse, PersonaSchema};
use super::provider::PersonaProvider;

/// HTTP client for the magickmind-api persona and runtime services.
///
/// Follows the same pattern as `MagickmindMemory` — uses `reqwest` + `Auth`
/// for authenticated requests against the magickmind REST API.
pub struct MagickmindPersonaClient {
    http: reqwest::Client,
    base_url: String,
    identity: Arc<dyn Auth>,
}

impl MagickmindPersonaClient {
    pub fn new(base_url: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            // Persona fetches gate message processing — a hung server must not
            // stall pipelines indefinitely.
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
            base_url: base_url.trim_end_matches('/').to_string(),
            identity,
        }
    }

    /// Fetch a persona definition by ID.
    ///
    /// `GET /v1/persona/{persona_id}`
    pub async fn get_persona(&self, persona_id: &str) -> Result<PersonaSchema> {
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
                .extend(&["v1", "persona", persona_id]);
            u
        };
        let headers = self.auth_headers().await?;

        let resp = self
            .http
            .get(url)
            .headers(headers)
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
                message: format!("Failed to get persona {persona_id}: {text}"),
                status_code: Some(status.as_u16()),
            });
        }

        let text = resp.text().await.map_err(|e| MindroidError::Api {
            message: format!("Failed to read persona response body: {e}"),
            status_code: None,
        })?;
        tracing::debug!("Persona API raw response: {text}");
        serde_json::from_str(&text).map_err(|e| MindroidError::Api {
            message: format!("error decoding persona response: {e}\nraw: {text}"),
            status_code: None,
        })
    }

    /// Fetch the effective (blended) personality from the runtime service.
    ///
    /// `GET /v1/runtime/effective-personality/{persona_id}[?user_id=...]`
    pub async fn get_effective_personality(
        &self,
        persona_id: &str,
        user_id: Option<&str>,
    ) -> Result<EffectivePersonalityResponse> {
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
                .extend(&["v1", "runtime", "effective-personality", persona_id]);
            u
        };
        let headers = self.auth_headers().await?;

        let mut req = self.http.get(url).headers(headers);
        if let Some(uid) = user_id {
            req = req.query(&[("user_id", uid)]);
        }

        let resp = req.send().await.map_err(|e| MindroidError::Api {
            message: e.to_string(),
            status_code: None,
        })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(MindroidError::Api {
                message: format!("Failed to get effective personality for {persona_id}: {text}"),
                status_code: Some(status.as_u16()),
            });
        }

        resp.json().await.map_err(|e| MindroidError::Api {
            message: e.to_string(),
            status_code: None,
        })
    }

    /// Invalidate the cached effective personality on the server.
    ///
    /// `POST /v1/runtime/invalidate-cache`
    pub async fn invalidate_cache(&self, persona_id: &str, user_id: Option<&str>) -> Result<()> {
        let url = format!("{}/v1/runtime/invalidate-cache", self.base_url);
        let headers = self.auth_headers().await?;
        let body = InvalidateCacheRequest {
            persona_id,
            user_id,
        };

        let resp = self
            .http
            .post(&url)
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
                message: format!("Failed to invalidate cache: {text}"),
                status_code: Some(status.as_u16()),
            });
        }

        Ok(())
    }

    async fn auth_headers(&self) -> Result<reqwest::header::HeaderMap> {
        crate::auth::build_auth_header_map(self.identity.as_ref()).await
    }
}

#[async_trait]
impl PersonaProvider for MagickmindPersonaClient {
    fn name(&self) -> &str {
        "magickmind"
    }

    async fn get_persona(&self, persona_id: &str) -> Result<PersonaSchema> {
        self.get_persona(persona_id).await
    }

    async fn get_effective_personality(
        &self,
        persona_id: &str,
        user_id: Option<&str>,
    ) -> Result<EffectivePersonalityResponse> {
        self.get_effective_personality(persona_id, user_id).await
    }
}

#[derive(Serialize)]
struct InvalidateCacheRequest<'a> {
    persona_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<&'a str>,
}
