use async_trait::async_trait;
use std::sync::Arc;

use serde::Serialize;

use crate::auth::Auth;
use crate::error::{MindroidError, Result};
use crate::http::AuthenticatedHttpClient;

use super::models::{EffectivePersonalityResponse, PersonaSchema};
use super::provider::PersonaProvider;

/// HTTP client for the magickmind-api persona and runtime services.
pub struct MagickmindPersonaClient {
    client: AuthenticatedHttpClient,
}

impl MagickmindPersonaClient {
    pub fn new(base_url: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            client: AuthenticatedHttpClient::new(base_url, identity),
        }
    }

    fn build_url(&self, segments: &[&str]) -> Result<reqwest::Url> {
        let mut u =
            reqwest::Url::parse(self.client.base_url()).map_err(|e| MindroidError::Api {
                message: format!("invalid base_url: {e}"),
                status_code: None,
            })?;
        u.path_segments_mut()
            .map_err(|_| MindroidError::Api {
                message: "base_url cannot be a base URL".to_string(),
                status_code: None,
            })?
            .extend(segments);
        Ok(u)
    }

    /// Fetch a persona definition by ID.
    ///
    /// `GET /v1/persona/{persona_id}`
    pub async fn get_persona(&self, persona_id: &str) -> Result<PersonaSchema> {
        let url = self.build_url(&["v1", "persona", persona_id])?;

        let req = self.client.request(reqwest::Method::GET, url).await?;
        let resp = self.client.send_and_check(req).await?;

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
        let url = self.build_url(&["v1", "runtime", "effective-personality", persona_id])?;

        let mut req = self.client.request(reqwest::Method::GET, url).await?;
        if let Some(uid) = user_id {
            req = req.query(&[("user_id", uid)]);
        }

        let resp = self.client.send_and_check(req).await?;

        resp.json().await.map_err(|e| MindroidError::Api {
            message: e.to_string(),
            status_code: None,
        })
    }

    /// Invalidate the cached effective personality on the server.
    ///
    /// `POST /v1/runtime/invalidate-cache`
    pub async fn invalidate_cache(&self, persona_id: &str, user_id: Option<&str>) -> Result<()> {
        let body = InvalidateCacheRequest {
            persona_id,
            user_id,
        };

        let url = self.build_url(&["v1", "runtime", "invalidate-cache"])?;
        let req = self
            .client
            .request(reqwest::Method::POST, url)
            .await?
            .json(&body);

        self.client.send_and_check(req).await?;

        Ok(())
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
