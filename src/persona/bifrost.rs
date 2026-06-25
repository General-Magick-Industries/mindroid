use async_trait::async_trait;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::auth::Auth;
use crate::core::context::Context;
use crate::error::{MindroidError, Result};
use crate::models::{LlmMessage, SenderType};
use crate::pipeline::PipelineStage;
#[cfg(feature = "transport-audio")]
use crate::pipeline::extensions::TextInput;

/// A pipeline stage that delegates persona prompt construction to Bifrost's
/// `PreparePersona` endpoint and uses the returned system prompt verbatim.
///
/// Unlike [`PersonaContextBuilder`](super::PersonaContextBuilder), which fetches
/// structured persona + effective-personality data and formats the prompt
/// in-process, this stage offloads *all* formatting to Bifrost. Bifrost fans out
/// to the persona and runtime services over gRPC, runs its own
/// `buildSystemPrompt` / `formatEffectiveTrait` (trait banding, structured
/// trait-ref parsing), and returns a finished `system_prompt` string.
///
/// Use this when Bifrost is the single source of truth for prompt rendering.
///
/// `POST {base_url}/v1/persona/{persona_id}/prepare`
pub struct BifrostPersonaStage {
    http: reqwest::Client,
    base_url: String,
    persona_id: String,
    identity: Arc<dyn Auth>,
    /// Pre-fetched conversation history injected at construction time.
    history: Arc<Vec<LlmMessage>>,
}

impl BifrostPersonaStage {
    /// Create a new `BifrostPersonaStage`.
    ///
    /// No network call is made at construction time — Bifrost computes the
    /// entire prompt per-request in `process()`.
    pub fn new(base_url: &str, persona_id: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            persona_id: persona_id.to_string(),
            identity,
            history: Arc::new(Vec::new()),
        }
    }

    /// Provide conversation history for inclusion in the LLM prompt.
    pub fn with_history(mut self, history: Arc<Vec<LlmMessage>>) -> Self {
        self.history = history;
        self
    }

    /// Call Bifrost's prepare endpoint and return the finished system prompt.
    async fn prepare(&self, user_id: Option<&str>) -> Result<String> {
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
                .extend(&["v1", "persona", &self.persona_id, "prepare"]);
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
                message: format!("Failed to prepare persona {}: {text}", self.persona_id),
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
impl PipelineStage for BifrostPersonaStage {
    fn name(&self) -> &str {
        "BifrostPersonaStage"
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

        debug!(
            "BifrostPersonaStage: preparing persona={} user={:?}",
            self.persona_id, user_id
        );
        let system_prompt = self.prepare(user_id).await?;

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
            "BifrostPersonaStage: {} history messages, {} total llm_messages",
            self.history.len(),
            messages.len(),
        );

        ctx.llm_messages = messages;

        Ok(())
    }
}

/// Request body for `POST /v1/persona/{id}/prepare`.
///
/// Mirrors Bifrost's `PreparePersonaRequest` (the `id` is a path segment).
#[derive(Serialize)]
struct PreparePersonaRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<&'a str>,
}

/// Response body for `POST /v1/persona/{id}/prepare`.
///
/// Mirrors Bifrost's `PreparePersonaResponse`.
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
}
