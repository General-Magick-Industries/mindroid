use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::core::context::Context;
use crate::llm_client::{AuthStyle, LlmClient, LlmClientConfig};
use crate::pipeline::context::ContextProvider;
use crate::pipeline::stages::{GenericLlmProcessor, PostProcessor};
use crate::{Auth, LlmMessage, MindroidError, Pipeline, PipelineStage, Result};

// ── Magickmind API types ──────────────────────────────────────────────────────

#[derive(Serialize)]
struct MagickmindSaveRequest<'a> {
    sender_id: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reply_to_message_id: Option<&'a str>,
}

#[derive(Deserialize)]
struct MagickmindSaveResponse {
    id: Option<String>,
}

// ── Context Prepare API types (POST /v1/magickspaces/:id/context) ─────────────

#[derive(Serialize)]
struct PrepareContextRequest<'a> {
    participant_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_history: Option<ChatHistoryParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pelican: Option<PelicanParams<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    corpus: Option<CorpusParams<'a>>,
}

#[derive(Serialize)]
struct ChatHistoryParams {
    limit: i32,
}

#[derive(Serialize)]
struct PelicanParams<'a> {
    query: &'a str,
}

#[derive(Serialize)]
struct CorpusParams<'a> {
    query: &'a str,
}

#[derive(Deserialize)]
struct PrepareContextResponse {
    #[serde(default)]
    chat_history: Vec<ChatHistoryItem>,
    #[serde(default)]
    fetcher: String,
    #[serde(default)]
    corpus: Vec<CorpusItem>,
}

#[derive(Deserialize)]
struct ChatHistoryItem {
    #[serde(default)]
    sent_by_user_id: String,
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct CorpusItem {
    content: String,
}

// ── MagickmindClient ──────────────────────────────────────────────────────────

pub struct MagickmindClient {
    http: reqwest::Client,
    base_url: String,
    identity: Arc<dyn Auth>,
    api_key: Option<String>,
    credential_kind: crate::models::CredentialKind,
}

impl MagickmindClient {
    pub fn new(base_url: impl Into<String>, identity: Arc<dyn Auth>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            identity,
            api_key: None,
            credential_kind: crate::models::CredentialKind::ServiceUser,
        }
    }

    /// x-api-key for the pelican fetcher, sent only on context prepare (the one
    /// route that uses pelican).
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Credential surface for the magickspace routes. Default `ServiceUser`.
    pub fn with_credential_kind(mut self, credential_kind: crate::models::CredentialKind) -> Self {
        self.credential_kind = credential_kind;
        self
    }

    async fn auth_headers(&self) -> Result<reqwest::header::HeaderMap> {
        crate::auth::build_auth_header_map(self.identity.as_ref()).await
    }

    pub async fn prepare_context(
        &self,
        magickspace_id: &str,
        participant_id: &str,
        query: &str,
        config: &MagickmindContextConfig,
        exclude_sender: Option<&str>,
    ) -> Result<Vec<LlmMessage>> {
        // Service-user → tenant-scoped route; end-user JWT → membership-scoped
        // /v1/end-user/... route (participant = token subject).
        let url = match self.credential_kind {
            crate::models::CredentialKind::ServiceUser => format!(
                "{}/v1/magickspaces/{}/context",
                self.base_url, magickspace_id
            ),
            crate::models::CredentialKind::EndUser => format!(
                "{}/v1/end-user/magickspaces/{}/context",
                self.base_url, magickspace_id
            ),
        };
        let mut headers = self.auth_headers().await?;

        let body = PrepareContextRequest {
            participant_id,
            chat_history: if config.include_chat_history {
                Some(ChatHistoryParams {
                    limit: config.chat_history_limit,
                })
            } else {
                None
            },
            pelican: if config.include_pelican {
                Some(PelicanParams { query })
            } else {
                None
            },
            corpus: if config.include_corpus {
                Some(CorpusParams { query })
            } else {
                None
            },
        };

        if let Some(key) = &self.api_key {
            headers.insert(
                reqwest::header::HeaderName::from_static("x-api-key"),
                reqwest::header::HeaderValue::from_str(key).map_err(|e| MindroidError::Auth {
                    message: format!("Invalid api key header value: {e}"),
                    source: None,
                })?,
            );
        }

        debug!("MagickmindClient::prepare_context POST {url}");

        let resp = self
            .http
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| MindroidError::Api {
                message: format!("Magickmind prepare_context request failed: {e}"),
                status_code: None,
            })?;

        let status = resp.status();
        if !status.is_success() {
            return Err(MindroidError::Api {
                message: format!("Magickmind prepare_context returned {status}"),
                status_code: Some(status.as_u16()),
            });
        }

        let parsed: PrepareContextResponse = resp.json().await.map_err(|e| MindroidError::Api {
            message: format!("Failed to parse Magickmind prepare_context response: {e}"),
            status_code: None,
        })?;

        Ok(convert_context_response(parsed, exclude_sender))
    }

    pub async fn save_message(
        &self,
        magickspace_id: &str,
        sender_id: &str,
        content: &str,
        reply_to_message_id: Option<&str>,
    ) -> Result<Option<String>> {
        // Same credential split as prepare_context (end-user send route: MM-378).
        let url = match self.credential_kind {
            crate::models::CredentialKind::ServiceUser => format!(
                "{}/v1/magickspaces/{}/messages",
                self.base_url, magickspace_id
            ),
            crate::models::CredentialKind::EndUser => format!(
                "{}/v1/end-user/magickspaces/{}/messages",
                self.base_url, magickspace_id
            ),
        };
        let headers = self.auth_headers().await?;
        let body = MagickmindSaveRequest {
            sender_id,
            content,
            reply_to_message_id,
        };

        debug!("MagickmindClient::save_message POST {url}");

        let resp = self
            .http
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| MindroidError::Api {
                message: format!("Magickmind save_message request failed: {e}"),
                status_code: None,
            })?;

        let status = resp.status();
        if !status.is_success() {
            return Err(MindroidError::Api {
                message: "Magickmind save_message returned non-success status".to_string(),
                status_code: Some(status.as_u16()),
            });
        }

        let parsed: MagickmindSaveResponse = resp.json().await.map_err(|e| MindroidError::Api {
            message: format!("Failed to parse Magickmind save_message response: {e}"),
            status_code: None,
        })?;

        Ok(parsed.id)
    }
}

// ── MagickmindContext: ContextProvider implementation ────────────────────────

/// Configuration for the Magickmind context provider.
pub struct MagickmindContextConfig {
    /// Maximum number of chat history messages to retrieve.
    pub chat_history_limit: i32,
    /// Include chat history in context.
    pub include_chat_history: bool,
    /// Include pelican (episodic memory + web search) in context.
    pub include_pelican: bool,
    /// Include corpus (semantic document search) in context.
    pub include_corpus: bool,
}

impl Default for MagickmindContextConfig {
    fn default() -> Self {
        Self {
            chat_history_limit: 20,
            include_chat_history: true,
            include_pelican: true,
            include_corpus: false,
        }
    }
}

/// Fetches context from Magickmind's context preparation endpoint.
///
/// Calls `POST /v1/magickspaces/{channel_id}/context` using the message's
/// `channel_id` as the magickspace ID and `sender_id` as the participant.
///
/// ```ignore
/// use mindroid::{ContextPreparer, MagickmindContext};
///
/// let preparer = ContextPreparer::new()
///     .add_provider(MagickmindContext::new(magickmind.clone()));
///
/// let context = preparer.prepare(&message).await.into_messages();
/// ```
pub struct MagickmindContext {
    client: Arc<MagickmindClient>,
    config: MagickmindContextConfig,
    /// When set, chat history messages from this sender are excluded from context.
    /// Prevents the LLM from seeing its own previous responses (which confuses it).
    exclude_self_id: Option<String>,
}

impl MagickmindContext {
    pub fn new(client: Arc<MagickmindClient>) -> Self {
        Self {
            client,
            config: MagickmindContextConfig::default(),
            exclude_self_id: None,
        }
    }

    pub fn with_config(client: Arc<MagickmindClient>, config: MagickmindContextConfig) -> Self {
        Self {
            client,
            config,
            exclude_self_id: None,
        }
    }

    /// Identify the agent so its previous messages get the correct `assistant` role.
    ///
    /// Pass the agent's `agent_id` so chat history messages sent by this agent
    /// become `assistant`-role messages, while messages from others become `user`-role.
    /// Without this, all chat history appears as `user` role and the LLM gets confused
    /// seeing its own previous responses attributed to a user.
    pub fn with_self_id(mut self, agent_id: impl Into<String>) -> Self {
        self.exclude_self_id = Some(agent_id.into());
        self
    }
}

#[async_trait]
impl ContextProvider for MagickmindContext {
    fn name(&self) -> &str {
        "MagickmindContext"
    }

    async fn fetch(&self, message: &crate::models::Message) -> Result<Vec<LlmMessage>> {
        let magickspace_id = &message.channel_id;
        if magickspace_id.is_empty() {
            debug!("MagickmindContext: no channel_id, skipping");
            return Ok(Vec::new());
        }

        self.client
            .prepare_context(
                magickspace_id,
                &message.sender_id,
                &message.content,
                &self.config,
                self.exclude_self_id.as_deref(),
            )
            .await
    }
}

fn convert_context_response(
    resp: PrepareContextResponse,
    self_id: Option<&str>,
) -> Vec<LlmMessage> {
    let mut messages = Vec::new();

    // Chat history: split into proper roles so the LLM recognizes its own responses.
    for item in &resp.chat_history {
        if let Some(id) = self_id
            && item.sent_by_user_id == id
        {
            // Agent's own previous response → assistant role
            messages.push(LlmMessage::assistant(&item.content));
            continue;
        }
        // Other participants → user role with sender attribution
        messages.push(LlmMessage::user(format!(
            "[{}]: {}",
            item.sent_by_user_id, item.content
        )));
    }

    // Knowledge and documents → system context
    let mut context_parts = Vec::new();

    if !resp.fetcher.is_empty() {
        context_parts.push(format!("Relevant knowledge:\n{}", resp.fetcher));
    }

    if !resp.corpus.is_empty() {
        let corpus_text: Vec<&str> = resp.corpus.iter().map(|c| c.content.as_str()).collect();
        context_parts.push(format!(
            "Reference documents:\n{}",
            corpus_text.join("\n---\n")
        ));
    }

    if !context_parts.is_empty() {
        messages.push(LlmMessage::system(format!(
            "Context:\n\n{}",
            context_parts.join("\n\n")
        )));
    }

    messages
}

// ── MagickmindPersistence ─────────────────────────────────────────────────────

pub struct MagickmindPersistence {
    magickmind: Arc<MagickmindClient>,
}

impl MagickmindPersistence {
    pub fn new(magickmind: Arc<MagickmindClient>) -> Self {
        Self { magickmind }
    }
}

#[async_trait]
impl PipelineStage for MagickmindPersistence {
    fn name(&self) -> &str {
        "MagickmindPersistence"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let magickspace_id = &ctx.message.channel_id;
        if magickspace_id.is_empty() {
            debug!("MagickmindPersistence: no magickspace_id in message, skipping save");
            return Ok(());
        }

        let content = ctx.response.as_deref().unwrap_or("").to_string();

        self.magickmind
            .save_message(
                magickspace_id,
                &ctx.agent_config.agent_id,
                &content,
                Some(&ctx.message.id),
            )
            .await
            .map_err(|e| MindroidError::Pipeline {
                stage: "MagickmindPersistence".into(),
                message: e.to_string(),
                source: None,
            })?;

        Ok(())
    }
}

// ── Constructors ─────────────────────────────────────────────────────────────

pub fn magickmind_pipeline(
    identity: Arc<dyn Auth>,
    base_url: &str,
    api_key: &str,
    compute_power: u8,
) -> crate::Result<Pipeline> {
    let magickmind = Arc::new(MagickmindClient::new(base_url, identity));

    let mut config = LlmClientConfig::new(format!("{base_url}/v1"));
    config.api_key = Some(api_key.to_string());
    config.auth_style = AuthStyle::Bearer;
    config.custom_headers = HashMap::from([("X-Compute-Power".into(), compute_power.to_string())]);
    let client = LlmClient::new(config)?;

    Ok(Pipeline::new()
        .add_streaming_stage(GenericLlmProcessor::new(client))
        .add_stage(PostProcessor)
        .add_stage(MagickmindPersistence::new(magickmind)))
}

/// Magick Mind pipeline with Ollama as the LLM backend.
///
/// Uses Magickmind for context/memory and Ollama (OpenAI-compatible endpoint) for inference.
pub fn magickmind_ollama_pipeline(
    identity: Arc<dyn Auth>,
    magickmind_url: &str,
    ollama_url: &str,
    model: &str,
) -> crate::Result<Pipeline> {
    let magickmind = Arc::new(MagickmindClient::new(magickmind_url, identity));

    let mut config = LlmClientConfig::new(format!("{ollama_url}/v1"));
    config.default_model = Some(model.to_string());
    config.auth_style = AuthStyle::None;
    let client = LlmClient::new(config)?;

    Ok(Pipeline::new()
        .add_streaming_stage(GenericLlmProcessor::new(client))
        .add_stage(PostProcessor)
        .add_stage(MagickmindPersistence::new(magickmind)))
}
