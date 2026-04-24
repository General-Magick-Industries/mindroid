use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::auth::Auth;
use crate::error::Result;
use crate::http::AuthenticatedHttpClient;
use crate::models::{LlmMessage, Message};
use crate::pipeline::context::ContextProvider;

// ── API types ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct QueryCorpusBody<'a> {
    query: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<&'a str>,
    only_need_context: bool,
}

#[derive(Deserialize)]
struct QueryCorpusResponse {
    result: String,
}

// ── CorpusClient ─────────────────────────────────────────────────────────────

/// HTTP client for the Magick Mind corpus (RAG) service.
///
/// Talks to `POST /v1/corpus/{corpus_id}/query` to retrieve relevant
/// document context for a given query.
pub struct CorpusClient {
    client: AuthenticatedHttpClient,
}

impl CorpusClient {
    pub fn new(base_url: impl Into<String>, identity: Arc<dyn Auth>) -> Self {
        Self {
            client: AuthenticatedHttpClient::new(base_url, identity),
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.client = self.client.with_api_key(api_key);
        self
    }

    /// Query a corpus for relevant context.
    ///
    /// Uses `only_need_context = true` so the endpoint returns raw retrieved
    /// documents without LLM synthesis — the caller's own pipeline handles that.
    pub async fn query(
        &self,
        corpus_id: &str,
        query: &str,
        mode: Option<&str>,
    ) -> Result<String> {
        let path = format!("/v1/corpus/{}/query", corpus_id);

        debug!("CorpusClient::query POST {}{path}", self.client.base_url());

        let body = QueryCorpusBody {
            query,
            mode,
            only_need_context: true,
        };

        let parsed: QueryCorpusResponse = self.client.post_json(&path, &body).await?;

        Ok(parsed.result)
    }
}

// ── CorpusContextProvider ────────────────────────────────────────────────────

/// A [`ContextProvider`] that queries a Magick Mind corpus for relevant
/// document context based on the incoming message.
///
/// ```ignore
/// use mindroid::{ContextPreparer, CorpusClient, CorpusContextProvider};
///
/// let corpus = Arc::new(CorpusClient::new(base_url, identity));
/// let preparer = ContextPreparer::new()
///     .add_provider(CorpusContextProvider::new(corpus, "my-corpus-id"));
///
/// let context = preparer.prepare(&message).await?;
/// ```
pub struct CorpusContextProvider {
    client: Arc<CorpusClient>,
    corpus_id: String,
}

impl CorpusContextProvider {
    pub fn new(client: Arc<CorpusClient>, corpus_id: impl Into<String>) -> Self {
        Self {
            client,
            corpus_id: corpus_id.into(),
        }
    }
}

#[async_trait]
impl ContextProvider for CorpusContextProvider {
    fn name(&self) -> &str {
        "CorpusContext"
    }

    async fn fetch(&self, message: &Message) -> Result<Vec<LlmMessage>> {
        if message.content.is_empty() {
            debug!("CorpusContextProvider: empty message content, skipping");
            return Ok(Vec::new());
        }

        let result = self
            .client
            .query(&self.corpus_id, &message.content, None)
            .await?;

        if result.is_empty() {
            return Ok(Vec::new());
        }

        Ok(vec![LlmMessage::system(format!(
            "Reference documents:\n{result}"
        ))])
    }
}
