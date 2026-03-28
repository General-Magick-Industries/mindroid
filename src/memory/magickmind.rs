use async_trait::async_trait;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{Auth, Memory, Message, MindroidError, Result};

pub struct MagickmindMemory {
    http: reqwest::Client,
    base_url: String,
    identity: Arc<dyn Auth>,
}

impl MagickmindMemory {
    pub fn new(base_url: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            identity,
        }
    }

    async fn auth_headers(&self) -> Result<reqwest::header::HeaderMap> {
        crate::auth::build_auth_header_map(self.identity.as_ref()).await
    }
}

#[derive(Serialize)]
struct SaveMessageRequest<'a> {
    channel_id: &'a str,
    sender_id: &'a str,
    content: &'a str,
    reply_to_id: Option<&'a str>,
}

#[derive(Deserialize)]
struct SaveMessageResponse {
    id: String,
}

#[derive(Deserialize)]
struct GetHistoryResponse {
    messages: Vec<Message>,
}

#[async_trait]
impl Memory for MagickmindMemory {
    async fn save_message(
        &self,
        channel_id: &str,
        sender_id: &str,
        content: &str,
        reply_to_id: Option<&str>,
    ) -> Result<Option<String>> {
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
                .extend(&["v1", "mindspaces", channel_id, "messages"]);
            u
        };
        let headers = self.auth_headers().await?;
        let body = SaveMessageRequest { channel_id, sender_id, content, reply_to_id };

        let resp = self
            .http
            .post(url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| MindroidError::Api { message: e.to_string(), status_code: None })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(MindroidError::Api {
                message: text,
                status_code: Some(status.as_u16()),
            });
        }

        let data: SaveMessageResponse = resp
            .json()
            .await
            .map_err(|e| MindroidError::Api { message: e.to_string(), status_code: None })?;

        Ok(Some(data.id))
    }

    async fn get_history(&self, channel_id: &str, limit: usize) -> Result<Vec<Message>> {
        let channel_id = channel_id.to_string();
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
                .extend(&["v1", "mindspaces", &channel_id, "messages"]);
            u
        };
        let headers = self.auth_headers().await?;

        let resp = self
            .http
            .get(url)
            .headers(headers)
            .query(&[("channel_id", channel_id.as_str()), ("limit", &limit.to_string())])
            .send()
            .await
            .map_err(|e| MindroidError::Api { message: e.to_string(), status_code: None })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(MindroidError::Api {
                message: text,
                status_code: Some(status.as_u16()),
            });
        }

        let data: GetHistoryResponse = resp
            .json()
            .await
            .map_err(|e| MindroidError::Api { message: e.to_string(), status_code: None })?;

        Ok(data.messages)
    }

    async fn clear_history(&self, channel_id: &str) -> Result<()> {
        let channel_id = channel_id.to_string();
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
                .extend(&["v1", "mindspaces", &channel_id, "messages"]);
            u
        };
        let headers = self.auth_headers().await?;

        let resp = self
            .http
            .delete(url)
            .headers(headers)
            .query(&[("channel_id", channel_id.as_str())])
            .send()
            .await
            .map_err(|e| MindroidError::Api { message: e.to_string(), status_code: None })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(MindroidError::Api {
                message: text,
                status_code: Some(status.as_u16()),
            });
        }

        Ok(())
    }
}
