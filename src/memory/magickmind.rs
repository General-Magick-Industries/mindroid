use async_trait::async_trait;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::http::AuthenticatedHttpClient;
use crate::{Auth, Memory, Message, MindroidError, Result};

pub struct MagickmindMemory {
    client: AuthenticatedHttpClient,
}

impl MagickmindMemory {
    pub fn new(base_url: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            client: AuthenticatedHttpClient::new(base_url, identity),
        }
    }

    fn build_url(&self, channel_id: &str) -> Result<reqwest::Url> {
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
            .extend(&["v1", "mindspaces", channel_id, "messages"]);
        Ok(u)
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
        let url = self.build_url(channel_id)?;
        let body = SaveMessageRequest {
            channel_id,
            sender_id,
            content,
            reply_to_id,
        };

        let req = self
            .client
            .request(reqwest::Method::POST, url)
            .await?
            .json(&body);

        let resp = self.client.send_and_check(req).await?;

        let data: SaveMessageResponse = resp.json().await.map_err(|e| MindroidError::Api {
            message: e.to_string(),
            status_code: None,
        })?;

        Ok(Some(data.id))
    }

    async fn get_history(&self, channel_id: &str, limit: usize) -> Result<Vec<Message>> {
        let channel_id = channel_id.to_string();
        let url = self.build_url(&channel_id)?;

        let req = self
            .client
            .request(reqwest::Method::GET, url)
            .await?
            .query(&[
                ("channel_id", channel_id.as_str()),
                ("limit", &limit.to_string()),
            ]);

        let resp = self.client.send_and_check(req).await?;

        let data: GetHistoryResponse = resp.json().await.map_err(|e| MindroidError::Api {
            message: e.to_string(),
            status_code: None,
        })?;

        Ok(data.messages)
    }

    async fn clear_history(&self, channel_id: &str) -> Result<()> {
        let channel_id = channel_id.to_string();
        let url = self.build_url(&channel_id)?;

        let req = self
            .client
            .request(reqwest::Method::DELETE, url)
            .await?
            .query(&[("channel_id", channel_id.as_str())]);

        self.client.send_and_check(req).await?;

        Ok(())
    }
}
