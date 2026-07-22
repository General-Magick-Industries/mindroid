use async_trait::async_trait;
use std::sync::Arc;

use serde::Serialize;
use tracing::{debug, warn};

use crate::auth::Auth;
use crate::core::context::Context;
use crate::error::{MindroidError, Result};
use crate::models::ChannelType;
use crate::persona::PersonaCaller;
use crate::pipeline::PipelineStage;

/// Shared HTTP client for the episode-ingest endpoint.
///
/// Ingest is best-effort: the stages that use this log and continue on failure,
/// so a memory outage never blocks message processing.
///
/// The route follows the credential, exactly like the persona prepare stage:
///
/// - [`PersonaCaller::ServiceUser`] — `POST /v1/episodes/process`, `agent_id` in
///   the body names the memory owner.
/// - [`PersonaCaller::EndUser`] — `POST /v1/end-user/episodes/process`, owner is
///   the token subject, so no `agent_id` is sent.
struct EpisodeClient {
    http: reqwest::Client,
    base_url: String,
    identity: Arc<dyn Auth>,
    caller: PersonaCaller,
    allow_insecure: bool,
}

impl EpisodeClient {
    const HTTP_TIMEOUT_SECS: u64 = 10;

    fn new(base_url: &str, identity: Arc<dyn Auth>, caller: PersonaCaller) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(Self::HTTP_TIMEOUT_SECS))
                .build()
                .expect("failed to build HTTP client"),
            base_url: base_url.trim_end_matches('/').to_string(),
            identity,
            caller,
            allow_insecure: false,
        }
    }

    fn process_url(&self) -> Result<reqwest::Url> {
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
                PersonaCaller::ServiceUser => segments.extend(&["v1", "episodes", "process"]),
                PersonaCaller::EndUser => {
                    segments.extend(&["v1", "end-user", "episodes", "process"])
                }
            };
        }
        Ok(u)
    }

    /// Send one message to the ingest endpoint.
    ///
    /// `agent_id` names the memory owner on the service-user route and is
    /// omitted on the end-user route (the token subject owns).
    async fn ingest(&self, agent_id: &str, msg: &EpisodeMessage<'_>) -> Result<()> {
        if self.base_url.starts_with("http://") && !self.allow_insecure {
            return Err(MindroidError::Api {
                message: format!(
                    "refusing to send auth headers over plaintext {}: use https://, \
                     or set episodes.allow_insecure = true for local development",
                    self.base_url
                ),
                status_code: None,
            });
        }

        let url = self.process_url()?;
        let headers = crate::auth::build_auth_header_map(self.identity.as_ref()).await?;
        let body = ProcessEpisodeRequest {
            agent_id: match self.caller {
                PersonaCaller::ServiceUser => Some(agent_id),
                PersonaCaller::EndUser => None,
            },
            magickspace_id: msg.magickspace_id,
            sender_id: msg.sender_id,
            message: msg.message,
            message_id: msg.message_id,
            display_name: msg.display_name,
            is_group: msg.is_group,
        };

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
                message: format!("episode ingest failed: {text}"),
                status_code: Some(status.as_u16()),
            });
        }
        Ok(())
    }
}

/// The fields of one message to ingest, mapped from a pipeline [`Context`].
struct EpisodeMessage<'a> {
    magickspace_id: &'a str,
    sender_id: &'a str,
    message: &'a str,
    message_id: &'a str,
    display_name: Option<&'a str>,
    is_group: bool,
}

/// Pipeline stage that ingests **inbound** messages into episodic memory.
///
/// Place this early — before any gate that may halt the pipeline — so every
/// received message is remembered regardless of whether the agent responds.
/// The agent's own outbound reply is dropped before the pipeline runs
/// ([`runtime`](crate::core::runtime)), so it is captured separately by
/// [`EpisodeReplyIngestStage`].
///
/// Ingest is best-effort: a failure is logged and the message proceeds.
pub struct EpisodeIngestStage {
    client: EpisodeClient,
}

impl EpisodeIngestStage {
    /// Create a stage that ingests inbound messages via the given credential route.
    pub fn new(base_url: &str, identity: Arc<dyn Auth>, caller: PersonaCaller) -> Self {
        Self {
            client: EpisodeClient::new(base_url, identity, caller),
        }
    }

    /// Permit sending auth headers over plaintext `http://` (local dev only).
    pub fn with_allow_insecure(mut self, allow_insecure: bool) -> Self {
        self.client.allow_insecure = allow_insecure;
        self
    }
}

#[async_trait]
impl PipelineStage for EpisodeIngestStage {
    fn name(&self) -> &str {
        "EpisodeIngestStage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let is_group = ctx.message.channel_type == ChannelType::Group;
        let msg = EpisodeMessage {
            magickspace_id: &ctx.message.channel_id,
            sender_id: &ctx.message.sender_id,
            message: &ctx.message.content,
            message_id: &ctx.message.id,
            // mindroid's Message has no display name; fall back to sender_id so
            // episodes always has a non-blank label.
            display_name: Some(&ctx.message.sender_id),
            is_group,
        };

        match self.client.ingest(&ctx.agent_config.agent_id, &msg).await {
            Ok(()) => debug!("EpisodeIngestStage: ingested inbound {}", ctx.message.id),
            Err(e) => warn!("EpisodeIngestStage: ingest failed (continuing): {e}"),
        }
        Ok(())
    }
}

/// Pipeline stage that ingests the agent's **outbound reply** into episodic
/// memory. Place it after response generation (near persistence).
///
/// The reply has no message id of its own, so one is derived deterministically
/// as `{inbound_id}:reply` — a retry of the same turn de-dupes instead of
/// storing the reply twice. Ingest is best-effort.
pub struct EpisodeReplyIngestStage {
    client: EpisodeClient,
}

impl EpisodeReplyIngestStage {
    pub fn new(base_url: &str, identity: Arc<dyn Auth>, caller: PersonaCaller) -> Self {
        Self {
            client: EpisodeClient::new(base_url, identity, caller),
        }
    }

    pub fn with_allow_insecure(mut self, allow_insecure: bool) -> Self {
        self.client.allow_insecure = allow_insecure;
        self
    }
}

#[async_trait]
impl PipelineStage for EpisodeReplyIngestStage {
    fn name(&self) -> &str {
        "EpisodeReplyIngestStage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let Some(reply) = ctx.response.as_deref() else {
            debug!("EpisodeReplyIngestStage: no response to ingest");
            return Ok(());
        };
        if reply.is_empty() {
            return Ok(());
        }

        let reply_id = format!("{}:reply", ctx.message.id);
        let is_group = ctx.message.channel_type == ChannelType::Group;
        let msg = EpisodeMessage {
            magickspace_id: &ctx.message.channel_id,
            // The agent is the sender of its own reply.
            sender_id: &ctx.agent_config.agent_id,
            message: reply,
            message_id: &reply_id,
            display_name: Some(&ctx.agent_config.name),
            is_group,
        };

        match self.client.ingest(&ctx.agent_config.agent_id, &msg).await {
            Ok(()) => debug!("EpisodeReplyIngestStage: ingested reply {reply_id}"),
            Err(e) => warn!("EpisodeReplyIngestStage: ingest failed (continuing): {e}"),
        }
        Ok(())
    }
}

/// Request body for both `/process` routes. `agent_id` is omitted on the
/// end-user route, where the owner is the token subject.
#[derive(Serialize)]
struct ProcessEpisodeRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<&'a str>,
    magickspace_id: &'a str,
    sender_id: &'a str,
    message: &'a str,
    message_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    display_name: Option<&'a str>,
    is_group: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::static_id::StaticAuth;

    fn client(caller: PersonaCaller) -> EpisodeClient {
        EpisodeClient::new("https://x", Arc::new(StaticAuth::new("t")), caller)
    }

    #[test]
    fn service_user_route() {
        let u = client(PersonaCaller::ServiceUser).process_url().unwrap();
        assert_eq!(u.path(), "/v1/episodes/process");
    }

    #[test]
    fn end_user_route() {
        let u = client(PersonaCaller::EndUser).process_url().unwrap();
        assert_eq!(u.path(), "/v1/end-user/episodes/process");
    }

    #[test]
    fn service_user_body_carries_agent_id() {
        let body = ProcessEpisodeRequest {
            agent_id: Some("a-1"),
            magickspace_id: "ms",
            sender_id: "u",
            message: "hi",
            message_id: "m1",
            display_name: None,
            is_group: false,
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["agent_id"], "a-1");
    }

    #[test]
    fn end_user_body_omits_agent_id() {
        let body = ProcessEpisodeRequest {
            agent_id: None,
            magickspace_id: "ms",
            sender_id: "u",
            message: "hi",
            message_id: "m1",
            display_name: None,
            is_group: false,
        };
        let json = serde_json::to_value(&body).unwrap();
        assert!(json.get("agent_id").is_none());
    }
}
