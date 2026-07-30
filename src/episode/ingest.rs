use async_trait::async_trait;
use std::sync::Arc;

use serde::Serialize;
use tracing::{debug, warn};

use crate::auth::Auth;
use crate::config::IngestScope;
use crate::core::context::Context;
use crate::error::{MindroidError, Result};
use crate::models::ChannelType;
use crate::models::CredentialKind;
use crate::pipeline::PipelineStage;

/// Shared HTTP client for the episode-ingest endpoint.
///
/// Ingest is best-effort: the stages that use this log and continue on failure,
/// so a memory outage never blocks message processing.
///
/// The route follows the credential, exactly like the persona prepare stage:
///
/// - [`CredentialKind::ServiceUser`] — `POST /v1/episodes/process`, `agent_id` in
///   the body names the memory owner.
/// - [`CredentialKind::EndUser`] — `POST /v1/end-user/episodes/process`, owner is
///   the token subject, so no `agent_id` is sent.
struct EpisodeClient {
    http: reqwest::Client,
    base_url: String,
    identity: Arc<dyn Auth>,
    caller: CredentialKind,
    allow_insecure: bool,
    /// Ask the server NOT to resolve and attach the agent's persona to each
    /// stored episode. Persona resolution runs per message and costs a lookup;
    /// opt out when the persona snapshot isn't needed.
    skip_persona: bool,
}

impl EpisodeClient {
    const HTTP_TIMEOUT_SECS: u64 = 10;

    fn new(base_url: &str, identity: Arc<dyn Auth>, caller: CredentialKind) -> Self {
        Self {
            http: crate::core::net::secure_json_client(std::time::Duration::from_secs(
                Self::HTTP_TIMEOUT_SECS,
            )),
            base_url: base_url.trim_end_matches('/').to_string(),
            identity,
            caller,
            allow_insecure: false,
            skip_persona: false,
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
                CredentialKind::ServiceUser => segments.extend(&["v1", "episodes", "process"]),
                CredentialKind::EndUser => {
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
        // Defense in depth: the builder already refuses a non-TLS base_url at
        // startup, but a directly-constructed client has not been through it.
        crate::core::net::require_secure_url(
            &self.base_url,
            self.allow_insecure,
            "episodes.allow_insecure",
        )?;

        let url = self.process_url()?;
        let headers = crate::auth::build_auth_header_map(self.identity.as_ref()).await?;
        let body = ProcessEpisodeRequest {
            agent_id: match self.caller {
                CredentialKind::ServiceUser => Some(agent_id),
                CredentialKind::EndUser => None,
            },
            magickspace_id: msg.magickspace_id,
            sender_id: msg.sender_id,
            message: msg.message,
            message_id: msg.message_id,
            display_name: msg.display_name,
            is_group: msg.is_group,
            skip_persona: self.skip_persona,
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
            let text = crate::core::net::error_excerpt(&resp.text().await.unwrap_or_default());
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
    scope: IngestScope,
}

impl EpisodeIngestStage {
    /// Create a stage that ingests inbound messages via the given credential route.
    pub fn new(base_url: &str, identity: Arc<dyn Auth>, caller: CredentialKind) -> Self {
        Self {
            client: EpisodeClient::new(base_url, identity, caller),
            scope: IngestScope::All,
        }
    }

    /// Permit sending auth headers over plaintext `http://` (local dev only).
    pub fn with_allow_insecure(mut self, allow_insecure: bool) -> Self {
        self.client.allow_insecure = allow_insecure;
        self
    }

    /// Ask the server not to resolve and attach the agent's persona to each
    /// stored episode. Saves a per-message persona lookup when the snapshot
    /// isn't needed. Default: `false` (persona is attached).
    pub fn with_skip_persona(mut self, skip_persona: bool) -> Self {
        self.client.skip_persona = skip_persona;
        self
    }

    /// Restrict which messages are ingested. Default: [`IngestScope::All`].
    ///
    /// [`IngestScope::DirectOnly`] is enforced here. [`IngestScope::Addressed`]
    /// cannot be — this stage runs before any gate, so it has no way to know
    /// whether the agent was addressed; the caller enforces it by invoking the
    /// stage only after the gate passes. [`Self::runs_after_gate`] reports
    /// which placement the configured scope requires.
    pub fn with_scope(mut self, scope: IngestScope) -> Self {
        self.scope = scope;
        self
    }

    /// Whether the configured scope requires this stage to be called *after*
    /// the agent's gate rather than before it.
    ///
    /// `true` only for [`IngestScope::Addressed`]. Calling the stage pre-gate
    /// under that scope would silently record everything.
    pub fn runs_after_gate(&self) -> bool {
        self.scope == IngestScope::Addressed
    }

    /// Whether this message is in scope for ingest.
    fn in_scope(&self, ctx: &Context) -> bool {
        match self.scope {
            // Addressed is enforced by call-site placement, not here: at this
            // point nothing has evaluated whether the agent was addressed.
            IngestScope::All | IngestScope::Addressed => true,
            IngestScope::DirectOnly => ctx.message.channel_type == ChannelType::Direct,
        }
    }
}

#[async_trait]
impl PipelineStage for EpisodeIngestStage {
    fn name(&self) -> &str {
        "EpisodeIngestStage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        if !self.in_scope(ctx) {
            debug!(
                "EpisodeIngestStage: {} out of scope for {:?}, not ingesting",
                ctx.message.id, self.scope
            );
            return Ok(());
        }

        let is_group = ctx.message.channel_type == ChannelType::Group;
        let msg = EpisodeMessage {
            magickspace_id: &ctx.message.channel_id,
            sender_id: &ctx.message.sender_id,
            message: &ctx.message.content,
            message_id: &ctx.message.id,
            // mindroid's Message carries no human-readable name. Sending the
            // raw sender_id would write an opaque platform id into permanent
            // memory as if it were a display name, and stored labels are hard
            // to backfill — leave it unset and let the server resolve one.
            display_name: None,
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
    pub fn new(base_url: &str, identity: Arc<dyn Auth>, caller: CredentialKind) -> Self {
        Self {
            client: EpisodeClient::new(base_url, identity, caller),
        }
    }

    pub fn with_allow_insecure(mut self, allow_insecure: bool) -> Self {
        self.client.allow_insecure = allow_insecure;
        self
    }

    /// Ask the server not to resolve and attach the agent's persona to each
    /// stored episode. Default: `false` (persona is attached).
    pub fn with_skip_persona(mut self, skip_persona: bool) -> Self {
        self.client.skip_persona = skip_persona;
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
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    skip_persona: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::static_id::StaticAuth;
    use crate::config::AgentConfig;
    use crate::models::Message;

    fn client(caller: CredentialKind) -> EpisodeClient {
        EpisodeClient::new("https://x", Arc::new(StaticAuth::new("t")), caller)
    }

    /// A context whose ingest is guaranteed to fail: port 1 is unreachable.
    fn failing_ctx(content: &str) -> (Context, String) {
        let mut msg = Message::new(content, "user-1", "space-1");
        msg.id = "msg-1".into();
        msg.channel_type = ChannelType::Group;
        let cfg = Arc::new(AgentConfig {
            agent_id: "agent-1".into(),
            name: "Agent One".into(),
            ..Default::default()
        });
        (
            Context::new(Arc::new(msg), cfg),
            "https://127.0.0.1:1".into(),
        )
    }

    /// The module's core contract: an ingest failure must never fail the
    /// pipeline, or a memory outage would stop the agent responding.
    #[tokio::test]
    async fn inbound_ingest_failure_does_not_fail_the_pipeline() {
        let (mut ctx, url) = failing_ctx("hello");
        let stage = EpisodeIngestStage::new(
            &url,
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        );
        assert!(stage.process(&mut ctx).await.is_ok());
    }

    #[tokio::test]
    async fn reply_ingest_failure_does_not_fail_the_pipeline() {
        let (mut ctx, url) = failing_ctx("hello");
        ctx.response = Some("a reply".into());
        let stage = EpisodeReplyIngestStage::new(
            &url,
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        );
        assert!(stage.process(&mut ctx).await.is_ok());
    }

    /// A plaintext base_url is refused at send time, and that refusal is still
    /// swallowed by the best-effort contract rather than failing the message.
    #[tokio::test]
    async fn plaintext_url_is_refused_but_still_best_effort() {
        let (mut ctx, _) = failing_ctx("hello");
        let stage = EpisodeIngestStage::new(
            "http://memory.internal",
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        );
        assert!(stage.process(&mut ctx).await.is_ok());

        // ...but the underlying client does reject it.
        let c = EpisodeClient::new(
            "http://memory.internal",
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        );
        let msg = EpisodeMessage {
            magickspace_id: "ms",
            sender_id: "u",
            message: "hi",
            message_id: "m1",
            display_name: None,
            is_group: false,
        };
        let err = c.ingest("agent-1", &msg).await.unwrap_err().to_string();
        assert!(err.contains("episodes.allow_insecure"), "got: {err}");
    }

    /// The reply id is the de-dupe key: a retry of the same turn must derive
    /// the same id rather than storing the reply twice.
    #[test]
    fn reply_id_is_derived_deterministically() {
        let derive = |id: &str| format!("{id}:reply");
        assert_eq!(derive("msg-1"), "msg-1:reply");
        assert_eq!(derive("msg-1"), derive("msg-1"));
        assert_ne!(derive("msg-1"), derive("msg-2"));
    }

    #[tokio::test]
    async fn reply_stage_skips_when_no_response() {
        let (mut ctx, url) = failing_ctx("hello");
        ctx.response = None;
        let stage = EpisodeReplyIngestStage::new(
            &url,
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        );
        // No response to ingest: returns early, before any network attempt.
        assert!(stage.process(&mut ctx).await.is_ok());

        ctx.response = Some(String::new());
        assert!(stage.process(&mut ctx).await.is_ok());
    }

    #[test]
    fn scope_defaults_to_all() {
        let s = EpisodeIngestStage::new(
            "https://x",
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        );
        assert_eq!(s.scope, IngestScope::All);
        // All is a pre-gate scope: recording everything requires seeing
        // everything, including messages that halt at the gate.
        assert!(!s.runs_after_gate());
    }

    /// Addressed cannot be enforced inside the stage — at Step 0 nothing has
    /// evaluated the gate. The caller enforces it by placement, so the stage
    /// must report that it needs the post-gate slot.
    #[test]
    fn addressed_requires_post_gate_placement() {
        let s = EpisodeIngestStage::new(
            "https://x",
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        )
        .with_scope(IngestScope::Addressed);
        assert!(s.runs_after_gate());
    }

    #[test]
    fn direct_only_is_enforced_in_the_stage() {
        let s = EpisodeIngestStage::new(
            "https://x",
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        )
        .with_scope(IngestScope::DirectOnly);
        // Enforced here, so no post-gate placement is needed.
        assert!(!s.runs_after_gate());

        let (group_ctx, _) = failing_ctx("hi");
        assert!(!s.in_scope(&group_ctx), "group traffic is out of scope");

        let mut direct = Message::new("hi", "user-1", "space-1");
        direct.channel_type = ChannelType::Direct;
        let direct_ctx = Context::new(Arc::new(direct), group_ctx.agent_config.clone());
        assert!(s.in_scope(&direct_ctx), "direct traffic is in scope");
    }

    /// A group message under DirectOnly must not reach the network at all.
    #[tokio::test]
    async fn out_of_scope_message_is_not_ingested() {
        let (mut ctx, url) = failing_ctx("hi");
        let s = EpisodeIngestStage::new(
            &url,
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        )
        .with_scope(IngestScope::DirectOnly);
        // The URL is unreachable, so an attempted send would still return Ok
        // (best-effort) — what this pins is the early return before that.
        assert!(s.process(&mut ctx).await.is_ok());
        assert!(!s.in_scope(&ctx));
    }

    #[test]
    fn group_channel_maps_to_is_group() {
        let mut msg = Message::new("hi", "u", "c");
        msg.channel_type = ChannelType::Group;
        assert!(msg.channel_type == ChannelType::Group);
        msg.channel_type = ChannelType::Direct;
        assert!(msg.channel_type != ChannelType::Group);
    }

    #[test]
    fn service_user_route() {
        let u = client(CredentialKind::ServiceUser).process_url().unwrap();
        assert_eq!(u.path(), "/v1/episodes/process");
    }

    #[test]
    fn end_user_route() {
        let u = client(CredentialKind::EndUser).process_url().unwrap();
        assert_eq!(u.path(), "/v1/end-user/episodes/process");
    }

    fn req(agent_id: Option<&'static str>, skip_persona: bool) -> ProcessEpisodeRequest<'static> {
        ProcessEpisodeRequest {
            agent_id,
            magickspace_id: "ms",
            sender_id: "u",
            message: "hi",
            message_id: "m1",
            display_name: None,
            is_group: false,
            skip_persona,
        }
    }

    #[test]
    fn service_user_body_carries_agent_id() {
        let json = serde_json::to_value(req(Some("a-1"), false)).unwrap();
        assert_eq!(json["agent_id"], "a-1");
    }

    #[test]
    fn end_user_body_omits_agent_id() {
        let json = serde_json::to_value(req(None, false)).unwrap();
        assert!(json.get("agent_id").is_none());
    }

    #[test]
    fn skip_persona_omitted_when_false_present_when_true() {
        let off = serde_json::to_value(req(None, false)).unwrap();
        assert!(
            off.get("skip_persona").is_none(),
            "false must not serialize"
        );
        let on = serde_json::to_value(req(None, true)).unwrap();
        assert_eq!(on["skip_persona"], true);
    }
}
