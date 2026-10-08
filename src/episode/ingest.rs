use async_trait::async_trait;
use std::{collections::HashMap, sync::Arc};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::auth::Auth;
use crate::config::IngestScope;
use crate::core::context::Context;
use crate::core::prompt_text::sanitize_line;
use crate::error::{MindroidError, Result};
use crate::models::ChannelType;
use crate::models::CredentialKind;
use crate::models::Message;
use crate::persona::{RuntimeAffectSnapshot, RuntimeStateEnvelope};
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
    credential_kind: CredentialKind,
    allow_insecure: bool,
    /// Ask the server NOT to resolve and attach the agent's persona to each
    /// stored episode. Persona resolution runs per message and costs a lookup;
    /// opt out when the persona snapshot isn't needed.
    skip_persona: bool,
}

impl EpisodeClient {
    const HTTP_TIMEOUT_SECS: u64 = 10;

    fn new(base_url: &str, identity: Arc<dyn Auth>, credential_kind: CredentialKind) -> Self {
        Self {
            http: crate::core::net::secure_json_client(std::time::Duration::from_secs(
                Self::HTTP_TIMEOUT_SECS,
            )),
            base_url: base_url.trim_end_matches('/').to_string(),
            identity,
            credential_kind,
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
            match self.credential_kind {
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
    async fn ingest(
        &self,
        agent_id: &str,
        msg: &EpisodeMessage<'_>,
    ) -> Result<Option<RuntimeStateEnvelope>> {
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
            agent_id: match self.credential_kind {
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
        crate::core::net::note_auth_status(self.identity.as_ref(), status);
        if !status.is_success() {
            let text = crate::core::net::error_excerpt(&resp.text().await.unwrap_or_default());
            return Err(MindroidError::Api {
                message: format!("episode ingest failed: {text}"),
                status_code: Some(status.as_u16()),
            });
        }
        // The write has succeeded by now; what follows only adds affect state.
        // The response is a few hundred bytes, so anything large is not ours.
        if resp
            .content_length()
            .is_some_and(|len| len > MAX_RESPONSE_BYTES)
        {
            warn!("EpisodeClient: ingest response too large to carry runtime state; ignoring it");
            return Ok(None);
        }
        let body = match read_capped(resp, MAX_RESPONSE_BYTES).await {
            Ok(Some(body)) => body,
            Ok(None) => {
                warn!(
                    "EpisodeClient: ingest response too large to carry runtime state; ignoring it"
                );
                return Ok(None);
            }
            Err(e) => {
                warn!("EpisodeClient: ingest succeeded but its response could not be read: {e}");
                return Ok(None);
            }
        };
        // An older Bifrost answered with an empty 2xx.
        if body.iter().all(u8::is_ascii_whitespace) {
            return Ok(None);
        }
        match serde_json::from_slice::<ProcessEpisodeResponse>(&body) {
            Ok(response) => Ok(response.runtime_state),
            Err(e) => {
                let text = crate::core::net::error_excerpt(&e.to_string());
                warn!("EpisodeClient: ingest succeeded but its response was not decodable: {text}");
                Ok(None)
            }
        }
    }
}

const MAX_RESPONSE_BYTES: u64 = 64 * 1024;

/// The body, or `None` once it grows past `cap`; a chunked response has no
/// length to check up front.
async fn read_capped(
    mut resp: reqwest::Response,
    cap: u64,
) -> std::result::Result<Option<Vec<u8>>, reqwest::Error> {
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await? {
        if (body.len() + chunk.len()) as u64 > cap {
            return Ok(None);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Some(body))
}

const MAX_RUNTIME_STATES: usize = 256;

/// Whose affect a held state is. Persona can hold a row per agent and one per
/// agent-and-user, and an envelope does not say which it came from, so each
/// sender's state is kept apart. `sender_id: None` is the agent's own row,
/// learned from ingesting the agent's reply, where the agent is the sender.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct RuntimeStateKey {
    agent_id: String,
    sender_id: Option<String>,
}

impl RuntimeStateKey {
    fn sender(ctx: &Context) -> Self {
        Self {
            agent_id: ctx.agent_config.agent_id.clone(),
            sender_id: Some(ctx.message.sender_id.clone()),
        }
    }

    fn agent(ctx: &Context) -> Self {
        Self {
            agent_id: ctx.agent_config.agent_id.clone(),
            sender_id: None,
        }
    }
}

/// Affect states as last reported by the server, one per persona row.
#[derive(Default)]
struct RuntimeStateCache {
    states: RwLock<HashMap<RuntimeStateKey, RuntimeStateEnvelope>>,
}

enum AcceptOutcome {
    Accepted,
    /// Stored, but already past its TTL on arrival: this host's clock is more
    /// than one TTL ahead of the persona service's, so no affect will ever
    /// render until that is fixed.
    AcceptedExpired,
    Stale,
    Invalid(&'static str),
}

impl RuntimeStateCache {
    async fn accept(
        &self,
        key: RuntimeStateKey,
        state: RuntimeStateEnvelope,
        now: DateTime<Utc>,
    ) -> AcceptOutcome {
        if let Err(reason) = state.validate_at(now) {
            return AcceptOutcome::Invalid(reason);
        }

        let mut states = self.states.write().await;
        if let Some(held) = states.get(&key)
            && !held.is_expired_at(now)
            && (state.state_version < held.state_version
                || (state.state_version == held.state_version
                    && state.computed_at < held.computed_at))
        {
            return AcceptOutcome::Stale;
        }
        if states.len() >= MAX_RUNTIME_STATES && !states.contains_key(&key) {
            states.retain(|_, held| !held.is_expired_at(now));
            if states.len() >= MAX_RUNTIME_STATES
                && let Some(oldest) = states
                    .iter()
                    .min_by_key(|(_, held)| held.computed_at)
                    .map(|(key, _)| key.clone())
            {
                states.remove(&oldest);
            }
        }
        let arrived_expired = state.is_expired_at(now);
        states.insert(key, state);
        if arrived_expired {
            return AcceptOutcome::AcceptedExpired;
        }
        AcceptOutcome::Accepted
    }

    async fn is_held(&self, key: &RuntimeStateKey) -> bool {
        self.states.read().await.contains_key(key)
    }

    async fn current(
        &self,
        sender: &RuntimeStateKey,
        agent: &RuntimeStateKey,
        at: DateTime<Utc>,
    ) -> Option<RuntimeAffectSnapshot> {
        let states = self.states.read().await;
        [sender, agent]
            .into_iter()
            .find_map(|key| states.get(key)?.decayed_at(at))
    }
}

fn log_accept(stage: &str, outcome: AcceptOutcome) {
    match outcome {
        AcceptOutcome::Accepted => debug!("{stage}: accepted runtime affect state"),
        AcceptOutcome::AcceptedExpired => warn!(
            "{stage}: runtime affect state was already expired on arrival; check clock \
             skew against the persona service"
        ),
        AcceptOutcome::Stale => debug!("{stage}: ignored stale runtime affect state"),
        AcceptOutcome::Invalid(reason) => {
            warn!("{stage}: ignored invalid runtime affect state: {reason}")
        }
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
    runtime_states: Arc<RuntimeStateCache>,
}

impl EpisodeIngestStage {
    /// Create a stage that ingests inbound messages via the given credential route.
    pub fn new(base_url: &str, identity: Arc<dyn Auth>, credential_kind: CredentialKind) -> Self {
        Self {
            client: EpisodeClient::new(base_url, identity, credential_kind),
            scope: IngestScope::All,
            runtime_states: Arc::default(),
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
    ///
    /// Also turns off live affect: with no persona resolved, the server returns
    /// no `runtime_state`.
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

    /// Whether this message should be ingested at all.
    ///
    /// Being control traffic is a property of the message, so it is checked
    /// before scope — a manifest is not an episode on any scope setting.
    fn should_ingest(&self, ctx: &Context) -> bool {
        // Control traffic is protocol, not something anyone said. A manifest or
        // a tool result is not an episode, and topic detection would otherwise
        // mint micro-episodes from them.
        if ctx.message.message_type.is_control() {
            return false;
        }
        match self.scope {
            // Addressed is enforced by call-site placement, not here: at this
            // point nothing has evaluated whether the agent was addressed.
            IngestScope::All | IngestScope::Addressed => true,
            IngestScope::DirectOnly => ctx.message.channel_type == ChannelType::Direct,
        }
    }

    /// Put the agent's latest valid, locally decayed affect toward this
    /// message's sender into the run-scoped pipeline context. With no state of
    /// the sender's own, the agent's is used, which the reply stage learns once
    /// it shares this stage's state ([`EpisodeReplyIngestStage::with_runtime_state_of`]).
    ///
    /// `Context::reset_output` clears run-scoped extensions. Call this again
    /// after such a reset when ingest and persona execution use separate
    /// pipeline runs.
    pub async fn apply_runtime_state(&self, ctx: &mut Context) {
        let _ = ctx.take_ext::<RuntimeAffectSnapshot>();
        let sender = RuntimeStateKey::sender(ctx);
        let agent = RuntimeStateKey::agent(ctx);
        match self
            .runtime_states
            .current(&sender, &agent, Utc::now())
            .await
        {
            Some(affect) => ctx.set_ext(affect),
            None if self.runtime_states.is_held(&sender).await
                || self.runtime_states.is_held(&agent).await =>
            {
                debug!("EpisodeIngestStage: held runtime affect state has expired; none applied")
            }
            None => debug!("EpisodeIngestStage: no runtime affect state to apply"),
        }
    }
}

#[async_trait]
impl PipelineStage for EpisodeIngestStage {
    fn name(&self) -> &str {
        "EpisodeIngestStage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        if !self.should_ingest(ctx) {
            debug!(
                "EpisodeIngestStage: not ingesting {} ({:?}, scope {:?})",
                ctx.message.id, ctx.message.message_type, self.scope
            );
            self.apply_runtime_state(ctx).await;
            return Ok(());
        }

        let is_group = ctx.message.channel_type == ChannelType::Group;
        let display_name = trusted_display_name(&ctx.message);
        let msg = EpisodeMessage {
            magickspace_id: ctx.message.conversation_id(),
            sender_id: &ctx.message.sender_id,
            message: &ctx.message.content,
            message_id: &ctx.message.id,
            // Never the raw sender_id as a fallback: an opaque platform id
            // written as if it were a display name is hard to backfill out of
            // permanent memory, and the server resolves a name of its own from
            // the sender's record when this is unset.
            display_name: display_name.as_deref(),
            is_group,
        };

        match self.client.ingest(&ctx.agent_config.agent_id, &msg).await {
            Ok(runtime_state) => {
                debug!("EpisodeIngestStage: ingested inbound {}", ctx.message.id);
                if let Some(state) = runtime_state {
                    let key = RuntimeStateKey::sender(ctx);
                    let outcome = self.runtime_states.accept(key, state, Utc::now()).await;
                    log_accept(self.name(), outcome);
                }
            }
            Err(e) => warn!("EpisodeIngestStage: ingest failed (continuing): {e}"),
        }
        self.apply_runtime_state(ctx).await;
        Ok(())
    }
}

/// The sender's display name as the backend stamped it on the envelope.
///
/// Publisher-supplied, so it is read only from a sender the transport can name
/// — the rule display names already follow into the prompt — and flattened to
/// one line: episodic stores it as `<id>:<name>: message`, where a newline in a
/// name would forge a second speaker in permanent memory.
fn trusted_display_name(message: &Message) -> Option<String> {
    message.trusted_sender_id()?;
    let name = message
        .metadata
        .get("sent_by_user_name")
        .and_then(serde_json::Value::as_str)
        .map(sanitize_line)?;
    (!name.is_empty()).then_some(name)
}

/// Pipeline stage that ingests the agent's **outbound reply** into episodic
/// memory. Place it after response generation (near persistence).
///
/// The reply has no message id of its own, so one is derived deterministically
/// as `{inbound_id}:reply` — a retry of the same turn de-dupes instead of
/// storing the reply twice. Ingest is best-effort.
pub struct EpisodeReplyIngestStage {
    client: EpisodeClient,
    runtime_states: Option<Arc<RuntimeStateCache>>,
}

impl EpisodeReplyIngestStage {
    pub fn new(base_url: &str, identity: Arc<dyn Auth>, credential_kind: CredentialKind) -> Self {
        Self {
            client: EpisodeClient::new(base_url, identity, credential_kind),
            runtime_states: None,
        }
    }

    /// Keep the affect state returned by reply ingest in `inbound`'s state.
    ///
    /// The reply is ingested with the agent as its sender, so its state is the
    /// agent's own rather than any user's, and `inbound` falls back to it for a
    /// sender it holds no state for. Without this the reply's state is dropped.
    pub fn with_runtime_state_of(mut self, inbound: &EpisodeIngestStage) -> Self {
        self.runtime_states = Some(Arc::clone(&inbound.runtime_states));
        self
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
            magickspace_id: ctx.message.conversation_id(),
            // The agent is the sender of its own reply.
            sender_id: &ctx.agent_config.agent_id,
            message: reply,
            message_id: &reply_id,
            display_name: Some(&ctx.agent_config.name),
            is_group,
        };

        match self.client.ingest(&ctx.agent_config.agent_id, &msg).await {
            Ok(runtime_state) => {
                debug!("EpisodeReplyIngestStage: ingested reply {reply_id}");
                if let (Some(states), Some(state)) = (&self.runtime_states, runtime_state) {
                    let key = RuntimeStateKey::agent(ctx);
                    log_accept(self.name(), states.accept(key, state, Utc::now()).await);
                }
            }
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

#[derive(Deserialize)]
struct ProcessEpisodeResponse {
    runtime_state: Option<RuntimeStateEnvelope>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::static_id::StaticAuth;
    use crate::config::AgentConfig;
    use crate::models::{Message, MessageType};
    use crate::persona::RuntimeAffectState;

    fn client(credential_kind: CredentialKind) -> EpisodeClient {
        EpisodeClient::new("https://x", Arc::new(StaticAuth::new("t")), credential_kind)
    }

    fn http_200(body: &[u8]) -> Vec<u8> {
        let mut response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();
        response.extend_from_slice(body);
        response
    }

    fn http_200_chunked(body: &[u8]) -> Vec<u8> {
        let mut response =
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n"
                .to_vec();
        for chunk in body.chunks(8 * 1024) {
            response.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
            response.extend_from_slice(chunk);
            response.extend_from_slice(b"\r\n");
        }
        response.extend_from_slice(b"0\r\n\r\n");
        response
    }

    async fn serve_once(response: Vec<u8>) -> (String, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut req = Vec::new();
            let mut buf = [0u8; 4096];
            while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = sock.read(&mut buf).await.unwrap();
                if n == 0 {
                    return;
                }
                req.extend_from_slice(&buf[..n]);
            }
            sock.write_all(&response).await.ok();
            sock.shutdown().await.ok();
        });
        (format!("http://{addr}"), server)
    }

    async fn ingest_against(response: Vec<u8>) -> Result<Option<RuntimeStateEnvelope>> {
        let (url, server) = serve_once(response).await;
        let mut c = EpisodeClient::new(
            &url,
            Arc::new(StaticAuth::new("t")),
            CredentialKind::ServiceUser,
        );
        c.allow_insecure = true;
        let msg = EpisodeMessage {
            magickspace_id: "ms",
            sender_id: "u",
            message: "hi",
            message_id: "m1",
            display_name: None,
            is_group: false,
        };
        let out = c.ingest("agent-1", &msg).await;
        server.await.unwrap();
        out
    }

    fn envelope_json(computed_at: DateTime<Utc>) -> serde_json::Value {
        let at = computed_at.to_rfc3339();
        serde_json::json!({
            "runtime_state": {
                "affect": {
                    "pleasure": 0.8, "arousal": 0.4, "dominance": -0.2,
                    "baseline_pleasure": 0.0, "baseline_arousal": 0.0, "baseline_dominance": 0.0,
                    "pleasure_half_life_seconds": 600,
                    "arousal_half_life_seconds": 1200,
                    "dominance_half_life_seconds": 1800,
                    "updated_at": at
                },
                "state_version": 3,
                "computed_at": at,
                "ttl_seconds": 60
            }
        })
    }

    fn envelope_body() -> Vec<u8> {
        serde_json::to_vec(&envelope_json(Utc::now())).unwrap()
    }

    fn key(agent_id: &str, sender_id: Option<&str>) -> RuntimeStateKey {
        RuntimeStateKey {
            agent_id: agent_id.into(),
            sender_id: sender_id.map(Into::into),
        }
    }

    #[tokio::test]
    async fn a_2xx_envelope_is_returned() {
        let state = ingest_against(http_200(&envelope_body())).await.unwrap();
        assert_eq!(state.unwrap().state_version, 3);
    }

    #[tokio::test]
    async fn an_undecodable_2xx_is_a_successful_ingest_with_no_affect() {
        let mut body = envelope_json(Utc::now());
        body["runtime_state"]["state_version"] = serde_json::json!("x".repeat(70_000));
        let body = serde_json::to_vec(&body).unwrap();
        assert!(ingest_against(http_200(&body)).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn an_oversized_2xx_is_a_successful_ingest_with_no_affect() {
        let mut body = envelope_body();
        body.resize(MAX_RESPONSE_BYTES as usize + 1, b' ');
        assert!(ingest_against(http_200(&body)).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn a_chunked_2xx_is_capped_while_it_streams() {
        let within = ingest_against(http_200_chunked(&envelope_body())).await;
        assert!(within.unwrap().is_some());

        let mut body = envelope_body();
        body.resize(MAX_RESPONSE_BYTES as usize + 1, b' ');
        let over = ingest_against(http_200_chunked(&body)).await;
        assert!(over.unwrap().is_none());
    }

    #[tokio::test]
    async fn an_envelope_expired_on_arrival_is_stored_but_reported() {
        let cache = RuntimeStateCache::default();
        let now = Utc::now();
        let k = key("agent-1", Some("user-1"));
        // Host clock more than one ttl ahead of the server's.
        let state = runtime_state(1, now - chrono::Duration::seconds(120));
        assert!(matches!(
            cache.accept(k.clone(), state, now).await,
            AcceptOutcome::AcceptedExpired
        ));
        assert!(cache.is_held(&k).await, "stored despite being expired");
        assert!(
            cache.current(&k, &k, now).await.is_none(),
            "nothing renders"
        );
    }

    #[tokio::test]
    async fn out_of_range_affect_is_rejected_as_invalid() {
        let cache = RuntimeStateCache::default();
        let now = Utc::now();
        let k = key("agent-1", Some("user-1"));

        let mut over = runtime_state(1, now);
        over.affect.pleasure = 1.5;
        let mut under = runtime_state(1, now);
        under.affect.baseline_dominance = -1.5;
        let mut zero_half_life = runtime_state(1, now);
        zero_half_life.affect.arousal_half_life_seconds = 0;
        let from_the_future = runtime_state(1, now + chrono::Duration::seconds(60));

        for state in [over, under, zero_half_life, from_the_future] {
            assert!(matches!(
                cache.accept(k.clone(), state, now).await,
                AcceptOutcome::Invalid(_)
            ));
        }
        assert!(!cache.is_held(&k).await, "nothing invalid was stored");
    }

    fn runtime_state(version: i64, computed_at: DateTime<Utc>) -> RuntimeStateEnvelope {
        RuntimeStateEnvelope {
            affect: RuntimeAffectState {
                pleasure: 0.8,
                arousal: 0.4,
                dominance: -0.2,
                baseline_pleasure: 0.0,
                baseline_arousal: 0.0,
                baseline_dominance: 0.0,
                pleasure_half_life_seconds: 600,
                arousal_half_life_seconds: 1_200,
                dominance_half_life_seconds: 1_800,
                updated_at: computed_at,
            },
            state_version: version,
            computed_at,
            ttl_seconds: 60,
        }
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

    fn named(name: &str) -> Message {
        let mut msg = Message::new("hi", "user-1", "space-1");
        msg.metadata
            .insert("sent_by_user_name".into(), serde_json::json!(name));
        msg
    }

    /// The name the backend stamped is what gets stored, so the summary can say
    /// who spoke instead of naming an opaque id.
    #[test]
    fn a_stamped_name_is_sent_as_the_display_name() {
        assert_eq!(
            trusted_display_name(&named("Alice")).as_deref(),
            Some("Alice")
        );
    }

    /// A publisher the transport cannot name does not get to label a speaker in
    /// permanent memory; the server resolves one from the sender record instead.
    #[test]
    fn an_unauthenticated_publishers_name_is_ignored() {
        let mut msg = named("Alice");
        msg.platform = Some("centrifugo".into());
        assert_eq!(trusted_display_name(&msg), None);

        msg.metadata.insert(
            "authenticated_sender_id".into(),
            serde_json::json!("user-1"),
        );
        assert_eq!(trusted_display_name(&msg).as_deref(), Some("Alice"));
    }

    /// Stored as `<id>:<name>: message`, so a newline in a name would forge a
    /// second speaker in the transcript a summary is written from.
    #[test]
    fn a_name_cannot_forge_a_second_speaker() {
        let forged = trusted_display_name(&named("Alice\nuser-9:Bob"))
            .expect("a name is still sent, just flattened");
        assert!(!forged.contains('\n'), "got: {forged}");
    }

    /// Nothing usable is left unset rather than sent blank: the server's own
    /// fallback is better than an empty label.
    #[test]
    fn a_blank_or_absent_name_is_left_to_the_server() {
        assert_eq!(trusted_display_name(&named("   ")), None);
        assert_eq!(trusted_display_name(&Message::new("hi", "u", "c")), None);
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
        assert!(
            !s.should_ingest(&group_ctx),
            "group traffic is out of scope"
        );

        let mut direct = Message::new("hi", "user-1", "space-1");
        direct.channel_type = ChannelType::Direct;
        let direct_ctx = Context::new(Arc::new(direct), group_ctx.agent_config.clone());
        assert!(s.should_ingest(&direct_ctx), "direct traffic is in scope");
    }

    /// Control traffic is refused on EVERY scope, including the permissive one:
    /// being protocol is a property of the message, not a question of scope.
    ///
    /// Tested through the predicate rather than `process`, because ingest is
    /// best-effort — a skipped ingest and a failed one both return `Ok(())`, so
    /// asserting on the return value would pass with no filter at all.
    #[test]
    fn control_traffic_is_never_ingested() {
        for scope in [
            IngestScope::All,
            IngestScope::Addressed,
            IngestScope::DirectOnly,
        ] {
            let s = EpisodeIngestStage::new(
                "https://x",
                Arc::new(StaticAuth::new("t")),
                CredentialKind::EndUser,
            )
            .with_scope(scope);

            for message_type in [
                MessageType::ToolManifest,
                MessageType::ToolResult,
                MessageType::ToolCall,
            ] {
                let mut msg = Message::new("hi", "user-1", "space-1");
                msg.channel_type = ChannelType::Direct;
                msg.message_type = message_type.clone();
                let ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));

                assert!(
                    !s.should_ingest(&ctx),
                    "{message_type:?} must not be ingested under {scope:?}"
                );
            }
        }
    }

    /// The counterpart: an ordinary turn on the same channel still ingests, so
    /// the filter above cannot be passing for an unrelated reason.
    #[test]
    fn an_ordinary_turn_is_still_ingested() {
        let s = EpisodeIngestStage::new(
            "https://x",
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        )
        .with_scope(IngestScope::DirectOnly);

        let mut msg = Message::new("hi", "user-1", "space-1");
        msg.channel_type = ChannelType::Direct;
        let ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));

        assert!(s.should_ingest(&ctx));
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
        assert!(!s.should_ingest(&ctx));
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

    #[test]
    fn process_response_decodes_runtime_state_envelope() {
        let response: ProcessEpisodeResponse = serde_json::from_str(
            r#"{
                "message_processed": true,
                "runtime_state": {
                    "affect": {
                        "pleasure": 0.8,
                        "arousal": 0.4,
                        "dominance": -0.2,
                        "baseline_pleasure": 0.0,
                        "baseline_arousal": 0.0,
                        "baseline_dominance": 0.0,
                        "pleasure_half_life_seconds": 600,
                        "arousal_half_life_seconds": 1200,
                        "dominance_half_life_seconds": 1800,
                        "updated_at": "2026-08-13T10:00:00Z"
                    },
                    "state_version": 9,
                    "computed_at": "2026-08-13T10:00:01Z",
                    "ttl_seconds": 60
                }
            }"#,
        )
        .unwrap();

        let state = response.runtime_state.unwrap();
        assert_eq!(state.state_version, 9);
        assert_eq!(state.affect.arousal_half_life_seconds, 1_200);
    }

    #[tokio::test]
    async fn runtime_cache_rejects_older_versions() {
        let cache = RuntimeStateCache::default();
        let now = Utc::now();
        let k = key("agent-1", Some("user-1"));
        let secs = chrono::Duration::seconds;

        assert!(matches!(
            cache.accept(k.clone(), runtime_state(2, now), now).await,
            AcceptOutcome::Accepted
        ));
        assert!(matches!(
            cache
                .accept(k.clone(), runtime_state(1, now + secs(1)), now)
                .await,
            AcceptOutcome::Stale
        ));
        assert!(matches!(
            cache
                .accept(k.clone(), runtime_state(2, now - secs(1)), now)
                .await,
            AcceptOutcome::Stale
        ));

        assert_eq!(cache.current(&k, &k, now).await.unwrap().state_version, 2);
    }

    /// Per-user persona rows carry independent version counters, so one
    /// sender's higher version must not shut another sender's state out.
    #[tokio::test]
    async fn runtime_cache_keeps_senders_apart() {
        let cache = RuntimeStateCache::default();
        let now = Utc::now();
        let agent = key("agent-1", None);
        let alice = key("agent-1", Some("alice"));
        let bob = key("agent-1", Some("bob"));

        let mut bobs = runtime_state(2, now);
        bobs.affect.pleasure = -0.8;
        cache
            .accept(alice.clone(), runtime_state(7, now), now)
            .await;
        assert!(matches!(
            cache.accept(bob.clone(), bobs, now).await,
            AcceptOutcome::Accepted
        ));

        let seen_by_alice = cache.current(&alice, &agent, now).await.unwrap();
        let seen_by_bob = cache.current(&bob, &agent, now).await.unwrap();
        assert_eq!(seen_by_alice.state_version, 7);
        assert_eq!(seen_by_bob.state_version, 2);
        assert!(seen_by_bob.pleasure < 0.0);
    }

    #[tokio::test]
    async fn a_sender_without_state_falls_back_to_the_agents() {
        let cache = RuntimeStateCache::default();
        let now = Utc::now();
        let agent = key("agent-1", None);
        let alice = key("agent-1", Some("alice"));
        let carol = key("agent-1", Some("carol"));

        assert!(cache.current(&carol, &agent, now).await.is_none());
        cache
            .accept(agent.clone(), runtime_state(5, now), now)
            .await;
        cache
            .accept(alice.clone(), runtime_state(9, now), now)
            .await;

        let carols = cache.current(&carol, &agent, now).await.unwrap();
        assert_eq!(carols.state_version, 5, "no state of carol's own");
        let alices = cache.current(&alice, &agent, now).await.unwrap();
        assert_eq!(alices.state_version, 9, "a sender's own state wins");
    }

    #[tokio::test]
    async fn runtime_cache_keeps_agents_apart() {
        let cache = RuntimeStateCache::default();
        let now = Utc::now();
        let theirs = key("agent-1", Some("user-1"));
        cache.accept(theirs, runtime_state(7, now), now).await;
        cache
            .accept(key("agent-1", None), runtime_state(7, now), now)
            .await;

        let other = key("agent-2", Some("user-1"));
        let other_agent = key("agent-2", None);
        assert!(cache.current(&other, &other_agent, now).await.is_none());
        assert!(matches!(
            cache
                .accept(other.clone(), runtime_state(2, now), now)
                .await,
            AcceptOutcome::Accepted
        ));
    }

    #[tokio::test]
    async fn runtime_cache_stays_bounded() {
        let cache = RuntimeStateCache::default();
        let now = Utc::now();
        let senders: Vec<_> = (0..MAX_RUNTIME_STATES + 10)
            .map(|i| key("agent-1", Some(&format!("user-{i}"))))
            .collect();
        for sender in &senders {
            cache
                .accept(sender.clone(), runtime_state(1, now), now)
                .await;
        }
        assert_eq!(cache.states.read().await.len(), MAX_RUNTIME_STATES);
        assert!(
            cache.is_held(senders.last().unwrap()).await,
            "the newest arrival is kept"
        );
    }

    #[tokio::test]
    async fn runtime_cache_replaces_an_expired_state_regardless_of_version() {
        // A hostile or buggy version number must not pin the cache forever.
        let cache = RuntimeStateCache::default();
        let now = Utc::now();
        let k = key("agent-1", Some("user-1"));
        cache
            .accept(k.clone(), runtime_state(i64::MAX, now), now)
            .await;

        let later = now + chrono::Duration::seconds(120);
        let lapsed = cache.current(&k, &k, later).await;
        assert!(lapsed.is_none(), "60 s ttl has lapsed");
        assert!(matches!(
            cache
                .accept(k.clone(), runtime_state(1, later), later)
                .await,
            AcceptOutcome::Accepted
        ));
        assert_eq!(cache.current(&k, &k, later).await.unwrap().state_version, 1);
    }

    /// The reply is ingested as the agent, so what it returns is the agent's
    /// own state, and a sender the inbound stage holds nothing for reads it.
    #[tokio::test]
    async fn reply_ingest_feeds_the_agent_state_the_inbound_stage_falls_back_to() {
        let (url, server) = serve_once(http_200(&envelope_body())).await;
        let inbound = EpisodeIngestStage::new(
            "https://unused.invalid",
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        );
        let reply = EpisodeReplyIngestStage::new(
            &url,
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        )
        .with_allow_insecure(true)
        .with_runtime_state_of(&inbound);

        let (mut ctx, _) = failing_ctx("hello");
        ctx.response = Some("a reply".into());
        reply.process(&mut ctx).await.unwrap();
        server.await.unwrap();

        let states = &inbound.runtime_states;
        assert!(states.is_held(&key("agent-1", None)).await);
        assert!(!states.is_held(&key("agent-1", Some("user-1"))).await);
        inbound.apply_runtime_state(&mut ctx).await;
        let affect = ctx.get_ext::<RuntimeAffectSnapshot>().unwrap();
        assert_eq!(affect.state_version, 3);
    }

    #[tokio::test]
    async fn apply_runtime_state_clears_a_snapshot_once_the_state_expires() {
        let (mut ctx, url) = failing_ctx("hello");
        let stage = EpisodeIngestStage::new(
            &url,
            Arc::new(StaticAuth::new("t")),
            CredentialKind::EndUser,
        );
        let now = Utc::now();
        let then = now - chrono::Duration::seconds(3_600);
        assert!(matches!(
            stage
                .runtime_states
                .accept(RuntimeStateKey::sender(&ctx), runtime_state(1, then), then)
                .await,
            AcceptOutcome::Accepted
        ));
        ctx.set_ext(RuntimeAffectSnapshot {
            pleasure: 0.9,
            arousal: 0.0,
            dominance: 0.0,
            state_version: 1,
        });

        stage.apply_runtime_state(&mut ctx).await;

        assert!(
            ctx.get_ext::<RuntimeAffectSnapshot>().is_none(),
            "an expired state must not leave yesterday's mood in the prompt"
        );
    }

    #[test]
    fn process_response_without_runtime_state_decodes_to_none() {
        for body in [
            r#"{"message_processed": true}"#,
            r#"{"runtime_state": null}"#,
        ] {
            let response: ProcessEpisodeResponse = serde_json::from_str(body).unwrap();
            assert!(response.runtime_state.is_none(), "{body}");
        }
    }
}
