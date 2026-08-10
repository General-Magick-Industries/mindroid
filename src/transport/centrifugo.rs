// Centrifugo WebSocket transport for Mindroid

use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use futures::{SinkExt, StreamExt};
use std::sync::atomic::Ordering;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::{Auth, Message, MindroidError, Response, Result, Transport};

fn transport_err(msg: impl Into<String>) -> MindroidError {
    MindroidError::Transport {
        message: msg.into(),
        source: None,
    }
}

/// Reject plaintext `ws://` URLs when an auth token would ride on them.
///
/// The JWT is sent in the connect frame and every refresh frame; over an
/// unencrypted socket any on-path observer can read and replay it (or inject
/// a malicious refresh TTL). Require `wss://` unless the caller explicitly
/// opts into insecure transport for local development.
fn check_url_security(ws_url: &str, token: &str, allow_insecure: bool) -> Result<()> {
    if ws_url.starts_with("ws://") && !token.is_empty() && !allow_insecure {
        return Err(transport_err(format!(
            "refusing to send auth token over plaintext {ws_url}: use wss://, \
             or set transport allow_insecure = true for local development"
        )));
    }
    Ok(())
}

/// Minimum interval between token refreshes.
///
/// Guards against a hostile or buggy server handing back a tiny (or zero) TTL:
/// `tokio::time::interval` panics on a zero period, and a very small period
/// would hammer the auth backend.
const MIN_REFRESH_INTERVAL: Duration = Duration::from_secs(30);

/// When to next act on the token.
///
/// This tick serves two different mechanisms, and they want different cadences:
/// a service user sends a Centrifugo `refresh` frame (any time before expiry
/// works), while a proxy-routed credential must land inside its own rotation
/// window or the poll is a no-op. The stricter of the two governs.
///
/// The TTL is server-controlled, so the arithmetic saturates rather than
/// overflowing on a hostile value.
fn refresh_interval(ttl_secs: u64) -> Duration {
    #[cfg(feature = "magickmind")]
    let candidate = crate::auth::enduser::rotation_deadline(ttl_secs);
    #[cfg(not(feature = "magickmind"))]
    let candidate = Duration::from_secs(ttl_secs.saturating_mul(80) / 100);

    candidate.max(MIN_REFRESH_INTERVAL)
}

/// Lifetime assumed when the connect reply carries no TTL.
///
/// Not "sleep forever": a missing TTL means unknown, not infinite, and the
/// server's default is one hour. The end-user route always lands here — the
/// connect proxy owns expiry, so no TTL ever comes back on the reply.
const ASSUMED_TTL_SECS: u64 = 3600;

/// Tick used when the connect reply carries no TTL.
///
/// Derived from [`ASSUMED_TTL_SECS`] through [`refresh_interval`] rather than
/// used raw: a tick equal to the assumed lifetime fires at or after expiry,
/// outside the rotation window, so the first poll would present an already-dead
/// token and latch the credential.
fn unknown_ttl_interval() -> Duration {
    refresh_interval(ASSUMED_TTL_SECS)
}

/// Extract the `sub` claim from a JWT without verifying the signature.
fn extract_jwt_sub(token: &str) -> Option<String> {
    let parts: Vec<&str> = token.splitn(3, '.').collect();
    if parts.len() < 2 {
        return None;
    }
    let payload = parts[1];
    // JWT uses base64url without padding
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .ok()?;
    let json: serde_json::Value = serde_json::from_slice(&decoded).ok()?;
    json.get("sub")?.as_str().map(|s| s.to_string())
}

/// A deployment's Centrifugo conventions: where the token rides in the `connect`
/// payload, what the inbound channel is called, and whether the client must
/// subscribe explicitly.
///
/// These are cluster-specific, not protocol-level. The default
/// [`ProxyChannelNaming`] encodes the conventions Magick Mind's deployment uses;
/// point mindroid at your own cluster and implement this instead.
pub trait ChannelNaming: Send + Sync + 'static {
    /// The `connect` payload body (the value under `"connect"`). `token` is the
    /// live credential; `kind` distinguishes an end user from a service user.
    fn connect_payload(
        &self,
        token: &str,
        kind: crate::models::CredentialKind,
    ) -> serde_json::Value;

    /// The channel this agent receives messages on. `service_user_id` is the
    /// JWT `sub` claim.
    fn inbound_channel(
        &self,
        agent_id: &str,
        service_user_id: &str,
        kind: crate::models::CredentialKind,
    ) -> String;

    /// Whether the client sends an explicit `subscribe` for `inbound_channel`.
    /// Return `false` when the server auto-subscribes — a duplicate is rejected.
    fn needs_explicit_subscribe(&self, kind: crate::models::CredentialKind) -> bool {
        let _ = kind;
        true
    }

    /// Whether periodic client-side token refresh frames apply. Return `false`
    /// when a connect proxy governs expiry instead.
    fn sends_refresh_frames(&self, kind: crate::models::CredentialKind) -> bool {
        let _ = kind;
        true
    }
}

/// The default [`ChannelNaming`]: a service-user token is JWKS/HMAC-verified by
/// Centrifugo itself (top-level `token`) and subscribes to `personal:{agent}#{sub}`;
/// an end-user token is routed to a connect proxy (`data.token`, no top-level
/// field), which auto-subscribes it to `user:{agent}#{agent}` and owns expiry.
#[derive(Debug, Clone, Default)]
pub struct ProxyChannelNaming;

impl ChannelNaming for ProxyChannelNaming {
    fn connect_payload(
        &self,
        token: &str,
        kind: crate::models::CredentialKind,
    ) -> serde_json::Value {
        match kind {
            crate::models::CredentialKind::ServiceUser => serde_json::json!({
                "token": token,
                "name": "mindroid",
            }),
            crate::models::CredentialKind::EndUser => serde_json::json!({
                "data": { "token": token },
                "name": "mindroid",
            }),
        }
    }

    fn inbound_channel(
        &self,
        agent_id: &str,
        service_user_id: &str,
        kind: crate::models::CredentialKind,
    ) -> String {
        match kind {
            // NOTE: this channel has a second writer — mindroid-voice publishes
            // to it as an output sink. Inbound pushes here are therefore not
            // guaranteed to be from a third party, which is why `parse_push`
            // drops unattributed messages rather than synthesizing a sender.
            crate::models::CredentialKind::EndUser => format!("user:{agent_id}#{agent_id}"),
            crate::models::CredentialKind::ServiceUser => {
                format!("personal:{agent_id}#{service_user_id}")
            }
        }
    }

    fn needs_explicit_subscribe(&self, kind: crate::models::CredentialKind) -> bool {
        kind != crate::models::CredentialKind::EndUser
    }

    fn sends_refresh_frames(&self, kind: crate::models::CredentialKind) -> bool {
        kind != crate::models::CredentialKind::EndUser
    }
}

/// Build the Centrifugo `connect` command from the naming strategy's payload.
fn build_connect_cmd(
    id: u64,
    token: &str,
    kind: crate::models::CredentialKind,
    naming: &dyn ChannelNaming,
) -> serde_json::Value {
    serde_json::json!({ "id": id, "connect": naming.connect_payload(token, kind) })
}

/// Liveness flag. An atomic, not a lock: `try_read` on a write-preferring
/// `RwLock` fails while a writer is merely pending, reporting "disconnected"
/// for a healthy agent.
type Connected = std::sync::atomic::AtomicBool;

type WsSink = futures::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    WsMessage,
>;

type WsStream = futures::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

/// Result of a successful Centrifugo handshake.
struct HandshakeResult {
    sink: WsSink,
    stream: WsStream,
    channel: String,
    /// Token TTL in seconds from the connect reply, if the token expires.
    ttl: Option<u64>,
}

pub struct CentrifugoTransport {
    ws_url: String,
    agent_id: String,
    identity: Arc<dyn Auth>,
    kind: crate::models::CredentialKind,
    naming: Arc<dyn ChannelNaming>,
    connected: Arc<Connected>,
    allow_insecure: bool,
    /// Cancels the spawned listener. `disconnect` triggers it and awaits the
    /// handle, so a cancelled agent leaves no socket or subscription behind.
    cancel: CancellationToken,
    listener: std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl CentrifugoTransport {
    pub fn new(ws_url: &str, agent_id: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            ws_url: ws_url.to_string(),
            agent_id: agent_id.to_string(),
            identity,
            kind: crate::models::CredentialKind::ServiceUser,
            naming: Arc::new(ProxyChannelNaming),
            connected: Arc::new(Connected::new(false)),
            allow_insecure: false,
            cancel: CancellationToken::new(),
            listener: std::sync::Mutex::new(None),
        }
    }

    /// Override the deployment's channel conventions. Defaults to
    /// [`ProxyChannelNaming`].
    pub fn with_channel_naming(mut self, naming: Arc<dyn ChannelNaming>) -> Self {
        self.naming = naming;
        self
    }

    /// Select the credential kind. An end-user credential connects via a
    /// connect proxy (token in `connect.data`) and listens on its own
    /// `user:` channel; a service user is JWKS-verified (top-level `token`) and
    /// listens on `personal:`.
    pub fn with_credential_kind(mut self, kind: crate::models::CredentialKind) -> Self {
        self.kind = kind;
        self
    }

    /// Permit sending the auth token over plaintext `ws://` (local development only).
    pub fn with_allow_insecure(mut self, allow_insecure: bool) -> Self {
        self.allow_insecure = allow_insecure;
        self
    }
}

/// Perform the Centrifugo connect + subscribe handshake.
async fn handshake(
    ws_url: &str,
    agent_id: &str,
    identity: &dyn Auth,
    kind: crate::models::CredentialKind,
    naming: &dyn ChannelNaming,
    allow_insecure: bool,
) -> Result<HandshakeResult> {
    let token = identity.get_token().await?;
    check_url_security(ws_url, &token, allow_insecure)?;

    let service_user_id = extract_jwt_sub(&token)
        .ok_or_else(|| transport_err("Failed to extract sub claim from JWT"))?;

    // On the end-user route the channel is built from config's `agent_id` on both
    // sides, and the proxy validates the token rather than the channel name — so a
    // wrong `agent_id` connects, logs "Listening on channel", and then receives
    // nothing forever, indistinguishable from an idle channel.
    if kind == crate::models::CredentialKind::EndUser && service_user_id != agent_id {
        warn!(
            agent_id = %agent_id,
            token_subject = %service_user_id,
            "Configured agent_id does not match the token subject; this agent will \
             listen on a channel nothing publishes to"
        );
    }

    let url = if ws_url.contains('?') {
        format!("{}&cf_ws_frame_ping_pong=true", ws_url)
    } else {
        format!("{}?cf_ws_frame_ping_pong=true", ws_url)
    };

    debug!("Connecting to Centrifugo at {}", url);
    let (ws_stream, _) = connect_async(&url)
        .await
        .map_err(|e| MindroidError::Transport {
            message: format!("WebSocket connection failed: {e}"),
            source: Some(Box::new(e)),
        })?;

    let (mut sink, mut stream) = ws_stream.split();

    // Send connect command.
    let connect_cmd = build_connect_cmd(1, &token, kind, naming);
    sink.send(WsMessage::Text(connect_cmd.to_string()))
        .await
        .map_err(|e| MindroidError::Transport {
            message: format!("Failed to send connect command: {e}"),
            source: Some(Box::new(e)),
        })?;

    // Read connect reply and extract TTL
    let mut ttl: Option<u64> = None;
    let reply = stream
        .next()
        .await
        .ok_or_else(|| transport_err("WebSocket closed before connect reply"))?
        .map_err(|e| MindroidError::Transport {
            message: format!("WebSocket error reading connect reply: {e}"),
            source: Some(Box::new(e)),
        })?;

    match &reply {
        WsMessage::Text(text) => {
            let val: serde_json::Value = serde_json::from_str(text)
                .map_err(|e| transport_err(format!("Invalid connect reply JSON: {e}")))?;
            if val.get("error").is_some() {
                return Err(transport_err(format!("Centrifugo connect error: {val}")));
            }
            // Extract TTL if the connection expires
            let expires = val
                .pointer("/connect/expires")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if expires {
                ttl = val.pointer("/connect/ttl").and_then(|v| v.as_u64());
            }
            debug!("Centrifugo connected: {val} (ttl={ttl:?})");
        }
        other => {
            return Err(transport_err(format!(
                "Unexpected connect reply frame: {other:?}"
            )));
        }
    }

    // Inbound channel and whether to claim it explicitly are the deployment's
    // conventions, not the protocol's — a server that auto-subscribes rejects a
    // duplicate subscribe, so the strategy decides.
    let channel = naming.inbound_channel(agent_id, &service_user_id, kind);
    if naming.needs_explicit_subscribe(kind) {
        let subscribe_cmd = serde_json::json!({
            "id": 2,
            "subscribe": { "channel": channel },
        });
        sink.send(WsMessage::Text(subscribe_cmd.to_string()))
            .await
            .map_err(|e| MindroidError::Transport {
                message: format!("Failed to send subscribe command: {e}"),
                source: Some(Box::new(e)),
            })?;

        let sub_reply = stream
            .next()
            .await
            .ok_or_else(|| transport_err("WebSocket closed before subscribe reply"))?
            .map_err(|e| MindroidError::Transport {
                message: format!("WebSocket error reading subscribe reply: {e}"),
                source: Some(Box::new(e)),
            })?;

        match &sub_reply {
            WsMessage::Text(text) => {
                let val: serde_json::Value = serde_json::from_str(text)
                    .map_err(|e| transport_err(format!("Invalid subscribe reply JSON: {e}")))?;
                if val.get("error").is_some() {
                    return Err(transport_err(format!("Centrifugo subscribe error: {val}")));
                }
            }
            other => {
                return Err(transport_err(format!(
                    "Unexpected subscribe reply frame: {other:?}"
                )));
            }
        }
    }
    info!("Listening on channel: {channel}");

    Ok(HandshakeResult {
        sink,
        stream,
        channel,
        ttl,
    })
}

/// Parse a Centrifugo push frame into a `Message`.
///
/// Push format: `{"push":{"channel":"...","pub":{"data":{...}}}}`
/// A stable id for a payload that carries none, so redelivery of an identical
/// message dedupes instead of looking new on every arrival.
fn content_derived_id(channel: &str, payload: &serde_json::Value) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    channel.hash(&mut h);
    payload.to_string().hash(&mut h);
    format!("centrifugo-{:016x}", h.finish())
}

fn parse_push(text: &str, subscribed_channel: &str) -> Option<Message> {
    let val: serde_json::Value = serde_json::from_str(text).ok()?;
    let push = val.get("push")?;
    let channel = push.get("channel")?.as_str()?.to_string();

    // Without this a server-side misroute is indistinguishable from real traffic.
    if !subscribed_channel.is_empty() && channel != subscribed_channel {
        warn!(
            got = %channel,
            expected = %subscribed_channel,
            "Dropping push for a channel this connection is not subscribed to"
        );
        return None;
    }

    let outer = push.get("pub")?.get("data")?;

    // The actual message fields may be nested inside outer.data or outer.payload.
    // magickmind publishes a WsMessage envelope: {"type":"chat_message","payload":{...}}
    // so we try "payload" as well as the legacy "data" key.
    let inner = outer
        .get("data")
        .or_else(|| outer.get("payload"))
        .unwrap_or(outer);

    debug!("Centrifugo push data: {outer}");

    // A fresh UUID per delivery would defeat the runtime's dedupe guard, so an
    // id-less payload gets a stable content-derived one instead.
    let id = inner
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| content_derived_id(&channel, outer));

    // A client tool result arrives as {type:"tool_result",payload:{…}} in the
    // envelope; rewrite it into the <tool_result> form the LLM history expects.
    let content = if let Some(framed) =
        crate::tools::remote::normalize_tool_result(&outer.to_string())
            .or_else(|| crate::tools::remote::normalize_tool_result(&inner.to_string()))
    {
        framed
    } else {
        inner
            .get("content")
            .or_else(|| inner.get("text"))
            .or_else(|| inner.get("message"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    };

    // A missing sender is fatal: the self-echo guard compares `sender_id` to the
    // agent id, and a synthesized placeholder can never match — so a co-tenanted
    // agent would consume its own output forever.
    let Some(sender_id) = inner
        .get("sent_by_user_id")
        .or_else(|| inner.get("sender_id"))
        .or_else(|| inner.get("user_id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
    else {
        warn!(
            channel = %channel,
            "Dropping push with no sender: it cannot be checked against the \
             self-echo guard"
        );
        return None;
    };

    // `channel_id` becomes the artifact scope, a trust boundary — so it comes from
    // the subscribed channel, never the payload.
    let channel_id = match outer
        .get("magickspace_id")
        .or_else(|| inner.get("magickspace_id"))
        .or_else(|| inner.get("channel_id"))
        .and_then(|v| v.as_str())
    {
        Some(claimed) if claimed != channel => {
            warn!(
                claimed = %claimed,
                actual = %channel,
                "Push claims a different channel id than the one it arrived on; \
                 using the actual channel"
            );
            channel.clone()
        }
        _ => channel.clone(),
    };

    let mut msg = Message::new(content, sender_id, channel_id).with_id(id);
    msg.platform = Some("centrifugo".into());
    if let Some(authenticated_sender) = push
        .get("pub")
        .and_then(|publication| publication.get("info"))
        .and_then(|info| info.get("user"))
        .and_then(serde_json::Value::as_str)
        .filter(|sender| !sender.is_empty())
    {
        msg.metadata.insert(
            "authenticated_sender_id".into(),
            serde_json::Value::String(authenticated_sender.to_string()),
        );
    }
    attach_image_metadata(&mut msg, inner);
    Some(msg)
}

/// Extract an image from a centrifugo payload into `msg.metadata`, supporting
/// both a hosted URL and an inline base64 blob.
///
/// Recognized fields on the inner message object:
/// - `image_url` (or `image` when it holds an `http(s)://` URL) → stored as
///   `metadata["image_url"]`.
/// - `image_data` (or `image` when it holds a `data:` URL or raw base64) →
///   stored as `metadata["image_data"]`, with the MIME type in
///   `metadata["image_mime"]` (parsed from a `data:` prefix, or from an
///   explicit `image_mime` field, defaulting to `image/png`).
///
/// The downstream `AttachImage` stage reads these keys; URL takes precedence
/// when both are present.
fn attach_image_metadata(msg: &mut Message, inner: &serde_json::Value) {
    use serde_json::Value;

    let as_str = |v: &Value| v.as_str().map(|s| s.to_string());

    // 1. Explicit URL field, or `image` that looks like a URL.
    let url = inner.get("image_url").and_then(as_str).or_else(|| {
        inner
            .get("image")
            .and_then(as_str)
            .filter(|s| s.starts_with("http://") || s.starts_with("https://"))
    });
    if let Some(url) = url {
        msg.metadata.insert("image_url".into(), Value::String(url));
        return;
    }

    // 2. Inline base64: explicit `image_data`, or `image` holding a data: URL
    //    or raw base64.
    let blob = inner
        .get("image_data")
        .and_then(as_str)
        .or_else(|| inner.get("image").and_then(as_str));
    let Some(blob) = blob else { return };

    // Split an optional `data:<mime>;base64,` prefix from the payload.
    let (mime_from_prefix, b64) = match blob.strip_prefix("data:") {
        Some(rest) => match rest.split_once(";base64,") {
            Some((mime, data)) => (Some(mime.to_string()), data.to_string()),
            None => (None, blob.clone()),
        },
        None => (None, blob.clone()),
    };

    let mime = inner
        .get("image_mime")
        .and_then(as_str)
        .or(mime_from_prefix)
        .unwrap_or_else(|| "image/png".to_string());

    msg.metadata.insert("image_data".into(), Value::String(b64));
    msg.metadata
        .insert("image_mime".into(), Value::String(mime));
}

/// Check if a text frame is a Centrifugo refresh reply and extract the new TTL.
fn parse_refresh_ttl(text: &str) -> Option<u64> {
    let val: serde_json::Value = serde_json::from_str(text).ok()?;
    val.pointer("/refresh/ttl").and_then(|v| v.as_u64())
}

#[async_trait]
impl Transport for CentrifugoTransport {
    fn name(&self) -> &str {
        "centrifugo"
    }

    async fn connect(&mut self) -> Result<()> {
        // Just validate that we can get a token and that it won't ride on a
        // plaintext socket. The actual WebSocket connection is established by
        // listen(), which owns the connection lifecycle.
        let token = self.identity.get_token().await?;
        check_url_security(&self.ws_url, &token, self.allow_insecure)?;
        Ok(())
    }

    /// Cancel the listener and wait for it to unwind. Flipping a flag is not
    /// enough — the task overwrites it on reconnect, and its only other exit is
    /// a failed `tx.send`, which a quiet channel never triggers.
    async fn disconnect(&mut self) -> Result<()> {
        self.cancel.cancel();
        self.connected.store(false, Ordering::SeqCst);

        let handle = self.listener.lock().unwrap().take();
        if let Some(handle) = handle
            && tokio::time::timeout(Duration::from_secs(5), handle)
                .await
                .is_err()
        {
            warn!("Centrifugo listener did not stop within 5s");
        }
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn listen(&self, tx: mpsc::Sender<Message>) -> Result<()> {
        let ws_url = self.ws_url.clone();
        let agent_id = self.agent_id.clone();
        let identity = Arc::clone(&self.identity);
        let kind = self.kind;
        let naming = Arc::clone(&self.naming);
        let connected = Arc::clone(&self.connected);
        let allow_insecure = self.allow_insecure;
        let cancel = self.cancel.clone();

        let handle = tokio::spawn(async move {
            let mut backoff = Duration::from_secs(1);
            const MAX_BACKOFF: Duration = Duration::from_secs(30);
            let mut cmd_id: u32 = 3; // 1=connect, 2=subscribe, 3+ for refresh

            loop {
                if cancel.is_cancelled() {
                    info!("Centrifugo listener cancelled");
                    break;
                }

                match handshake(
                    &ws_url,
                    &agent_id,
                    identity.as_ref(),
                    kind,
                    naming.as_ref(),
                    allow_insecure,
                )
                .await
                {
                    Ok(HandshakeResult {
                        mut sink,
                        mut stream,
                        channel: subscribed_channel,
                        ttl,
                    }) => {
                        connected.store(true, Ordering::SeqCst);
                        backoff = Duration::from_secs(1); // reset on success

                        info!("Centrifugo listen loop started");

                        // Act on the token before it expires. A missing TTL means
                        // unknown, not infinite — poll at the server's default.
                        let refresh_duration = ttl
                            .map(refresh_interval)
                            .unwrap_or_else(unknown_ttl_interval);
                        let mut refresh_timer = tokio::time::interval(refresh_duration);
                        refresh_timer.tick().await; // consume the immediate first tick

                        loop {
                            tokio::select! {
                                biased;
                                _ = cancel.cancelled() => {
                                    info!("Centrifugo listener cancelled; closing socket");
                                    let _ = sink.close().await;
                                    connected.store(false, Ordering::SeqCst);
                                    return;
                                }
                                frame = stream.next() => {
                                    match frame {
                                        Some(Ok(WsMessage::Text(text))) => {
                                            // Centrifugo batches replies/pushes as
                                            // newline-delimited JSON in one frame, so
                                            // parse each line independently.
                                            for line in text.split('\n') {
                                                let line = line.trim();
                                                if line.is_empty() {
                                                    continue;
                                                }

                                                // Check if this is a refresh reply with a new TTL
                                                if let Some(new_ttl) = parse_refresh_ttl(line) {
                                                    let new_duration = refresh_interval(new_ttl);
                                                    refresh_timer = tokio::time::interval(new_duration);
                                                    refresh_timer.tick().await; // consume immediate tick
                                                    debug!("Token refreshed, next refresh in {}s", new_duration.as_secs());
                                                    continue;
                                                }

                                                if let Some(message) = parse_push(line, &subscribed_channel) {
                                                    if tx.send(message).await.is_err() {
                                                        warn!("Message receiver dropped, stopping listen loop");
                                                        connected.store(false, Ordering::SeqCst);
                                                        return;
                                                    }
                                                } else {
                                                    debug!("Ignoring non-push frame: {line}");
                                                }
                                            }
                                        }
                                        Some(Ok(WsMessage::Close(_))) => {
                                            info!("Centrifugo WebSocket closed");
                                            break;
                                        }
                                        Some(Ok(WsMessage::Ping(_) | WsMessage::Pong(_))) => {}
                                        Some(Ok(WsMessage::Binary(bytes))) => {
                                            // Length only: Debug-formatting the payload
                                            // dumps the whole buffer into the logs.
                                            warn!(
                                                bytes = bytes.len(),
                                                "Dropping binary frame: binary is not a supported \
                                                 inbound encoding (pushes must be JSON text)"
                                            );
                                        }
                                        Some(Ok(other)) => {
                                            debug!("Ignoring WebSocket frame: {other:?}");
                                        }
                                        Some(Err(e)) => {
                                            error!("WebSocket read error: {e}");
                                            break;
                                        }
                                        None => {
                                            info!("WebSocket stream ended");
                                            break;
                                        }
                                    }
                                }
                                _ = refresh_timer.tick() => {
                                    // Refresh frames don't apply to a proxy-routed
                                    // credential, but rotation still must be driven:
                                    // EndUserAuth rotates lazily in get_token and owns
                                    // no timer, so skipping the tick entirely lets an
                                    // idle agent die at expiry.
                                    if !naming.sends_refresh_frames(kind) {
                                        if let Err(e) = identity.get_token().await {
                                            error!("Token rotation failed: {e}");
                                            break;
                                        }
                                        continue;
                                    }
                                    // Time to refresh the token
                                    match identity.get_token().await {
                                        Ok(new_token) => {
                                            // Re-check before every refresh so the
                                            // "no token over plaintext" invariant
                                            // holds by design, not by accident.
                                            if let Err(e) = check_url_security(&ws_url, &new_token, allow_insecure) {
                                                error!("Refusing token refresh: {e}");
                                                break;
                                            }
                                            cmd_id += 1;
                                            let refresh_cmd = serde_json::json!({
                                                "id": cmd_id,
                                                "refresh": {
                                                    "token": new_token
                                                }
                                            });
                                            if let Err(e) = sink.send(WsMessage::Text(refresh_cmd.to_string())).await {
                                                error!("Failed to send refresh command: {e}");
                                                break;
                                            }
                                            debug!("Sent token refresh command (id={cmd_id})");
                                        }
                                        Err(e) => {
                                            error!("Failed to get new token for refresh: {e}");
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        connected.store(false, Ordering::SeqCst);
                    }
                    Err(e) => {
                        error!("Centrifugo handshake failed: {e}");
                    }
                }

                // A rejected credential cannot be fixed by waiting. Retrying forever
                // holds `tx` open, so `run()` never returns — a zombie that looks
                // alive to a supervisor.
                if identity.is_terminal() {
                    error!(
                        "Credential is permanently rejected; stopping the Centrifugo \
                         listener so the runtime can shut down"
                    );
                    break;
                }

                warn!("Reconnecting to Centrifugo in {}s", backoff.as_secs());
                tokio::select! {
                    _ = cancel.cancelled() => {
                        info!("Centrifugo listener cancelled during backoff");
                        break;
                    }
                    _ = tokio::time::sleep(backoff) => {}
                }
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }

            connected.store(false, Ordering::SeqCst);
        });

        *self.listener.lock().unwrap() = Some(handle);
        Ok(())
    }

    async fn send(&self, _response: &Response) -> Result<Option<String>> {
        // No-op: response delivery is handled by the pipeline's persistence stage.
        Ok(None)
    }
}

/// Extension to `Message` to allow overriding the auto-generated id.
trait MessageExt {
    fn with_id(self, id: String) -> Self;
}

impl MessageExt for Message {
    fn with_id(mut self, id: String) -> Self {
        self.id = id;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_token_over_plaintext_ws() {
        assert!(check_url_security("ws://example.com/ws", "jwt-token", false).is_err());
    }

    #[test]
    fn allows_wss_with_token() {
        assert!(check_url_security("wss://example.com/ws", "jwt-token", false).is_ok());
    }

    #[test]
    fn allows_plaintext_ws_without_token() {
        assert!(check_url_security("ws://localhost:8000/ws", "", false).is_ok());
    }

    #[test]
    fn allows_plaintext_ws_with_explicit_insecure_flag() {
        assert!(check_url_security("ws://localhost:8000/ws", "jwt-token", true).is_ok());
    }

    #[test]
    fn connect_cmd_service_user_uses_top_level_field() {
        let cmd = build_connect_cmd(
            1,
            "jwt-abc",
            crate::models::CredentialKind::ServiceUser,
            &ProxyChannelNaming,
        );
        let connect = &cmd["connect"];
        assert_eq!(connect["token"], "jwt-abc");
        assert!(connect.get("data").is_none());
        assert_eq!(connect["name"], "mindroid");
    }

    #[test]
    fn connect_cmd_end_user_routes_to_proxy() {
        // Token in data.token, top-level absent → Centrifugo skips JWKS and calls the proxy.
        let cmd = build_connect_cmd(
            1,
            "eu-hs256",
            crate::models::CredentialKind::EndUser,
            &ProxyChannelNaming,
        );
        let connect = &cmd["connect"];
        assert_eq!(connect["data"]["token"], "eu-hs256");
        assert!(connect.get("token").is_none());
        assert_eq!(connect["name"], "mindroid");
    }

    #[test]
    fn default_naming_matches_the_deployment_conventions() {
        use crate::models::CredentialKind::{EndUser, ServiceUser};
        let n = ProxyChannelNaming;

        assert_eq!(
            n.inbound_channel("agent7", "svc1", ServiceUser),
            "personal:agent7#svc1"
        );
        // An end-user channel is keyed by the agent on both sides, not by `sub`.
        assert_eq!(
            n.inbound_channel("agent7", "svc1", EndUser),
            "user:agent7#agent7"
        );

        // The proxy auto-subscribes end users and owns their expiry.
        assert!(n.needs_explicit_subscribe(ServiceUser));
        assert!(!n.needs_explicit_subscribe(EndUser));
        assert!(n.sends_refresh_frames(ServiceUser));
        assert!(!n.sends_refresh_frames(EndUser));
    }

    /// A third party pointing mindroid at their own cluster must be able to
    /// replace the channel grammar wholesale.
    #[test]
    fn a_custom_naming_overrides_channel_and_connect_shape() {
        struct FlatNaming;
        impl ChannelNaming for FlatNaming {
            fn connect_payload(
                &self,
                token: &str,
                _kind: crate::models::CredentialKind,
            ) -> serde_json::Value {
                serde_json::json!({ "token": token })
            }
            fn inbound_channel(
                &self,
                agent_id: &str,
                _service_user_id: &str,
                _kind: crate::models::CredentialKind,
            ) -> String {
                format!("agents.{agent_id}")
            }
        }

        let n = FlatNaming;
        assert_eq!(
            n.inbound_channel("a1", "svc", crate::models::CredentialKind::EndUser),
            "agents.a1"
        );

        let cmd = build_connect_cmd(1, "tok", crate::models::CredentialKind::EndUser, &n);
        assert_eq!(cmd["connect"]["token"], "tok");
        assert!(cmd["connect"].get("name").is_none());

        // Defaulted methods keep the conservative behavior: subscribe and refresh.
        assert!(n.needs_explicit_subscribe(crate::models::CredentialKind::EndUser));
        assert!(n.sends_refresh_frames(crate::models::CredentialKind::EndUser));
    }

    /// A credential with no notion of terminal failure must not stop a
    /// reconnect loop — the default keeps a static token retrying.
    #[test]
    fn auth_is_not_terminal_by_default() {
        let auth = crate::auth::static_id::StaticAuth::new("tok");
        assert!(!Auth::is_terminal(&auth));
    }

    /// The listener must stop on cancellation rather than leaking its task,
    /// socket, and subscription. Uses an unroutable address so the loop is in
    /// its reconnect-backoff path when cancelled — the case a quiet channel
    /// produces, and the one a `tx.send` failure can never reach.
    #[tokio::test]
    async fn disconnect_stops_the_listener_instead_of_leaking_it() {
        let auth = Arc::new(crate::auth::static_id::StaticAuth::new(""));
        let mut transport = CentrifugoTransport::new("ws://127.0.0.1:1/ws", "agent", auth)
            .with_allow_insecure(true);

        let (tx, _rx) = mpsc::channel(8);
        transport.listen(tx).await.unwrap();

        // Wait out four failed handshakes (backoff 1s, 2s, 4s) so the task is
        // parked in an 8s sleep, not between attempts. The sleep is what has to
        // be interruptible: a `tx.send` failure can never reach a sleeping task,
        // and a bare `sleep(backoff).await` would make disconnect wait it out.
        tokio::time::sleep(Duration::from_millis(7500)).await;

        let start = std::time::Instant::now();
        tokio::time::timeout(Duration::from_secs(30), transport.disconnect())
            .await
            .expect("disconnect must not hang on a listener stuck in backoff")
            .expect("clean disconnect");

        assert!(
            start.elapsed() < Duration::from_secs(3),
            "disconnect must interrupt the backoff sleep, not wait it out (took {:?})",
            start.elapsed()
        );
        assert!(!transport.is_connected());
    }

    /// `is_connected` must not report "disconnected" merely because a writer is
    /// in flight — a supervisor would bounce a healthy agent.
    #[tokio::test]
    async fn is_connected_is_not_confused_by_concurrent_writers() {
        let auth = Arc::new(crate::auth::static_id::StaticAuth::new("t"));
        let transport = CentrifugoTransport::new("wss://example.com/ws", "agent", auth);

        transport.connected.store(true, Ordering::SeqCst);
        let flag = Arc::clone(&transport.connected);

        // Hammer the flag from another task while reading it here.
        let writer = tokio::spawn(async move {
            for _ in 0..1000 {
                flag.store(true, Ordering::SeqCst);
            }
        });

        assert!(transport.is_connected());
        writer.await.unwrap();
        assert!(transport.is_connected());
    }

    fn push(channel: &str, data: serde_json::Value) -> String {
        serde_json::json!({ "push": { "channel": channel, "pub": { "data": data } } }).to_string()
    }

    /// The runtime's self-echo guard compares `sender_id` to the agent id. A
    /// synthesized placeholder can never match, so an agent co-tenanted on a
    /// channel it also writes to would consume its own output forever.
    #[test]
    fn a_push_with_no_sender_is_dropped() {
        let frame = push("user:a1#a1", serde_json::json!({ "content": "hi" }));
        assert!(
            parse_push(&frame, "user:a1#a1").is_none(),
            "an unattributable message must not be delivered"
        );
    }

    /// The dedupe guard keys off `msg.id`. A fresh UUID per delivery makes an
    /// identical redelivery look new every time, so the id must be derived from
    /// the payload when the publisher supplies none.
    #[test]
    fn an_id_less_push_dedupes_across_identical_redeliveries() {
        let data = serde_json::json!({ "content": "hi", "sender_id": "u1" });
        let a = parse_push(&push("user:a1#a1", data.clone()), "user:a1#a1").unwrap();
        let b = parse_push(&push("user:a1#a1", data), "user:a1#a1").unwrap();
        assert_eq!(a.id, b.id, "identical payloads must produce the same id");

        let other = parse_push(
            &push(
                "user:a1#a1",
                serde_json::json!({ "content": "different", "sender_id": "u1" }),
            ),
            "user:a1#a1",
        )
        .unwrap();
        assert_ne!(a.id, other.id, "different payloads must not collide");
    }

    #[test]
    fn a_push_for_another_channel_is_dropped() {
        let frame = push(
            "user:someone-else#x",
            serde_json::json!({ "content": "hi", "sender_id": "u1" }),
        );
        assert!(parse_push(&frame, "user:a1#a1").is_none());
    }

    /// `channel_id` becomes the artifact scope, so a payload must not be able to
    /// name a different one and read another conversation's artifacts.
    #[test]
    fn a_payload_cannot_choose_its_own_scope() {
        let frame = push(
            "user:a1#a1",
            serde_json::json!({
                "content": "hi",
                "sender_id": "u1",
                "magickspace_id": "someone-elses-space",
            }),
        );
        let msg = parse_push(&frame, "user:a1#a1").expect("message is still delivered");
        assert_eq!(
            msg.channel_id, "user:a1#a1",
            "scope must come from the subscribed channel, not the payload"
        );
    }

    #[test]
    fn an_attributed_push_is_delivered_with_its_own_id() {
        let frame = push(
            "user:a1#a1",
            serde_json::json!({ "id": "m-7", "content": "hello", "sender_id": "u1" }),
        );
        let msg = parse_push(&frame, "user:a1#a1").expect("valid push");
        assert_eq!(msg.id, "m-7");
        assert_eq!(msg.sender_id, "u1");
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn authenticated_publication_identity_is_separate_from_payload_sender() {
        let frame = serde_json::json!({
            "push": {
                "channel": "user:a1#a1",
                "pub": {
                    "info": { "user": "authenticated-user" },
                    "data": { "content": "hello", "sender_id": "payload-user" }
                }
            }
        })
        .to_string();
        let msg = parse_push(&frame, "user:a1#a1").unwrap();

        assert_eq!(msg.sender_id, "payload-user");
        assert_eq!(msg.trusted_sender_id(), Some("authenticated-user"));
    }

    #[test]
    fn refresh_interval_clamps_hostile_ttl() {
        assert_eq!(refresh_interval(0), MIN_REFRESH_INTERVAL);
        assert_eq!(refresh_interval(1), MIN_REFRESH_INTERVAL);
        // A server-controlled TTL must not overflow the arithmetic.
        assert!(refresh_interval(u64::MAX) > MIN_REFRESH_INTERVAL);
    }

    /// The tick has to land inside `EndUserAuth`'s rotation window or the poll
    /// returns the cached token and rotates nothing — the token then expires and
    /// the next rotation presents an expired JWT, which is terminal.
    ///
    /// A fixed fraction of the TTL cannot satisfy this: 80% of the server's
    /// 3600s default leaves 720s, six times `ROTATE_BEFORE`.
    #[cfg(feature = "magickmind")]
    #[test]
    fn the_refresh_tick_lands_inside_the_rotation_window() {
        use crate::auth::enduser::ROTATE_BEFORE;

        for ttl in [600u64, 900, 1800, 3600, 7200, 86400] {
            let tick = refresh_interval(ttl);
            let remaining = Duration::from_secs(ttl).saturating_sub(tick);
            assert!(
                remaining <= ROTATE_BEFORE,
                "ttl={ttl}s: tick at {tick:?} leaves {remaining:?}, outside the \
                 {ROTATE_BEFORE:?} rotation window — the poll would be a no-op"
            );
            assert!(
                tick < Duration::from_secs(ttl),
                "ttl={ttl}s: tick at {tick:?} fires at or after expiry"
            );
        }
    }

    /// A connect reply without a TTL must not be read as "no expiry", and must
    /// leave real headroom before the assumed expiry — a tick landing *at* it is
    /// already too late. The end-user route always takes this path.
    #[test]
    fn an_unknown_ttl_polls_within_the_server_default_lifetime() {
        let interval = unknown_ttl_interval();
        assert!(
            interval < Duration::from_secs(ASSUMED_TTL_SECS),
            "a missing TTL means unknown, not infinite"
        );
        assert_eq!(
            interval,
            refresh_interval(ASSUMED_TTL_SECS),
            "the unknown-TTL tick must respect the same rotation window as a known one"
        );
    }
}

#[cfg(test)]
mod image_metadata_tests {
    use super::attach_image_metadata;
    use crate::models::Message;
    use serde_json::json;

    fn msg() -> Message {
        Message::new("hi", "user", "ch")
    }

    #[test]
    fn explicit_image_url() {
        let mut m = msg();
        attach_image_metadata(&mut m, &json!({ "image_url": "https://x.test/a.png" }));
        assert_eq!(m.metadata["image_url"], json!("https://x.test/a.png"));
        assert!(!m.metadata.contains_key("image_data"));
    }

    #[test]
    fn image_field_holding_url() {
        let mut m = msg();
        attach_image_metadata(&mut m, &json!({ "image": "http://x.test/a.jpg" }));
        assert_eq!(m.metadata["image_url"], json!("http://x.test/a.jpg"));
    }

    #[test]
    fn explicit_base64_blob_with_mime() {
        let mut m = msg();
        attach_image_metadata(
            &mut m,
            &json!({ "image_data": "QUJD", "image_mime": "image/jpeg" }),
        );
        assert_eq!(m.metadata["image_data"], json!("QUJD"));
        assert_eq!(m.metadata["image_mime"], json!("image/jpeg"));
    }

    #[test]
    fn data_url_prefix_is_stripped_and_mime_parsed() {
        let mut m = msg();
        attach_image_metadata(&mut m, &json!({ "image": "data:image/webp;base64,QUJD" }));
        assert_eq!(m.metadata["image_data"], json!("QUJD"));
        assert_eq!(m.metadata["image_mime"], json!("image/webp"));
    }

    #[test]
    fn url_wins_over_blob() {
        let mut m = msg();
        attach_image_metadata(
            &mut m,
            &json!({ "image_url": "https://x.test/a.png", "image_data": "QUJD" }),
        );
        assert!(m.metadata.contains_key("image_url"));
        assert!(!m.metadata.contains_key("image_data"));
    }

    #[test]
    fn no_image_fields_is_noop() {
        let mut m = msg();
        attach_image_metadata(&mut m, &json!({ "content": "just text" }));
        assert!(m.metadata.is_empty());
    }

    #[test]
    fn base64_without_mime_defaults_to_png() {
        let mut m = msg();
        attach_image_metadata(&mut m, &json!({ "image_data": "QUJD" }));
        assert_eq!(m.metadata["image_mime"], json!("image/png"));
    }
}
