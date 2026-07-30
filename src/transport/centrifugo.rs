// Centrifugo WebSocket transport for Mindroid

use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use futures::{SinkExt, StreamExt};
use tokio::sync::{RwLock, mpsc};
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

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

/// Compute the refresh interval as 80% of the token TTL, clamped to a sane minimum.
///
/// The TTL is server-controlled, so the multiplication saturates rather than
/// overflowing on a hostile value.
fn refresh_interval(ttl_secs: u64) -> Duration {
    Duration::from_secs(ttl_secs.saturating_mul(80) / 100).max(MIN_REFRESH_INTERVAL)
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

/// Build the Centrifugo `connect` command. A service user's token goes
/// top-level (Centrifugo-verified, JWKS/HMAC); an end user's token goes in
/// `data.token` with the top-level field omitted, so Centrifugo routes to the
/// bifrost connect proxy.
fn build_connect_cmd(
    id: u64,
    token: &str,
    kind: crate::models::CredentialKind,
) -> serde_json::Value {
    let connect = match kind {
        crate::models::CredentialKind::ServiceUser => serde_json::json!({
            "token": token,
            "name": "mindroid",
        }),
        crate::models::CredentialKind::EndUser => serde_json::json!({
            "data": { "token": token },
            "name": "mindroid",
        }),
    };
    serde_json::json!({ "id": id, "connect": connect })
}

#[derive(Debug, Default)]
struct State {
    connected: bool,
}

type WsSink = futures::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    WsMessage,
>;

type WsStream = futures::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

/// Result of a successful Centrifugo handshake.
#[allow(dead_code)]
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
    state: Arc<RwLock<State>>,
    allow_insecure: bool,
}

impl CentrifugoTransport {
    pub fn new(ws_url: &str, agent_id: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            ws_url: ws_url.to_string(),
            agent_id: agent_id.to_string(),
            identity,
            kind: crate::models::CredentialKind::ServiceUser,
            state: Arc::new(RwLock::new(State::default())),
            allow_insecure: false,
        }
    }

    /// Select the credential kind. An end-user credential connects via the
    /// bifrost connect proxy (token in `connect.data`) and listens on its own
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
    allow_insecure: bool,
) -> Result<HandshakeResult> {
    let token = identity.get_token().await?;
    check_url_security(ws_url, &token, allow_insecure)?;

    let service_user_id = extract_jwt_sub(&token)
        .ok_or_else(|| transport_err("Failed to extract sub claim from JWT"))?;

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
    let connect_cmd = build_connect_cmd(1, &token, kind);
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

    // Inbound channel follows the credential. An end-user token is already
    // auto-subscribed to its own `user:{agent}#{agent}` channel by the connect
    // proxy (the MM-378 fan-out target), so no explicit subscribe is sent — a
    // duplicate would be rejected. A service-user credential must explicitly
    // subscribe to `personal:{agent}#{sub}`.
    let channel = match kind {
        crate::models::CredentialKind::EndUser => format!("user:{agent_id}#{agent_id}"),
        crate::models::CredentialKind::ServiceUser => {
            let channel = format!("personal:{agent_id}#{service_user_id}");
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
            channel
        }
    };
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
fn parse_push(text: &str) -> Option<Message> {
    let val: serde_json::Value = serde_json::from_str(text).ok()?;
    let push = val.get("push")?;
    let channel = push.get("channel")?.as_str()?.to_string();
    let outer = push.get("pub")?.get("data")?;

    // The actual message fields may be nested inside outer.data or outer.payload.
    // magickmind publishes a WsMessage envelope: {"type":"chat_message","payload":{...}}
    // so we try "payload" as well as the legacy "data" key.
    let inner = outer
        .get("data")
        .or_else(|| outer.get("payload"))
        .unwrap_or(outer);

    debug!("Centrifugo push data: {outer}");

    let id = inner
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let content = inner
        .get("content")
        .or_else(|| inner.get("text"))
        .or_else(|| inner.get("message"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let sender_id = inner
        .get("sent_by_user_id")
        .or_else(|| inner.get("sender_id"))
        .or_else(|| inner.get("user_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let channel_id = outer
        .get("magickspace_id")
        .or_else(|| inner.get("magickspace_id"))
        .or_else(|| inner.get("channel_id"))
        .and_then(|v| v.as_str())
        .unwrap_or(&channel)
        .to_string();

    let mut msg = Message::new(content, sender_id, channel_id).with_id(id);
    msg.platform = Some("centrifugo".into());
    Some(msg)
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

    async fn disconnect(&mut self) -> Result<()> {
        self.state.write().await.connected = false;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        // Blocking read of the async RwLock is safe here because the lock is never held long.
        self.state.try_read().map(|s| s.connected).unwrap_or(false)
    }

    async fn listen(&self, tx: mpsc::Sender<Message>) -> Result<()> {
        let ws_url = self.ws_url.clone();
        let agent_id = self.agent_id.clone();
        let identity = Arc::clone(&self.identity);
        let kind = self.kind;
        let state = Arc::clone(&self.state);
        let allow_insecure = self.allow_insecure;

        tokio::spawn(async move {
            let mut backoff = Duration::from_secs(1);
            const MAX_BACKOFF: Duration = Duration::from_secs(30);
            let mut cmd_id: u32 = 3; // 1=connect, 2=subscribe, 3+ for refresh

            loop {
                match handshake(&ws_url, &agent_id, identity.as_ref(), kind, allow_insecure).await {
                    Ok(HandshakeResult {
                        mut sink,
                        mut stream,
                        ttl,
                        ..
                    }) => {
                        state.write().await.connected = true;
                        backoff = Duration::from_secs(1); // reset on success

                        info!("Centrifugo listen loop started");

                        // Schedule token refresh at 80% of TTL
                        let refresh_duration = ttl
                            .map(refresh_interval)
                            .unwrap_or(Duration::from_secs(86400)); // no expiry: sleep forever
                        let mut refresh_timer = tokio::time::interval(refresh_duration);
                        refresh_timer.tick().await; // consume the immediate first tick

                        loop {
                            tokio::select! {
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

                                                if let Some(message) = parse_push(line) {
                                                    if tx.send(message).await.is_err() {
                                                        warn!("Message receiver dropped, stopping listen loop");
                                                        state.write().await.connected = false;
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
                                    // Refresh frames carry only a top-level `token`
                                    // (JWKS-gated, no `data`), so skip them for
                                    // proxy-routed creds — the proxy governs expiry.
                                    if kind == crate::models::CredentialKind::EndUser {
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

                        state.write().await.connected = false;
                    }
                    Err(e) => {
                        error!("Centrifugo handshake failed: {e}");
                    }
                }

                warn!("Reconnecting to Centrifugo in {}s", backoff.as_secs());
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }
        });

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
        let cmd = build_connect_cmd(1, "jwt-abc", crate::models::CredentialKind::ServiceUser);
        let connect = &cmd["connect"];
        assert_eq!(connect["token"], "jwt-abc");
        assert!(connect.get("data").is_none());
        assert_eq!(connect["name"], "mindroid");
    }

    #[test]
    fn connect_cmd_end_user_routes_to_proxy() {
        // Token in data.token, top-level absent → Centrifugo skips JWKS and calls the proxy.
        let cmd = build_connect_cmd(1, "eu-hs256", crate::models::CredentialKind::EndUser);
        let connect = &cmd["connect"];
        assert_eq!(connect["data"]["token"], "eu-hs256");
        assert!(connect.get("token").is_none());
        assert_eq!(connect["name"], "mindroid");
    }

    #[test]
    fn refresh_interval_clamps_hostile_ttl() {
        assert_eq!(refresh_interval(0), MIN_REFRESH_INTERVAL);
        assert_eq!(refresh_interval(1), MIN_REFRESH_INTERVAL);
        assert_eq!(refresh_interval(3600), Duration::from_secs(2880));
        // A server-controlled TTL must not overflow the 80% computation.
        assert_eq!(
            refresh_interval(u64::MAX),
            Duration::from_secs(u64::MAX / 100)
        );
    }
}
