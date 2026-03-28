// Centrifugo WebSocket transport for Mindroid

use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use futures::{SinkExt, StreamExt};
use tokio::sync::{mpsc, RwLock};
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{Auth, Message, MindroidError, Response, Result, Transport};

fn transport_err(msg: impl Into<String>) -> MindroidError {
    MindroidError::Transport { message: msg.into(), source: None }
}

/// Extract the `sub` claim from a JWT without verifying the signature.
fn extract_jwt_sub(token: &str) -> Option<String> {
    let parts: Vec<&str> = token.splitn(3, '.').collect();
    if parts.len() < 2 {
        return None;
    }
    let payload = parts[1];
    // JWT uses base64url without padding
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&decoded).ok()?;
    json.get("sub")?.as_str().map(|s| s.to_string())
}

#[derive(Debug, Default)]
struct State {
    connected: bool,
}

type WsSink = futures::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    WsMessage,
>;

type WsStream = futures::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
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
    state: Arc<RwLock<State>>,
}

impl CentrifugoTransport {
    pub fn new(ws_url: &str, agent_id: &str, identity: Arc<dyn Auth>) -> Self {
        Self {
            ws_url: ws_url.to_string(),
            agent_id: agent_id.to_string(),
            identity,
            state: Arc::new(RwLock::new(State::default())),
        }
    }
}

/// Perform the Centrifugo connect + subscribe handshake.
async fn handshake(ws_url: &str, agent_id: &str, identity: &dyn Auth) -> Result<HandshakeResult> {
    let token = identity.get_token().await?;

    let service_user_id = extract_jwt_sub(&token).ok_or_else(|| {
        transport_err("Failed to extract sub claim from JWT")
    })?;

    let url = if ws_url.contains('?') {
        format!("{}&cf_ws_frame_ping_pong=true", ws_url)
    } else {
        format!("{}?cf_ws_frame_ping_pong=true", ws_url)
    };

    debug!("Connecting to Centrifugo at {}", url);
    let (ws_stream, _) = connect_async(&url).await.map_err(|e| {
        MindroidError::Transport {
            message: format!("WebSocket connection failed: {e}"),
            source: Some(Box::new(e)),
        }
    })?;

    let (mut sink, mut stream) = ws_stream.split();

    // Send connect command
    let connect_cmd = serde_json::json!({
        "id": 1,
        "connect": {
            "token": token,
            "name": "mindroid"
        }
    });
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
            let val: serde_json::Value =
                serde_json::from_str(text).map_err(|e| transport_err(format!("Invalid connect reply JSON: {e}")))?;
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
            return Err(transport_err(format!("Unexpected connect reply frame: {other:?}")));
        }
    }

    // Subscribe to personal channel
    let channel = format!("personal:{}#{}", agent_id, service_user_id);
    let subscribe_cmd = serde_json::json!({
        "id": 2,
        "subscribe": {
            "channel": channel
        }
    });
    sink.send(WsMessage::Text(subscribe_cmd.to_string()))
        .await
        .map_err(|e| MindroidError::Transport {
            message: format!("Failed to send subscribe command: {e}"),
            source: Some(Box::new(e)),
        })?;

    // Read subscribe reply
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
            info!("Subscribed to channel: {channel}");
        }
        other => {
            return Err(transport_err(format!("Unexpected subscribe reply frame: {other:?}")));
        }
    }

    Ok(HandshakeResult { sink, stream, channel, ttl })
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
    let inner = outer.get("data")
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
        .get("mindspace_id")
        .or_else(|| inner.get("mindspace_id"))
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
        // Just validate that we can get a token. The actual WebSocket connection
        // is established by listen(), which owns the connection lifecycle.
        let _ = self.identity.get_token().await?;
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
        let state = Arc::clone(&self.state);

        tokio::spawn(async move {
        let mut backoff = Duration::from_secs(1);
        const MAX_BACKOFF: Duration = Duration::from_secs(30);
        let mut cmd_id: u32 = 3; // 1=connect, 2=subscribe, 3+ for refresh

        loop {
            match handshake(&ws_url, &agent_id, identity.as_ref()).await {
                Ok(HandshakeResult { mut sink, mut stream, ttl, .. }) => {
                    state.write().await.connected = true;
                    backoff = Duration::from_secs(1); // reset on success

                    info!("Centrifugo listen loop started");

                    // Schedule token refresh at 80% of TTL
                    let refresh_duration = ttl
                        .map(|t| Duration::from_secs(t * 80 / 100))
                        .unwrap_or(Duration::from_secs(86400)); // no expiry: sleep forever
                    let mut refresh_timer = tokio::time::interval(refresh_duration);
                    refresh_timer.tick().await; // consume the immediate first tick

                    loop {
                        tokio::select! {
                            frame = stream.next() => {
                                match frame {
                                    Some(Ok(WsMessage::Text(text))) => {
                                        // Check if this is a refresh reply with a new TTL
                                        if let Some(new_ttl) = parse_refresh_ttl(&text) {
                                            let new_duration = Duration::from_secs(new_ttl * 80 / 100);
                                            refresh_timer = tokio::time::interval(new_duration);
                                            refresh_timer.tick().await; // consume immediate tick
                                            debug!("Token refreshed, next refresh in {}s", new_duration.as_secs());
                                            continue;
                                        }

                                        if let Some(message) = parse_push(&text) {
                                            if tx.send(message).await.is_err() {
                                                warn!("Message receiver dropped, stopping listen loop");
                                                state.write().await.connected = false;
                                                return;
                                            }
                                        } else {
                                            debug!("Ignoring non-push frame: {text}");
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
                                // Time to refresh the token
                                match identity.get_token().await {
                                    Ok(new_token) => {
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
