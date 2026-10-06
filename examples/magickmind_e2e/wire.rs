use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use futures::{Sink, SinkExt, Stream, StreamExt};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::tungstenite::{Error as WsError, Message as Frame};

pub struct WireObserver {
    label: &'static str,
    rx: mpsc::UnboundedReceiver<Value>,
    seen: Vec<Value>,
    closed: Arc<Mutex<Option<String>>>,
}

impl WireObserver {
    pub async fn connect(url: &str, token: &str, label: &'static str) -> Result<Self> {
        let (ws, _) = tokio_tungstenite::connect_async(url)
            .await
            .with_context(|| format!("{label}: websocket connect to {url} failed"))?;
        let (mut sink, stream) = ws.split();
        let connect = json!({
            "id": 1,
            "connect": { "data": { "token": token }, "name": "mindroid-e2e" },
        });
        sink.send(Frame::Text(connect.to_string())).await?;

        let (tx, rx) = mpsc::unbounded_channel();
        let (ready_tx, ready_rx) = oneshot::channel::<Result<()>>();
        let closed = Arc::new(Mutex::new(None));
        let close_reason = Arc::clone(&closed);
        tokio::spawn(async move {
            let reason = pump(stream, sink, tx, ready_tx, label).await;
            *close_reason.lock().unwrap() = Some(reason);
        });

        match tokio::time::timeout(Duration::from_secs(10), ready_rx).await {
            Ok(Ok(outcome)) => outcome?,
            Ok(Err(_)) => bail!("{label}: connection closed before the connect reply"),
            Err(_) => bail!("{label}: no connect reply within 10s"),
        }
        Ok(Self {
            label,
            rx,
            seen: Vec::new(),
            closed,
        })
    }

    /// `Err` says why nothing matched, phrased to follow "nothing reached …".
    pub async fn wait_for(
        &mut self,
        within: Duration,
        matches: impl Fn(&Value) -> bool,
    ) -> Result<Value, String> {
        if let Some(hit) = self.seen.iter().find(|m| matches(m)) {
            return Ok(hit.clone());
        }
        let deadline = tokio::time::Instant::now() + within;
        while let Ok(next) = tokio::time::timeout_at(deadline, self.rx.recv()).await {
            let Some(msg) = next else {
                let reason = self.closed.lock().unwrap().clone().unwrap_or_default();
                return Err(format!(
                    "before the {} connection dropped ({reason})",
                    self.label
                ));
            };
            self.seen.push(msg.clone());
            if matches(&msg) {
                return Ok(msg);
            }
        }
        Err(format!("within {within:?}"))
    }
}

async fn pump(
    mut stream: impl Stream<Item = Result<Frame, WsError>> + Unpin,
    mut sink: impl Sink<Frame> + Unpin,
    tx: mpsc::UnboundedSender<Value>,
    ready: oneshot::Sender<Result<()>>,
    label: &'static str,
) -> String {
    let mut ready = Some(ready);
    while let Some(frame) = stream.next().await {
        let text = match frame {
            Ok(Frame::Text(text)) => text,
            Ok(Frame::Close(Some(close))) => {
                return format!("closed {} {}", close.code, close.reason);
            }
            Ok(Frame::Close(None)) => return "closed with no reason".into(),
            Ok(_) => continue,
            Err(e) => return e.to_string(),
        };
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(msg) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            // A bare `{}` is the server's ping; an unanswered one drops the connection.
            if msg.as_object().is_some_and(|o| o.is_empty()) {
                if sink.send(Frame::Text("{}".into())).await.is_err() {
                    return "answering a ping failed".into();
                }
                continue;
            }
            if msg["id"] == 1 {
                let outcome = match msg.get("error") {
                    Some(err) => Err(anyhow::anyhow!("{label}: connect refused: {err}")),
                    None => Ok(()),
                };
                if let Some(ready) = ready.take() {
                    let _ = ready.send(outcome);
                }
                continue;
            }
            let data = &msg["push"]["pub"]["data"];
            if data["type"] == "chat_message" && tx.send(data["payload"].clone()).is_err() {
                return "the harness stopped listening".into();
            }
        }
    }
    "the server ended the stream".into()
}
