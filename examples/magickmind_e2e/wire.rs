use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::tungstenite::Message as Frame;

pub struct WireObserver {
    rx: mpsc::UnboundedReceiver<Value>,
    seen: Vec<Value>,
}

impl WireObserver {
    pub async fn connect(url: &str, token: &str, label: &'static str) -> Result<Self> {
        let (ws, _) = tokio_tungstenite::connect_async(url)
            .await
            .with_context(|| format!("{label}: websocket connect to {url} failed"))?;
        let (mut sink, mut stream) = ws.split();
        let connect = json!({
            "id": 1,
            "connect": { "data": { "token": token }, "name": "mindroid-e2e" },
        });
        sink.send(Frame::Text(connect.to_string())).await?;

        let (tx, rx) = mpsc::unbounded_channel();
        let (ready_tx, ready_rx) = oneshot::channel::<Result<()>>();
        tokio::spawn(async move {
            let mut ready = Some(ready_tx);
            while let Some(Ok(frame)) = stream.next().await {
                let Frame::Text(text) = frame else { continue };
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    let Ok(msg) = serde_json::from_str::<Value>(line) else {
                        continue;
                    };
                    // A bare `{}` is the server's ping; an unanswered one drops the connection.
                    if msg.as_object().is_some_and(|o| o.is_empty()) {
                        if sink.send(Frame::Text("{}".into())).await.is_err() {
                            return;
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
                        return;
                    }
                }
            }
        });

        match tokio::time::timeout(Duration::from_secs(10), ready_rx).await {
            Ok(Ok(outcome)) => outcome?,
            Ok(Err(_)) => bail!("{label}: connection closed before the connect reply"),
            Err(_) => bail!("{label}: no connect reply within 10s"),
        }
        Ok(Self {
            rx,
            seen: Vec::new(),
        })
    }

    pub async fn expect(
        &mut self,
        within: Duration,
        matches: impl Fn(&Value) -> bool,
    ) -> Option<Value> {
        if let Some(hit) = self.seen.iter().find(|m| matches(m)) {
            return Some(hit.clone());
        }
        let deadline = tokio::time::Instant::now() + within;
        while let Ok(Some(msg)) = tokio::time::timeout_at(deadline, self.rx.recv()).await {
            self.seen.push(msg.clone());
            if matches(&msg) {
                return Some(msg);
            }
        }
        None
    }
}
