use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

pub type Script = Arc<dyn Fn(&Value) -> String + Send + Sync>;

#[derive(Clone)]
pub struct StubLlm {
    pub base_url: String,
    requests: Arc<Mutex<Vec<Value>>>,
}

impl StubLlm {
    pub async fn start(script: Script) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}", listener.local_addr()?);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&requests);
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                let script = Arc::clone(&script);
                let log = Arc::clone(&log);
                tokio::spawn(async move {
                    if let Err(e) = serve(sock, script, log).await {
                        tracing::warn!("stub llm connection failed: {e}");
                    }
                });
            }
        });
        Ok(Self { base_url, requests })
    }

    pub fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }
}

async fn serve(mut sock: TcpStream, script: Script, log: Arc<Mutex<Vec<Value>>>) -> Result<()> {
    let mut raw = Vec::new();
    let mut buf = [0u8; 8192];
    let body = loop {
        let n = sock.read(&mut buf).await?;
        if n == 0 {
            anyhow::bail!("client closed before sending a full request");
        }
        raw.extend_from_slice(&buf[..n]);
        let text = String::from_utf8_lossy(&raw);
        if let Some((head, body)) = text.split_once("\r\n\r\n") {
            let len = head
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().to_string())
                })
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            if body.len() >= len {
                break body.to_string();
            }
        }
    };

    let request: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
    let reply = script(&request);
    let streaming = request["stream"].as_bool().unwrap_or(false);
    log.lock().unwrap().push(request);

    let (content_type, payload) = if streaming {
        let chunk = json!({
            "id": "stub", "object": "chat.completion.chunk", "created": 0, "model": "stub",
            "choices": [{ "index": 0, "delta": { "role": "assistant", "content": reply }, "finish_reason": "stop" }],
        });
        (
            "text/event-stream",
            format!("data: {chunk}\n\ndata: [DONE]\n\n"),
        )
    } else {
        let completion = json!({
            "id": "stub", "object": "chat.completion", "created": 0, "model": "stub",
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": reply }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 },
        });
        ("application/json", completion.to_string())
    };
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
        payload.len()
    );
    sock.write_all(response.as_bytes()).await?;
    sock.shutdown().await.ok();
    Ok(())
}

pub fn message_text(message: &Value) -> String {
    match &message["content"] {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

pub fn last_user_text(request: &Value) -> String {
    request["messages"]
        .as_array()
        .and_then(|msgs| msgs.iter().rev().find(|m| m["role"] == "user"))
        .map(message_text)
        .unwrap_or_default()
}
