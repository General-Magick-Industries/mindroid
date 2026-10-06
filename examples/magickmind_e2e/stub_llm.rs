use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const MAX_REQUEST_BYTES: usize = 8 << 20;

pub type Script = Arc<dyn Fn(&Value) -> String + Send + Sync>;

pub struct StubLlm {
    pub base_url: String,
    requests: Arc<Mutex<Vec<Value>>>,
}

impl StubLlm {
    pub async fn start(script: Script) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let secret = uuid::Uuid::new_v4().simple().to_string();
        let base_url = format!("http://{}/{secret}", listener.local_addr()?);
        let route = Arc::new(format!("POST /{secret}/"));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&requests);
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                let (script, log, route) =
                    (Arc::clone(&script), Arc::clone(&log), Arc::clone(&route));
                tokio::spawn(async move {
                    if let Err(e) = serve(sock, &route, script, log).await {
                        tracing::warn!("stub llm connection failed: {e:#}");
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

async fn read_request(sock: &mut TcpStream) -> Result<(String, Vec<u8>)> {
    let mut raw = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = sock.read(&mut buf).await?;
        ensure!(n > 0, "client closed before sending a full request");
        raw.extend_from_slice(&buf[..n]);
        ensure!(
            raw.len() <= MAX_REQUEST_BYTES,
            "request exceeds {MAX_REQUEST_BYTES} bytes"
        );
        let Some(end) = raw.windows(4).position(|w| w == b"\r\n\r\n") else {
            continue;
        };
        let head = String::from_utf8_lossy(&raw[..end]).into_owned();
        let body = end + 4;
        let len = content_length(&head)?;
        ensure!(
            len <= MAX_REQUEST_BYTES - body,
            "content-length {len} exceeds {MAX_REQUEST_BYTES} bytes"
        );
        if raw.len() >= body + len {
            return Ok((head, raw[body..body + len].to_vec()));
        }
    }
}

fn content_length(head: &str) -> Result<usize> {
    let header = |name: &str| {
        head.lines().skip(1).find_map(|line| {
            line.split_once(':')
                .filter(|(key, _)| key.trim().eq_ignore_ascii_case(name))
                .map(|(_, value)| value.trim())
        })
    };
    ensure!(
        header("transfer-encoding").is_none(),
        "chunked request bodies are not supported"
    );
    header("content-length")
        .context("request has no content-length")?
        .parse()
        .context("content-length is not a number")
}

async fn serve(
    mut sock: TcpStream,
    route: &str,
    script: Script,
    log: Arc<Mutex<Vec<Value>>>,
) -> Result<()> {
    let (head, body) = read_request(&mut sock).await?;
    if !head.starts_with(route) {
        sock.write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
            .await?;
        return Ok(());
    }
    let request: Value = serde_json::from_slice(&body).context("request body is not JSON")?;
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
