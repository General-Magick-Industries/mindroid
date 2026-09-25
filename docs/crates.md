# Implementation Crate Reference

Reference for all implementation crates that provide concrete backends for Mindroid's core traits.

This document covers the 9 implementation crates that extend Mindroid's core capabilities. See [Core API](core.md) for trait definitions and [Architecture](architecture.md) for design principles.

---

## Transport Crates

### mindroid-transport-centrifugo

WebSocket transport connecting to Centrifugo real-time messaging server for distributed agent deployment.

**Purpose**: Enable remote message delivery through Centrifugo, allowing agents to receive messages from external systems over persistent WebSocket connections.

**Constructor**:
```rust
CentrifugoTransport::new(ws_url: &str, agent_id: &str, identity: Arc<dyn Identity>)
```

**Protocol Details**:
- Connects via WebSocket to Centrifugo server at `ws_url`
- Requires `wss://` when an auth token is configured — plaintext `ws://` is refused unless explicitly opted in via `.with_allow_insecure(true)` (or `transport.allow_insecure = true` in config; local development only)
- Performs automatic handshake with JWT token extracted from `Identity.get_token()`
- Subscribes to personal channel: `personal:{agent_id}#{service_user_id}`
  - `service_user_id` extracted from JWT `sub` claim via base64 decode
- Parses Centrifugo push frames (format: `{"push":{"channel":"...","pub":{"data":{...}}}}`)
- Extracts message fields: `id`, `content`, `sender_id`, `channel_id`

**Reconnection**:
- Automatic with exponential backoff (1s → 30s max) on connection loss
- Maintains connection state across reconnects

**send() Behavior**:
No-op. Response delivery is handled by pipeline persistence stages (e.g., MagickmindPersistence). Responses do not go through WebSocket.

**Dependencies**:
`tokio-tungstenite`, `base64`, `serde_json`

**Example**:
```rust
use mindroid_transport_centrifugo::CentrifugoTransport;

let transport = CentrifugoTransport::new(
    "wss://centrifugo:8000/connection/websocket",
    "my_agent",
    identity,
);
runtime.with_transport(transport).build().await?;
```

---

### mindroid-transport-stdio

Simple stdin/stdout transport for local development and interactive testing.

**Purpose**: Enable agent interaction via terminal I/O without external dependencies, ideal for prototyping and debugging.

**Constructor**:
```rust
StdioTransport::new()
// or
StdioTransport::default()
```

**Behavior**:
- `listen()`: Reads lines from stdin; each line becomes a Message
  - `sender_id`: `"stdin"`
  - `channel_id`: `"stdio"`
  - `message_type`: `Text`
- `send()`: Prints response content to stdout
- `connect()`/`disconnect()`: No-ops (always connected)
- `is_connected()`: Always returns `true`

**Dependencies**:
`tokio` (stdin/stdout)

**Example**:
```rust
use mindroid_transport_stdio::StdioTransport;

let transport = StdioTransport::new();
runtime.with_transport(transport).build().await?;

// Agent now reads from stdin and prints to stdout
```

---

## Pipeline Crates

### mindroid-pipeline-magickmind

Full Magick Mind platform pipeline with MagickMind context/persistence and Cortex LLM inference.

**Purpose**: Provide end-to-end agent processing: fetch conversation history, route to appropriate LLM, stream inference, and persist responses.

**Constructor**:
```rust
magickmind_pipeline(
    identity: Arc<dyn Identity>,
    base_url: &str,  // Magick Mind platform URL
    api_key: &str,   // Cortex API key
) -> Pipeline
```

**Pipeline Stages** (5 stages):

1. **ContextBuilder**
   - Calls `MagickmindClient.prepare_context(magickspace_id, participant_id, query, config, exclude_sender)`
   - The magickspace id comes from the message's `channel_id` (populated by the transport)
   - Fetches conversation history as `Vec<LlmMessage>`
   - Sets `ctx.llm_messages` (or system + user messages if no magickspace)

2. **Router**
   - Copies `model_type`, `model_ids`, `compute_power` from `ctx.agent_config`
   - Prepares these fields for Cortex request

3. **CortexProcessor** (StreamingStage)
   - Sends LLM request to Cortex API with SSE streaming
   - Non-streaming: Collects full response into `ctx.raw_response`
   - Streaming: Emits `StreamEvent::Thinking`, `Chunk`, `Complete`, `Error`
   - Handles `[DONE]` sentinel to end stream

4. **PostProcessor**
   - Copies `ctx.raw_response` to `ctx.final_response`
   - Trims whitespace

5. **MagickmindPersistence**
   - Saves final response to MagickMind via `MagickmindClient.save_message()`
   - Skips when the message carries no `channel_id` (magickspace id)

**MagickmindClient API**:
```rust
pub fn new(base_url: impl Into<String>, identity: Arc<dyn Auth>) -> Self
pub fn with_api_key(self, api_key: impl Into<String>) -> Self

pub async fn prepare_context(
    &self,
    magickspace_id: &str,
    participant_id: &str,
    query: &str,
    config: &MagickmindContextConfig,
    exclude_sender: Option<&str>,
) -> Result<PreparedContext>
// POST /v1/magickspaces/{id}/context
// Body: { participant_id, chat_history?, pelican?, corpus? }
// PreparedContext { messages: Vec<LlmMessage>, corpora: Vec<CorpusCatalogEntry> }

pub async fn save_message(
    &self,
    magickspace_id: &str,
    sender_id: &str,
    content: &str,
    reply_to_message_id: Option<&str>,
) -> Result<Option<String>>
// POST /v1/magickspaces/{id}/messages
// Body: { sender_id, content, reply_to_message_id? }
```

**CortexClient API**:
```rust
pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self

// Internal: stream_owned() returns BoxStream<'static, StreamEvent>
// Sends CortexRequest with { messages, model_type, compute_power, model_ids, magickspace_id, user_id }
// Receives SSE events: event_type (thinking, chunk, complete, error), content, data fields
```

**Dependencies**:
`reqwest`, `reqwest-eventsource`, `async-stream`, `serde_json`

**Example**:
```rust
use mindroid_pipeline_magickmind::magickmind_pipeline;

let pipeline = magickmind_pipeline(
    identity,
    "https://magickmind.platform",
    "sk-cortex-xxx",
);
runtime.with_pipeline(pipeline).build().await?;
```

---

### mindroid-pipeline-ollama

Local LLM pipeline using Ollama for offline inference.

**Purpose**: Enable agents to run completely locally with Ollama, no external API dependencies.

**Constructor**:
```rust
ollama_pipeline(base_url: &str, model: &str) -> Pipeline
// Example: ollama_pipeline("http://localhost:11434", "llama2")
```

**Pipeline Stages** (3 stages):

1. **SimpleContextBuilder**
   - Builds `Vec<LlmMessage>` from agent persona (system message) + user input (user message)
   - Sets `ctx.llm_messages`

2. **OllamaProcessor** (StreamingStage)
   - POST to `{base_url}/api/chat` with `{ model, messages, stream: bool }`
   - Non-streaming: Reads full response, sets `ctx.raw_response`
   - Streaming: Reads newline-delimited JSON chunks
     - Each line is `OllamaChunk: { message: { content }, done: bool }`
     - Emits `StreamEvent::Chunk` for each chunk
     - Emits `StreamEvent::Complete` when `done: true`

3. **PostProcessor**
   - Copies `ctx.raw_response` to `ctx.final_response`
   - Trims whitespace

**Ollama API Format**:
```rust
{
    "model": "llama2",
    "messages": [ /* LlmMessage array */ ],
    "stream": true  // or false
}
```

Response (streaming):
```json
{"message":{"content":"Hello","role":"assistant"},"done":false}
{"message":{"content":" world"},"role":"assistant"},"done":true}
```

**Dependencies**:
`reqwest`, `async-stream`, `serde_json`

**Example**:
```rust
use mindroid_pipeline_ollama::ollama_pipeline;

let pipeline = ollama_pipeline("http://localhost:11434", "llama2");
runtime.with_pipeline(pipeline).build().await?;
```

---

## Memory Crates

### mindroid-memory-magickmind

Remote message storage via MagickMind REST API for centralized conversation history.

**Purpose**: Persist all agent messages to a remote MagickMind service, enabling conversation continuity and cross-agent history sharing.

**Constructor**:
```rust
MagickmindMemory::new(base_url: &str, identity: Arc<dyn Auth>)
```

The magickspace id is passed per call as `channel_id` — it is not fixed at
construction time.

**API Endpoints** (all authenticated):

```rust
// Save a message
POST /v1/magickspaces/{magickspace_id}/messages
Body: { channel_id, sender_id, content, reply_to_id }
Response: { id: String }

// Get message history
GET /v1/magickspaces/{magickspace_id}/messages?channel_id={id}&limit={n}
Response: { messages: Vec<Message> }

// Clear all messages for a channel
DELETE /v1/magickspaces/{magickspace_id}/messages?channel_id={id}
```

**Authentication**:
- All requests include auth headers from `Identity.get_auth_headers()`
- Header format: `Authorization: Bearer {token}`

**Dependencies**:
`reqwest`, `serde_json`

**Example**:
```rust
use mindroid_memory_magickmind::MagickmindMemory;

let memory = MagickmindMemory::new(
    "https://magickmind.service",
    "magickspace_123",
    identity,
);
runtime.with_memory(memory).build().await?;
```

---

### mindroid-memory-sqlite

Local SQLite-backed message storage for offline agents.

**Purpose**: Provide persistent local message storage without external dependencies, suitable for edge agents and offline-first applications.

**Constructor**:
```rust
SqliteMemory::new(path: &str) -> Result<Self>
// Use ":memory:" for in-memory database
// Use "messages.db" for persistent file-based storage
```

**Schema**:
```sql
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    content TEXT NOT NULL,
    reply_to_id TEXT,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_channel
    ON messages(channel_id, timestamp);
```

**Async Pattern**:
- Uses `tokio::task::spawn_blocking` to run synchronous rusqlite operations in async context
- Wraps connection in `Arc<Mutex<Connection>>` for thread-safe access
- All async methods await the blocking task

**Timestamps**:
- Stored as RFC 3339 strings (e.g., `2025-02-21T15:30:45.123Z`)
- Parsed back to `DateTime<Utc>` on retrieval

**Dependencies**:
`rusqlite` (with bundled feature), `tokio`, `uuid`, `chrono`

**Example**:
```rust
use mindroid_memory_sqlite::SqliteMemory;

// In-memory (tests)
let memory = SqliteMemory::new(":memory:")?;

// Persistent file
let memory = SqliteMemory::new("./agent_memory.db")?;

runtime.with_memory(memory).build().await?;
```

---

## Identity Crates

### mindroid-identity-apikey

Email/password authentication with automatic token refresh.

**Purpose**: Authenticate with a backend service using email and password, managing token lifecycle and refresh automatically.

**Constructor**:
```rust
ApiKeyIdentity::new(base_url: &str, email: &str, password: &str)
```

**Auth Flow**:

1. **First call to `get_token()`** triggers login:
   - POST `{base_url}/v1/auth/login`
   - Body: `{ email, password }`
   - Receives: `{ access_token, refresh_token, expires_in }`
   - Caches token state in `Arc<RwLock<Option<TokenState>>>`

2. **Subsequent calls check expiration**:
   - If token expires within 10 seconds: refresh automatically
   - POST `{base_url}/v1/auth/refresh` with `{ refresh_token }`
   - Updates cached state with new tokens

3. **Double-check locking** for concurrency:
   - Acquire read lock first (fast path)
   - If refresh needed, upgrade to write lock only then
   - Prevents thundering herd during concurrent token requests

**Methods**:
```rust
async fn get_token() -> Result<String>
    // Returns valid access token (refreshes if needed)

async fn get_auth_headers() -> Result<Vec<(String, String)>>
    // Returns [("Authorization", "Bearer {token}")]

fn is_authenticated() -> bool
    // Non-async, checks cached expiration

async fn refresh() -> Result<()>
    // Manually trigger token refresh
```

**Dependencies**:
`reqwest`, `serde_json`, `chrono`, `tokio` (RwLock)

**Example**:
```rust
use mindroid_identity_apikey::ApiKeyIdentity;

let identity = Arc::new(ApiKeyIdentity::new(
    "https://auth.service",
    "user@example.com",
    "password123",
));
runtime.with_identity(identity).build().await?;
```

---

### mindroid-identity-static

Simple hardcoded token for development and testing.

**Purpose**: Provide a zero-configuration identity implementation for local testing and development.

**Constructor**:
```rust
StaticIdentity::new(token: impl Into<String>)
```

**Behavior**:
- `get_token()`: Always returns the hardcoded token
- `is_authenticated()`: Always returns `true`
- `refresh()`: No-op
- `get_auth_headers()`: Returns `[("Authorization", "Bearer {token}")]`

**No external dependencies** beyond core.

**Example**:
```rust
use mindroid_identity_static::StaticIdentity;

let identity = Arc::new(StaticIdentity::new("test-token-abc123"));
runtime.with_identity(identity).build().await?;
```

**Security Note**: Never use in production. For development/testing only.

---

## Observer Crate

### mindroid-observer-log

Structured logging for all agent lifecycle events via tracing.

**Purpose**: Provide comprehensive observability into agent runtime, with structured logs at appropriate detail levels.

**Constructor**:
```rust
LogObserver::new()
// or
LogObserver::default()
```

**Log Levels**:

| Event | Level | Details |
|-------|-------|---------|
| `on_start()` | `info!` | "Agent started" |
| `on_shutdown()` | `info!` | "Agent shutting down" |
| `on_message_received(msg)` | `info!` | `id`, `sender`, `channel`, message content length |
| `on_response_sent(channel, content)` | `info!` | `channel`, content length |
| `on_stream_event(event)` | `debug!` | Full event type details |
| `on_error(error)` | `error!` | Error message |

**Log Output Examples**:
```
INFO: Agent started
INFO: Message received id=msg_123 sender=user1 channel=general len=45
INFO: Response sent channel=general len=128
DEBUG: Stream event Chunk { content: "Hello " }
ERROR: Error occurred message="LLM request failed"
INFO: Agent shutting down
```

**Integration**:
- Requires `tracing` crate setup (e.g., `tracing-subscriber` for console output)
- Uses structured fields for programmatic filtering and analysis

**Dependencies**:
`tracing`

**Example**:
```rust
use mindroid_observer_log::LogObserver;

// Initialize tracing subscriber (typically in main)
tracing_subscriber::fmt::init();

// Add observer to runtime
runtime
    .with_observer(LogObserver::new())
    .build()
    .await?;
```

---

## Dependency Summary

| Crate | HTTP | Async | Storage | Auth | Serialization |
|-------|------|-------|---------|------|---------------|
| centrifugo | tokio-tungstenite | ✓ | — | Identity | serde_json |
| stdio | tokio | ✓ | — | — | — |
| magickmind | reqwest | ✓ | — | Identity | serde_json |
| ollama | reqwest | ✓ | — | — | serde_json |
| magickmind | reqwest | ✓ | — | Identity | serde_json |
| sqlite | rusqlite | ✓ | ✓ (local) | — | — |
| apikey | reqwest | ✓ | — | ✓ | serde_json |
| static | — | — | — | — | — |
| log | — | — | — | — | tracing |

---

## Common Patterns

### Combining Transports and Pipelines

```rust
// Local development: stdio + ollama
let runtime = RuntimeBuilder::new()
    .with_transport(StdioTransport::new())
    .with_pipeline(ollama_pipeline("http://localhost:11434", "llama2"))
    .with_identity(StaticIdentity::new("dev-token"))
    .build()
    .await?;

// Production: Centrifugo + Magick Mind
let identity = Arc::new(ApiKeyIdentity::new(
    "https://auth.service",
    email,
    password,
));
let runtime = RuntimeBuilder::new()
    .with_transport(CentrifugoTransport::new(
        "wss://centrifugo.example.com/connection/websocket",
        agent_id,
        identity.clone(),
    ))
    .with_pipeline(magickmind_pipeline(
        identity,
        "https://magickmind.platform",
        api_key,
    ))
    .with_memory(MagickmindMemory::new(
        "https://magickmind.service",
        identity.clone(),
    ))
    .with_observer(LogObserver::new())
    .build()
    .await?;
```

### Custom Memory + Remote Transport

```rust
// Local storage + remote messaging
let runtime = RuntimeBuilder::new()
    .with_transport(CentrifugoTransport::new(ws_url, agent_id, identity.clone()))
    .with_memory(SqliteMemory::new("./agent.db")?)
    .with_pipeline(ollama_pipeline("http://localhost:11434", "mistral"))
    .with_observer(LogObserver::new())
    .build()
    .await?;
```

---

## See Also

- [Core API](core.md) — Trait definitions and interfaces
- [Architecture](architecture.md) — Design principles and message lifecycle
