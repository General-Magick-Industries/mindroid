# Core Crate API Reference

Complete reference for `mindroid-core` — the foundation crate defining all traits, types, and the runtime.

The core crate exports the minimal API surface needed to build an agent: error types, data models, trait definitions, configuration, and the runtime. All other crates depend on core and implement its traits.

## Error Types

All fallible operations return `Result<T>` where the error is `MindroidError`.

```rust
#[derive(thiserror::Error, Debug)]
pub enum MindroidError {
    #[error("Auth failed: {message}")]
    Auth {
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Transport error: {message}")]
    Transport {
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Pipeline error at stage '{stage}': {message}")]
    Pipeline {
        stage: String,
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Memory error: {message}")]
    Memory {
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("API error: {message} (HTTP {status_code:?})")]
    Api {
        message: String,
        status_code: Option<u16>,
    },

    #[error("Config error: {message}")]
    Config {
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl MindroidError {
    /// Convenience constructor for config errors without a source chain.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config { message: message.into(), source: None }
    }
}

pub type Result<T> = std::result::Result<T, MindroidError>;
```

### Error Variants

| Variant | When Used | Fields |
|---------|-----------|--------|
| `Auth` | Identity/authentication failures | `message`, optional `source` |
| `Transport` | Send/receive failures, connection issues | `message`, optional `source` |
| `Pipeline` | Stage processing errors | `stage` name, `message`, optional `source` |
| `Memory` | Memory backend failures (save, load, clear) | `message`, optional `source` |
| `Api` | HTTP/API errors from downstream services | `message`, optional HTTP `status_code` |
| `Config` | Configuration parsing or validation errors | `message`, optional `source` for chaining IO/parse errors |
| `Other` | All other errors (wraps anyhow::Error) | underlying anyhow::Error |

The `source` field allows chaining underlying errors. Use it when wrapping errors from dependencies:

```rust
match do_something() {
    Err(e) => Err(MindroidError::Transport {
        message: "Connection failed".into(),
        source: Some(Box::new(e)),
    }),
    Ok(v) => Ok(v),
}
```

## Data Models

### Message

Represents an incoming message from a transport (user, API, webhook, etc.).

**Fields:**
- `id: String` — Unique message identifier (auto-generated as UUID)
- `content: String` — Message body text
- `sender_id: String` — Who sent it (user ID, bot ID, webhook identifier)
- `sender_type: SenderType` — Origin type: User, Agent, System (default: User)
- `channel_id: String` — Where it came from (chat room ID, webhook source, etc.)
- `channel_type: ChannelType` — Channel kind: Direct, Group, Broadcast (default: Direct)
- `message_type: MessageType` — Content type: Text, Command, System, Image, Audio (default: Text)
- `timestamp: DateTime<Utc>` — When it was sent (auto-set to now)
- `metadata: HashMap<String, serde_json::Value>` — Transport-specific data (attachments, user info, etc.)

**Constructor:**

```rust
let msg = Message::new("Hello", "user123", "chat-room-1");
// Produces:
// Message {
//     id: "<uuid>",
//     content: "Hello",
//     sender_id: "user123",
//     sender_type: SenderType::User,
//     channel_id: "chat-room-1",
//     channel_type: ChannelType::Direct,
//     message_type: MessageType::Text,
//     timestamp: <now>,
//     metadata: {},
// }
```

All fields except `content`, `sender_id`, and `channel_id` have sensible defaults. Modify fields directly after construction:

```rust
let mut msg = Message::new("Hello", "user123", "chat-room-1");
msg.message_type = MessageType::Command;
msg.metadata.insert("priority".into(), json!("high"));
```

### Response

Represents an agent response to be sent back through the transport.

**Fields:**
- `content: String` — Response body text
- `channel_id: String` — Where to send it (same channel as incoming message)
- `sender_id: String` — Who sends it (usually the agent ID)
- `reply_to_id: Option<String>` — If set, marks this as a reply to a specific message
- `metadata: HashMap<String, serde_json::Value>` — Transport-specific data (attachments, formatting, etc.)

**Constructor:**

```rust
let resp = Response::new("Got it!", "chat-room-1", "agent-1");
// Produces:
// Response {
//     content: "Got it!",
//     channel_id: "chat-room-1",
//     sender_id: "agent-1",
//     reply_to_id: None,
//     metadata: {},
// }
```

**Methods:**

```rust
// Chain to set reply_to_id
let resp = Response::new("Acknowledged", "chat-room-1", "agent-1")
    .reply_to("msg-uuid-here");
// reply_to_id is now Some("msg-uuid-here")
```

### StreamEvent

Enum representing token-streaming events from a pipeline streaming stage. Used when you want to stream LLM output token-by-token rather than waiting for the full response.

```rust
pub enum StreamEvent {
    Thinking { content: String },           // Agent reasoning
    Chunk { content: String },               // Partial output token(s)
    ToolCall { name: String, arguments: String }, // Function call
    ToolResult { name: String, result: String }, // Function result
    Complete { content: String, usage: Option<TokenUsage> }, // Final output
    Error { message: String },               // Streaming failed
    Heartbeat,                               // Keep-alive signal
}
```

| Variant | When Emitted | Notes |
|---------|--------------|-------|
| `Thinking` | LLM is reasoning | Used for multi-step or chain-of-thought |
| `Chunk` | Token(s) arrive | Raw LLM output; may come in batches |
| `ToolCall` | LLM requests a tool | Agent will process and send ToolResult |
| `ToolResult` | Tool completes | Result fed back to LLM |
| `Complete` | Stream ends | Final output, optional token usage stats |
| `Error` | Stream fails | Message contains error details |
| `Heartbeat` | Periodic signal | Keep WebSocket/connection alive |

Example client code:

```rust
let ctx = message_context;
let mut stream = ctx.process_streaming();
while let Some(event) = stream.next().await {
    match event {
        StreamEvent::Chunk { content } => println!("{}", content),
        StreamEvent::Complete { usage, .. } => {
            if let Some(u) = usage {
                println!("Tokens: {}/{}", u.prompt_tokens, u.completion_tokens);
            }
        }
        StreamEvent::Error { message } => eprintln!("Error: {}", message),
        _ => {}
    }
}
```

### LlmMessage

Represents a single message in an LLM conversation history (system prompt, user input, assistant output, etc.).

**Fields:**
- `role: String` — Message source: "system", "user", "assistant", or custom
- `content: String` — Message text

**Constructors:**

```rust
let sys = LlmMessage::system("You are a helpful assistant");
let usr = LlmMessage::user("What is 2+2?");
let ast = LlmMessage::assistant("The answer is 4.");

// Or construct directly
let msg = LlmMessage {
    role: "custom_role".into(),
    content: "Custom message".into(),
};
```

Use these in a pipeline stage to build the conversation history (llm_messages field of PipelineContext):

```rust
ctx.llm_messages.push(LlmMessage::system("System prompt"));
ctx.llm_messages.push(LlmMessage::user(&ctx.message.content));
```

### TokenUsage

Tracks token consumption from an LLM call.

**Fields:**
- `prompt_tokens: u32` — Tokens in the input
- `completion_tokens: u32` — Tokens in the output
- `total_tokens: u32` — Sum

```rust
let usage = TokenUsage {
    prompt_tokens: 45,
    completion_tokens: 20,
    total_tokens: 65,
};
```

Typically set by a Processor stage and included in `StreamEvent::Complete`.

### Enums: MessageType, SenderType, ChannelType

Three simple enums representing message classifications. All serialize to snake_case and have `#[default]` on the first variant.

**MessageType** — What the message contains:
```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum MessageType {
    #[default]
    Text,    // Plain text
    Command, // User/agent command
    System,  // System message
    Image,   // Image attachment
    Audio,   // Audio attachment
}
```

**SenderType** — Who sent it:
```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum SenderType {
    #[default]
    User,   // Human user
    Agent,  // AI agent
    System, // System event
}
```

**ChannelType** — Where it came from:
```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ChannelType {
    #[default]
    Direct,    // 1-to-1 chat
    Group,     // Group chat
    Broadcast, // Broadcast/announcement
}
```

## Transport Trait

The `Transport` trait defines how the runtime sends and receives messages.

```rust
#[async_trait]
pub trait Transport: Send + Sync + 'static {
    fn name(&self) -> &str;
    async fn connect(&mut self) -> Result<()>;
    async fn disconnect(&mut self) -> Result<()>;
    async fn listen(&self, tx: mpsc::Sender<Message>) -> Result<()>;
    async fn send(&self, response: &Response) -> Result<Option<String>>;
    fn is_connected(&self) -> bool;
    async fn send_typing(&self, _channel_id: &str) -> Result<()> { Ok(()) }
    async fn health_check(&self) -> bool { self.is_connected() }
}
```

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `name()` | `fn(&self) -> &str` | Human-readable name (e.g., "stdio", "centrifugo") |
| `connect()` | `async fn(&mut self) -> Result<()>` | Establish connection to message source |
| `disconnect()` | `async fn(&mut self) -> Result<()>` | Gracefully close connection |
| `listen()` | `async fn(&self, tx: mpsc::Sender<Message>) -> Result<()>` | Poll for messages, send them via `tx` channel |
| `send()` | `async fn(&self, response: &Response) -> Result<Option<String>>` | Send a response, return optional message ID |
| `is_connected()` | `fn(&self) -> bool` | True if currently connected (used by runtime health checks) |
| `send_typing()` | `async fn(&self, _channel_id: &str) -> Result<()>` | Optional: send "typing..." indicator. Default: no-op. |
| `health_check()` | `async fn(&self) -> bool` | Optional: check transport health. Default: calls `is_connected()`. |

**Implementation notes:**

- `connect()` is called at runtime startup. Use it to establish network connections, subscribe to channels, etc.
- `listen()` is a long-running async task. Implement it to poll/subscribe to messages and send them via the provided channel.
- `send()` is called for each response. Return the remote message ID if the transport supports it.
- `is_connected()` is checked frequently; keep it fast.
- `send_typing()` and `health_check()` have default no-op/fallback implementations.

Example from mindroid-transport-stdio:

```rust
pub struct StdioTransport;

#[async_trait]
impl Transport for StdioTransport {
    fn name(&self) -> &str { "stdio" }

    async fn connect(&mut self) -> Result<()> {
        println!("Listening on stdin...");
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        Ok(())
    }

    async fn listen(&self, tx: mpsc::Sender<Message>) -> Result<()> {
        let stdin = std::io::stdin();
        // Read lines and send as messages
        for line in stdin.lock().lines() {
            let msg = Message::new(line?, "user", "stdin");
            tx.send(msg).await.ok();
        }
        Ok(())
    }

    async fn send(&self, response: &Response) -> Result<Option<String>> {
        println!("{}", response.content);
        Ok(None)
    }

    fn is_connected(&self) -> bool { true }
}
```

## Identity Trait

The `Identity` trait provides authentication and credential management.

```rust
#[async_trait]
pub trait Identity: Send + Sync + 'static {
    async fn get_token(&self) -> Result<String>;
    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>>;
    fn is_authenticated(&self) -> bool;
    async fn refresh(&self) -> Result<()>;
}
```

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `get_token()` | `async fn(&self) -> Result<String>` | Return auth token (JWT, API key, etc.) |
| `get_auth_headers()` | `async fn(&self) -> Result<Vec<(String, String)>>` | Return headers to include in requests (e.g., `Authorization: Bearer ...`) |
| `is_authenticated()` | `fn(&self) -> bool` | True if credentials are available (fast check) |
| `refresh()` | `async fn(&self) -> Result<()>` | Refresh token/credentials (OAuth token rotation, etc.) |

Pipeline stages call these to add auth headers to API requests:

```rust
// In a pipeline stage
let headers = identity.get_auth_headers().await?;
for (name, value) in headers {
    request.header(name, value);
}
```

**Blanket Impl for Arc<T>:**

Arc-wrapped identity implementations are also valid identities:

```rust
let identity: Arc<dyn Identity> = Arc::new(StaticIdentity::new("token"));
// Can be used directly:
let token = identity.get_token().await?;
```

This allows sharing identity across multiple threads/tasks.

## Memory Trait

The `Memory` trait handles message persistence (history, search, etc.).

```rust
#[async_trait]
pub trait Memory: Send + Sync + 'static {
    async fn save_message(
        &self,
        channel_id: &str,
        sender_id: &str,
        content: &str,
        reply_to_id: Option<&str>,
    ) -> Result<Option<String>>;

    async fn get_history(&self, channel_id: &str, limit: usize) -> Result<Vec<Message>>;

    async fn clear_history(&self, channel_id: &str) -> Result<()>;
}
```

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `save_message()` | Store a message by channel, sender, and optional reply target | Return optional message ID if backend supports it |
| `get_history()` | Retrieve message history for a channel (limited to N most recent) | Used by ContextBuilder stages to construct conversation context |
| `clear_history()` | Delete all messages for a channel | Called on `reset_conversation` command |

The runtime automatically calls `save_message()` for:
- Incoming messages (in `Runtime::run`)
- Outgoing responses (in `MessageContext::respond()`)

Pipeline stages call `get_history()` to build conversation history for the LLM:

```rust
// In a ContextBuilder stage
let history = memory.get_history(&ctx.message.channel_id, 10).await?;
for msg in history {
    ctx.llm_messages.push(LlmMessage::user(&msg.content));
}
```

### NoMemory

Built-in no-op memory implementation. Returns empty history, stores nothing.

```rust
pub struct NoMemory;

// All methods are no-ops; always returns Ok
```

Used by default if no memory is configured in RuntimeBuilder.

## Observer Trait

The `Observer` trait provides lifecycle hooks for logging, metrics, and custom logic.

```rust
#[async_trait]
pub trait Observer: Send + Sync + 'static {
    async fn on_start(&self) {}
    async fn on_shutdown(&self) {}
    async fn on_message_received(&self, _msg: &Message) {}
    async fn on_response_sent(&self, _channel: &str, _content: &str) {}
    async fn on_stream_event(&self, _event: &StreamEvent) {}
    async fn on_error(&self, _error: &MindroidError) {}
}
```

**Methods (all optional):**

| Method | Called When | Parameters |
|--------|-------------|------------|
| `on_start()` | Runtime starts | — |
| `on_shutdown()` | Runtime stops | — |
| `on_message_received()` | Message arrives from transport | `Message` reference |
| `on_response_sent()` | Response sent via transport | `channel_id`, `content` |
| `on_stream_event()` | Streaming event emitted | `StreamEvent` reference |
| `on_error()` | Any operation fails | `MindroidError` reference |

All methods have default empty implementations. Implement only the ones you need.

Example observer for logging:

```rust
pub struct LogObserver;

#[async_trait]
impl Observer for LogObserver {
    async fn on_message_received(&self, msg: &Message) {
        tracing::info!("Message: {} from {}", msg.id, msg.sender_id);
    }

    async fn on_response_sent(&self, channel: &str, content: &str) {
        tracing::info!("Sent to {}: {}", channel, content);
    }

    async fn on_error(&self, error: &MindroidError) {
        tracing::error!("Error: {}", error);
    }
}
```

Add multiple observers:

```rust
let runtime = Runtime::builder()
    .observer(LogObserver)
    .observer(MetricsObserver)
    .observer(AlertingObserver)
    .build()?;
```

### NoObserver

Built-in no-op observer. All hooks are empty.

```rust
pub struct NoObserver;

// All methods are no-ops
```

Used by default if no observer is configured.

## Pipeline System

The pipeline is an ordered sequence of processing stages that transform a message into a response.

### PipelineContext

The context passed through each stage, accumulating data.

**Fields:**

| Field | Type | Purpose | Set By |
|-------|------|---------|--------|
| `message` | `Message` | Incoming message | Runtime (immutable) |
| `agent_config` | `AgentConfig` | Agent settings | Runtime (immutable) |
| `llm_messages` | `Vec<LlmMessage>` | Conversation history | ContextBuilder stage |
| `model_type` | `String` | LLM variant (e.g., "chat", "completion") | Router stage |
| `model_ids` | `Vec<String>` | Ordered list of model IDs to try | Router stage |
| `compute_power` | `u8` | Resource level 0–100 | Router stage |
| `raw_response` | `Option<String>` | LLM output (unformatted) | Processor stage |
| `final_response` | `Option<String>` | Agent response (formatted) | PostProcessor stage |
| `extensions` | `HashMap<String, Value>` | Custom cross-stage data | Any stage |

Constructor:

```rust
let ctx = PipelineContext::new(message, agent_config);
// llm_messages, model_ids, etc. are empty
// compute_power defaults to 50
// extensions is empty
```

Typical pipeline flow:

```
ContextBuilder:
  ctx.llm_messages = build_history(...)

Router:
  ctx.model_type = "chat"
  ctx.model_ids = vec!["gpt-4", "gpt-3.5"]
  ctx.compute_power = 80

Processor:
  ctx.raw_response = call_llm(...)

PostProcessor:
  ctx.final_response = format_response(ctx.raw_response)
```

### PipelineStage Trait

A processing stage reads from and writes to `PipelineContext`.

```rust
#[async_trait]
pub trait PipelineStage: Send + Sync {
    fn name(&self) -> &str;
    async fn process(&self, ctx: &mut PipelineContext) -> Result<()>;
}
```

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `name()` | `fn(&self) -> &str` | Unique stage identifier for logging and errors |
| `process()` | `async fn(&self, ctx: &mut PipelineContext) -> Result<()>` | Transform the context; return error to halt pipeline |

Example stage:

```rust
pub struct DebugStage;

#[async_trait]
impl PipelineStage for DebugStage {
    fn name(&self) -> &str { "debug" }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        tracing::debug!("Processing: {}", ctx.message.content);
        Ok(())
    }
}
```

### StreamingStage Trait

Extends `PipelineStage` to emit events token-by-token.

```rust
#[async_trait]
pub trait StreamingStage: PipelineStage {
    fn stream<'a>(&'a self, ctx: &'a mut PipelineContext) -> BoxStream<'a, StreamEvent>;
}
```

**Key points:**

- Implementors must provide both `PipelineStage::process()` (non-streaming fallback) and `StreamingStage::stream()` (streaming).
- `process()` should collect the full response into `ctx.raw_response`.
- `stream()` yields `StreamEvent`s as they arrive.
- Only one streaming stage per pipeline.

Example:

```rust
pub struct OllamaStreamingStage;

#[async_trait]
impl PipelineStage for OllamaStreamingStage {
    fn name(&self) -> &str { "ollama-streaming" }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        // Non-streaming: collect full response
        let response = call_ollama_blocking(&ctx.llm_messages).await?;
        ctx.raw_response = Some(response);
        Ok(())
    }
}

#[async_trait]
impl StreamingStage for OllamaStreamingStage {
    fn stream<'a>(&'a self, ctx: &'a mut PipelineContext) -> BoxStream<'a, StreamEvent> {
        Box::pin(async_stream::stream! {
            let mut stream = call_ollama_streaming(&ctx.llm_messages).await;
            while let Some(chunk) = stream.next().await {
                yield StreamEvent::Chunk { content: chunk };
            }
            yield StreamEvent::Complete {
                content: String::new(),
                usage: None,
            };
        })
    }
}
```

### Pipeline Struct

Composes multiple stages into an executable pipeline.

```rust
pub struct Pipeline {
    stages: Vec<StageEntry>,
    streaming_idx: Option<usize>,
}

impl Pipeline {
    pub fn new() -> Self { /* ... */ }
    pub fn add_stage(mut self, stage: impl PipelineStage + 'static) -> Self { /* ... */ }
    pub fn add_streaming_stage(mut self, stage: impl StreamingStage + 'static) -> Self { /* ... */ }
    pub async fn run(&self, ctx: &mut PipelineContext) -> Result<String> { /* ... */ }
    pub fn run_streaming<'a>(&'a self, ctx: &'a mut PipelineContext) -> BoxStream<'a, StreamEvent> { /* ... */ }
}
```

**Methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `new()` | `fn() -> Self` | Create empty pipeline |
| `add_stage()` | Takes `impl PipelineStage`, returns self | Add a normal processing stage (fluent) |
| `add_streaming_stage()` | Takes `impl StreamingStage`, returns self | Add a streaming stage (fluent, max 1) |
| `run()` | Async, returns `Result<String>` | Execute non-streaming; returns final response |
| `run_streaming()` | Returns `BoxStream<StreamEvent>` | Execute with streaming; yields events |

Construction (fluent builder):

```rust
let pipeline = Pipeline::new()
    .add_stage(ContextBuilder)
    .add_stage(Router)
    .add_streaming_stage(OllamaProcessor)
    .add_stage(PostProcessor);
```

**Execution model:**

Non-streaming (`run()`):
```
Stage 1 → Stage 2 → ... → Stage N → Returns ctx.final_response or ctx.raw_response
```

Streaming (`run_streaming()`):
```
Pre-stages (1..k) run normally
         ↓
Streaming stage yields BoxStream<StreamEvent>
         ↓
Post-stages (k+1..N) run after stream completes
         ↓
Returns BoxStream<StreamEvent> with pre/stream/post events combined
```

**Key constraint:** At most one `StreamingStage` per pipeline. Stages before it run via `process()`, the streaming stage emits events, stages after it run on the completed context.

## Configuration

### MindroidConfig

Top-level configuration from TOML or environment.

```rust
pub struct MindroidConfig {
    pub agent: AgentConfig,
    pub transport: TransportConfig,
    pub pipeline: PipelineConfig,
    pub identity: IdentityConfig,
    pub memory: MemoryConfig,
    pub observer: ObserverConfig,
}
```

**Loading:**

```rust
// From file
let cfg = MindroidConfig::from_file("./mindroid.toml")?;

// From TOML string
let cfg = MindroidConfig::from_toml_str(toml_content)?;

// Resolve (tries: explicit path → ./mindroid.toml → ~/.mindroid/config.toml → defaults)
let cfg = MindroidConfig::resolve(Some("./my-config.toml"))?;
let cfg = MindroidConfig::resolve(None)?; // Auto-discover
```

**Resolution order:**
1. Explicit path (if provided)
2. `./mindroid.toml` (current directory)
3. `~/.mindroid/config.toml` (home directory)
4. Default (empty config)

Environment variables override config file values:

| Env Var | Applies To | Example |
|---------|-----------|---------|
| `MINDROID_API_KEY` | `identity.api_key` | `MINDROID_API_KEY=sk-...` |
| `MINDROID_EMAIL` | `identity.email` | `MINDROID_EMAIL=user@example.com` |
| `MINDROID_PASSWORD` | `identity.password` | `MINDROID_PASSWORD=secret` |
| `MINDROID_BASE_URL` | `pipeline.base_url`, `transport.url` | `MINDROID_BASE_URL=http://localhost:11434` |
| `MINDROID_AGENT_ID` | `agent.agent_id` | `MINDROID_AGENT_ID=my-agent` |

### AgentConfig

Agent identity and model settings.

**Fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `agent_id` | String | `""` | Unique agent identifier |
| `name` | String | `"Mindroid Agent"` | Human-readable name |
| `persona` | String | `""` | Personality/role description |
| `magickspace_id` | Option<String> | None | Optional workspace/organization ID |
| `model_type` | String | `"chat"` | LLM type (chat, completion, etc.) |
| `model_ids` | Vec<String> | `[]` | Ordered list of model IDs to try |
| `compute_power` | u8 | 50 | Resource level 0–100 |
| `metadata` | HashMap<String, Value> | `{}` | Custom data |

Example TOML:

```toml
[agent]
agent_id = "my-agent"
name = "Assistant"
persona = "You are helpful"
model_type = "chat"
model_ids = ["gpt-4", "gpt-3.5-turbo"]
compute_power = 80
```

### Other Config Sections

**TransportConfig** — Transport-specific settings:
```rust
pub struct TransportConfig {
    pub transport_type: Option<String>,  // "stdio", "centrifugo", etc.
    pub url: Option<String>,
    pub channels: Vec<String>,
    pub options: HashMap<String, Value>,
}
```

**PipelineConfig** — LLM/pipeline settings:
```rust
pub struct PipelineConfig {
    pub pipeline_type: Option<String>,   // "ollama", "magickmind", etc.
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub options: HashMap<String, Value>,
}
```

**IdentityConfig** — Auth settings:
```rust
pub struct IdentityConfig {
    pub identity_type: Option<String>,   // "static", "apikey", etc.
    pub email: Option<String>,
    pub password: Option<String>,
    pub api_key: Option<String>,
    pub token: Option<String>,
    pub base_url: Option<String>,
}
```

**MemoryConfig** — Persistence settings:
```rust
pub struct MemoryConfig {
    pub memory_type: Option<String>,    // "sqlite", "magickmind", etc.
    pub path: Option<String>,
    pub base_url: Option<String>,
    pub options: HashMap<String, Value>,
}
```

**ObserverConfig** — Logging/observability settings:
```rust
pub struct ObserverConfig {
    pub observer_type: Option<String>,  // "log", custom, etc.
    pub level: Option<String>,          // "debug", "info", "error"
    pub options: HashMap<String, Value>,
}
```

## Runtime

The runtime wires all components together and manages the main loop.

### MessageContext

Context available during message handling. Provides access to the pipeline, transport, memory, and observers.

**Fields (public):**
- `message: Message` — The incoming message
- `agent_config: Arc<AgentConfig>` — Agent configuration

**Methods:**

```rust
pub async fn process(&self) -> Result<String>
```
Run the pipeline on this message and return the result (non-streaming).

```rust
pub fn process_streaming(&self) -> BoxStream<'_, StreamEvent>
```
Run the pipeline with streaming and return a stream of events.

```rust
pub async fn respond(&self, content: &str) -> Result<Option<String>>
```
Send a response back through the transport. Triggers `on_response_sent()` observer hook and saves response to memory.

```rust
pub async fn process_and_respond(&self) -> Result<String>
```
Convenience: run pipeline then automatically send the response.

Example handler:

```rust
.on_message(|ctx| async {
    // Option 1: process and respond separately
    match ctx.process().await {
        Ok(content) => {
            let _ = ctx.respond(&content).await;
        }
        Err(e) => {
            tracing::error!("Pipeline failed: {e}");
        }
    }

    // Option 2: convenience method
    if let Err(e) = ctx.process_and_respond().await {
        tracing::error!("Handler failed: {e}");
    }

    // Option 3: streaming
    let mut stream = ctx.process_streaming();
    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::Chunk { content } => print!("{}", content),
            StreamEvent::Complete { .. } => println!(),
            _ => {}
        }
    }
})
```

### RuntimeBuilder

Fluent builder for constructing a runtime.

**Methods (all chainable):**

| Method | Takes | Returns Self | Purpose |
|--------|-------|--------------|---------|
| `config()` | `MindroidConfig` | ✓ | Set configuration |
| `transport()` | `impl Transport` | ✓ | Set transport (required) |
| `pipeline()` | `Pipeline` | ✓ | Set pipeline (required) |
| `identity()` | `impl Identity` | ✓ | Set identity (required) |
| `memory()` | `impl Memory` | ✓ | Set memory (optional; defaults to NoMemory) |
| `observer()` | `impl Observer` | ✓ | Add observer (repeatable) |
| `transport_sender()` | `TransportSender` | ✓ | Set response sender |
| `channel_buffer()` | `usize` | ✓ | Set mpsc channel size (default: 256) |
| `on_message()` | Closure `(MessageContext) -> impl Future` | ✓ | Set message handler |
| `build()` | — | `Result<Runtime>` | Build and validate |

**Required fields:**
- `transport` — Where messages come from/go to
- `pipeline` — How to process messages
- `identity` — How to authenticate
- `on_message` — Message handler (has sensible default: `process_and_respond()`)

**Optional fields:**
- `config` — Configuration (default: MindroidConfig::default())
- `memory` — Message storage (default: NoMemory)
- `observer` — Lifecycle hooks (default: none)
- `channel_buffer` — Channel size (default: 256)

Example:

```rust
let runtime = Runtime::builder()
    .config(MindroidConfig::resolve(None)?)
    .transport(StdioTransport)
    .pipeline(
        Pipeline::new()
            .add_stage(ContextBuilder)
            .add_stage(Router)
            .add_streaming_stage(OllamaProcessor)
            .add_stage(PostProcessor)
    )
    .identity(StaticIdentity::new("token"))
    .memory(SqliteMemory::new("./messages.db").await?)
    .observer(LogObserver)
    .on_message(|ctx| async {
        let _ = ctx.process_and_respond().await;
    })
    .build()?;

runtime.run().await?;
```

### Runtime

The running agent.

**Methods:**

```rust
pub fn builder() -> RuntimeBuilder
```
Create a builder.

```rust
pub async fn run(&mut self) -> Result<()>
```
Start the main loop. Blocks until transport disconnects or error. Calls:
1. `transport.connect()`
2. Observer `on_start()` hooks
3. `transport.listen()` to poll for messages
4. For each message: save to memory, notify observers, spawn handler task
5. Handler task calls `on_message` closure

```rust
pub async fn shutdown(&mut self) -> Result<()>
```
Gracefully disconnect. Calls:
1. Observer `on_shutdown()` hooks
2. `transport.disconnect()`

The runtime maintains state:
- `transport`: Connection to message source
- `pipeline`: Processing stages
- `identity`: Auth provider
- `memory`: Message storage
- `observers`: Lifecycle hooks
- `agent_config`: Agent settings

---

See [Architecture Overview](architecture.md) for design rationale and [Crate Reference](crates.md) for implementation crates.
