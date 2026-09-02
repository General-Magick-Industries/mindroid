# Magick Mind Integration

Connect your Mindroid agent to the Magick Mind platform using Centrifugo, MagickMind, and Cortex.

## Architecture Overview

The Magick Mind integration connects three platform services that work together to deliver real-time messaging, conversation context, and LLM inference:

- **Centrifugo** — Real-time WebSocket messaging server. The agent connects via WebSocket, subscribes to a personal channel, and receives messages as Centrifugo push frames.
- **MagickMind** — REST API for conversation context and message persistence. Provides context retrieval (conversation history as LlmMessages) and message storage via authenticated POST/GET requests.
- **Cortex** — LLM inference API with Server-Sent Events (SSE) streaming. Accepts model configuration and messages, returns streaming responses with thinking, chunks, and completion events.

### Message Flow Diagram

```
User → Centrifugo (WebSocket) → Agent
                                  │
                          ┌───────┴───────┐
                          │   Pipeline     │
                          │                │
                          │ 1. Context  ←──── MagickMind (GET context)
                          │ 2. Router      │
                          │ 3. Cortex  ←──── Cortex (SSE stream)
                          │ 4. PostProc    │
                          │ 5. Persist ───→── MagickMind (POST message)
                          └────────────────┘
```

## Required Credentials and Environment Variables

Configure these environment variables or use mindroid.toml:

| Variable | Purpose |
|----------|---------|
| MINDROID_EMAIL | Email for API key authentication |
| MINDROID_PASSWORD | Password for API key authentication |
| MINDROID_BASE_URL | Base URL for MagickMind and Identity APIs |
| MINDROID_API_KEY | API key for Cortex LLM service |
| MINDROID_AGENT_ID | Agent identifier for channel subscription |

## Configuration File (mindroid.toml)

Place this file at `./mindroid.toml` or `~/.mindroid/config.toml`:

```toml
[agent]
agent_id = "agent-001"
name = "My Agent"
model_type = "chat"
model_ids = ["gpt-4o"]
compute_power = 50

[transport]
type = "centrifugo"
url = "wss://centrifugo.example.com/connection/websocket"
# The transport refuses to send the auth token over plaintext ws://.
# For local development against a non-TLS Centrifugo, opt in explicitly:
# allow_insecure = true

[pipeline]
type = "magickmind"
base_url = "https://api.magickmind.io"
api_key = "sk-..."

[identity]
type = "apikey"
email = "agent@example.com"
password = "secret"
base_url = "https://api.magickmind.io"

[memory]
type = "magickmind"
base_url = "https://api.magickmind.io"

[observer]
type = "log"
```

Key fields for Magick Mind:
- `transport.type = "centrifugo"` — Enables real-time messaging
- `pipeline.type = "magickmind"` — Activates the 5-stage pipeline
- `identity.type = "apikey"` — Enables JWT-based authentication
- `memory.type = "magickmind"` — Enables remote message persistence

There is no `agent.magickspace_id` config field: the magickspace ID travels
with each message as `channel_id`, extracted by the Centrifugo transport from
the push payload's `magickspace_id` field. Context retrieval and persistence
use it per-message.

## The 5-Stage Pipeline

The magickmind_pipeline() creates a sequential processing pipeline with these stages:

### Stage 1: ContextBuilder

Retrieves conversation context from MagickMind.

- Calls `MagickmindClient.prepare_context(magickspace_id, participant_id, query, config, exclude_sender)`
- The magickspace ID is the message's `channel_id`
- Makes POST request to `/v1/magickspaces/{magickspace_id}/context` with `{ participant_id, chat_history?, pelican?, corpus? }`
- Receives conversation history, knowledge, and documents, converted to `Vec<LlmMessage>`
- Sets `ctx.llm_messages` to the retrieved messages
- If the message has no `channel_id`, the provider is skipped and the pipeline falls back to system persona + user message

### Stage 2: Router

Copies agent configuration into pipeline context for downstream stages.

- Copies `ctx.agent_config.model_type` → `ctx.model_type`
- Copies `ctx.agent_config.model_ids` → `ctx.model_ids`
- Copies `ctx.agent_config.compute_power` → `ctx.compute_power`

This allows the pipeline to route requests with correct model and resource configuration.

### Stage 3: CortexProcessor (StreamingStage)

Sends the message to Cortex for LLM inference with optional SSE streaming.

Builds CortexRequest with:
- `messages` — from ctx.llm_messages
- `model_type` — from ctx.model_type (e.g., "chat")
- `model_ids` — from ctx.model_ids (e.g., ["gpt-4o"])
- `compute_power` — from ctx.compute_power (0-100)
- `magickspace_id` — optional, from the incoming message's `channel_id`
- `user_id` — from incoming message sender_id

Sends POST request to `/v1/cortex/stream` with `x-api-key` header.

**Non-streaming mode** (`process()`): Collects the full SSE response and sets `ctx.raw_response`.

**Streaming mode** (`stream()`): Emits StreamEvent variants from SSE events:
- `thinking` event → `StreamEvent::Thinking { content }`
- `chunk` event → `StreamEvent::Chunk { content }`
- `complete` event → `StreamEvent::Complete { content }`
- `error` event → `StreamEvent::Error { message }`
- `[DONE]` marker → stream termination

### Stage 4: PostProcessor

Formats the raw response into final response.

- Copies `ctx.raw_response` to `ctx.final_response`
- Trims leading and trailing whitespace
- Handles empty responses gracefully

### Stage 5: MagickmindPersistence

Saves the agent's response to MagickMind for conversation history.

- Calls `MagickmindClient.save_message(magickspace_id, sender_id, content, reply_to_message_id)`
- Makes POST request to `/v1/magickspaces/{magickspace_id}/messages`
- Stores response with `reply_to_message_id` linking to the incoming message ID
- Skips persistence when the message carries no `channel_id` (magickspace ID)

## MagickmindClient API

Create a client and fetch context or save messages:

```rust
let client = MagickmindClient::new(base_url, identity);

// Fetch conversation context (magickspace_id = the message's channel_id)
let prepared = client
    .prepare_context(
        magickspace_id,
        participant_id,
        query,
        &MagickmindContextConfig::default(),
        None, // exclude_sender
    )
    .await?;
let messages: Vec<LlmMessage> = prepared.messages;
// prepared.corpora lists the space's bound knowledge bases ({id, name,
// description}); pass them to the shipped `CorpusTool` by putting a
// `CorpusCatalog(prepared.corpora)` in the tool run scope for the turn. The
// catalog system block is rendered only when
// `MagickmindContextConfig::include_corpus_catalog` is set.

// Save a message
let msg_id: Option<String> = client
    .save_message(magickspace_id, sender_id, content, reply_to_message_id)
    .await?;
```

Both methods require authentication headers obtained from the Identity provider.

## CortexClient SSE Event Format

Cortex returns Server-Sent Events in this format:

```
event: thinking
data: {"event_type": "thinking", "content": "Let me think about this..."}

event: chunk
data: {"event_type": "chunk", "content": "Here is"}

event: chunk
data: {"event_type": "chunk", "content": " my response."}

event: complete
data: {"event_type": "complete", "content": "Here is my response.", "final_answer": "Here is my response."}

[DONE]
```

Event types:
- `thinking` — Internal model reasoning (not displayed to user)
- `chunk` — Streamed text content
- `complete` — Final answer with optional final_answer field
- `error` — Error message string

The client parses each event as JSON and emits the corresponding StreamEvent variant.

## Full Example Walkthrough

The `magickmind.rs` example demonstrates the complete integration:

```rust
use std::sync::Arc;

use mindroid_core::{MindroidConfig, Runtime};
use mindroid_identity_apikey::ApiKeyIdentity;
use mindroid_memory_magickmind::MagickmindMemory;
use mindroid_observer_log::LogObserver;
use mindroid_pipeline_magickmind::magickmind_pipeline;
use mindroid_transport_centrifugo::CentrifugoTransport;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,mindroid=debug")
        .init();

    // Load config from mindroid.toml or environment variables
    let config = MindroidConfig::resolve(None)?;

    // Extract configuration with defaults
    let base_url = config
        .pipeline
        .base_url
        .as_deref()
        .unwrap_or("http://localhost:8080");

    let api_key = config
        .identity
        .api_key
        .as_deref()
        .unwrap_or("");

    let email = config.identity.email.as_deref().unwrap_or("");
    let password = config.identity.password.as_deref().unwrap_or("");

    // Create identity provider (handles JWT token refresh automatically)
    let identity = Arc::new(ApiKeyIdentity::new(base_url, email, password));

    let ws_url = config
        .transport
        .url
        .as_deref()
        .unwrap_or("wss://localhost:8000/connection/websocket");

    let agent_id = &config.agent.agent_id;

    // Create transport (connects to Centrifugo WebSocket)
    let transport = CentrifugoTransport::new(ws_url, agent_id, identity.clone());

    // Create pipeline (5-stage magickmind pipeline)
    let pipeline = magickmind_pipeline(identity.clone(), base_url, api_key);

    // Create memory (MagickMind remote storage; the magickspace ID comes from
    // each message's channel_id)
    let memory = MagickmindMemory::new(base_url, identity.clone());

    // Wire everything together and start the runtime
    let mut runtime = Runtime::builder()
        .config(config)
        .transport(transport)
        .pipeline(pipeline)
        .identity(identity)
        .memory(memory)
        .observer(LogObserver::new())
        .on_message(|ctx| async move {
            if let Err(e) = ctx.process_and_respond().await {
                tracing::error!("Error processing message: {e}");
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
```

### Setup Walkthrough

1. **Config Resolution** — `MindroidConfig::resolve(None)` loads from mindroid.toml, env vars, or defaults
2. **Identity Creation** — `ApiKeyIdentity::new()` handles email/password login and JWT token refresh (10 seconds before expiry)
3. **Transport** — `CentrifugoTransport::new()` connects via WebSocket using JWT for authentication
4. **Pipeline** — `magickmind_pipeline()` creates the 5-stage pipeline with MagickMind and Cortex clients
5. **Memory** — `MagickmindMemory::new()` provides remote message storage independent of the pipeline
6. **Observer** — `LogObserver::new()` logs lifecycle events (connect, message, error, disconnect)
7. **Runtime** — Wires all components, spawns message handler tasks, and starts the event loop

## Troubleshooting

### Token Refresh Failures

ApiKeyIdentity automatically refreshes tokens 10 seconds before expiry. If you see Auth errors:

1. Verify `MINDROID_EMAIL` and `MINDROID_PASSWORD` are correct
2. Check that the login endpoint is reachable at `{base_url}/v1/auth/login`
3. Ensure the login credentials correspond to an active Magick Mind account

### WebSocket Reconnection Issues

CentrifugoTransport reconnects automatically with exponential backoff. If WebSocket fails:

1. Verify the Centrifugo server URL is reachable (check firewall, DNS, TLS cert)
2. Confirm `MINDROID_AGENT_ID` matches your agent configuration
3. Check that the agent has permission to connect to Centrifugo
4. Use `wss://` — the transport refuses to send the auth token over plaintext `ws://` unless `transport.allow_insecure = true` is set (local development only)

### Channel Subscription Errors

The personal channel format is `personal:{agent_id}#{service_user_id}`. If subscription fails:

1. Verify your JWT contains the correct `sub` (subject) claim
2. Check that `service_user_id` extracted from JWT matches your Magick Mind user ID
3. Ensure the agent_id in the channel name matches your configured `MINDROID_AGENT_ID`

### Cortex SSE Streaming Errors

If streaming fails or times out:

1. Verify `MINDROID_API_KEY` is valid (check API key expiration in Magick Mind console)
2. Ensure the Cortex endpoint is reachable at `{base_url}/v1/cortex/stream`
3. Confirm the API key is passed in the `x-api-key` header (not as Bearer token)
4. Check that the model_ids in agent config are available in your Magick Mind workspace

### MagickMind Context Retrieval Failures

If ContextBuilder fails to fetch context:

1. Verify incoming messages carry a `channel_id` (the transport extracts the magickspace ID from the Centrifugo push payload)
2. Verify the magickspace exists in your Magick Mind workspace
3. Check that the MagickMind endpoint is reachable at `{base_url}/v1/magickspaces/{magickspace_id}/context`
4. Confirm authentication headers are being sent (check logs for auth errors)

## Cross-references

- [Getting Started](getting-started.md) — Simpler setup with stdio transport and Ollama
- [Building Custom Pipelines](custom-pipeline.md) — Create your own pipeline stages
- [Core API — Pipeline System](../core.md#pipeline-system) — PipelineContext and Pipeline trait
- [Architecture Overview](../architecture.md) — System design and trait-driven composition
- [Crate Reference](../crates.md) — Implementation details for all crates
