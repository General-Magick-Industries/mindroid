# Architecture Overview

Mindroid is a modular Rust SDK for building AI agents, designed around trait-driven composition. Every core subsystem—transport, pipeline, identity, memory, and observer—is defined as a trait, enabling you to swap implementations without touching the rest of the runtime.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Runtime                                  │
│  ┌──────────────┐  ┌──────────────────────────────────────────┐ │
│  │  Transport   │→ │     MessageContext                        │ │
│  │  (listen)    │  │  ┌────────────────────────────────────┐  │ │
│  └──────────────┘  │  │  Pipeline (Sequential Stages)      │  │ │
│                    │  │  ┌──────────┐                       │  │ │
│  ┌──────────────┐  │  │  │ Stage 1  │ → ┐                  │  │ │
│  │  Identity    │  │  │  └──────────┘   │                  │  │ │
│  │  (auth)      │  │  │  ┌──────────┐   ├→ Processing     │  │ │
│  └──────────────┘  │  │  │ Stage N  │ → ┘   (Sequential)  │  │ │
│                    │  │  └──────────┘                       │  │ │
│  ┌──────────────┐  │  │  ┌──────────────────────────────┐  │  │ │
│  │   Memory     │  │  │  │ Optional: Streaming Stage    │  │  │ │
│  │  (storage)   │  │  │  │ (token-by-token output)      │  │  │ │
│  └──────────────┘  │  │  └──────────────────────────────┘  │  │ │
│                    │  │  ┌────────────────────────────────┐  │  │
│  ┌──────────────┐  │  │  │ Post-Stages (after streaming) │  │  │
│  │   Observer   │  │  │  └────────────────────────────────┘  │  │
│  │  (hooks)     │  │  └────────────────────────────────────┘  │ │
│  └──────────────┘  └──────────────────────────────────────────┘ │
│                                                                  │
│  Agent Configuration & Metadata                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Design Philosophy

**Trait-driven composition**: Every subsystem (Transport, Pipeline, Identity, Memory, Observer) is defined as a trait. This allows you to:

- Implement custom versions for your use case
- Swap implementations without changing runtime code
- Test with mock implementations
- Compose multiple implementations (e.g., multiple observers)

**Async-first**: Built on tokio with async/await throughout. All I/O operations (transport, identity, memory) are non-blocking.

**Builder pattern**: RuntimeBuilder wires components together with a fluent API, validating required components at build time.

**Context passing**: Data flows through MessageContext and PipelineContext, accumulating state as it passes through pipeline stages.

## Core Traits Overview

| Trait | Purpose | Key Methods |
|-------|---------|-------------|
| Transport | Message I/O | connect, disconnect, listen, send, is_connected, health_check, set_health_reporter, reports_own_health |
| PipelineStage | Processing step | process(&mut ctx) |
| StreamingStage | Token-by-token output (extends PipelineStage) | stream(&mut ctx) → BoxStream<StreamEvent> |
| Identity | Authentication | get_token, get_auth_headers, is_authenticated, refresh |
| Memory | Message persistence | save_message, get_history, clear_history |
| Observer | Lifecycle hooks | on_start, on_shutdown, on_message_received, on_response_sent, on_stream_event, on_error |

## Message Lifecycle

1. **Transport receives incoming message** — Transport.listen() polls for messages from external source and sends them through an mpsc channel.

2. **Runtime creates MessageContext** — Runtime receives Message from channel, pairs it with agent config, pipeline, and other services into a MessageContext.

3. **Memory saves incoming message** — Runtime saves incoming message to memory (if configured) and notifies observers via on_message_received().

4. **Message handler spawned** — Runtime spawns a tokio task with the configured message handler (default: ctx.process_and_respond()).

5. **Pipeline processes message** — MessageContext.process() creates a PipelineContext from the Message and AgentConfig, then runs pipeline stages.

6. **Pipeline stages execute** — Each stage reads from and writes to PipelineContext. Common stages:
   - Router: sets model_type and model_ids
   - ContextBuilder: builds llm_messages conversation history
   - Processor: calls LLM and sets raw_response
   - PostProcessor: formats raw_response into final_response

7. **Streaming (optional)** — If pipeline has a StreamingStage:
   - Pre-stages run normally
   - StreamingStage.stream() emits BoxStream<StreamEvent>
   - Post-stages run after stream completes
   - Observer.on_stream_event() fires for each chunk

8. **Response sent** — MessageContext.respond() sends Response through Transport.send().

9. **Observer hooks** — Observer.on_response_sent() fired, response saved to memory.

## Pipeline Model

Pipeline is an ordered sequence of processing stages. Execution model:

```
Non-streaming execution (Pipeline::run):
  Stage 1 → Stage 2 → ... → Stage N
  Returns: String (ctx.final_response or ctx.raw_response)

Streaming execution (Pipeline::run_streaming):
  Pre-stages (Stage 1..k)
         ↓
  Streaming Stage (k+1) emits BoxStream<StreamEvent>
         ↓
  Post-stages (Stage k+2..N)
  Returns: BoxStream<'a, StreamEvent>
```

**Key constraint**: At most one StreamingStage per pipeline. Stages before it run via process(), the streaming stage emits events, stages after it run via process() on the completed context.

**PipelineContext** accumulates data:
- message: incoming Message
- agent_config: AgentConfig
- llm_messages: built by ContextBuilder
- model_type, model_ids, compute_power: set by Router
- raw_response: set by Processor (LLM output)
- final_response: set by PostProcessor (formatted output)
- extensions: HashMap for custom cross-stage data

## RuntimeBuilder Composition

RuntimeBuilder wires all components together with a fluent API:

```rust
let runtime = Runtime::builder()
    .config(mindroid_config)              // Optional: MindroidConfig
    .transport(stdio_transport)            // Required: impl Transport
    .pipeline(pipeline)                    // Required: Pipeline
    .identity(static_identity)             // Required: impl Identity
    .memory(sqlite_memory)                 // Optional: impl Memory (default: NoMemory)
    .observer(log_observer)                // Optional: impl Observer (repeatable)
    .on_message(|ctx| async {              // Optional: message handler
        let _ = ctx.process_and_respond().await;
    })
    .channel_buffer(256)                   // Optional: mpsc channel size (default: 256)
    .build()?;

// Start the runtime (blocks until shutdown)
runtime.run().await?;
```

**Required**: transport, pipeline, identity, on_message (has sensible default)
**Optional**: config, memory, observer, channel_buffer

## Workspace Crate Map

All 11 crates with dependency relationships (all depend on mindroid-core):

| Crate | Type | Purpose |
|-------|------|---------|
| mindroid-core | foundation | Traits, models, runtime, error types |
| mindroid-transport-stdio | transport | Stdin/stdout communication |
| mindroid-transport-centrifugo | transport | Centrifugo websocket messaging |
| mindroid-pipeline-ollama | pipeline | Ollama LLM integration |
| mindroid-pipeline-magickmind | pipeline | Magickmind LLM integration |
| mindroid-identity-static | identity | Static token authentication |
| mindroid-identity-apikey | identity | API key authentication |
| mindroid-memory-sqlite | memory | SQLite message history |
| mindroid-memory-magickmind | memory | MagickMind remote storage |
| mindroid-observer-log | observer | Logging observer |
| mindroid-examples | examples | Usage examples |

Each crate exports a single implementation (e.g., StdioTransport, OllamaStage). Mix and match them, or implement your own.

## Configuration

MindroidConfig (from TOML or env) has sections for each subsystem:

```rust
pub struct MindroidConfig {
    pub agent: AgentConfig,           // agent_id, name, persona, model_ids, etc.
    pub transport: TransportConfig,   // type, url, channels, options
    pub pipeline: PipelineConfig,     // type, base_url, api_key, model
    pub identity: IdentityConfig,     // type, email, password, api_key, token
    pub memory: MemoryConfig,         // type, path, base_url, options
    pub observer: ObserverConfig,     // type, level, options
}
```

Config resolution order: explicit path → ./mindroid.toml → ~/.mindroid/config.toml → defaults. Environment variables (MINDROID_API_KEY, MINDROID_AGENT_ID, etc.) override config file values.

## Error Handling

All trait methods return Result<T> where Err is MindroidError. The runtime:

- Logs errors via tracing
- Notifies observers via observer.on_error()
- Continues processing subsequent messages
- Graceful shutdown on transport disconnect

## Extension Points

Use PipelineContext.extensions (HashMap<String, serde_json::Value>) to pass custom data between stages without modifying trait signatures.

Example:
```rust
// Stage A writes to extensions
ctx.extensions.insert("custom_key".to_string(), json!({"data": "value"}));

// Stage B reads from extensions
if let Some(val) = ctx.extensions.get("custom_key") {
    // use val
}
```

---

See [Core API Reference](core.md) for detailed trait documentation, [Crate Reference](crates.md) for each implementation, and [Getting Started](guides/getting-started.md) for a working example.
