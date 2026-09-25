# Mindroid Design: Modularity and Swappability

Mindroid is a Rust SDK for building AI agents. Its central design principle is **trait-driven composition**: every subsystem is defined as a Rust trait, and every concrete backend is an independent implementation of that trait. You swap components by changing which implementation you pass to the builder — nothing else changes.

---

## The Five Trait Subsystems

```
┌──────────────────────────────────────────────────────────────┐
│                          Runtime                             │
│                                                              │
│   Transport    →   mpsc channel   →   MessageContext         │
│   (listen)                            │                      │
│                                       ▼                      │
│   Identity     ──────────────────►  Pipeline                 │
│   (auth)                              │  Stage 1             │
│                                       │  Stage 2  (stages    │
│   Memory       ──────────────────►    │  Stage N   compose)  │
│   (history)                           │                      │
│                                       ▼                      │
│   Observer     ──────────────────►  respond()                │
│   (hooks)                                                    │
└──────────────────────────────────────────────────────────────┘
```

Each row in the diagram is an independent trait. You choose one implementation per subsystem (or write your own) and hand it to `RuntimeBuilder`. The runtime wires them together — it does not know or care which concrete types you chose.

### Transport — message I/O

```rust
#[async_trait]
pub trait Transport: Send + Sync + 'static {
    fn name(&self) -> &str;
    async fn connect(&mut self) -> Result<()>;
    async fn disconnect(&mut self) -> Result<()>;
    async fn listen(&self, tx: mpsc::Sender<Message>) -> Result<()>;
    async fn send(&self, response: &Response) -> Result<Option<String>>;
    fn is_connected(&self) -> bool;
    // optional with defaults:
    async fn send_typing(&self, _channel_id: &str) -> Result<()> { Ok(()) }
    async fn health_check(&self) -> bool { self.is_connected() }
}
```

`listen()` is a long-running task that pushes `Message`s into an mpsc channel. The runtime drains that channel and dispatches each message to the configured handler. `send()` is called to deliver responses back to the transport.

Built-in implementations:

| Feature flag | Type | What it does |
|---|---|---|
| `stdio` | `StdioTransport` | Reads lines from stdin, writes to stdout. Good for local dev. |
| `centrifugo` | `CentrifugoTransport` | WebSocket connection to a Centrifugo real-time server with JWT auth and auto-reconnect. |

Implement `Transport` yourself to integrate any message source: HTTP webhooks, MQTT, gRPC, Discord, Slack, etc.

---

### Identity — authentication

```rust
#[async_trait]
pub trait Identity: Send + Sync + 'static {
    async fn get_token(&self) -> Result<String>;
    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>>;
    fn is_authenticated(&self) -> bool;
    async fn refresh(&self) -> Result<()>;
}
```

Identity is called by pipeline stages and memory backends that make authenticated HTTP requests. A blanket impl over `Arc<T>` lets you share one identity across multiple components without cloning credentials.

Built-in implementations:

| Feature flag | Type | What it does |
|---|---|---|
| `static-id` | `StaticIdentity` | Hardcoded token. Zero dependencies. Use for local dev. |
| `apikey` | `ApiKeyIdentity` | Login with email/password, cache token, auto-refresh before expiry. Uses double-checked locking so concurrent refreshes don't cause thundering herd. |

---

### Memory — message persistence

```rust
#[async_trait]
pub trait Memory: Send + Sync + 'static {
    async fn save_message(&self, channel_id, sender_id, content, reply_to_id) -> Result<Option<String>>;
    async fn get_history(&self, channel_id: &str, limit: usize) -> Result<Vec<Message>>;
    async fn clear_history(&self, channel_id: &str) -> Result<()>;
}
```

The runtime calls `save_message` automatically for every incoming message and every outgoing response. Pipeline stages call `get_history` to build conversation context.

Built-in implementations:

| Feature flag | Type | What it does |
|---|---|---|
| *(default)* | `NoMemory` | No-op. Returns empty history, stores nothing. |
| `sqlite` | `SqliteMemory` | Local SQLite database via `rusqlite`. Uses `spawn_blocking` for async compatibility. |
| `magickmind` | `MagickmindMemory` | Remote REST API (`POST/GET/DELETE /v1/magickspaces/:id/messages`). Auth from `Identity`. |

---

### ArtifactStore — out-of-band media storage

Chat history should stay cheap. When a user attaches media (an image, audio, a
file), inlining the raw bytes into every turn bloats the context window and the
token bill. `ArtifactStore` solves this: the bytes are offloaded to a pluggable
store that returns an opaque **id**, and the conversation keeps only a compact
reference. The model re-fetches the bytes on demand via the `get_artifact` tool.

```rust
#[async_trait]
pub trait ArtifactStore: Send + Sync + 'static {
    async fn save(&self, scope, data, mime_type) -> Result<StoredArtifact>;
    async fn load(&self, scope: &str, id: &str) -> Result<Artifact>;
    async fn delete(&self, scope: &str, id: &str) -> Result<()>;
}
```

It is a pure store — `save` persists bytes and returns a reference, `load`/`delete`
retrieve and remove by id. `save` returns a [`StoredArtifact`](#storedartifact):
at minimum an id, optionally a metadata map the store chose to attach. A store that
doesn't enrich returns `StoredArtifact::new(id)` — the zero-cost path.

**Security:** `scope` MUST come from trusted session context (e.g.
`ctx.message.channel_id`), never from model- or user-supplied input. The store
enforces path containment but NOT tenant authorization — an attacker-chosen scope
could read another tenant's artifacts.

Built-in implementations:

| Feature flag | Type | What it does |
|---|---|---|
| *(default)* | `NoArtifactStore` | No-op. Both `save` and `load` error — a fabricated id would discard the bytes and never load. |
| `artifacts` | `LocalArtifactStore` | On-disk store, path-jailed under a base dir. Bytes + a JSON sidecar per `(scope, id)`. Caps an artifact at 64 MiB and its sidecar at 64 KiB, and on Windows refuses ids that name a device (`NUL`, `COM1`, …) — see [ADR-0006](adr/0006-artifact-path-jail.md). |
| `magickmind` | *(reserved)* | Remote backend; a user impl calling the artifact service. Placeholder today. |

#### StoredArtifact

The result of `save`: the reference `id` plus any enrichment the store attached.

```rust
pub struct StoredArtifact {
    pub id: String,
    pub metadata: ContentMetadata,     // caption, backend facts (etag/region), hashes, entities…
}
```

`ContentMetadata` is `serde_json::Map<String, Value>` — arbitrary, store-defined.
A store that wants to caption or tag on save fills it; a minimal store leaves it
empty (`StoredArtifact::new(id)`). Empty metadata is omitted from serialization, so a
minimal store costs nothing extra in storage or tokens. There is no separate
`description` field — a caption is just a metadata key (e.g. `metadata["caption"]`),
visible to the model like any other.

The `artifact_agent` example demonstrates this with a `DescribedLocalStore` — a
custom `ArtifactStore` that wraps `LocalArtifactStore` (the decorator pattern) and
returns `metadata: { "type": "locally saved artifact", "_directory": ... }` on save,
which then rides along on the persisted `File` reference.

**Metadata is visible to the model by default.** Artifact metadata is usually
descriptive (captions, tags, entities) and meant to inform the model, so when an
artifact reference is sent to the model its plain metadata keys are rendered into the
reference line alongside the id. To keep a key **code-only** (backend plumbing —
filesystem paths, S3 etags, internal ids), prefix its name with an underscore:
`_directory`, `_etag`. Underscore-prefixed keys are never rendered. The escape-hatch
is opt-*out*, because for artifacts exposure is the norm and hiding is the exception.

#### The offload / rehydrate flow

Three pieces cooperate, all built from one shared `Arc<dyn ArtifactStore>` (mirroring
how `Identity` is shared) so the offload stage and the load tool operate on the
same store:

1. **`ArtifactManager`** — orchestration over the store. `offload()` walks a
   message's parts, saves each inline media part, and replaces it with a
   `ContentPart::File { source: Uri { id }, metadata }` reference, stamping in
   whatever the store returned.
2. **`ArtifactOffload`** (pipeline stage) — runs `offload()` on the turn. Place it
   *early* (before the LLM, so the model never sees inline bytes) or *late* (after
   the LLM, so only history keeps the reference). Decoupled from persistence — it
   has no memory knowledge; `MemoryPersistence` later saves whatever the parts
   became.
3. **`GetArtifactTool`** (`get_artifact`) — the model calls it with an id;
   `XmlToolExecutorStage` resolves the bytes via the store's `load` and re-injects
   them as a multimodal `Role::Tool` message the model can see. A round
   re-attaches at most 8 artifacts, deduplicated — every one is held in memory
   and base64-expanded into the request — and the message names any left out.

Because a reference is just a `ContentPart::File` carrying an opaque id, swapping
the storage backend (local → remote → S3 → encrypted) changes nothing downstream —
`get_artifact`, the LLM wire conversion, and history round-tripping all treat the
id as an opaque token.

**Not** an artifact concern: "derive text from media instead of storing it" (OCR an
image to text, transcribe audio) is ordinary pipeline composition — a stage that
rewrites the `ContentPart` in place (e.g. `*part = ContentPart::text(...)`),
composed *instead of* `ArtifactOffload`. No new trait; `ArtifactStore` stays a pure
store. (`ContentPart` is the multi-modal message-part enum — `Text`, `Image`,
`Audio`, `Video`, `File` — where the media variants each carry a `metadata`
`ContentMetadata` map.)

---

### Observer — lifecycle hooks

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

All methods have default no-op implementations — implement only what you need. Unlike the other traits, the runtime accepts **multiple observers** and calls them all for each event. Add one for logging, one for metrics, one for alerting.

Built-in implementations:

| Feature flag | Type | What it does |
|---|---|---|
| *(default)* | `NoObserver` | No-op. |
| `log-observer` | `LogObserver` | Structured `tracing` logs at appropriate levels for each event. |

---

### Pipeline and PipelineStage — processing logic

The pipeline is where messages are transformed into responses. It is the most composable subsystem.

```rust
#[async_trait]
pub trait PipelineStage: Send + Sync {
    fn name(&self) -> &str;
    async fn process(&self, ctx: &mut PipelineContext) -> Result<()>;
}
```

A `PipelineStage` reads from and writes to `PipelineContext`. Stages compose sequentially: each stage sees the changes made by all previous stages. Any stage can set `ctx.halted = true` to abort the pipeline cleanly without an error — useful for gates that decide the agent should not respond.

`StreamingStage` extends `PipelineStage` to emit token-by-token output:

```rust
pub trait StreamingStage: PipelineStage {
    fn stream<'a>(&'a self, ctx: &'a mut PipelineContext) -> BoxStream<'a, StreamEvent>;
}
```

Implementors must provide both `process()` (collects full response into `ctx.raw_response`) and `stream()` (yields `StreamEvent`s incrementally). At most one streaming stage per pipeline. Stages before it run normally; stages after it run on the collected result.

---

## PipelineContext: the data bus

```rust
pub struct PipelineContext {
    pub message: Arc<Message>,       // immutable: incoming message
    pub agent_config: Arc<AgentConfig>, // immutable: agent settings
    pub context: Vec<LlmMessage>,    // pre-fetched context (from ContextPreparer)
    pub llm_messages: Vec<LlmMessage>, // built by SimpleContextBuilder
    pub model_type: String,          // set by Router
    pub model_ids: Vec<String>,      // set by Router
    pub compute_power: u8,           // set by Router
    pub raw_response: Option<String>,   // set by Processor stage
    pub final_response: Option<String>, // set by PostProcessor stage
    pub halted: bool,                // set by any gate to stop the pipeline
    pub extensions: HashMap<String, Value>, // escape hatch for custom stage data
}
```

`extensions` is the cross-stage escape hatch. Any stage can insert arbitrary JSON values under a key; later stages can read them. Use it to pass custom data between stages without changing the `PipelineContext` struct or the trait signatures.

`reset_output()` clears `llm_messages`, `raw_response`, `final_response`, and `halted` while preserving `context`, `model_ids`, and `extensions`. This lets you run multiple pipelines on the same context (e.g., classify then respond) without re-fetching context each time.

---

## Built-in pipeline stages

These are the building blocks you compose to build pipelines:

| Stage | Role | What it does |
|---|---|---|
| `SimpleContextBuilder` | Context builder | Assembles `llm_messages` from `ctx.context` + agent persona + incoming message |
| `Router` | Model selector | Copies `model_type`, `model_ids`, `compute_power` from agent config. Falls back to simple prompt if `llm_messages` is empty. |
| `PostProcessor` | Output cleaner | Trims `raw_response` and writes it to `final_response`. Usually the last stage. |
| `GenericLlmProcessor` | LLM caller | Calls any OpenAI-compatible endpoint via `LlmClient`. Backend-agnostic. |
| `OllamaProcessor` | Ollama LLM | `GenericLlmProcessor` preset for Ollama. Takes `base_url` and `model` directly. |
| `CortexProcessor` | Cortex LLM | `GenericLlmProcessor` preset for Magick Mind Cortex. Passes compute power via header. |
| `MagickmindPersistence` | Response saver | Saves `final_response` to MagickMind after the LLM stage completes. |
| `CoordinationGate` | Multi-agent gate (fast) | Halts pipeline if the agent already responded and not enough new messages have arrived. |
| `RelevanceGate` | Multi-agent gate (LLM) | Halts pipeline if a classifier LLM judges the message outside the agent's domain. |

---

## Pipeline presets

Two preset constructors wire the common stage combinations for you:

```rust
// Local inference: SimpleContextBuilder → OllamaProcessor (streaming) → PostProcessor
pub fn ollama_pipeline(base_url: &str, model: &str) -> Pipeline

// Magick Mind platform: Router → CortexProcessor (streaming) → PostProcessor → MagickmindPersistence
pub fn magickmind_pipeline(identity, base_url, api_key, compute_power) -> Pipeline

// Magick Mind + local Ollama: Router → GenericLlmProcessor (streaming) → PostProcessor → MagickmindPersistence
pub fn magickmind_ollama_pipeline(identity, magickmind_url, ollama_url, model) -> Pipeline
```

These are just functions that return a `Pipeline`. Nothing stops you from copying them and modifying the stage list.

---

## LlmClient: the OpenAI-compatible HTTP layer

`LlmClient` (feature flag: `llm-client`, enabled by `ollama` and `magickmind`) wraps `async-openai` to provide a unified client that works with any OpenAI-compatible endpoint.

```rust
pub struct LlmClientConfig {
    pub base_url: String,                      // e.g. "http://localhost:11434/v1"
    pub api_key: Option<String>,
    pub default_model: Option<String>,
    pub default_temperature: Option<f32>,
    pub default_max_tokens: Option<u32>,
    pub auth_style: AuthStyle,                 // Bearer, XApiKey, or None
    pub custom_headers: HashMap<String, String>,
}

pub struct LlmClient { /* ... */ }

impl LlmClient {
    pub async fn chat(&self, req: ChatRequest<'_>) -> Result<(String, Option<TokenUsage>)>
    pub fn stream_chat(&self, req: ChatRequest<'_>) -> BoxStream<'static, StreamEvent>
}
```

`AuthStyle` covers the three auth patterns in practice: `Bearer` (OpenAI, OpenRouter), `XApiKey` (custom services), `None` (local Ollama). Custom headers like `X-Compute-Power` attach to every request automatically.

---

## ContextProvider and ContextPreparer

For multi-pipeline workflows, context fetching can be separated from pipeline execution:

```rust
#[async_trait]
pub trait ContextProvider: Send + Sync {
    fn name(&self) -> &str;
    async fn fetch(&self, message: &Message) -> Result<Vec<LlmMessage>>;
}

pub struct ContextPreparer { /* providers: Vec<Box<dyn ContextProvider>> */ }

impl ContextPreparer {
    pub fn add_provider(self, provider: impl ContextProvider + 'static) -> Self
    pub async fn prepare(&self, message: &Message) -> Result<Vec<LlmMessage>>
    //                                         ^--- runs all providers in parallel
}
```

`ContextPreparer::prepare()` runs all registered providers in parallel with `join_all`, merges results, and returns the combined `Vec<LlmMessage>`. You store this in `pctx.context` once and reuse it across multiple `reset_output()` + pipeline runs.

Built-in implementation: `MagickmindContext` calls `POST /v1/magickspaces/:id/context` to retrieve chat history, episodic memory (Pelican), and corpus documents. Role mapping converts the agent's own prior messages to `assistant` role so the LLM correctly understands conversation flow.

Implement `ContextProvider` yourself for: vector databases, SQL history, static system prompts, or any other context source.

---

## Persona — multi-dimensional personality adaptation

Personas define what your agent is: name, role, background story, and personality traits. Unlike a simple string persona, Mindroid treats personas as composable, versionable data structures with per-user adaptation.

```rust
#[async_trait]
pub trait PersonaProvider: Send + Sync {
    fn name(&self) -> &str;
    async fn get_persona(&self, persona_id: &str) -> Result<PersonaSchema>;
    async fn get_effective_personality(
        &self,
        persona_id: &str,
        user_id: Option<&str>,
    ) -> Result<EffectivePersonalityResponse>;
}
```

`PersonaProvider` abstracts the data source. Two implementations are included:

**LocalPersonaProvider** — Files and JSON

Loads persona definitions from the filesystem:

```
{data_dir}/{persona_id}/persona.md         # TOML frontmatter + markdown body
{data_dir}/{persona_id}/dyadic/{user_id}.json  # Per-user trait overrides
```

Use for offline agents, version-controlled personas, and evolving local definitions:

```rust
let provider = LocalPersonaProvider::load("./agents", "assistant")?;
```

See [Local Persona guide](./guides/local-persona.md) for file format and trait locking.

**MagickmindPersonaClient** — Remote REST API

Fetches personas from a managed platform via HTTP:

```rust
let identity = Arc::new(ApiKeyIdentity::new(base_url, email, password));
let provider = MagickmindPersonaClient::new(api_url, identity);
```

Calls:
- `GET /v1/persona/{persona_id}` — fetch static persona schema
- `GET /v1/runtime/effective-personality/{persona_id}?user_id={user_id}` — fetch blended personality

Use for centrally-managed personas with instant updates across agents.

Both implementations blend authored traits with optional per-user dyadic overrides. Lock levels control how much dyadic learning can adapt each trait:

- `"HARD"`: Authored value only
- `"SOFT"`: Dyadic value clamped to ±0.3 of authored
- `none`: Dyadic fully overrides authored

### PersonaContextBuilder — system prompt composition

`PersonaContextBuilder` is a `PipelineStage` that fetches the effective personality and builds a structured system prompt:

```rust
pub struct PersonaContextBuilder {
    provider: Arc<dyn PersonaProvider>,
    persona_id: String,
    history: Arc<Vec<LlmMessage>>,
    // ... cache, persona schema ...
}
```

In `process()`, it:

1. Determines user_id for dyadic blending (prefers canonical ID from identity resolution)
2. Fetches effective personality from provider (with caching)
3. Builds a system prompt from persona info + effective traits
4. Prepends system prompt + history to LLM messages

```rust
let persona = PersonaContextBuilder::new(provider, "assistant").await?;
let pipeline = Pipeline::new()
    .add_stage(persona)
    .add_streaming_stage(GenericLlmProcessor::new(llm_client))
    .add_stage(PostProcessor);
```

Replace `SimpleContextBuilder` with `PersonaContextBuilder` when personas are configured.

---

## Identity resolution — canonical user IDs across platforms

When users contact your agent via multiple platforms (Telegram, Centrifugo, web socket), each platform assigns a different ID. Identity resolution maps all these platform IDs to a single **canonical user ID**, enabling consistent per-user personalization and context tracking.

```rust
pub struct IdentityResolver {
    registry: RwLock<IdentityRegistry>,
    index: RwLock<HashMap<(String, String), String>>,
    registry_path: PathBuf,
}

impl IdentityResolver {
    pub async fn resolve(&self, platform: &str, platform_id: &str) -> String
    pub async fn link(&self, canonical_id: &str, platform: &str, platform_id: &str) -> Result<()>
}
```

**Workflow:**

1. Transport sets `message.platform` (e.g., "telegram", "centrifugo")
2. `IdentityResolutionStage` intercepts the message
3. Calls `resolver.resolve(platform, platform_id)`
4. Auto-creates canonical ID on first contact, persists to disk
5. Stores result as `CanonicalUserId` extension in `PipelineContext`
6. Later stages (like `PersonaContextBuilder`) read this extension for dyadic adaptation

Registry is stored as JSON:

```json
{
  "users": {
    "alice": {
      "canonical_id": "alice",
      "identities": [
        { "platform": "telegram", "platform_id": "789", "linked_at": "..." },
        { "platform": "centrifugo", "platform_id": "user#alice", "linked_at": "..." }
      ]
    }
  }
}
```

### IdentityResolutionStage

```rust
pub struct IdentityResolutionStage {
    resolver: Arc<IdentityResolver>,
}

#[async_trait]
impl PipelineStage for IdentityResolutionStage {
    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let platform = ctx.message.platform.as_deref().unwrap_or("unknown");
        let canonical = self.resolver.resolve(platform, &ctx.message.sender_id).await;
        ctx.set_ext(CanonicalUserId(canonical));
        Ok(())
    }
}
```

Place early in the pipeline, before stages that need user context:

```rust
let resolver = Arc::new(IdentityResolver::load("./identity.json")?);
let pipeline = Pipeline::new()
    .add_stage(IdentityResolutionStage::new(resolver))
    .add_stage(PersonaContextBuilder::new(provider, "assistant").await?)
    // ...
```

With canonical IDs, alice's interactions via telegram, centrifugo, and web all map to a single identity. Dyadic trait learning, conversation history, and personalization are then consistent across all channels.

See [Identity Resolution guide](./guides/identity-resolution.md) for configuration and multi-channel setup.

---

## Multi-agent coordination

When multiple agents share a channel, naive implementations loop — agent A responds, agent B responds to A, A responds to B, forever. Mindroid provides two gates to prevent this:

### CoordinationGate (deterministic, no LLM)

Scans `ctx.context` for `assistant`-role messages (the agent's own prior responses). If the agent already responded and fewer than `min_new_messages` user messages have appeared since, it sets `ctx.halted = true`. Fast — pure role counting, no network call.

```rust
Pipeline::new()
    .add_stage(CoordinationGate::new(2)) // require 2+ new messages before re-engaging
    .add_stage(RelevanceGate::new(...))
    // ...
```

### RelevanceGate (LLM-based topic classification)

Sends a structured JSON prompt to a cheap, fast model (typically a small local Ollama model). The schema enforces `{"relevant": true}` or `{"relevant": false}` — no ambiguity from free-text classification. Halts the pipeline when `relevant` is false.

```rust
let gate = RelevanceGate::new(
    "budget planning — budgets, expenses, savings, financial goals",
    "http://localhost:11434",
    "smallthinker",
)
.instructions("Always engage when the word 'cost' appears.")
.strict(false); // let message through on LLM failure (default)
```

`strict(false)` means the gate fails open — if the classifier call fails (network error, timeout), the agent still responds. `strict(true)` fails closed, blocking the message.

The two gates are designed to layer: `CoordinationGate` runs first (cheap, deterministic), `RelevanceGate` second (LLM call, only when the coordination check passes).

### EngagementTracker

For simpler multi-agent scenarios, `EngagementTracker` tracks which sender the agent is actively engaged with per channel. Messages from a different sender during an active engagement are silently dropped until the cooldown expires.

```rust
let tracker = Arc::new(EngagementTracker::new(Duration::from_secs(120)));

builder.on_message(move |ctx| {
    let tracker = Arc::clone(&tracker);
    async move {
        if !tracker.should_engage(&ctx.message.channel_id, &ctx.message.sender_id).await {
            return;
        }
        ctx.process_and_respond().await.ok();
        tracker.record(&ctx.message.channel_id, &ctx.message.sender_id).await;
    }
});
```

---

## The Runtime and RuntimeBuilder

`RuntimeBuilder` is the assembly point. It validates required components at `.build()` time and wires everything together:

```rust
let mut runtime = Runtime::builder()
    .transport(StdioTransport::new())          // required
    .pipeline(pipeline)                        // required (defaults to empty)
    .identity(StaticIdentity::new("dev"))      // required
    .memory(SqliteMemory::new("./agent.db")?)  // optional, default: NoMemory
    .observer(LogObserver::new())              // optional, repeatable
    .channel_buffer(256)                       // optional, default: 256
    .on_message(|ctx| async move {             // optional, default: process_and_respond()
        ctx.process_and_respond().await.ok();
    })
    .build()?;

runtime.run().await?;
```

`Runtime::from_config()` auto-constructs identity, transport, memory, and observer from a `MindroidConfig`, returning a builder you can complete with `.pipeline()` and `.on_message()`:

```rust
let config = MindroidConfig::resolve_from_args()?;
let mut runtime = Runtime::from_config(config)?
    .pipeline(my_pipeline)
    .on_message(|ctx| async move { ctx.process_and_respond().await.ok(); })
    .build()?;
```

### MessageContext

`MessageContext` is what the `on_message` closure receives. It provides:

```rust
ctx.process().await                  // → Result<Option<String>>  (non-streaming)
ctx.process_streaming()              // → BoxStream<StreamEvent>  (streaming)
ctx.respond("content").await         // → send through transport + fire observer hooks
ctx.process_and_respond().await      // → process() then respond() in one call

// Run a different pipeline on the same message:
ctx.run_pipeline(&other_pipeline).await
ctx.run_pipeline_streaming(&other_pipeline)

// Reuse a PipelineContext across multiple pipelines:
ctx.run_with_context(&pipeline, &mut pctx).await
ctx.run_streaming_with_context(&pipeline, &mut pctx)
```

The last two methods let you pre-fetch context once (via `ContextPreparer`), store it in a `PipelineContext`, call `reset_output()` between runs, and execute multiple pipelines without redundant API calls.

---

## Configuration system

`MindroidConfig` is the TOML-loadable configuration that mirrors the five trait subsystems plus two new sections for LLM provider management:

```toml
[agent]
agent_id = "my-agent"
name = "Assistant"
persona = "You are a helpful AI assistant."
model_type = "fast"
model_ids = ["gpt-4o"]
compute_power = 80

[transport]
type = "stdio"   # or "centrifugo"

[pipeline]
type = "ollama"  # or "magickmind"
model = "llama3.2"

[identity]
type = "static"  # or "apikey"
token = "dev-token"

[memory]
type = "none"    # or "sqlite" or "magickmind"

[observer]
type = "log"     # or "none"

# Named LLM providers — define once, reference from multiple models
[providers.cortex]
base_url = "https://api.cortex.example.com"
api_key = "sk-..."
auth_style = "bearer"  # or "x-api-key" or "none"

[providers.local]
base_url = "http://localhost:11434"
auth_style = "none"

# Per-call LLM configs that inherit from a provider
[models.main]
provider = "cortex"
model = "gpt-4o"
compute_power = 80

[models.gate]
provider = "local"
model = "smallthinker"
```

Resolve config with:

```rust
MindroidConfig::resolve_from_args()?  // --config <path> → MINDROID_CONFIG env → ./mindroid.toml → ~/.mindroid/config.toml → defaults
MindroidConfig::resolve(Some("./my.toml"))?
MindroidConfig::from_file("./my.toml")?
```

Environment variables (`MINDROID_API_KEY`, `MINDROID_EMAIL`, `MINDROID_PASSWORD`, `MINDROID_BASE_URL`, `MINDROID_AGENT_ID`) override TOML values after parsing.

`config.llm("gate")` resolves a named model entry into a ready-to-use `LlmClientConfig` by merging model overrides onto provider defaults:

```rust
let gate_config = config.llm("gate")?;
let gate = RelevanceGate::from_config("budget planning", gate_config);
```

---

## Feature flags

All implementations are behind feature flags. The `full` feature enables everything. For minimal binaries, opt in only to what you need:

| Flag | Enables |
|---|---|
| `stdio` | `StdioTransport` |
| `centrifugo` | `CentrifugoTransport` |
| `ollama` | `ollama_pipeline`, `OllamaProcessor` (implies `llm-client`) |
| `magickmind` | `magickmind_pipeline`, `MagickmindClient`, `CortexProcessor` (implies `llm-client`) |
| `llm-client` | `LlmClient`, `GenericLlmProcessor`, `RelevanceGate`, `CoordinationGate`, `collect_stream` |
| `static-id` | `StaticIdentity` |
| `apikey` | `ApiKeyIdentity` |
| `sqlite` | `SqliteMemory` |
| `magickmind` | `MagickmindMemory` |
| `artifacts` | `LocalArtifactStore`, `ArtifactManager`, `ArtifactOffload`, `GetArtifactTool` |
| `log-observer` | `LogObserver` |
| `full` | All of the above |

This keeps the binary footprint small for edge or embedded deployments and prevents pulling in `reqwest`, `rusqlite`, or `tokio-tungstenite` when they are not needed.

---

## Composing a custom agent

Everything above composes freely. Here is a multi-agent setup with two-layer gating and context pre-fetching:

```rust
// Shared components
let identity = Arc::new(ApiKeyIdentity::new(base_url, email, password));
let magickmind = Arc::new(MagickmindClient::new(base_url, identity.clone()));

// Context: fetch chat history + knowledge from MagickMind, run providers in parallel
let preparer = ContextPreparer::new()
    .add_provider(MagickmindContext::new(magickmind.clone()).with_self_id(&agent_id));

// Gate: deterministic check first, then LLM relevance check
let gate_config = config.llm("gate")?;
let pipeline = Pipeline::new()
    .add_stage(CoordinationGate::new(2))
    .add_stage(RelevanceGate::from_config("budget planning", gate_config))
    .add_stage(SimpleContextBuilder)
    .add_streaming_stage(CortexProcessor::from_config(config.llm("main")?))
    .add_stage(PostProcessor)
    .add_stage(MagickmindPersistence::new(magickmind.clone()));

let tracker = Arc::new(EngagementTracker::new(Duration::from_secs(120)));

let mut runtime = Runtime::from_config(config)?
    .pipeline(pipeline)
    .on_message(move |ctx| {
        let preparer = preparer.clone();
        let tracker = tracker.clone();
        async move {
            if !tracker.should_engage(&ctx.message.channel_id, &ctx.message.sender_id).await {
                return;
            }

            // Fetch context once, reuse across multiple pipeline runs
            let context = preparer.prepare(&ctx.message).await.unwrap_or_default();
            let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());
            pctx.context = context;

            if let Ok(Some(_)) = ctx.run_with_context(&ctx_pipeline, &mut pctx).await {
                tracker.record(&ctx.message.channel_id, &ctx.message.sender_id).await;
            }
        }
    })
    .build()?;

runtime.run().await?;
```

Each piece — transport, identity, memory, observer, context providers, pipeline stages, gates — is independently replaceable. Add a new transport by implementing `Transport`. Add a new LLM backend by implementing `PipelineStage` (or `StreamingStage`). Add new context sources by implementing `ContextProvider`. None of these changes touch the runtime or any other component.
