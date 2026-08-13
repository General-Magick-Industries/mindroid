# Mindroid

Rust AI agent SDK. Trait-based modularity — every subsystem is swappable.

> This is the **source of truth** for agent + contributor guidance. `CLAUDE.md` imports it (`@AGENTS.md`); every other agent tool reads this file directly. Edit here, not there.

## Design invariants (read the ADR before changing these)

These are load-bearing. If a change touches one, read the linked ADR in `docs/adr/` first and **do not contradict an accepted ADR** — supersede it with a new one instead.

- **Functional core, imperative shell.** Compose transformational logic from `PipelineStage` combinators; use imperative `select!`/actor loops only where concurrent `&mut` ownership requires it. → `docs/adr/0000-principles.md`
- **Concurrency = structured.** Prefer `JoinSet` / `select!` / `CancellationToken` over detached `tokio::spawn`. Fan-out collects results; it never shares `&mut Context` across tasks. → `docs/adr/0001-concurrency.md`
- **Observability = middleware, never a mutable observer registry.** Cross-cutting concerns wrap traits (tower-style) or ride the `PipelineEvent` / callback stream. → `docs/adr/0002-observability.md`
- **OmniSession is a separate execution model**, not an extended `Pipeline`. → `docs/adr/0003-omnisession.md`
- **Accept traits, return structs.** Every subsystem is a swappable trait; keep them small and object-safe.

See `docs/adr/README.md` for the full index. ADRs hold the *why* + rejected alternatives; this file holds the *operational* rules.

## Build & Test Commands

```bash
cargo build --all-features          # Build (needs libasound2-dev on Linux)
cargo test --all-features --lib     # Unit tests only
cargo clippy --all-features --all-targets -- -D warnings  # Lint (zero warnings policy)
cargo fmt --all -- --check          # Format check
```

**Workspace examples** build separately — each is its own crate under `examples/`. Run one with `cargo run -p <package> --bin <bin> -- [args]` (not `cargo run --example`; `autoexamples = false` disables that form).

## Rust Edition & Toolchain

- Edition: **2024** (Cargo.toml `edition = "2024"`)
- Resolver: **3** (workspace-level)
- Release profile: `opt-level = "z"`, thin LTO, `panic = "abort"` — this crate's own
  binaries/examples only. Cargo ignores a dependency's profile, so an embedding host
  picks its own (see [Embedding many runtimes](#embedding-many-runtimes-in-one-process))

## Feature Flags

Default: `llm-local` only. Use `--all-features` for full build/test.

| Flag | Pulls in | Key types unlocked |
|------|----------|--------------------|
| `llm-client` | `async-openai`, `reqwest` | `GenericLlmProcessor`, `ToolExecutorStage` |
| `llm-local` | (includes `llm-client`) | `ollama_pipeline` preset |
| `llm-hosted` | (includes `llm-client`) | `magickmind_pipeline` preset |
| `transport-ws` | `tokio-tungstenite` | `CentrifugoTransport` |
| `transport-audio` | `cpal`, `hound`, `rodio`, VAD | `AudioTransport`, `AudioOutputStage` |
| `speech` | `reqwest` | `OpenAiStt`, `DeepgramTts`, etc. |
| `apikey` | `reqwest` | `ApiKeyAuth` |
| `persistence` | `reqwest`, `rusqlite` | `SqliteMemory`, `MagickmindMemory` |
| `persona` | `reqwest` | `PersonaContextBuilder`, `MagickmindPersonaStage`, `PersonaId`, `ConversationHistory`, `LocalPersonaProvider` |
| `identity` | (none) | `IdentityResolver`, `IdentityResolutionStage` |
| `artifacts` | `base64` (+ `llm-client`) | `ArtifactStore`, `LocalArtifactStore`, `ArtifactOffload`, `GetArtifactTool` |
| `magickmind` | (includes `artifacts`, `persona`) | `EndUserAuth`, `EpisodicMemoryTool`, `AgentCredentials`, `auth.type = "enduser"` |
| `full` | everything above | All types |

Backend-specific code lives behind `magickmind`, not `persona` — enabling the
persona system must not compile in a token client for one service.

## Architecture

```
Runtime (core/runtime.rs)
├── Transport  → mpsc → MessageContext → Pipeline → respond()
├── Auth       → token/headers for authed subsystems
├── Memory     → save/get/clear history
├── Observer   → lifecycle hooks (on_start, on_message, on_error, …)
├── Routines   → background poll/act loops (reminders, etc.)
└── Pipeline   → ordered stages, at most ONE StreamingStage
```

For real-time bidirectional audio, `OmniSession` runs alongside `Pipeline` as a separate model (see ADR-0003).

### Core Traits (always available, no feature gate)

| Trait | File | Key methods |
|-------|------|-------------|
| `Transport` | `transport/mod.rs` | `connect`, `listen`, `send` |
| `Auth` | `auth/mod.rs` | `get_token`, `get_auth_headers`, `refresh` |
| `Memory` | `memory/mod.rs` | `save_message`, `get_history`, `clear_history` |
| `Observer` | `observer/mod.rs` | `on_start`, `on_message_received`, `on_error`, … |
| `PipelineStage` | `pipeline/mod.rs` | `name`, `process` |
| `StreamingStage` | `pipeline/mod.rs` | `stream` (extends `PipelineStage`) |
| `Tool` | `tools/mod.rs` | `name`, `description`, `parameters_schema`, `execute` |

### Pipeline Stages Order

Stages execute sequentially. Typical order:
1. **ContextBuilder** — builds `llm_messages` from memory + system prompt
2. **Gate** (optional) — halts pipeline if message irrelevant (`ctx.halted = true`)
3. **LLM Processor** (streaming) — calls LLM, streams tokens
4. **ToolExecutor** — parses XML tool calls, executes, loops back to LLM
5. **PostProcessor** — transforms final response

Set `ctx.halted = true` to stop the pipeline early from any stage.

### Combinators

`pipeline/combinators.rs` provides `BranchStage` (gate → pass/fail), `RouterStage` (N-way), `RetryStage` (backoff), `ApprovalStage<T>` (HITL via `watch_session`). These are the functional composition layer — reach for them before writing bespoke control flow. Parallel fan-out (`ParallelStage`) is value-returning, not shared-`&mut` (ADR-0001).

### Extension Map

`PipelineContext` has a typed extension map (`set_ext<T>`, `get_ext<T>`, `take_ext<T>`) for passing data between stages without coupling them.

## Key Patterns

### RuntimeBuilder Resolution Order

**code > config > built-in default**. Subsystems set in builder code always win over `mindroid.toml`.

```rust
// Config as fallback, code overrides pipeline
let config = MindroidConfig::from_file("./mindroid.toml")?;
Runtime::builder().config(config).pipeline(my_pipeline).build()?
```

Use `Runtime::from_config(config)?` when you need `auth_arc()` before `build()` (e.g., to share auth with a `MagickmindClient`).

### Adding a New Tool

Implement `Tool` trait → register in `ToolRegistry` → `ToolExecutorStage` picks it up automatically. Schema is JSON Schema for arguments.

```rust
async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<String>
```

`ctx` carries the **trusted** per-message `channel_id` / `sender_id` plus a typed
extension map for backend data (credentials, agent id) set by a stage. Never take
identity from `args` — that's model-generated. Use `_ctx` if unused. → ADR-0005

### Adding a New Pipeline Stage

Implement `PipelineStage` (or `StreamingStage` for streaming). Use `pipeline.add_stage()` / `pipeline.add_streaming_stage()`. Only one streaming stage per pipeline.

### Config Resolution

`MindroidConfig::from_file` searches: CLI `--config` → `MINDROID_CONFIG` env → `./mindroid.toml` → `~/.mindroid/config.toml` → defaults.

### Error Handling

Use `MindroidError` variants (`Auth`, `Transport`, `Pipeline`, `Memory`, `Api`, `Config`, `Other`). Return `mindroid::Result<T>`. `MindroidError::config("msg")` for config errors.

### Embedding many runtimes in one process

A control plane can run N agents as N tokio tasks in one process: build a
`MindroidConfig` per agent, `Runtime::from_config_with_auth(cfg, auth)`, then
`tokio::spawn(rt.run_until_cancelled(token))`. The SDK supports this — no global
singletons, no library-level `tracing_subscriber` init, `Runtime` is `Send`.

Four things the SDK does **not** do for you; the host owns them:

1. **Build each config in code.** Never `MindroidConfig::resolve*` per agent — those
   read CLI args and env, which are process-wide and assume one agent per process.
   `merge_toml` layers a per-agent delta onto a shared base.
2. **Per-agent credentials.** `from_config_with_auth` injects an `Auth` per runtime;
   config-derived auth would be shared. Same reasoning as (1).
3. **Per-agent tracing spans.** One subscriber interleaves every agent's logs.
4. **`panic = "unwind"` + `catch_unwind` per task.** With `abort`, one agent's panic
   takes the whole host down. Cargo ignores this crate's profile when it's a
   dependency, so this is the host's choice to make.

`run_until_cancelled(token)` is the stop path — it disconnects the transport and
cancels routines, unlike `handle.abort()`, which drops mid-await. `Runtime::health()`
reports `Starting`/`Ready`/`Reconnecting`/`Stopped` so a supervisor can tell a working
agent from one that is alive but cannot receive.

## Gotchas

- **One streaming stage max** — `Pipeline` panics at runtime if you add two `StreamingStage`s
- **`#[async_trait]` everywhere** — all core traits use `async_trait` crate, not native async traits
- **Feature gating is pervasive** — check `#[cfg(feature = "...")]` before touching impl modules; code compiles with default features only
- **`pv_cobra` yanked** — Cobra VAD infra exists but the real crate is unavailable on crates.io; see `Cargo.toml` comment
- **macOS builds work out of the box** — Linux needs `libasound2-dev` for `transport-audio`
- **CI runs `--lib` tests only** — no integration or doc tests in CI (`cargo test --all-features --lib`), on ubuntu and windows. The artifact store's symlink tests skip themselves where the OS denies symlink creation, so a green local run does not mean they ran; set `MINDROID_REQUIRE_SYMLINKS=1` to turn that skip into a failure, as the Windows CI job does
- **`NoMemory` / `NoObserver`** are the no-op defaults — don't create new empty impls

## Anti-Patterns

- **NEVER** return responses outside the pipeline — all output flows through `Transport::send` via `MessageContext::process_and_respond()`
- **NEVER** add stages that assume a specific prior stage's output — use the extension map for inter-stage data
- **NEVER** skip `--all-features` in CI/local test — default features only cover `llm-local`; most code is feature-gated
- **NEVER** add a mutable observer registry for cross-cutting concerns — use middleware or the event stream (ADR-0002)

## Source Layout

```
src/
├── core/           # Runtime, builder, config, error, models
├── auth/           # Auth trait + static_id, apikey impls
├── transport/      # Transport trait + stdio, centrifugo, audio
├── pipeline/
│   ├── stages/     # ContextBuilder, Gate, LlmProcessor, ToolExecutor, PostProcessor, STT/TTS
│   ├── presets/    # ollama, magickmind ready-made pipelines
│   ├── combinators.rs  # Branch/Router/Retry/Approval (+ Parallel/Fusion)
│   ├── context.rs  # ContextPreparer, ContextProvider
│   └── coordination.rs  # EngagementTracker (multi-agent)
├── omni/           # OmniSession, OmniProvider, audio source/sink, VAD (ADR-0003)
├── artifacts/      # ArtifactStore trait + local, manager (ADR-0004)
├── ingest/         # Source/Encoder/MediaEncoder, Base64Source, ResolvedSource
├── memory/         # Memory trait + sqlite, magickmind impls
├── observer/       # Observer trait + log impl
├── tools/          # Tool trait + shell, open, reminder, get_artifact, remote
├── skills/         # SkillRegistry, manifest parsing, prefiltering
├── persona/        # PersonaProvider, cache, local/cloud providers
├── identity/       # IdentityResolver, cross-platform resolution
├── llm_client.rs   # Shared OpenAI-compatible client
├── prelude.rs      # Convenience re-exports
└── lib.rs          # Crate root, public API surface
```

## Documentation layout

- `AGENTS.md` (this file) — always-on operational rules + invariants. Keep terse; link, don't inline.
- `docs/adr/` — architecturally-significant decisions (the *why* + rejected alternatives). Read on-demand.
- `docs/` — architecture, guides, design reference.

**Scope:** ADRs here cover the **SDK's own architecture** only. Decisions about Magick Mind service integration (billing, cross-service wiring) live in the private monorepo, not this public repo.

## Git

Gita repo (own `.git`). PRs go to `General-Magick-Industries/mindroid` on GitHub, branch `main`.
