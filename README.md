# Mindroid

Modular AI agent SDK with trait-based component swappability.

Mindroid is a Rust framework for building AI agents that can reason, act, and communicate across multiple transports. Every subsystem -- transport, pipeline, auth, memory, and observer -- is defined as a trait, letting you swap implementations without touching the rest of the runtime.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          Runtime                             │
│                                                              │
│  Transport ──► mpsc channel ──► MessageContext                │
│  (listen)                       │                            │
│                                 ▼                            │
│  Auth      ─────────────────► Pipeline                       │
│  (identity)                     │ Stage 1: Context Builder   │
│                                 │ Stage 2: Gate (optional)   │
│  Memory    ─────────────────►   │ Stage 3: LLM Processor     │
│  (history)                      │ Stage 4: Tool Executor     │
│                                 │ Stage N: Post Processor    │
│  Observer  ─────────────────►   ▼                            │
│  (hooks)                      respond()                      │
│                                                              │
│  Routines  ─────────────────► Background tasks (timers)      │
└──────────────────────────────────────────────────────────────┘
```

## Features

- **Composable pipelines** -- Chain processing stages in any order. Add gates, tool execution, speech, or custom stages.
- **Any LLM backend** -- Works with Ollama, OpenAI, litellm, vLLM, OpenRouter, or any OpenAI-compatible endpoint.
- **Tool execution** -- Agents can run shell commands, open URLs, set reminders, or use custom tools you define.
- **Remote tools** -- Declare tools the *client* executes: the pipeline emits the call as its response instead of running it, and the client returns the result as a new message.
- **Artifact storage** -- Move images/audio out of conversation history after the model reads them, and re-fetch by id on demand instead of re-sending bytes every turn.
- **Skills system** -- On-demand domain knowledge with deterministic prefiltering and trust-based authority.
- **Multi-agent coordination** -- Gates and engagement tracking prevent feedback loops in multi-agent deployments.
- **Persona system** -- Structured personality traits with per-user dyadic adaptation, formatted in-process or server-prepared (`magickmind-prepared`) with per-message persona selection and prompt caching.
- **Streaming first** -- Token-by-token LLM output with transparent tool-call buffering.
- **Transport agnostic** -- stdio, Centrifugo WebSocket, or audio (microphone + speaker).

> **Security note:** transports and persona stages refuse to send credentials
> over plaintext connections. Existing configs pointing a Centrifugo transport
> at `ws://` with an auth token now fail at connect — switch to `wss://`, or
> set `transport.allow_insecure = true` for local development only. The same
> applies to `http://` persona base URLs (`persona.allow_insecure = true`).

## Quick Start

Add mindroid to your project:

```toml
[dependencies]
mindroid = { path = ".", features = ["llm-local"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

Minimal agent with Ollama:

```rust
use mindroid::prelude::*;
use mindroid::transport::stdio::StdioTransport;
use mindroid::pipeline::presets::ollama::ollama_pipeline;
use mindroid::auth::static_id::StaticAuth;

#[tokio::main]
async fn main() -> mindroid::Result<()> {
    let mut runtime = Runtime::builder()
        .transport(StdioTransport::new())
        .pipeline(ollama_pipeline("http://localhost:11434", "llama3.2")?)
        .auth(StaticAuth::new("dev"))
        .on_message(|ctx| async move {
            if let Err(e) = ctx.process_and_respond().await {
                tracing::error!("Error: {e}");
            }
        })
        .build()?;

    runtime.run().await
}
```

## Feature Flags

| Flag | Default | What it enables |
|------|---------|-----------------|
| `llm-local` | yes | OpenAI-compatible LLM client (Ollama, litellm, vLLM) |
| `llm-hosted` | no | Magick Mind hosted LLM pipeline |
| `transport-ws` | no | Centrifugo WebSocket transport |
| `transport-audio` | no | Microphone input + speaker output via cpal/rodio |
| `speech` | no | STT/TTS providers (OpenAI Whisper, Deepgram) |
| `apikey` | no | API key authentication |
| `persistence` | no | SQLite and Magick Mind memory backends |
| `persona` | no | Persona system (cloud and local providers) |
| `identity` | no | Cross-platform identity resolution |
| `artifacts` | no | Out-of-band media storage (offload + on-demand re-injection) |
| `magickmind` | no | Magick Mind service integration (end-user credentials, backend-routed tools) |
| `full` | no | All features |

## Configuration

Mindroid resolves configuration in order: CLI `--config` flag, `MINDROID_CONFIG` env, `./mindroid.toml`, `~/.mindroid/config.toml`, then defaults.

```toml
[agent]
agent_id = "my-agent"
name = "My Agent"
compute_power = 50

[transport]
type = "stdio"

[pipeline]
type = "ollama"
model = "llama3.2"

[auth]
type = "static"
token = "dev-token"

[memory]
type = "sqlite"
path = "./mindroid.db"

[observer]
type = "log"
level = "info"
```

Environment variable overrides: `MINDROID_API_KEY`, `MINDROID_EMAIL`, `MINDROID_PASSWORD`, `MINDROID_BASE_URL`, `MINDROID_AGENT_ID`.

## Documentation

- [Architecture Overview](docs/architecture.md) -- High-level design and trait system
- [Design: Modularity and Swappability](docs/design.md) -- Deep dive into trait-driven composition
- [Core API Reference](docs/core.md) -- Error types, models, configuration
- [Custom Pipelines](docs/guides/custom-pipeline.md) -- Building your own pipeline stages
- [Skills System](docs/guides/skills.md) -- On-demand domain knowledge
- [Tools System](docs/guides/tools.md) -- Agent action capabilities
- [Multi-Agent Coordination](docs/guides/coordination.md) -- Gates and engagement tracking
- [Persona System](docs/guides/persona.md) -- Structured personality and dyadic adaptation
- [Local Persona Setup](docs/guides/local-persona.md) -- File-based persona configuration
- [Identity Resolution](docs/guides/identity-resolution.md) -- Cross-platform user identity
- [Magick Mind Integration](docs/guides/magickmind-integration.md) -- Cloud pipeline and memory

## License

Licensed under either of [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) or [MIT License](http://opensource.org/licenses/MIT), at your option.
