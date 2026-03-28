# Building Custom Pipelines

Create custom processing stages to control how your agent handles messages.

## Implementing PipelineStage

The `PipelineStage` trait defines a single processing step in the pipeline. Implement it to create custom logic that reads from and writes to the pipeline context.

```rust
#[async_trait]
pub trait PipelineStage: Send + Sync {
    fn name(&self) -> &str;
    async fn process(&self, ctx: &mut PipelineContext) -> Result<()>;
}
```

### Example: Uppercase Echo Stage

Here's a simple stage that transforms the incoming message to uppercase:

```rust
use async_trait::async_trait;
use mindroid_core::prelude::*;

struct UppercaseEcho;

#[async_trait]
impl PipelineStage for UppercaseEcho {
    fn name(&self) -> &str {
        "UppercaseEcho"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        ctx.raw_response = Some(ctx.message.content.to_uppercase());
        Ok(())
    }
}
```

The stage reads `ctx.message.content` (the incoming message), transforms it, and stores the result in `ctx.raw_response`. Later stages can read and transform further, or the runtime uses `ctx.final_response` (or `ctx.raw_response` as fallback) as the final output.

## Implementing StreamingStage

For token-by-token output (e.g., streaming LLM responses), implement `StreamingStage`. This trait extends `PipelineStage`, so you must implement both `process()` (non-streaming fallback) and `stream()` (streaming implementation).

```rust
#[async_trait]
pub trait StreamingStage: PipelineStage {
    fn stream<'a>(&'a self, ctx: &'a mut PipelineContext) -> BoxStream<'a, StreamEvent>;
}
```

### Example: Word-by-Word Streaming

```rust
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use mindroid_core::prelude::*;

struct WordStreamer;

#[async_trait]
impl PipelineStage for WordStreamer {
    fn name(&self) -> &str {
        "WordStreamer"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        // Non-streaming fallback: collect full response
        ctx.raw_response = Some(ctx.message.content.to_uppercase());
        Ok(())
    }
}

#[async_trait]
impl StreamingStage for WordStreamer {
    fn stream<'a>(&'a self, ctx: &'a mut PipelineContext) -> BoxStream<'a, StreamEvent> {
        let content = ctx.message.content.clone();
        Box::pin(stream! {
            for word in content.split_whitespace() {
                yield StreamEvent::Chunk { content: format!("{word} ") };
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            yield StreamEvent::Complete {
                content: content.to_uppercase(),
                usage: None,
            };
        })
    }
}
```

**Key points:**

- The `process()` method is the non-streaming fallback. It runs when `Pipeline::run()` is called instead of `run_streaming()`.
- The `stream()` method yields `StreamEvent` variants: `Chunk { content }`, `Complete { content, usage }`, and `Error { message }`.
- `stream()` receives a mutable borrow of `ctx`, allowing you to read and write pipeline context fields.

## PipelineContext Field Reference

As your stages execute, they read and write fields in `PipelineContext`. Here's what each field is used for:

| Field | Type | Typical Usage |
|-------|------|---------------|
| message | Message | Read-only input. The incoming message (content, author, metadata). Set by the runtime. |
| agent_config | AgentConfig | Read-only. Agent persona, model settings, compute power defaults. |
| llm_messages | Vec<LlmMessage> | Context builders populate this with system/user/assistant messages for the LLM. |
| model_type | String | Router stages set this to classify the model (e.g., "chat", "completion"). |
| model_ids | Vec<String> | Router stages set this ordered list of model IDs to try. |
| compute_power | u8 | Router stages set this as a hint for model selection (0-100 scale). |
| raw_response | Option<String> | LLM processor stages write the raw model output here. |
| final_response | Option<String> | Post-processor stages write the cleaned/formatted output here. |
| extensions | HashMap<String, Value> | Free-form storage for custom data passed between stages (e.g., metrics, flags). |

## Composing Stages into a Pipeline

Build a pipeline by chaining stages in the order they should execute:

```rust
let pipeline = Pipeline::new()
    .add_stage(MyContextBuilder)
    .add_streaming_stage(MyLlmProcessor)
    .add_stage(MyPostProcessor);
```

### Execution Rules

- **Order matters**: Stages execute in the order they are added.
- **At most one streaming stage**: Adding a second `StreamingStage` panics. All other stages must be `PipelineStage`.
- **Non-streaming execution** (`Pipeline::run()`): All stages execute via their `process()` method sequentially. Returns the final response as a `String`.
- **Streaming execution** (`Pipeline::run_streaming()`):
  - Pre-stages (before the streaming stage) run via `process()`.
  - The streaming stage runs via `stream()`, yielding `StreamEvent`s.
  - Post-stages (after the streaming stage) run via `process()` after streaming completes.
  - Returns a `BoxStream<'a, StreamEvent>`.

## Full Custom Pipeline Example

Here's a complete example showing two stages working together: one transforms the input, and the next wraps the output.

```rust
use async_trait::async_trait;
use mindroid_core::{Pipeline, PipelineContext, PipelineStage, Result, Runtime};
use mindroid_identity_static::StaticIdentity;
use mindroid_transport_stdio::StdioTransport;

/// A stage that echoes the message content in uppercase.
struct UppercaseEcho;

#[async_trait]
impl PipelineStage for UppercaseEcho {
    fn name(&self) -> &str {
        "UppercaseEcho"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        ctx.raw_response = Some(ctx.message.content.to_uppercase());
        Ok(())
    }
}

/// A stage that wraps the response with a prefix and suffix.
struct Wrapper {
    prefix: String,
    suffix: String,
}

#[async_trait]
impl PipelineStage for Wrapper {
    fn name(&self) -> &str {
        "Wrapper"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let raw = ctx.raw_response.as_deref().unwrap_or("");
        ctx.final_response = Some(format!("{}{}{}", self.prefix, raw, self.suffix));
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pipeline = Pipeline::new()
        .add_stage(UppercaseEcho)
        .add_stage(Wrapper {
            prefix: ">>> ".into(),
            suffix: " <<<".into(),
        });

    let mut runtime = Runtime::builder()
        .transport(StdioTransport::new())
        .pipeline(pipeline)
        .identity(StaticIdentity::new("dev"))
        .on_message(|ctx| async move {
            if let Err(e) = ctx.process_and_respond().await {
                tracing::error!("Error: {e}");
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
```

**Message flow:**

- Input: "hello world"
- After `UppercaseEcho`: `raw_response = "HELLO WORLD"`
- After `Wrapper`: `final_response = ">>> HELLO WORLD <<<"`
- Output: ">>> HELLO WORLD <<<"

## Tips and Best Practices

### Passing Custom Data Between Stages

Use `ctx.extensions` (a `HashMap<String, serde_json::Value>`) to share data without modifying trait signatures:

```rust
// Stage A: Store custom metadata
ctx.extensions.insert("request_id".to_string(), json!("req-12345"));

// Stage B: Retrieve and use it
if let Some(val) = ctx.extensions.get("request_id") {
    let request_id = val.as_str().unwrap_or("unknown");
    // Log or use request_id
}
```

### Stage Design

- **Single responsibility**: Each stage should do one thing (route, build context, call LLM, post-process).
- **Clear stage names**: The `name()` method is used in error messages. Use descriptive names like "OllamaProcessor" or "JsonFormatter".
- **Error handling**: Return `Err(MindroidError::Pipeline { stage: self.name().into(), message, source })` to abort the pipeline with context.

### Accessing Agent Configuration

Stages can read agent settings from `ctx.agent_config`:

```rust
async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
    let agent_id = &ctx.agent_config.agent_id;
    let persona = &ctx.agent_config.persona;
    let model_ids = &ctx.agent_config.model_ids;
    // Use configuration to customize processing
    Ok(())
}
```

## See Also

- [Getting Started](getting-started.md) — Build your first agent
- [Core API — Pipeline System](../core.md#pipeline-system) — Full trait documentation
- [Magick Mind Integration](magickmind-integration.md) — Production pipeline example
