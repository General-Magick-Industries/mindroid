# Identity Resolution

Resolve cross-platform user identities to enable per-user personalization and context tracking across multiple communication channels.

## The Problem

Users can interact with your agent through multiple platforms with different ID schemes:

- Centrifugo: `user#centrifugo-user-123`
- Telegram: `tg#user-456`
- Web socket: `ws#user-789`
- Microphone input: `mic#speaker-001`

Without identity resolution, each platform ID is treated as a separate user. This breaks:

- Per-user trait adaptation (dyadic learning)
- Conversation context tracking across channels
- User engagement history
- Personalization based on interaction patterns

Identity resolution maps all of these to a **single canonical user ID**, enabling coherent personalization and context.

## How It Works

The `IdentityResolver` maintains a registry:

```
(platform, platform_id) → canonical_user_id
```

### First Contact: Auto-Creation

When a user contacts your agent from a platform for the first time:

1. The `IdentityResolutionStage` intercepts the message
2. It checks the registry for `(platform, platform_id)`
3. If not found, it auto-creates a canonical ID (UUID prefix)
4. It persists this mapping to disk

```rust
let resolver = IdentityResolver::load("./identity.json")?;

// First message from Telegram user tg#456
let canonical_id = resolver.resolve("telegram", "456").await;
// → auto-creates canonical ID "a1b2c3d4" if new
// → persists mapping to disk
```

### Registry File

The identity registry is stored as JSON at the path you provide:

**File: `identity.json`**

```json
{
  "users": {
    "a1b2c3d4": {
      "canonical_id": "a1b2c3d4",
      "display_name": null,
      "identities": [
        {
          "platform": "telegram",
          "platform_id": "456",
          "linked_at": "2025-03-26T10:00:00Z"
        }
      ]
    },
    "alice": {
      "canonical_id": "alice",
      "display_name": "Alice Johnson",
      "identities": [
        {
          "platform": "telegram",
          "platform_id": "789",
          "linked_at": "2025-03-26T09:00:00Z"
        },
        {
          "platform": "centrifugo",
          "platform_id": "user#alice",
          "linked_at": "2025-03-26T09:15:00Z"
        }
      ]
    }
  }
}
```

## Configuration

Define identity resolution in your config:

```toml
[identity]
registry_path = "./identity.json"

# Pre-configured identity links (optional)
# Links are loaded at runtime without requiring the user to contact all platforms
[identity.links]
alice = ["telegram:789", "centrifugo:user#alice", "web:alice@example.com"]
bob = ["telegram:456", "centrifugo:user#bob"]
```

### Loading Config-Driven Links

At startup, pass pre-configured links to the resolver:

```rust
let mut resolver = IdentityResolver::load("./identity.json")?;

let links = {
    let mut m = std::collections::HashMap::new();
    m.insert(
        "alice".to_string(),
        vec![
            "telegram:789".to_string(),
            "centrifugo:user#alice".to_string(),
        ],
    );
    m
};

resolver.load_config_links(&links);
```

Or let the runtime load from config automatically:

```rust
let config = MindroidConfig::resolve_from_args()?;
// Config contains [identity.links] table
let mut runtime = Runtime::from_config(config)?
    .build()?;
```

## Pipeline Integration

Use `IdentityResolutionStage` to resolve identities during message processing:

```rust
use mindroid::identity::IdentityResolutionStage;
use std::sync::Arc;

let resolver = Arc::new(IdentityResolver::load("./identity.json")?);

let pipeline = Pipeline::new()
    .add_stage(IdentityResolutionStage::new(resolver))
    // ... other stages
    .add_stage(PersonaContextBuilder::new(persona_provider, "assistant").await?)
    // ... LLM, output stages
```

The stage:

1. Checks if the message is from a user (not a system message)
2. Extracts `platform` and `platform_id` from the message
3. Calls `resolver.resolve(platform, platform_id).await`
4. Stores the result as `CanonicalUserId` extension in `PipelineContext`

Later stages (like `PersonaContextBuilder`) read this extension for dyadic adaptation.

## Accessing the Canonical ID

Any stage can retrieve the resolved canonical ID:

```rust
#[async_trait]
impl PipelineStage for MyStage {
    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        // Retrieve canonical user ID if available
        if let Some(canonical) = ctx.get_ext::<CanonicalUserId>() {
            let user_id = &canonical.0;
            println!("Processing for canonical user: {}", user_id);
            // Use for per-user personalization, logging, tracking, etc.
        }
        Ok(())
    }
}
```

## Transport Platform Field

Transports must set the `platform` field on outgoing `Message` objects:

```rust
pub struct Message {
    pub sender_id: String,
    pub platform: Option<String>,  // "telegram", "centrifugo", "web", etc.
    pub content: String,
    // ... other fields
}
```

**Built-in transports:**

- `StdioTransport`: Sets `platform = "stdio"`
- `CentrifugoTransport`: Sets `platform = "centrifugo"`

**Custom transport:**

```rust
#[async_trait]
impl Transport for MyCustomTransport {
    async fn listen(&self, tx: mpsc::Sender<Message>) -> Result<()> {
        // ... receive messages ...
        let msg = Message {
            sender_id: "user#123".to_string(),
            platform: Some("my-custom-platform".to_string()),
            content: "Hello".to_string(),
            // ... other fields
        };
        tx.send(msg).await.ok();
    }
}
```

## PersonaContextBuilder Integration

`PersonaContextBuilder` uses the canonical ID for dyadic trait blending:

```rust
let user_id = if ctx.message.sender_type == SenderType::User {
    // Prefer canonical ID from identity resolution
    ctx.get_ext::<CanonicalUserId>()
        .map(|c| c.0.clone())
        .as_deref()
        .or(Some(&ctx.message.sender_id))
} else {
    None
};

// Fetch effective personality for this user
let effective = provider.get_effective_personality(&persona_id, user_id).await?;
```

With canonical IDs, all interactions from alice—whether via telegram, centrifugo, or web—use the same user ID for dyadic adaptation. This ensures personality evolution is consistent across channels.

## Example: Multi-Channel Agent

Here's a complete setup:

**Config: `mindroid.toml`**

```toml
[agent]
agent_id = "multi-channel-bot"
name = "Assistant"

[transport]
type = "centrifugo"
url = "wss://centrifugo.example.com/connection/websocket"
channels = ["personal:agent-001#*"]

[pipeline]
type = "magickmind"
base_url = "https://api.magickmind.io"

[auth]
type = "apikey"
email = "bot@example.com"
password = "secret"

[memory]
type = "magickmind"

[persona]
type = "local"
data_dir = "./agents"
persona_id = "assistant"

[identity]
registry_path = "./identity.json"

[identity.links]
alice = ["telegram:789", "centrifugo:user#alice", "web:alice@example.com"]
```

**Code:**

```rust
use mindroid::{MindroidConfig, Runtime, Pipeline, IdentityResolutionStage, PersonaContextBuilder};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let config = MindroidConfig::resolve_from_args()?;

    // Build runtime from config (auto-creates identity resolver)
    let mut runtime = Runtime::from_config(config)?;

    // Get the resolver from runtime and load config links
    let resolver = runtime.identity_resolver();
    if let Some(mut resolver) = resolver.write().await.take() {
        resolver.load_config_links(&config.identity.links);
    }

    // Persona provider
    let persona_provider = Arc::new(
        LocalPersonaProvider::load("./agents", "assistant")?
    );

    // Build pipeline with identity + persona
    let pipeline = Pipeline::new()
        .add_stage(IdentityResolutionStage::new(Arc::clone(&resolver)))
        .add_stage(PersonaContextBuilder::new(persona_provider, "assistant").await?)
        .add_streaming_stage(/* LLM processor */)
        .add_stage(/* PostProcessor */);

    runtime
        .pipeline(pipeline)
        .on_message(|ctx| async move {
            ctx.process_and_respond().await.ok();
        })
        .build()?
        .run()
        .await?;

    Ok(())
}
```

## Advanced: Linking Identities at Runtime

To link a new platform identity to an existing canonical user:

```rust
let resolver = IdentityResolver::load("./identity.json")?;

// Link a new platform identity to alice
resolver.link("alice", "telegram", "new_phone_number").await?;

// Now both telegram:789 and telegram:new_phone_number map to alice
let id1 = resolver.resolve("telegram", "789").await;      // → "alice"
let id2 = resolver.resolve("telegram", "new_phone_number").await; // → "alice"
```

This enables:

- Adding email aliases to existing users
- Merging duplicate identities
- Supporting device migration (old phone → new phone)
- Multi-platform onboarding flows
