# Local Persona

Use local persona files for offline agents, privacy-sensitive deployments, and evolving personality definitions.

## When to Use Local Persona

**Local persona is ideal when:**

- Your agent runs offline or air-gapped (no remote API calls)
- You want version control over persona definitions (keep them in git)
- You're developing and iterating on personality traits
- Privacy regulations prevent sending persona data to external services
- You want to store dyadic (per-user) learned trait adaptations locally

**Use remote persona (`magickmind`) when:**

- Your team manages personas in a central platform
- You want instant updates without redeploying
- You need global trait learning across multiple agents
- You leverage a managed persona versioning system

## Directory Layout

Local personas live in a data directory with this structure:

```
{data_dir}/
├── {persona_id}/
│   ├── persona.md          # Static persona definition
│   └── dyadic/
│       ├── user1.json       # Per-user trait adaptations
│       ├── user2.json
│       └── ...
├── {other_persona_id}/
│   ├── persona.md
│   └── dyadic/
│       └── ...
```

**Example:**

```
./agents/
├── assistant/
│   ├── persona.md
│   └── dyadic/
│       ├── alice.json
│       ├── bob.json
│       └── charlie.json
├── expert/
│   ├── persona.md
│   └── dyadic/
│       └── ...
```

## Persona File Format

The `persona.md` file uses TOML frontmatter delimited by `+++`:

```toml
+++
name = "Assistant"
role = "helpful AI assistant"
tones = ["friendly", "professional", "patient"]

[traits]
helpfulness = { value = 0.9 }
humor = { value = 0.5, lock = "SOFT" }
formality = { value = 0.3, lock = "HARD" }
+++

You are a general-purpose assistant designed to help users with a wide range of tasks.
You provide clear, concise answers and ask clarifying questions when needed.

Your background includes training in multiple domains:
- Software engineering and DevOps
- Data science and analytics
- General knowledge and reference

Adapt your communication style to the user's needs and technical level.
```

## Trait Definitions

Each trait in the `[traits]` section has:

- **`value`** (float, required): Base trait value, typically 0.0–1.0 or -1.0–1.0
- **`lock`** (string, optional): Controls how dyadic learning can modify the trait
  - `"HARD"`: Authored value is final; dyadic overrides are ignored
  - `"SOFT"`: Dyadic value is used but clamped to ±0.3 of authored value
  - `none` (omitted): Dyadic value fully overrides authored value

Example traits:

```toml
[traits]
# Expert-level knowledge (immutable)
expertise = { value = 0.95, lock = "HARD" }

# Humor can adapt slightly per-user (±0.3 from 0.6)
humor = { value = 0.6, lock = "SOFT" }

# Formality can be fully adapted per-user
formality = { value = 0.5 }
```

## Dyadic Learned Traits

Per-user trait adaptations are stored as JSON files in the `dyadic/` subdirectory. Each file is named after the user ID (or canonical user ID from identity resolution) and contains learned trait overrides:

**File: `dyadic/alice.json`**

```json
{
  "user_id": "alice",
  "traits": {
    "humor": { "numeric_value": 0.8 },
    "formality": { "numeric_value": 0.2 },
    "engagement": { "numeric_value": 0.9 }
  },
  "updated_at": "2025-03-26T10:30:00Z"
}
```

**Trait Value Structure:**

Each value in the `traits` map is a `TraitValue` object with exactly one field set:

- `numeric_value`: A floating-point number
- `string_value`: A string value
- `string_list_value`: A list of strings

This flexible structure supports different trait types (numeric personality scales, descriptive text, lists of keywords, etc.).

## Blending Rules

When `PersonaContextBuilder` fetches the effective personality, it blends authored traits with dyadic overrides:

| Lock Level | Blending Behavior |
|---|---|
| `HARD` | Authored value wins; dyadic is ignored |
| `SOFT` | Dyadic value is clamped to ±0.3 of authored |
| `none` | Dyadic value fully overrides authored (if present) |

**Example:**

```
Authored traits:
  humor = 0.6 (lock: SOFT)
  formality = 0.3 (lock: none)
  expertise = 0.95 (lock: HARD)

User alice's dyadic overrides:
  humor = 0.8      → clamped to 0.9 (0.6 + 0.3)
  formality = 0.1  → used as-is
  expertise = 0.5  → ignored (HARD lock)

Effective personality for alice:
  humor = 0.9
  formality = 0.1
  expertise = 0.95
```

## Configuration

Enable local persona in your TOML config:

```toml
[agent]
agent_id = "my-agent"
name = "Assistant"
model_type = "fast"
model_ids = ["gpt-4o"]

[transport]
type = "stdio"

[pipeline]
type = "ollama"
model = "llama3.2"

[auth]
type = "static"
token = "dev-token"

[persona]
type = "local"
data_dir = "./agents"      # Path to data directory
persona_id = "assistant"   # Persona ID within data_dir

[memory]
type = "none"

[observer]
type = "log"
```

Resolve the config with:

```rust
let config = MindroidConfig::resolve_from_args()?;
```

Or pass `--config ./mindroid.toml` at runtime:

```sh
cargo run --example my_agent -- --config ./mindroid.toml
```

## Integration: PersonaProvider Trait

Both local and remote personas implement the `PersonaProvider` trait:

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

- `get_persona()`: Fetches static metadata (name, role, background story, tones)
- `get_effective_personality()`: Fetches trait values, blended with dyadic overrides for the given user

Load and use a local provider:

```rust
use mindroid::persona::LocalPersonaProvider;

// Load the persona at startup
let provider = LocalPersonaProvider::load("./agents", "assistant")?;

// Pass to PersonaContextBuilder
let builder = PersonaContextBuilder::new(Arc::new(provider), "assistant").await?;
```

Or let the runtime auto-construct it from config:

```rust
let config = MindroidConfig::resolve_from_args()?;
let mut runtime = Runtime::from_config(config)?
    .pipeline(my_pipeline)
    .build()?;
```

## Example: Using PersonaContextBuilder

The `PersonaContextBuilder` is a pipeline stage that:

1. Fetches the effective personality (with dyadic blending)
2. Builds a structured system prompt from traits and background story
3. Prepends this system prompt to the LLM message list

```rust
use mindroid::{PersonaContextBuilder, Pipeline, GenericLlmProcessor, PostProcessor};

// Build the persona stage (fetches schema at init)
let persona_provider = Arc::new(LocalPersonaProvider::load("./agents", "assistant")?);
let persona_stage = PersonaContextBuilder::new(persona_provider, "assistant").await?;

// Use it in a pipeline
let pipeline = Pipeline::new()
    .add_stage(persona_stage)
    .add_streaming_stage(GenericLlmProcessor::new(llm_client))
    .add_stage(PostProcessor);
```

The system prompt generated from the above persona file would look like:

```
You are Assistant, a helpful AI assistant.

You are a general-purpose assistant designed to help users with a wide range of tasks.
You provide clear, concise answers and ask clarifying questions when needed.

Your background includes training in multiple domains:
- Software engineering and DevOps
- Data science and analytics
- General knowledge and reference

Your personality traits:
- helpfulness: 0.9
- humor: 0.5 [lock: SOFT]
- formality: 0.3 [lock: HARD]

Communication tones: friendly, professional, patient
```

## Comparison: Local vs. Remote

| Feature | Local | Remote (magickmind) |
|---|---|---|
| File storage | YAML/TOML files | REST API |
| Persona updates | Redeploy agent | Live via API (no redeploy) |
| Dyadic learning | JSON files in `dyadic/` | Server-managed database |
| Version control | Git friendly | Central platform versioning |
| Network calls | None (offline) | Yes (trait fetches) |
| Privacy | Fully local | Data sent to API server |
| Use case | Development, air-gapped | Teams, managed platforms |

## Advanced: Custom Trait Learning

To update dyadic traits for a user, write or update the corresponding JSON file:

```rust
use mindroid::persona::local::DyadicLearnedTraits;
use std::collections::HashMap;

let dyadic = DyadicLearnedTraits {
    user_id: "alice".to_string(),
    traits: {
        let mut m = HashMap::new();
        m.insert(
            "humor".to_string(),
            TraitValue {
                numeric_value: Some(0.8),
                string_value: None,
                string_list_value: None,
            },
        );
        m
    },
    updated_at: chrono::Utc::now().to_rfc3339(),
};

// Write to disk
let path = Path::new("./agents/assistant/dyadic/alice.json");
let json = serde_json::to_string_pretty(&dyadic)?;
std::fs::write(path, json)?;
```

On the next message from alice, `PersonaContextBuilder` will automatically load and blend these overrides.
