# Persona System

The Persona system gives agents a structured, adaptable personality. Instead of a static system prompt, you define traits (warmth, directness, humor) with numeric values and let the runtime blend them into natural language -- optionally adapting per user.

## Why It Exists

Static system prompts have three problems:

1. **No adaptation** -- Every user gets the same personality, regardless of their interaction style.
2. **Scattered definitions** -- Updating personality means editing strings buried in code.
3. **No structured evolution** -- You cannot constrain, measure, or evolve specific traits independently.

The Persona system separates the *definition* of personality (structured data) from its *expression* (generated system prompt), with a blending layer that incorporates per-user learned adjustments.

---

## Core Models

### PersonaSchema

The static definition of a persona, fetched once at startup:

```rust
pub struct PersonaSchema {
    pub id: String,
    pub name: String,           // "Aria"
    pub role: String,           // "wellness coach"
    pub traits: Vec<String>,    // trait names
    pub tones: Vec<String>,     // ["empathetic", "calm"]
    pub background_story: String,
}
```

### EffectivePersonalityResponse

The blended snapshot returned per-request, scoped to a user:

```rust
pub struct EffectivePersonalityResponse {
    pub persona_id: String,
    pub user_id: Option<String>,
    pub traits: Vec<EffectiveTrait>,
    pub computed_at: String,    // RFC 3339
    pub ttl_seconds: u64,       // 0 = do not cache
}
```

Each `EffectiveTrait` carries the blended value plus its provenance (authored, globally learned, dyadically learned) and lock constraints.

---

## PersonaProvider Trait

All persona sources implement:

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

    fn is_prepared(&self) -> bool { false }
    async fn prepared_prompt(
        &self,
        id: &str,
        user_id: Option<&str>,
    ) -> Result<Option<PreparedPrompt>> { Ok(None) }
}
```

| Method | When called | Purpose |
|--------|------------|---------|
| `is_prepared` | Once at startup | Declare which path this provider uses |
| `prepared_prompt` | Per request | Return a server-assembled prompt, or `None` |
| `get_persona` | Once at startup | Fetch static definition (name, role, tones, background) |
| `get_effective_personality` | Per request | Return blended traits, optionally scoped to a user |

A provider takes one of two paths. **Prepared** providers override
`is_prepared` and `prepared_prompt`, and their prompt is used verbatim.
**Assembling** providers implement `get_persona` and
`get_effective_personality`, and `PersonaContextBuilder` composes the prompt
client-side via `build_system_prompt`. The last two methods have no defaults;
the first two default to "not prepared", so existing providers are unaffected.

### MagickmindAgentPersonaClient (cloud, prepared)

Calls `POST /v1/end-users/{agent_id}/persona/prepare`, which returns a fully
assembled `system_prompt` — identity, background, traits and tones are blended
server-side, so no client-side formatting happens.

```rust
let client = MagickmindAgentPersonaClient::new("https://api.magickmind.io", auth);
let stage = PersonaContextBuilder::new(Arc::new(client), agent_id).await?;
```

The path segment is an **agent id**, not a persona id. Passing a persona id
returns 404 ("Agent not found"). Configure with
`persona.type = "magickmind-prepared"`, which reads `agent.agent_id` and
ignores `persona.persona_id`.

The route takes either a service-user credential or an agent's own end-user
token (minted via `POST /v1/end-users/tokens`). Under an end-user token the
path id must equal the token subject; naming another agent returns 403
("agent_id does not match token subject"), so the route cannot be used to read
another agent's prompt.

`get_persona` and `get_effective_personality` return errors on this client —
the prepare endpoint exposes neither a schema nor a trait list. Reach for
`MagickmindPersonaClient` if you need that raw data.

### MagickmindPersonaClient (cloud, assembling)

The previous two-call path: fetches the static definition and the blended
traits separately, then assembles the prompt client-side. Keyed by persona id.
Behaviour is unchanged — swap to this type to keep it.

```rust
let client = MagickmindPersonaClient::new("https://api.magickmind.io", auth);
let stage = PersonaContextBuilder::new(Arc::new(client), persona_id).await?;
```

### LocalPersonaProvider (file-based)

Reads from local files. No network required. Ideal for development:

```rust
let provider = LocalPersonaProvider::load("./personas", "aria")?;
```

---

## Local Persona Format

### Directory layout

```
personas/
  aria/
    persona.md              # required
    dyadic/
      user-123.json         # optional per-user overrides
```

### persona.md

TOML frontmatter delimited by `+++`, followed by a markdown background story:

```
+++
name = "Aria"
role = "wellness coach"
tones = ["empathetic", "calm", "encouraging"]

[traits.warmth]
value = 0.8

[traits.directness]
value = 0.6
lock = "SOFT"

[traits.formality]
value = 0.3
lock = "HARD"
+++

Aria is a warm and grounding presence who has spent fifteen years
working with people navigating life transitions. She listens first,
speaks second, and believes small steps compound into lasting change.
```

### Dyadic override file

Per-user trait adjustments in `dyadic/{user_id}.json`:

```json
{
  "user_id": "user-123",
  "traits": {
    "warmth": { "numeric_value": 0.95 },
    "directness": { "numeric_value": 0.75 }
  }
}
```

Only traits that differ from authored values need to be listed.

---

## Blending Algorithm

When both authored and dyadic values exist for a trait, the lock level determines the outcome:

| Lock | Behavior |
|------|----------|
| `HARD` | Authored value always used. Dyadic override ignored. |
| `SOFT` | Dyadic value used but clamped to +/-0.3 of authored value. |
| None | Dyadic value used as-is. |

SOFT lock clamping:

```
effective = clamp(dyadic, authored - 0.3, authored + 0.3)
```

Example: authored warmth = 0.8, SOFT lock, dyadic = 1.0 --> effective = 0.8 + 0.3 = 1.1, clamped to 1.0 (max). If dyadic = 0.2 --> effective = 0.5 (0.8 - 0.3).

The cloud provider applies an equivalent algorithm server-side and returns the already-blended result.

---

## PersonaContextBuilder Pipeline Stage

`PersonaContextBuilder` implements `PipelineStage`. It replaces `SimpleContextBuilder` when a persona is configured.

### Construction

```rust
let stage = PersonaContextBuilder::new(
    Arc::new(provider),  // any PersonaProvider
    "aria",              // persona_id, or agent_id for prepared providers
).await?
.with_history(history);  // optional conversation history
```

For assembling providers, `new()` fetches the `PersonaSchema` once so static
fields are available without per-request network calls. For prepared providers
it fetches nothing — there is no schema to fetch.

### Per-request processing

On each `process()` call:

1. Determine user_id for dyadic blending (from identity resolution or sender_id)
2. **Prepared providers:** call `provider.prepared_prompt()` and use it verbatim
3. **Assembling providers:** check the TTL cache, call
   `provider.get_effective_personality()` on miss, then compose the prompt from
   persona fields and blended traits
4. Set `ctx.llm_messages` to: system prompt, history, current message

### Generated system prompt

```
You are Aria, a wellness coach.

Aria is a warm and grounding presence who has spent fifteen years
working with people navigating life transitions...

Your personality traits:
- warmth: 0.9
- directness: 0.6 [lock: SOFT]
- formality: 0.3 [lock: HARD]

Communication tones: empathetic, calm, encouraging
```

---

## Caching

`PersonaContextBuilder` holds an in-memory TTL cache keyed by `"{persona_id}:{user_id}"`.

- **TTL source:** Set by the server via `ttl_seconds` on each response. `0` means do not cache.
- **Eviction:** Expired entries evicted on access. Write-lock sweep at 200 entries.
- **Local provider:** Always returns `ttl_seconds = 0` (blending is in-process, no caching needed).
- **Prepared providers:** Not cached client-side — every request calls
  `prepared_prompt()`. `PreparedPrompt` carries `ttl_seconds` for callers that
  want to cache, but the stage does not act on it yet.

---

## Integration

```rust
use mindroid::persona::{LocalPersonaProvider, PersonaContextBuilder};

let provider = LocalPersonaProvider::load("./personas", "aria")?;

let persona_stage = PersonaContextBuilder::new(Arc::new(provider), "aria")
    .await?
    .with_history(history.clone());

let pipeline = Pipeline::new()
    .add_stage(persona_stage)
    .add_streaming_stage(llm_processor)
    .add_stage(post_processor);
```

For cloud-backed personas, prefer the prepared path (keyed by agent id):

```rust
use mindroid::persona::{PreparedPersonaClient, PreparedPersonaContextBuilder};

let client = PreparedPersonaClient::new("https://api.magickmind.io", auth);
let stage = PreparedPersonaContextBuilder::new(Arc::new(client), agent_id);
```

Or the legacy two-call path, keyed by persona id:

```rust
use mindroid::persona::MagickmindPersonaClient;

let client = MagickmindPersonaClient::new("https://api.magickmind.io", auth);
let stage = PersonaContextBuilder::new(Arc::new(client), "aria").await?;
```

See also: [local-persona.md](local-persona.md), [magickmind-integration.md](magickmind-integration.md).
