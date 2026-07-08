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
}
```

| Method | When called | Purpose |
|--------|------------|---------|
| `get_persona` | Once at startup | Fetch static definition (name, role, tones, background) |
| `get_effective_personality` | Per request | Return blended traits, optionally scoped to a user |

### MagickmindPersonaClient (cloud)

Contacts the Magick Mind REST API for server-side dyadic computation:

```rust
let client = MagickmindPersonaClient::new("https://api.magickmind.io", auth);
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
    "aria",              // persona_id
).await?
.with_history(history);  // optional conversation history
```

`new()` fetches the `PersonaSchema` once so static fields are available without per-request network calls.

### Per-request processing

On each `process()` call:

1. Determine user_id for dyadic blending (from identity resolution or sender_id)
2. Check the in-memory TTL cache
3. On cache miss, call `provider.get_effective_personality()`
4. Compose the system prompt from persona fields and blended traits
5. Set `ctx.llm_messages` to: system prompt, history, current message

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

## MagickMind-prepared Prompt (delegated formatting)

`PersonaContextBuilder` (above) fetches structured trait data and formats the
system prompt **in-process** in Rust. An alternative is to let the **MagickMind
server** own prompt construction: `MagickmindPersonaStage` calls the server's
prepare endpoint and uses the returned string verbatim.

```
POST {base_url}/v1/persona/{persona_id}/prepare
body:   { "user_id": "<optional>" }
returns: { "system_prompt": "..." }
```

The server computes the effective personality — including numeric trait
*banding* (e.g. "very high", "moderate") — and returns a finished
`system_prompt` string. Use this when the MagickMind server is the single
source of truth for prompt rendering and you do not want to duplicate
formatting logic in Rust.

| | `magickmind` (SDK formats) | `magickmind-prepared` (server formats) |
|---|---|---|
| Trait banding / custom phrasing | no — raw `- warmth: 0.8` | yes — banded server-side |
| Requests from the SDK | 2 (persona + effective personality) | 1 (prepare) |
| Formatting location | Rust (`format_trait`) | Server (single source of truth) |

### Configuration

```toml
[persona]
type = "magickmind-prepared"
persona_id = "your-persona-id"              # default persona (see below)
base_url = "https://magickmind.example.com" # falls back to memory/auth base_url
cache_ttl_secs = 600                        # optional; prepared-prompt cache TTL (default 600 = 10 min, 0 disables)
# allow_insecure = true                     # permit auth headers over plaintext http:// (local dev only)
```

Auth headers ride on every prepare request, so plaintext `http://` base URLs
are refused unless `allow_insecure = true` explicitly opts in for local
development.

`RuntimeBuilder::build_magickmind_persona_stage()` returns the configured stage
(no startup network call — the server computes everything per-request). The
`persona_agent` example auto-selects it when `type = "magickmind-prepared"`; see
`examples/persona_agent/magickmind-prepared.toml`.

### Per-message persona selection (`PersonaId`)

The configured `persona_id` is a **default**. To serve many personas from one
stage (e.g. a server whose mobile clients each send their own persona id), the
application sets a `PersonaId` extension on the inbound message context; the
stage uses it instead of the default. This mirrors how `IdentityResolutionStage`
sets `CanonicalUserId` — *the SDK owns the slot, the application owns the value.*

```rust
use mindroid::PersonaId;

// In an upstream, application-owned step (e.g. extracting it from message metadata):
ctx.set_ext(PersonaId(persona_id_from_mobile));
```

If no `PersonaId` extension is present, the stage falls back to the configured
`persona_id`. Source policy (where the id comes from) is the application's job;
the resolve-and-cache mechanism is the SDK's.

### Prepared-prompt caching

`MagickmindPersonaStage` caches each prepared `system_prompt` keyed by
`(persona_id, user_id)` so voice/chat turns don't call the server on every message
(the `prepare` response carries no TTL, so the cache lifetime is the SDK's
choice). The default is 10 minutes; override per stage with
`.with_ttl(Duration)` or via `cache_ttl_secs` in config. A TTL of `0` disables
caching entirely (nothing is stored).

- **Eviction:** Expired entries are swept on the write path once the cache
  exceeds 200 entries.
- **Degradation:** If a re-fetch fails, the last-good (stale) prompt is served
  so a persona-service outage doesn't drop messages. With caching disabled
  (TTL `0`) there is no stale fallback and the fetch error fails the message.

## Caching

`PersonaContextBuilder` holds an in-memory TTL cache keyed by `"{persona_id}:{user_id}"`.

- **TTL source:** Set by the server via `ttl_seconds` on each response. `0` means do not cache.
- **Eviction:** Expired entries evicted on access. Write-lock sweep at 200 entries.
- **Local provider:** Always returns `ttl_seconds = 0` (blending is in-process, no caching needed).

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

For cloud-backed personas, substitute `MagickmindPersonaClient`:

```rust
use mindroid::persona::MagickmindPersonaClient;

let client = MagickmindPersonaClient::new("https://api.magickmind.io", auth);
let stage = PersonaContextBuilder::new(Arc::new(client), "aria").await?;
```

See also: [local-persona.md](local-persona.md), [magickmind-integration.md](magickmind-integration.md).
