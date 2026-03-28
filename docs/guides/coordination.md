# Multi-Agent Coordination

When multiple agents share a communication channel, three problems emerge:

1. **Feedback loops** -- Agent A responds, Agent B responds to A, A responds to B, forever.
2. **Off-topic responses** -- An agent designed for coding answers a cooking question because it saw it in the channel.
3. **Re-engagement flooding** -- An agent keeps responding to the same conversation thread after its contribution is complete.

Mindroid solves these with **gates** (pipeline stages that halt processing) and **engagement tracking** (per-channel sender affinity).

---

## Gate System

A gate is a pipeline stage that decides whether to continue or halt:

```rust
#[async_trait]
pub trait Gate: Send + Sync {
    async fn classify(&self, ctx: &PipelineContext) -> Result<bool>;
}
```

Return `true` to proceed, `false` to halt. When a gate halts, `ctx.halted = true` and all subsequent pipeline stages are skipped. The pipeline returns `None` (no response).

### Combinators

Gates compose via boolean logic:

**OrGate** -- Passes if ANY inner gate passes (logical OR):

```rust
let gate = OrGate::new(vec![
    Box::new(MentionGate::new("@mybot")),
    Box::new(DirectMessageGate),
]);
```

**AndGate** -- Passes if ALL inner gates pass (logical AND):

```rust
let gate = AndGate::new(vec![
    Box::new(CoordinationGate::new(history.clone(), 2)),
    Box::new(RelevanceGate::new("coding assistant", url, model)),
]);
```

**Fail-open design** -- Both combinators treat gate errors as passes (log a warning, continue). This prevents a temporary LLM failure from silencing the agent entirely.

---

## CoordinationGate

**Purpose:** Prevent re-engagement flooding. Deterministic, no LLM call, fast.

**Algorithm:**

```
  Conversation history scan
  ─────────────────────────

  user:       "How do I sort a list?"
  assistant:  "Use .sort() or sorted()..."   ◄── Last agent message
  user:       "Thanks!"                       ◄── 1 new message
  user:       "Actually, one more question"   ◄── 2 new messages

  min_new_messages = 2  →  PASS (2 >= 2)
  min_new_messages = 3  →  HALT (2 < 3)
```

The gate scans conversation history backward, finds the last `assistant` message (the agent's previous response), and counts `user` messages after it. If the count is below the threshold, the agent stays silent.

```rust
let gate = CoordinationGate::new(history.clone(), 2);
```

**Why deterministic?** An LLM-based re-engagement check would be slower (network round-trip) and less predictable (different answers on retry). Message counting is instant, deterministic, and free.

---

## RelevanceGate

**Purpose:** Filter messages that aren't relevant to the agent's role. Uses an LLM call for nuanced classification.

**How it works:**

1. Builds a prompt: agent role + conversation history + latest message
2. Calls the LLM with JSON schema enforcement: `{"relevant": true/false}`
3. Parses the boolean response

```rust
let gate = RelevanceGate::new(
    "You are a coding assistant specializing in Rust.",
    "http://localhost:11434",
    "llama3.2",
)
.instructions("Only respond to programming questions. Ignore casual chat.")
.with_history(history.clone());
```

**Fail-open vs strict mode:**

| Mode | On LLM error | Use case |
|------|-------------|----------|
| Fail-open (default) | Let message through | Production -- agent stays responsive |
| Strict | Halt pipeline | Testing -- catch classification failures |

```rust
let gate = RelevanceGate::new(role, url, model).strict(true);
```

---

## EngagementTracker

**Purpose:** Prevent agents from talking over each other in group channels by tracking which user an agent is currently "engaged" with.

```rust
let tracker = EngagementTracker::new();

// Check before responding
if tracker.should_engage("channel-1", "user-42") {
    // Process and respond
    tracker.record("channel-1", "user-42");
}
```

**Rules:**
- No active engagement for this channel: engage (new conversation)
- Same sender as active engagement: engage (follow-up from same user)
- Different sender, active engagement not expired: skip (someone else's conversation)
- Active engagement expired (cooldown passed): engage (conversation moved on)

The tracker is sender-agnostic -- it doesn't know which agents are in the channel. It only tracks the agent's own engagement state.

---

## Recommended Pattern: Two-Layer Gate

For multi-agent deployments, combine both gates in an AndGate:

```
  Incoming     CoordinationGate        RelevanceGate        Pipeline
  Message ──►  (deterministic,   ──►   (LLM-based,    ──►  (process
               fast, free)              accurate,           and respond)
                                        costs 1 LLM call)
```

The coordination gate runs first as a cheap filter. Only messages that pass it trigger the more expensive relevance check.

```rust
let gate = AndGate::new(vec![
    Box::new(CoordinationGate::new(history.clone(), 2)),
    Box::new(
        RelevanceGate::from_config(llm_config)
            .instructions("Respond to Rust programming questions only.")
            .with_history(history.clone()),
    ),
]);

let pipeline = Pipeline::new()
    .add_stage(gate)
    .add_stage(context_builder)
    .add_streaming_stage(llm_processor)
    .add_stage(post_processor);
```

In a deployment with three agents (coding, writing, general), each agent uses the same two-layer pattern with a different role description in its RelevanceGate. Messages are routed to the right agent based on content, with coordination preventing re-engagement floods.
