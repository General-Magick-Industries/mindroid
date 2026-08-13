# ADR-0002: Observability via middleware, not a mutable observer registry

- Status: Accepted
- Deciders: Mindroid maintainers
- Applies to: logging, metrics, tracing, event capture across pipelines and sessions

## Context

The SDK needs cross-cutting observability: per-stage timing, provider call logging, metrics,
and structured session events (transcripts, tool calls, interruptions). The naive approach —
a global registry of mutable observers that the core loop calls into — couples concerns,
causes borrow-checker friction inside concurrent loops, and makes behavior non-local.

## Decision

Observability is **composed**, in three complementary forms:

1. **Trait middleware (tower-style) for wrapping.** Cross-cutting behavior wraps a trait
   implementation: `LoggingProvider<P: OmniProvider>` delegates every method to the inner
   provider and adds logging/metrics/retry. Middleware composes:
   `LoggingProvider<MetricsProvider<GeminiLiveProvider>>`. This is decorator composition, not
   mutation.
2. **The `PipelineEvent` stream** (fire-and-forget mpsc) for pipeline lifecycle events
   (`StageStarted`, `StageCompleted`, `Cancelled`, …). Opt-in; observers subscribe.
3. **Callback traits** (e.g. `OmniEventCallback`) for session-level events, injected via the
   builder — used for things like transcript persistence.

Read-only stream observation uses `StreamExt::inspect`; that is the one place stream
combinators are the right tool, because it is a linear read-only tap, not multi-resource dispatch.

## Alternatives considered

- **Mutable observer registry** (`Vec<Box<dyn Observer>>` + side-effecting `notify()`).
  Rejected: the OO observer anti-pattern — global mutable state, non-local effects, and
  borrow conflicts inside `select!` loops.
- **Logging inside the core loop.** Rejected: couples the loop to a logging policy and can't
  be swapped or layered per deployment.
- **Global `tracing` spans only.** Kept for ambient logs, but insufficient alone for
  structured, subscribable session events — hence the event stream + callbacks.

## Consequences

- New cross-cutting concerns must be a middleware wrapper, an event on the stream, or a
  callback — never a new mutable registry (this is also an AGENTS.md anti-pattern).
- [ADR-0007](0007-runtime-health.md) adds a fourth form for *level-triggered* state: a
  single-producer `watch` channel. The three forms above are all edge-triggered, and none
  can answer "what is the state right now?" without an observer shadowing its own copy —
  which is the duplicated mutable state the registry ban exists to prevent. The ban itself
  is unchanged.
- `OmniSession` (ADR-0003) is observed through provider middleware + callbacks rather than
  `PipelineEvent`, because it is not a `Pipeline`.
- The middleware pattern is where metrics/tracing hooks are added later without touching the
  core loop.
