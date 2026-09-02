# ADR-0007: Runtime health is a level-triggered `watch` sink, not a callback

- Status: Accepted (2026-08-13)
- Deciders: Mindroid maintainers
- Applies to: `src/core/health.rs`, `Transport::set_health_reporter` /
  `Transport::reports_own_health`, `Runtime::health()`
- Relates to: [ADR-0002](0002-observability.md) (extended, not superseded: this adds a
  fourth injection form and says why the existing three cannot express it),
  [ADR-0001](0001-concurrency.md) (the reporter is a channel, not a task)

## Context

A runtime can be alive and deaf at the same time. The Centrifugo listener reconnects in a
loop, so a process whose socket is down — or whose credential has latched terminal — is
still running, still green to any liveness probe, and answering nothing. A supervisor sees a
healthy process and leaves it alone. This is the failure MINDROID-13 describes: *"a zombie
that looks alive to any supervisor watching the future."*

The consumer is out-of-process supervision. `AGD_MM_SPAWNER` runs one OS process per agent
and surfaces per-agent state on `/status`; it needs to answer **"what is this agent's state
right now?"** at arbitrary times, not only at the moment a transition happens.

[ADR-0002](0002-observability.md) enumerates three forms for cross-cutting concerns —
trait middleware, the `PipelineEvent` stream, and callback traits — and forbids a new
mutable registry. Health fits none of them cleanly.

## Decision

Runtime health is a **single-producer `tokio::sync::watch` channel carrying a `Copy` enum**.
`HealthReporter` is the writer, `HealthWatcher` the reader; `Runtime::health()` hands out
cheap clones. The reporter reaches a transport through `Transport::set_health_reporter`,
called by the runtime before `connect`.

This is a fourth injection form, and deliberately so:

1. **Level-triggered, not edge-triggered.** A callback reports that a transition *occurred*.
   `watch` holds the current value, so a supervisor that attaches late, polls, or restarts
   still reads the truth. Under a callback every observer would shadow its own copy of the
   state — that duplicated mutable state is what ADR-0002's registry ban exists to prevent.
2. **Middleware cannot observe it.** The reconnect loop lives *inside*
   `CentrifugoTransport::listen`. A `HealthTransport<T: Transport>` decorator wrapping
   `listen` sees one call that does not return until the listener stops; the transitions it
   needs to report all happen within that call, invisible to the wrapper.
3. **It is not a registry.** One producer, no `Vec<Box<dyn Observer>>`, no fan-out, and no
   dyn dispatch back into user code from the core loop. Readers are channel receivers, so
   adding one cannot affect another or block the writer.

Two rules fall out of the mechanism:

- **`Stopped` latches.** `disconnect` may time out and retain a live listener that keeps
  publishing after the runtime has already returned. Without latching, a supervisor sees a
  terminal agent go `Ready` again and `wait_ready` loses its terminal short-circuit.
- **A transport that connects lazily owns its own `Ready`.** The runtime reports `Ready`
  once `connect` succeeds, which is only true where `connect` finishes connecting.
  Centrifugo opens its socket in `listen`, so it returns `true` from `reports_own_health`
  and the runtime stays out of the way. Without this the runtime announces `Ready` for a
  transport that has not reached its endpoint — the false green light the whole mechanism
  exists to prevent, and the defect this ADR's implementation shipped with before review.

## Alternatives considered

- **Extend the `Observer` trait** (`on_health_changed`). Rejected: edge-triggered. A
  supervisor asking "what is the state now?" would have to keep its own copy per agent,
  reintroducing the shadowed mutable state ADR-0002 bans, and a supervisor attaching after a
  transition would never learn it.
- **A `HealthTransport<T>` middleware decorator.** Rejected: cannot see inside `listen`,
  where every transition happens. It could only report "listen was called" and "listen
  returned".
- **Reuse `PipelineEvent`.** Rejected: wrong subsystem and wrong shape. Health is transport
  and credential lifecycle, not pipeline stage lifecycle, and the stream is fire-and-forget
  with no current-value semantics.
- **Poll `Transport::is_connected()` / `health_check()`.** Rejected: at a thousand agents
  that is a thousand timers doing nothing, and it cannot distinguish "reconnecting" from
  "terminal" — the distinction that tells a supervisor whether to wait or replace.
- **Expose the state over HTTP from the SDK.** Rejected: the SDK is a library. Transport of
  the signal is the host's concern; `HealthWatcher` is what it needs to build one.

## Consequences

- `Transport` gains two defaulted methods, so existing implementations keep compiling. A
  transport that establishes its connection in `listen` rather than `connect` **must**
  override `reports_own_health`, or the runtime will report `Ready` prematurely on its
  behalf.
- ADR-0002's registry ban stands unchanged. Its list of three forms is now four, with the
  constraint that the additional form is a single-producer channel carrying state — not a
  general licence for setter injection on core traits.
- Health says nothing about the pipeline or the LLM. A runtime can be `Ready` and still fail
  every turn for unrelated reasons; this reports whether the agent can *receive*.
- A cancelled `disconnect` still detaches its `JoinHandle` (the handle drops as a local).
  The transport recovers — the slot is left `Idle` and a later `listen` succeeds — but the
  task is not joined. Closing that needs a drop guard; recorded here rather than left
  implicit, per the ADR-0006 precedent.
