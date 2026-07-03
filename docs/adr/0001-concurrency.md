# ADR-0001: Structured concurrency over detached spawns

- Status: Accepted
- Deciders: Mindroid maintainers
- Applies to: any concurrent execution (fan-out, background work, real-time loops)

## Context

Parts of the SDK run work concurrently: the OmniSession real-time loop, and (incoming)
parallel fan-out for multi-branch patterns like Mixture-of-Agents / Fusion and
Tree-of-Thoughts. Concurrency is where lifecycle bugs (leaked tasks, un-cancellable work,
unhandled child errors) accumulate.

Tokio remains the de facto async runtime for Rust services, and the ecosystem has moved
toward **structured concurrency** — child tasks bounded by a parent scope, with explicit
cancellation — over fire-and-forget `tokio::spawn`.

## Decision

1. **Prefer structured primitives**: `JoinSet`, `select!`, and
   `tokio_util::sync::CancellationToken`. A detached `tokio::spawn` whose handle is dropped
   is disallowed except for genuinely fire-and-forget telemetry.
2. **Fan-out is value-returning, never shared-`&mut`.** A parallel stage snapshots the
   read-only inputs it needs (cheap `Arc` clones + the message list + a shared cancel
   token), runs N branches with **no shared mutable state**, collects `Vec<Result<_>>`, and
   folds the results back into `Context` in the single-threaded continuation. Branch panics
   are caught and converted to `Err` so one branch cannot poison the join.
3. **Cancellation propagates.** The parent `CancellationToken` aborts in-flight branches;
   racing/first-wins strategies cancel the losers immediately.
4. **Bound concurrency** (`buffer_unordered` / `JoinSet` with a cap) so a large panel can't
   exhaust the worker budget.

## Alternatives considered

- **Detached `tokio::spawn` per branch.** Rejected: tasks leak, cancellation is best-effort,
  child errors go unobserved.
- **Sharing `&mut Context` across concurrent tasks.** Impossible under the borrow checker and
  the reason fan-out returns values instead (see ADR-0003 for the same constraint in OmniSession).
- **A heavyweight actor framework (e.g. actix) for all concurrency.** Rejected: bounded-mpsc
  channels + `select!` already give the actor pattern where we need it, without the dependency.
- **Rayon.** Rejected: it's a CPU thread-pool for data parallelism, not async I/O concurrency.

## Consequences

- `ParallelStage` (the fan-out primitive underpinning Fusion) is specced as value-returning.
- Reviewers should reject new detached spawns and shared-`&mut`-across-tasks patterns.
- When the tokio `task::scope` structured-concurrency API stabilizes, revisit via a new ADR.
