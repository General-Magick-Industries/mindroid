# ADR-0000: Design principles — functional core, imperative shell

- Status: Accepted
- Deciders: Mindroid maintainers
- Applies to: the whole SDK

## Context

Mindroid is a Rust SDK that is primarily implemented by AI coding agents and reviewed
by humans. To stay coherent across many contributors and long timelines, the load-bearing
design stance needs to be written down once, with rationale, rather than re-derived per PR.

The recurring tension is *how much* to lean on functional-programming abstractions
(combinators, monadic composition) in a language whose borrow checker makes some of those
abstractions fight the compiler.

## Decision

**Functional core, imperative shell.**

1. **Compose transformational logic from combinators.** Anything that transforms a request
   into a response is a `PipelineStage`, and control flow is built from the combinator set
   (`BranchStage`, `RouterStage`, `RetryStage`, `ApprovalStage`, `ParallelStage`). Reach for
   a combinator before writing bespoke control flow.
2. **Use imperative code only where concurrency demands it.** Where multiple owned mutable
   resources must be driven concurrently (e.g. an audio sink + provider + tool executor),
   a `select!`/actor loop is the correct, idiomatic Rust — not a failure to be "functional
   enough" (see ADR-0003).
3. **Accept traits, return structs.** Every subsystem (`Transport`, `Memory`, `Tool`,
   `OmniProvider`, …) is a small, object-safe trait so implementations are swappable.
4. **Cross-cutting concerns compose as wrappers, not registries** (see ADR-0002).
5. **Progressive disclosure for docs.** Keep `AGENTS.md` terse and always-on; put rationale
   in ADRs read on-demand. Don't inline deep design into the always-loaded context file.

## Alternatives considered

- **Pure-FP-everywhere (monadic effect system, stream-combinator dispatch).** Rejected:
  Rust's ownership rules make `.map()`/monadic chains unable to hold `&mut` over multiple
  resources at once; forcing it produces worse, not cleaner, code.
- **OO-first (inheritance, mutable observer/manager objects).** Rejected: works against the
  trait-composition grain and makes behavior non-local.

## Consequences

- The combinator layer and the concurrent-loop layer look deliberately different. That is
  intended; the boundary is "does this need concurrent `&mut` ownership?"
- New contributors get a single rule to apply and a place (ADRs) to find the exceptions.
- Deviations should be raised as a new ADR, not smuggled into an implementation PR.
