# Architecture Decision Records

ADRs capture **architecturally-significant decisions** for the Mindroid SDK — choices
that are costly to reverse and constrain future work. They record the *context*, the
*decision*, the *alternatives considered and why they were rejected*, and the
*consequences*.

**They are not per-feature.** Most features add zero ADRs. Write one only when a
decision has real alternatives, cross-cutting impact, and a "why is it done this way?"
that a future contributor (human or agent) would otherwise have to reverse-engineer.

## Conventions

- Numbered sequentially, `NNNN-kebab-title.md`. Append-only.
- Once `Accepted`, an ADR is **immutable**. To change a decision, write a new ADR that
  **supersedes** the old one and set the old one's status to `Superseded by NNNN`.
- `AGENTS.md` points agents here. Before changing a design invariant, read the relevant
  ADR and do not contradict an accepted one.
- **Scope: SDK architecture only.** Magick Mind service-integration decisions (billing,
  cross-service wiring, private roadmap) belong in the private monorepo, not this public repo.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0000](0000-principles.md) | Design principles (functional core, imperative shell) | Accepted |
| [0001](0001-concurrency.md) | Structured concurrency over detached spawns | Accepted |
| [0002](0002-observability.md) | Observability via middleware, not a mutable observer registry | Accepted |
| [0003](0003-omnisession.md) | OmniSession as a separate execution model | Accepted |
| [0004](0004-artifact-store.md) | ArtifactStore as a pure out-of-band media store | Accepted |
| [0005](0005-tool-context.md) | Per-invocation `ToolContext` on `Tool::execute` | Accepted |
| [0006](0006-artifact-path-jail.md) | Artifact path jail — lexical validation plus no-follow opens | Accepted |
| [0007](0007-runtime-health.md) | Runtime health is a level-triggered `watch` sink, not a callback | Accepted |
