# ADR-0003: OmniSession as a separate execution model

- Status: Accepted
- Deciders: Mindroid maintainers
- Applies to: real-time bidirectional audio / omni-modal sessions (`src/omni/`)

## Context

The `Pipeline` model runs stages sequentially over a single `&mut Context`. Real-time
bidirectional sessions (Gemini Live, OpenAI Realtime) are fundamentally different: audio
input, provider events, and tool execution all happen **concurrently**, and interruption
(barge-in) must be able to pre-empt playback. This does not fit a linear stage pipeline.

## Decision

`OmniSession` is a **separate execution model** that lives alongside `Pipeline`, sharing the
same Context primitives (scoped state, `CancellationToken`, `ContentPart`) but not the
sequential stage machinery.

- Its core is a **biased `select!` loop** over owned resources — audio source/sink, the
  provider event stream, and the tool executor — with cancellation checked first and
  interruption prioritized over playback.
- Providers implement an `OmniProvider` trait returning an **owned** event stream
  (`Pin<Box<dyn Stream + Send>>`), so the stream can be driven independently of `&mut self`
  calls like `disconnect`.
- State is an explicit machine: `Connecting → Listening → Speaking → ToolCall → Closed`.

## Alternatives considered

- **Extend `Pipeline` to be bidirectional.** Rejected: a stage takes `&mut Context`, but the
  concurrent audio/provider/tool tasks each need mutable access to *different* owned
  resources at the same time. `&mut Context` cannot be shared across those tasks.
- **Stream-combinator dispatch** (`events().for_each(...)` / `.map()`). Rejected: a single
  combinator closure cannot capture `&mut sink` **and** `&mut provider` simultaneously;
  `select!` + `match` over the owned resources is the idiomatic Rust solution.
- **A generic `Pipe<T>` operator for the audio chain.** Rejected: the chain
  (resample → encode → frame) is short and fixed; a generic pipe adds type ceremony without
  benefit.

## Consequences

- Two execution models coexist; presets (`voice_pipeline()` vs `omni_session()`) keep the
  choice explicit for users.
- `OmniSession` is observed via provider middleware + callbacks, not `PipelineEvent` (ADR-0002).
- This is a concrete instance of the ADR-0000 rule: imperative shell where concurrent `&mut`
  ownership requires it; the combinator/functional layer stays in `Pipeline`.
