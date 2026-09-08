# ADR-0008: The pipeline refuses unconsumable control traffic before any stage runs

- Status: Accepted (2026-08-21)
- Deciders: Mindroid maintainers
- Applies to: `Pipeline::run` / `Pipeline::run_streaming`, `unconsumable_control`,
  `MessageType::{ToolCall, ToolResult}`
- Relates to: [ADR-0000](0000-principles.md) (**deviation** — §1 says control flow
  composes from stages and the combinator set; this is the exception and why),
  [ADR-0002](0002-observability.md) (the refusal emits no `PipelineEvent`; see
  Consequences)

## Context

MM-445 replaced body-sniffing with a sender-declared `MessageType`, because a
magickspace is multi-party: any authenticated participant can publish, so a body
that *looks* like a tool manifest cannot be trusted to *be* one. Declaring the
type on the wire fixed who gets to say what a message is.

It did not fix what happens to a declared message nothing consumes. Two cases:

- **`TOOL_CALL` inbound.** The runtime issues tool calls and a client answers
  them. Nothing anywhere consumes one arriving the other way.
- **`TOOL_RESULT` whose body failed to normalize.** The transports keep the raw
  body on purpose, so a malformed result stays visible rather than vanishing.
  But `RemoteResultGate` activated on the `<tool_result>` markup that such a
  body, by definition, no longer has.

In both cases the message walked past every tool-aware stage and was assembled
into the prompt as ordinary user content — the exact property the declared type
exists to deny. `<tool_result>` is the marker the runtime uses for genuinely
executed tools, so a forged one is fabricated execution output the model treats
as real.

The obvious home for the fix is a stage. The problem is that **no bundled preset
wires any of the tool-protocol stages** — `ManifestStage`, `PerTurnToolsStage`
and `RemoteResultGate` are all wired by the embedder. A defense that an embedder
must remember to install, and to install *early enough*, is not a defense; it is
a footgun with documentation. `AGD_MM_SPAWNER` had already worked around the
same gap locally, dropping inbound `TOOL_CALL` in its own `is_discarded` before
handing the message to the pipeline — evidence that the SDK was pushing a
protocol invariant onto its consumers.

## Decision

`Pipeline::run` and `Pipeline::run_streaming` call `unconsumable_control(ctx)`
**before any stage runs and before `PipelineStarted` is emitted**. It refuses:

- `MessageType::ToolCall`, unconditionally.
- `MessageType::ToolResult` whose content is not one complete `<tool_result>`
  envelope, *unless* a `CorrelatedRemoteResult` marker is already in run scope.

A refusal sets `ctx.halted` and returns no response, matching the existing
stage-halt contract.

The exemption matters: `RemoteResultGate` strips the `call` attribute once it
has claimed a result, and the stripped body deliberately no longer validates.
Without the exemption, re-entering the pipeline over the same context — what
`BranchStage`, `RouterStage` and `MessageContext::run_with_context` all do —
would refuse the very result the gate just correlated.

### Why this deviates from ADR-0000

ADR-0000 §1 says anything transforming a request into a response is a
`PipelineStage`. This is not a transformation. It is **admission control**: a
statement about which messages are eligible to enter a pipeline at all, which is
a property of the protocol rather than of any particular pipeline's shape. A
stage can be omitted, reordered, or wrapped in a combinator that swallows its
halt; an entrance check cannot. The invariant "control traffic with no consumer
is never prompt text" has to hold for *every* pipeline, including one an
embedder assembles wrongly.

Scope of the exception, deliberately narrow: only message types the SDK itself
defines as protocol traffic, only where the SDK can prove nothing downstream
consumes them. It is not a licence to put policy in the engine.

## Alternatives considered

- **A `ControlTrafficGate` stage every preset installs.** The honest ADR-0000
  answer, and it fails on the facts: the presets that would install it
  (`ollama_pipeline`, `magickmind_pipeline`) do not wire the other tool stages
  either, and an embedder building a `Pipeline` by hand — the common case, and
  what the spawner does — gets nothing. It also cannot bind itself to position
  zero, so "wired but too late" stays reachable.
- **Refuse in the transports.** Each transport would need its own copy, third-
  party transports would silently lack it, and `MessageContext` can be driven
  with a hand-built `Message` that never saw a transport.
- **Let `ContextBuilder` / persona stages filter.** Spreads one invariant across
  every prompt-assembling stage, present and future, and each is optional too.
- **Refuse `TOOL_MANIFEST` alongside the other two.** Cannot: `ManifestStage`
  *is* the consumer, so refusing at the entrance would stop tools ever being
  installed. Manifests therefore remain the embedder's responsibility, and an
  unwired `ManifestStage` lets the body through as an ordinary turn. That is no
  worse than the same sender writing the same text as `TEXT`, and
  `PerTurnToolsStage` skips control traffic, so no tool injection follows.

## Consequences

- The invariant holds regardless of how an embedder assembles its pipeline.
- **`Pipeline` now depends on `tools::remote::validated_tool_result`** — the
  generic engine knows one protocol's validator. Accepted as the cost of the
  guarantee; if a second protocol ever needs admission control, this should
  become a trait rather than a growing `match`.
- **A refused message emits no `PipelineEvent`**, only a `warn!` carrying the
  channel and a `&'static str` reason (never the body). An embedder watching the
  event stream per ADR-0002 cannot currently count refusals. Known gap; the
  event enum is public API and widening it is a breaking change, so it is left
  to a follow-up rather than smuggled in here.
- **Only normalize and validate are enforced here.** Authenticating a sender and
  claiming an outstanding call need `PendingRemoteCalls`, which is stage state a
  `Pipeline` cannot see, so a well-formed but *unsolicited* result still depends
  on `RemoteResultGate` being wired. This narrows the hole; it does not make a
  gate-less pipeline safe.
- `RetryStage` calls `Context::reset_output`, which clears run scope while
  leaving the gate's already-stripped body in place. A correlated result inside
  a retried sub-pipeline is therefore refused on the retry. Fails closed, and
  narrow enough to leave to a follow-up.
- An embedder whose agent legitimately *is* a tool executor can no longer
  receive inbound `TOOL_CALL` through a `Pipeline`. No such consumer exists
  today (the spawner already discards them). If one appears, it needs a
  successor ADR, not a quiet revert.
