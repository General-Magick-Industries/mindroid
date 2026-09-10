# ADR-0009: AgentLoop is a separate execution model

**Status:** Proposed
**Date:** 2026-09-10
**Supersedes:** none

## Context

`Pipeline` runs its stages once per inbound message. A turn that reasons, calls
a tool, reads the result and reasons again has to iterate somewhere, and in a
single-pass pipeline the only place left is inside a stage. That is where the
iteration lives today: `XmlToolExecutorStage` and `ToolExecutorStage` each own a
`run_loop` that calls the LLM, executes tools and calls the LLM again, entirely
within one `process`.

The consequence is not that mindroid cannot act agentically — it does. The
consequence is that everything which wants to happen *between* rounds is
unreachable by composition:

- **Compaction.** A tool-calling turn grows its own transcript; tool output is
  the largest thing in the conversation. Nothing trims it, so a long turn ends
  by overrunning the context window rather than by finishing.
- **Approval.** `ApprovalStage` exists and is documented for HITL, but it can
  only run before or after the whole executor — never before the third tool
  call, which is where a permission prompt belongs.
- **Retry and routing.** `RetryStage` and `RouterStage` wrap the executor, not a
  round. Per-round model routing (a cheap model that escalates mid-turn) cannot
  be expressed, which is why the dual-brain design was written bespoke in the
  downstream control plane instead of composed here.
- **Observability.** Rounds are not stages, so `PipelineEvent` reports the
  executor as a single unit however many times it looped.

Each of these has been solved once inside an executor, which is also how the two
executors came to diverge (`ends_turn` honoured by one, event timing and
artifact re-attachment differing).

## Decision

Add `AgentLoop`, a third execution model alongside `Pipeline` and `OmniSession`.
It is composed *of* pipelines rather than being an extended `Pipeline`:

```rust
AgentLoop::new(body)          // runs per iteration
    .with_setup(setup)        // runs once, before the first pass
    .with_finish(finish)      // runs once, after the last pass
```

One `Context` spans every phase, so run scope is the loop's state.

A body stage requests another pass by setting `Continue` in run scope. The loop
clears it before each pass, so the request cannot latch, and a body that never
asks runs exactly once — which is today's behaviour. `ctx.halted` keeps its
existing meaning: stop, and do not resume.

`Pipeline` is unchanged. Its admission control (ADR-0008) and its
one-streaming-stage rule are stated per run and stay per run;
`AgentLoop::run_streaming` concatenates each pass's stream, swallowing the
per-pass `Complete` and emitting one for the turn with the passes' usage summed.

The native round is split into two composable stages, `LlmRound` and
`ToolRound`, with the transcript in run scope as `Transcript`. It has to be
there rather than in `ctx.llm_messages` because `LlmMessage` cannot represent an
assistant turn carrying `tool_calls` or a `role: tool` result keyed by
`tool_call_id` — the same reason `LlmClient::chat_with_tools` takes async-openai
types directly.

### It is also a stage

`AgentLoop` implements `PipelineStage`, so a loop nests inside another loop's
body — a planning loop handing off to an executing loop within one turn. Two
things differ from calling `run` directly: the response is written back to the
context, because as a stage the turn is not over; and the enclosing loop's
`Continue` is held aside for the duration, so the inner loop neither consumes
its parent's request nor leaves its own behind.

Nesting shares run scope, which is the point when composing phases of one turn
and a hazard when the two are meant to be independent agents — both would write
the same `Transcript`. A genuine sub-agent wants its own context, which is what
`DelegationTool` builds.

## Alternatives rejected

**Make `Pipeline` itself loop.** Three things resist it. `SimpleContextBuilder`
rebuilds `llm_messages` from history on every `process` and assigns it, so
re-running it per iteration discards the rounds accumulated so far — meaning
every stage would need a phase flag, and every existing pipeline would have to
declare one. `run_streaming` splits the stage list at a single `streaming_idx`
into pre/streaming/post, a shape that assumes one stream per run and would need
rewriting. And `halted` is sticky, so it cannot also mean "this pass is done".
Separate phase pipelines make the distinction structural instead, and cost no
existing caller anything.

**Keep the loop in the executor and add hooks.** A callback per round is a
second, weaker composition mechanism next to `PipelineStage`, and it would have
to be added to both executors — widening the divergence this ADR exists to
narrow.

**Terminate on an explicit `Done` signal instead of the absence of `Continue`.**
Then a body stage that forgets to signal hangs until the iteration cap. Making
"nothing asked" the terminating case means a turn ends by default and loops only
on purpose.

## Consequences

- Compaction, approval, retry and per-round routing become ordinary stages in
  the body. `TranscriptCompaction` ships as the first of them.
- `ends_turn` becomes "the round stops asking for another pass" rather than a
  flag an executor interprets, so the XML/JSON divergence on it does not
  reappear in the split stages.
- `PipelineEvent` gains `LoopIterationStarted` and `LoopCompleted`.
- Nothing existing changes behaviour: no preset, example or embedder uses
  `AgentLoop` unless it opts in, and a body with no loop-aware stage runs once.
- `LlmRound`/`ToolRound` keep `ToolExecutorStage`'s remote-tool wire contract
  exactly: the call is framed as `{type: "tool_call"}`, the outstanding call is
  recorded with the same deadline, and the returning `TOOL_RESULT` clears the
  same gate. A remote call ends the turn without requesting another pass, which
  is the natural expression of "there is nothing to iterate on until the client
  answers" — the executor needed a distinct `RoundOutcome::Remote` to say it.
- **The correlation gate moves out of the stage and into `setup`.** A split
  round cannot run it inline the way `ToolExecutorStage` does: by the time
  `ToolRound` runs, `LlmRound` has already sent the context to the model. This
  is where the XML stage's own documentation already recommended putting the
  gate, since `setup` is ahead of context building. To keep the property that
  correlation cannot simply be forgotten, `LlmRound` refuses a turn whose
  declared `tool_result` nothing claimed — a missing gate costs a refused turn
  rather than fabricated tool output reaching the model.
- Artifact re-attachment stays in `ToolExecutorStage`, as do `ToolCall` /
  `ToolResult` stream events; the split stages do not emit them yet.
- `Runtime` still drives a `Pipeline`. An `AgentLoop` is run directly by the
  embedder for now; giving `MessageContext::process_and_respond` a loop to drive
  is follow-up work, and the point at which presets can offer one.
- `Transcript` being async-openai-typed means a compaction or summarizing stage
  is written against that type rather than against `LlmMessage`. Extending
  `LlmMessage` to carry tool calls and tool results would make these stages
  backend-agnostic; that is deliberately not attempted here.
