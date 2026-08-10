# Remote-tool reliability

Status: **partially implemented.** Process-local correlation, server-side
deduplication, sender binding, input bounds, and mandatory result gating are in
place. Durable outstanding-call storage, timeouts, client-side deduplication,
and reconnect recovery remain design work.

## Context

The remote-tool feature (see `src/tools/remote.rs`, `ToolExecutorStage`) emits a
tool call as terminal pipeline output; a client executes it and publishes the
result back, which re-enters as a new inbound message. This is
**async request-reply + correlation identifier**, resumed like a durable-execution
task token (cf. Temporal signals, Step Functions `waitForTaskToken`, A2A
`PushNotificationConfig`, ADK `LongRunningFunctionTool`).

The happy path and the process-local safety envelope are implemented. A client
can still disconnect, never respond, repeat execution, or outlive the runtime
process. Three durability and delivery gaps remain.

## The correlation token

Everything below keys off `tool_call_id`, minted in `frame_remote_call`.
`PendingRemoteCalls` already treats it as a process-local **task token**:
server-generated, single-use, TTL-bounded, and matched atomically with channel,
authenticated sender, and tool name. Durable storage would preserve that state
across process restarts.

## Durable outstanding-calls table (backs gaps 1 & 2)

A small store, backed by the existing `Memory`/persistence layer, one row per
in-flight remote call:

```
outstanding_tool_calls
  tool_call_id   TEXT PRIMARY KEY   -- the correlation token
  channel_id     TEXT               -- conversation (reply scope)
  tool_name      TEXT
  deadline       INTEGER            -- unix ts; when to give up
  consumed       INTEGER            -- 0 = outstanding, 1 = result accepted / timed out
```

- **On dispatch** (a remote call is framed and emitted): insert a row,
  `consumed = 0`, `deadline = now + timeout`.
- **On result ingest** (inbound `tool_result` with an id): preserve the current
  atomic identity/name match, then set `consumed = 1`. A duplicate or
  post-timeout result finds `consumed = 1` (or no row) and is **dropped**.

## Gap 1 — idempotency / dedup

Pub-sub is at-least-once; duplicates are normal, not edge cases.

- **Server side (result ingest): implemented process-locally.**
  `ToolExecutorStage` performs mandatory correlation and one-shot claim; an
  unknown, mismatched, or already-consumed result is dropped. Moving the same
  check to durable storage remains necessary for restart safety.
- **Client side (command execution):** the client must dedup by `tool_call_id`
  before executing, OR tools must be idempotent by construction (prefer absolute
  actions — `move_to(x,y)` — over relative — `move(dx,dy)` — so a double-execute
  is harmless). This is a client-contract note, documented in the manifest/protocol,
  not mindroid code.

## Gap 2 — timeout / orphaned tasks

A halted conversation waiting on a client that never returns hangs forever.

- Give every outstanding call a **deadline** (the table column).
- A timer sweeps for expired, unconsumed rows. Reuse the combinator layer
  (`pipeline/combinators.rs` `RetryStage`, or a small dedicated timeout routine
  in the `Routines` slot) rather than bespoke `tokio::spawn`.
- On expiry: mark `consumed = 1` and **synthesize a `tool_result`** (`<tool_result
  name="X">error: timed out</tool_result>`) injected as an inbound message, so the
  pipeline resumes and the LLM can react ("the door didn't respond…") instead of
  hanging.
- Because the row is now `consumed = 1`, a late real result is rejected (mirrors
  Step Functions invalidating the task token on timeout, and OpenAI Assistants
  refusing outputs on an `expired` run).

Timeout value: per-tool sensible default (game action ~ seconds; a long robot
task ~ minutes), overridable on the `RemoteTool` / manifest entry.

## Gap 3 — reconnect-safe delivery

"Publishes result → becomes inbound message" assumes the publish reaches the
pipeline. If the pipeline's subscriber is reconnecting, it's lost — symmetric, the
client can also miss the outbound command.

- **Use Centrifugo's built-in recovery** — publications carry an incrementing
  `offset` per channel with an `epoch`; on reconnect a subscriber resubscribes
  passing its last-seen offset and Centrifugo replays missed publications
  (`recovered: true`) within `history_size` / `history_ttl`.
- Apply on **both legs**: the client's command-receiving subscription *and* the
  pipeline's result-receiving subscription must resubscribe **with recovery**, not
  as a fresh subscription, after any disconnect.
- Config: enable channel history (`history_size`, `history_ttl`, optionally
  `force_recovery`) on the channels used for tool traffic.
- If Centrifugo's bounded history window is too short for very long tasks, the
  outstanding-calls table is the durable backstop: a result lost past the recovery
  window can still be reconciled by the client re-publishing (its own dedup makes
  that safe) or by a poll, since the server knows the call is still outstanding.

## Build order (smallest first)

1. **Persist the process-local outstanding-call set** — keeps the implemented
   server-side correlation and dedup guarantees across restarts.
2. **Timeout sweep + synthesized error result** (gap 2) — stops orphaned hangs;
   reuses `RetryStage`/routines.
3. **Centrifugo recovery config + resubscribe-with-recovery** (gap 3) — mostly
   configuration + a transport tweak.

Client-side dedup and idempotent-tool guidance (gap 1 client side) are a protocol
contract, documented for client authors, not mindroid code.

## What this deliberately does NOT change

The emit-and-halt execution model stays — this is the reliability envelope around
it, not a move to connection-held/blocking (which the research showed is the
wrong fit for unreliable clients). No new blocking, no held pipeline slots.
