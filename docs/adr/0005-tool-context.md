# ADR-0005: Per-invocation `ToolContext` on `Tool::execute`

- Status: Accepted (2026-08-06)
- Deciders: Mindroid maintainers
- Applies to: the `Tool` trait (`src/tools/`) and `XmlToolExecutorStage`

## Context

`Tool::execute(&self, args: Value) -> Result<String>` gave a tool exactly one input: the
JSON arguments the model produced. Everything else had to be baked into the tool at
construction time.

That works for a stateless tool like `shell`. It does not work once a tool needs to act
*on behalf of the current message*: which channel it arrived on, who sent it, and which
credential the runtime holds for that caller. Those vary per invocation, and a tool
constructed once at startup cannot know them. The workarounds available were all bad —
construct a fresh tool per message (defeats the registry), reach for a global (untestable,
and wrong under concurrency), or smuggle the values through `args` (model-supplied, hence
untrusted precisely where trust matters most).

## Decision

`Tool::execute` takes a second parameter: `async fn execute(&self, args: Value, ctx: &ToolContext)`.

- **`ToolContext` carries the trusted per-message identity** — `channel_id` and `sender_id`,
  copied by the executor from `ctx.message` on every invocation, never from `args`.
- **Backend-specific data rides in a typed extension map**, not in named fields. A stage
  places e.g. `AgentCredentials` into the map; the tool reads it back by type. This is what
  keeps the trait transport-agnostic — the SDK's own trait does not grow a field per backend.
- **The ext map is `Arc`-backed and shared across clones**, which is how a stage hands data
  forward to the executor without the two being coupled through a signature.
- **No default body.** The change is deliberately source-breaking for every `impl Tool`.

## Alternatives considered

- **Keep the signature and pass context through `args`.** Rejected outright on security
  grounds: `args` is model-generated. Merging trusted identity into it means a prompt-injected
  model can claim any channel or sender, and the tool has no way to tell the two apart.
- **Add a defaulted `execute_with_context` and leave `execute` intact.** Rejected: two entry
  points where only one carries identity is a trap — the safe path is the one you have to opt
  into, and every tool written from the old docs silently gets the unsafe one. A break that
  fails at compile time is cheaper than a footgun that fails in production.
- **Named fields on `ToolContext` for each backend's needs** (credentials, agent id, …).
  Rejected: it puts service-specific concepts in the SDK's core trait, and every new backend
  widens a struct that every unrelated tool must still see.
- **Thread `&Context` (the full pipeline context) into tools.** Rejected: it hands a tool
  mutable reach over the whole pipeline — history, extensions, halt flag — to answer "who sent
  this?". `ToolContext` is the narrow read-only slice that question actually needs.

## Consequences

- **Breaking for all downstream `impl Tool`.** The migration is mechanical (add the parameter,
  ignore it with `_ctx` if unused) but it is not automatic, and it is the headline break of
  this release.
- A tool that needs per-message identity no longer has any reason to be constructed per
  message, so `DynamicRegistry`'s single shared snapshot stays valid for the whole turn.
- The ext map is stringly-untyped at the boundary in the sense that a tool reading a type no
  stage ever set gets `None` — a wiring mistake surfaces as a missing value at runtime rather
  than a compile error. Tools should degrade explicitly on `None` rather than unwrap.
