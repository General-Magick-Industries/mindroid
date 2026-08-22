# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This crate is pre-1.0: breaking changes may land in any release, and are always
listed under **Breaking Changes** with a migration note.

## [Unreleased]

### Breaking Changes

#### 1. Tool-protocol traffic is declared, not sniffed

Tool manifests, per-turn tools and tool results were located by parsing
`Message::content` as JSON. Dispatch now comes from the sender-declared
`Message::message_type` plus transport metadata, so a participant on a
multi-party channel can no longer rewrite an agent's tool registry by quoting a
manifest envelope into what they say.

`ToolsManifest::from_envelope` and `ToolsManifest::per_turn_from_message` are
removed:

```rust
// before — parsed the message body
ToolsManifest::from_envelope(&msg.content)
ToolsManifest::per_turn_from_message(&msg.content)
// after — reads what the transport stamped
ToolsManifest::declared_manifest(&msg) // for a declared TOOL_MANIFEST
ToolsManifest::from_metadata(&msg)     // for per-turn tools on an ordinary turn
```

Transports must stamp `tools` and `context` into `Message::metadata` under
`TOOLS_METADATA_KEY` / `CONTEXT_METADATA_KEY` and set `message_type`. The
bundled Centrifugo and stdio transports already do.

#### 2. `normalize_tool_result` no longer self-dispatches

It no longer decides whether a body *is* a tool result; the caller dispatches on
`MessageType::ToolResult` first. It accepts the fields bare or in the historical
`{type, payload}` wrapper.

#### 3. `MessageType` gained variants and is now `#[non_exhaustive]`

`ToolCall`, `ToolResult` and `ToolManifest` were added. Exhaustive matches need a
wildcard arm; the attribute means this is the last release in which adding a
variant breaks you.

#### 4. `MagickmindClient::prepare_context` returns `PreparedContext`

It returned the context messages bare, which left no way to hand the parsed
corpus catalog to the embedding application. It now returns
`PreparedContext { messages, corpora }`:

```rust
// before
let messages = client.prepare_context(...).await?;
// after
let prepared = client.prepare_context(...).await?;
let messages = prepared.messages; // prompt-ready, as before
let corpora = prepared.corpora;   // Vec<CorpusCatalogEntry> {id, name, description}
```

`MagickmindContext` (the `ContextProvider`) is unchanged.

### Added

- Context prepare now parses the `corpora` catalog (the space's bound knowledge
  bases) and injects a sanitized system block listing each entry's id, name and
  description, so the model knows what a corpus-query tool can reach. The parsed
  entries are exposed on `PreparedContext::corpora` for tool wiring; the field
  deserializes to empty when the backend omits it.
- `RecallTimeWindowTool` (`recall_time_window`): recalls episodes in a date window, for questions about *when* rather than *what*. Requires an end-user credential.
- `MessageType::from_wire` and `MessageType::is_control`.
- `TOOLS_METADATA_KEY` / `CONTEXT_METADATA_KEY` in `core::models`.
- Per-turn `context` renders as a sanitized, bounded system block on the turn it
  rides, gated on an authenticated sender.

### Fixed

- `PerTurnToolsStage` now requires an authenticated sender, matching
  `ManifestStage`. Previously an unnameable publisher's tool names and
  descriptions reached the turn's system prompt.
- Manifest revocation. A backend that omits an empty array (`omitempty` and its
  equivalents) sends a withdrawal as `TOOL_MANIFEST` with no `tools` key;
  that now clears the remote set. Unusable metadata still refuses to clear, so
  garbage cannot be used to revoke. Embedders stamping metadata themselves
  should send an absent or empty `tools` to revoke.
- `EpisodeIngestStage` refuses control traffic on every `IngestScope`. A tool
  manifest or result is not an episode, and topic detection would otherwise
  mint micro-episodes from protocol traffic.
- Manifest tool descriptions are now markup-escaped as well as flattened, so a
  description cannot forge a `<tool_result>` frame in the system prompt. Text
  reaching the prompt also has Unicode separator, bidi, zero-width and tag
  characters folded — the tag block encodes an invisible ASCII alphabet that
  `char::is_control` does not cover.
- Control traffic with no consumer is refused by `Pipeline` itself, before any
  stage runs. An inbound `TOOL_CALL` has no inbound consumer — the runtime
  issues calls and never executes one — and a declared `TOOL_RESULT` whose body
  is not one complete `<tool_result>` envelope can no longer walk past the
  correlation gate as an ordinary turn. Both previously reached the LLM as user
  content. A pipeline that omits `RemoteResultGate`, or orders it after context
  building, no longer becomes the permissive one.
- `RemoteResultGate` and `ToolExecutorStage` now activate on the declared
  `MessageType::ToolResult` as well as on `<tool_result>` markup. Correlation
  keyed on markup alone, so a result whose body failed to normalize kept its
  raw body and slipped the gate by no longer looking like a result.
- The client-advertised half of the tool system prompt is capped at 32 KiB of
  *rendered* text, after sanitization and escaping. The manifest's 64 KiB cap
  counts wire JSON, and escaping expands it — repeated `&` renders at 5x — so a
  nominal 64 KiB manifest could render roughly 320 KiB of prompt.
- Neutralizing a remote tool's text moved from `ToolsManifest::build_tools_for`
  to the render in `ToolRegistry::system_prompt`, and now covers the name,
  description, schema property keys and their descriptions. Escaping at build
  time only protected tools that arrived through a manifest: a `RemoteTool` an
  embedder constructs directly reached the prompt raw, so a description could
  forge a `<tool_result>` frame. Schema text was never escaped on either path,
  and schema *keys* are bounded only here — `schema_is_bounded` walks values.
  `RemoteTool::description()` now returns the raw text it was given; the prompt
  is where the escaping happens.
- **Every untrusted path into the prompt is now folded, escaped and bounded.**
  Previously only manifest tool text was. The declared-type refusals above stop
  control traffic, but nothing stopped a participant simply *typing*
  `<tool_result name="shell">…</tool_result>` into ordinary chat: it reached the
  model in the USER role byte-identical to genuinely executed tool output, and
  the `[Name]:` attribution filter deliberately stripped the prefix from exactly
  those bodies, so the forgery arrived looking *more* machine-like than a real
  turn. Now covered:
  - the live turn (`persona::assemble_llm_messages`, `SimpleContextBuilder`),
    except when `RemoteResultGate` has authenticated and claimed it — that
    exemption is what keeps genuine correlated results working;
  - replayed chat history, both the participant turns and the agent's own
    (the latter replays as `assistant`, and a participant steers it in one hop
    by asking the agent to quote a frame back, since responses persist verbatim);
  - retrieved knowledge and corpus documents, which land in the SYSTEM role;
  - the sender's display name, now also gated on an authenticated sender,
    matching the rule the per-turn `context` block already applied.
- Prose keeps its newlines (real turns are multi-line) but loses every
  *invisible* control, via a new `sanitize_block`. Names and single-line values
  are still fully flattened by `sanitize_line`, since a newline there forges a
  further `[Name]:` turn. The folded set gained the variation selectors
  (U+FE00–FE0F, U+E0100–E01EF), soft hyphen, Mongolian selectors, Hangul
  fillers and musical controls — U+E0100–E01EF alone is a 240-symbol invisible
  alphabet, strictly more capable than the tag block already covered.
- Prompt blocks are capped at 8 KiB of **rendered** text, after escaping. Cap
  before escaping and `&` expands 5x on the way out, bounding the wire form and
  letting the prompt reach 40 KiB — the same defect the tool-prompt budget above
  exists to prevent. Corpus documents are bounded individually, so one oversized
  document cannot starve the rest or splice itself into the next.
- `CorrelatedRemoteResult` carries the id of the message it was claimed for.
  Run scope outlives one `Pipeline::run` and `run_with_context` shares a
  `Context` across runs, so a bare marker let one genuine claim exempt every
  later declared result on a reused `Context`.

## [0.0.2-a.1] — 2026-08-06

The "v2" release. Four breaking API changes, plus one silent behavior change —
all listed below with migrations.

### Breaking Changes

#### 1. `Tool::execute` takes a `&ToolContext`

```rust
// before
async fn execute(&self, args: Value) -> Result<String>
// after
async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<String>
```

Every `impl Tool` must be updated. `ToolContext` carries the trusted per-message
`channel_id` / `sender_id` plus a typed extension map for backend data set by a
pipeline stage.

**Migration:** add the parameter. If your tool does not need it, name it `_ctx`.
Never read identity out of `args` — that value is model-generated. See ADR-0005.

#### 2. `ContentPart` media variants gained a `metadata` field

`ContentPart::{Image, Audio, Video, File}` now carry `metadata: ContentMetadata`.

Serde round-trips are **unaffected** — existing payloads deserialize
byte-identically (pinned by `test_media_metadata_is_backward_compatible`). What
breaks is Rust-side struct-literal construction and exhaustive field patterns.

**Migration:** use the new constructors (`ContentPart::image`, `::audio`,
`::video`, `::file`) instead of struct literals, and add `..` to exhaustive
patterns.

`ContentPart` and its four media variants are now `#[non_exhaustive]`, so this
is the last release in which a field addition breaks you. That attribute is
itself part of this break: external exhaustive field patterns must use `..`,
and the media variants can no longer be built with struct literals at all.

#### 3. `PersonaCaller` renamed to `CredentialKind`

**Migration:** rename. A `#[deprecated] pub type PersonaCaller = CredentialKind`
alias is re-exported for one release, so existing code compiles with a warning.
The alias will be removed in the next breaking release.

`CredentialKind` is `#[non_exhaustive]` — match with a `_` arm.

#### 4. `Tool` gained a feature-conditional method

Under `feature = "artifacts"`, `Tool` gains `fn artifact_store(&self) -> Option<Arc<dyn ArtifactStore>>`.
It has a default body, so this is **not** source-breaking.

Be aware that the trait's shape now varies with a feature flag: under Cargo
feature unification, any crate in your dependency graph enabling `artifacts`
changes the trait for every consumer. The default body means this is benign
today; it would become a foot-gun if the default were ever removed.

#### 5. Backend-specific code moved from `persona` to `magickmind`

`auth/enduser.rs` (`EndUserAuth`) and `tools/magickmind.rs` (`EpisodicMemoryTool`,
`AgentCredentials`, `AgentCredentialsStage`) were gated on `persona`; they are now
gated on `magickmind`. The `auth.type = "enduser"` config variant moves with them.

Enabling the persona system no longer compiles in a token client for one service.

**Migration:** add `magickmind` to your feature list if you use any of the above.
`full` includes it, so `--all-features` and `features = ["full"]` are unaffected.

This release ships no hosted artifact backend; `magickmind` covers end-user
credentials and backend-routed tools only. Artifact storage is pluggable — use
`artifacts_from_store` with any `ArtifactStore` implementation, such as the
built-in `LocalArtifactStore`.

### Fixed

- **At most 8 artifacts are re-attached per tool round**, deduplicated. The
  model chooses how many `get_artifact` calls a round makes and every result was
  held in memory at once; a round asking for more now receives 8, and the
  message says which ids were left out.
- **`LocalArtifactStore` now bounds artifact size and rejects Windows device
  names.** Reads and writes are capped at 64 MiB (the mime sidecar at 64 KiB),
  and on Windows an id resolving to a device (`NUL`, `COM1`, …) is refused.
  These are behaviour changes: an artifact larger than the cap no longer saves
  or loads, and the device-name rule applies on Windows only, so a store
  directory written on Linux can hold ids that Windows will not read back.
  Previously the read was unbounded, so anyone able to plant a file in the
  store's directories could drive an allocation failure, which aborts the
  process.
- **The local artifact store no longer follows a symlink at the artifact path,
  even under a race.** Reads open with `O_NOFOLLOW` (on Windows, the opened
  handle is checked for a reparse point) and writes use `create_new`, so the
  final path component that was validated is the one that is used. The prior
  stat-then-open check left a window in which an attacker able to write into
  the scope directory could swap in a symlink. Scoped deliberately: this covers
  the final component only. Still reachable by anyone who can write into the
  store's directories: replacing the *scope directory* between validation and
  open, a hardlink at the artifact path, and a FIFO there — which blocks the
  *open*, pinning a blocking-pool thread for good rather than merely stalling a
  read. Closing those needs `openat`-style traversal pinned to a directory
  handle.
- **A model-supplied artifact id can no longer escape the store's base
  directory on Windows.** `LocalArtifactStore` rejected absolute paths, but a
  drive-relative component like `C:evil` is not absolute — and joining one
  discards the base it is joined onto, so the id resolved against that drive's
  working directory instead. Since ids reach the store from `get_artifact`, this
  was reachable by the model: an out-of-jail file-existence oracle, a bounded
  read, and — through `delete`, which needs no sidecar and ignores errors —
  arbitrary file deletion. Every path component must now be a single ordinary
  component, and containment is re-checked after the join.
- **An idle end-user agent no longer dies in its second hour.** The refresh tick
  was 80% of the token TTL while rotation only triggers inside the last 120s, so
  at the server's 3600s default the tick fired at 2880s with 720s remaining —
  served the cached token, rotated nothing, and let the token expire. The next
  rotation then presented an *expired* JWT, which is terminal. Only a TTL under
  600s would ever have worked. The tick now derives from the rotation window, so
  it lands inside it for every TTL. A connect reply with no TTL polls hourly
  instead of assuming a day of validity.
- **`auth.type` now decides the credential type, and `rotate_token` only decides
  how it stays fresh.** `type = "enduser"` with `rotate_token = false` previously
  returned a plain static token holder, which reported `ServiceUser` — routing an
  end-user token to the service-user surface — and could never latch a rejection,
  so a supervisor that stopped redelivering yielded an agent that 401'd every
  call while reporting healthy. Both modes now use `EndUserAuth`: `true`
  self-refreshes, `false` is supervised (holds the credential, tracks expiry,
  surfaces rejection, never rotates).
- **A supervised token is refused at startup** rather than dying at the first
  rotation. The server mints supervised tokens by default and bars them from the
  refresh route (403, terminal), so `rotate_token = true` plus a supervised token
  was an agent that started cleanly and killed itself an hour later.
- **An authorization denial no longer kills the credential.** `403` is used for
  ordinary outcomes — not a member of this space, belongs to another tenant — so
  an agent mentioned in a space it was removed from discarded a valid, unexpired
  token. Only `401` now marks a credential rejected; rotation still treats its
  own 403 as terminal, because there the verdict *is* about the credential.
- **A malformed rotation request is retryable.** `400` (an over-cap
  `token_ttl_seconds`, say) latched the credential dead, turning a config typo
  into an agent that never recovers.
- **Rate-limit backoff leaves headroom.** The ceiling was 60s against a 60/hour
  budget that counts rejected attempts, so sustained retries consumed the whole
  allowance and a recovering client found its legitimate rotation refused.
- **A token that expires while rotation is backing off now latches terminal.**
  It was the one unrecoverable path that left `is_terminal()` false, so the
  reconnect loop spun forever holding the runtime open — the exact zombie that
  flag exists to prevent.

- **The Centrifugo listener is now cancellable.** `listen` spawned a detached
  task whose only exit was a failed `tx.send` — which a quiet channel never
  triggers — so `run_until_cancelled` leaked a task, socket, and subscription per
  activation, and a second `run()` subscribed twice. It now takes a
  `CancellationToken`; `disconnect` cancels and awaits with a timeout.
- **End-user credentials rotate again on an idle connection.** The refresh tick
  skipped `get_token()` entirely for proxy-routed credentials, but `EndUserAuth`
  rotates lazily inside `get_token` and owns no timer, so that removed the only
  rotation heartbeat: an idle agent died at expiry and the reconnect loop then
  spun forever, holding `tx` so `run()` never returned. The tick now drives
  rotation without sending a frame, and the loop exits on a terminal credential.
- **`parse_push` no longer defeats the runtime's two loop-breakers.** A missing
  sender became `"unknown"` (never equal to an agent id, so the self-echo guard
  could not fire) and a missing id became a fresh UUID per delivery (so dedupe
  could not fire). Unattributed pushes are now dropped, and an id-less payload
  gets a stable content-derived id. Pushes for an unsubscribed channel are
  rejected, and `channel_id` — the artifact scope — comes from the subscribed
  channel rather than the payload.
- **Remote tool results are correlated against outstanding calls.** An
  unsolicited `tool_result` — for a call the agent never made — was accepted and
  rendered into history as genuine execution output. `ToolExecutorStage` now
  records each emitted call's `tool_call_id`, and the new `RemoteResultGate`
  stage (`ToolExecutorStage::result_gate`) claims results against it. Claiming is
  one-shot, so at-least-once redelivery cannot append the same result twice, and
  the pending set is per-channel, bounded, and expiring.
- **Inbound tool results can no longer forge tool executions.** `name` and
  `content` were interpolated into `<tool_result>` unescaped and unbounded, so a
  payload could close the tag and open another that the model reads as genuine
  execution output. `name` is validated, `content` escaped and capped.
- **Manifest tools are validated.** Names and descriptions reach the system
  prompt; names are now charset-checked, descriptions flattened and capped, and
  entries colliding with a local tool name rejected. `ManifestStage::trust_sender`
  restricts registry writes to a known client.

- **A revoked credential is now visible.** `TokenFate::Dead` was only ever set
  inside rotation, so a revocation discovered through a 401 on any other call
  left `is_terminal()` false and `is_authenticated()` true while every request
  failed — including to the reconnect loop that exists to stop exactly that.
  All six credential-bearing call sites now report rejections to the credential.
- **A credential can no longer contradict its configured routing.**
  `from_config_with_auth` took the credential from the caller and
  `CredentialKind` from config with nothing reconciling them, so an injected
  end-user JWT could be presented to service-user surfaces and fail as opaque
  401s. Mismatches now fail at construction, where the cause is legible.

### Changed

- **A persistent `404` on the token-refresh route is now terminal.** A route
  that is genuinely absent cannot be waited out, and retrying one forever left
  the agent alive but unable to renew — reporting healthy right up until its
  token quietly expired. It latches only after five consecutive 404s, roughly
  ten minutes once backoff reaches its ceiling: a wrong `auth.base_url` still
  fails fast, while a blue/green rollout or a restarted ingress recovers
  instead of killing the agent. Any other outcome resets the count. Unchanged:
  `401`/`403` latch on the first response, because those are verdicts about the
  credential; a missing route is not.
- **`Transport::disconnect` can now return `Err`, and `Runtime::shutdown`
  propagates it.** A listener that survives both a cooperative stop and an abort
  is reported as a shutdown failure with its handle retained, rather than
  silently detached. Embedders that ignored `disconnect`'s result now get an
  error they must handle; treat it as "shutdown did not complete", not as a
  reason to retry immediately.
- **`tool_result_name` and `tool_result_call_id` reject tags they previously
  parsed.** Both now return `None` for an open tag carrying an unknown attribute
  or a duplicate `name`/`call`, instead of returning the first match. A frame
  like `<tool_result name="peek" evil="x">` no longer yields `Some("peek")`.
  This closes a path where an attribute invisible to validation reached the
  model; downstream callers relying on the looser behavior must handle `None`.

- `Auth` gains `is_terminal()` (defaulting `false`) so a caller can distinguish
  a retryable failure from a dead credential, plus `kind()` (defaulting
  `ServiceUser`) and `note_rejection()` (defaulting to a no-op). All three are
  defaulted, so existing `impl Auth` types need no changes.
- Binary WebSocket frames are logged at `warn` with byte length only, instead of
  being Debug-formatted into the logs at `debug`.
- **`CentrifugoTransport` channel conventions are now configurable.** The channel
  grammar (`personal:{agent}#{sub}` / `user:{agent}#{agent}`), the connect-payload
  shape, and the subscribe/refresh rules were hardcoded to one deployment. They now
  come from a `ChannelNaming` strategy, defaulting to `ProxyChannelNaming` — the
  previous behavior exactly. Point mindroid at your own Centrifugo cluster with
  `.with_channel_naming(..)`. Not breaking: the default preserves current behavior.
- **`NoArtifactStore::save` now returns `Err`** instead of a fabricated id. This
  is a silent behavior change invisible to the type system. The previous
  fake-success discarded the bytes and returned an id that could never load, so
  a configuration mistake surfaced as data loss at read time rather than at
  write time. If you relied on `save` succeeding with the default store, you
  need a real `ArtifactStore` impl.
- Inline images are no longer silently dropped when `transport-ws` is disabled.
  Base64 encoding moved behind the `artifacts` feature's `dep:base64` so it no
  longer requires the WebSocket transport.

### Added

- **Observable runtime health** — `Health` (`Starting`/`Ready`/`Reconnecting`/
  `Stopped`), `HealthReporter`, `HealthWatcher`, and `Runtime::health()`. `run`
  only distinguishes running from exited; this reports the state in between, so
  a supervisor can tell a working agent from one that is alive but reconnecting
  and answering nothing. `Stopped` latches — a retained listener cannot walk a
  terminal runtime back to `Ready`.
- **`Transport::set_health_reporter`** (defaulted no-op) and
  **`Transport::reports_own_health`** (defaulted `false`). Both are defaulted,
  so existing implementations need no changes. Override `reports_own_health` to
  `true` if your transport establishes its connection in `listen` rather than
  `connect` — otherwise the runtime reports `Ready` as soon as `connect`
  returns, which for such a transport is before anything is connected.
- **`Message::conversation_id()`** — the backend conversation a message belongs
  to, separate from `channel_id`. A Magick Mind delivery channel names a
  *subscriber* (`user:{id}#{id}`), not a conversation, so the magickspace
  travels in the envelope and is carried in `metadata["magickspace_id"]`.
  `channel_id` stays the *delivery* scope, derived from the subscribed channel
  and therefore trusted: it keys the artifact store, local history, and
  per-channel call correlation, none of which consult a server. Use
  `conversation_id()` only where the value is handed to a service that
  re-authorizes the caller against it — never as a local scope or a path
  component. See `ArtifactStore`'s contract and ADR-0004.
- **Artifact storage** (`artifacts` feature) — `ArtifactStore` trait,
  `LocalArtifactStore` (path-jailed on-disk), `ArtifactOffload` stage, and
  `GetArtifactTool`. Moves media out of conversation history after the model
  reads it, re-fetching by id on demand instead of re-sending bytes every turn.
  See ADR-0004.
- **Remote tools** — tools declared to the LLM but executed by the client. The
  pipeline emits the call as its response rather than running it; the client
  returns the result as a new inbound message. Known reliability gaps are
  documented in `docs/design/remote-tool-reliability.md`.
- **End-user credentials** — `EndUserAuth` with single-flight token rotation, a
  terminal-vs-retryable failure taxonomy, and a saturating clamp against hostile
  server TTLs.
- **Episodic ingest** — best-effort, never fails the pipeline.
- `Runtime::run_until_cancelled` for cooperative shutdown.
- `src/ingest/` — `Source` / `Encoder` / `MediaEncoder` / `Base64Source` /
  `ResolvedSource`, re-exported at the crate root.
- `magickmind` feature gating the Magick Mind service integration.
- `ChannelNaming` / `ProxyChannelNaming` in `transport::centrifugo`, plus
  `CentrifugoTransport::with_channel_naming`.
- Forward-compatible `expires_in` handling. The rotation response's lifetime is
  read from `expires_in` (a duration, needing no clock reading) when the server
  sends it, falling back to `expires_at` until then. Converting an absolute
  timestamp requires subtracting the host's own wall clock, so a slow clock
  leaves the runtime believing a token outlives its real expiry — serving a dead
  credential while reporting healthy. The field is optional, so nothing changes
  until the server starts sending it.
- `auth.token_file` — a path a control plane writes freshly minted credentials
  to, re-read before rotating and whenever the credential is terminal. The
  cross-process counterpart to `replace_token`, for a pod-per-agent deployment
  where the control plane holds no handle to the agent's auth object. Read-only:
  the runtime never writes it, so a self-refreshed token stays memory-only.
  Per-credential rather than ambient, so agents sharing a process cannot cross
  tokens.
- `EndUserAuth::replace_token` — a control plane's delivery channel into a
  *running* agent. A chain that hit its absolute cap cannot be extended by
  rotation, and minting past it needs the service-user credential the agent
  deliberately does not hold; without this the only recovery was restarting the
  process. A method rather than an environment variable because `std::env` is
  per-process, so a host running several agents would otherwise let one adopt
  another's credential.

### Removed

- The `webcam` feature and its `nokhwa` dependency from `examples/artifact_agent`.
  `/snap` now generates a synthetic frame instead of capturing one, so the example
  demonstrates the offload path with no camera and no OS backends. This also drops
  `paste` (RUSTSEC-2024-0436, unmaintained) from the tree entirely.
- `docs/artifact-storage-summary.md` — a non-technical stakeholder summary that
  duplicated `docs/design.md` less accurately.

### Documentation

- ADR-0004 (ArtifactStore as a pure store; "derive text from media" rejected).
- ADR-0005 (per-invocation `ToolContext`).
- Corrected `docs/design.md`'s `NoArtifactStore` row, which described the
  superseded placeholder-id behavior.
- Corrected `artifacts_local`'s rustdoc, which claimed its `scope` was what the
  tool loads under. The authoritative scope is the live per-message
  `ctx.message.channel_id` held by the executor; the tool's pinned scope only
  feeds its confirmation string.
- Corrected `ToolContext::channel_id`'s rustdoc, which said "empty for stdio"
  while the stdio transport sets `"stdio"`.

## [0.0.1-a.1]

Initial alpha.
