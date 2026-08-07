# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This crate is pre-1.0: breaking changes may land in any release, and are always
listed under **Breaking Changes** with a migration note.

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

The panicking `artifacts_magickmind` stub moved to a narrower
`magickmind-artifacts` flag, so `magickmind` itself is now part of `full` while
the stub stays opt-in.

### Fixed

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
- `magickmind` feature gating the Magick Mind service integration, and
  `magickmind-artifacts` gating the panicking remote-artifact stub (excluded
  from `full`).
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
