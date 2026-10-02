# ADR-0004: ArtifactStore as a pure out-of-band media store

- Status: Accepted (2026-08-06)
- Deciders: Mindroid maintainers
- Applies to: out-of-band media storage (`src/artifacts/`, `ArtifactOffload`, `GetArtifactTool`)

## Context

Multimodal turns carry media inline as base64 in `ContentPart::{Image,Audio,Video,File}`.
Re-sending that media on every subsequent turn is the default behavior of a naive history
builder, and it is quadratic: an image attached once is re-encoded into every later request
in the conversation, burning context window and tokens for bytes the model already saw.

The SDK needed a way to move media *out* of the message history after the model has
consumed it, and to bring it *back* on demand when a later turn genuinely needs it — without
coupling that mechanism to any particular storage backend, and without letting a model-supplied
id reach the filesystem.

## Decision

`ArtifactStore` is a **core swappable trait** alongside `Auth`/`Memory`/`Transport`, and it is
a **pure store**: `save` bytes in, id out; `load` id in, bytes out. It knows nothing about
conversations, models, or relevance.

- **Scope is a trust boundary, not a parameter.** Every operation is keyed by `(scope, id)`,
  where `scope` comes from trusted context (`ctx.message.channel_id`) and never from model or
  user input. `LocalArtifactStore` validates both as single path components — rejecting
  separators, `..`, absolute markers, nulls, and empty — because `canonicalize` cannot jail a
  path that does not exist yet.
- **The offload stage and the load tool are minted as a matched pair.** `ArtifactManager::into_stage_and_tool`
  returns `(ArtifactOffload, GetArtifactTool)` sharing one store, so a store mismatch between
  the writer and the reader is unrepresentable.
- **Re-injection is the executor's job, under the live per-message scope.** `GetArtifactTool`'s
  own pinned scope is advisory — used only for its confirmation string. The authoritative
  scope for the bytes the model actually sees is the one the executor holds for that message.
- **`NoArtifactStore::save` errors.** The default no-op store refuses to fabricate an id.

## Alternatives considered

- **Derive text from media instead of storing it** (caption the image, transcribe the audio,
  keep only the text in history). Rejected as an `ArtifactStore` concern: it is lossy and
  irreversible, it bakes a model call into what should be a byte store, and it makes the
  trait's behavior depend on a captioning provider. A caption stage is a perfectly good thing
  to build — it just composes *over* the store rather than replacing it. Keeping the store
  pure leaves both options open.
- **Let `NoArtifactStore::save` return a placeholder id** so the default path never errors.
  Rejected: a fake success silently discards the bytes and hands back an id that can never
  load, converting a configuration mistake into data loss discovered at read time.
- **Store keyed by id alone, with scope as an ordinary argument.** Rejected: it makes tenant
  isolation a caller discipline rather than a store invariant, and one forgotten scope check
  reads another tenant's artifacts.
- **`canonicalize`-based path jailing.** Rejected: it fails outright for not-yet-existing
  files, which is exactly the `save` path.

## Consequences

- Media offloaded from history is retrievable but not automatically re-attached — a turn that
  needs it must call `get_artifact`, which costs a tool round-trip.
- A transport that leaves `channel_id` empty cannot use artifact offload; the store rejects an
  empty scope by design rather than falling back to a shared global namespace.
- All artifact filesystem IO is async, so no blocking syscall lands on the runtime.
- A remote backend is a user impl of the same trait; the SDK ships no hosted client.
