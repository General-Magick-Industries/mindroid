# ADR-0009: An artifact store chooses its scope; the SDK ships a hosted store

- Status: Accepted (2026-10-06)
- Deciders: Mindroid maintainers
- Applies to: `ArtifactStore::scope_for`, `ArtifactOffload`, both tool executors'
  artifact re-injection, `MagickmindArtifactStore`
- Supersedes: two points of [ADR-0004](0004-artifact-store.md) — "scope is
  `ctx.message.channel_id`" and "the SDK ships no hosted client". The rest of
  ADR-0004 stands.

## Context

ADR-0004 keys every artifact by `(scope, id)` with `scope = ctx.message.channel_id`,
because the delivery channel is the one value no publisher controls. That is right for
a store that enforces nothing itself, such as `LocalArtifactStore`.

It is wrong for a store whose backend scopes artifacts differently. On Centrifugo the
delivery channel names a *subscriber* (`user:{id}#{id}`), not a conversation. Magick
Mind's artifact service files uploads under a magickspace and checks, on every call,
that the caller may act in that space. Called with the delivery channel it has nothing
to resolve.

## Decision

- `ArtifactStore` gains `scope_for(&self, message: &Message) -> String`, defaulting to
  `message.channel_id`. Every caller that picked a scope — the offload stage and both
  executors' re-injection — now asks the store. Existing stores keep the old behaviour
  without a change.
- A store may override it only when its backend **authorizes every operation itself**,
  so a scope a publisher influenced can at most name something the caller is already
  allowed to reach.
- `MagickmindArtifactStore` (feature `magickmind`) overrides it with
  `Message::conversation_id()`, the magickspace id, and lives in the SDK beside
  `MagickmindMemory` and `MagickmindClient`: the backend integrations that already ship.
  What Bifrost checks, by credential:
  - **end user:** presign, finalize and the space download require membership of that
    space; the fallback download and delete use the caller's *own* routes, which serve
    only artifacts this identity uploaded, from any space;
  - **service user:** finalize, download and delete are tenant-wide.

  The guarantee is therefore the platform's tenant boundary plus, for end users, space
  membership on space routes, not per-space isolation of one identity's own uploads.
  The store only puts a scope or id into a URL when it is a plain token.

## Alternatives considered

- **Keep `channel_id` and map it to a space inside the store.** The store would need a
  subscriber-to-conversation table that only the transport has, and a channel names a
  subscriber who may be in many spaces.
- **Always scope by `conversation_id`.** Breaks `LocalArtifactStore`'s isolation: a
  publisher could name another conversation's local directory.
- **Ship the hosted store from the embedding host instead.** Every host on Magick Mind
  would carry a copy of the same HTTP client; the SDK already ships the other Magick
  Mind clients behind the same feature.

## Consequences

- A store overriding `scope_for` owns the security argument for it; the default stays
  the safe choice.
- The model-facing `get_artifact` tool still never supplies a scope.
