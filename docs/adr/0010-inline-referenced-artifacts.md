# ADR-0010: Referenced images may be inlined by an opt-in stage

- Status: Accepted (2026-10-07)
- Deciders: Mindroid maintainers
- Applies to: `InlineArtifacts`, `ArtifactStore::load_bounded`, `ArtifactManager::offload`,
  both tool executors' artifact re-injection
- Supersedes: one point of [ADR-0004](0004-artifact-store.md) — "media offloaded from
  history is not automatically re-attached; a turn that needs it must call
  `get_artifact`". The rest of ADR-0004 stands.

## Context

ADR-0004 made re-attachment on-demand: the model sees a reference line and calls
`get_artifact`, so a conversation does not resend its media on every turn.

Since `artifact_data` (ADR-0009's hosted store), a sender uploads its own media and
attaches it by reference, so even a photo sent *this* turn reaches the model as a
reference line. Small models measurably skip the tool call and describe the photo
anyway, wrongly. Matching a question to an earlier photo has the same failure.

## Decision

- `InlineArtifacts` is an opt-in stage. It loads the images referenced by the inbound
  message's `artifact_data` and attaches them inline to the current user turn. With
  `with_history_messages(n)` (default 0) it does the same for references in the `n`
  messages before it. Nothing is persisted: it changes this run's `llm_messages` only.
- Every load is bounded: at most `with_max_images` loads per turn, whether or not they
  inline, an id at most once, and `ArtifactStore::load_bounded` with the bytes left,
  so a store can stop reading at the limit. The current turn is served first.
- Only PNG, JPEG, GIF and WebP are inlined, by byte signature, here and in both
  executors' `get_artifact` re-attachment: a provider rejects a whole request over one
  image it cannot read.
- Each inlined image keeps its reference id, name and metadata as a short text label,
  and carries its id in code-only metadata so a later `ArtifactOffload` restores the
  original reference instead of storing a second copy.

## Alternatives considered

- **Keep on-demand only, and prompt the model harder.** Measured on gpt-4o-mini: it
  still described photos it had not loaded, and copied its own earlier refusals.
- **Inline by default.** Reintroduces the per-turn resend ADR-0004 avoided for every
  pipeline, including ones whose endpoint cannot take images.
- **Resolve references in the transport.** The transport has no store or scope, and
  inline bytes would then reach persistence.

## Consequences

- With a history window, the window's images are resent every turn, bounded by the
  stage's budgets; that cost is the embedder's choice.
- The stage's byte budget does not cover `get_artifact` re-attachment, which still
  has only a count cap.
- An image with a valid signature but a corrupt body can still fail a request.
