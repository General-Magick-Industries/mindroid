# ADR-0006: Artifact path jail — lexical validation plus no-follow opens

- Status: Accepted (2026-08-11)
- Deciders: Mindroid maintainers
- Applies to: `src/artifacts/local.rs` (`LocalArtifactStore`)
- Relates to: [ADR-0004](0004-artifact-store.md) (unchanged; this refines how its
  on-disk store resolves paths), [ADR-0005](0005-tool-context.md) (identity is never
  taken from model-generated arguments)

## Context

`LocalArtifactStore` resolves `<base>/<scope>/<id>`. `id` reaches it from
`GetArtifactTool`'s model-generated arguments — under this SDK's threat model,
attacker-controlled. `scope` is caller-supplied and must come from trusted session
context, never from arguments (ADR-0005). The store is therefore a jail, not a
convenience wrapper.

ADR-0004 considered and rejected "`canonicalize`-based path jailing" because
`canonicalize` fails for files that do not yet exist, which is exactly the `save` path.
That reasoning stands for *files*. It does not settle how *directories* are confined,
and it predates three concrete escapes found in review:

1. A symlinked scope directory pointing outside the base.
2. A Windows drive-relative component (`C:evil`), which is not `is_absolute` yet makes
   `Path::join` discard the base entirely.
3. A symlink swapped in at the artifact path between a `symlink_metadata` check and the
   subsequent open — a check-then-use race.

## Decision

Layer three mechanisms, each covering what the others cannot:

1. **Lexical validation stays the primary jail.** Every component must be a single
   `Component::Normal` — no separators, `.`/`..`, null bytes, absolute paths, or drive
   prefixes. On Windows, reserved device names (`NUL`, `COM1`, …) are rejected too, since
   those resolve to devices rather than files and opening one can block. This works for
   paths that do not yet exist, which is why ADR-0004's objection does not apply to it.
2. **Directories are canonicalized and confined.** The base and scope directory are
   resolved and the scope must sit under the base. Only directories are canonicalized;
   the artifact id is never resolved this way, so the not-yet-existing-file problem
   ADR-0004 names does not arise.
3. **Containment is asserted at open time, not inferred from a prior stat.** Reads use
   `O_NOFOLLOW`; on Windows, which has no equivalent, the opened handle is checked for
   a reparse point. Writes use `create_new` (`O_EXCL`), which POSIX requires to fail on
   an existing symlink. A `symlink_metadata` pre-check is kept for a clearer error, and
   is explicitly *not* the guarantee.

## Consequences

- The guarantee is scoped to the **final path component**. Replacing the scope directory
  between validation and open is still followed, because confinement holds a path
  string rather than a pinned inode.
- Hardlinks are not detected: `O_NOFOLLOW` does not refuse one and `symlink_metadata`
  reports a plain file. A FIFO at the artifact path blocks the open and pins a
  blocking-pool thread.
- Reads and writes are capped at `MAX_ARTIFACT_BYTES` (64 MiB), the sidecar at
  `MAX_SIDECAR_BYTES` (64 KiB). Reads apply it with `take` rather than a `metadata()`
  pre-size, so no attacker-supplied length is trusted. Both are private consts, so an
  embedder storing larger media has to fork — a builder knob is the obvious successor.
- The device-name rejection is Windows-only, so the same id is accepted on Linux and
  refused on Windows. A store directory shared between hosts is not portable in that edge.
- `libc` is a dependency on unix, gated on the `artifacts` feature, for `O_NOFOLLOW`.

## Alternatives considered

- **`openat`/`openat2(RESOLVE_BENEATH)` or `cap-std`, pinning a directory handle.**
  This is the correct end state: it closes the intermediate-component race and the
  hardlink vector together, because containment becomes a property of the handle rather
  than of a path string. Not adopted yet — `openat2` is Linux-only and recent, the
  portable dirfd walk forks the read path per platform, and `cap-std` is a substantial
  dependency for one module. Recorded as the intended direction rather than rejected.
- **Dropping the `symlink_metadata` pre-check** now that opens assert containment
  themselves. Kept: it is the only symlink gate on `delete`, which uses `remove_file`
  rather than an open, and it produces a specific error instead of a raw `ELOOP`.
- **Narrowing the Windows check by reparse tag**, so cloud-placeholder and dedup files
  are not rejected. Not adopted: reading the tag means trusting an ioctl that can fail,
  and failing *open* on a symlink is worse than failing *closed* on a OneDrive file.
