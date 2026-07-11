---
project: rlevo
status: active
type: decision
date: 2026-07-11
tags: [adr, decision, core, environment, snapshot, metadata, time-limit, box2d, locomotion]
---

# ADR 0042: `SnapshotBase` carries optional metadata

## Status

**Accepted (2026-07-11).** Resolves issue #128. Amends `docs/rules.md §10`
(the `SnapshotBase`/`Snapshot` bullet, reworded below). Narrows ADR 0019's
"`State`, `Environment`, `Snapshot`, and `SnapshotBase` are byte-unchanged"
statement — that line described **that ADR's own scope** (an additive,
unrelated trait), not a freeze on `SnapshotBase` for all future work; this ADR
is the first to change `SnapshotBase` itself. Supersedes nothing.

## Context

Two environment families hand-roll their own `impl Snapshot<1>`:
`LunarLanderSnapshot` (`crates/rlevo-environments/src/box2d/lunar_lander/snapshot.rs:25`)
and `LocomotionSnapshot<O>` (`crates/rlevo-environments/src/locomotion/common.rs:245`).
They are structurally identical — `observation` / `reward: ScalarReward` /
`status: EpisodeStatus` / a private `metadata: SnapshotMetadata` field, plus
`running`/`terminated`/`truncated` constructors — and are the **only** two
metadata-carrying snapshots in the workspace.

They exist because `SnapshotBase<R, ObservationType, RewardType>`
(`crates/rlevo-core/src/environment.rs:165`) cannot express metadata: it has no
metadata field, no constructor that accepts one, and does not override
`Snapshot::metadata()` (`:148`), so it inherits the trait's `None` default —
this despite `rlevo-core` itself declaring both `SnapshotMetadata` (`:54`) and
the `metadata()` seam it fills. Core declares a capability its own default
carrier structurally cannot carry.

**The decider is `TimeLimit`.** `wrappers/time_limit.rs:91` binds `impl
Environment<D, SD, AD> for TimeLimit<E>` where `E::SnapshotType =
SnapshotBase<D, Obs, Rew>`, then mutates `snap.status` in place to enforce
truncation. A bespoke `SnapshotType` fails that bound, so it **forfeits
`TimeLimit` composition** — already admitted as a wart in
`lunar_lander/snapshot.rs:21-23` ("`TimeLimit<LunarLander*>` does not compile
… the step limit is enforced internally via `config.max_steps`"). Six
environments (the four locomotion envs plus `LunarLanderDiscrete` /
`LunarLanderContinuous`) are locked out of `TimeLimit` — and every future
`SnapshotBase`-bound wrapper — solely because they needed a metadata field.

Verification done before deciding: `cargo check -p rlevo-environments
--no-default-features --features box2d` builds clean today (`box2d` and
`locomotion` are orthogonal Cargo features — neither implies the other, and
`locomotion::common` is `#[cfg(feature = "locomotion")]`). This matters
because the naive fix below fails exactly that check, and no existing CI job
would have caught it.

## Decision

### 1. `SnapshotBase` gains an optional metadata field and a fluent builder

```rust
#[derive(Debug, Clone)]
pub struct SnapshotBase<const R: usize, ObservationType: Observation<R>, RewardType: Reward> {
    pub observation: ObservationType,
    pub reward: RewardType,
    pub status: EpisodeStatus,
    pub metadata: Option<SnapshotMetadata>,
}

impl<const R: usize, ObservationType: Observation<R>, RewardType: Reward>
    SnapshotBase<R, ObservationType, RewardType>
{
    #[must_use]
    pub fn with_metadata(mut self, metadata: SnapshotMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }
}
```

`running`/`terminated`/`truncated` set `metadata: None`; a caller that wants
metadata chains `.with_metadata(meta)` onto the constructor. `Snapshot::metadata()`
is overridden on `SnapshotBase` to return `self.metadata.as_ref()`, replacing
the inherited `None` default.

### 2. Both bespoke types collapse to aliases

```rust
// locomotion/common.rs
pub type LocomotionSnapshot<O> = SnapshotBase<1, O, ScalarReward>;

// box2d/lunar_lander/snapshot.rs
pub type LunarLanderSnapshot = SnapshotBase<1, LunarLanderObservation, ScalarReward>;
```

Each alias is defined **locally, in its own feature-gated module**, not as one
shared cross-family type. After this, direct `impl Snapshot<_>` blocks in-tree
drop from 2 to **0** — every snapshot in the workspace is a `SnapshotBase`
instance.

### 3. Metadata generalises beyond rank-1 / `ScalarReward`

Because the field lives on `SnapshotBase<R, ObservationType, RewardType>`
itself rather than on a rank-1/`ScalarReward`-specific bespoke type, any future
environment at any rank or reward type gets the same metadata seam for free —
it is no longer coupled to the two families that happened to need it first.

## Consequences

### Positive

- One snapshot type in the entire workspace. Core's declared contract
  (`SnapshotMetadata` + `Snapshot::metadata()`) is closed by its own default
  carrier instead of leaning on two bespoke duplicates.
- `TimeLimit` composes with all six previously-locked-out environments
  (4 locomotion + 2 lunar lander) for the first time, and with every future
  environment that needs metadata — the actual defect this ADR fixes.
- Metadata is no longer coupled to rank-1/`ScalarReward`; any `SnapshotBase`
  instantiation can carry it.

### Neutral

- `SnapshotMetadata` still has **no production consumer**. `BenchStep`
  (`crates/rlevo-core/src/evaluation.rs:25`, `{observation, reward, done}`)
  drops it entirely, and `EpisodeRecord`
  (`crates/rlevo-benchmarks/src/record/schema.rs`) has no metadata field. It
  remains a debug/analysis affordance read only by env unit tests (e.g.
  `LunarLanderSnapshot`'s `test_metadata_shaping_key_present`). This ADR does
  not change that — wiring metadata into the bench/record path is a separate,
  unfiled decision.
- No serde derives are added to `SnapshotBase` or `SnapshotMetadata` by this
  change, so there is **no record `FORMAT_VERSION` bump** (ADR 0014).

### Negative / accepted costs

- **Breaking at alpha.** `SnapshotBase`'s field list changes, so every
  struct-literal construction site (5, all in-tree) must add `metadata:
  None`, and every call site that previously depended on
  `LunarLanderSnapshot::running(obs, reward, shaping)` /
  `LocomotionSnapshot::running(obs, reward, metadata)`'s bespoke constructor
  signatures (~20 sites) moves the metadata argument to a `.with_metadata(...)`
  tail instead. This is precedented: ADRs 0028, 0030, and 0038 each took an
  atomic breaking change to a core contract type at alpha rather than
  maintaining a parallel non-breaking path.
- **No `#[deprecated]` shim.** The type *names* `LocomotionSnapshot<O>` and
  `LunarLanderSnapshot` survive unchanged as aliases — callers that only name
  the type are unaffected. The break is in **constructor arity**, and a
  `#[deprecated]` shim cannot be layered on a constructor for a type that is
  now a foreign type alias (`SnapshotBase` is defined in `rlevo-core`; Rust's
  orphan rules forbid a local inherent `impl` block for a foreign type,
  E0116). CHANGELOG carries the breaking note instead (see `CHANGELOG.md`).
- **Unmeasured size cost.** Every `SnapshotBase` instance — including the vast
  majority that never carry metadata — grows by an `Option<SnapshotMetadata>`.
  `SnapshotMetadata` is two `BTreeMap`s; an empty `BTreeMap` does not
  heap-allocate, so the steady-state cost is stack-only, on the order of
  48-56 bytes depending on target pointer width and enum niche-packing. **This
  was not measured.** If a hot loop is ever suspected to regress from this,
  the `k_armed` bandit's `step` loop
  (`crates/rlevo-environments/src/classic/bandit/k_armed.rs`) is the place to
  bench — it is the tightest, highest-frequency `SnapshotBase`-construction
  path in the workspace.
- **`LocomotionSnapshot`'s "always has metadata" guarantee becomes an
  `Option`.** Before this change, `LocomotionSnapshot<O>::metadata()` always
  returned `Some`; after, it is possible (though not exercised by any current
  env) to construct one with `metadata: None`. This was already unobservable
  past the trait boundary: `Snapshot::metadata()` returns `Option<&SnapshotMetadata>`
  regardless of the implementing type, so no caller could rely on the
  always-`Some` guarantee through the trait object/generic-bound surface
  in the first place.

## Alternatives considered

- **`pub type LunarLanderSnapshot = LocomotionSnapshot<LunarLanderObservation>`
  (share one alias across both families).** Rejected on evidence: fails
  `cargo check -p rlevo-environments --no-default-features --features box2d`
  with E0433 (`locomotion::common` is unresolved), because `box2d` and
  `locomotion` are orthogonal Cargo features — neither implies the other.
  Verified before rejecting. Also the reason a CI guard is added (see
  `.github/workflows/crate-tests.yml`) — no existing job exercised
  `--no-default-features`, so this class of break was invisible to CI.
- **Promote a separate `MetadataSnapshot<R, O, Rw>` into `rlevo-core` beside
  `SnapshotBase`.** Rejected: institutionalizes the exception rather than
  closing it — every future env author would face a taxonomy fork
  ("does my env need `SnapshotBase` or `MetadataSnapshot`?") — and does
  **not** fix `TimeLimit`, since `TimeLimit` binds specifically to
  `SnapshotBase`, not to any metadata-carrying sibling.
- **An ungated shared metadata-carrying type inside `rlevo-environments`
  itself (no core change).** Rejected: leaves core's own declared contract
  (`SnapshotMetadata` + `Snapshot::metadata()`) unfulfilled by its default
  carrier, still does not fix `TimeLimit` (which binds to `SnapshotBase`
  specifically), and inverts the dependency story — a *contract* type
  (metadata carriage) would live in the *implementation* crate rather than
  the crate that defines the contract.

## References

- Issue #128 — `LunarLanderSnapshot` duplicates `LocomotionSnapshot` and
  mis-documents its metadata.
- ADR [0019](0019-observable-projection-trait.md) — the "byte-unchanged"
  statement this ADR narrows (scoped to that ADR's own additive change, not a
  freeze on `SnapshotBase`).
- ADR [0011](0011-lift-construction-off-environment-trait.md) — the
  standalone-trait-over-supertrait precedent this ADR does not need, since
  `metadata()` is already a provided trait method; this ADR only makes
  `SnapshotBase` fulfil it.
- ADR [0028](0028-tensor-batch-conversion-seam.md), ADR
  [0030](0030-permutation-tensorgenome-and-population-nonempty-invariant.md),
  ADR [0038](0038-continuous-action-components-const.md) — precedent for
  atomic breaking changes to core contract types at alpha, no deprecation
  shim.
- ADR [0014](0014-record-schema-v6-single-agent-richness-and-provenance.md) —
  the record `FORMAT_VERSION` this change does not bump (no serde change).
- `docs/rules.md §10` — Architecture Invariants, amended by this ADR.
- Code: `crates/rlevo-core/src/environment.rs` (`SnapshotBase`,
  `SnapshotMetadata`, `Snapshot::metadata`), `crates/rlevo-core/src/evaluation.rs`
  (`BenchStep`, the metadata-dropping consumer), `crates/rlevo-environments/src/locomotion/common.rs`
  (`LocomotionSnapshot` alias), `crates/rlevo-environments/src/box2d/lunar_lander/snapshot.rs`
  (`LunarLanderSnapshot` alias), `crates/rlevo-environments/src/wrappers/time_limit.rs:91`
  (the `SnapshotBase`-bound `TimeLimit` impl this unblocks).
