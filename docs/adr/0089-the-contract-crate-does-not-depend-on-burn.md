---
project: rlevo
status: active
type: decision
date: 2026-10-18
tags: [adr, decision, architecture, rlevo-core, burn, tensor-convertible, host-row, crate-boundaries, breaking-change]
---

# ADR 0089: The contract crate does not depend on Burn

## Status

**Accepted (2026-10-18).**

Supersedes these clauses. The rest of each ADR stands.

- ADR [0028](0028-tensor-batch-conversion-seam.md):
  - Decision 2, the `stack_to_tensor` helper and its `BR = R + 1` chokepoint;
  - Decision 5, whose deferral of a decode method becomes a deletion of the
    one that existed.

  Decision 1 (the row writer is the primitive, and `to_tensor` is derived
  from it) and Decision 3 (rewards stay a rank-1 batch) stand.
- ADR [0052](0052-hostrow-supertrait-splits-layout-from-backend.md):
  - Decision 4, that `HostRow` carries no `Sized`;
  - Decision 5, where to put a future decoder;
  - Decision 6, `stack_to_tensor` relaxed to `HostRow`;
  - the deferred half of Decision 8, answered here in the other direction.

  Decisions 1, 2, 3 and 7 stand: the split itself, how the ambiguity
  resolves, the one-rank hazard, and the migration note.
- ADR [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md)
  Decision 2, the two-clause contract on `TensorConvertible` (tensor-image
  fidelity and no fabrication). Both clauses are about decoding, and
  nothing decodes. Decision 1, `agent_direction: Option<Direction>`,
  stands. Its source of `None` was the decoder this ADR deletes (see
  Consequences).
- ADR [0080](0080-library-crates-select-no-burn-backend.md) is extended, not
  superseded: `rlevo-core`'s Burn dependency is optional, and when it is on
  it resolves at `[std]` as that ADR requires.

## Context

`rlevo-core` is the contract crate: `Environment`, the observation and action
traits, `EpisodeGuard`, the evaluation drive seam. An environment author with
no learner depends on it and nothing else. Every such author compiled Burn
all the same. `cargo tree -p rlevo-core -e normal` listed 110
packages, 8 of them `burn*`, at `45a47b3` (the measurement method is under
References).

Burn appeared in one file of the crate, `base.rs`, and in two items there.
One was `TensorConvertible<R, B>`, with a derived `to_tensor` and a
hand-written `from_tensor`. The other was the free function
`stack_to_tensor`. Everything else in the contract was already Burn-free.

The decode half had no consumer. Counted at `45a47b3` by a scan that drops
`#[cfg(test)]` items and comment lines:

- 30 library implementations of `from_tensor` (29 in `rlevo-environments`,
  one in `rlevo-test-support`);
- 65 constructions of `TensorConversionError`, every one inside a
  `from_tensor` body;
- no call of `from_tensor` in library code, benches, examples or tests
  outside the unit tests;
- no caller of `stack_to_tensor` anywhere; its own documentation said so.

The decoders the workspace does use are `DiscreteAction::from_index` and
`ContinuousAction::from_slice`. ADR 0028 Decision 5 had deferred a required
`from_host_row` for the same reason: no consumer decodes.

Three test modules in `rlevo-reinforcement-learning` (DDPG, TD3, SAC) did call
`from_tensor`, as a constructor. Their `make_obs` helper built a
`MaskObservation` through it because the struct's field is private. That is
a fixture's need, not a decoder's.

Two facts about `HostRow` came out of the same count:

- **`row_shape()` was a second definition of `shape()`.** All 18 observation
  types that implement `HostRow` implement it at the same rank as
  `Observation`, with `row_shape()` returning what `shape()` returns. Nothing
  checked that the two agreed. RL library code never called `row_shape()`:
  it sizes its buffers with `O::shape()` (10 sites) and fills them with
  `write_host_row`. ADR 0052 Decision 8 saw the duplication and deferred it.
- **Eleven action types and one state type implemented `HostRow` for no
  consumer.** Nothing converts an action or a state to a tensor. Actions
  reach a network through `DiscreteAction::to_index` and
  `ContinuousAction::as_slice`, and every RL bound on `TensorConvertible` is
  on an observation.

## Decision

### 1. `rlevo-core` depends on Burn only under its `burn` feature

`burn` is an optional dependency of `rlevo-core`, and the `burn` feature turns
on `TensorConvertible` and nothing else. It is off by default.
`rlevo-reinforcement-learning` and the `rlevo` umbrella request it. When it
is on, Burn resolves at `[std]`, unchanged from ADR 0080.

### 2. `TensorConvertible` is encode-only and implemented for every `HostRow`

```rust
#[cfg(feature = "burn")]
pub trait TensorConvertible<const R: usize, B: Backend>: HostRow<R> {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, R> { /* derived */ }
}

#[cfg(feature = "burn")]
impl<const R: usize, B: Backend, T: HostRow<R>> TensorConvertible<R, B> for T {}
```

No type writes the trait by hand. A hand-written impl is now a coherence
error (E0119), which also retires the "do not override `to_tensor`" note:
nobody can. The RL crate's 39 bounds on the trait and its 20 calls of
`to_tensor` are unchanged.

### 3. The decode path is deleted

`TensorConvertible::from_tensor`, `TensorConversionError` and `stack_to_tensor`
are removed, along with the environments' `argmax` helper that only the action
decoders called. No decode trait is added.

A decoder returns with the first code that calls one. As ADR 0028 Decision 5
already allowed, that is an additive change. It goes beside the row writer it
inverts, with `where Self: Sized` on the method rather than on the trait, as
ADR 0052 Decision 5 advised.

### 4. `rlevo-environments` does not depend on Burn

Neither `[dependencies]` nor `[dev-dependencies]` names it. An environment
writes `HostRow` and no Burn code. Its tests assert the row that
`write_host_row` produces, including its length. The upload is core's, and it
is tested once, in core.

Where a deleted round-trip test was the only witness of an encoding, it is
rewritten against the row:

- the grid stride and channel order;
- the mission byte in every cell;
- the all-zero row of an unseen view;
- the pixel and `CarRacing` `/255` scaling;
- the contextual bandit's one-hot;
- the Santa Fe food flag;
- the 8 and 24 values of the two Box2D observations.

### 5. A host row belongs to an observation, and its shape is the observation's

```rust
pub trait HostRow<const R: usize>: Observation<R> {
    fn write_host_row(&self, buf: &mut Vec<f32>);
    fn row_is_finite(&self, scratch: &mut Vec<f32>) -> bool { /* provided */ }
}
```

- `row_shape()` is deleted. A row has `Self::shape()` elements per axis, and
  `write_host_row` pushes exactly their product.
- `Observation::shape()` is the row's extent for every type that writes one,
  and its documentation now says so.
- The 11 action impls and `KArmedBanditState`'s impl are deleted.

Requiring `Observation` brings `Clone`, so `HostRow` is now `Sized`. That
closes the unsized door ADR 0052 Decision 4 kept open, which had no
implementor.

ADR 0052 Decision 8 deferred `Observation<R>: HostRow<R>`, because that
direction would force a row layout on 12 observations that have none, and
that layout is a design question with learning consequences. This ADR takes
the other direction. A row writer is an observation, and an observation that
writes no row loses nothing.

### 6. CI keeps the core Burn-free

The `libraries-select-no-backend` job checks a crate whose `burn` dependency
is optional in both configurations: with the feature on, where `burn` must
resolve at `[std]`, and with it off, where the crate must still build and
document. A separate step fails if `rlevo-core` reaches any `burn*` package at
default features.

## Consequences

### Positive

- `cargo tree -p rlevo-core -e normal` lists 25 packages, against 110, and
  no `burn*`.
- The `consumer-core` probe builds an environment file against
  `rlevo-core` alone. It now reports 24 packages in its graph, against 108;
  0 `burn*` packages, against 8; and a cold build of 3.4 s, against 11.7 s.
  Both runs are one cold `dev` build each, on an Apple M2 Pro (12 cores,
  32 GiB), rustc 1.99.0, Burn 0.21.0, with the "before" run on the parent
  commit the same day.
- `rlevo-environments` at default features lists 75 packages, against 153,
  and no `burn*`. The harness, which reached Burn only through core, lists
  37 against 118 at default features.
- `rlevo` at default features is unchanged at 210 packages, because it
  selects a backend.
- An environment author implements `HostRow` (one method, plus an optional
  override) and writes no Burn code. The shape of the row is stated once.
- About 725 lines of decode bodies, their tests and a helper module go.

### Negative / costs

- **Breaking for direct users of `rlevo-core`:**
  - `from_tensor`, `TensorConversionError` and `stack_to_tensor` are gone.
  - A hand-written `TensorConvertible` impl no longer compiles; delete it.
  - Naming `TensorConvertible` now needs the `burn` feature. A crate that
    reaches it only through feature unification compiles in a workspace
    and fails alone. The per-library CI check is the guard.
  - `HostRow` on a type that is not an `Observation` no longer compiles,
    and `row_shape` is gone.
- An observation's row now has the observation's rank. A type that wanted a
  row at a different rank cannot have one. None of the 18 does.
- The only runtime check that `write_host_row` pushes the right count is the
  `debug_assert_eq!` in `to_tensor`, which environments no longer reach.
  Their tests assert the length instead.
- `agent_direction: None` on the two grid observations is no longer produced
  by any code in the workspace. It reaches the field only through the public
  field or `Deserialize`. Whether the field returns to `Direction` is a
  separate break, left to the next pruning of unused members.

### Neutral

- RL library source is unchanged. Its test fixtures lose their hand-written
  impls, and the three `make_obs` helpers call a test-only
  `MaskObservation::new`.
- `rlevo-environments` still reaches Burn under its `record` feature, through
  the harness's `record` dependency on `rlevo-evolution`. That edge is ADR
  0084's, not this one's.
- Packaging is unaffected. `cargo package --workspace --no-verify`, with the
  unpublished members excluded, packages all eight crates with the new
  feature requested by a sibling. This was measured before the change, on
  `45a47b3` with the manifests edited.

## Alternatives considered

- **A `FromHostRow` decode trait in place of `from_tensor`.** It would re-home
  a capability with no caller, for the reason ADR 0028 Decision 5 gave for not
  adding it. Rejected.
- **Moving `TensorConvertible` into `rlevo-reinforcement-learning`**, its only
  consumer, so that core has no feature at all. Rejected. `rlevo-hybrid`
  does not depend on the RL crate, so the first consumer outside RL (a world
  model in hybrid) would have to add that edge or move the trait back. The
  feature costs only the `cfg` gates and the CI check.
- **A free function `to_tensor(&x, device)` in place of the trait.** It would
  change 39 bound lines and 20 call sites for no behaviour. Rejected.
- **Keeping `row_shape` and pinning `row_shape() == shape()` in a test.** It
  derives nothing, and an outside implementor gets no check. Rejected.
- **Deleting `Observation::shape()` and `Action::shape()` and keeping
  `row_shape`.** It would break the four RL sites that size an action buffer
  with `A::shape()`, on actions that write no row. It would also change every
  environment author's `Observation` impl. Rejected.
- **Keeping Burn as a dev-dependency of `rlevo-environments`.** Twenty-nine
  files would keep testing an upload they no longer implement. Rejected.

## References

- `crates/rlevo-core/src/base.rs`: `HostRow`, `TensorConvertible`, the blanket
  impl, and the `to_tensor` tests behind `burn`.
- `crates/rlevo-core/Cargo.toml`: the optional dependency and the docs.rs
  feature list.
- `.github/workflows/crate-tests.yml`, job `libraries-select-no-backend`.
- The measurements:
  - Package counts are `cargo tree -p <crate> -e normal --prefix none -f '{p}'`,
    unique lines, the crate itself included, on Burn 0.21.0 and rustc 1.99.0.
  - The probe's figures are from `cargo xtask probe consumer-core`.
- ADRs [0028](0028-tensor-batch-conversion-seam.md),
  [0052](0052-hostrow-supertrait-splits-layout-from-backend.md),
  [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md),
  [0067](0067-non-finite-observations-are-dropped-at-replay-ingestion.md),
  [0080](0080-library-crates-select-no-burn-backend.md),
  [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md),
  [0084](0084-the-umbrella-carries-the-evaluation-harness.md).
