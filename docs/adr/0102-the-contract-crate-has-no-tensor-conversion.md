---
project: rlevo
status: active
type: decision
date: 2026-11-14
tags: [adr, decision, tensor-convertible, host-row, burn, feature-flags, rlevo-core, rlevo, public-surface, breaking-change]
---

# ADR 0102: The contract crate has no tensor conversion

## Status

**Accepted (2026-11-14).**

- Amends ADR [0089](0089-the-contract-crate-does-not-depend-on-burn.md)
  Decisions 1, 2 and 6. `rlevo-core` has no `burn` feature and no Burn
  dependency, `TensorConvertible` is deleted, and CI checks that the core
  reaches no Burn package with any feature on.
- Answers the question ADR [0101](0101-observation-carries-no-rank.md) left
  open: `TensorConvertible` does not keep a consumer.
- Leaves ADR 0089's other decisions as they are: conversion stays
  encode-only (Decision 3), and a host row belongs to an observation
  (Decision 5).

## Context

ADR 0089 made `rlevo-core` depend on Burn only under a `burn` feature, and
put one item behind it: `TensorConvertible`, implemented for every `HostRow`
by a blanket impl, whose provided `to_tensor` uploads one row as a tensor of
the observation's rank.

ADR 0101 moved every agent off it. Its Decision 4 stages a batch of rows into
one host buffer and uploads `[n, ..SHAPE]` once, through crate-private
functions in `rlevo-reinforcement-learning` (`upload_rows`, `upload_one`).
ADR 0101 kept the trait and left one question open, to be decided by the
count of outside-style callers: does it keep a consumer?

### Who calls it

On `main` at `96b2e943`, after ADR 0101 landed:

| Caller | Use |
|--------|-----|
| `rlevo-core`'s own tests | two tests of `to_tensor` itself |
| `staging_bench.rs` | the retired per-row round trip, kept as a baseline |
| `learn_step_bench.rs` | the same baseline, inside a learn step |
| `rlevo::prelude` | a re-export |

No library code in the workspace calls it. No probe and no seam of the
`rlevo-algo` template names it, and those stand for outside authors (an
environment author, a landscape author, a learner, two algorithm authors and
the template's three seams).

### What it costs

- It is the only reason `rlevo-core` has a `burn` feature. The feature takes
  three `cfg` lines in `base.rs`, a feature request in two manifests (the
  umbrella and the RL crate, which turned it on for its benches only), a
  `features` override in the `test-fast` tier, a second configuration in the
  `libraries-select-no-backend` CI job, and a paragraph in each of
  `docs/rules.md`, `CLAUDE.md`, the core README and the core's crate docs.
- It points the wrong way. It uploads one row at the observation's rank.
  The pattern ADR 0101 removed from 21 sites was exactly this call followed
  by an `unsqueeze`, and the right path, a staged batch uploaded once, is
  crate-private. A prelude item that does the first invites it back.

### What it gives

- An author who builds a Burn agent on `rlevo-core` alone gets a conversion
  without writing one. The conversion is four lines: write the row into a
  `Vec<f32>`, then `Tensor::from_data(TensorData::new(row, O::SHAPE),
  device)`.
- Its `to_tensor::<R>` asserts `SHAPE.len() == R` when it is built. A
  hand-written upload checks the same thing at run time, in
  `TensorData::new` and `Tensor::from_data`, or at build time with the same
  `const` block, as the agents and the documented example do.

## Decision

### 1. `TensorConvertible` is deleted

The trait, its blanket impl and its two tests are removed from
`rlevo-core`, and `rlevo::prelude` no longer re-exports it. No replacement
is added to `rlevo-core`: ADR
[0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md) asks a
public item for a consumer, and none exists.

### 2. `rlevo-core` does not depend on Burn

The optional `burn` dependency and the `burn` feature are removed, and so is
the core's Burn dev-dependency, which only those tests used. The core builds,
tests and documents with no Burn package in its graph. The umbrella and the
RL crate stop requesting the feature, and the `test-fast` tier runs the core
at its default features.

### 3. The upload is documented where Burn is

The `rlevo` crate docs gain a section, "Uploading observations", with a
doctest: a function generic over `O: HostRow` and the batch rank, which
asserts `O::SHAPE.len() + 1 == BOR` in a `const` block, stages every row into
one buffer and uploads `[n, ..SHAPE]` once. One row is a batch of one. The
`HostRow` documentation points to it.

The two benches keep their retired baseline with a bench-local
`row_to_tensor`, the old `to_tensor` body, so the staging delta stays
measurable.

### 4. CI checks the core has no Burn at all

The `libraries-select-no-backend` job no longer treats an optional `burn`
dependency as a second configuration; no crate has one. Its last step runs
`cargo tree -p rlevo-core --all-features` and fails if any `burn*` package
appears.

## Consequences

### Positive

- `rlevo-core` is Burn-free in every configuration, not only by default. An
  environment author's dependency graph cannot reach Burn through the core.
- One prelude item, one feature and one CI configuration fewer, and no
  public path that uploads a single row.

### Negative

- A breaking change for an outside author who called `to_tensor` or
  requested `rlevo-core/burn`. Both are named in the CHANGELOG with the
  replacement. It ships in the same release as ADRs 0100 and 0101, so
  implementors meet one break.
- An outside Burn author writes the upload, about four lines, or copies the
  documented function.

### Neutral

- `HostRow` is unchanged, and the row layout stays with the observation.
- ADR 0089 Decision 4, `rlevo-environments` not depending on Burn, is
  unchanged and was already true.

### Surface delta (ADR 0081)

Removed: `rlevo_core::base::TensorConvertible` and its method `to_tensor`,
and the re-export in `rlevo::prelude`. Removed feature: `rlevo-core/burn`.
Added: nothing public. `tools/public-surface.sh` at `96b2e943` and with this
change shows exactly these three lines removed (the trait and its method in
`rlevo-core`, the `use` in `rlevo`) and none added.

## Alternatives considered

- **Keep it** (ADR 0101's choice). Keeps the one-row upload in the prelude
  with no caller, and the feature and its CI configuration with it.
- **Make the batch upload public in `rlevo-core`**, as a function over
  `&[O]`. The right shape, but it needs Burn in the core, so the feature
  stays, and nothing outside the RL crate would call it today.
- **Make `upload_rows` public in `rlevo-reinforcement-learning`.** An outside
  agent author could call it, but none does, and ADR 0081 asks for the
  consumer first. It can be made public when one appears, as an addition.

## References

- ADR [0028](0028-tensor-batch-conversion-seam.md): the batch-upload
  chokepoint ADR 0101 restored.
- ADR [0052](0052-hostrow-supertrait-splits-layout-from-backend.md): the split
  of the row layout (`HostRow`) from the upload.
- ADR [0080](0080-library-crates-select-no-burn-backend.md): no library crate
  selects a backend.
- ADR [0089](0089-the-contract-crate-does-not-depend-on-burn.md): the
  feature this ADR removes.
- ADR [0101](0101-observation-carries-no-rank.md) Decision 4: the staging
  functions every agent uploads through.
- `crates/rlevo-core/src/base.rs` (`HostRow`), `crates/rlevo/src/lib.rs`
  ("Uploading observations"),
  `crates/rlevo-reinforcement-learning/src/algorithms/shared.rs`,
  `.github/workflows/crate-tests.yml` (`libraries-select-no-backend`).
