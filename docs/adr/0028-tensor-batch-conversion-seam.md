---
project: rlevo
status: active
type: decision
date: 2026-07-05
tags: [adr, decision, tensor-convertible, batch-conversion, rlevo-core, burn]
---

# ADR 0028: Host-row primitive on `TensorConvertible` for single-upload batching

## Status

**Accepted (2026-07-05).** Filed under issue #195 (cross-crate
`TensorConvertible` batch seam + `unwrap_or_default` audit). Decides the API
shape (spec Options A/B/C) for a batch/slice-producing conversion seam on
`rlevo-core`'s `TensorConvertible` trait, and records the Burn reuse-or-reject
verdict.

**Chosen shape:** a refinement of spec **Option B** — make the host-only,
flat-`f32` **row-writer the required primitive** of `TensorConvertible`
(`row_shape()` + `write_host_row()`), *derive* `to_tensor` from it so the
single-item and batch layouts share one definition, and add one free function
`stack_to_tensor` in `rlevo-core` that stages an entire batch on the host and
issues a **single** `Tensor::from_data` upload. The refinement over the spec's
Option B: only the **forward** path is made required; `from_tensor` keeps its
current hand-written contract and the inverse batch decode (`from_host_row` /
`from_tensor_batch`) is deferred as additive future work, because no consumer
(the `memory.rs` migration target) decodes batches. This is a **breaking**
trait change that migrates all ~27 existing impls in the same PR — acceptable in
alpha, where trait/API stability is the declared focus.

## Context

`TensorConvertible<R, B>::to_tensor(&self, device) -> Tensor<B, R>` converts one
item at a time. Every replay-buffer and (future) population path uploads
observations and actions **one host↔device round-trip per element** and then
stacks. The canonical site is
`crates/rlevo-reinforcement-learning/src/memory.rs`
(`PrioritizedExperienceReplay::sample_batch`, pre-0028 lines 388-417): it maps
`to_tensor` over the sampled batch (N uploads) before `Tensor::stack`. A batch
seam that stages the whole batch on the host and uploads once is a per-step
cost paid by every algorithm (DQN, C51, DDPG, TD3, SAC, QR-DQN, PPO) at once.

The P1 half of #195 (`unwrap_or_default()` masking tensor-transfer failures) was
audited separately and found **clean** in `rlevo-environments` and
`rlevo-reinforcement-learning` production code; it is handled by a
`docs/rules.md` convention bullet plus a CI grep guard (spec decision D1) and is
not this ADR's subject.

### Burn reuse-or-reject (mandatory pre-decision audit)

Per the project convention (audit Burn built-ins before hand-rolling), the
question "does Burn 0.21 already ship an assemble-on-host, upload-once path?"
was researched to completion (maintainer-vault research note,
`research/2026-07-05-burn-batched-upload.md`). Verdict: **reject the premise;
build the helper.**

- `TensorData` has **no** host-side `stack`/`cat`/`concat`/`merge` — only
  construction, access, and dtype conversion methods.
- `Tensor::stack` is `unsqueeze_dim` + `cat`, executed purely on-device. It does
  not incur extra host transfers, but it also does not remove the N transfers
  the per-element `to_tensor` loop already paid.
- Burn's own canonical batcher (`Batcher` trait, MNIST example) does exactly the
  N-transfers-then-`cat` pattern — structurally identical to today's
  `memory.rs`. It is **not** a single-upload path.
- `Tensor::from_data(TensorData::new(flat_vec, [N, d0..]), device)` **is** a
  supported, documented single-upload idiom — Burn simply does not use it in its
  examples. Building it as a hand-rolled helper is additive, not redundant.

The decisive architectural implication: a helper that truly achieves one upload
**cannot** be built on top of `to_tensor`, because each `to_tensor` call is
itself a transfer. The helper needs a **host-only per-item row seam** (flat
`f32` payload + per-item shape) so it can flatten N rows into one `Vec<f32>`
before ever touching the device. A helper that internally still calls
`to_tensor` per element and then `Tensor::stack` merely re-derives Burn's
existing idiom and buys nothing.

### What every impl already does

All ~27 production `TensorConvertible` impls already round-trip through a flat
`f32` row:

- rank-1 impls (cartpole, acrobot, mountain_car(_continuous), pendulum, k_armed,
  contextual bandit, santa_fe_ant, lunar_lander observation, the action impls,
  reward) build via `Tensor::from_floats(array)`;
- the two rank-3 impls (`pixel_grid.rs`, `grids/core/observation.rs`)
  **already** hand-roll a flatten-to-`Vec<f32>` plus an explicit shape and call
  `TensorData::new(flat, shape)` → `Tensor::from_data`.

The row-writer model is therefore not a new concept imposed on impls; it is the
one thing they all already do, lifted into a single named place. Two impls carry
type-level const params (`ContextualBanditObservation<C>`,
`KArmedBanditAction<K>`); `row_shape()` as an associated fn returns `[C]` / `[K]`
for those without difficulty.

## Decision

### 1. The row-writer becomes the required primitive; `to_tensor` is derived

```rust
pub trait TensorConvertible<const R: usize, B: Backend>: Sized {
    /// Per-item row shape, e.g. `[4]` (cartpole) or `[H, W, C]` (pixels).
    /// Required — a flat length alone cannot reshape a rank-R row.
    fn row_shape() -> [usize; R];

    /// Append this item's row-major `f32` payload. Single source of truth for
    /// the item's layout; must push exactly `row_shape().iter().product()`
    /// elements.
    fn write_host_row(&self, buf: &mut Vec<f32>);

    /// Derived: one row + reshape. Do NOT override — overriding reintroduces the
    /// single-item / batch layout-drift hazard this design exists to remove.
    fn to_tensor(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, R> {
        let row = Self::row_shape();
        let mut buf = Vec::with_capacity(row.iter().product());
        self.write_host_row(&mut buf);
        debug_assert_eq!(buf.len(), row.iter().product::<usize>());
        Tensor::from_data(TensorData::new(buf, row), device)
    }

    /// Reconstruct a value from a single-item tensor. Contract unchanged from
    /// the pre-0028 trait; still hand-written per impl (shape validation +
    /// de-normalisation live here).
    fn from_tensor(tensor: Tensor<B, R>) -> Result<Self, TensorConversionError>;
}
```

Required surface after this ADR: `row_shape`, `write_host_row`, `from_tensor`.
`to_tensor` becomes a provided (default) method. `from_tensor` is deliberately
**not** made derived — decoders carry per-impl shape checks and de-normalisation
(e.g. pixel `b/255.0` inverse), and no consumer needs a derived inverse.

### 2. The single-upload helper (single `BR = R + 1` chokepoint)

```rust
/// Stage `items` as one host buffer and upload once. Batched rank-bumping
/// analogue of `to_tensor`, for `[N, row_shape..]` observation / action batches.
pub fn stack_to_tensor<const R: usize, const BR: usize, T, B>(
    items: &[T],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, BR>
where
    T: TensorConvertible<R, B>,
    B: Backend,
{
    assert_eq!(BR, R + 1, "batched rank BR must equal row rank R + 1");
    let row = T::row_shape();
    let row_len: usize = row.iter().product();
    let mut buf = Vec::with_capacity(items.len() * row_len);
    for item in items {
        item.write_host_row(&mut buf);
    }
    debug_assert_eq!(buf.len(), items.len() * row_len);
    let mut shape = [0usize; BR];
    shape[0] = items.len();
    shape[1..].copy_from_slice(&row); // sound: BR == R + 1
    Tensor::from_data(TensorData::new(buf, shape), device)
}
```

`R + 1` is not expressible in stable Rust types, so the batch rank rides a second
const param (`BR`) with a runtime `assert_eq!`. That assert lives at **one
library chokepoint** — inside `stack_to_tensor` — matching the "resolve at one
chokepoint" idiom.

`memory.rs` **keeps** its two pre-existing caller-facing asserts (`BD == D + 1`,
`BAD == AD + 1`) at the top of `sample_batch` rather than dropping them as
originally proposed: the `InsufficientData` early return sits between those
asserts and the first `stack_to_tensor` call, so removing them would let a
wrong-rank call paired with an oversized batch request return `Err` instead of
panicking — a silent behaviour change on that path, and one pinned by existing
`should_panic` regression tests. The helper's assert remains the single
*library* chokepoint; `sample_batch`'s asserts are a caller-facing precondition
check that fires before any early return.

### 3. Rewards stay rank-1 `[batch]` — do NOT route them through `stack_to_tensor`

Rewards are `TensorConvertible<1, B>` with `row_shape() == [1]`. Passing them to
`stack_to_tensor` would produce shape `[N, 1]` (`BR = 2`), but `memory.rs`
produces shape `[batch]` (rank 1) via `Tensor::cat` of `[1]` rows, and
bit-identical output is an acceptance criterion. Rewards therefore use the
row-writer **directly** and concatenate via `from_floats`, preserving the rank
and value layout:

```rust
let mut rewards_buf = Vec::with_capacity(batch_size);
for r in &rewards_vec { r.write_host_row(&mut rewards_buf); }
let rewards: Tensor<B, 1> = Tensor::from_floats(rewards_buf.as_slice(), device);
```

`dones` is unchanged: a plain `Vec<f32>` → `Tensor::from_floats` → `[batch]`
(it is not a `TensorConvertible`). `stack_to_tensor` is for the rank-bumping
`[N, d0..]` case only (observations, actions, next-observations).

### 4. Do not generalise over tensor `Kind` now

`TensorConvertible::to_tensor` returns `Tensor<B, R>`, which defaults to the
`Float` kind, so the trait is **already** `f32`/`Float`-locked. None of the ~27
impls produce `Int`/`Bool` output: the `Tensor<B, 1, Int>` tensors in
`dqn`/`c51`/`qrdqn` are internal gather/argmax indices, not trait output. The
flat-`f32` row adds **no new constraint** — it inherits exactly what the trait
already imposes.

Generalising `TensorConvertible` over `Kind` (an `Int`/`Bool` observation path)
is therefore **declined** here: no impl consumes it today, and blocking the
batch seam on an unused feature is unwarranted. If an `Int`/`Bool` observation
path ever lands, it is the prerequisite and the batch seam is redesigned against
the generalised trait at that time. (Pre-existing and independent of this work:
`f32` represents integers exactly only to 2²⁴, so a large discrete
encoding-as-float is already lossy under the current trait.)

### 5. The inverse batch path is deferred, not designed-in

The spec's Option B also made `from_host_row` required (an inverse row seam) so
that `from_tensor_batch` could decode a batch row-wise. No consumer decodes
batches — `memory.rs` only assembles them. Making `from_host_row` required on
all ~27 impls now is migration cost for an unused capability. It is deferred: it
can be added later as a **purely additive** pair (`from_host_row` +
`from_tensor_batch`) without touching the forward primitive.

## Consequences

### Positive

- **One upload, not N.** `stack_to_tensor` issues a single `from_data` for the
  whole batch — the actual goal of #195 P2, achievable only through a host-only
  row seam (Burn ships no built-in for it).
- **One layout definition.** `to_tensor` is derived from `write_host_row`, so the
  single-item and batch layouts cannot silently drift. An additive second trait
  (Option A) would leave two independent layout definitions that can diverge.
- **The `R + 1` assert lives at one library chokepoint** (the helper);
  `sample_batch`'s retained asserts are a caller-facing precondition, not a
  second layout definition.
- **Rests on what impls already do** — every impl already produces a flat `f32`
  row; the two rank-3 impls already flatten-plus-shape by hand. The migration
  lifts existing code into one place rather than inventing a new obligation.

### Negative / costs

- **Breaking trait change; wide migration.** All ~27 `TensorConvertible` impls
  are edited in the same PR: each replaces its `to_tensor` body with a
  `write_host_row` body (the flatten it already contains) and adds a trivial
  `row_shape()`; `from_tensor` is untouched. Mechanical but wide. Justified by
  alpha-stage API-stability focus and by the fact that even an "additive" new
  trait (Option A) would force the same impls anyway (see Alternatives).
- **Public trait shape is a one-way door.** Flagged for scrutiny; acceptable
  because there are no external consumers in alpha, and the alternative (two
  traits) bakes in a permanent drift hazard rather than a one-time migration.
- **`to_tensor` is a default method and thus overridable.** Drift prevention is a
  documented convention ("do NOT override"), not a compiler-enforced seal; the
  correct default plus the `debug_assert` length check make an errant override
  the only failure mode, and it is loud in debug builds.

### Neutral

- No serde / record-schema change (perf + trait change only).
- Reuses `rlevo-core`'s existing Burn dependency; `stack_to_tensor` adds no new
  dependency.

## Alternatives considered

- **Option A — additive `BatchTensorConvertible<R, BR, B>` second trait.** Leaves
  `TensorConvertible` untouched; a blanket default over `TensorConvertible`
  reproduces today's per-element stack, and impls opt into a fused override.
  Rejected. The blanket default that delegates to `to_tensor` does **not** achieve
  one upload (each `to_tensor` is a transfer) — it re-derives Burn's own idiom and
  buys nothing. To achieve one upload, each impl must hand-roll a parallel flatten
  in its override, which can silently drift from `to_tensor`'s layout — the exact
  hazard this ADR removes. And the "additive, no migration" selling point is
  illusory for the types that matter: `sample_batch`'s bounds must move to the
  row-capable trait regardless, so every batched observation/action type needs the
  new impl anyway. Option A therefore buys a second layout definition and a drift
  hazard for **no** reduction in the migration of the consuming types.

- **Option B as specified (inverse row seam required).** Same forward design as
  chosen, but also makes `from_host_row` required on all ~27 impls. Rejected in
  favour of the forward-only refinement: no consumer decodes batches, so requiring
  the inverse now is migration cost for an unused capability. It is deferred as a
  purely additive future pair (§5).

- **Option C — default batch method on `TensorConvertible` itself.** Folds the
  `R + 1` problem and its assert into the existing single-item trait signature,
  widening every impl's const-param surface and the trait's blast radius. It also
  returns a device tensor rather than exposing a host-only seam, so — like Option
  A's delegating default — it cannot achieve a single upload. Least attractive;
  rejected.

- **Generalise `TensorConvertible` over tensor `Kind`.** Deferred, not chosen
  (§4): the trait is already `Float`-locked, no impl needs `Int`/`Bool`, and the
  batch seam should not block on an unconsumed feature.

## References

- Issue #195 — cross-crate parent (batch seam + `unwrap_or_default` audit).
- ADR [0023](0023-objective-sense-and-maximize-convention.md),
  [0026](0026-shared-config-validation-convention.md),
  [0027](0027-bounds-newtype-for-closed-ranges.md) — the "small typed primitive /
  resolve at one chokepoint" pattern this ADR follows.
- Code: `crates/rlevo-core/src/base.rs` (the trait + `stack_to_tensor`);
  `crates/rlevo-reinforcement-learning/src/memory.rs` (`sample_batch`, the first
  consumer); `crates/rlevo-environments/src/pixel_grid.rs`,
  `crates/rlevo-environments/src/grids/core/observation.rs` (the two rank-3
  impls whose hand-rolled flatten the row-writer model generalises).
- Maintainer-vault working notes (not in this repo): spec
  `specs/2026-07-05-tensor-convertible-batch-seam/spec.md`, research
  `research/2026-07-05-burn-batched-upload.md`.
