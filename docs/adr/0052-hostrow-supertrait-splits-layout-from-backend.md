---
project: rlevo
status: active
type: decision
date: 2026-07-19
tags: [adr, decision, tensor-convertible, host-row, rlevo-core, burn, type-safety]
---

# ADR 0052: `HostRow<R>` supertrait splits row layout from the backend

## Status

**Accepted (2026-07-19).** **Extends ADR
[0028](0028-tensor-batch-conversion-seam.md); does not supersede it.** Every
substantive 0028 decision survives verbatim: the host-only flat-`f32` row-writer
is still the required primitive, `to_tensor` is still derived from it and still
must not be overridden, `from_tensor` is still hand-written per impl, and rewards
still bypass `stack_to_tensor` in favour of the row-writer plus one `from_floats`
(0028 §3). What changes is *where* the two required row methods live, not what
they mean. 0028's batching rationale — the Burn reuse-or-reject audit, the
single-upload argument, the `BR = R + 1` chokepoint — is load-bearing and
unaltered; marking it superseded would falsely invalidate it.

Splits `TensorConvertible<R, B>` into a backend-independent supertrait
`HostRow<R>` and a slimmed device-facing `TensorConvertible<R, B>: HostRow<R>`.
Breaking trait change; every `TensorConvertible` impl in the workspace was split
in the same PR — 35 impl blocks in code plus 8 rustdoc/doctest examples, across 6
crates (`rlevo-core`, `rlevo-environments`, `rlevo-reinforcement-learning`,
`rlevo-test-support`, `rlevo-examples`, `rlevo`).

## Context

### The contract that the type system did not enforce

0028 gave `write_host_row` an explicitly backend-independent contract. Its
rustdoc required implementors to

> Push **plain `f32`** — do *not* pre-convert to `B::FloatElem`.
> `TensorData::new` performs the element-type conversion at upload time.

That is a real, load-bearing invariant: it is what lets `stack_to_tensor` stage N
rows from N different sources into one contiguous `Vec<f32>` and issue a single
upload. But under 0028's trait shape it was **prose only**. `write_host_row`
lived on a trait parameterised by `B`, so nothing stopped an implementor from
writing a backend-specialised body — rounding differently under one backend,
emitting a different element count, or branching on `B` outright. The compiler
had no opinion. The contract said "this method does not depend on `B`" while the
signature said "this method may depend on `B`", and only the docs bridged the
gap.

### Where that bit: the dual-bound turbofish

The off-policy agents (`dqn`, `c51`, `qrdqn`, `ddpg`, `td3`, `sac`) stage replay
batches on the host, and their observation type carries an autodiff-aware pair of
bounds:

```rust
O: TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>
```

Calling `t.obs.write_host_row(&mut buf)` under those bounds does not compile.
Two distinct trait obligations apply to the same receiver, differing only in a
parameter the method never mentions, and inference cannot choose between them —
**E0284, "type annotations needed"** (reproduced on rustc 1.97.0). The six call
sites therefore carried a qualified-path workaround:

```rust
// The turbofish is required: `O`'s dual `TensorConvertible` bound is ambiguous.
<O as TensorConvertible<DO, B::InnerBackend>>::write_host_row(&t.obs, &mut obs_flat);
```

This line names a backend that is **provably irrelevant to its own result**. Every
reader must independently re-derive that irrelevance in order to trust the code,
and the derivation depends on the prose contract above rather than on anything
checkable. Two engineers, working independently, picked *different* irrelevant
backends for structurally identical staging loops — which is the diagnostic
symptom: when a required annotation has no correct answer, authors supply
arbitrary ones.

The failure mode this sets up is the serious part. Had anyone ever written a
backend-specialised `write_host_row` — permitted by the signature, forbidden only
by docs — then `<O as TensorConvertible<DO, B>>::write_host_row` and
`<O as TensorConvertible<DO, B::InnerBackend>>::write_host_row` would push
**different bytes into the staging buffer**, silently. 0028's length
`debug_assert` would not catch it: both bodies write a correctly-sized row, so
the assert passes and the model trains on subtly wrong data. The guard that
exists checks the row's *length*, and the defect is in its *contents*.

### What the weak argument would have been

An earlier framing justified this split on ergonomics: `stack_to_tensor` had zero
call sites, and the dual-bound friction was said to explain why. That argument was
reviewed and **rejected**, and is recorded here so it is not revived. It is
post-hoc storytelling. `stack_to_tensor`'s only nominal consumer was
`memory.rs::sample_batch`, which was dead code (#188) and was deliberately retired
by ADR [0050](0050-replay-strategy-seam.md). Its call count is explained by that
retirement, not by friction. The justification for this ADR is the type-system
one above, and it stands on its own.

## Decision

### 1. Split the trait; the layout half loses `B`

```rust
/// Host-side, backend-independent row serialization.
pub trait HostRow<const R: usize> {
    fn row_shape() -> [usize; R];
    fn write_host_row(&self, buf: &mut Vec<f32>);
}

pub trait TensorConvertible<const R: usize, B: Backend>: HostRow<R> + Sized {
    fn to_tensor(&self, device: &<B as BackendTypes>::Device) -> Tensor<B, R> {
        /* unchanged from ADR 0028 */
    }
    fn from_tensor(tensor: Tensor<B, R>) -> Result<Self, TensorConversionError>;
}
```

**The backend-independence of `write_host_row` moves from prose to the type
system.** `HostRow` has no `B` in scope, so a backend-specialised row-writer is
not merely discouraged — it is **unrepresentable**. The `debug_assert` that could
not catch a wrong-contents row is now guarding a case that cannot arise.

This is the project's recurring invariants-in-types move, applied to a trait
boundary rather than a value: ADR [0027](0027-bounds-newtype-for-closed-ranges.md)
(`Bounds` makes `lo > hi` unconstructible), ADR
[0031](0031-probability-rate-newtypes.md) (rate newtypes make an out-of-range
probability unconstructible), ADR
[0046](0046-slot-newtype-replaces-option-take-around-learn-step.md) (`Slot` makes
the `None` window unrepresentable). In each case a documented rule became a
compile-time one and a class of silent-wrong-answer bugs stopped existing.

### 2. The ambiguity resolves by *shared supertrait obligation*, not relocation

This is the mechanism, and it is worth stating precisely because the obvious
worry — "you moved the ambiguity from `B` to `R`" — is wrong.

`O: TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>` elaborates
to two supertrait obligations: `O: HostRow<DO>` and `O: HostRow<DO>`. They are the
**same obligation**. `DO` is fixed by the enclosing generic signature; `B` has
been projected away entirely, because `HostRow` cannot mention it. There is
exactly one candidate, so `t.obs.write_host_row(&mut buf)` resolves with no
annotation. All six turbofish workarounds were deleted, and the comment
explaining them deleted with them.

The ambiguity is not relocated to `R` for the reason that `R` was never ambiguous
in these positions: at every call site it is either a concrete literal or a const
parameter already pinned by the function's own bounds.

### 3. Residual hazard: one type at two ranks

`HostRow<R>` is still generic in `R`, so the ambiguity *class* is not eliminated —
it is narrowed to a case that does not occur. A type implementing `HostRow` at two
different ranks reintroduces exactly the old problem: an unqualified
`row_shape()` / `write_host_row()` call on a concrete value of that type becomes
ambiguous again (**E0283**, with an accompanying **E0284** on the const parameter;
both reproduced on rustc 1.97.0).

**There are zero such types in the workspace today**, and the accepted mitigation
is a documented trait invariant, recorded in `docs/rules.md`'s invariant table and
in `HostRow`'s rustdoc:

> A domain type implements `HostRow` at exactly **one** rank `R`.

This is a genuine, if modest, cost, and it is accepted on two grounds. First, the
failure mode is a **loud compile error, never silent** — the opposite of the
divergent-bytes hazard §Context describes, which is precisely the trade this ADR
is making. Second, should a two-rank type ever be wanted, the fix is the *same
qualified-path syntax that existed before this ADR*
(`<T as HostRow<3>>::write_host_row(..)`). The downside case is therefore no worse
than the status quo ante, while the ordinary case is strictly better.

### 4. `HostRow` deliberately carries no `Sized`, and is deliberately not `dyn`-compatible

`Sized` stays on `TensorConvertible` (where `from_tensor -> Result<Self, _>`
requires it) and is deliberately **not** placed on `HostRow`. Neither `HostRow`
method needs it: `row_shape` returns an array of `usize`, and `write_host_row`
takes `&self`. Imposing an unnecessary bound would exclude unsized implementors
for no benefit.

`HostRow` is nonetheless **not** `dyn`-compatible, because `row_shape()` is an
associated function with no `self` receiver. This is deliberate and is not a
regression — `TensorConvertible` was not `dyn`-compatible before the split either,
for the same reason. It is also a **two-way door**: adding `where Self: Sized` to
`row_shape` alone would exclude it from the vtable and make `dyn HostRow<R>`
legal, purely additively and without touching any impl. No consumer needs runtime
heterogeneity over row types today (batch staging is monomorphic in `T`), so the
door is left closed.

### 5. Interaction with #201 — read this before implementing the inverse seam

Issue #201 (the inverse batch-decode seam, deferred by 0028 §5) proposes adding
`from_host_row(row: &[f32]) -> Self` "alongside `write_host_row`", paired with a
`from_tensor_batch` free function.

**The split makes #201 cleaner.** Under 0028's single trait, both new methods
would have landed on a `B`-parameterised trait, and `from_host_row` — which
decodes a plain `&[f32]` and touches no device — would have inherited the exact
irrelevant-backend ambiguity this ADR just removed. Post-split the assignment is
forced and obvious: **`from_host_row` belongs on `HostRow`** (backend-independent,
mirroring `write_host_row` on the same trait, so encode and decode share one
home), and **`from_tensor_batch` belongs beside `to_tensor`** (it consumes a
`Tensor<B, BR>` and genuinely needs `B`).

**But there is a trap, and it is the reason this section is normative rather than
informational.** `from_host_row(row: &[f32]) -> Self` returns `Self` by value, so
it requires `Sized` — the bound §4 just declined to put on `HostRow`. The
path of least resistance for whoever implements #201 is to add `Sized` back to the
trait, which compiles, passes every test, and **silently undoes §4's reasoning**
without anyone noticing.

The required approach: put the bound on **that one method**, not on the trait.

```rust
pub trait HostRow<const R: usize> {
    fn row_shape() -> [usize; R];
    fn write_host_row(&self, buf: &mut Vec<f32>);

    // #201, when a consumer exists. `where Self: Sized` goes HERE.
    // Do NOT move it to the trait — see ADR 0052 §4.
    fn from_host_row(row: &[f32]) -> Self
    where
        Self: Sized;
}
```

This keeps unsized implementors viable, and (as a bonus) `where Self: Sized` on
both `row_shape` and `from_host_row` is exactly what §4's dyn-compatibility door
requires — so #201 done this way opens that door rather than welding it shut.

### 6. `stack_to_tensor` relaxes to `T: HostRow<R>`

```rust
pub fn stack_to_tensor<const R: usize, const BR: usize, T, B>(
    items: &[T],
    device: &<B as BackendTypes>::Device,
) -> Tensor<B, BR>
where
    T: HostRow<R>,   // was: T: TensorConvertible<R, B>
    B: Backend,
{ /* body unchanged */ }
```

The body was already backend-independent up to the final `Tensor::from_data`; the
old bound over-constrained it. `B` remains a parameter of the *function* (it names
the upload target) but is no longer demanded of the *row type*. A type can now be
staged into a batch without implementing `from_tensor` at all — which is the
correct factoring, since staging never decodes. The `BR = R + 1` assert stays
where 0028 put it: one library chokepoint.

### 7. Migration: `E0207`

Every impl was split (35 in code, 8 more in rustdoc examples). The mechanical trap, hit immediately and
worth recording because the error message does not name the cause:

```rust
impl<B: Backend> HostRow<1> for CartPoleObservation { ... }
// error[E0207]: the type parameter `B` is not constrained by the
//               impl trait, self type, or predicates
```

`HostRow` does not mention `B`, so a `B` introduced by the impl header is
unconstrained. Each split must **drop `B` from the `HostRow` half**:

```rust
impl HostRow<1> for CartPoleObservation { /* row_shape, write_host_row */ }
impl<B: Backend> TensorConvertible<1, B> for CartPoleObservation { /* from_tensor */ }
```

`from_tensor` bodies were not touched. Verified on rustc 1.97.0.

### 8. `Observation<R>: HostRow<R>` is now expressible — deferred to a follow-up

A latent consequence worth recording: requiring every observation to be
row-writable was **not expressible** before this ADR. `Observation<R>` carries no
`B`, so it could not name `TensorConvertible<R, B>` as a supertrait without
growing a backend parameter that has no business on a domain trait.
`Observation<R>: HostRow<R>` is a well-formed bound today.

It is **deferred to a follow-up issue**, not adopted here, for a substantive
reason: 12 `Observation` types in the workspace (`FrozenLakeObservation`,
`TaxiObservation`, `CliffWalkingObservation`, `BlackjackObservation`,
`KArmedBanditObservation`, and assorted test/doc stubs) have no
`TensorConvertible` impl at all. Most are discrete/tabular, and choosing their row
layout — one-hot, ordinal index-as-float, or something else — is a **design
question with learning consequences**, not a mechanical migration. 0028 §4 already
notes that `f32` represents integers exactly only to 2²⁴, which bears directly on
the ordinal option. Forcing that decision as a side effect of a trait-split PR
would be the wrong sequencing.

One limitation to record so the follow-up does not over-promise: the supertrait
bound alone would **not** force `shape() == row_shape()`. They would remain two
independently-implemented functions that happen to describe the same value's
extent, and a mismatch would stay a runtime concern. Unifying them is a separate
question from requiring the bound.

## Consequences

### Positive

- **A backend-specialised `write_host_row` is unrepresentable**, not merely
  discouraged. The silent divergent-bytes failure mode described in §Context
  cannot occur.
- **Six qualified-path workarounds deleted**, each of which named a
  provably-irrelevant backend and obliged every reader to verify that
  irrelevance.
- **`stack_to_tensor` is bounded on what it actually uses.** Host-side staging no
  longer demands a device-facing decode impl.
- **`Observation<R>: HostRow<R>` becomes expressible** (§8), and **#201 gets a
  forced, clean method assignment** (§5).

### Negative / costs

- **Breaking public-trait change; 35 impl blocks across 6 crates.** Mechanical (split
  the impl block, drop `B` from the `HostRow` half) but wide, and it lands with
  the `E0207` trap of §7. Acceptable on the same alpha-stage grounds 0028 invoked.
- **Two impl blocks where there was one.** Slightly more ceremony per type; the
  compensating simplification is at every *call* site.
- **A new documented invariant to uphold** — one rank per type (§3) — enforced by
  convention, with a loud compile error as the backstop.
- **A trap for #201's implementer** (§5), mitigated only by this ADR and by the
  rustdoc note on `HostRow`.

### Neutral

- **No performance change whatsoever.** Not one byte moves differently, no upload
  count changes, no allocation changes. `to_tensor`'s derived body and
  `stack_to_tensor`'s staging loop are untouched; this is a type-safety and
  ergonomics change and nothing else. 0028's single-upload win is neither extended
  nor eroded.
- No serde / record-schema change; no persisted data format is affected.
- No new dependency; no change to `rlevo-core`'s Burn usage.

## Alternatives considered

- **Do nothing — keep the turbofish.** Zero migration cost, and the six call sites
  do compile and are correct *today*. Rejected: it leaves the backend-independence
  contract as unenforced prose, leaves every reader re-deriving the irrelevance of
  a named backend, and leaves the silent divergent-bytes hazard live. The
  observed fact that two engineers picked different irrelevant backends is
  evidence the annotation carries no meaning a reader can check.

- **An associated const `RANK` instead of a const parameter** (`trait HostRow {
  const RANK: usize; fn row_shape() -> [usize; Self::RANK]; }`), which would make
  the one-rank-per-type invariant of §3 a compiler-enforced consequence rather
  than a documented rule. **Dead on stable Rust**: `[usize; Self::RANK]` is a
  generic const operation and needs `generic_const_exprs`, still unstable
  (verified on rustc 1.97.0: *"error: generic parameters may not be used in const
  operations"*). Not rejected on design grounds — genuinely unavailable. Worth
  revisiting if that feature ever stabilises.

- **A free function `write_host_row_of::<R, T>(item, buf)` instead of a
  supertrait.** Would sidestep the dual-bound ambiguity at call sites by making
  `R` and `T` explicit. Rejected: it addresses the *symptom* (call-site
  annotation) and not the *cause* (the method's signature admits a backend it must
  not use). A free function still has to dispatch to some trait method, so a
  backend-specialised row-writer would remain writable, and callers would trade a
  turbofish for a differently-shaped turbofish.

- **Give `write_host_row` a `B`-mentioning signature** — e.g. take a
  `&B::Device`, or return `B::FloatElem`. **Actively harmful**, and rejected
  outright: it directly contradicts the method's reason for existing. The whole
  value of a host-side row seam (0028's central finding — a true single-upload
  path *cannot* be built on a per-item device call) is that staging happens with
  no device involvement at all. Making the row-writer name a backend would
  reintroduce the per-item device coupling that 0028 removed and would defeat
  `stack_to_tensor`.

- **Supersede ADR 0028 rather than extend it.** Rejected as factually wrong. 0028
  decided *what the primitive is* and *why batching needs a host-only seam*; this
  ADR decides *which trait holds it*. Every 0028 decision — required row-writer,
  derived `to_tensor`, hand-written `from_tensor`, rewards bypassing
  `stack_to_tensor`, the `BR = R + 1` chokepoint, the Burn reuse-or-reject verdict
  — survives unchanged and remains load-bearing. Marking it superseded would
  invalidate a rationale that is still in force.

## References

- ADR [0028](0028-tensor-batch-conversion-seam.md) — **extended, not superseded**;
  the row-writer primitive, the derived `to_tensor`, and the single-upload
  batching rationale this ADR builds on.
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md),
  [0031](0031-probability-rate-newtypes.md),
  [0046](0046-slot-newtype-replaces-option-take-around-learn-step.md) — the same
  invariants-in-types move applied to values; this ADR applies it to a trait
  boundary.
- ADR [0050](0050-replay-strategy-seam.md) — retired `memory.rs::sample_batch`,
  `stack_to_tensor`'s only nominal consumer; the reason its call count is not
  evidence of ergonomic friction (§Context).
- Issue #201 — the additive inverse batch-decode seam (`from_host_row` +
  `from_tensor_batch`); see §5 for the `Sized` placement it must use.
- Issue #195 — 0028's cross-crate parent.
- Code: `crates/rlevo-core/src/base.rs` (both traits + `stack_to_tensor`);
  `crates/rlevo-reinforcement-learning/src/algorithms/{dqn,c51,qrdqn,ddpg,td3,sac}/`
  (the six deleted turbofish staging sites); `docs/rules.md` (the one-rank-per-type
  invariant, invariant table).
