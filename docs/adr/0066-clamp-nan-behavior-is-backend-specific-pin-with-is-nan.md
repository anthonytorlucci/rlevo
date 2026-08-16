---
project: rlevo
status: active
type: decision
date: 2026-07-27
tags: [adr, decision, numerical-stability, nan, gpu, wgpu, cubecl, c51, scatter, backend-parity, issue-1044]
---

# ADR 0066: `Tensor::clamp`'s NaN behavior is backend-unspecified — pin it with an explicit `is_nan` mask

## Status

**Accepted (2026-07-27).** Resolves issue #1044, the open question ADR 0065's
own "C51 does not escape the loss guard, and the escape route is worth
tracing precisely" subsection filed by number ("this has **not** been tested
on a GPU backend and is **not** claimed to break; it is recorded as an open
question … filed as **#1044** so a maintainer with GPU hardware can settle
it"). Corrects a false premise stated in ADR 0065's own "C51 does not escape
the loss guard…" subsection and ADR 0056's own "Out of scope, deliberately"
bullet — both are immutable, so the correction lives here rather than as an
edit to either. Purely additive — supersedes nothing, changes no public API,
changes no Flex-backend numerics.

**Chosen shape.** `Tensor::clamp`'s NaN behavior is unspecified across Burn
backends. Where a computation's correctness depends on what happens to a NaN,
rlevo pins it with an explicit `is_nan` mask and never with an op's implicit
NaN handling. The primitives are `clamp_preserving_nan` (preserve) alongside the
existing `sanitize_fitness_tensor` shape (replace); a `Tensor::clamp` on
possibly-NaN data with neither is a review-flagged defect.

## Context

### The measured divergence

*(Measured on Apple M2 Pro, macOS 26.5.2, Metal 4; `burn` 0.21.0, `cubecl`
0.10.0, `wgpu` 29.0.x, `rustc` 1.94.1 — these are backend/hardware facts about
today's dependency versions, not a Burn guarantee, and are attributed to that
combination throughout this ADR.)*

`Tensor::clamp`'s NaN behavior is not part of Burn's documented contract and
differs by backend today:

- **`burn-flex`** lowers `clamp` to `f32::clamp`, which **propagates** NaN:
  `clamp(NaN, -10, 10) == NaN`.
- **`burn-cubecl`/wgpu** emits the WGSL `clamp` builtin, which naga lowers to
  Metal's `fmin(fmax(x, lo), hi)`. Measured: `clamp(NaN, -10, 10) == -10.0` —
  the NaN is **rescued** to the lower bound.
- `Ordered::clamp` in `burn-backend` documents neither behavior.

`clamp_min`/`clamp_max` are **inverted** relative to `clamp`: Flex rescues a NaN
via `f32::max`/`f32::min`, Metal propagates it. There is no single rule that
holds across the four operations and two backends; each combination must be
checked, not assumed.

### The C51 instance

`project_distribution` (`crates/rlevo-reinforcement-learning/src/algorithms/c51/projection.rs`)
clamps twice on the NaN-reward path: once on the Bellman-shifted coordinate
`tz`, once on the derived atom index `b`. With a NaN reward:

- **Flex**: `tz` is NaN, the clamp propagates it, the row comes out
  `[NaN, 0, …]` summing to `NaN` — loud. ADR 0056's `FiniteLossGuard` fires as
  designed.
- **Metal**: the clamp rescues `tz` to `v_min`in the first clamp and rescues
  `b` to `0.0` in the second, and the projection returns `[1.0, 0, …, 0]` — a
  row that sums to **exactly 1.0**. It contains no NaN and is not merely
  finite-looking; it is a well-formed categorical distribution asserting
  certainty of the worst possible return. `FiniteLossGuard` has nothing to
  catch, because there is nothing wrong with the tensor by any test that
  inspects only its values.

**A row-sum postcondition does not detect this**, stated explicitly so nobody
proposes one later: the corrupted row's sum is not merely close to 1.0, it is
1.0 to the same tolerance a healthy row would pass.

### `Tensor::is_nan` is safe to build a guard on

cubecl does not lower `is_nan` as `x != x` (a pattern fast-math compilers are
permitted to fold to `false`). It emits an integer bit-pattern polyfill —
`bitcast<u32>`, mask the sign bit, compare `> 0x7f800000u` — at
`cubecl-wgpu-0.10.0/src/compiler/wgsl/extension.rs:302`. This is a source-level
fact about the dependency, not a measurement, so it holds independent of this
machine.

**Caveat worth recording so it isn't rediscovered the hard way:** if
`cubecl-wgpu`'s `msl` or `spirv` compiler features are ever enabled, the MSL
path emits Metal's native `isnan` builtin instead, and that call site is marked
`can_optimize() -> true` at `cubecl-cpp-0.10.0/src/shared/unary.rs:406` — which
*is* fast-math-vulnerable. `is_nan` being safe today is a property of the WGSL
polyfill path this workspace currently exercises, not of the operation's name.

### The unchecked-scatter hazard

`burn-cubecl`'s scatter kernel is `#[cube(launch_unchecked)]`, and
`cubecl-wgpu` sets `bounds_checks: false`
(`cubecl-wgpu-0.10.0/src/backend/base.rs:94-101`, with an in-source comment
that WebGPU's own bounds guarantees are "too loose" to rely on). An
out-of-range scatter index on this path does not panic — it writes into
whatever tensor currently occupies that address in the shared memory pool.

**State the hazard as unbounded, not fixed-offset.** The corrupted write's
target tracks the pool allocator's 256-byte stride (`align_up(n, 64)` f32
elements), and reordering allocations changed both *which* tensor was hit and
*how many* writes landed there: allocating the eventual victim first, one bad
index reached two distinct tensors; allocating it last, the write vanished
into pool slack with no observable effect at all. The behavior is
deterministic for one fixed program, but there is no fixed offset a caller
could reason about or guard against structurally — it is a property of
allocation order, which is not part of any API contract. Flex panics on the
identical input. Burn's own `Tensor::scatter` documentation carries a
`# Warning` about this exact class of hazard.

### Scope

A workspace sweep found `c51/projection.rs` is the **only** tensor-side
float→`Int` index derivation feeding `scatter`/`gather`/`select`. QR-DQN has no
projection step — its Bellman backup uses integer-origin gather indices
throughout — and is structurally immune to this class of defect.

### Correcting the record

Three in-tree documents assert host `f32::clamp` NaN-propagation semantics as
if they were universal, when they are Flex-specific:

- `c51/projection.rs:96-101` **as of the commit this ADR supersedes** (the
  `# Panics` doc): *"`f32::clamp` propagates `NaN` rather than rescuing it, and
  `NaN as i32` saturates to `0`…"* — corrected in the same change that adds
  this ADR, so the line numbers above resolve only against the parent commit.
- ADR 0065's own "C51 does not escape the loss guard…" subsection (`:63-73`),
  which cites `projection.rs:97` as its authority and traces a NaN reward
  through *"the `clamp` at `:142` propagates NaN rather than rescuing it"*.
- ADR 0056's own "Out of scope, deliberately" bullet (`:112-115`), reasoning
  about `min(unclamped, clamped)` masking `Inf` but not `NaN` on the premise
  that clamp does not rescue NaN.

ADR 0065 and ADR 0056 are immutable — this ADR does not edit either — so the
correction is recorded here: **the clamp-propagates-NaN premise is true on
Flex and false on the wgpu/Metal backend measured above.** This does not
retroactively invalidate either ADR's *conclusion*. ADR 0065's decision to
drop a non-finite reward at replay ingestion is correct regardless of which
backend's clamp semantics apply downstream — dropping the reward removes it
from the tensor pipeline entirely, before either clamp runs. What was
backend-specific was the *trace* used to reason about C51's exposure, not the
action taken.

### The fix landing concurrently (described here, not implemented by this ADR)

A `clamp_preserving_nan(t, lo, hi)` helper in `algorithms/shared.rs`:

```text
let nan = t.clone().is_nan();
t.clamp(lo, hi).mask_fill(nan, f32::NAN)
```

applied at **both** existing clamps in `project_distribution` (the `tz` clamp
and the `b` clamp), plus a clamp of the derived `Int` indices (`l_idx`,
`u_idx`) to `[0, num_atoms - 1]`.

**The coupling is the substantive part of this decision, stated plainly:**
preserving the NaN through the two coordinate clamps *creates* the exact
out-of-range index issue #1044 was filed to check for. On the wgpu/Metal
backend, `NaN.floor().int()` is `i32::MIN`, not the host `as`-cast's saturating
`0` that ADR 0065's trace relied on — so once the NaN survives the coordinate
clamp, the derived index is no longer merely wrong, it is a scatter index that
reaches the unchecked GPU kernel described above. The index clamp is what
makes carrying the NaN through safe. **The two changes are only correct
together**: preserving without clamping the index reopens the out-of-range
scatter; clamping the index without preserving the NaN silently launders a
diverging reward into a confident, undetectable target distribution, which is
the defect this ADR exists to close. Landing one without the other is a
regression, not a partial fix.

`is_nan` is used rather than `is_finite` deliberately: $\pm\infty$ is handled
correctly and identically by the ordinary clamp on both backends measured here
— an unbounded coordinate saturates to `v_min`/`v_max` exactly as intended,
which *is* the algorithm's semantics (Bellemare et al. 2017 Algorithm 1's
Bellman shift is meant to saturate at the support boundary). Only NaN — a
value with no ordering — needs the explicit mask.

## Decision

1. **`Tensor::clamp`'s NaN behavior is backend-unspecified** and must never be
   relied upon, implicitly, for correctness. This applies to `clamp`,
   `clamp_min`, and `clamp_max` alike — the two are inverted relative to each
   other across backends, so "I know which one propagates" is not a safe
   assumption to carry from one op to the other.
2. **Two primitives cover the two shapes a computation needs:**
   - **Preserve** — `clamp_preserving_nan(t, lo, hi)`: bound a value while
     keeping a NaN input NaN, so a downstream finiteness guard (ADR 0056's
     `FiniteLossGuard`, ADR 0065's `FiniteRewardGuard`) can still see it and
     fire. This is the C51 shape: the projection must not *manufacture*
     confidence out of a reward the environment never validly emitted.
   - **Replace** — the existing `sanitize_fitness_tensor` shape
     (`rlevo-evolution/src/fitness.rs:324-329`): mask NaN to a defined
     sentinel (`-inf` under maximise-native ordering) *before* any clamp runs,
     so the clamp only ever sees already-finite input and its NaN behavior is
     moot by construction. This is the right shape when the caller's contract
     is "NaN loses," not "NaN must remain visible."
3. **A bare `Tensor::clamp` call on data that can carry NaN, with neither
   primitive present, is a review-flagged defect** — not a lint (no clippy
   lint distinguishes "this tensor can be NaN" from one that provably cannot),
   a convention enforced the same way rules.md's Trait Design Constraints
   section already enforces `total_cmp`-over-`partial_cmp` and the
   sanitize-before-compare rule.

## Rejected alternatives

- **Sanitize the whole row to a fixed distribution (e.g. all-zero) whenever
  the reward is non-finite.** A zeroed row contributes exactly zero loss and
  zero gradient for that sample — indistinguishable downstream from a sample
  that has already converged perfectly. That is *harder* to notice than the
  bug it replaces: at least the Metal all-mass-on-`v_min` row is wrong in a
  way that, once suspected, is checkable; an all-zero row looks like nothing
  happened at all.
- **A row-sum postcondition check** (`assert!((row.sum() - 1.0).abs() <
  eps)`). Rejected on the measured evidence in this ADR's own "The C51
  instance" subsection above: the corrupted row sums to exactly 1.0. This
  check would pass on the exact input it is meant to catch.
- **A host-side finiteness check on the already-projected target.** This
  would be free — `c51_agent.rs` already round-trips the projected target
  through host memory for its priority writeback — but it is useless for the
  same reason the row-sum check is: the corrupted target is finite. A
  finiteness check has nothing to see.
- **Switch to Dopamine's dense tent-kernel projection**, which derives no
  discrete indices at all and is structurally immune to this defect. Rejected
  for two reasons. First, it does not actually fix the problem it would be
  adopted to fix: Dopamine's `project_distribution` clips both intermediate
  quantities on the NaN path with plain `clip`/`clamp` calls, so on the
  wgpu/Metal backend a NaN reward yields an all-zero row under that
  implementation too — contributing zero loss and zero gradient, which is
  *harder* to notice than rlevo's current corruption, not easier. Second, even
  if it were a clean fix, replacing the index-based projection with a dense
  tent kernel is a performance-characterizing rewrite of the whole operator,
  and coupling that decision to a correctness fix for one reward-path defect
  is scope creep this ADR declines to take on.

## Consequences

- **CI does not cover this class of defect, and this ADR does not claim
  otherwise.** Every workflow in `.github/workflows/` runs on `ubuntu-latest`.
  The workspace's only cross-backend test,
  `crates/rlevo-evolution/tests/backend_parity.rs:128`
  (`wgpu_matches_flex_on_sphere_d10`), is `#[ignore]`d because CI runners have
  no GPU adapter. Regression protection for the fix this ADR describes is a
  **manual GPU run** on hardware that has one — there is no automated gate.
- The correction to ADR 0065's own "C51 does not escape the loss guard…"
  subsection and ADR 0056's own "Out of scope, deliberately" bullet (above)
  is the only change either of those records will ever receive; both stay
  immutable and their own conclusions stand unamended.
- The two convention primitives (`clamp_preserving_nan`,
  `sanitize_fitness_tensor`) are now the two named shapes a reviewer checks a
  new tensor `clamp` call against. A workspace sweep for the third,
  unguarded shape is *not* performed by this ADR — see the next section for
  the specific sites already found and left open.

## Known-affected sites not fixed here

Each of the following is the same class of defect — a `clamp` (or
`clamp_min`) whose NaN behavior is silently backend-dependent — found during
the sweep for this ADR but out of scope for the C51 fix landing alongside it.
Each is to be filed as its own issue rather than bundled in.

- **`ppo/policies/gaussian.rs:460`** (`clamped_log_std`) and
  **`sac/sac_policy.rs:176`** (`mean_and_log_std`). Both clamp `log_std`, a
  **persistent, gradient-updated** module parameter, not a transient
  per-batch value — a materially different risk profile than a one-shot
  projection. On the wgpu/Metal backend a NaN gradient step could rescue
  `log_std` to `log_std_min`, giving $\sigma = \exp(-20) \approx 2.06 \times 10^{-9}$ — finite,
  plausible, and consumed **consistently** by both sampling and log-prob
  evaluation, so nothing downstream ever disagrees with anything else. Worse,
  the ADR 0049/#347 host-side warn scan (`gaussian.rs:497,499`) uses `v <
  self.log_std_min` / `v > self.log_std_max` comparisons, both of which are
  `false` for a NaN operand under IEEE 754 — so the warn scan **never fires**
  for this case, on *either* backend, independent of which way the clamp
  itself resolves the NaN.
- **`ppo/losses.rs`'s `min_elem`.** Measured: with a NaN `log_ratio`, the PPO
  policy loss is `NaN` on Flex (ADR 0056's guard fires) but `-0.50172365` on
  Metal — finite, plausible-looking, and the guard never fires.
- **`c51/loss.rs:92` — audited and CLEAR, recorded so it is not re-opened.**
  `clamp_min` guards `log(0)` in the entropy term of the KL priority signal fed
  to prioritized replay (`c51_agent.rs:707-715`), and `clamp_min`'s NaN
  behavior is divergent in the *opposite* direction to `clamp` (Flex rescues
  via `f32::max`; Metal propagates). It does not matter here: the expression is
  `target_probs * target_probs.clamp_min(FLOOR).log()`, and the **left factor
  is unclamped**, so a NaN in `target_probs` reaches the product on either
  backend no matter which way the clamp goes. Same structural property that
  saves `sanitize_fitness_tensor` below — the clamp's result is never the sole
  carrier of the NaN.
- **The good news, recorded so a future sweep does not re-open it:**
  `rlevo-evolution/src/fitness.rs:324-329` (`sanitize_fitness_tensor`) masks
  `is_nan → -inf` **before** `clamp_max` runs, so it is correct on both
  backends measured here regardless of what `clamp_max` would do with a NaN
  input directly — cite it as the idiom to copy. Only its explanatory comment
  (`fitness.rs:319-320`, *"so no `NaN` reaches the clamp, which would
  propagate it"*) repeats the false universal-propagation premise this ADR
  corrects; the code itself was already right by construction.
- **`sac_alpha.rs`'s clamps operate on a plain host `f32`**
  (`self.log_alpha.clamp(LOG_ALPHA_MIN, LOG_ALPHA_MAX)`, `sac_alpha.rs:200,374`),
  never on a `Tensor`, so this ADR's backend divergence does not apply to
  them at all.

## References

- Issue #1044 — the open question ADR 0065's own "C51 does not escape the
  loss guard…" subsection filed by number.
- ADR [0065](0065-non-finite-reward-is-dropped-at-replay-ingestion.md) —
  its own "C51 does not escape the loss guard, and the escape route is
  worth tracing precisely" subsection traces C51's NaN-reward propagation
  and states, correctly for Flex only, that the coordinate clamp propagates
  NaN; corrected, not edited, by this ADR. Its own Decision section —
  dropping the reward at replay ingestion — is unaffected.
- ADR [0056](0056-non-finite-loss-skip-and-warn-guard.md) — its own "Out of
  scope, deliberately" bullet reasons about `min(unclamped, clamped)`
  masking `Inf` but not `NaN` on the same Flex-specific premise; corrected,
  not edited, by this ADR.
- ADR [0034](0034-fitness-hygiene-chokepoint-convention.md) — the
  `sanitize_fitness`/`sanitize_fitness_tensor` "replace" shape this ADR names
  as one of its two blessed primitives.
- `crates/rlevo-reinforcement-learning/src/algorithms/c51/projection.rs` — the
  two coordinate clamps this ADR governs, at `:194` (`tz`) and `:236` (`b`),
  both now routed through `clamp_preserving_nan`, plus the derived-index clamps
  at `:238-239`. The rewritten `# Panics` doc is at `:123`. These are post-fix
  line numbers; the pre-fix layout cited in this ADR's own "Correcting the
  record" subsection resolves only against this change's parent commit.
- `crates/rlevo-evolution/src/fitness.rs:319-329` — `sanitize_fitness_tensor`,
  the "replace" idiom to copy, and the comment repeating the corrected
  premise.
- `crates/rlevo-reinforcement-learning/src/algorithms/ppo/policies/gaussian.rs:460,483-502`
  — `clamped_log_std` and the NaN-blind warn scan.
- `crates/rlevo-reinforcement-learning/src/algorithms/sac/sac_policy.rs:176`
  — `mean_and_log_std`'s `log_std` clamp.
- `crates/rlevo-reinforcement-learning/src/algorithms/ppo/losses.rs:173-176`
  — `min_elem`, the PPO policy-loss guard bypassed by a rescued NaN.
- `crates/rlevo-reinforcement-learning/src/algorithms/c51/loss.rs:92` —
  `clamp_min` on the KL priority signal; audited clear (the unclamped left
  factor carries the NaN on either backend).
- `crates/rlevo-reinforcement-learning/src/algorithms/sac/sac_alpha.rs:200,374`
  — the host-`f32` clamps unaffected by this ADR.
- `crates/rlevo-evolution/tests/backend_parity.rs:128` —
  `wgpu_matches_flex_on_sphere_d10`, `#[ignore]`d for want of a GPU CI runner;
  the only cross-backend test in the workspace and the manual-run gate this
  ADR's own Consequences section names.
- Bellemare, M. G., Dabney, W., Munos, R. *A Distributional Perspective on
  Reinforcement Learning.* ICML 2017. arXiv:1707.06887. Algorithm 1, Eq. (7),
  Section 4.1.
- CleanRL `c51.py` — clamps both the continuous coordinate and the derived
  integer indices; the `l == u` mass-preservation fix this workspace also
  applies is implementation folklore with no published erratum, and CleanRL's
  commented fix is the most defensible citation for it.
- Dopamine `rainbow_agent.py::project_distribution` — a dense tent-kernel
  implementation of Eq. (7) that derives no indices and is structurally immune
  to this defect, but does not fix it: its two `clip` calls sit on the NaN
  path too, so a wgpu/Metal NaN reward yields an all-zero row under that
  implementation as well.
- Full ecosystem citations, code excerpts, and the measured backend table:
  `docs/.private/research/2026-07-27-issue-1044-clamp-nan-backend-divergence.md`.
