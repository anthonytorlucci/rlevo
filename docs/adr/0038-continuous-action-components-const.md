---
project: rlevo
status: active
type: decision
date: 2026-07-09
tags: [adr, decision, action, continuous-action, rlevo-core, box2d, panic]
---

# ADR 0038: A `COMPONENTS` associated constant on `ContinuousAction`

## Status

**Accepted (2026-07-09).** Resolves issue #100 (`ContinuousAction::random()`
panics for multi-component rank-1 actions). Supersedes nothing.

## Context

### Terminology: rank, shape, and component count are three different things

`rlevo-core`'s own doc comments are precise about this and worth restating,
because #100 is exactly a case where the distinction was not respected:

- **Rank** (`Action::RANK`, `= R`) is the number of axes — "the count of
  indices needed to address an element" (NumPy `ndim`, Burn's `Tensor<B,
  R>`), *not* matrix rank, *not* a size, and *not* a product of anything. A
  rank-1 action has exactly one axis; that axis can hold one value or a
  thousand.
- **Shape** (`Action::shape() -> [usize; R]`) is the per-axis cardinality: an
  array of length `R` (the rank), where element `i` is the size along axis
  `i`. Shape has length `R`; it does not by itself give a scalar count.
- **Component count** — the quantity `ContinuousAction::as_slice()` returns
  and `from_slice()` consumes — is the total number of flattened `f32`
  scalars in the action's representation. For a tensor-shaped value whose
  `shape()` *faithfully* describes its storage (one array axis per stored
  dimension, sized correctly), component count equals `shape().iter()
  .product()` — the standard `numel` = product-of-axis-sizes identity.
  `rlevo-core::State` even defines exactly this as its own `numel()`
  default (`crates/rlevo-core/src/base.rs`). **`Action` has no such
  `numel()`,** and, as the next section shows, nothing in the trait forces
  `ContinuousAction::shape()` to faithfully describe storage — so even
  `shape().iter().product()` cannot be assumed to equal component count for
  every impl.

`ContinuousAction<R>: Action<R>` inherits `RANK == R`. The bug in #100 was
treating that inherited rank as if it were the component count.

### The defect

The default `ContinuousAction::random()` sampled `Self::RANK` values and
passed them to `from_slice`, which requires the full component count:

```rust
fn random() -> Self
where
    Self: Sized,
{
    use rand::RngExt;
    let mut rng = rand::rng();
    let values: Vec<f32> = (0..Self::RANK)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    Self::from_slice(&values)
}
```

Every production continuous action in the workspace is `Action<1>` — one
axis — regardless of how many components it holds. So for any type with more
than one component, `random()` samples exactly one value while `from_slice`
`assert_eq!`s the input length against the true component count, and panics.
This is an unconditional panic reachable through the public trait's own
default method, not a caller error.

The blast radius, reconciled against every `ContinuousAction` impl in the
workspace: three box2d types are rank-1 with more than one component and
have no override, so they panic —

| Type | `Action<R>` | components | overrides `random()`? |
|------|-------------|-----------|-----------------------|
| `BipedalWalkerAction` | `<1>` | 4 | no — **panics** |
| `CarRacingAction` | `<1>` | 3 | no — **panics**, and would also sample out of bounds |
| `LunarLanderContinuousAction` | `<1>` | 2 | no — **panics** |
| `ReacherAction`, `SwimmerAction` | `<1>` | 2 | yes — already safe |
| `InvertedPendulumAction`, `InvertedDoublePendulumAction`, `PendulumAction`, `MountainCarContinuousAction` | `<1>` | 1 | yes / coincidentally safe (rank happens to equal component count when both are 1) |

### The trap: two incompatible conventions for what `shape()` means

Fixing `random()` by substituting some *other* expression already on the
trait for `Self::RANK` is not safe, because the codebase carries **two
different conventions** for how a `ContinuousAction`'s `shape()` relates to
its component count:

- **Convention A — every production impl.** `Action<1>` (rank 1, a single
  axis), and `shape()` returns `[C]` where `C` is the true component count
  (e.g. `BipedalWalkerAction::shape() == [4]`, 4 components on one axis).
  Here `shape()` faithfully describes storage, so `shape().iter().product()
  == C` — but `RANK == 1 != C` for every `C > 1`.
- **Convention B — the trait's own in-module test type,
  `ContinuousActionTest`.** `Action<3>` (rank 3, three axes), yet it stores
  3 independent `f32`s and declares `shape()` as `[1, 1, 1]` — one
  size-1 axis per stored scalar, rather than the one-axis-of-size-3 shape
  Convention A would use. Here `RANK == 3 == C` (rank happens to match
  component count), but `shape()` does **not** faithfully describe storage
  as a single flat axis, so `shape().iter().product() == 1 != C`.

Both types compile against `ContinuousAction` today and both pass the
trait's own test suite (`crates/rlevo-core/src/action.rs`). A fix that
derives component count from `RANK` is correct under B and wrong under A
(reproduces #100 for every Convention-A multi-component type); a fix that
derives it from `shape().iter().product()` is correct under A and wrong
under B. **Neither `RANK` nor any derivation from `shape()` is universally
correct across the impls that already exist.** The component count has to
be supplied directly by each implementor, not computed from something else
on the trait.

## Decision

1. **Add a required associated constant** `const COMPONENTS: usize` to
   `ContinuousAction`, defined as the length of the slice returned by
   `as_slice()` and consumed by `from_slice()` — explicitly **not** `RANK`
   and **not** `shape().iter().product()`, per the Context section above.
   Every one of the 11 existing `ContinuousAction` impls in the workspace
   must declare it.

2. **No default value.** `const COMPONENTS: usize = R;` would compile for
   every impl and silently reproduce the #100 panic for every Convention-A
   multi-component action — the exact bug this ADR fixes. No
   `shape()`-derived default is available either: it would be wrong under
   Convention B, and an associated-`fn` call (`shape().iter().product()`)
   is not usable to define an associated-`const` default on stable Rust in
   any case. Requiring every implementor to write the literal component
   count is deliberate: it forces the value to be *stated*, not derived,
   and makes the RANK/component-count trap impossible to re-inherit by
   omission — a missing `COMPONENTS` is a compile error, not a latent
   panic.

3. **The default `random()` is corrected to loop over `COMPONENTS`, not
   `RANK`:**

   ```rust
   fn random() -> Self
   where
       Self: Sized,
   {
       use rand::RngExt;
       let mut rng = rand::rng();
       let values: Vec<f32> = (0..Self::COMPONENTS)
           .map(|_| rng.random_range(-1.0..1.0))
           .collect();
       Self::from_slice(&values)
   }
   ```

   This samples every component uniformly in `[-1.0, 1.0)`. Combined with
   `COMPONENTS`, the default now produces a **correctly-sized** action for
   every impl, and additionally a **correctly-distributed** one for the two
   symmetric-bound box2d types it previously panicked on:
   `BipedalWalkerAction` (Gymnasium `Box(-1.0, 1.0, (4,))`, all four
   components `[-1, 1]`) and `LunarLanderContinuousAction` (Gymnasium
   `Box(-1, +1, (2,))`, both components `[-1, 1]`) [1][3].

4. **`[-1, 1)` is a symmetric-bound assumption; document it as such.** The
   trait docs state that the default `random()` assumes every component's
   valid range is symmetric about zero and equal to `[-1, 1)`; an
   implementor whose action space is not symmetric **must** override
   `random()`. `CarRacingAction` is the canonical case: its Gymnasium action
   space is `Box([-1,0,0], [1,1,1], (3,))` — steer `∈ [-1,1]` but gas and
   brake `∈ [0,1]` (asymmetric) [2]. It overrides `random()` to sample
   within its true per-component bounds rather than relying on the default.
   "No panic" (component count, fixed universally by `COMPONENTS`) and
   "correct distribution" (bounds, still a per-type concern) are separate
   axes of correctness; this ADR fixes the first for every impl and leaves
   the second a documented per-type override responsibility, the same
   responsibility `BoundedAction` implementors already carry.

5. **Sync safeguard.** Add a contract test, run against every
   `ContinuousAction` impl, asserting `T::COMPONENTS ==
   T::random().as_slice().len()` — keyed on `as_slice()`, **not**
   `shape().iter().product()`, because the latter fails under Convention B
   (`ContinuousActionTest`). This test is what caught a pre-existing,
   unrelated defect while being written: `CarRacingAction::as_slice()`
   returned only 1 of its 3 components (a truncation bug). It is fixed in
   the same change, since the new contract test would otherwise fail on
   landing.

## Consequences

### Positive
- The panic class named in #100 is eliminated at the trait seam, not
  patched per call site: `ContinuousAction::random()` cannot panic for any
  conforming impl, and a new impl that omits `COMPONENTS` fails to compile
  rather than silently inheriting the trap.
- Two of the three previously-panicking types (`BipedalWalkerAction`,
  `LunarLanderContinuousAction`) become correct via the default alone — no
  override needed.
- The `as_slice()`-keyed contract test is a stronger invariant than a
  shape-keyed one and surfaced (and fixed) an independent bug
  (`CarRacingAction::as_slice()` truncation) as a side effect of adding it.

### Negative / accepted costs
- **Breaking change to all 11 existing `ContinuousAction` impls.** Every
  implementor must add a `COMPONENTS` declaration; this lands atomically in
  one PR. Acceptable in alpha (no external consumers).
- **Convention B is not deprecated by this ADR.** `ContinuousActionTest`
  (rank 3, `shape() == [1,1,1]`, storage not faithfully described by
  `shape()`) keeps compiling and stays correct under the new contract,
  because `COMPONENTS` is supplied directly rather than derived from
  `shape()`. Whether Convention B should eventually be retired in favor of
  the uniform rank-1 `[C]` form (Convention A) — which would make
  `shape().iter().product()` a viable, no-const-needed derivation — is an
  **open question this ADR does not decide**.

### Explicitly out of scope

`BoundedAction<R>::low()` / `high()` return `[f32; R]` — keyed on the tensor
**rank**, not on component count. For a rank-1, multi-component action
(e.g. `BipedalWalkerAction`, `Action<1>`, 4 components) this means
`low()`/`high()` can only express **one** bound value across all 4
components, not per-component bounds — the identical rank-vs-component
confusion this ADR fixes for `random()`, still present on `BoundedAction`.

This is **not** fixed here, and cannot be fixed by mechanically substituting
`[f32; Self::COMPONENTS]` for `[f32; R]`: an array length keyed on an
associated constant (`[f32; Self::COMPONENTS]`) requires the unstable
`generic_const_exprs` feature on stable Rust. Making `BoundedAction`
component-correct needs either a second const generic parameter (distinct
from `R`) threaded through the trait and every impl, or a `Vec`/slice-typed
bounds representation — either is a materially larger, differently-shaped
change than adding one associated constant. It is deferred to a **future
ADR**, which should also decide the Convention A/B question noted above as
part of picking the component-keying scheme.

**Concrete blast radius already resting on this gap.** This is not a purely
hypothetical follow-up — three shipped `rlevo-reinforcement-learning`
agents already consume `A::RANK` as if it were `A::COMPONENTS` on
`A: BoundedAction<R>`, the identical conflation this ADR fixes for
`ContinuousAction::random()`, one layer up the stack:

- `DdpgAgent::act` / `act_with`
  (`crates/rlevo-reinforcement-learning/src/algorithms/ddpg/ddpg_agent.rs`,
  ~L290, L301, L336) loop `0..A::RANK` for the warm-up sampler and
  `slice.iter().take(A::RANK)` to pull the actor network's output out of its
  raw tensor, then index `self.low[i]`/`self.high[i]` (from `A::low()` /
  `A::high()`, `[f32; R]`) over that same range.
- `Td3Agent::act` / `act_with`
  (`crates/rlevo-reinforcement-learning/src/algorithms/td3/td3_agent.rs`,
  ~L335, L346, L381) — the identical pattern.
- `SacAgent::act`
  (`crates/rlevo-reinforcement-learning/src/algorithms/sac/sac_agent.rs`,
  ~L376, L408) — the identical pattern.
- `SacAgent`'s default target entropy,
  `crates/rlevo-reinforcement-learning/src/algorithms/sac/sac_agent.rs:275`:
  `config.target_entropy.unwrap_or_else(|| -(A::RANK as f32))`, documented at
  `sac_config.rs:43` as "the common heuristic from Haarnoja et al. 2018b."
  The canonical heuristic is `−dim(action_space)` — negative **component
  count** — not negative rank.

All four sites are **coincidentally correct today only** because every
`BoundedAction` impl that exists (`PendulumAction`, `MountainCarContinuousAction`,
`LinearAction` — all `<1>`, 1 component; the in-trait `ContinuousActionTest`
— `<3>`, 3 components under Convention B) happens to have `RANK ==
COMPONENTS`. None of the three multi-component box2d actions
(`BipedalWalkerAction`, `CarRacingAction`, `LunarLanderContinuousAction`)
implement `BoundedAction` today — only `ContinuousAction`. The moment a
`BoundedAction<1>` impl is added for any of them, `.take(A::RANK)` silently
truncates the actor's real output to 1 of `C` components and `from_slice`
panics on the length mismatch — the #100 panic resurfacing one layer above
the trait default this ADR fixes, plus a wrong SAC entropy target. This is
the concrete problem statement the follow-up `BoundedAction` ADR must
resolve, not an abstract risk.

## Alternatives considered

- **Default `COMPONENTS` to `R`.** Rejected: this is exactly the bug. Every
  Convention-A multi-component rank-1 type (`BipedalWalkerAction`,
  `CarRacingAction`, `LunarLanderContinuousAction`) would inherit
  `COMPONENTS = 1` and the default `random()` would keep panicking — a
  required-but-defaulted constant provides no safety over the status quo.
- **Derive `COMPONENTS` from `shape().iter().product()`.** Rejected on two
  grounds: (1) it is wrong under Convention B, where `shape() == [1,1,1]`
  products to `1` but the true component count is `3`; (2) even where it
  would be numerically correct (Convention A), an associated-`fn` call is
  not usable to define an associated-`const` default on stable Rust, so
  this was never available as a *default* — only as a manually-implemented
  per-type constant, which is no safer than requiring the literal directly.
- **A second const generic parameter (e.g. `ContinuousAction<const R:
  usize, const C: usize>`).** Would resolve both `random()` and the
  deferred `BoundedAction` bound-array problem in one shape, since `[f32;
  C]` is const-generic-native (no `generic_const_exprs` needed). Rejected
  for **this** issue: it changes the trait's generic arity, touching every
  `ContinuousAction<R>` bound across the workspace (RL algorithms, bounded
  warm-up sampling, tests) for a fix #100 does not require — `random()`
  only needs a `const`, not a second type parameter. Recorded here as the
  leading candidate for the follow-up ADR that resolves `BoundedAction`.
- **Fix only the three broken types with hand-rolled `random()` overrides,
  no trait change.** Rejected: this papers over the seam instead of closing
  it — a twelfth `ContinuousAction` impl could reintroduce the identical
  panic with no compiler signal. The associated constant is what makes the
  omission a compile error instead of a latent bug.

## References
- Issue #100 — `Env::ContinuousAction::random` default panics for
  multi-component rank-1 actions. Resolved for `random()`/`COMPONENTS`; the
  `BoundedAction` rank-vs-component gap is split to a follow-up ADR.
- Vault research note
  `docs/.private/research/2026-07-09-issue-100-continuous-action-random.md` —
  blast-radius table and Gymnasium citations.
- [1] Farama Gymnasium — BipedalWalker action space (`Box(-1.0, 1.0, (4,),
  float32)`). https://gymnasium.farama.org/environments/box2d/bipedal_walker/
- [2] Farama Gymnasium — CarRacing (continuous) action space
  (`Box([-1,0,0], [1,1,1], (3,), float32)`, asymmetric gas/brake bounds).
  https://gymnasium.farama.org/environments/box2d/car_racing/
- [3] Farama Gymnasium — LunarLander (Continuous) action space (`Box(-1, +1,
  (2,), float32)`). https://gymnasium.farama.org/environments/box2d/lunar_lander/
- Code: `crates/rlevo-core/src/action.rs` (`ContinuousAction`,
  `BoundedAction`, and the in-module `ContinuousActionTest` demonstrating
  Convention B); `crates/rlevo-core/src/base.rs` (`Action::RANK`/`shape()`
  doc contract, `State::numel()` for the faithful-shape comparison case).
