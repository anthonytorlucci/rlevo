---
project: rlevo
status: active
type: decision
date: 2026-07-19
tags: [adr, decision, action, bounded-action, continuous-action, rlevo-core, ddpg, td3, sac]
---

# ADR 0053: `BoundedAction` returns per-component bound slices

## Status

**Accepted (2026-07-19).** Resolves issue #253 (`BoundedAction::low()`/`high()`
are keyed on tensor rank, not component count). Discharges the follow-up ADR
that ADR [0038](0038-continuous-action-components-const.md) named in its
"Explicitly out of scope" section, including the Convention A/B question 0038
attached to it. Supersedes nothing; ADR 0038 remains accepted and unmodified.

**Chosen shape:** `fn low() -> &'static [f32]` / `fn high() -> &'static [f32]`
on an otherwise-unchanged `BoundedAction<const R: usize>`, with
`low().len() == high().len() == Self::COMPONENTS` as a documented, contract-
tested invariant. The trait's generic arity does not change, so all 12
`A: BoundedAction<DA>` bound sites compile untouched; only the return type and
the sites that *index* it change.

## Context

### The defect

`crates/rlevo-core/src/action.rs:380-392`:

```rust
pub trait BoundedAction<const R: usize>: ContinuousAction<R> {
    /// Returns the per-component lower bounds of this action space.
    fn low() -> [f32; R];
    /// Returns the per-component upper bounds of this action space.
    fn high() -> [f32; R];
}
```

`R` is the tensor **rank** — the number of axes — not the flattened scalar
count. ADR 0038 established that these are three distinct quantities (rank,
shape, component count) and added `ContinuousAction::COMPONENTS` as the single
authority for the third. `BoundedAction` never got the same treatment: for a
rank-1 action with `C > 1` components — which is *every* multi-component action
in this workspace, since every production continuous action is `Action<1>` —
`[f32; R]` is `[f32; 1]` and can express exactly **one** bound for all `C`
components. The doc comment directly above each method already promises
"per-component lower bounds"; the signature cannot deliver it. This is the same
rank-vs-component conflation as issue #100, one layer up the stack.

### Why it is latent, and exactly what makes it fire

All five current `BoundedAction` impls happen to have `R == COMPONENTS`, so
nothing is wrong today:

| Impl | site | `R` / `COMPONENTS` |
|------|------|--------------------|
| `PendulumAction` | `classic/pendulum.rs:244` | 1 / 1 |
| `MountainCarContinuousAction` | `classic/mountain_car_continuous.rs:269` | 1 / 1 |
| `LinearAction` | `rlevo-test-support/src/env.rs:130` | 1 / 1 |
| `MaskContinuousAction` | `bootstrap_mask.rs:154` | 1 / 1 |
| `ContinuousActionTest` | `rlevo-core/src/action.rs:547` | 3 / 3 (Convention B) |

Five multi-component rank-1 `ContinuousAction` impls have **no**
`BoundedAction` impl, and are therefore the environments currently locked out
of DDPG / TD3 / SAC:

| Type | components | Gymnasium action space |
|------|-----------|------------------------|
| `BipedalWalkerAction` | 4 | `Box(-1.0, 1.0, (4,))` [1] |
| `CarRacingAction` | 3 | `Box([-1,0,0], [1,1,1], (3,))` — asymmetric [2] |
| `LunarLanderContinuousAction` | 2 | `Box(-1, +1, (2,))` [3] |
| `ReacherAction` | 2 | — |
| `SwimmerAction` | 2 | — |

Adding a `BoundedAction<1>` impl for any one of them fires the bug. The trigger
is not hypothetical: "give the box2d/locomotion continuous envs a continuous-
control agent" is the obvious next feature, and it is blocked precisely here.

### The three consumers, and how much they actually depend on the array shape

**(a) The exploration-noise layer needs no change at all.**
`algorithms/ddpg/exploration.rs:43-49`:

```rust
pub fn apply<R: Rng + ?Sized>(&self, mean: &[f32], low: &[f32], high: &[f32], rng: &mut R) -> Vec<f32>
```

It already takes **slices**, already length-checks at runtime (`:50-51`), and
its own test at `:86-87` already exercises four distinct per-component bounds.
TD3 reuses this same `GaussianNoise`. The layer that does the per-component
work was written correctly from the start; only the trait feeding it is wrong.
This is the single strongest constraint on the fix: the correct-consumer side
of the seam is slice-shaped, so a slice-shaped trait is the *smaller* change,
not the larger one.

**(b) The three agents conflate `RANK` with the component count.** Each of the
following loops `0..A::RANK` (or `.take(A::RANK)`) over a `COMPONENTS`-wide
actor output, then feeds the truncated result to `from_slice`, whose documented
`assert_eq!` panics:

- `ddpg_agent.rs:288-289, 299, 305, 339-340`
- `td3_agent.rs:329-330, 340, 346, 381-382`
- `sac_agent.rs:369-370, 401-402`

Agent bound-storage fields are `[f32; DA]` (`ddpg_agent.rs:152-153`,
`td3_agent.rs:180-181`, `sac_agent.rs:188-189`).

**(c) A second, unnamed defect: the scalar target-action clip.** Neither #253
nor ADR 0038 named this. `ddpg_agent.rs:473-474` and `td3_agent.rs:521-522`
collapse the bounds to a single scalar for the target-action clip:

```rust
// CleanRL uses low[0]/high[0] as a scalar clip on the target action;
// adopt the same convention (documented in BoundedAction).
let low_scalar = self.low[0];
let high_scalar = self.high[0];
```

TD3 threads these into `smoothed_target_action(...)` at `:526-533`. This is
correct only while every component shares one bound — i.e. under exactly the
condition that makes the main bug latent. For `CarRacingAction`
(`Box([-1,0,0], [1,1,1])`), `low[0]/high[0]` is `-1/1`, which would admit
**negative gas and negative brake** into the target action: literally the
"values of impossible actions" that TD3's target clip exists to suppress
(Fujimoto et al. 2018, Eq. 14, which clips against the Gym `Box(low, high)`
*vectors*) [5]. Same trigger condition, same root cause, different failure mode
— a silently degraded target rather than a panic.

The relevant Burn 0.21 constraint, verified against vendored source
(`burn-tensor-0.21.0/src/tensor/api/orderable.rs`): `clamp` (`:986`),
`clamp_min` (`:1022`) and `clamp_max` (`:1054`) take a **scalar**
`E: ElementConversion` and cannot express per-component limits. Per-component
tensor clipping must go through `max_pair` (`:722`) / `min_pair` (`:950`),
which take a full tensor operand and therefore require a bounds tensor
broadcastable to the batched action shape.

### What the literature says the bound is

Per-component bound *vectors* are the only canonical form across the three
algorithms that consume this trait:

- DDPG (Lillicrap et al. 2015, Eq. 7) bounds actions with an elementwise
  `tanh` [4].
- TD3 (Fujimoto et al. 2018, Eq. 14) clips the smoothed target action against
  the Gym `Box(low, high)` vectors [5].
- SAC (Haarnoja et al. 2018b, Appendix C) applies `tanh` elementwise to
  `u ∈ R^D` with a per-index log-prob correction; Appendix D Table 1 gives the
  entropy target as `−dim(A)`, the action-space **dimensionality**, e.g. `−6`
  for HalfCheetah-v1 [6].

Note that the SAC target-entropy sub-claim attached to #253 is **already
fixed**: `sac_agent.rs:280` reads `-(A::COMPONENTS as f32)` as of commit
`e0ce1cd`. This ADR does not re-open it.

### Does the stable-Rust obstacle ADR 0038 cited still hold?

Yes — re-verified rather than inherited. On `rustc 1.97.0`:

```
error: generic parameters may not be used in const operations
 --> fn low() -> [f32; Self::COMPONENTS];
  = note: type parameters may not be used in const expressions
```

`[f32; Self::COMPONENTS]` requires the unstable `generic_const_exprs`, exactly
as ADR 0038 predicted and as ADR 0052 independently found for
`[usize; Self::RANK]`. The mechanical substitution is unavailable; a real
design choice is required.

## Decision

### 1. `low()` / `high()` return `&'static [f32]`

```rust
/// A [`ContinuousAction`] with statically-known per-component `[low, high]` bounds.
///
/// # Invariants
///
/// - `low().len() == high().len() == Self::COMPONENTS`.
/// - `low()[i] < high()[i]` for every component `i`.
/// - [`ContinuousAction::clip`] is a no-op on an action already within bounds.
pub trait BoundedAction<const R: usize>: ContinuousAction<R> {
    fn low() -> &'static [f32];
    fn high() -> &'static [f32];
}
```

The const generic `R` is **retained** — it is required to name the
`ContinuousAction<R>` supertrait — but it no longer appears in any signature.
Consequently the trait's generic arity is unchanged and every `A: BoundedAction<DA>`
bound in the workspace compiles verbatim.

Every impl becomes a static-slice return:

```rust
impl BoundedAction<1> for PendulumAction {
    fn low() -> &'static [f32] { &[-2.0] }
    fn high() -> &'static [f32] { &[2.0] }
}
```

### 2. `&'static [f32]`, not `Vec<f32>` or `Box<[f32]>`

`low()`/`high()` are called on the hot path — once per `act()` in DDPG's
warm-up sampler, and (today) once per `learn_step` for the target clip. A
`Vec`-returning signature allocates on every call for a value that is, in every
existing impl and every impl this ADR anticipates, a compile-time literal.
`&'static [f32]` is `Copy`, allocation-free, and lets the agents store the
bounds as `&'static [f32]` fields rather than `Vec<f32>` — no clone, no
lifetime plumbing, no per-step allocation.

The escape hatch for a genuinely computed bound is `OnceLock` + `Box::leak`,
which is documented in the trait docs. No impl needs it today.

### 3. The trait's "runtime env config" doc claim is false and is deleted

The current trait docs justify static *methods* over associated *constants*
with: "so implementors can still derive bounds from a runtime env config (e.g.
a `max_torque` field)". This is not true. `low()`/`high()` take no `self` and
no argument; they cannot observe any instance, so they cannot reach a
`max_torque` field on a config. All five existing impls return literals. The
claim is removed rather than preserved, and `&'static [f32]` therefore
forecloses nothing that was actually reachable.

If per-*instance* bounds are ever genuinely needed (a config-parameterised
torque limit), that is a different seam — an instance method, or bounds carried
on the environment rather than the action *type* — and needs its own ADR. It is
explicitly **not** what the current signature provides, so this ADR is not
removing that capability; it is deleting a doc comment that promised it falsely.

### 4. The `COMPONENTS`-agreement invariant is contract-tested, not compiled

`&'static [f32]` gives up the compiler's guarantee that `low()` and `high()`
have equal length. This ADR accepts that loss because **the guarantee that
matters was never available in either option** — see §Alternatives A. The
replacement is a contract test, extending the ADR 0038 `COMPONENTS` test to
every `BoundedAction` impl:

```rust
assert_eq!(T::low().len(), T::COMPONENTS);
assert_eq!(T::high().len(), T::COMPONENTS);
for i in 0..T::COMPONENTS { assert!(T::low()[i] < T::high()[i]); }
```

Additionally, the agents `assert_eq!` the stored bound lengths against
`A::COMPONENTS` **at construction**, so a nonconforming impl fails loudly at
agent-build time rather than mid-episode.

`docs/rules.md` §3's Core Trait Invariants table row for `BoundedAction<D>` is
updated from `low()[i] < high()[i]` to include the length agreement.

### 5. All `A::RANK`-keyed action loops migrate to `A::COMPONENTS`

`ddpg_agent.rs:288-289, 299, 305, 339-340`; `td3_agent.rs:329-330, 340, 346,
381-382`; `sac_agent.rs:369-370, 401-402`; and
`rlevo-test-support/src/baseline.rs:41` (`uniform_bounded`'s `(0..AR)` loop).
`uniform_bounded<const AR: usize, A: BoundedAction<AR>>` keeps its signature
and loops `0..A::COMPONENTS` instead.

Agent bound fields change from `[f32; DA]` to `&'static [f32]`
(`ddpg_agent.rs:152-153`, `td3_agent.rs:180-181`, `sac_agent.rs:188-189`).

### 6. The DDPG/TD3 scalar target clip is fixed **in this ADR's scope**

Not deferred. The trigger condition for the scalar collapse is identical to the
trigger condition for the main bug — the first asymmetric multi-component
`BoundedAction` impl — so deferring it would mean landing a component-correct
trait alongside a known-wrong target clip, and then having to *gate which impls
are allowed to land* to keep the wrongness latent. That coupling is worse than
the implementation cost.

Both agents replace `self.low[0]` / `self.high[0]` with per-component bound
tensors, built **once at agent construction** (the device is available there)
and shaped `[1, C]` for broadcast against the `[batch, C]` action tensor:

- DDPG (`ddpg_agent.rs:473-478`): `target_actions.max_pair(low_t).min_pair(high_t)`
  in place of `.clamp(low_scalar, high_scalar)`.
- TD3 (`td3_agent.rs:521-533`): `smoothed_target_action` changes its `low: f32,
  high: f32` parameters to the two bound tensors and applies `max_pair`/
  `min_pair`. Its **noise** clip (`noise_clip`) stays a scalar — that is a
  bound on the smoothing noise magnitude, not on the action space, and is
  scalar in the paper too [5].

The comments citing "CleanRL convention" are removed; CleanRL's usage is
`low[0]/high[0]` only because its reference envs are symmetric, not as a
deliberate departure from Eq. 14.

SAC needs no equivalent change: it has no bound-clipping target path (the
squashed-Gaussian `tanh` handles bounding), and its `act()` clamp at
`sac_agent.rs:401-402` is already per-component — it only needs the
`RANK → COMPONENTS` fix from §5.

### 7. The five blocked impls land as the final step of this work

`BipedalWalkerAction`, `CarRacingAction`, `LunarLanderContinuousAction`,
`ReacherAction`, and `SwimmerAction` get `BoundedAction<1>` impls in the same
change series, as its last step. Rationale: a component-correct trait with zero
multi-component implementors is unfalsifiable. `CarRacingAction`
(`[-1,0,0] .. [1,1,1]`) is specifically the regression witness for §6 — it is
the only action in the workspace whose components disagree on their bounds, and
therefore the only one that can distinguish a correct per-component target clip
from the scalar collapse. Landing the fix without it would leave §6 untested.

They land as a separate, final PR so the mechanical trait migration and the
agent-semantics change can be reviewed independently.

### 8. Convention A/B: not retired; `shape()` is documented as never a source of component count

ADR 0038 asked the follow-up ADR to decide whether Convention B
(`ContinuousActionTest`: `Action<3>`, `shape() == [1,1,1]`, 3 components)
should be retired in favour of the uniform rank-1 `[C]` form. **Decision: it
stays permitted.** The motive 0038 recorded for retiring it was that
Convention A would make `shape().iter().product()` a viable derivation of
component count, removing the need for a const. That motive is now moot in both
directions: `COMPONENTS` exists, this ADR keys bounds on it, and after this
change **nothing in the workspace derives a component count from `shape()`**.
Retiring Convention B would be a breaking change to an in-trait test type
bought with no correctness gain.

What *is* added is the prohibition itself: the `Action::shape()` docs state
that `shape()` describes axis cardinality and is **never** a source of
component count — use `ContinuousAction::COMPONENTS`. This turns the
0038 Convention-A/B trap from an undocumented hazard into a stated rule.

### 9. Breaking-change surface and migration

`BoundedAction` is publicly re-exported on two paths — `rlevo_core::action::BoundedAction`
(`rlevo-core/src/lib.rs:122`) and the facade prelude (`crates/rlevo/src/lib.rs:84`)
— so this is a breaking change on the public API. It is accepted: the project is
alpha with no external consumers, and the alternative is shipping a trait whose
signature contradicts its own documentation.

The migration for a downstream implementor is mechanical and compiler-guided:
change `-> [f32; R]` to `-> &'static [f32]` and `[x, y]` to `&[x, y]`. Every
non-conforming impl is a type error, not a silent behaviour change. Downstream
*callers* that merely bound `A: BoundedAction<AR>` need no change at all;
callers that index `A::low()[i]` need no change either (slice indexing is
identical); only callers that move the array by value (`let arr: [f32; R] = A::low();`)
break, and there are none in-workspace.

## Consequences

### Positive

- `BoundedAction`'s signature finally matches its documented contract; the five
  blocked continuous-control environments become implementable.
- The exploration-noise layer, already slice-shaped and per-component correct,
  is now fed by a trait of the same shape — the impedance mismatch disappears
  rather than being bridged.
- The TD3/DDPG target clip becomes Eq.-14-correct for asymmetric spaces, fixing
  a defect that neither #253 nor ADR 0038 had named.
- Zero change to the 12 `A: BoundedAction<DA>` generic-plumbing sites and zero
  change to trait arity: the diff is concentrated in the impls and the ~11
  indexing sites, where the semantics actually change.
- No allocation is introduced on any hot path.

### Negative / accepted costs

- **Compile-time length agreement between `low()` and `high()` is lost.**
  Replaced by a contract test plus a construction-time `assert_eq!` in each
  agent. See §Alternatives A for why this costs less than it appears to.
- **Breaking change on two public export paths.** Mechanical, compiler-guided,
  acceptable in alpha (§9).
- **A bounds tensor is now built per agent** (two small `[1, C]` tensors, once
  at construction). Negligible memory; one extra device allocation per agent.
- **`max_pair`/`min_pair` replace `clamp`** in the DDPG/TD3 target path — two
  broadcast elementwise ops instead of one fused scalar clamp. The cost is not
  measured here and no multiple is claimed; the operation is elementwise over a
  `[batch, C]` tensor either way, so the expected difference is small relative
  to the surrounding critic forward pass.
- **`&'static` forecloses computed bounds** without an `OnceLock`/`Box::leak`
  escape hatch. Nothing needs it today, and the capability the old docs claimed
  (per-instance config-derived bounds) was never actually reachable (§3).

### Neutral

- Convention B survives (§8); `ContinuousActionTest` needs only the return-type
  edit.
- ADR 0038 is untouched and remains accepted; this ADR discharges the follow-up
  it named.

## Alternatives considered

- **A. Second const generic: `BoundedAction<const R: usize, const C: usize>`
  returning `[f32; C]`.** Rejected. It appears to preserve compile-time safety,
  but the guarantee it actually buys is only that `low()` and `high()` agree
  with *each other* and with `C`. The invariant that matters —
  `C == Self::COMPONENTS`, i.e. that the bounds match what `from_slice` will
  accept — is **not** expressible: there is no way to write
  `where C == Self::COMPONENTS` on stable, and an associated-const equality
  bound is not available. So the obligation would be discharged by exactly the
  same contract test as Option B, and Option A's extra compile-time strength
  reduces to "the two bound arrays are the same length as a number the
  implementor typed twice." For that it costs: a change to the trait's generic
  arity, edits to all 12 `A: BoundedAction<DA>` plumbing sites (three agents ×
  three sites, three `train.rs`, plus `baseline.rs`), a breaking arity change on
  both public export paths, and a second const parameter that every future
  bound must thread. Additionally, a wrong `C` would be *silently* wrong (bounds
  of the wrong length, agent panics later) exactly as in Option B — so it does
  not even improve the failure mode. Paying a materially larger migration for a
  guarantee that stops short of the real invariant is a bad trade.

- **B-variant: `Vec<f32>`.** Rejected in favour of `&'static [f32]`: identical
  type-level properties, but allocates on every call including inside
  `act()`/`learn_step`, and forces `Vec<f32>` agent fields with clones. There is
  no impl that needs the ownership.

- **B-variant: `Box<[f32]>`.** Rejected for the same allocation reason as
  `Vec<f32>`, with worse ergonomics (no `Copy`, awkward to store) and no
  compensating benefit.

- **`&'static [Bounds]` (per-component ADR 0027 `Bounds` newtype).**
  Attractive on paper: it makes `low[i] < high[i]` valid by construction per
  component, which is half the trait's invariant. Rejected because every
  consumer wants the two bounds as **separate parallel slices** — `exploration.rs`'s
  `apply(mean, low, high, ..)` takes `&[f32]` twice, and the DDPG/TD3 bound
  tensors are built one per side. A `&[Bounds]` return would force materialising
  two `Vec<f32>`s at each of those sites, reintroducing the allocation this ADR
  avoids, in exchange for an invariant the contract test already covers.
  Revisit only if a consumer appears that wants the paired form.

- **`[f32; Self::COMPONENTS]` directly.** Not available: requires
  `generic_const_exprs`, re-verified on `rustc 1.97.0` (see Context). The
  workspace pins stable in `rust-toolchain.toml`; adopting a nightly-only
  feature for one trait signature is far out of proportion.

- **Do nothing / document the limitation.** Rejected: the limitation is that the
  five most-wanted continuous-control environments cannot be used with the three
  continuous-control agents the workspace ships. That is not a documentable
  caveat, it is the feature being blocked.

- **Fix only `BoundedAction`, defer the DDPG/TD3 scalar target clip to a
  follow-up issue.** Seriously considered — it is the smaller change and would
  keep this ADR to one seam. Rejected on §6's reasoning: the deferral is only
  safe if no asymmetric multi-component impl lands, which means the follow-up
  issue would have to *block* §7, and a deferral that gates other work is not
  really a deferral. Fixing both together also means `CarRacingAction` serves as
  the regression witness for both halves at once.

## References

- Issue #253 — `BoundedAction` `low`/`high` keyed on rank, not component count.
- ADR [0038](0038-continuous-action-components-const.md) — added
  `ContinuousAction::COMPONENTS`; named this `BoundedAction` gap in its
  "Explicitly out of scope" section, including the blast-radius list of
  `A::RANK`-consuming agent sites and the Convention A/B question §8 answers.
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) — the `Bounds` newtype
  considered and rejected as the return shape here.
- ADR [0052](0052-hostrow-supertrait-splits-layout-from-backend.md) —
  independently found `[usize; Self::RANK]` dead on stable; same obstacle.
- Issue #100 — the original rank-vs-component panic, one layer down.
- [1] Farama Gymnasium — BipedalWalker action space (`Box(-1.0, 1.0, (4,), float32)`).
  https://gymnasium.farama.org/environments/box2d/bipedal_walker/
- [2] Farama Gymnasium — CarRacing (continuous) action space
  (`Box([-1,0,0], [1,1,1], (3,), float32)`).
  https://gymnasium.farama.org/environments/box2d/car_racing/
- [3] Farama Gymnasium — LunarLander (Continuous) action space (`Box(-1, +1, (2,))`).
  https://gymnasium.farama.org/environments/box2d/lunar_lander/
- [4] Lillicrap et al., "Continuous control with deep reinforcement learning"
  (DDPG), arXiv:1509.02971, Eq. 7 — elementwise `tanh` action bounding.
- [5] Fujimoto, van Hoof, Meger, "Addressing Function Approximation Error in
  Actor-Critic Methods" (TD3), arXiv:1802.09477, Eq. 14 — target-action clip
  against the `Box(low, high)` vectors; scalar noise clip is separate.
- [6] Haarnoja et al., "Soft Actor-Critic Algorithms and Applications",
  arXiv:1812.05905, Appendix C (elementwise `tanh` with per-index log-prob
  correction) and Appendix D Table 1 (entropy target `−dim(A)`).
- Burn 0.21 `burn-tensor/src/tensor/api/orderable.rs`: `clamp` (:986),
  `clamp_min` (:1022), `clamp_max` (:1054) are scalar-only; `max_pair` (:722),
  `min_pair` (:950) take tensor operands.
- Code: `crates/rlevo-core/src/action.rs:380-392` (the trait);
  `crates/rlevo-reinforcement-learning/src/algorithms/ddpg/exploration.rs:43-51`
  (the already-correct slice consumer);
  `crates/rlevo-reinforcement-learning/src/algorithms/ddpg/ddpg_agent.rs:473-478`
  and `.../td3/td3_agent.rs:521-533` (the scalar target-clip collapse).
