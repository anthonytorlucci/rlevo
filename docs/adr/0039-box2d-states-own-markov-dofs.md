---
project: rlevo
status: active
type: decision
date: 2026-07-09
tags: [box2d, state, markov, rapier, pomdp, observable, encapsulation, issue-117]
---

# ADR 0039: box2d states own their Markov DOFs; CarRacing re-models as `Observable<3>`

## Status

Active. Adopted 2026-07-09. Resolves issue #117. **Step 1 (encapsulation + honest
`is_valid()`) ships in #117; the DOF-ownership refactor is deferred to tracked
follow-ups #255 (CarRacing) and #256 (LunarLander + BipedalWalker).**

## Context

The three box2d environment `State` structs — `CarRacingState`,
`LunarLanderState`, `BipedalWalkerState` — are bundles of Rapier arena handles
(`RigidBodyHandle`, `ImpulseJointHandle`, `ColliderHandle`) into a `RapierWorld`
owned by the *environment*, not by the state. The Markov degrees of freedom
(pose, twist, per-wheel/joint dynamics) live *behind* those handles, read live out
of the world each step; they are not values in the struct.

A handle is a reference into a mutable world, not a value capturing state at time
*t*. This is a structural violation of the Markov property (Sutton & Barto,
*RL: An Introduction* 2nd ed. §3.5: a Markov state "summarizes past sensations …
all relevant information retained"): `#[derive(Clone)]` yields a shallow alias
whose handles are only meaningful alongside the specific world they were taken
from, and dangle after a `reset()`/rebuild. The code review
(`code-review-01-07-2026-env-box2d.md`) flagged this as car_racing §3.1/§3.2 (🔴),
lunar_lander §1.1/§3.1/§3.2 (🔴/🟠), bipedal_walker §3.2 (🟡).

Verified facts that scope the fix:

- `Environment` has **no `state()` accessor** and never calls
  `State::is_valid()`/`observe()` on env state — the runtime loop flows through
  `SnapshotType`. So today the defect is a **contract/testability lie, not a live
  bug**: `is_valid()` has zero callers, and the states are never cloned/serialized
  externally (derives are `Debug, Clone` only; no serde). The dangling-handle-on-
  clone is a **latent hazard**. But the state model is a one-way door per env, so
  the target deserves a recorded decision.
- CarRacing was forced to `Environment<3,3,1>` (its "state" typed as the
  96×96×3 pixel buffer) **only because `Observable<OR>` (ADR 0019) did not exist
  yet** (session 2026-04-17). That constraint is now lifted; `PixelGridEnv`
  (ADR 0020) is the first synthetic `Observable` consumer (#65, closed).

**The load-bearing projection constraint.** Both `State::observe(&self)` and
`Observable::project(&self)` see only the state, never the world. BipedalWalker's
10 lidar dims are raycasts against world geometry, and CarRacing's entire
observation is a rendered frame. Neither can be *recomputed* from a self-contained
state value — they are irreducible projections of the whole physics world. So a
"self-contained Markov state" target does **not** eliminate the cached
observation; it demotes the cache to a *sensor snapshot* that the env computes
during `step` and the state merely returns. That per-env behavior change to the
projection path (and, for CarRacing, a compact dynamical model that does not exist
in the struct today) is a milestone, not a bug fix — hence the split.

Canonical Markov state per env (Gymnasium/Farama; research note
`docs/.private/research/2026-07-09-issue-117-box2d-state-ownership.md`):

| Env | Canonical state | Modality |
|-----|-----------------|----------|
| LunarLander | 8-dim: pose(x,y,angle) + twist(vx,vy,angvel) + 2 leg contacts — obs **==** state | same |
| BipedalWalker | hull pose+twist + 4 joint (angle,speed) pairs + 2 contacts + 10 lidar | lidar is a world-projection |
| CarRacing | hull pose+twist + per-wheel omega/phase/slip + fuel + track progress | pixels-over-physics POMDP |

## Decision

### 1. Target model (deferred to #255/#256)

Each box2d `State` owns its Markov DOFs **as values** (pose + twist +
per-joint/wheel/contact dynamical scalars). Physics/world handles are env
infrastructure and move onto the env "core" struct, normalizing the current
asymmetry (LunarLander stores `ground_handle` on the state; BipedalWalker stores
it on the env struct) to **handles-on-core uniformly**. BipedalWalker's
motor-driven joint handles stay reachable from the stepping code (on core).

### 2. CarRacing re-models as `Observable<3>` (deferred to #255)

CarRacing becomes `Environment<3,1,1>`: a compact `State<1>` (hull pose+twist,
per-wheel omega/phase/slip, fuel, track progress) plus `Observable<3>` projecting
the pixel frame. This lands a second, physics-based production `Observable`
consumer after `PixelGridEnv`.

### 3. World-projected sensors stay cached

BipedalWalker lidar and CarRacing pixels remain computed env-side during
`step`/`reset` and cached; `project()`/`observe()` return the cached value. A
cached sensor snapshot *is* a value and does not reintroduce the handle-aliasing
defect. This reconciles the `&self`-only projection signature (ADR 0019) with
world-derived sensors.

### 4. Step 1 ships in #117 (this ADR's immediate scope)

Contained encapsulation + invariant honesty, no handle relocation, no struct
restructuring, no `Observable`:

- All fields of the three states become **`pub(crate)`** with `#[must_use]` read
  accessors named after each field. Fields stay `pub(crate)` (not private) so the
  existing struct-literal-then-post-hoc-assign build pattern in each `env.rs`
  keeps compiling unchanged.
- Real `is_valid()` per env: handle-validity (`!= ::invalid()`) + finiteness
  (`last_obs.is_finite()`, **and `prev_shaping.is_finite()` for lunar** — the field
  the current check silently skips) + structural invariants (CarRacing:
  `tiles_visited <= total_tiles`, `current_tile.is_none_or(|i| i < total_tiles)`).
- CarRacing `current_tile: usize` → `Option<usize>`, killing the `0`-as-both-
  "tile-zero"-and-"no-tile" sentinel.
- `is_valid()` is **wired into a caller**: a `debug_assert!(self.state.is_valid())`
  at the end of each env's `reset`/`step` (after full assembly + `last_obs` write),
  giving the invariant teeth in debug/test builds at zero release cost. This is
  cross-arch safe — a NaN/handle-validity check is not a chaotic-magnitude bound.
- Delete the redundant `numel()` overrides on **both** lunar (`-> 8`) and bipedal
  (`-> 24`); the default `shape().iter().product()` computes them.
- Honest handle-lifetime doc caveats on each struct, citing #255/#256.

### 5. Constructor deviation from rules §3, recorded

box2d states use `pub(crate)` fields + read accessors + a runtime `is_valid()`
invariant, **without** a validating `try_new`/`new`. This is a deliberate deviation
from rules §3's "offer a `try_new`" wording, justified because: (a) construction is
in-module and *incremental* — the struct is built with `::invalid()` handle
placeholders and filled post-hoc across the world-build path, so a validating
`try_new` could not be the construction entry point (it would reject its own
initial state); and (b) there is no public or `Deserialize`-based construction path
(the rules §3 threat model — external struct-literal construction of invalid
values). `pub(crate)` alone fully closes that surface; invariant enforcement lives
in `is_valid()`, not a constructor.

## Consequences

**Positive**
- (#117) The representable-but-invalid surface shrinks to `pub(crate)`; `is_valid()`
  stops being a rubber stamp and gains a caller; the lunar `prev_shaping` NaN hole
  (it feeds the reward) and the CarRacing `0`-tile sentinel are closed.
- (target) States become genuinely Markov and unit-testable against a real
  invariant; the dangling-handle-on-clone latent hazard is retired when handles
  leave the state.
- (target) `Observable<OR>` gains a second, physics-based production consumer.

**Negative / accepted costs**
- The DOF-ownership refactor is a per-env behavior change to the projection path;
  CarRacing requires a compact dynamical model built from scratch (highest effort).
  Deferred deliberately to #255/#256.
- Lidar/pixels stay cached; `project()`/`observe()` are pure reads of an
  env-populated cache, not pure functions of the owned DOFs. Documented, not a
  regression.
- The three envs are asymmetric in effort (lunar clean, bipedal intermediate, car
  richest); follow-ups are sized accordingly.

**Neutral**
- The `try_new`-omission deviates from rules §3's letter while honoring its intent;
  called out explicitly here so a future reader does not "normalize" it.

## Alternatives considered

- **Do DOF-ownership in #117.** Rejected: converts a contained ~1-file-per-env
  encapsulation PR into a three-env physics-modeling change plus a new trait
  consumer.
- **Keep CarRacing at `Environment<3,3,1>` and only document.** Rejected: entrenches
  "state == pixel buffer," forcing the `State` Markov invariant and `numel()` onto a
  frame; `Observable<3>` is the idiomatic fix now that ADR 0019/0020 exist.
- **Move `project()` onto the environment.** Rejected: contradicts ADR 0019
  (`project` lives on the state type); the cache pattern reconciles the `&self`-only
  signature with world-derived sensors.
- **Fully self-contained bipedal state (recompute lidar in `observe`).** Rejected:
  `observe(&self)` has no world access; lidar is structurally a world-projection.
- **Validating `try_new` per rules §3 letter.** Rejected (decision 5): contradicts
  the incremental invalid-handle-placeholder build pattern; no public/serde
  construction path to guard.

## References

- Issue #117 (this decision); follow-ups #255 (CarRacing `Observable<3>`),
  #256 (LunarLander + BipedalWalker DOF-ownership).
- ADR [0019](0019-observable-projection-trait.md) / [0020](0020-synthetic-pixel-over-grid-env.md) — the `Observable<OR>` seam and its first consumer.
- ADR [0011](0011-lift-construction-off-environment-trait.md) — separate-concern → standalone-trait pattern.
- ADR [0026](0026-shared-config-validation-convention.md) / [0027](0027-bounds-newtype-for-closed-ranges.md) / [0031](0031-probability-rate-newtypes.md) — validation idioms.
- `docs/rules.md` §3 Struct Field Encapsulation; §12 Deferred Work Gets a GitHub Issue.
- Code review `.scratchpad/code-review-30-06-2026/code-review-01-07-2026-env-box2d.md`.
- Research note `docs/.private/research/2026-07-09-issue-117-box2d-state-ownership.md`
  (Gymnasium canonical state definitions; Sutton & Barto §3.5).
