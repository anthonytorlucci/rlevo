---
project: rlevo
status: active
type: decision
date: 2026-07-17
tags: [core, state, observation, sensor, pomdp, emission-model, rank, issue-329]
---

# ADR 0047: Relocate the emission model `O` to an env-side `Sensor` trait

## Status

Active. Adopted 2026-07-17. Implements issue #329.

**Supersedes [ADR 0019](0019-observable-projection-trait.md)** (the
`Observable<OR>` trait as the home for observation modality change): observation
production no longer lives on `State` at all, so the framing of `Observable` as
"the typed home for the rank-changing projection *instead of* `State::observe`"
is obsolete. `Observable` is retained — demoted to an optional pure-projection
helper — but it is no longer the seam observation flows through.

**Annotates [ADR 0039](0039-box2d-states-own-markov-dofs.md)**: that ADR's
*deferred* target relied on `observe`/`project` being `&self`-only, so
world-derived sensors (BipedalWalker lidar, CarRacing pixels) had to be cached
into the state as "cached sensor snapshots." That cached-sensor requirement is
**obsoleted** by this change — `Sensor::observe(&self = env, …)` reads the world
directly. ADR 0039's shipped Step 1 (encapsulated box2d DOFs, real `is_valid`)
stands; only its deferred cached-sensor rationale is retired.

**Breaking** (alpha): `State::observe` and `State::Observation` are removed, and
`BeliefState`'s associated-type signature changes. Migrated atomically across the
workspace in one PR.

## Context

`State<SR>` carried the observation function:

```rust
pub trait State<const R: usize>: … {
    type Observation: Observation<R>;
    fn observe(&self) -> Self::Observation;
    // …
}
```

In the POMDP tuple ⟨S, A, T, R, Ω, O⟩ the emission model `O: S × A → Π(Ω)` is a
property of the **problem** (the environment), not of a point `s ∈ S` in state
space. Hanging `O` on `State` was a category error that produced three
documented symptoms:

1. **Rank inflation.** `observe()` binds the observation's tensor order to the
   state's own order (`type Observation: Observation<R>` at the same `R`). To
   observe a compact physics state as a pixel frame, the state had to *pretend*
   to be that frame — CarRacing was forced to `Environment<3,3,1>`, its state
   masquerading as a `96×96×3` buffer (research:
   `2026-07-09-issue-117-box2d-state-ownership`).
2. **`&self`-only weakness → the cache hack.** A bare state value cannot see the
   simulator world, so ADR 0039 cached world-derived sensor readings (lidar
   raycasts, rendered frames) into the state purely so `observe(&self)` /
   `project(&self)` could return them.
3. **Dual-impl redundancy.** A modality-changing state implemented a dead
   `State::observe` alongside `Observable::project`.

`Environment<R, SR, AR>` already declares `type ObservationType: Observation<R>`
fully independent of `StateType`, with `R` decoupled from `SR` (ADR 0019 proved
`R != SR` reachable). The missing piece was never the environment contract — it
was *where the emission function lives*. ADR 0019 evaluated (a) a second const
generic on `State`, (b) a blanket impl, and (c) a standalone `Observable`, and
chose (c). It never evaluated **removing `observe` from `State` and giving `O` to
the environment**. This ADR takes that path, so it supersedes 0019 rather than
contradicting a considered rejection.

## Decision

Remove `fn observe()` and `type Observation` from `State`. `State<SR>` now
carries only what genuinely belongs to a point in state space: `RANK`,
`shape()`, `numel()`, `is_valid()`. Add a new env-side trait to
`crates/rlevo-core/src/environment.rs`:

```rust
pub trait Sensor<const OR: usize, const AR: usize, const SR: usize> {
    type Action: Action<AR>;
    type State: State<SR>;
    type Observation: Observation<OR>;

    /// O(a, s') — observation after taking `action` and arriving at `next_state`.
    fn observe(&self, action: &Self::Action, next_state: &Self::State) -> Self::Observation;

    /// Initial observation at episode start, before any action exists.
    fn observe_reset(&self, state: &Self::State) -> Self::Observation;
}
```

Environments implement `Sensor` on their own struct and build snapshots by
calling `self.observe(&action, &next_state)` in `step` and
`self.observe_reset(&next_state)` in `reset`. `&self` is the environment, so a
sensor may read world / simulator geometry directly — no cache needed.

### 1. Signature is `O(a, s')`, with a reset companion

The canonical emission model is a function of the last action and the resulting
state. `reset()` has no preceding action, so rather than widen the stepping
method to `action: Option<&A>` (which pushes a `None` check into every impl and
muddies the canonical signature), the trait carries a dedicated
`observe_reset(&self, &state)`. Envs whose observation ignores the action forward
both methods to one shared body; envs that genuinely use the action keep the two
distinct. **Two-way door:** collapsing to `Option<&A>` later is mechanical.

### 2. Env-*implements*-`Sensor`, not env-*holds*-`Sensor`

For this change the environment struct implements `Sensor` for its own
`(Action, State, Observation)` triple. This needs no new field, gives `&self`
world access for free, and matches the alpha's YAGNI posture. A *held* sensor
(`generic field` or `Box<dyn Sensor>`) would let one environment expose swappable
modalities (vector vs. pixels), which the roadmap's pixel/vector env variants may
want — recorded here as a deliberate **two-way door**. Migrating from
env-implements to env-holds is additive (introduce a field, forward the two
methods to it); nothing in this decision forecloses it.

### 3. Generic surface: three const generics, three associated types

`Sensor<OR, AR, SR>` with `type Action`/`type State`/`type Observation` mirrors
the workspace's existing const-generic trait vocabulary (`Action<AR>`,
`State<SR>`, `Observation<OR>`; cf. `BeliefState<…>`, `LatentState<R, AR>`). Each
associated type needs its rank const to state its bound, so all three const
generics are load-bearing. The three ranks are independent — `OR` is free of
`SR` — which is exactly what retires symptom 1: a compact `State<SR>` can be
observed through a `Sensor` of any `OR` without inflating the state's rank.

### 4. `Observable<OR>` is demoted, not deleted

`Observable` remains for the case where an observation is a **pure function of
the state** — no world context. It is no longer required and no longer the seam
observation flows through; it is a helper a `Sensor` *may* delegate to
(`sensor.observe(..) == next_state.project()`). The `pixel_grid` environment is
the reference for that delegation: its `PixelGridState: Observable<3>` renders the
pixel image, and `PixelGridEnv: Sensor<3,1,1>` delegates to `state.project()`.
The genuine modality-change use `Observable` was introduced for (ADR 0019) is
unchanged; only its *framing* (as the replacement for `State::observe`) is
retired.

### 5. Grid family: the shared `build_snapshot` chokepoint (documented exception)

The eleven gridworld environments share **one** observation projection that is a
pure function of `GridState` (the `7×7×3` egocentric view). Rather than eleven
identical `Sensor` impls, `GridState` implements `Observable<3>` (the projection,
moved verbatim from the old `State::observe` body) and the shared free function
`grids/core/mod.rs::build_snapshot` builds every grid snapshot via
`state.project()`. This is the single structural exception to "the env implements
`Sensor`": the emission model is shared across eleven envs and is state-pure, so
it lives on the state as the demoted-`Observable` projection and flows through one
chokepoint. `GoToDoorEnv` is the grid family's one env that *does* implement
`Sensor`, because its observation is goal-conditioned — it reads the episode
mission colour, which lives on the environment, not the state (ADR 0043).

### 6. `BeliefState` repointed off `State::Observation`

`BeliefState::update` was the sole generic bound consuming `State::Observation`.
It now carries its own associated observation type, mirroring
`HiddenState::Observation` / `LatentState::Observation`:

```rust
pub trait BeliefState<const OR: usize, const SR: usize, const AR: usize,
                      S: State<SR>, A: Action<AR>>: Clone {
    type Observation: Observation<OR>;
    fn update(&self, action: &A, observation: &Self::Observation) -> Self;
    // …
}
```

Like the `Environment` contract, the belief's observation rank `OR` is decoupled
from its state rank `SR`. No `BeliefState` impls exist in the workspace, so this
is a contract-only change.

## Consequences

**Positive**

- `O` sits where the POMDP formalism puts it. A `State` is a point in state
  space; the environment owns the emission model.
- Rank inflation is gone: an env whose state is rank `SR` observes through a
  `Sensor` of any `OR`. CarRacing no longer *needs* to masquerade as
  `Environment<3,3,1>` (the honest compact-state re-model is a separate,
  now-unblocked follow-up, #255).
- The ADR-0039 cached-sensor hack is retired for world-derived sensors:
  BipedalWalker lidar and CarRacing pixels are read from the world via
  `&self = env` at observe time, not stashed on the state first.
  - *One honest consequence for `is_valid`.* box2d states still hold Rapier
    *handles*, not value DOFs — the DOF-ownership refactor is the deferred #256.
    Because the observation cache is gone, ADR 0039's `last_obs`-finiteness clause
    (which stood in for a genuine DOF check on a handle-only state) is dropped with
    it. `LunarLanderState` is unaffected — its `is_valid` catches physics
    divergence through `prev_shaping.is_finite()` (a genuine state field, tested).
    `BipedalWalkerState`, which owns no float field, is left with a handle-only
    `is_valid` until #256 moves its DOFs onto the state as values and restores a
    real finiteness check. This is a transitional, debug-assert-only gap tracked
    on #256; the locomotion family (whose states retain a small `last_obs` field
    feeding `is_valid`) is the mirror case the same #256-class cleanup will reach.
- The dual-impl redundancy is gone: `pixel_grid` has one projection
  (`Observable::project`) that the `Sensor` delegates to, not a dead
  `State::observe` beside it.

**Negative / accepted costs**

- Breaking API (alpha): every `State` impl drops `type Observation`/`observe`;
  every environment gains a `Sensor` impl and routes snapshots through it;
  `BeliefState` callers (none today) adopt the new associated-type shape.
- One documented inconsistency: the grid family builds snapshots via
  `Observable` through `build_snapshot` rather than a per-env `Sensor`
  (decision 5). Justified by the shared, state-pure projection across eleven envs;
  called out here and in the module docs so it is not read as an oversight.
- `Sensor` is a fourth "seam" trait a new env author must know. Mitigated by
  making the common case a two-method impl whose bodies usually match the old
  `observe` body verbatim.

**Neutral**

- `State`'s residual role once networks consume observations (not states) — does
  it still need `shape()`/`numel()`/`TensorConvertible`? — is **explicitly out of
  scope** and deferred to a follow-up (see Open questions). This change removes
  only `observe`/`type Observation`.

## Open questions (deferred, not decided here)

- **State's residual tensor machinery.** With observation production gone, the
  question of whether `State` still needs `shape()`/`numel()`/`RANK`/
  `TensorConvertible` is real but separate. Deferred; filed as its own issue if
  pursued.
- **CarRacing honest re-model** to `Environment<3,1,1>` = compact `State<1>` +
  pixel `Sensor<3, …>`. Unblocked by this change; tracked as #255.
- **Env-holds-`Sensor`** for swappable modalities (decision 2's two-way door),
  if/when a pixel-vs-vector env variant needs it.

## Alternatives considered

**Keep `State::observe`, add the rank escape via `Observable` only (ADR 0019's
path).** Rejected: leaves `O` on `State` (the category error), keeps the
`&self`-only cache hack for world sensors, and keeps the dual-impl redundancy.
0019 solved rank change but not the ownership of `O`.

**`observe(&self, action: Option<&A>, next_state)` instead of a reset
companion.** Rejected as noisier: every impl carries a `None` branch and the
canonical `O(a, s')` signature is muddied. The reset companion keeps the stepping
method clean; collapsing later is a two-way door (decision 1).

**Env-*holds*-`Sensor` now (`Box<dyn Sensor>` / generic field).** Rejected for
this change as premature (YAGNI, alpha): it adds a field and indirection for a
swappable-modality capability nothing needs yet. Recorded as a two-way door
(decision 2).

**Per-env `Sensor` for all eleven grid envs.** Rejected: eleven identical impls
of a state-pure projection, versus one `Observable<3>` + one shared
`build_snapshot` (decision 5). The env-implements rule earns its keep where the
emission model is env-specific (box2d world sensors, GoToDoor's mission), not
where it is shared and state-pure.

## References

- Sutton & Barto, *Reinforcement Learning* §3.1, §17.3 (the POMDP tuple and the
  emission model `O`).
- Issue #329 — the design tracker and locked decisions.
- [ADR 0019](0019-observable-projection-trait.md) — superseded (the `Observable`
  modality-change trait).
- [ADR 0020](0020-synthetic-pixel-over-grid-env.md) — the `pixel_grid`
  `Observable` consumer, now the "`Sensor` delegates to `Observable`" reference.
- [ADR 0039](0039-box2d-states-own-markov-dofs.md) — annotated (its deferred
  cached-sensor target is obsoleted); its shipped Step 1 stands.
- [ADR 0043](0043-grid-observation-contract.md) — `GoToDoorEnv`'s goal-conditioned
  observation, now its `Sensor` impl.
- `crates/rlevo-core/src/environment.rs` — the `Sensor` trait + in-source tests.
- `crates/rlevo-core/src/state.rs` — the repointed `BeliefState`, the demoted
  `Observable`.
- `crates/rlevo-core/src/base.rs` — `State` with observation production removed.
- Research: `2026-07-09-issue-117-box2d-state-ownership`,
  `rank-vs-dimensionality-in-pomdp-observations`.
