---
project: rlevo
status: active
type: decision
date: 2026-10-18
tags: [adr, decision, architecture, rlevo-core, environment, snapshot, reward, trait-pruning, breaking-change]
---

# ADR 0090: The environment contract keeps only what generic code reads

## Status

**Accepted (2026-10-18).**

Supersedes these clauses. The rest of each ADR stands.

- ADR [0003](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md),
  "Scope decision: conservative dead-code policy". The trait shells it kept
  for features not yet written are deleted (Decision 6); a planned feature
  is no reason for a `pub trait` (Decision 8).
- ADR [0011](0011-lift-construction-off-environment-trait.md), its
  Implementation section: the `ConstructableEnv` trait. Construction stays
  off `Environment`, as its Decision has it; the environments implement
  `Default`, the shape its Decision listed as Option 2 (Decision 4 here).
- ADR [0019](0019-observable-projection-trait.md), already superseded in its
  framing by ADR 0047: the `Observable<OR>` trait itself is deleted.
- ADR [0020](0020-synthetic-pixel-over-grid-env.md) Decision 5, the dual
  `State<1>` + `Observable<3>` impls on `PixelGridState`. The state type has
  an inherent `project` and no trait impl.
- ADR [0042](0042-snapshotbase-carries-optional-metadata.md):
  - Decision 1 is restated on the struct `Snapshot<O>`: the optional
    `metadata` field and the `with_metadata` builder stand; the overridden
    `Snapshot::metadata()` accessor does not exist, since there is no trait to
    override and the field is read directly.
  - Decision 2, the `LocomotionSnapshot` and `LunarLanderSnapshot` aliases.
    They are deleted, with the grid family's `GridSnapshot` and
    `GoToDoorSnapshot`.
- ADR [0047](0047-sensor-relocates-emission-model-to-environment.md):
  Decisions 1 to 4 (the `Sensor` trait, its signature, its generic surface,
  and `Observable` as the helper it delegates to) and Decision 6
  (`BeliefState` repointed onto `Sensor`). Its placement stands: the emission
  model lives on the environment, not on the state. Decision 5, the grid
  family's shared `build_snapshot`, stands.
- ADR [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md)
  Decision 1: `agent_direction` is a `Direction`, not an
  `Option<Direction>`. Its source of `None` was the decoder ADR 0089
  deleted.
- ADR [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md)
  Decision 5, last bullet, is answered: `TransitionDynamics`, `BeliefState`,
  `StateAggregation` and `MultiDiscreteAction` are deleted here.

ADR [0026](0026-shared-config-validation-convention.md) is restored, not
superseded: its fail-first `validate` is again the whole of `Validate`. The
`validate_all` method and the `Violations` accumulator, added later without an
ADR, are deleted (Decision 7).

## Context

`Environment` is the seam an outside author implements to bring a problem to
`rlevo`, and the first thing the `cargo generate` templates of the next steps
will stamp. At `0d29f87` an author implemented, to reach a report:

- `Environment<R, SR, AR>` with five associated types: `StateType`,
  `ObservationType`, `ActionType`, `RewardType`, `SnapshotType`;
- `State<SR>`, `Observation<R>` and `Action<AR>`;
- `Sensor<OR, AR, SR>`, by convention on every environment;
- `ConstructableEnv`, by convention on every environment;
- `Snapshot<R>` through `SnapshotBase<R, Obs, Rew>`, and `Reward` through
  `ScalarReward`.

ADR 0081's test is that a public item needs an implementor and a consumer.
Read against the workspace (counts are lines from
`rg -c <pattern> crates xtask --glob '*.rs'`, summed, all targets):

- **`State`, `StateType` and `SR`.** 39 `impl State<N> for` lines, 57
  `type StateType` lines. No generic code outside `rlevo-core` bounds on
  `State`; no method of `Environment` mentions `StateType`; nothing reads
  `E::StateType`. Inside core the five bounds were `Environment::StateType`
  and four traits deleted here. Four test fixtures said in a comment that
  their `State` impl existed only to satisfy `StateType`.
- **`Reward`.** One production implementor, `ScalarReward`. The evaluator
  already required `RewardType = ScalarReward`; the ten RL `train` functions
  carried a `Rew: Reward + Copy` parameter whose only use was to read the
  reward as an `f32`.
- **`Snapshot`.** One implementor, `SnapshotBase`, and `rules.md` forbade a
  second. `TimeLimit` had to bind its inner environment's
  `SnapshotType = SnapshotBase<..>` to set `status` in place.
- **`Sensor`.** 42 impl lines, no bound, no `dyn`, and every call made on
  `self` from the environment's own `reset` and `step`. Its associated types
  were not tied to `Environment`'s, and its ranks were ordered differently
  (`<OR, AR, SR>` against `<R, SR, AR>`).
- **`ConstructableEnv`.** 35 environment impls and one forwarding impl on
  `TimeLimit`, whose only reason was "so generic `E: ConstructableEnv` code
  composes"; no such code existed. All 54 calls of `new(bool)` passed
  `false`. The harness builds environments through `EnvFactory`, a closure
  from a trial's seed.
- **`Observable`.** One implementor, `PixelGridState`, no bound.
- **Placeholders for unwritten features.** `BeliefState`, `StateAggregation`,
  `TransitionDynamics` and `MultiDiscreteAction` had no implementor, no
  bound and no caller; `PhysicsSnapshot` with `RapierWorld::snapshot` and
  `restore` had one caller, its own unit test.
- **Members nothing calls.** The action traits' `random()` methods, which
  drew from an unseeded thread-local generator, and their overrides;
  `DiscreteAction::enumerate`; `ContinuousAction::clip`, required of every
  continuous action and called only by tests; the three `RANK` constants;
  `Bounds::clamp_slice`; `combinations` and `checked_combinations`, which
  `rlevo-core`'s own docs described as having zero consumers;
  `Validate::validate_all` with `Violations`, used by one config whose
  `validate` kept the first error and dropped the rest; two `StateError`
  variants and `EnvironmentError::IoError`, which nothing constructs.
- **`LocomotionBackend`.** One implementor, `Rapier3DBackend`. The four
  locomotion environments took `B: LocomotionBackend = Rapier3DBackend`, but
  every impl was written for `X<Rapier3DBackend>` and every call spelled
  `Rapier3DBackend::..`. Four `*Rapier` aliases named the default.

## Decision

### 1. `Environment<OR, AR>` has two associated types

```rust
pub trait Environment<const OR: usize, const AR: usize> {
    type ObservationType: Observation<OR>;
    type ActionType: Action<AR>;

    fn reset(&mut self) -> Result<Snapshot<Self::ObservationType>, EnvironmentError>;
    fn step(
        &mut self,
        action: Self::ActionType,
    ) -> Result<Snapshot<Self::ObservationType>, EnvironmentError>;
}
```

`State`, `Environment::StateType` and the state rank `SR` are deleted. What an
environment keeps between steps is a plain type it owns. The rank parameters
remain because they are how the trait bounds its two associated types.

### 2. The snapshot is a struct, and it is read through its fields

```rust
pub struct Snapshot<O> {
    pub observation: O,
    pub reward: ScalarReward,
    pub status: EpisodeStatus,
    pub metadata: Option<SnapshotMetadata>,
}
```

`running`, `terminated`, `truncated` and `with_metadata` are its methods.
The `Snapshot` trait, `SnapshotBase` and `Environment::SnapshotType` are
deleted. The struct has no rank parameter and no bound on `O`: the
environment already bounds its observation type, and a bound on the struct
would be repeated wherever it is named. Its eight accessors are not kept:
`snap.observation`, `snap.reward`, `snap.status` and `snap.metadata` are the
one way to read each fact, and `snap.status.is_done()` is `EpisodeStatus`'s.

The four step-result aliases (`GridSnapshot`, `GoToDoorSnapshot`,
`LunarLanderSnapshot`, `LocomotionSnapshot`) are deleted:
`Snapshot<GridObservation>` is as short, and the first collided with the
render payload of the same name.

### 3. The reward is `ScalarReward`

The `Reward` trait and `Environment::RewardType` are deleted. `ScalarReward`
carries the finiteness obligation the trait documented. A vector reward, if
one is needed, returns as a new type beside it; named reward components
already travel in `SnapshotMetadata`.

### 4. The emission model and construction are patterns, not traits

`Sensor`, `ConstructableEnv` and `Observable` are deleted.

- **The emission model** is the environment's own: typically an inherent
  method that `reset` and `step` call. It lives on the environment, so it can
  read world context a state value does not hold (ADR 0047). A stochastic
  observation is drawn there from the environment's own generator (ADR
  0029). Observation variants of one environment are a type parameter on it;
  noise, latency and history are wrappers beside `TimeLimit`. Each
  environment's `Sensor` impl became private inherent methods with the same
  bodies; `PixelGridState::project` is public, because tests and benches
  call it.
- **Construction** is not part of the contract (ADR 0011). An environment
  enters the evaluation harness through `EnvFactory`. The workspace's 35
  environments implement `Default` with the body `new(false)` had; the render
  flag stays a config field. `rules.md`'s infallible-`Default` rule holds,
  because a default config validates.

### 5. The grid facing is a `Direction`

`GridObservation::agent_direction` and `GoToDoorObservation::agent_direction`
are `Direction`. Every observation is built with the agent's facing, and no
code produces `None`.

### 6. Five seams with no implementor are deleted

`BeliefState`, `StateAggregation`, `TransitionDynamics`, `MultiDiscreteAction`,
and `PhysicsSnapshot` with `RapierWorld::snapshot` and `restore`. Each returns
in the change that writes the example that implements it, with the signature
the redesign specified for it:

- `BeliefState::sample` takes the caller's generator;
- `TransitionDynamics` is a sampling model, `step(&mut self, s, a, rng)` with
  the reward and status beside the next state, plus `reward(s, a, s')`;
- `StateAggregation` is generic over the agent's input type, with no `State`
  bound;
- `MultiDiscreteAction` returns with the chess move encoding;
- `PhysicsSnapshot` returns as `checkpoint` / `rewind` on the Box2D
  environments.

### 7. Members nothing calls are deleted

`DiscreteAction::random` and `enumerate`; `ContinuousAction::random` and
`clip`; the `RANK` constants of `Observation`, `State` and `Action`;
`Bounds::clamp_slice`; `util::combinations` and `checked_combinations`;
`Validate::validate_all` and `Violations` (`CmaEsConfig::validate` returns the
first violation, as it did); `StateError::InvalidShape` and `InvalidSize`;
`EnvironmentError::IoError`. Each returns with its first caller. Ten action
impls lose `clip`; seven lose a `random` override.

`EnvironmentError` is `#[non_exhaustive]`, so an I/O variant can return
additively with the first environment that loads data. Until then an
environment that does maps the failure into an existing variant.

### 8. A trait's reason to exist is generic code that reads it

The test ADR 0081 applies to every public item is the standing test for a
`pub trait` in a published crate: at least two diverging implementors and a
generic consumer, or the seam an audience is asked to implement, or a
coherence or associated-item need. A trait with one legal implementor is a
type, and a feature not yet written is no reason for one: the trait returns
with the code that implements it.

### 9. One simulator, no backend trait

`LocomotionBackend` and the `B` parameter of the four locomotion environments
are deleted, with the four `*Rapier` aliases. The trait's five methods are
associated functions of `Rapier3DBackend`, documented with the contract the
trait carried, so every call site is unchanged. A trait returns with a second
simulator.

## Consequences

### Positive

- An environment author implements `Observation`, `Action` and
  `Environment<OR, AR>` with two associated types, and returns a
  `Snapshot`. The `byoe-1-thermostat` template's environment file loses a
  `State` impl, three associated-type lines and the `SnapshotBase` spelling;
  the tank farm's also loses `clip` and a `FINDING:` comment about
  `random()`.
- `TimeLimit<E>` is an `Environment` for every `E`, with no binding on the
  snapshot type.
- The ten RL `train` functions lose two parameters (`SR` and `Rew`), the
  harness taps and the evaluator lose `SD`, and hybrid's bounds lose a rank.
- Random actions are drawn from a seeded generator at the caller
  (`CarRacingAction::random_valid` takes one), not from a thread-local one
  inside the contract crate.

### Negative / costs

- **Breaking for every implementor and every reader:**
  - Every `Environment` impl loses three associated types and a rank, and
    spells its return type `Snapshot<Obs>`.
  - Every read of a snapshot changes from a method to a field.
  - Code that named `State`, `Reward`, `Sensor`, `ConstructableEnv`,
    `Observable`, a deleted seam or a deleted member stops compiling.
  - `ConstructableEnv::new(false)` becomes `Default::default()`.
- The locomotion and Box2D states keep a `last_obs` field whose reader was
  `State::is_valid`. Tests read it; nothing else does.
- A test that pinned the panic-contract guard's wrapped-impl-header handling
  on a live file now uses a fixture: no impl header in the workspace wraps
  once the state rank leaves.

### Neutral

- The package graph of `rlevo-core` is unchanged; this ADR removes items,
  not dependencies.
- `StateError` stays in `rlevo_core::state` with its one constructed variant;
  its users are environment constructors that still refuse an illegal state.

## Alternatives considered

- **Keeping `State` as documentation of the MDP's state set.** It would ask
  every author for an impl nothing reads. Rejected.
- **Keeping `SnapshotBase` as an alias, or `RewardType` and `SnapshotType`
  with defaults.** Associated type defaults are unstable, and an alias would
  leave every author's source unchanged and nothing simplified. Rejected.
- **`Snapshot<R, O: Observation<R>>`.** The rank is spelled at every use for a
  bound the environment already states. Rejected.
- **Keeping the accessors beside the public fields.** Two ways to read every
  fact. Rejected; so was making the fields private, which loses
  destructuring and forces `TimeLimit` to rebuild a snapshot to truncate it.
- **A plain `f32` reward instead of `ScalarReward`.** A newtype is no harder
  to widen later. Rejected.
- **An inherent `new(render: bool)` on each environment.** It keeps a flag no
  caller sets in 35 signatures. Rejected.
- **A `pruned-contract` scan in the BYOE-1 probe** that refuses a template
  writing a deleted name. Once a name is deleted, a template that writes it
  does not compile, and step 3 reports that. Rejected; the thermostat as
  written before this change was compiled against it once, and fails.

## References

- `crates/rlevo-core/src/environment.rs`: `Environment`, `Snapshot`,
  `EnvironmentError`.
- `crates/rlevo-core/src/reward.rs`: `ScalarReward` and the finiteness
  obligation.
- `crates/rlevo-environments/src/wrappers/time_limit.rs`.
- `crates/rlevo-environments/src/locomotion/backend/`.
- `xtask/probes/byoe-1-thermostat/` and `xtask/probes/byoe-1-tank-farm/`.
- ADRs [0003](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md),
  [0011](0011-lift-construction-off-environment-trait.md),
  [0019](0019-observable-projection-trait.md),
  [0020](0020-synthetic-pixel-over-grid-env.md),
  [0026](0026-shared-config-validation-convention.md),
  [0029](0029-host-rng-seeding-convention.md),
  [0042](0042-snapshotbase-carries-optional-metadata.md),
  [0047](0047-sensor-relocates-emission-model-to-environment.md),
  [0061](0061-optional-facing-and-tensorconvertible-no-fabrication.md),
  [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md),
  [0089](0089-the-contract-crate-does-not-depend-on-burn.md).
