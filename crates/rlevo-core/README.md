# rlevo-core

![Alt Text](https://raw.githubusercontent.com/anthonytorlucci/rlevo/main/rlevo-logo.png)

Core abstractions for the `rlevo` Deep Reinforcement Learning with Evolutionary Optimization library.

This crate defines the foundational trait hierarchy and concrete types shared across all other `rlevo` crates. It deliberately contains no learning algorithms or environment implementations — only the interfaces those depend on.

## Overview

`rlevo-core` models a reinforcement learning problem as a typed interaction loop:

```
Environment ──reset/step──► Snapshot ──observation──► Agent ──action──► Environment
                                   └──reward──────────────────────────────────────┘
```

The rank (number of tensor axes) of every state, observation and action is enforced at compile time via const generics, so rank mismatches become type errors rather than runtime panics.

## Modules

### `base` — Core Traits

The primitive building blocks of the RL abstraction.

| Trait | Description |
|---|---|
| `State<R>` | Full MDP state (`shape()`, `is_valid()`, `numel()`); the environment's `Sensor` produces the observation |
| `Observation<R>` | What the agent perceives (may be a partial view of state) |
| `Action<R>` | Base action constraint (`is_valid()`) |
| `Reward` | Scalar feedback signal; must be `Clone + Add + Into<f32> + Debug` and provide `zero()` |
| `HostRow<R>` | An observation's flat `f32` row (`write_host_row`), in the shape `shape()` reports |
| `TensorConvertible<R, B>` | With the `burn` feature: uploads a `HostRow` as a Burn tensor; implemented for every `HostRow` |
| `TransitionDynamics<SR, AR, S, A>` | Deterministic state-transition function |

To feed an observation to a neural-network agent, implement `HostRow` on it. `TensorConvertible` comes with the `burn` feature, which `rlevo-reinforcement-learning` turns on, and needs no impl of its own. Without the feature the crate does not depend on Burn.

### `action` — Action Spaces

Three orthogonal action abstractions cover the standard taxonomy:

| Trait | Use Case |
|---|---|
| `DiscreteAction<R>` | Finite enumerable actions (e.g., 4-directional movement). Provides `from_index`, `to_index`, `enumerate`, `random`. |
| `MultiDiscreteAction<R>` | Multiple independent discrete dimensions (e.g., direction $\times$ attack). Combinatorial enumeration and uniform random sampling. |
| `ContinuousAction<R>` | Real-valued vectors with `clip`, `from_slice`, `as_slice`, `random` (uniform in \[$-$1, 1)). |
| `BoundedAction<R>` | Extends `ContinuousAction<R>` with per-component `low()` / `high()` bounds. |

### `state` — Advanced State Abstractions

Extensions to `State<R>` for partial observability and hierarchy:

| Trait | Purpose |
|---|---|
| `Observable<OR>` | Optional pure state→observation projection (`project()`) that a `Sensor` may delegate to (ADR 0047); `OR` is independent of `SR` (`OR == SR` included), so it can also change modality, e.g. RAM bytes viewed as pixels (ADR 0019) |
| `BeliefState<OR, SR, AR, S, A>` | POMDP belief distribution over latent states, updated from action `A` and an `Observation<OR>` |
| `StateAggregation<SR, S>` | Maps concrete states to abstract representatives for hierarchical RL |

### `environment` — Environment Protocol

| Item | Description |
|---|---|
| `Environment<R, SR, AR>` | Core trait: `reset()`, `step(action)` |
| `Sensor<OR, AR, SR>` | Env-side emission model $O(a, s')$: `observe(action, next_state)` and `observe_reset(state)` produce the agent's observation (ADR 0047) |
| `ConstructableEnv` | Default-construction factory (`new(render)`), lifted off `Environment` per ADR 0011 |
| `Snapshot<R>` | Per-step result: `observation()`, `reward()`, `status()` |
| `EpisodeStatus` | `Running \| Terminated \| Truncated` — distinguishes natural episode ends from step-limit truncations (important for value bootstrapping) |
| `SnapshotBase<R, ObservationType, RewardType>` | Default `Snapshot` implementation with named constructors: `running()`, `terminated()`, `truncated()` |
| `SnapshotMetadata` | Builder for named reward components and 3-D positions (visualization / debugging) |

### `episode` — Episode-Lifecycle Guard

`EpisodeGuard` is the state machine behind `Environment::step`'s post-terminal contract: a step taken after the episode ended returns `EnvironmentError::StepAfterEpisodeEnd` and changes nothing. Every environment holds one. `check()` it first in `step`, `record()` the status of each snapshot emitted, and `reset()` it in `reset`.

### `reward` — Concrete Reward Types

`ScalarReward(f32)` is a thin newtype implementing `Reward`. It covers the vast majority of environments. Custom reward types can implement `Reward` directly.

### `evaluation` — Drive Seam for Self-Paced Work

The seam a benchmark harness drives when the work is *not* an episodic agent/environment interaction. Episodic work needs nothing here: `rlevo-harness` binds on `Environment` directly and infers its rank parameters from the suite (ADR 0076).

Lives in core so `rlevo-evolution` (which implements it) and `rlevo-harness` (which drives it) can share it without a cross-crate dependency — the same argument ADR 0004 made for the `BenchEnv` surface that used to live here, and that ADR 0077 removed.

| Item | Description |
|---|---|
| `GenerationProbe` | Drive seam for self-paced work (evolutionary generation loops): `begin()`, then `advance() -> Option<Metrics>` until the budget is spent |

### `fitness` — Evaluation & Fitness Traits

The agent/landscape side of the benchmark harness, also hoisted into core per ADR 0004. (Note: the `FitnessFn`/`BatchFitnessFn`/`GenomeKind` evolutionary traits now live in `rlevo-evolution` per ADR 0002 — they are **not** in this crate.)

| Trait / Type | Description |
|---|---|
| `BenchableAgent<Obs, Act>` | Minimal inference interface (`act`) for an externally-driven evaluator; deterministic given a harness-owned RNG |
| `FitnessEvaluable` | Optimizer-vs-landscape evaluation, used when the benchmark *is* the fitness function |
| `Landscape` | Self-evaluating numerical landscape (Sphere, Ackley, Rastrigin) carrying both parameters and `f(x)` |
| `MetricsProvider` | Reports method-specific `Metric` values at trial boundaries |
| `Metric` | Method-specific signal emitted by an agent or aggregator |

### `util` — Shared Utilities

| Item | Description |
|---|---|
| `combinations(n, k)` | Binomial coefficient (folded in from the former `rlevo-utils` crate per ADR 0003) |
| `seed::SeedStream` | Reproducible host-side seed source for downstream RNGs |

### `bounds`, `config`, `objective`, `probability`, `rate` — Validated Value Types

Small validated newtypes and helpers shared across config validation in environments and algorithms. Their only external dependencies are `serde` (the newtypes and `ObjectiveSense` derive `Serialize` / `Deserialize`, and deserialising a newtype validates it) and `thiserror` (the error types):

| Item | Description |
|---|---|
| `Bounds` / `BoundsError` | Validated `[low, high]` interval type |
| `Validate` | Trait for config structs to self-check: `validate()` returns the first `ConfigError`, `validate_all()` every one (collected with `Violations`) |
| `ConfigError` / `ConstraintKind` / `Violations` | Structured config-validation error reporting, plus helper checks (`positive`, `in_range`, `ordered`, `distinct`, `nonzero`, `at_least`, `at_most`) |
| `ObjectiveSense` | `Minimize \| Maximize`, used to disambiguate landscape/fitness direction |
| `Probability` / `ProbabilityError` | Validated `[0, 1]` newtype |
| `NonNegativeRate` / `NonNegativeRateError` | Validated non-negative `f32` newtype (learning rates, decay rates, etc.) |

### `agent` — Agent Traits (reserved)

Placeholder module reserved for a future unified agent trait hierarchy (`act` / `learn` / checkpoint). Empty while the unified agent trait hierarchy stabilizes — concrete algorithms currently live in `rlevo-reinforcement-learning` and `rlevo-evolution` and will migrate behind these traits once the API stabilizes.

### `render` — Rendering Abstractions

The rendering surface feeds the two visualisation products (ADR 0013): a live metrics-only `ratatui` TUI and a post-run static-HTML report that replays env state from structured payloads. `rlevo-core` ships zero terminal-side dependencies — the conversion into `ratatui` styles lives downstream in `rlevo-harness`, and the rendering of frames and payloads into HTML in `rlevo-benchmarks-report-client`.

**`ascii` — optional text rendering**

| Item | Description |
|---|---|
| `AsciiRenderable` | Optional debug helper (not a library invariant, per ADR 0013): a grep-friendly text dump for logs, snapshot tests, and the report's legacy `<pre>` fallback |

**`styled` — colour-aware projection** (the report tier renders `StyledFrame`s; the live TUI uses only the `SpanStyle` / `Color` / `Modifier` styles)

| Item | Description |
|---|---|
| `StyledFrame` / `StyledLine` / `StyledSpan` | A small, ratatui-shaped vocabulary for styled text output |
| `SpanStyle` | Foreground/background `Color` plus a `Modifier` set |
| `Color` | ANSI colour enum: the 16-colour palette plus `Indexed(u8)` 256-colour entries |
| `Modifier` | Bitset of `BOLD` / `REVERSED` / `UNDERLINED` / … hints |

**`palette` — semantic colours.** Project-wide named colour constants, each paired with a `*_MODIFIER` companion so meaning is never carried by hue alone (accessibility contract — see the `palette` module docs on hue-redundant signalling).

**`payload` — per-family structured snapshots** (the report tier's rich view). Pure-data snapshot types plus an opt-in `*PayloadSource` trait per family; envs that don't implement one fall back to the ASCII payload.

| Family | Snapshot type | Opt-in source trait |
|---|---|---|
| Landscapes | `Landscape2DSnapshot` (`Point2`) | none: a landscape is not an environment |
| Box2D | `Box2dSnapshot` (`RigidBody2D`, `BodyKind`) | `Box2dPayloadSource` |
| Locomotion | `Locomotion2DSnapshot` | `Locomotion2DPayloadSource` |
| Grid | `GridSnapshot` (`GridTile`, `GridDir`, `GridColor`, `GridDoorState`, `GridAgentMarker`) | `GridPayloadSource` |
| Tabular | `TabularSnapshot` (`TabularGrid`, `TabularCell`, `TabularMarker`, `TabularLayout`, `CardTable`) | `TabularPayloadSource` |
| Classic 2D | `Classic2DSnapshot` (`Classic2DBody`, `Classic2DRole`) | `Classic2DPayloadSource` |

## Design Principles

**Const generics for ranks.** `State`, `Observation` and `Action`, and the traits built on them, carry a const rank `R` (`SR`, `AR`, `OR` where several meet) so that rank mismatches fail at compile time, not at runtime.

**Separation of state and observation.** `State<R>` represents the true MDP state; `Observation<R>` is what the agent receives. This separation makes POMDPs first-class citizens rather than workarounds.

**Terminated vs. Truncated.** `EpisodeStatus` distinguishes a natural terminal state from a step-limit cutoff. Algorithms that bootstrap values (e.g., TD-learning) need this distinction to avoid incorrect target computation.

**Trait-based composition.** No inheritance. New action spaces, state types, and genome kinds are added by implementing the appropriate trait.

**Zero-cost abstractions.** `Copy` reward/state types and const-generic ranks incur no runtime cost.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
rlevo-core = "0.5"
```

The example below uses `EpisodeGuard`, which is on `main` and ships with the next release; the published 0.5.0 has no `episode` module. On `main` the crate also depends on Burn only under its `burn` feature; the published 0.5.0 always depends on it.

### Minimal Environment

```rust
use rlevo_core::{
    base::{Action, Observation, Reward, State},
    environment::{ConstructableEnv, Environment, EnvironmentError, Snapshot, SnapshotBase},
    episode::EpisodeGuard,
    reward::ScalarReward,
};

struct MyEnv { guard: EpisodeGuard, /* ... */ }

impl Environment<1, 1, 1> for MyEnv {
    type StateType = MyState;
    type ObservationType = MyObservation;
    type ActionType = MyAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, MyObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.guard.reset();
        // reset internal state, return initial snapshot
        Ok(SnapshotBase::running(MyObservation::default(), ScalarReward::new(0.0)))
    }

    fn step(&mut self, action: MyAction) -> Result<Self::SnapshotType, EnvironmentError> {
        // first: a step after the episode ended is an error and changes nothing
        self.guard.check()?;
        // apply action, compute reward
        let snapshot = SnapshotBase::terminated(MyObservation::default(), ScalarReward::new(1.0));
        self.guard.record(snapshot.status());
        Ok(snapshot)
    }
}

impl ConstructableEnv for MyEnv {
    fn new(render: bool) -> Self { MyEnv { guard: EpisodeGuard::new(), /* ... */ } }
}
```

### Replay Buffer

Replay lives in `rlevo-reinforcement-learning`, not `rlevo-core` (per ADR 0003 —
replay and metrics moved to where they are consumed). The replay-strategy
seam is `rlevo_reinforcement_learning::replay` (ADR 0050): `ReplayStrategy`, the
default `UniformReplay`, and the opt-in `PrioritizedReplay`. See that crate's
README for the API and the config-level PER opt-in.

## Academic References

The following papers directly informed the algorithms and concepts in this crate:

- Sutton, R. S., & Barto, A. G. (2018). **Reinforcement Learning: An Introduction** (2nd ed.). MIT Press. — foundational framework for the `Environment` / `Snapshot` / `Reward` trait design and the Terminated vs. Truncated episode status distinction.

- Puterman, M. L. (1994). **Markov Decision Processes: Discrete Stochastic Dynamic Programming**. Wiley [acm digital library](https://dl.acm.org/doi/10.5555/528623). — theoretical grounding for `BeliefState` and `TransitionDynamics`.

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or [MIT License](../../LICENSE-MIT) at your option.
