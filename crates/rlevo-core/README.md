# rlevo-core

Core abstractions for the `rlevo` Deep Reinforcement Learning with Evolutionary Optimization library.

This crate defines the foundational trait hierarchy and concrete types shared across all other `rlevo` crates. It deliberately contains no learning algorithms or environment implementations — only the interfaces those depend on.

## Overview

`rlevo-core` models a reinforcement learning problem as a typed interaction loop:

```
Environment ──reset/step──► Snapshot ──observation──► Agent ──action──► Environment
                                   └──reward──────────────────────────────────────┘
```

All dimensions (state, observation, action) are enforced at compile time via const generics, so shape mismatches become type errors rather than runtime panics.

## Modules

### `base` — Core Traits

The primitive building blocks of the RL abstraction.

| Trait | Description |
|---|---|
| `State<R>` | Full MDP state; produces an `Observation<R>` via `observe()` |
| `Observation<R>` | What the agent perceives (may be a partial view of state) |
| `Action<R>` | Base action constraint (`is_valid()`) |
| `Reward` | Scalar feedback signal; must be `Clone + Add + Into<f32> + Debug` and provide `zero()` |
| `TensorConvertible<R, B>` | Bidirectional conversion between domain types and Burn tensors |
| `TransitionDynamics<SR, AR, S, A>` | Deterministic state-transition function |
| `UpdateFunction<Input, Output>` | Generic parameterized update (policy gradient step, etc.) |

`TensorConvertible` is the integration point with Burn: implement it on your state/action types to enable neural-network-based agents.

### `action` — Action Spaces

Three orthogonal action abstractions cover the standard taxonomy:

| Trait | Use Case |
|---|---|
| `DiscreteAction<R>` | Finite enumerable actions (e.g., 4-directional movement). Provides `from_index`, `to_index`, `enumerate`, `random`. |
| `MultiDiscreteAction<R>` | Multiple independent discrete dimensions (e.g., direction × attack). Combinatorial enumeration and weighted random sampling. |
| `ContinuousAction<R>` | Real-valued vectors with `clip`, `from_slice`, `as_slice`, `random` (uniform in \[−1, 1\]). |
| `BoundedAction<R>` | Extends `ContinuousAction<D>` with per-component `low()` / `high()` bounds. |

### `state` — Advanced State Abstractions

Extensions to `State<R>` for partial observability, recurrence, and hierarchy:

| Trait | Purpose |
|---|---|
| `MarkovState` | Asserts the Markov property holds for this state representation |
| `Observable<OR>` | Modality-changing state→observation projection for `OR != SR` (ADR 0019) |
| `BeliefState<SR, AR, S, A>` | POMDP belief distribution over latent states, updated by action `A` |
| `HiddenState<R>` | RNN-style recurrent memory (`update`, `reset`) |
| `LatentState<R, AR>` | World-model compressed representation (`encode`, `predict`, `decode`) |
| `StateAggregation<SR, S>` | Maps concrete states to abstract representatives for hierarchical RL |

### `environment` — Environment Protocol

| Item | Description |
|---|---|
| `Environment<D, SD, AD>` | Core trait: `reset()`, `step(action)` |
| `ConstructableEnv` | Default-construction factory (`new(render)`), lifted off `Environment` per ADR 0011 |
| `Snapshot<D>` | Per-step result: `observation()`, `reward()`, `status()` |
| `EpisodeStatus` | `Running \| Terminated \| Truncated` — distinguishes natural episode ends from step-limit truncations (important for value bootstrapping) |
| `SnapshotBase<D, O, R>` | Default `Snapshot` implementation with named constructors: `running()`, `terminated()`, `truncated()` |
| `SnapshotMetadata` | Builder for named reward components and 3-D positions (visualization / debugging) |

### `reward` — Concrete Reward Types

`ScalarReward(f32)` is a thin newtype implementing `Reward`. It covers the vast majority of environments. Custom reward types can implement `Reward` directly.

### `evaluation` — Benchmark Environment Protocol

A lightweight evaluation surface for the benchmark harness (`rlevo-benchmarks`), moved into core per ADR 0004 so it can be shared without a cross-crate dependency.

| Item | Description |
|---|---|
| `BenchEnv` | Minimal stateful-environment interface an external evaluator drives |
| `BenchStep<Obs>` | Per-step result returned by a `BenchEnv` |
| `BenchError` | Recoverable error wrapping the typed upstream `EnvironmentError` |

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

Small, dependency-free newtypes and helpers shared across config validation in environments and algorithms:

| Item | Description |
|---|---|
| `Bounds` / `BoundsError` | Validated `[low, high]` interval type |
| `Validate` | Trait for config structs to self-check via `Violations` |
| `ConfigError` / `ConstraintKind` / `Violations` | Structured config-validation error reporting, plus helper checks (`positive`, `in_range`, `ordered`, `distinct`, `nonzero`, `at_least`) |
| `ObjectiveSense` | `Minimize \| Maximize`, used to disambiguate landscape/fitness direction |
| `Probability` / `ProbabilityError` | Validated `[0, 1]` newtype |
| `NonNegativeRate` / `NonNegativeRateError` | Validated non-negative `f32` newtype (learning rates, decay rates, etc.) |

### `agent` — Agent Traits (reserved)

Placeholder module reserved for a future unified agent trait hierarchy
(`act` / `learn` / checkpoint). Empty in v0.1.0 — concrete algorithms currently
live in `rlevo-reinforcement-learning` and `rlevo-evolution` and will migrate behind these traits
once the API stabilizes.

### `render` — Rendering Abstractions

The rendering surface feeds the two visualisation products (ADR 0013): a live metrics-only `ratatui` TUI and a post-run static-HTML report that replays env state from structured payloads. `rlevo-core` ships zero terminal-side dependencies — the conversions into `ratatui`/HTML live downstream in `rlevo-benchmarks`.

**Core renderer trait**

| Item | Description |
|---|---|
| `Renderer<E>` | Generic over environment and associated `Frame` type (`String`, `Vec<u8>`, `()`, …) |
| `NullRenderer` | Zero-cost no-op; compiles away entirely |

**`ascii` — optional text rendering**

| Item | Description |
|---|---|
| `AsciiRenderable` | Optional debug helper (not a library invariant, per ADR 0013): a grep-friendly text dump for logs, snapshot tests, and the report's legacy `<pre>` fallback |
| `AsciiRenderer` | Renderer adapter over an `AsciiRenderable` env |

**`styled` — colour-aware projection** (consumed by the live TUI and report tiers)

| Item | Description |
|---|---|
| `StyledFrame` / `StyledLine` / `StyledSpan` | A small, ratatui-shaped vocabulary for styled text output |
| `SpanStyle` | Foreground/background `Color` plus a `Modifier` set |
| `Color` | Semantic colour enum |
| `Modifier` | Bitset of `BOLD` / `REVERSED` / `UNDERLINED` / … hints |

**`palette` — semantic colours.** Project-wide named colour constants, each paired with a `*_MODIFIER` companion so meaning is never carried by hue alone (accessibility contract — see ADR/feedback on hue-redundant signalling).

**`payload` — per-family structured snapshots** (the report tier's rich view). Pure-data snapshot types plus an opt-in `*PayloadSource` trait per family; envs that don't implement one fall back to the ASCII payload.

| Family | Snapshot type | Opt-in source trait |
|---|---|---|
| Landscapes | `Landscape2DSnapshot` (`Point2`) | `Landscape2DPayloadSource` |
| Box2D | `Box2dSnapshot` (`RigidBody2D`, `BodyKind`) | `Box2dPayloadSource` |
| Locomotion | `Locomotion2DSnapshot` | `Locomotion2DPayloadSource` |
| Grid | `GridSnapshot` (`GridTile`, `GridDir`, `GridColor`, `GridDoorState`, `GridAgentMarker`) | `GridPayloadSource` |
| Tabular | `TabularSnapshot` (`TabularGrid`, `TabularCell`, `TabularMarker`, `TabularLayout`, `CardTable`) | `TabularPayloadSource` |
| Classic 2D | `Classic2DSnapshot` (`Classic2DBody`, `Classic2DRole`) | `Classic2DPayloadSource` |

## Design Principles

**Const generics for shapes.** Every trait is parameterised by a const `D` (or `SD`, `AD`) so that dimension mismatches fail at compile time, not at runtime.

**Separation of state and observation.** `State<R>` represents the true MDP state; `Observation<R>` is what the agent receives. This separation makes POMDPs first-class citizens rather than workarounds.

**Terminated vs. Truncated.** `EpisodeStatus` distinguishes a natural terminal state from a step-limit cutoff. Algorithms that bootstrap values (e.g., TD-learning) need this distinction to avoid incorrect target computation.

**Trait-based composition.** No inheritance. New action spaces, state types, and genome kinds are added by implementing the appropriate trait.

**Zero-cost abstractions.** `NullRenderer`, `Copy` reward/state types, and const-generic shapes incur no runtime cost.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
rlevo-core = { path = "../rlevo-core" }  # or version once published
```

### Minimal Environment

```rust
use rlevo_core::{
    base::{Action, Observation, Reward, State},
    environment::{ConstructableEnv, Environment, EnvironmentError, SnapshotBase},
    reward::ScalarReward,
};

struct MyEnv { /* ... */ }

impl Environment<1, 1, 1> for MyEnv {
    type StateType = MyState;
    type ObservationType = MyObservation;
    type ActionType = MyAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, MyObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        // reset internal state, return initial snapshot
        Ok(SnapshotBase::running(MyObservation::default(), ScalarReward::new(0.0)))
    }

    fn step(&mut self, action: MyAction) -> Result<Self::SnapshotType, EnvironmentError> {
        // apply action, compute reward
        Ok(SnapshotBase::terminated(MyObservation::default(), ScalarReward::new(1.0)))
    }
}

impl ConstructableEnv for MyEnv {
    fn new(render: bool) -> Self { MyEnv { /* ... */ } }
}
```

### Replay Buffer

The prioritized replay buffer lives in `rlevo-reinforcement-learning`, not `rlevo-core`
(per ADR 0003 — replay/experience/metrics moved to where they are
consumed):

```rust
use rlevo_reinforcement_learning::memory::PrioritizedExperienceReplayBuilder;

let mut buffer = PrioritizedExperienceReplayBuilder::default()
    .with_capacity(100_000)
    .with_alpha(0.6)
    .build();

// buffer.add(obs, action, reward, next_obs, is_done, priority);
// let batch = buffer.sample_batch::<2, 2, MyBackend>(32, &device)?;
```

## Academic References

The following papers directly informed the algorithms and concepts in this crate:

- Sutton, R. S., & Barto, A. G. (2018). **Reinforcement Learning: An Introduction** (2nd ed.). MIT Press. — foundational framework for the `Environment` / `Snapshot` / `Reward` trait design and the Terminated vs. Truncated episode status distinction.

- Puterman, M. L. (1994). **Markov Decision Processes: Discrete Stochastic Dynamic Programming**. Wiley [acm digital library](https://dl.acm.org/doi/10.5555/528623). — theoretical grounding for `MarkovState`, `BeliefState`, and `TransitionDynamics`.

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or [MIT License](../../LICENSE-MIT) at your option.
