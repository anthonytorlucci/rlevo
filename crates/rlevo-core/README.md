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
| `State<D>` | Full MDP state; produces an `Observation<D>` via `observe()` |
| `Observation<D>` | What the agent perceives (may be a partial view of state) |
| `Action<D>` | Base action constraint (`is_valid()`) |
| `Reward` | Scalar feedback signal; must be `Clone + Add + Into<f32> + Debug` and provide `zero()` |
| `TensorConvertible<D, B>` | Bidirectional conversion between domain types and Burn tensors |
| `TransitionDynamics<SD, AD, S, A>` | Deterministic state-transition function |
| `UpdateFunction<Input, Output>` | Generic parameterized update (policy gradient step, etc.) |

`TensorConvertible` is the integration point with Burn: implement it on your state/action types to enable neural-network-based agents.

### `action` — Action Spaces

Three orthogonal action abstractions cover the standard taxonomy:

| Trait | Use Case |
|---|---|
| `DiscreteAction<D>` | Finite enumerable actions (e.g., 4-directional movement). Provides `from_index`, `to_index`, `enumerate`, `random`. |
| `MultiDiscreteAction<D>` | Multiple independent discrete dimensions (e.g., direction × attack). Combinatorial enumeration and weighted random sampling. |
| `ContinuousAction<D>` | Real-valued vectors with `clip`, `from_slice`, `as_slice`, `random` (uniform in \[−1, 1\]). |
| `BoundedAction<D>` | Extends `ContinuousAction<D>` with per-component `low()` / `high()` bounds. |

### `state` — Advanced State Abstractions

Extensions to `State<D>` for partial observability, recurrence, and hierarchy:

| Trait | Purpose |
|---|---|
| `MarkovState` | Asserts the Markov property holds for this state representation |
| `BeliefState<SD, AD, S, A>` | POMDP belief distribution over latent states, updated by action `A` |
| `HiddenState<D>` | RNN-style recurrent memory (`update`, `reset`) |
| `LatentState<D, AD>` | World-model compressed representation (`encode`, `predict`, `decode`) |
| `StateAggregation<SD, S>` | Maps concrete states to abstract representatives for hierarchical RL |

### `environment` — Environment Protocol

| Item | Description |
|---|---|
| `Environment<D, SD, AD>` | Core trait: `new(render)`, `reset()`, `step(action)` |
| `Snapshot<D>` | Per-step result: `observation()`, `reward()`, `status()` |
| `EpisodeStatus` | `Running \| Terminated \| Truncated` — distinguishes natural episode ends from step-limit truncations (important for value bootstrapping) |
| `SnapshotBase<D, O, R>` | Default `Snapshot` implementation with named constructors: `running()`, `terminated()`, `truncated()` |
| `SnapshotMetadata` | Builder for named reward components and 3-D positions (visualization / debugging) |

### `reward` — Concrete Reward Types

`ScalarReward(f32)` is a thin newtype implementing `Reward`. It covers the vast majority of environments. Custom reward types can implement `Reward` directly.

### `memory` — Experience Replay Buffers

| Item | Description |
|---|---|
| `PrioritizedExperienceReplay<D, AD, O, A, R>` | Off-policy replay buffer with priority-weighted sampling. Priorities are raised to power `alpha` before normalization. FIFO eviction at capacity. |
| `PrioritizedExperienceReplayBuilder` | Fluent builder: `.with_capacity(n).with_alpha(0.6).build()` |
| `TrainingBatch<BD, BAD, B>` | GPU-ready tensor bundle (`observations`, `actions`, `rewards`, `next_observations`, `dones`) consumed by learning algorithms |

Prioritized Experience Replay follows the algorithm described in:

> Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). **Prioritized Experience Replay**. *ICLR 2016*. [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)

### `experience` — Trajectory Storage

| Item | Description |
|---|---|
| `ExperienceTuple<D, AD, O, A, R>` | Single `(s, a, r, s′, done)` transition |
| `History<D, AD, O, A, R>` | Fixed-capacity FIFO buffer (VecDeque); indexable, iterable |
| `HistoryRepresentation` | Trait for summaries built from a `History` |
| `SufficientStatistic` | Extends `HistoryRepresentation + MarkovState`; checks whether a history summary is Markov-sufficient |

### `evolution` — Evolutionary Optimization

Minimal trait surface that `rlevo-evolution` builds on:

| Trait | Description |
|---|---|
| `Fitness` | Scalar fitness score: `worst()`, `is_finite()`, `as_f32()`. Blanket impls for `f32` and `f64`. Convention: **higher is better**. |
| `GenomeKind` | Zero-sized marker with `const DIM: usize` and associated `Element` type; lets evolutionary strategies select operators at compile time. |
| `MultiFitness` | Multi-objective fitness: `objectives() -> &[f32]`. Pareto/NSGA machinery lives in `rlevo-evolution`. |

### `metrics` — Performance Tracking

| Item | Description |
|---|---|
| `PerformanceRecord` | Per-episode outcome: `score() -> f32`, `duration() -> usize` |
| `AgentStats<T>` | Running counters (`total_episodes`, `total_steps`, `best_score`) with a configurable sliding-window average |

### `agent` — Agent Traits (reserved)

Placeholder module reserved for a future unified agent trait hierarchy
(`act` / `learn` / checkpoint). Empty in v0.1.0 — concrete algorithms currently
live in `rlevo-reinforcement-learning` and `rlevo-evolution` and will migrate behind these traits
once the API stabilizes.

### `render` — Rendering Abstractions

| Item | Description |
|---|---|
| `Renderer<E>` | Generic over environment and associated `Frame` type (`String`, `Vec<u8>`, `()`, …) |
| `NullRenderer` | Zero-cost no-op; compiles away entirely |

## Design Principles

**Const generics for shapes.** Every trait is parameterised by a const `D` (or `SD`, `AD`) so that dimension mismatches fail at compile time, not at runtime.

**Separation of state and observation.** `State<D>` represents the true MDP state; `Observation<D>` is what the agent receives. This separation makes POMDPs first-class citizens rather than workarounds.

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
    environment::{Environment, EnvironmentError, SnapshotBase},
    reward::ScalarReward,
};

struct MyEnv { /* ... */ }

impl Environment<1, 1, 1> for MyEnv {
    type StateType = MyState;
    type ObservationType = MyObservation;
    type ActionType = MyAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, MyObservation, ScalarReward>;

    fn new(render: bool) -> Self { MyEnv { /* ... */ } }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        // reset internal state, return initial snapshot
        Ok(SnapshotBase::running(MyObservation::default(), ScalarReward::new(0.0)))
    }

    fn step(&mut self, action: MyAction) -> Result<Self::SnapshotType, EnvironmentError> {
        // apply action, compute reward
        Ok(SnapshotBase::terminated(MyObservation::default(), ScalarReward::new(1.0)))
    }
}
```

### Replay Buffer

The prioritized replay buffer lives in `rlevo-rl`, not `rlevo-core`
(per ADR 0003 — replay/experience/metrics moved to where they are
consumed):

```rust
use rlevo_rl::memory::PrioritizedExperienceReplayBuilder;

let mut buffer = PrioritizedExperienceReplayBuilder::default()
    .with_capacity(100_000)
    .with_alpha(0.6)
    .build();

// buffer.add(obs, action, reward, next_obs, is_done, priority);
// let batch = buffer.sample_batch::<2, 2, MyBackend>(32, &device)?;
```

## Examples

```bash
# Egocentric grid agent: State, Observation, TensorConvertible,
# DiscreteAction, and MultiDiscreteAction (move + interact)
cargo run -p rlevo-core --example grid_agent

# Constrained continuous state (2D robot workspace with orientation bounds)
cargo run -p rlevo-core --example continuous_state_with_constraints
```

## Academic References

The following papers directly informed the algorithms and concepts in this crate:

- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). **Prioritized Experience Replay**. *International Conference on Learning Representations (ICLR)*. [arXiv:1511.05952](https://arxiv.org/abs/1511.05952) — basis for `PrioritizedExperienceReplay` and the priority-weighted sampling scheme.

- Sutton, R. S., & Barto, A. G. (2018). **Reinforcement Learning: An Introduction** (2nd ed.). MIT Press. — foundational framework for the `Environment` / `Snapshot` / `Reward` trait design, Bellman targets in `ExperienceTuple`, and the Terminated vs. Truncated episode status distinction.

- Puterman, M. L. (1994). **Markov Decision Processes: Discrete Stochastic Dynamic Programming**. Wiley [acm digital library](https://dl.acm.org/doi/10.5555/528623). — theoretical grounding for `MarkovState`, `BeliefState`, and `TransitionDynamics`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
