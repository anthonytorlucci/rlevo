# evorl-envs

Standard benchmark environments for the `burn-evorl` workspace.

This crate provides a collection of RL environments ranging from classic tabular problems through continuous-control physics simulations. All environments implement the `evorl-core` `Environment` trait — a common `reset` / `step` interface that makes them drop-in compatible with every algorithm in `evorl-rl`.

---

## Environments

### Classic Control

Ports of the canonical Gymnasium control tasks implemented in pure Rust.

| Environment | Module | Observation | Action | Notes |
|---|---|---|---|---|
| `CartPole` | `classic::CartPole` | 4-D continuous | Discrete(2) | Physics match Gymnasium `CartPole-v1` exactly |
| `Acrobot` | `classic::Acrobot` | 6-D continuous | Discrete(3) | Sutton two-link dynamics |
| `MountainCar` | `classic::MountainCar` | 2-D continuous | Discrete(3) | Sparse reward; needs exploration |
| `MountainCarContinuous` | `classic::MountainCarContinuous` | 2-D continuous | Continuous(1) | Dense reward variant |
| `Pendulum` | `classic::Pendulum` | 3-D continuous | Continuous(1) | Underactuated swing-up |
| `TenArmedBandit` | `classic::TenArmedBandit` | — | Discrete(10) | ε-greedy / UCB / Thompson via features |

---

### Toy Text

Tabular MDPs for baseline algorithm validation. Every state is fully observable.

| Environment | Module | State space | Action |
|---|---|---|---|
| `FrozenLake` | `toy_text::FrozenLake` | 16 or 64 discrete | Discrete(4) |
| `CliffWalking` | `toy_text::CliffWalking` | 48 discrete | Discrete(4) |
| `Taxi` | `toy_text::Taxi` | 500 discrete | Discrete(6) |
| `Blackjack` | `toy_text::Blackjack` | 32×11×2 discrete | Discrete(2) |

---

### Gridworlds

Twelve partially observable grid environments inspired by [Farama Minigrid](https://minigrid.farama.org). All share a common egocentric 7×7×3 observation and a 7-action discrete space. Physics are implemented once in `grids::core` and reused across every variant.

| Environment | Grid size | Key mechanic |
|---|---|---|
| `EmptyEnv` | 6×6 | Reach the goal |
| `DoorKeyEnv` | 8×8 | Pick up key, unlock door, reach goal |
| `LavaGapEnv` | 7×7 | Navigate a gap in a lava wall |
| `FourRoomsEnv` | 19×19 | Long-horizon exploration |
| `UnlockEnv` | 6×6 | Single lock/key |
| `UnlockPickupEnv` | 7×7 | Unlock then pick up object |
| `MemoryEnv` | 5×… | Remember seen target |
| `MultiRoomEnv` | variable | Chain of rooms |
| `CrossingEnv` | 11×11 | Navigate obstacles |
| `DistShiftEnv` | 9×… | Adaption to distribution shift |
| `DynamicObstaclesEnv` | 6×6 | Moving obstacles |
| `GoToDoorEnv` | 5×5 | Reach specified door |

---

### Box2D-style Physics

Continuous-control environments powered by [Rapier2D](https://rapier.rs). Enabled by the `box2d` feature (default on).

| Environment | Module | Observation | Action |
|---|---|---|---|
| `BipedalWalker` | `box2d::BipedalWalker` | 24-D continuous | Continuous(4) |
| `LunarLander` | `box2d::LunarLander` | 8-D continuous | Discrete(4) or Continuous(2) |
| `CarRacing` | `box2d::CarRacing` | 96×96 pixel | Continuous(3) |

---

### Locomotion

MuJoCo-style locomotion environments in pure Rust via [Rapier3D](https://rapier.rs). Enabled by the `locomotion` feature (default on).

| Environment | Module | Notes |
|---|---|---|
| `InvertedPendulum` | `locomotion::InvertedPendulum` | 1-D balance |
| `InvertedDoublePendulum` | `locomotion::InvertedDoublePendulum` | Harder balance; sparser failure |
| `Reacher` | `locomotion::Reacher` | 2-DOF arm reaching |
| `Swimmer` | `locomotion::Swimmer` | 3-link swimmer |

---

### Games

| Environment | Module | Notes |
|---|---|---|
| `Chess` | `games::chess` | Full legal-move board; uses `chess` crate |
| `ConnectFour` | `games::connect_four` | Two-player; 7×6 board |

---

### Optimization Benchmarks

Continuous single-objective functions for evaluating evolutionary algorithms.

| Function | Module | Notes |
|---|---|---|
| Sphere | `benchmarks::sphere` | Convex, unimodal |
| Ackley | `benchmarks::ackley` | Multimodal; exponential traps |
| Rastrigin | `benchmarks::rastrigin` | Highly multimodal |

---

## Quick Start

```rust,no_run
use evorl_core::environment::{Environment, Snapshot};
use evorl_envs::classic::{CartPole, CartPoleConfig, CartPoleAction};
use evorl_envs::wrappers::TimeLimit;

let env = CartPole::with_config(CartPoleConfig::default());
let mut env = TimeLimit::new(env, 500);

let mut snap = env.reset().expect("reset");
while !snap.is_done() {
    snap = env.step(CartPoleAction::Right).expect("step");
}
```

Gridworld environments use the shared `GridAction` type:

```rust,no_run
use evorl_core::environment::{Environment, Snapshot};
use evorl_envs::grids::{EmptyEnv, EmptyConfig};
use evorl_envs::grids::core::GridAction;

let mut env = EmptyEnv::with_config(EmptyConfig::default(), false);
let mut snap = env.reset().expect("reset");
while !snap.is_done() {
    snap = env.step(GridAction::Forward).expect("step");
}
```

---

## Cargo Features

| Feature | Default | Description |
|---|---|---|
| `box2d` | yes | Box2D-style physics environments via `rapier2d` |
| `locomotion` | yes | Locomotion environments via `rapier3d` + `nalgebra` |
| `ucb` | no | Upper Confidence Bound bandit strategy |
| `thompson` | no | Thompson Sampling bandit strategy |
| `mujoco-ffi` | no | Reserved for future FFI-bound MuJoCo backend (compile-errors at v1) |

Disable physics environments to shrink compile time:

```toml
[dependencies]
evorl-envs = { path = "…", default-features = false }
```

---

## Running Examples

```bash
# Classic control
cargo run -p evorl-envs --example cartpole_random
cargo run -p evorl-envs --example cartpole_timelimit
cargo run -p evorl-envs --example pendulum_random
cargo run -p evorl-envs --example mountain_car_random
cargo run -p evorl-envs --example mountain_car_continuous_random
cargo run -p evorl-envs --example acrobot_random

# Gridworlds
cargo run -p evorl-envs --example grid_empty_random
cargo run -p evorl-envs --example grid_door_key_scripted
cargo run -p evorl-envs --example grid_memory_random

# Bandits
cargo run -p evorl-envs --example ten_armed_bandit_training

# Box2D (requires box2d feature, enabled by default)
cargo run -p evorl-envs --example bipedal_walker_random
cargo run -p evorl-envs --example lunar_lander_discrete_random
cargo run -p evorl-envs --example lunar_lander_continuous_random
cargo run -p evorl-envs --example car_racing_random

# Locomotion (requires locomotion feature, enabled by default)
cargo run -p evorl-envs --example reacher_random
cargo run -p evorl-envs --example inverted_double_pendulum_random
cargo run -p evorl-envs --example swimmer_random
```

See [`examples/README.md`](examples/README.md) for patterns, conventions, and how to write new examples.

---

## Design

### Configuration Builder

Every environment ships with a `XyzConfig` struct and a `XyzConfigBuilder` with a fluent API. Defaults match the reference Gymnasium implementation where one exists.

```rust,no_run
use evorl_envs::classic::{CartPole, CartPoleConfig};

let env = CartPole::with_config(
    CartPoleConfig {
        seed: 42,
        ..CartPoleConfig::default()
    }
);
```

### Reproducibility

`reset()` re-seeds the environment's internal RNG from `config.seed` so the same seed always produces the same episode trajectory. Pass different seeds across parallel workers to get independent rollouts.

### Wrappers

`wrappers::TimeLimit<E>` wraps any `Environment` and injects a truncation signal after a configurable number of steps. The underlying environment's `Terminated` status is preserved separately from the wrapper's `Truncated` — algorithms that distinguish the two (PPO, SAC) can act on both signals correctly.

### Episode Status

`Snapshot::status()` returns one of three variants:

- `Running` — episode continues
- `Terminated` — natural episode end (goal reached, fell over, game over)
- `Truncated` — wall-clock time limit exceeded (`TimeLimit` wrapper)

### Const Generics

State and observation dimensions are encoded as const generics (`State<D>`, `Observation<D>`). Mismatched dimensions produce a compile-time error rather than a runtime panic.

---

## Testing

```bash
# Unit tests (all environments)
cargo test -p evorl-envs

# Gridworld solvability integration tests
# (scripted optimal policies verify physics correctness)
cargo test -p evorl-envs --test grids_solvable
```

---

## References

- Brockman et al. (2016), *OpenAI Gym* — https://arxiv.org/abs/1606.01540
- Chevalier-Boisvert et al. (2023), *Minigrid & Miniworld* — https://arxiv.org/abs/2306.13831
- Barto, Sutton & Anderson (1983), *Neuronlike adaptive elements that can solve difficult learning control problems*, IEEE SMC — CartPole physics
- Rapier physics engine — https://rapier.rs
- Burn framework — https://burn.dev
