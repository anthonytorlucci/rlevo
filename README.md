# burn-evorl

> [!CAUTION]
> **Alpha Stage**: This project is in early development. Core abstractions and the environment suite are stable; RL algorithm implementations are still in progress.

**Survival of the fittest, implemented in Rust.**

Gradient descent is powerful, but it is a local optimizer. If an agent finds a mediocre solution that is "good enough," it often gets trapped in a local optimum — a mathematical rut that no amount of hyperparameter tuning can escape.

`burn-evorl` takes a different path. Built on [Burn](https://burn.dev/), this library implements **Evolutionary Reinforcement Learning (ERL)**: a population-based approach that uses crossover, mutation, and natural selection to optimize neural networks across complex, non-convex search spaces.

## Why ERL?

| Feature | Standard RL (Gradient-Based) | Evolutionary RL (ERL) |
| :--- | :--- | :--- |
| **Optimization** | Gradient descent | Black-box / genetic operators |
| **Agent focus** | Individual policy refinement | Population-wide evolution |
| **Learning signal** | Step-level rewards (TD-learning) | Episodic fitness (total reward) |
| **Search space** | Susceptible to local optima | Robust to noise & non-convexity |
| **Scaling** | Complex distributed synchronization | Embarrassingly parallel |
| **Sample efficiency** | High | Low (offset by parallelism) |

Because evaluating individuals is independent, ERL maps naturally onto Rust's fearless concurrency and Burn's backend-agnostic tensor operations — turning the sample-efficiency trade-off into a raw-throughput advantage.

## Workspace Crates

### `evorl-core`
Foundational trait definitions used across the entire workspace. Defines the core abstractions: `State`, `Observation`, `Action`, `Environment`, `Snapshot`, `ReplayBuffer`, and reward types. Everything else is built on top of these.

### `evorl-envs`
A growing suite of benchmark environments implementing the `Environment` trait.

**Classic Control**
- `CartPole` — balance a pole on a moving cart
- `MountainCar` / `MountainCarContinuous` — escape a valley with sparse rewards
- `Pendulum` — swing-up and stabilization
- `Acrobot` — underactuated double pendulum
- `TenArmedBandit` — multi-armed bandit testbed

**Box2D Physics**
- `BipedalWalker` — bipedal locomotion over varied terrain
- `LunarLander` / `LunarLanderContinuous` — fuel-efficient touchdown
- `CarRacing` — top-down racing with visual observations

**MuJoCo-style Locomotion**
- `InvertedPendulum` / `InvertedDoublePendulum` — balance tasks
- `Reacher` — goal-reaching with a two-link arm
- `Swimmer` — fluid locomotion with drag dynamics

**Grid Worlds**
- Configurable grid environments with optional memory, keyed doors, and partial observability

### `evorl-evolution`
The evolutionary engine. Implements tensor-native genetic algorithms and evolutionary strategies, with custom CubeCL kernels on hot paths and full swarm intelligence coverage.

**Classical Algorithms**
- Genetic Algorithm (GA) with crossover and mutation operators
- Evolution Strategies (ES), Evolutionary Programming (EP)
- Differential Evolution (DE), Cartesian Genetic Programming (CGP)

**Swarm Intelligence**
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)
- Firefly, Cuckoo Search, Bat Algorithm
- Grey Wolf Optimizer (GWO), Artificial Bee Colony (ABC)
- Whale Optimization Algorithm (WOA), Salp Swarm

### `evorl-rl`
Standard deep RL algorithm implementations. Currently in active development.

- **DQN** — core architecture in progress (config, model, agent scaffolded)
- **PPO** — planned
- **SAC** — planned

### `evorl-hybrid`
Combines gradient-based RL with evolutionary optimization. The integration layer connecting `evorl-rl` and `evorl-evolution` for hybrid training strategies.

### `evorl-benchmarks`
A reproducible benchmarking harness with deterministic seeding, checkpoint management, and JSON / TUI reporting. Used to evaluate and compare algorithms across environments.

### `evorl-utils`
Shared math utilities used across the workspace.

## Development Status

| Crate | Status |
| :--- | :--- |
| `evorl-core` | Stable — trait API largely settled |
| `evorl-envs` | Active — 13+ environments implemented |
| `evorl-evolution` | Active — full classical EA + swarm suite |
| `evorl-benchmarks` | Active — evaluation harness working |
| `evorl-rl` | In progress — DQN scaffolded, PPO/SAC pending |
| `evorl-hybrid` | Early — integration layer in design |
| `evorl-utils` | Minimal — grows with need |

## Quick Start

```bash
# Build the workspace
cargo build

# Run tests
cargo test

# Run a specific environment example
cargo run -p evorl-core --example grid_position

# Generate documentation
cargo doc --workspace --no-deps --open
```

## Dependencies

- **[Burn](https://burn.dev/) 0.19** — backend-agnostic tensor operations with `wgpu`, `ndarray`, training loop, and TUI metrics
- **rand 0.9** — randomness with deterministic seeding via `splitmix64`
- **serde 1.0** — serialization for checkpoints and configs
- **tracing 0.1** — structured logging
- **rapier2d / rapier3d** — physics simulation with enhanced determinism
- **criterion** — benchmarking

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
