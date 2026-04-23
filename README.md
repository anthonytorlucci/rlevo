# burn-evorl

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

## Why burn-evorl?

Most ERL implementations are Python research prototypes built around flat vector observations and fixed-dimension action spaces. `burn-evorl` is designed differently from the ground up:

**Const-generic dimensional safety.** `State<D>`, `Observation<D>`, and `Action<AD>` carry their dimensionality as const generic parameters. Dimension mismatches are compile-time errors, not runtime panics — a guarantee no existing Rust RL crate provides.

**Unified evolutionary and gradient-based RL.** The `evorl-evolution`, `evorl-rl`, and `evorl-hybrid` crates share the same core trait abstractions, so evolutionary and gradient-based agents run against identical environments and compose naturally in a single training loop.

**Backend-agnostic tensors via Burn.** Neural network weights, population tensors, and replay buffers are all Burn tensors. Hardware backends (CPU, WGPU, CUDA) swap without touching algorithm code.

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
Standard deep RL algorithm implementations.

**Value-Based**
- **DQN** — Deep Q-Network with experience replay and target network
- **C51** — Categorical DQN (distributional RL over 51 atoms)
- **QR-DQN** — Quantile Regression DQN

**Policy Gradient**
- **PPO** — Proximal Policy Optimization with clipped surrogate objective (categorical and Gaussian policies)
- **PPG** — Phasic Policy Gradient with auxiliary phase and distillation

**Actor-Critic (Continuous Control)**
- **DDPG** — Deep Deterministic Policy Gradient with Ornstein-Uhlenbeck exploration
- **TD3** — Twin Delayed DDPG with target policy smoothing
- **SAC** — Soft Actor-Critic with automatic entropy tuning

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
| `evorl-rl` | Active — 8 algorithms implemented (DQN, C51, QR-DQN, PPO, PPG, DDPG, TD3, SAC) |
| `evorl-hybrid` | Early — integration layer in design |
| `evorl-utils` | Minimal — grows with need |

## Quick Start

```bash
# Build the workspace
cargo build

# Run tests
cargo test

# Run a specific environment example
cargo run -p evorl-core --example grid_agent

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

## Development

This crate was developed with the assistance of AI coding tools (Claude by Anthropic).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
