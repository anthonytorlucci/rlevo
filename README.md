# rlevo

**Survival of the fittest, implemented in Rust.**

Gradient descent is powerful, but it is a local optimizer. If an agent finds a mediocre solution that is "good enough," it often gets trapped in a local optimum — a mathematical rut that no amount of hyperparameter tuning can escape.

`rlevo` takes a different path. Built on [Burn](https://burn.dev/), this library implements **Deep Reinforcement Learning with Evolutionary Optimization**: a population-based approach that uses crossover, mutation, and natural selection to optimize neural networks across complex, non-convex search spaces.

## Why Evolutionary Optimization with Deep Reinforcement Learning?

| Feature | Standard RL (Gradient-Based) | Evolutionary RL (ERL) |
| :--- | :--- | :--- |
| **Optimization** | Gradient descent | Black-box / genetic operators |
| **Agent focus** | Individual policy refinement | Population-wide evolution |
| **Learning signal** | Step-level rewards (TD-learning) | Episodic fitness (total reward) |
| **Search space** | Susceptible to local optima | Robust to noise & non-convexity |
| **Scaling** | Complex distributed synchronization | Embarrassingly parallel |
| **Sample efficiency** | High | Low (offset by parallelism) |

Because evaluating individuals is independent, ERL maps naturally onto Rust's fearless concurrency and Burn's backend-agnostic tensor operations — turning the sample-efficiency trade-off into a raw-throughput advantage.

## Why `rlevo`?

Most ERL implementations are Python research prototypes built around flat vector observations and fixed-dimension action spaces. `rlevo` is designed differently from the ground up:

**Const-generic dimensional safety.** `State<D>`, `Observation<D>`, and `Action<AD>` carry their dimensionality as const generic parameters. Dimension mismatches are compile-time errors, not runtime panics — a guarantee no existing Rust RL crate provides.

**Unified evolutionary and gradient-based RL.** Evolutionary and gradient-based agents share the same core trait abstractions, so they run against identical environments and compose naturally in a single training loop.

**Backend-agnostic tensors via Burn.** Neural network weights, population tensors, and replay buffers are all Burn tensors. Hardware backends (CPU, WGPU, CUDA) swap without touching algorithm code.

## What's Included

### Environments

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

### Deep RL Algorithms

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

### Evolutionary & Swarm Algorithms

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

### Hybrid RL + Evolution

Hybrid training strategies that combine gradient-based RL with evolutionary search are in active design. See the roadmap for details.

## Quick Start

```toml
[dependencies]
rlevo = "0.1"
```

```rust
use rlevo::environments::classic::CartPole;
use rlevo::core::{Environment, EpisodeStatus};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut env = CartPole::new(false);
    let snapshot = env.reset()?;
    println!("Initial observation: {:?}", snapshot.observation());

    loop {
        // Replace with your policy — here we pick action 0 unconditionally
        let action = env.sample_action();
        let snapshot = env.step(action)?;

        if matches!(snapshot.status(), EpisodeStatus::Terminated | EpisodeStatus::Truncated) {
            break;
        }
    }
    Ok(())
}
```

```bash
# Build the workspace
cargo build

# Run tests
cargo test

# Generate documentation
cargo doc --workspace --no-deps --open
```

## Development Status

`rlevo` is **alpha software**. The core trait API is largely settled; algorithm implementations and environments are under active development. Breaking changes may occur before 1.0.

| Area | Status |
| :--- | :--- |
| Core trait API | Stable |
| Environments (13+) | Active |
| Deep RL algorithms (8) | Active |
| Evolutionary & swarm algorithms | Active |
| Benchmarking harness | Active |
| Hybrid RL + evolution | Early design |

## Dependencies

- **[Burn](https://burn.dev/) 0.19** — backend-agnostic tensor operations with `wgpu`, `ndarray`, training loop, and TUI metrics
- **rand 0.9** — randomness with deterministic seeding via `splitmix64`
- **serde 1.0** — serialization for checkpoints and configs
- **tracing 0.1** — structured logging
- **rapier2d / rapier3d** — physics simulation with enhanced determinism
- **criterion** — benchmarking

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, scope, and how to open a PR.

## Development

This crate was developed with the assistance of AI coding tools (Claude by Anthropic).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
