# evorl-rl

Deep reinforcement learning algorithms for the `burn-evorl` workspace.

This crate ports the [CleanRL](https://github.com/vwxyzjn/cleanrl) algorithm set into Rust using the [Burn](https://burn.dev) deep learning framework. The design follows CleanRL's *single-file-per-algorithm* pedagogical model: each algorithm lives under `src/algorithms/<algo>/` and its training loop is a linear, readable `train.rs` function.

All algorithms operate over the `evorl-core` environment and agent abstractions. Neural-network operations go through Burn's `AutodiffBackend` so the same code runs on `ndarray` (CPU, bit-deterministic) and `wgpu` (GPU, within `1e-5` relative tolerance).

---

## Implemented Algorithms

### Deep Q-Network (DQN)

**Module:** `algorithms::dqn`

Mnih et al.'s seminal value-based algorithm. A policy Q-network maps observations to per-action Q-values; actions are selected by ε-greedy exploration; targets are bootstrapped from a periodically-synced frozen copy (target network). Supports both vanilla DQN (max-Q targets) and Double-DQN (action selected by online network, evaluated by target).

Key components:

| File | Purpose |
|------|---------|
| `dqn_config.rs` | `DqnTrainingConfig` + `DqnTrainingConfigBuilder` |
| `dqn_model.rs` | `DqnModel` Burn `Module` trait |
| `dqn_agent.rs` | `DqnAgent` — act / remember / learn |
| `exploration.rs` | `EpsilonGreedy` schedule (shared with C51, QR-DQN) |
| `train.rs` | End-to-end training loop |

Default hyperparameters match the Nature DQN paper:

| Hyperparameter | Default | Source |
|----------------|---------|--------|
| `batch_size` | 32 | Nature DQN |
| `gamma` (γ) | 0.99 | Nature DQN |
| `tau` (soft-update) | 0.005 | CleanRL |
| `learning_rate` | 0.001 | CleanRL |
| `epsilon_start` | 1.0 | Nature DQN |
| `epsilon_end` | 0.01 | Nature DQN |
| `replay_buffer_capacity` | 10 000 | CleanRL |
| `learning_starts` | 1 000 | CleanRL |
| `train_frequency` | 4 | Nature DQN |

**References**

- Mnih et al. (2015), *Human-level control through deep reinforcement learning*, Nature — https://www.nature.com/articles/nature14236
- van Hasselt, Guez & Silver (2016), *Deep Reinforcement Learning with Double Q-learning*, AAAI — https://arxiv.org/abs/1509.06461
- CleanRL DQN implementation — https://docs.cleanrl.dev/rl-algorithms/dqn/

---

### Categorical DQN (C51)

**Module:** `algorithms::c51`

Bellemare, Dabney & Munos's distributional extension of DQN. Instead of predicting a scalar Q-value, the network outputs a probability distribution over a fixed set of N atoms spanning `[v_min, v_max]`. Bootstrap targets are computed by projecting the Bellman-shifted distribution back onto the support (Cătălina projection); the loss is categorical cross-entropy.

Key components:

| File | Purpose |
|------|---------|
| `c51_config.rs` | `C51TrainingConfig` + `C51TrainingConfigBuilder` |
| `c51_model.rs` | `C51Model` Burn `Module` trait |
| `c51_agent.rs` | `C51Agent` — act / remember / learn |
| `projection.rs` | `project_distribution` — Bellman projection onto atom support |
| `loss.rs` | `categorical_cross_entropy` loss |
| `train.rs` | End-to-end training loop |

Distributional-specific hyperparameters:

| Hyperparameter | Default | Source |
|----------------|---------|--------|
| `num_atoms` | 51 | Bellemare et al. (2017) — gives the algorithm its name |
| `v_min` | −10.0 | Bellemare et al. (2017) |
| `v_max` | 10.0 | Bellemare et al. (2017) |

All standard DQN hyperparameters are inherited unchanged.

**References**

- Bellemare, Dabney & Munos (2017), *A Distributional Perspective on Reinforcement Learning*, ICML — https://arxiv.org/abs/1707.06887
- CleanRL C51 implementation — https://docs.cleanrl.dev/rl-algorithms/c51/

---

### Quantile Regression DQN (QR-DQN)

**Module:** `algorithms::qrdqn`

Dabney et al.'s distributional algorithm that replaces C51's fixed atom support with implicit quantile targets. The network outputs N quantile values `θ_i(s, a)` for midpoints `τ_i = (i + 0.5) / N`. No `[v_min, v_max]` range is required — the distribution is unconstrained. The loss is the quantile Huber loss (asymmetric Huber weighted by `|τ_i − 𝟙[u < 0]|`).

Key components:

| File | Purpose |
|------|---------|
| `qrdqn_config.rs` | `QrDqnTrainingConfig` + `QrDqnTrainingConfigBuilder` |
| `qrdqn_model.rs` | `QrDqnModel` Burn `Module` trait |
| `qrdqn_agent.rs` | `QrDqnAgent` — act / remember / learn |
| `quantile_loss.rs` | `quantile_huber_loss` |
| `train.rs` | End-to-end training loop |

Distributional-specific hyperparameters:

| Hyperparameter | Default | Source |
|----------------|---------|--------|
| `num_quantiles` | 200 | Dabney et al. (2018) |
| `kappa` (Huber threshold) | 1.0 | Dabney et al. (2018) |

**References**

- Dabney et al. (2018), *Distributional Reinforcement Learning with Quantile Regression*, AAAI — https://arxiv.org/abs/1710.10044
- CleanRL QR-DQN implementation — https://docs.cleanrl.dev/rl-algorithms/c51/#qr-dqnpy

---

## Planned Algorithms

The following algorithms are planned for future releases:

| Algorithm | Family | Action space | Status |
|-----------|--------|-------------|--------|
| Rainbow | Distributional value | Discrete | Deferred stub |
| PPO | Policy gradient | Both | Planned |
| PPG | Policy gradient | Discrete | Planned |
| DDPG | Actor-critic | Continuous | Planned |
| TD3 | Actor-critic | Continuous | Planned |
| SAC | Max-entropy actor-critic | Both | Planned |
| RND | Exploration | Discrete | Deferred stub |
| QDagger | Distillation | Discrete | Deferred stub |

---

## Shared Infrastructure

### Exploration (`algorithms::dqn::exploration`)

`EpsilonGreedy` is shared across all discrete value-based algorithms (DQN, C51, QR-DQN). The schedule decays ε multiplicatively from `epsilon_start` to `epsilon_end` at rate `epsilon_decay`.

### Utilities (`utils`)

`compute_target_q_values` — Bellman target computation shared by DQN variants.

---

## Configuration Pattern

Every algorithm follows the same config pattern:

```rust
// Default config (CleanRL reference values)
let cfg = DqnTrainingConfig::default();

// Builder for custom overrides
let cfg = DqnTrainingConfigBuilder::new()
    .learning_rate(0.0005)
    .batch_size(64)
    .double_q(true)
    .build();
```

All config structs are `Clone + Debug`. Serde support is planned.

---

## Reproducibility

- Every stochastic entry point (`act`, `learn`, `sample`) takes an explicit `rng` — no thread-locals or module-owned RNG.
- `ndarray` backend is bit-deterministic; `wgpu` is within `1e-5` relative tolerance.
- Unit tests that assert exact equality run on `ndarray`.

---

## References

- CleanRL — https://github.com/vwxyzjn/cleanrl
- CleanRL algorithm overview — https://docs.cleanrl.dev/rl-algorithms/overview/
- Burn framework — https://burn.dev
- Mnih et al. (2015), *Human-level control through deep reinforcement learning* — https://www.nature.com/articles/nature14236
- van Hasselt, Guez & Silver (2016), *Deep Reinforcement Learning with Double Q-learning* — https://arxiv.org/abs/1509.06461
- Bellemare, Dabney & Munos (2017), *A Distributional Perspective on Reinforcement Learning* — https://arxiv.org/abs/1707.06887
- Dabney et al. (2018), *Distributional Reinforcement Learning with Quantile Regression* — https://arxiv.org/abs/1710.10044
- Huang et al. (2022), *The 37 Implementation Details of Proximal Policy Optimization* — https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
