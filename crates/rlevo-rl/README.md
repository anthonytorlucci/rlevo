# rlevo-rl

Deep reinforcement learning algorithms for the `rlevo` workspace.

This crate ports the [CleanRL](https://github.com/vwxyzjn/cleanrl) algorithm set into Rust using the [Burn](https://burn.dev) deep learning framework. The design follows CleanRL's *single-file-per-algorithm* pedagogical model: each algorithm lives under `src/algorithms/<algo>/` and its training loop is a linear, readable `train.rs` function.

All algorithms operate over the `rlevo-core` environment and agent abstractions. Neural-network operations go through Burn's `AutodiffBackend` so the same code runs on `ndarray` (CPU, bit-deterministic) and `wgpu` (GPU, within `1e-5` relative tolerance).

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

Bellemare, Dabney & Munos's distributional extension of DQN. Instead of predicting a scalar Q-value, the network outputs a probability distribution over a fixed set of N atoms spanning `[v_min, v_max]`. Bootstrap targets are computed by projecting the Bellman-shifted distribution back onto the support (Cramer projection); the loss is categorical cross-entropy.

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

### Proximal Policy Optimization (PPO)

**Module:** `algorithms::ppo`

Schulman et al.'s on-policy policy-gradient algorithm with a clipped surrogate objective. Supports both discrete and continuous action spaces through two built-in policy heads: `CategoricalPolicyHead` (softmax over logits) and `TanhGaussianPolicyHead` (state-independent `log_std` with `scale · tanh(z)` squashing). The rollout buffer computes GAE advantages; the update step runs `update_epochs` passes over `num_minibatches`-sized minibatches with an optional early-stop on `approx_kl`. Implementation details follow Huang et al. 2022.

Key components:

| File | Purpose |
|------|---------|
| `ppo_config.rs` | `PpoTrainingConfig` + `PpoTrainingConfigBuilder` + `annealed_learning_rate` |
| `ppo_policy.rs` | `PpoPolicy` trait with associated `ActionTensor` type |
| `ppo_value.rs` | `PpoValue` trait |
| `policies/categorical.rs` | `CategoricalPolicyHead` (discrete) |
| `policies/gaussian.rs` | `TanhGaussianPolicyHead` (continuous) |
| `rollout.rs` | `RolloutBuffer` + free `compute_gae` |
| `losses.rs` | `clipped_surrogate`, `clipped_value_loss`, `approx_kl`, `clip_fraction` |
| `ppo_agent.rs` | `PpoAgent` — `act`, `record_step`, `finalize_rollout`, `update` |
| `train.rs` | `train_discrete` / `train_continuous` entry points |

Default hyperparameters follow CleanRL's `ppo.py`:

| Hyperparameter | Default | Source |
|----------------|---------|--------|
| `num_steps` (rollout horizon) | 128 | CleanRL |
| `num_minibatches` | 4 | CleanRL |
| `update_epochs` | 4 | CleanRL |
| `learning_rate` | 2.5e-4 | CleanRL |
| `anneal_lr` | true | Huang et al. #4 |
| `gamma` (γ) | 0.99 | CleanRL |
| `gae_lambda` (λ) | 0.95 | Schulman et al. (2015) |
| `clip_coef` (ε) | 0.2 | Schulman et al. (2017) |
| `clip_value_loss` | true | Huang et al. #8 |
| `entropy_coef` | 0.01 | CleanRL |
| `value_coef` | 0.5 | CleanRL |
| `max_grad_norm` | 0.5 | Huang et al. #10 |
| Adam `epsilon` | 1e-5 | Huang et al. #3 |

`num_envs` is fixed at `1` in v0.1.0; vectorised rollout is deferred.

**References**

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms* — https://arxiv.org/abs/1707.06707
- Huang et al. (2022), *The 37 Implementation Details of PPO* — https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- CleanRL PPO implementation — https://docs.cleanrl.dev/rl-algorithms/ppo/

---

### Phasic Policy Gradient (PPG)

**Module:** `algorithms::ppg`

Cobbe et al.'s on-policy algorithm that interleaves a standard PPO policy phase with a periodic auxiliary phase that retrains the value function plus an auxiliary value head on the policy network, distilling the pre-aux-phase policy via a KL penalty. v0.1.0 ships a discrete-only `PpgCategoricalPolicyHead`; continuous support is deferred. CartPole parity with PPO is the v1 target — Procgen-scale gains require vectorised envs and CNN encoders.

Key components:

| File | Purpose |
|------|---------|
| `ppg_config.rs` | `PpgConfig` + `PpgConfigBuilder` (wraps `PpoTrainingConfig`) |
| `ppg_policy.rs` | `PpgPolicy` trait with auxiliary-value head accessor |
| `policies/categorical.rs` | `PpgCategoricalPolicyHead` (discrete-only in v0.1.0) |
| `aux_buffer.rs` | `AuxBuffer` accumulating rollouts between auxiliary phases |
| `losses.rs` | Auxiliary-phase losses (value, behavioural cloning / KL) |
| `ppg_agent.rs` | `PpgAgent` — policy-phase + auxiliary-phase updates |
| `train.rs` | End-to-end training loop |

The PPG-specific hyperparameters layer on top of PPO's defaults:

| Hyperparameter | Default | Source |
|----------------|---------|--------|
| `n_iteration` (policy phases per aux phase) | 32 | CleanRL |
| `e_aux` (aux epochs) | 6 | CleanRL |
| `beta_clone` (KL distillation coefficient) | 1.0 | CleanRL |
| `aux_batch_size` | 256 | CleanRL |

**References**

- Cobbe et al. (2020), *Phasic Policy Gradient* — https://arxiv.org/abs/2009.04416
- CleanRL PPG implementation — https://docs.cleanrl.dev/rl-algorithms/ppg/

---

### Deep Deterministic Policy Gradient (DDPG)

**Module:** `algorithms::ddpg`

Lillicrap et al.'s off-policy actor-critic for continuous action spaces. Pairs a deterministic actor with a Q-critic, each with a Polyak-averaged target copy. Explores via Gaussian noise on the actor output and learns off a uniform replay buffer. CleanRL's `ddpg_continuous_action.py` is the reference implementation.

Key components:

| File | Purpose |
|------|---------|
| `ddpg_config.rs` | `DdpgTrainingConfig` + `DdpgTrainingConfigBuilder` |
| `ddpg_model.rs` | `DeterministicPolicy` actor trait + `ContinuousQ` critic trait |
| `exploration.rs` | `GaussianNoise` (shared with TD3) |
| `ddpg_agent.rs` | `DdpgAgent` — act / remember / learn |
| `train.rs` | End-to-end collect-learn loop with warm-up |

Default hyperparameters follow CleanRL:

| Hyperparameter | Default | Source |
|----------------|---------|--------|
| `buffer_capacity` | 1 000 000 | CleanRL |
| `batch_size` | 256 | CleanRL |
| `learning_starts` | 25 000 | CleanRL |
| `actor_lr` | 3e-4 | CleanRL |
| `critic_lr` | 3e-4 | CleanRL |
| `gamma` (γ) | 0.99 | CleanRL |
| `tau` (Polyak) | 0.005 | CleanRL |
| `exploration_noise` (σ) | 0.1 | CleanRL |
| `policy_frequency` | 2 | CleanRL |

**References**

- Lillicrap et al. (2015), *Continuous Control with Deep Reinforcement Learning* — https://arxiv.org/abs/1509.02971
- CleanRL DDPG implementation — https://docs.cleanrl.dev/rl-algorithms/ddpg/

---

### Twin Delayed DDPG (TD3)

**Module:** `algorithms::td3`

Fujimoto et al.'s three-delta refinement of DDPG that addresses deterministic-policy Q-overestimation: a `min`-of-twin-critics bootstrap target, Gaussian target-policy smoothing, and delayed actor + Polyak updates every `policy_frequency`-th critic step. Reuses DDPG's `GaussianNoise`, `DeterministicPolicy`, and `ContinuousQ` trait surfaces unchanged. CleanRL's `td3_continuous_action.py` is the reference.

Key components:

| File | Purpose |
|------|---------|
| `td3_config.rs` | `Td3TrainingConfig` + `Td3TrainingConfigBuilder` |
| `td3_model.rs` | Re-exports / aliases atop `ddpg::ddpg_model` for twin-critic setups |
| `target_smoothing.rs` | `smooth_target_actions` — clipped Gaussian noise on target actor |
| `td3_agent.rs` | `Td3Agent` — twin critics + delayed actor update |
| `train.rs` | End-to-end collect-learn loop |

TD3 inherits DDPG's defaults and adds target-smoothing parameters:

| Hyperparameter | Default | Source |
|----------------|---------|--------|
| `policy_noise` (target σ) | 0.2 | Fujimoto et al. (2018) |
| `noise_clip` (target σ clip) | 0.5 | Fujimoto et al. (2018) |
| `policy_frequency` (delayed actor) | 2 | Fujimoto et al. (2018) |

**References**

- Fujimoto, van Hoof, Meger (2018), *Addressing Function Approximation Error in Actor-Critic Methods* — https://arxiv.org/abs/1802.09477
- CleanRL TD3 implementation — https://docs.cleanrl.dev/rl-algorithms/td3/

---

### Soft Actor-Critic (SAC)

**Module:** `algorithms::sac`

Haarnoja et al.'s off-policy max-entropy algorithm for continuous action spaces. Pairs a squashed-Gaussian stochastic actor with two critics (each with a Polyak-averaged target), a scalar learnable temperature α, and a uniform replay buffer. The Bellman target includes the entropy term `−α·log π(a'|s')`; the actor is trained via reparameterization; α is auto-tuned toward the heuristic target entropy `-|A|` by default. CleanRL's `sac_continuous_action.py` is the reference.

Key components:

| File | Purpose |
|------|---------|
| `sac_config.rs` | `SacTrainingConfig` + `SacTrainingConfigBuilder` |
| `sac_model.rs` | `ContinuousQ` critic trait (SAC uses twin critics) |
| `sac_policy.rs` | `SquashedGaussianPolicyHead` stochastic actor |
| `sac_alpha.rs` | `LogAlpha` — learnable log-temperature with auto-tuning toward `target_entropy` |
| `sac_agent.rs` | `SacAgent` — twin-critic updates + entropy-regularised actor loss |
| `train.rs` | End-to-end collect-learn loop |

Default hyperparameters follow CleanRL:

| Hyperparameter | Default | Source |
|----------------|---------|--------|
| `buffer_capacity` | 1 000 000 | CleanRL |
| `batch_size` | 256 | CleanRL |
| `learning_starts` | 5 000 | CleanRL |
| `actor_lr` | 3e-4 | CleanRL |
| `critic_lr` | 1e-3 | CleanRL |
| `alpha_lr` | 1e-3 | CleanRL |
| `gamma` (γ) | 0.99 | CleanRL |
| `tau` (Polyak) | 0.005 | CleanRL |
| `autotune` (α auto-tune) | true | Haarnoja et al. (2018b) |
| `initial_alpha` | 1.0 | CleanRL |
| `target_entropy` | `-|A|` heuristic | Haarnoja et al. (2018b) |
| `log_std_min` | -5.0 | CleanRL |
| `log_std_max` | 2.0 | CleanRL |
| `policy_frequency` | 2 | CleanRL |
| `target_update_frequency` | 1 | CleanRL |

**References**

- Haarnoja et al. (2018a), *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor* — https://arxiv.org/abs/1801.01290
- Haarnoja et al. (2018b), *Soft Actor-Critic Algorithms and Applications* — https://arxiv.org/abs/1812.05905
- CleanRL SAC implementation — https://docs.cleanrl.dev/rl-algorithms/sac/

---

## Planned / Deferred Algorithms

The following algorithms are deferred stubs — scope is parked until prerequisites land.

| Algorithm | Family | Action space | Status |
|-----------|--------|-------------|--------|
| Rainbow | Distributional value | Discrete | Deferred stub |
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
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms* — https://arxiv.org/abs/1707.06707
- Huang et al. (2022), *The 37 Implementation Details of Proximal Policy Optimization* — https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Cobbe et al. (2020), *Phasic Policy Gradient* — https://arxiv.org/abs/2009.04416
- Lillicrap et al. (2015), *Continuous Control with Deep Reinforcement Learning* — https://arxiv.org/abs/1509.02971
- Fujimoto, van Hoof & Meger (2018), *Addressing Function Approximation Error in Actor-Critic Methods* — https://arxiv.org/abs/1802.09477
- Haarnoja et al. (2018a), *Soft Actor-Critic* — https://arxiv.org/abs/1801.01290
- Haarnoja et al. (2018b), *Soft Actor-Critic Algorithms and Applications* — https://arxiv.org/abs/1812.05905

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
