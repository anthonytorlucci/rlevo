# PPO (Proximal Policy Optimization)

On-policy policy-gradient algorithm with a clipped surrogate objective. Handles
both discrete and continuous action spaces through two built-in policy heads:

- `CategoricalPolicyHead` — discrete, softmax over logits.
- `TanhGaussianPolicyHead` — continuous, state-independent `log_std` with
  `scale · tanh(z)` squashing applied at the env boundary.

Reference: [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/);
implementation details follow
[Huang et al. 2022, *The 37 Implementation Details of PPO*](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

## Quick start

Discrete env (CartPole, Acrobot, ...):

```bash
cargo run -p rlevo-reinforcement-learning --release --example ppo_cart_pole -- \
  --seed 42 --total-timesteps 50000 --num-steps 128 --log-every 4096
```

Continuous env (Pendulum, MountainCarContinuous, ...):

```bash
cargo run -p rlevo-reinforcement-learning --release --example ppo_pendulum -- \
  --seed 42 --total-timesteps 100000 --num-steps 2048 --log-every 4096
```

## 37 details — coverage in v1

**Implemented** (roughly half of Huang et al.'s 37, covering the correctness-
critical subset):

| # | Detail | Where |
|---|---|---|
| 3 | Adam epsilon = 1e-5 | `ppo_config::PpoTrainingConfig::default` |
| 4 | Linear LR annealing | `ppo_config::annealed_learning_rate` |
| 5 | GAE advantages | `rollout::compute_gae` |
| 6 | Batch-level advantage normalization | `losses::normalize_advantages` |
| 7 | Clipped surrogate objective | `losses::clipped_surrogate` |
| 8 | Clipped value loss | `losses::clipped_value_loss` |
| 9 | Entropy bonus | `ppo_agent::PpoAgent::update` |
| 12 | Separate policy / value networks | `PpoAgent` fields |
| — | Early stop on target_kl | `PpoAgent::update` (threshold `1.5 · target_kl`) |
| — | Gaussian log-prob summed across action dims | `gaussian::log_prob_entropy` |
| — | Terminated / truncated stored distinctly | `rollout::RolloutBuffer` |

**Deferred** (documented gaps, not bugs):

- #1 Vectorised architecture — sequential `num_envs = 1` in v1; vec-envs are
  a follow-up.
- #10 Global gradient-norm clipping — **not implemented**.
  `PpoTrainingConfig::clip_grad` is `None` by default (no clipping at all), and
  when set, Burn's `GradientClippingConfig::Norm` rescales each parameter
  tensor against its *own* L2 norm. Detail #10 requires rescaling against the
  norm of the entire flattened parameter vector, which Burn's optimizer hook
  cannot express. Tracked in issue #328.
- #11 Vectorised reset handling — N/A while sequential.
- Running observation normalization / returns-based reward scaling — a
  follow-up. Pendulum reward targets note the expected gap.
- #14–37 Continuous- and Atari-specific details — covered as the relevant
  envs/wrappers land.
- Truncation bootstrap bias: when an episode ends via `Truncated` mid-rollout,
  the GAE currently zeros the bootstrap (matching CleanRL). Strict correctness
  would bootstrap from `V(s_continuation)`, which isn't available post-reset.
  See `rollout::compute_gae`'s doc-comment.

## Module layout

| File | Role |
|---|---|
| `ppo_config.rs` | `PpoTrainingConfig` + builder + `annealed_learning_rate`. |
| `ppo_policy.rs` | `PpoPolicy` trait with associated `ActionTensor` type. |
| `ppo_value.rs` | `PpoValue` trait. |
| `policies/categorical.rs` | `CategoricalPolicyHead` built-in. |
| `policies/gaussian.rs` | `TanhGaussianPolicyHead` built-in. |
| `rollout.rs` | `RolloutBuffer` + free `compute_gae`. |
| `losses.rs` | Stateless loss / diagnostic functions. |
| `ppo_agent.rs` | `PpoAgent` with `act`, `record_step`, `finalize_rollout`, `update`. |
| `train.rs` | End-to-end `train_discrete` / `train_continuous` entry points. |

The trait surface is deliberately scoped to this module rather than promoted
to a crate-level "stochastic policy" abstraction: each algorithm in
`rlevo-reinforcement-learning` currently owns its model trait (`DqnModel`, `C51Model`, ...).
Cross-algorithm refactoring waits until SAC/PPG drive a concrete second
consumer.
