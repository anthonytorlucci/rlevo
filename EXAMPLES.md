# Examples

This guide catalogs the runnable examples across the `burn-evorl` workspace. Examples are ordered as a learning path — start at the top if you are new to the library, and skim the later sections for algorithm-specific entry points.

Run any example with:

```bash
cargo run -p <crate> --example <name> --release
```

---

## 1. Core abstractions — `evorl-core`

These examples have no neural networks and no training loop. They exist to make the central traits (`State`, `Observation`, `Action`) concrete before you touch any algorithm.

### `grid_position`
A 2D grid with a discrete "move N/S/E/W" action. Demonstrates the minimum viable implementation of `State<D>`, `Observation<D>`, and `DiscreteAction`: how states encode validity, how observations expose agent-visible data, and how a discrete action round-trips between an enum and its integer index. **Start here** to understand the shape of an environment before the RL machinery shows up.

### `combat_action`
Extends `grid_position` with a compound action — four directions × three attack strengths — via `MultiDiscreteAction`. The payoff is seeing how a multi-dimensional action space decomposes into independent sub-dimensions and how hierarchical enum structures flatten to an index array. Pick this up once `grid_position` feels obvious.

### `continuous_state_with_constraints`
A 1000×1000 mm robot workspace with orientation in [-180°, 180°], built with a constraint-driven builder pattern. Covers validation, distance metrics for reward shaping, angle normalization, and trajectory checks. This is the most heavily commented of the core examples and is intentionally written as a design-pattern reference for physics and robotics domains.

---

## 2. Benchmarking harness — `evorl-benchmarks`

These examples show how to plug an agent — evolutionary or RL — into the `Suite` / `Evaluator` / `BenchmarkReport` pipeline. The algorithms are deliberately simple so the harness remains the focus.

### `tabular_bandit`
A sample-average ε-greedy agent on the 10-armed bandit, evaluated under a frozen policy. The minimum reproducible path through the benchmark harness; use it as the template when wiring a new agent into `BenchEnv` and `Evaluator::run_suite`. A DQN version is deliberately deferred until `TenArmedBandit` implements the neural-network-facing traits.

### `ga_rastrigin`
A hand-rolled genetic algorithm (tournament selection + Gaussian mutation) optimizing the Rastrigin landscape, wrapped as a `BenchEnv`. Shows that the same benchmark instrumentation serves evolutionary search and reinforcement learning — a single reporting surface regardless of optimizer family.

---

## 3. Evolutionary strategies — `evorl-evolution`

### `sphere_showcase`
Runs the full roster — GA (tournament + BLX), ES variants from (1+1) to (μ,λ), EP, and several DE flavors — on a 10D sphere for 500 generations and prints a convergence comparison. Treat it as a menu: useful for picking a strategy, less useful as a tutorial.

---

## 4. Deep RL algorithms — `evorl-rl`

Each example stands alone (config struct, model, agent, training loop). Environments are intentionally minimal — CartPole for discrete control, Pendulum for continuous — so the differences between examples isolate the *algorithm* rather than the problem.

### Value-based (CartPole)

#### `dqn_cart_pole`
Vanilla Deep Q-Network: a two-hidden-layer MLP, replay buffer, ε-decay, and Polyak target updates. The baseline entry point into DRL in this library — read this before any of the other RL examples.

#### `c51_cart_pole`
Categorical DQN: the scalar Q-value is replaced by a categorical distribution over 51 fixed atoms in [0, 500]. The output tensor becomes `[batch, actions, atoms]`. Demonstrates distributional RL with an explicit, hand-tuned support.

#### `qrdqn_cart_pole`
Quantile Regression DQN: distributional RL again, but the support is *learned* as quantile locations rather than specified up front. The practical payoff over C51 is no `v_min` / `v_max` tuning; compare the two side by side to see the tradeoff between a fixed grid and an implicit one.

### Policy gradient

#### `ppo_cart_pole`
PPO with a categorical policy head — your first on-policy actor-critic. Shows the clipped surrogate objective, GAE, and a separate value network. The transition point from value-based (DQN family) to policy-gradient methods.

#### `ppo_pendulum`
The same PPO algorithm with a tanh-Gaussian policy head on continuous actions. The diff against `ppo_cart_pole` is exactly the action-space change: reparameterized sampling, state-conditional log-std, and log-probability computation. Read both to internalize that the discrete/continuous split lives in the policy head, not the algorithm.

#### `ppg_cart_pole`
Phasic Policy Gradient: separates policy and value updates into phases and adds an auxiliary value objective. Pedagogically, its job is to prove that `RolloutBuffer`, `PpoValue`, and the on-policy loss abstractions generalize beyond PPO. Note: wall-clock parity with PPO on CartPole is expected — PPG's speedups live in CNN + vectorized settings (e.g. Procgen), not in this toy.

### Deterministic & maximum-entropy continuous control (Pendulum)

These three share a CLI and environment so you can run them back-to-back and compare.

#### `ddpg_pendulum`
Deep Deterministic Policy Gradient: deterministic actor (tanh-scaled to [-2, 2]), separate critic `Q(s, a)`, replay, Polyak updates, action-space exploration noise. The first off-policy continuous-control recipe in the library.

#### `td3_pendulum`
Twin Delayed DDPG: two critics combined with `min(Q1, Q2)` for pessimism, delayed actor updates, and target-policy smoothing. Same plumbing as DDPG — read the diff to see exactly how each mitigation targets Q-value overestimation.

#### `sac_pendulum`
Soft Actor-Critic: stochastic squashed-Gaussian policy with per-state log-σ, twin critics, and entropy regularization with automatic temperature tuning. Completes the trio by swapping determinism plus external noise for learned action variance plus an explicit entropy objective.
