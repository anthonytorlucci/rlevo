# Deep Q-Network (DQN)

`rlevo-reinforcement-learning`'s DQN implementation follows the Nature-style Q-learning algorithm
with an optional Double-DQN target path. It is the anchor algorithm for the
`rlevo-reinforcement-learning` crate — later algorithms (C51, QR-DQN, PPO, ...) reuse the trait
surface and example scaffolding introduced here.

## Modules

- [`dqn_agent`](dqn_agent.rs) — `DqnAgent`, `DqnMetrics`, `DqnAgentError`.
- [`dqn_config`](dqn_config.rs) — `DqnTrainingConfig`, with all
  hyperparameters (batch size, $\gamma$, $\tau$, $\epsilon$ schedule, `learning_starts`,
  `train_frequency`, `double_q`, replay capacity, grad clip).
- [`dqn_model`](dqn_model.rs) — the `DqnModel` trait: `forward`, to
  Q-values, or at output rank 3 to a distribution per action, which is what
  C51 and QR-DQN take.
- [`exploration`](exploration.rs) — multiplicative $\epsilon$-greedy schedule.
- [`train`](train.rs) — end-to-end collect-learn-sync loop.

## Implementing `DqnModel`

There are two networks in DQN:

1. **Policy network** — the agent switches it to training (`train()`) on the
   autodiff form of the device you pass, and trains it on minibatches.
2. **Target network** — a `valid()` snapshot of the policy, with no autodiff,
   updated by `utils::polyak_update` (Polyak averaging, or a periodic hard
   copy at $\tau = 1$).

Both are the same type `M`, so the trait asks for one method.

`MlpQNetwork` is the default: an MLP with `ReLU` hidden layers of the widths
you give, at any observation rank (it flattens the observation first).

```rust
let net = MlpQNetworkConfig {
    obs_dim: 4,
    hidden: vec![64, 64],
    num_actions: 2,
}
.try_init(&device)?;
```

A network of your own implements the trait:

```rust
#[derive(Module, Debug)]
pub struct MyDqn {
    l1: Linear,
    l2: Linear,
}

impl DqnModel<2> for MyDqn {
    fn forward(&self, obs: Tensor<2>) -> Tensor<2> { /* ... */ }
}
```

## Tests

The CartPole integration tests live in the umbrella crate:

```bash
cargo test -p rlevo --test dqn_integration
```

The convergence and reproducibility tests are `#[ignore]`d and run in the
heavy tier:

```bash
cargo xtask test-heavy dqn_integration
```

A reference learning curve, kept as a record (no test or bench reads it), is
`crates/rlevo/tests/baselines/dqn_cartpole.csv` in the workspace repository.

## References

- Mnih et al. (2015), *Human-level control through deep reinforcement
  learning*, Nature.
- van Hasselt, Guez, Silver (2016), *Deep Reinforcement Learning with
  Double Q-learning*, AAAI.
- [CleanRL DQN docs](https://docs.cleanrl.dev/rl-algorithms/dqn/)
