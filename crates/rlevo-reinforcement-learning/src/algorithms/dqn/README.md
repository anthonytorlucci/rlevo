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
- [`dqn_model`](dqn_model.rs) — the `DqnModel` trait: `forward`.
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

## Running the example

```bash
cargo run -p rlevo-reinforcement-learning --release --example dqn_cart_pole -- \
    --seed 42 --total-timesteps 50000 --log-every 1000
```

On ndarray the 100-episode moving average climbs past 100 within ~15k env
steps and past 180 within 30k. Reference curve, kept as a record (no test or
bench reads it): `crates/rlevo/tests/baselines/dqn_cartpole.csv` in the
workspace repository.

## Tests

```bash
cargo test -p rlevo-reinforcement-learning --release --test dqn_integration
```

Two reproducibility/smoke tests are gated behind `#[ignore]` because they
perturb Burn's global ndarray RNG when run in parallel. Exercise them with:

```bash
cargo test -p rlevo-reinforcement-learning --release --test dqn_integration -- \
    --ignored --test-threads=1
```

## References

- Mnih et al. (2015), *Human-level control through deep reinforcement
  learning*, Nature.
- van Hasselt, Guez, Silver (2016), *Deep Reinforcement Learning with
  Double Q-learning*, AAAI.
- [CleanRL DQN docs](https://docs.cleanrl.dev/rl-algorithms/dqn/)
