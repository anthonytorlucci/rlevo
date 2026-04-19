# Deep Q-Network (DQN)

`evorl-rl`'s DQN implementation follows the Nature-style Q-learning algorithm
with an optional Double-DQN target path. It is the anchor algorithm for the
`evorl-rl` crate — later specs (C51, Rainbow, PPO) will reuse the trait
surface and example scaffolding introduced here.

## Modules

- [`dqn_agent`](dqn_agent.rs) — `DqnAgent`, `DqnMetrics`, `DqnAgentError`.
- [`dqn_config`](dqn_config.rs) — `DqnTrainingConfig` + builder with all
  hyperparameters (batch size, γ, τ, ε schedule, `learning_starts`,
  `train_frequency`, `double_q`, replay capacity, grad clip).
- [`dqn_model`](dqn_model.rs) — the `DqnModel` trait: `forward`,
  `forward_inner`, and `soft_update` for Polyak averaging.
- [`exploration`](exploration.rs) — multiplicative ε-greedy schedule.
- [`train`](train.rs) — end-to-end collect-learn-sync loop.

## Implementing `DqnModel`

There are two networks in DQN:

1. **Policy network** — autodiff backend `B`, trained on minibatches.
2. **Target network** — `M::InnerModule` on `B::InnerBackend`, updated via
   Polyak averaging or periodic hard copy.

```rust
#[derive(Module, Debug)]
pub struct MyDqn<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

impl<B: AutodiffBackend> DqnModel<B, 2> for MyDqn<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> { /* ... */ }
    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> { /* ... */ }
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Self::InnerModule { /* use ModuleVisitor + ModuleMapper */ }
}
```

See `examples/dqn_cart_pole.rs` for a complete `ModuleVisitor` /
`ModuleMapper`-based Polyak update.

## Running the example

```bash
cargo run -p evorl-rl --release --example dqn_cart_pole -- \
    --seed 42 --total-timesteps 50000 --log-every 1000
```

On ndarray the 100-episode moving average climbs past 100 within ~15k env
steps and past 180 within 30k. Reference curve: `tests/baselines/dqn_cartpole.csv`.

## Tests

```bash
cargo test -p evorl-rl --release --test dqn_integration
```

Two reproducibility/smoke tests are gated behind `#[ignore]` because they
perturb Burn's global ndarray RNG when run in parallel. Exercise them with:

```bash
cargo test -p evorl-rl --release --test dqn_integration -- \
    --ignored --test-threads=1
```

## References

- Mnih et al. (2015), *Human-level control through deep reinforcement
  learning*, Nature.
- van Hasselt, Guez, Silver (2016), *Deep Reinforcement Learning with
  Double Q-learning*, AAAI.
- [CleanRL DQN docs](https://docs.cleanrl.dev/rl-algorithms/dqn/)
