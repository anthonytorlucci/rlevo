# Categorical DQN (C51)

`rlevo-reinforcement-learning`'s C51 implementation learns a **discrete return distribution**
over a fixed atom support instead of a scalar Q-value. It reuses the DQN
scaffolding (ε-greedy schedule, replay buffer, target-network sync, Adam
optimizer) and adds:

- a categorical projection operator that back-propagates the Bellman update
  through a fixed support (`project_distribution`);
- a cross-entropy loss between the projected target distribution and the
  policy's log-probabilities for the taken action;
- an expectation-based action selector: `argmax_a Σ_i z_i · softmax(logits)_i`.

## Modules

- [`c51_agent`](c51_agent.rs) — `C51Agent`, `C51Metrics`, `C51AgentError`.
- [`c51_config`](c51_config.rs) — `C51TrainingConfig` + builder with all
  hyperparameters, including `num_atoms`, `v_min`, `v_max`.
- [`c51_model`](c51_model.rs) — the `C51Model` trait returning a rank-3
  tensor of atom logits `(batch, num_actions, num_atoms)`.
- [`projection`](projection.rs) — `project_distribution` (Algorithm 1,
  Bellemare et al. 2017) implemented via Burn's `scatter`.
- [`loss`](loss.rs) — `categorical_cross_entropy` free function.
- [`train`](train.rs) — end-to-end collect-learn-sync loop.

## Implementing `C51Model`

```rust
#[derive(Module, Debug)]
pub struct MyC51<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    head: Linear<B>, // output = num_actions * num_atoms
    num_atoms: usize,
}

impl<B: AutodiffBackend> C51Model<B, 2> for MyC51<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _] = obs.dims();
        let x = activation::relu(self.l1.forward(obs));
        let x = activation::relu(self.l2.forward(x));
        self.head
            .forward(x)
            .reshape([batch, NUM_ACTIONS, self.num_atoms])
    }
    // forward_inner / soft_update as for DqnModel.
}
```

See `examples/c51_cart_pole.rs` for a runnable implementation.

## Running the example

```bash
cargo run -p rlevo-reinforcement-learning --release --example c51_cart_pole -- \
    --seed 42 --total-timesteps 50000 --log-every 1000 --num-atoms 51
```

## Tests

```bash
cargo test -p rlevo-reinforcement-learning --release --test c51_integration
```

## References

- Bellemare, Dabney, Munos (2017), *A Distributional Perspective on
  Reinforcement Learning*, ICML.
- [CleanRL C51 docs](https://docs.cleanrl.dev/rl-algorithms/c51/)
