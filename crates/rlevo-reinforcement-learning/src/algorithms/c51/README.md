# Categorical DQN (C51)

`rlevo-reinforcement-learning`'s C51 implementation learns a **discrete return distribution**
over a fixed atom support instead of a scalar Q-value. It reuses the DQN
scaffolding ($\epsilon$-greedy schedule, replay buffer, target-network sync, Adam
optimizer) and adds:

- a categorical projection operator that back-propagates the Bellman update
  through a fixed support (`project_distribution`);
- a cross-entropy loss between the projected target distribution and the
  policy's log-probabilities for the taken action;
- an expectation-based action selector: $\arg\max_a \sum_i z_i \cdot \mathrm{softmax}(\text{logits})_i$.

## Modules

- [`c51_agent`](c51_agent.rs) — `C51Agent` and its `Agent` impl.
- [`c51_config`](c51_config.rs) — `C51TrainingConfig`, with all
  hyperparameters, including `num_atoms`, `v_min`, `v_max`.
- [`projection`](projection.rs) — `project_distribution` (Algorithm 1,
  Bellemare et al. 2017) implemented via Burn's `scatter`.
- [`loss`](loss.rs) — `categorical_cross_entropy_per_sample` free function.

## Implementing the network

C51's network is a [`DqnModel`](../dqn/dqn_model.rs) at output rank 3: it
returns atom logits `(batch, num_actions, num_atoms)`, and the agent takes
the softmax over atoms. QR-DQN uses the same trait and reads the same shape
as quantile values. `MlpDistributionalQ` is the default for both:

```rust
let net = MlpDistributionalQConfig {
    obs_dim: 4,
    hidden: vec![64, 64],
    num_actions: 2,
    num_atoms: 51,
}
.try_init(&device)?;
```

A network of your own implements the trait:

```rust
#[derive(Module, Debug)]
pub struct MyC51 {
    l1: Linear,
    l2: Linear,
    head: Linear, // output = num_actions * num_atoms
    num_atoms: usize,
}

impl DqnModel<2, 3> for MyC51 {
    fn forward(&self, obs: Tensor<2>) -> Tensor<3> {
        let [batch, _] = obs.dims();
        let x = activation::relu(self.l1.forward(obs));
        let x = activation::relu(self.l2.forward(x));
        self.head
            .forward(x)
            .reshape([batch, NUM_ACTIONS, self.num_atoms])
    }
}
```

## Tests

The CartPole integration tests live in the umbrella crate; the `#[ignore]`d
convergence tests run in the heavy tier:

```bash
cargo test -p rlevo --test c51_integration
cargo xtask test-heavy c51_integration
```

## References

- Bellemare, Dabney, Munos (2017), *A Distributional Perspective on
  Reinforcement Learning*, ICML.
- [CleanRL C51 docs](https://docs.cleanrl.dev/rl-algorithms/c51/)
