# rlevo-hybrid

![Alt Text](rlevo-logo.png)

Hybrid evolutionary + deep reinforcement learning strategies for the `rlevo` workspace,
built on [Burn](https://burn.dev/) tensor operations.

## Status

**Alpha, evolution-guided policy init shipped.** This crate is the
dependency boundary between pure evolution (`rlevo-evolution`, which never
depends on `rlevo-core`) and anything that couples an evolutionary strategy
to an `Environment` rollout. Three modules ship today:

- `rollout_fitness::RolloutFitness` — a `BatchFitnessFn` that scores a
  population of flat policy parameters by running episodes against an
  `Environment`.
- `policy_neuroevolution::PolicyNeuroevolution` — pairs a `WeightOnly`
  evolutionary strategy with `RolloutFitness` inside an
  `EvolutionaryHarness`, evolving the weights of a fixed-topology policy
  network directly against environment return.
- `policy::{StatefulPolicy, ReactivePolicy}` — the rollout contract by
  which a policy module carries per-episode (recurrent or stateless) state
  across a `RolloutFitness` rollout.

Exercised end-to-end by `tests/cartpole_smoke.rs` (evolves a real CartPole
MLP policy) and `tests/stateful_rollout.rs`. Population-based training,
ERL-style concurrent evolution + gradient RL, and CEM-RL remain future work.

## Planned strategies

| Strategy | Description | Status |
|---|---|---|
| Evolution-Guided Policy Init | Evolve policy weights directly against rollout fitness (`WeightOnly` + `RolloutFitness`) | **Shipped** — `PolicyNeuroevolution` |
| Population-Based Training (PBT) | Jaderberg et al.'s hyperparameter and weight exploitation/exploration schedule | Planned |
| ERL (Evolutionary RL) | Khadka & Tumer: concurrent evolutionary population + gradient-trained RL actor | Planned |
| CEM-RL | Pourchot & Sigaud: interleaved CMA-ES population + TD3 actor | Planned |

## Design intent

`PolicyNeuroevolution<B, S, M, E>` wires `rlevo-evolution`'s `WeightOnly`
strategy and `RolloutFitness` into an `EvolutionaryHarness`, so a
fixed-topology policy module `M` is optimized purely by black-box search
against environment `E`'s return — no gradients. Planned strategies that
interleave gradient-based RL with the evolutionary loop (PBT, ERL, CEM-RL)
will build on this same rollout/fitness seam once `rlevo-reinforcement-learning`
integration lands.

## Related crates

- [`rlevo-evolution`](../rlevo-evolution) — tensor-native evolutionary algorithms
- [`rlevo-reinforcement-learning`](../rlevo-reinforcement-learning) — deep RL algorithms (DQN, PPO, SAC, …)
- [`rlevo-core`](../rlevo-core) — shared traits and environment abstractions

## References

- S. Khadka and K. Tumer, "Evolution-guided policy gradient in reinforcement learning," in *Advances in Neural Information Processing Systems*, vol. 31, 2018. [arXiv](https://arxiv.org/abs/1805.07917)
- A. Pourchot and O. Sigaud, "CEM-RL: Combining evolutionary and gradient-based methods for policy search," in *Proc. ICLR*, 2019. [arXiv](https://arxiv.org/abs/1810.01222)
- M. Jaderberg, V. Dalibard, S. Osindero, W. M. Czarnecki, J. Donahue, A. Razavi, O. Vinyals, T. Green, I. Dunning, K. Simonyan, C. Fernando, and K. Kavukcuoglu, "Population based training of neural networks," arXiv:1711.09846, 2017. [arXiv](https://arxiv.org/abs/1711.09846)

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or [MIT License](../../LICENSE-MIT) at your option.
