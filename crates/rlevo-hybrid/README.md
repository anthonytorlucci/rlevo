# rlevo-hybrid

![Alt Text](https://raw.githubusercontent.com/anthonytorlucci/rlevo/main/rlevo-logo.png)

Hybrid evolutionary + deep reinforcement learning strategies for the `rlevo` workspace,
built on [Burn](https://burn.dev/) tensor operations.

## Status

**Alpha, evolution-guided policy init shipped.** This crate is the
boundary between pure evolution (`rlevo-evolution`, which never names an
`Environment`) and anything that couples an evolutionary strategy to an
`Environment` rollout. It depends on `rlevo-core` and `rlevo-evolution`.
Two modules ship today:

- `rollout_fitness::RolloutFitness` — a `BatchFitnessFn` that scores a
  population of flat policy parameters by running episodes against an
  `Environment`. Hand it to an `EvolutionaryLoop` with any flat-genome
  strategy to evolve the weights of a fixed-topology policy network
  directly against environment return.
- `policy::StatefulPolicy` — the rollout contract by which a policy module
  carries per-episode state across a `RolloutFitness` rollout (`Hidden = ()`
  for a memoryless policy).

Exercised end-to-end by `tests/cartpole_smoke.rs` (evolves a real CartPole
MLP policy) and `tests/stateful_rollout.rs`. Population-based training,
ERL-style concurrent evolution + gradient RL, and CEM-RL remain future work.

## Planned strategies

| Strategy | Description | Status |
|---|---|---|
| Evolution-Guided Policy Init | Evolve policy weights directly against rollout fitness (`RolloutFitness` in an `EvolutionaryLoop`) | **Shipped** |
| Population-Based Training (PBT) | Jaderberg et al.'s hyperparameter and weight exploitation/exploration schedule | Planned |
| ERL (Evolutionary RL) | Khadka & Tumer: concurrent evolutionary population + gradient-trained RL actor | Planned |
| CEM-RL | Pourchot & Sigaud: interleaved CMA-ES population + TD3 actor | Planned |

## Design intent

`RolloutFitness<M, E>` scores flat policy weights by rollout, so any
flat-genome strategy in an `EvolutionaryLoop` optimizes a fixed-topology
policy module `M` purely by black-box search against environment `E`'s
return — no gradients. Planned strategies that interleave gradient-based RL
with the evolutionary loop (PBT, ERL, CEM-RL) will build on this same
rollout/fitness seam. The first of them brings a dependency on
`rlevo-reinforcement-learning` with it; the crate has none today, because
nothing in it uses that crate.

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
