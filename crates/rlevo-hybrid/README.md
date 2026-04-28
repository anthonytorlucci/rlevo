# rlevo-hybrid

Hybrid evolutionary + deep reinforcement learning strategies for the `rlevo` workspace,
built on [Burn](https://burn.dev/) tensor operations.

## Status

**Alpha stub.** The crate is reserved for hybrid algorithms that combine
`rlevo-evolution` and `rlevo-reinforcement-learning` — for example,
evolution-guided policy initialisation, population-based training (PBT),
and ERL-style concurrent evolution + RL actor pools. No hybrid strategies
are implemented in v0.1.0; this release establishes the crate skeleton and
dependency wiring.

## Planned strategies

| Strategy | Description | Target |
|---|---|---|
| Evolution-Guided Policy Init | Run a few generations of an evolutionary algorithm to seed a good starting policy for PPO/SAC | v0.2.0 |
| Population-Based Training (PBT) | Jaderberg et al.'s hyperparameter and weight exploitation/exploration schedule | v0.2.0 |
| ERL (Evolutionary RL) | Khadka & Tumer: concurrent evolutionary population + gradient-trained RL actor | v0.3.0 |
| CEM-RL | Pourchot & Sigaud: interleaved CMA-ES population + TD3 actor | v0.3.0 |

## Design intent

The crate sits at the intersection of `rlevo-evolution`'s
`Strategy<B>` trait and `rlevo-reinforcement-learning`'s gradient-trained
agents. The planned abstraction is a `HybridHarness<B, S, A>` that runs an
evolutionary outer loop and a gradient inner loop, sharing parameters or
fitness signals between them.

## Related crates

- [`rlevo-evolution`](../rlevo-evolution) — tensor-native evolutionary algorithms
- [`rlevo-reinforcement-learning`](../rlevo-reinforcement-learning) — deep RL algorithms (DQN, PPO, SAC, …)
- [`rlevo-core`](../rlevo-core) — shared traits and environment abstractions

## References

- S. Khadka and K. Tumer, "Evolution-guided policy gradient in reinforcement learning," in *Advances in Neural Information Processing Systems*, vol. 31, 2018. [arXiv](https://arxiv.org/abs/1805.07917)
- A. Pourchot and O. Sigaud, "CEM-RL: Combining evolutionary and gradient-based methods for policy search," in *Proc. ICLR*, 2019. [arXiv](https://arxiv.org/abs/1810.01222)
- M. Jaderberg, V. Dalibard, S. Osindero, W. M. Czarnecki, J. Donahue, A. Razavi, O. Vinyals, T. Green, I. Dunning, K. Simonyan, C. Fernando, and K. Kavukcuoglu, "Population based training of neural networks," arXiv:1711.09846, 2017. [arXiv](https://arxiv.org/abs/1711.09846)

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
