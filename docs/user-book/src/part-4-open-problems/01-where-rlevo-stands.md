# Where rlevo Stands Today

`rlevo` is alpha-stage, so treat this page as a snapshot rather than a
contract. "Stable" below means *implemented, exercised by tests, and used in at
least one benchmark or example* — not *frozen API*. APIs are still moving.

For the full algorithm catalogue with derivations and configuration knobs, see
[Appendix A (EC)](../appendix-a-ec-algorithms/index.md),
[Appendix B (RL)](../appendix-b-rl-algorithms/index.md), and
[Appendix C (Hybrid)](../appendix-c-hybrid-algorithms/index.md).

## What is implemented

### Evolutionary algorithms (`rlevo::evo`)

| Algorithm | Status | Notes |
| --------- | ------ | ----- |
| Genetic Algorithm — binary | Stable | Bit-string genomes, uniform crossover, bit-flip mutation, elitist replacement |
| Genetic Algorithm — real-valued | Stable | BLX-α / uniform crossover, Gaussian mutation, generational and elitist replacement |
| Evolution Strategies | Stable | `(1+1)`, `(1+λ)`, `(μ,λ)`, `(μ+λ)`; 1/5th rule and log-normal σ adaptation |
| Differential Evolution | Stable | `Rand1Bin/Exp`, `Rand2Bin`, `Best1Bin`, `CurrentToBest1Bin`; greedy per-slot replacement |
| Evolutionary Programming | Stable | Fogel-style; per-individual log-normal σ, q-tournament survivor selection |
| CMA-ES / CMSA-ES | Planned | Covariance-matrix adaptation; tracked in [issue #59](https://github.com/anthonytorlucci/rlevo/issues/59) |

**Estimation-of-distribution algorithms.** All share the `EdaStrategy`
`fit → sample` driver; only the `ProbabilityModel` changes.

| Algorithm | Status | Notes |
| --------- | ------ | ----- |
| UMDA (`UnivariateGaussian`) | Stable | Per-dimension Gaussian; unweighted MLE with variance floor |
| PBIL (`UnivariateBernoulli`) | Stable | Per-bit probability vector; best/worst update |
| cGA (`CompactGenetic`) | Stable | Virtual-population probability vector |
| MIMIC (`DependencyChain`) | Stable | Pairwise dependency chain |
| BOA (`BayesianNetwork`) | Stable | BIC-scored Bayesian-network DAG over binary genes; ancestral sampling |

**Program evolution, memetics, neuroevolution, and swarm metaheuristics.**

| Algorithm | Status | Notes |
| --------- | ------ | ----- |
| Cartesian Genetic Programming | Stable | Fixed-grid CGP; configurable function set |
| Gene Expression Programming | Stable | Head/tail chromosome; symbolic regression |
| Memetic wrapper | Stable | Hill climbing, Nelder–Mead, SA, random restart; Lamarckian / Baldwinian / partial write-back |
| Neuroevolution — weight-only | Stable | Evolves flattened weights of any Burn `Module` with any real-valued strategy |
| Neuroevolution — architecture NAS | Stable | Co-evolves fixed-topology module variant + weights |
| Neuroevolution — NEAT | Stable | Topology growth, speciation, innovation-aligned crossover |
| PSO | Stable | Particle Swarm Optimization |
| ACO — continuous (`ACO_R`) | Stable | Ant Colony for continuous domains |
| ACO — permutation (`aco_perm`) | Stub | Deferred to a future release |
| ABC, Cuckoo Search, Firefly | Stable | Nature-inspired continuous metaheuristics |
| GWO, WOA, Bat, SSA | Stable (legacy comparator) | Retained for comparison; see Sörensen (2015), Camacho-Villalón et al. (2023) |

### Reinforcement learning (`rlevo::rl`)

All eight agents have integration tests and benchmark harnesses.

| Algorithm | Status | Notes |
| --------- | ------ | ----- |
| DQN | Stable | Discrete actions; experience replay, target network, optional Double-DQN (`double_q`) |
| C51 | Stable | Categorical (distributional) DQN; discrete actions |
| QR-DQN | Stable | Quantile-regression (distributional) DQN; discrete actions |
| PPO | Stable | On-policy; clipped surrogate, GAE, tanh-Gaussian + categorical heads |
| PPG | Stable | Phasic Policy Gradient; auxiliary value phase with KL distillation (v1 discrete-only) |
| DDPG | Stable | Off-policy, continuous actions; deterministic actor + Q-critic, Polyak targets |
| TD3 | Stable | Off-policy, continuous; twin critics, target-policy smoothing, delayed updates |
| SAC | Stable | Off-policy, continuous; squashed-Gaussian actor, twin critics, auto-tuned temperature α |

### Environments (`rlevo::envs`)

| Family | Environments |
| ------ | ------------ |
| Classic control | CartPole, Acrobot, MountainCar, MountainCarContinuous, Pendulum |
| Box2D | LunarLander, BipedalWalker, CarRacing |
| Locomotion (MuJoCo-style) | InvertedPendulum, InvertedDoublePendulum, Reacher, Swimmer |
| Grid worlds | Empty, FourRooms, DoorKey, MultiRoom, LavaGap, Crossing, DynamicObstacles, Memory, Unlock(+Pickup), GoToDoor, DistShift |
| Toy text | FrozenLake, CliffWalking, Taxi, Blackjack |
| K-Armed Bandit | Stationary and non-stationary variants |
| Landscapes | Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Michalewicz, Concatenated Trap, and ~15 more (see Appendix D) |

## Known gaps and limitations

**DQN on LunarLander.** A DQN agent trained on LunarLander for 150,000 steps
performed worse than random. The implementation is believed correct (it learns
CartPole, Acrobot, and grid tasks); the root cause is insufficient
hyperparameter tuning and training budget. This benchmark is deferred pending a
proper tuning effort — DQN itself is not in question.

**Multi-objective optimisation.** `rlevo` does not implement NSGA-II or any
Pareto-based selection. Single-objective only.

**Covariance-matrix adaptation.** CMA-ES / CMSA-ES are not yet implemented
([issue #59](https://github.com/anthonytorlucci/rlevo/issues/59)). For problems
with strong variable dependencies, this is the most-missed strategy.

**Distributed evaluation.** Population evaluation is parallelised locally via
`rayon`. There is no built-in support for distributing evaluation across
machines.

**Reproducibility with GPU backends.** The `wgpu` backend uses GPU kernels whose
non-determinism can break exact reproducibility even with a fixed seed.
Deterministic runs require both a seeded backend RNG and pinning `rayon` to a
single thread; the `ndarray` backend is fully deterministic.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
