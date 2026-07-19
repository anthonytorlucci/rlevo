# Where rlevo Stands Today

`rlevo` is alpha-stage, so treat this page as a snapshot rather than a
contract. "Stable" below means *implemented, exercised by tests, and used in at
least one benchmark or example* — not *frozen API*. APIs are still moving.

Where an algorithm is implemented and tested but not yet exercised by a
benchmark or example, we mark it **Implemented** rather than Stable. The
distinction matters to you: a Stable entry has been run end-to-end on a real
problem, an Implemented one has only been run against its own tests.

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
| CMA-ES / CMSA-ES | Implemented | Covariance-matrix adaptation with self-contained strategy state ([ADR 0021](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0021-cma-es-placement-and-self-contained-strategy.md)); 43 tests including Sphere-d10 and Rastrigin-d10 convergence. Not yet *Stable* by the definition above — no benchmark or example exercises it, and it is reachable only as `rlevo::evo::algorithms::cma_es::CmaEs`, not from the crate root |

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

**Co-evolution.** Where a fitness function is not fixed but depends on the rest
of the population — predator against prey, or a solution assembled from
sub-populations — we drive evaluation through `CoEvolutionaryHarness` rather
than a plain fitness call.

| Algorithm | Status | Notes |
| --------- | ------ | ----- |
| Competitive co-evolution | Stable | `CompetitiveCoEA`; `HallOfFameFitness` guards against cyclic forgetting |
| Cooperative co-evolution | Stable | `CooperativeCoEA`; `CoupledFitness` composes a full solution from per-species partials |

### Reinforcement learning (`rlevo::rl`)

All eight agents have integration tests and benchmark harnesses.

| Algorithm | Status | Notes |
| --------- | ------ | ----- |
| DQN | Stable | Discrete actions; experience replay (uniform default, opt-in prioritised via `prioritized_replay`), target network, optional Double-DQN (`double_q`) |
| C51 | Stable | Categorical (distributional) DQN; discrete actions; opt-in prioritised replay (prioritises by KL, per Rainbow) |
| QR-DQN | Stable | Quantile-regression (distributional) DQN; discrete actions; opt-in prioritised replay |
| PPO | Stable | On-policy; clipped surrogate, GAE with partial-episode bootstrapping (PEB) on truncation, tanh-Gaussian + categorical heads |
| PPG | Stable | Phasic Policy Gradient; shares PPO's PEB-corrected GAE; auxiliary value phase with KL distillation (v1 discrete-only) |
| DDPG | Stable | Off-policy, continuous actions; deterministic actor + Q-critic, Polyak targets |
| TD3 | Stable | Off-policy, continuous; twin critics, target-policy smoothing, delayed updates |
| SAC | Stable | Off-policy, continuous; squashed-Gaussian actor, twin critics, auto-tuned temperature α |

### Hybrid strategies (`rlevo::hybrid`)

Evolution and RL meet here and nowhere else. `rlevo-evolution` never depends on
`rlevo-core`, so anything that couples a strategy to an
[`Environment`](https://docs.rs/rlevo-core) rollout lives in this crate — the
boundary is deliberate, and it is what keeps the pure-EC side usable without
dragging in the RL stack.

| Component | Status | Notes |
| --------- | ------ | ----- |
| `RolloutFitness` | Stable | Scores a population of flat policy parameters by running episodes against an `Environment` |
| `PolicyNeuroevolution` | Stable | Pairs a `WeightOnly` strategy with `RolloutFitness` inside an `EvolutionaryHarness` |
| `StatefulPolicy` / `ReactivePolicy` | Stable | The recurrent rollout contract, and its memoryless `Hidden = ()` convenience |

### Environments (`rlevo::envs`)

| Family | Environments |
| ------ | ------------ |
| Classic control | CartPole, Acrobot, MountainCar, MountainCarContinuous, Pendulum, Santa Fe Ant Trail |
| Box2D | LunarLander, BipedalWalker, CarRacing |
| Board games | Chess, Connect Four |
| Locomotion (MuJoCo-style) | InvertedPendulum, InvertedDoublePendulum, Reacher, Swimmer |
| Grid worlds | Empty, FourRooms, DoorKey, MultiRoom, LavaGap, Crossing, DynamicObstacles, Memory, Unlock(+Pickup), GoToDoor, DistShift |
| Toy text | FrozenLake, CliffWalking, Taxi, Blackjack |
| K-Armed Bandit | Stationary, non-stationary, adversarial, and contextual variants |
| Landscapes | Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Michalewicz, Concatenated Trap, and ~15 more (see Appendix D) |

## Known gaps and limitations

**DQN on LunarLander.** A DQN agent trained on LunarLander for 150,000 steps
settled into a degenerate hovering policy and scored worse than random. The
implementation is believed correct — it learns CartPole, Acrobot, and the grid
tasks — so DQN itself is not in question.

We originally attributed this to insufficient tuning and training budget, and
that attribution is no longer safe to state as fact: the measurement predates
the fix to LunarLander's crash-termination reward
([issue #122](https://github.com/anthonytorlucci/rlevo/issues/122)), where
crashes failed to terminate the episode and could net positive reward. An agent
that learned to hover is exactly what that reward signal would have selected
for. The benchmark needs re-running against the corrected environment before
any cause is assigned ([issue #270](https://github.com/anthonytorlucci/rlevo/issues/270)).

**Multi-objective optimisation.** `rlevo` does not implement NSGA-II or any
Pareto-based selection. Single-objective only.

**Distributed evaluation.** Population evaluation is parallelised locally via
`rayon`. There is no built-in support for distributing evaluation across
machines.

**Residual GAE weighting bias at episode boundaries.** Partial-episode
bootstrapping ([ADR 0048](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0048-partial-episode-bootstrapping-in-gae.md))
corrects the terminal *bootstrap value* for a truncated episode, but GAE's
geometric weighting still collapses its mass onto the longest available k-step
estimator right at any episode boundary — a separate issue Doering et al.
(2026) identify and partially correct on environments with strong terminal
signal, such as Lunar Lander. This is a **live research question, not a defect
in `rlevo`'s PEB fix**; see [Rewards](../part-1-foundations/reinforcement-learning/33-reward.md#gae-and-truncation-two-masks-not-one)
for where it is discussed alongside the fix it does not replace.

**Reproducibility with GPU backends.** The `wgpu` backend uses GPU kernels whose
non-determinism can break exact reproducibility even with a fixed seed.
Deterministic runs require both a seeded backend RNG and pinning `rayon` to a
single thread — `rayon`'s work-stealing changes reduction order, and the CPU
path (`burn-flex`) parallelises its GEMM through it. The examples do exactly
this; see the note in `crates/rlevo-examples/Cargo.toml`.

We enable the `wgpu` and `flex` Burn backends only. Burn's `ndarray` backend is
*not* wired up in this workspace, so it is not an option you can reach for
today — if you need a deterministic CPU run, pin `rayon` on `flex` rather than
switching backend.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
