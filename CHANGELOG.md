# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] – 2026-04-28

Initial alpha release. All crates are published together at the same version.

### `rlevo-core`

**Added**

- `State<D>` / `Observation<D>` traits for typed, const-generic environment state and agent perception.
- `Action<D>` trait hierarchy (`DiscreteAction`, `ContinuousAction`, `MultiDiscreteAction`, `MultiBinaryAction`) for compile-time action-space safety.
- `Environment<D, SD, AD>` trait with `reset` / `step` / `render` contract; `Snapshot<D>` trait with `SnapshotBase<D, O, R>` concrete type.
- `Reward` trait with `ScalarReward` and `VectorReward` implementations.
- `TensorConvertible<B, D>` bridge trait for lifting state/action types onto Burn tensors.
- `Agent` and `BenchableAgent` traits for uniform agent interaction.
- `FitnessEvaluable` and `Landscape` traits for benchmarking evolutionary algorithms.
- `BenchEnv`, `BenchError`, `BenchStep`, `Metric`, `MetricsProvider`, and `SeedStream` (moved from `rlevo-benchmarks` per ADR-0004).
- `util::seed` — deterministic `SeedStream` for reproducible multi-run experiments.
- `EnvironmentError` and `StateError` error types with `thiserror` derives.

### `rlevo-environments`

**Added**

- **Classic control** — `CartPole`, `MountainCar`, `MountainCarContinuous`, `Pendulum`, `Acrobot`.
- **Bandits** — `KArmedBandit<K>`, `ContextualBandit`, `NonStationaryBandit`, `AdversarialBandit`.
- **Toy text** — `Blackjack`, `CliffWalking`, `FrozenLake`, `Taxi`.
- **Gridworlds** (MiniGrid-style) — `Empty`, `DoorKey`, `Memory`, `FourRooms`, `Crossing`, `LavaGap`, `MultiRoom`, `Unlock`, `UnlockPickup`, `GoToDoor`, `DistShift`, `DynamicObstacles`; shared `GridCore` (grid, entity, action, direction, observation, render, reward, dynamics).
- **Box2D physics** (`box2d` feature, rapier2d) — `BipedalWalker`, `LunarLander` (discrete and continuous action spaces), `CarRacing`.
- **Locomotion** (`locomotion` feature, rapier3d) — `Reacher`, `Swimmer`, `InvertedPendulum`, `InvertedDoublePendulum`.
- **Games** — `Chess` (full move generation and board state), `ConnectFour`.
- **Optimisation landscapes** — `Sphere`, `Ackley`, `Rastrigin` for benchmarking evolutionary algorithms.
- **Wrappers** — `TimeLimit` wraps any `Environment` with an episode step cap.
- **Bench adapter** (`bench` feature) — `BenchAdapter` and preset `Suite` factories to drive any environment from `rlevo-benchmarks`.
- ASCII render backend for text-based environments.

### `rlevo-evolution`

**Added**

- `Strategy<B>` pure trait (`init` / `ask` / `tell` / `best`) — stateless, parallelism-friendly, trivially checkpointable.
- `EvolutionaryHarness<B, S, F>` — wraps any `Strategy` as a `BenchEnv`.
- `BatchFitnessFn` trait with `FromFitnessEvaluable` adapter.
- `GenomeKind` enum (`RealValued`, `Binary`, `Integer`, `Program`).
- **Classical families** — `GeneticAlgorithm` (real-valued, SBX crossover + polynomial mutation), `BinaryGeneticAlgorithm` (one-point/uniform crossover + bit-flip mutation), `EvolutionStrategy` (`(1+1)`, `(1+λ)`, `(μ,λ)`, `(μ+λ)` with self-adaptive σ), `EvolutionaryProgramming` (Gaussian perturbation + tournament), `DifferentialEvolution` (Rand/1/Bin, Best/1/Bin, CurrentToBest/1/Bin), `CartesianGeneticProgramming` (symbolic regression via CGP graph).
- **Metaheuristics** — `ParticleSwarmOptimization`, `AntColonyOptimizationReal`, `AntColonyOptimizationPermutation`, `ArtificialBeeColony`, `FireflyAlgorithm`, `BatAlgorithm`, `CuckooSearch` (Lévy flights via Mantegna), `GreyWolfOptimizer`, `SalpSwarmAlgorithm`, `WhaleOptimizationAlgorithm`.
- **Genetic operators** (`ops`) — selection (tournament, roulette, rank, SUS, elitism, NSGA-II crowding), crossover (uniform, one-point, multi-point, SBX, BLX-α, intermediate), mutation (Gaussian, uniform, polynomial, bit-flip, inversion), replacement (generational, steady-state, elitist, comma, plus).
- **Custom CubeCL kernels** (`custom-kernels` feature) — fused pairwise-attract (Firefly large-N path) and fused Lévy-flight (Cuckoo/Bat) kernels; pure-tensor fallbacks used when feature is off.
- `PopulationState` tensor wrapper; `ShapingFn` fitness shaping (linear rank, exponential rank, truncation).

### `rlevo-reinforcement-learning`

**Added**

- **Replay memory** — `PrioritizedExperienceReplay` (uniform-sampling mode in v0.1.0); `TrainingBatch` typed container.
- **Experience** — `ExperienceTuple` (s, a, r, s', done), `History` trajectory buffer.
- **Metrics** — `AgentStats` (per-step), `PerformanceRecord` (per-episode).
- **DQN** — `DqnModel`, `DqnAgent`, `DqnTrainingConfig`; ε-greedy exploration schedule; Double-DQN target option.
- **C51** — `C51Model`, `C51Agent`, `C51TrainingConfig`; Bellman projection onto N-atom support; categorical cross-entropy loss.
- **QR-DQN** — `QrDqnModel`, `QrDqnAgent`, `QrDqnTrainingConfig`; quantile Huber loss; no `[v_min, v_max]` required.
- **PPO** — `PpoAgent`, `PpoTrainingConfig`, `RolloutBuffer`, GAE advantages, `CategoricalPolicyHead` (discrete), `TanhGaussianPolicyHead` (continuous), clipped surrogate + value loss, early-stop on `approx_kl`.
- **PPG** — `PpgAgent`, `PpgConfig`, `AuxBuffer`, `PpgCategoricalPolicyHead`; interleaved policy-phase + auxiliary-phase with KL distillation.
- **DDPG** — `DdpgAgent`, `DdpgTrainingConfig`; deterministic actor + Q-critic; Polyak target sync; Gaussian exploration noise.
- **TD3** — `Td3Agent`, `Td3TrainingConfig`; twin-critic min-bootstrap; delayed actor updates; target-policy smoothing.
- **SAC** — `SacAgent`, `SacTrainingConfig`; squashed-Gaussian stochastic actor; twin critics; learnable temperature α with auto-tuning toward `-|A|`.
- Shared `EpsilonGreedy` schedule (DQN / C51 / QR-DQN) and `GaussianNoise` exploration (DDPG / TD3).

### `rlevo-benchmarks`

**Added**

- `Evaluator` — drives any `BenchEnv` for N episodes, collecting per-step and per-episode metrics.
- `Suite` — ordered sequence of `(env, evaluator)` pairs with shared reporter.
- **Metrics** — `EaMetrics` (best fitness, population diversity, convergence rate), `RlMetrics` (episode return, episode length, sample efficiency).
- **Reporters** — `JsonReporter` (newline-delimited JSON), `LoggingReporter` (tracing spans), `TuiReporter` (ratatui live dashboard, `tui` feature).
- `Checkpoint` and `Storage` traits for saving/resuming benchmark state.
- `rayon`-parallel episode evaluation for multi-seed sweeps.

### `rlevo-hybrid`

**Added**

- Stub crate establishing the dependency wiring between `rlevo-evolution` and `rlevo-reinforcement-learning`. No hybrid strategies are implemented in v0.1.0; see the crate README for the v0.2.0 roadmap.

### `rlevo` (umbrella)

**Added**

- Re-exports all public APIs from every workspace crate behind a single `rlevo` entry point.
- `keywords`: `reinforcement-learning`, `evolutionary`, `deep-learning`, `burn`, `neural-network`.
- `categories`: `science`, `algorithms`, `simulation`.
- Full example suite (35 examples across gridworlds, classic control, Box2D, locomotion, evolutionary showcases, RL algorithms, and benchmarks harness).
- Cross-crate integration tests.

---

[0.1.0]: https://github.com/anthonytorlucci/rlevo/releases/tag/v0.1.0
