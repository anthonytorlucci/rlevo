# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Breaking changes

- **`SnapshotBase` gains `metadata: Option<SnapshotMetadata>`** (ADR 0042,
  resolves #128) — `SnapshotBase<R, ObservationType, RewardType>` now carries
  an optional `SnapshotMetadata` field and a fluent `#[must_use]
  with_metadata(self, SnapshotMetadata) -> Self` builder; `Snapshot::metadata()`
  is overridden on `SnapshotBase` to return it instead of the inherited `None`
  default. `running`/`terminated`/`truncated` now construct with `metadata:
  None`; attach metadata with a `.with_metadata(...)` tail. The two bespoke
  hand-rolled `impl Snapshot<1>` types collapse to type aliases over
  `SnapshotBase` — `LocomotionSnapshot<O>` (`rlevo-environments::locomotion::common`)
  and `LunarLanderSnapshot` (`rlevo-environments::box2d::lunar_lander::snapshot`) —
  so their type names are unaffected, but their constructors' metadata
  arguments move to the `.with_metadata()` tail (no `#[deprecated]` shim: a
  constructor cannot be deprecated-and-retained on a type that is now a
  foreign alias). This unblocks `TimeLimit` composition for all six
  previously-locked-out environments (4 locomotion + `LunarLanderDiscrete` /
  `LunarLanderContinuous`).

- **`MemoryEnv` and `GoToDoorEnv` config and observation surfaces change**
  (ADR 0043, resolves #109) — both environments claimed properties they did not
  have, and fixing them removes the config fields that caused the defect.
  - `MemoryConfig::swap_fork` is **removed**, and `MemoryConfig::new` changes
    arity: `new(size, max_steps, seed)` (was `new(max_steps, seed, swap_fork)`).
    `size` is a new field — odd and `>= 11`, rejected by `Validate` otherwise.
    Defaults are `size = 13`, `max_steps = 845` (`5 * size²`), `seed = 0`.
    The default sits deliberately **above** the minimum: `11` is the smallest
    size at which the cue is unobservable from the fork (Invariant M), but it is
    also the size at which the cue-free corridor run collapses to a single cell,
    so it is the *weakest* recall task the layout supports. `13` gives a
    three-cell cue-free run for ~40% more step budget.
    A `swap_fork=…` key in a `MemoryConfig` config string is now an error, not
    a silently ignored no-op.
  - `GoToDoorConfig::target_color` is **removed**, and `GoToDoorConfig::new`
    changes arity: `new(size, max_steps, seed)` (was
    `new(size, max_steps, seed, target_color)`). The target is sampled per
    episode. `target_color=…` / `color=…` config-string keys are now errors.
  - **`GoToDoorEnv`'s observation and snapshot types change.** It no longer
    emits the shared `GridObservation` / `GridSnapshot`; it emits
    `GoToDoorObservation` (`[7, 7, 4]`) and `GoToDoorSnapshot`. Rank is still
    `3`, so `Environment<3, 3, 1>` is unchanged, but any code naming its
    `ObservationType` / `SnapshotType`, or feeding a `7×7×3` model, must be
    updated. This is the grid family's only 4-channel observation.
  - Both configs pinned a quantity the environment is supposed to sample every
    episode; determinism for tests is served by the new `reset_with_seed` (ADR
    0029) instead, which exercises the real sampling environment.

- **`ContextualBanditObservation` closes its construction surface** (resolves
  #124) — the `pub context: usize` field is **private**; read it with the new
  `context()` accessor and build one with the fallible
  `ContextualBanditObservation::<C>::new(context) -> Result<Self, StateError>`,
  which rejects `context >= C` with `StateError::InvalidData`. The public field
  let a caller construct an out-of-range context that then panicked with an
  index-out-of-bounds inside `TensorConvertible::write_host_row`'s one-hot
  encoder — a panic on user-supplied data, which `docs/rules.md` §4 forbids.
  `Deserialize` is now hand-written and validates through the same constructor,
  so the identical hole is closed on the serde path (the wire format is
  byte-identical — a single `context` field — so existing persisted
  observations still load; an out-of-range one now errors instead of panicking
  later). The **`Default` derive is removed**: it yielded `context: 0`, which is
  out of range at `C == 0` and was the only construction path that skipped
  validation. `context < C` is now an invariant no public API can break.

### `rlevo-core`

**Changed**

- `SnapshotBase` struct gains a `pub metadata: Option<SnapshotMetadata>` field
  and a `with_metadata` builder; `Snapshot::metadata()` is now overridden on
  `SnapshotBase` (ADR 0042).

### `rlevo-environments`

**Added**

- `GoToDoorObservation` (`[7, 7, 4]`), `GoToDoorSnapshot`, and the consts
  `MISSION_CHANNEL`, `GO_TO_DOOR_OBS_CHANNELS`, `DOOR_COUNT` — re-exported from
  `grids` alongside `GoToDoorEnv` (ADR 0043).
- `MemoryEnv::reset_with_seed`, `MemoryEnv::cue`, `MemoryEnv::size`, and
  `GoToDoorEnv::reset_with_seed`, `GoToDoorEnv::doors` — the accessors a
  scripted oracle or a replay needs now that both envs sample per episode.

**Fixed**

- **`MemoryEnv` now actually requires memory** (ADR 0043, #109). The cue was a
  compile-time constant (`Key(Yellow)`), the stored RNG was never read, and the
  reward was keyed to a coordinate independent of the cue — a reactive
  feedforward policy solved it outright. The cue type and the fork order are now
  drawn from a live persistent RNG each episode, all three objects are green so
  colour cannot leak the answer, `match_pos()` is derived from the sampled cue,
  and the layout is size-configurable on the canonical Minigrid geometry with a
  `Validate`-enforced `size >= 11` (Invariant M: with no occlusion in
  `egocentric_view`, only distance can hide the cue from the fork). A reactive
  policy is now capped at chance on the binary fork.
- **`GoToDoorEnv`'s mission now reaches the policy** (ADR 0043, #109). The
  instruction previously existed only on the env — `mission()` had zero callers
  workspace-wide — so simply sampling the target (what #109 asked for) would have
  made the task unsolvable at 25%. The mission colour is now broadcast into
  channel 3 of every observation cell, in the same ordinal encoding
  `Entity::color_u8` uses for perceived door colours, so a network can learn
  equality between the two. The four door colours are rejection-sampled distinct
  per episode and the fixed Red=North / Green=East / Blue=South / Yellow=West map
  is gone.
- Both envs re-seeded their RNG from `config.seed` inside `reset()`, violating
  ADR 0029 — harmless only because the RNG was never read, and a live landmine
  the moment sampling was added. Both re-seed lines are deleted; `reset()` draws
  from the persistent stream and `reset_with_seed` covers deterministic replay.
- **`ContextualBandit`'s ASCII render no longer transposes its arm and context
  counts** (#124). The `AsciiRenderable` impl was declared
  `impl<const K, const C> … for ContextualBandit<K, C>` while the struct is
  `ContextualBandit<C, K>`, so inside the impl `K` was bound to the context
  count and `C` to the arm count and the `"Contextual (K=…, C=…)"` label printed
  them backwards: `ContextualBandit<10, 4>` (10 contexts, 4 arms) rendered as
  `K=10, C=4`. Only the labels lied — the arm-mean lookup and `best@ctx` are
  indexed positionally and were always correct — and no test asserted on the
  label text, so the existing suite could not catch it. The impl now matches the
  struct's `<C, K>` order (as `Display` already did) and a regression test pins
  the labels to the const generics.

**Changed**

- `LocomotionSnapshot<O>` and `LunarLanderSnapshot` are now type aliases over
  `SnapshotBase` instead of hand-rolled `Snapshot` impls; construction moves
  metadata onto a `.with_metadata(...)` builder call (ADR 0042).
- `grids_solvable` integration tests for `MemoryEnv` and `GoToDoorEnv` are now
  **seed-driven oracles**: they read the sampled cue / mission back from the env
  after `reset` and derive the script, across a range of seeds. The old
  hard-coded scripts ("walk north to the Red door", "the match is on the top
  fork") encode answers that are now per-episode random — they would pass by luck.

### Infrastructure

**Added**

- CI: `rlevo-environments` feature-orthogonality check — `cargo check
  --no-default-features --features box2d` and `--features locomotion` run in
  isolation to catch a type gated behind only one of the two orthogonal
  features silently breaking the other (`.github/workflows/crate-tests.yml`).

---

## [0.2.0] – 2026-06-07

### Breaking changes

- **`DIM` → `RANK`** — the const generic parameter on `State<D>`, `Observation<D>`, `Action<D>`, and `Environment<D, SD, AD>` is renamed to `RANK` (or `R`, `SR`, `AR` at usage sites) across all crates. Update any downstream `impl State<D>` / `impl Environment<D, …>` declarations accordingly.
- **`fn new` removed from `Environment` trait** (ADR 0011) — construction is no longer part of the shared trait contract. Replace call sites with the new `ConstructableEnv` factory trait or a concrete `new` method.

### New crates

- **`rlevo-examples`** — heavy visualisation, recording, and report examples extracted from the `rlevo` umbrella (ADR 0012). Lightweight environment/algorithm examples stay in `crates/rlevo/`.
- **`rlevo-metrics-registry`** — wasm32-compatible leaf crate that owns the canonical metric descriptor list (`CANONICAL_METRICS`, `MetricDescriptor`, domain grouping). Eliminates the hand-copied duplicate that previously existed between `rlevo-benchmarks` and `rlevo-benchmarks-report-client` (ADR 0015).
- **`rlevo-benchmarks-report-client`** — Leptos/WASM static-HTML post-run report viewer. Served from an embedded `axum` server. Shares the metric registry with `rlevo-benchmarks` without pulling in `burn` or `rand`.

### Dependency upgrades

- **burn** `0.20.0` → `0.21.0`; migrated `ndarray` backend to the new `flex` backend.
- **rand** `0.9.x` → `0.10.1`, **rand_distr** → `0.6.0`.

### `rlevo-core`

**Added**

- `ConstructableEnv` factory trait — standalone `fn new(render: bool) -> Self` replacement for the removed `Environment::new` (ADR 0011).
- `StyledFrame`, `StyledLine`, `StyledSpan`, `SpanStyle`, `Color`, `Modifier`, semantic `palette` module, and `AsciiRenderable`/`AsciiRenderer` hoisted from `rlevo-environments::render` into `rlevo-core::render` (ADR 0009). Import paths inside `rlevo-environments` are preserved via a re-export shim.

**Changed**

- `AsciiRenderable` demoted from a required library invariant to an optional debug helper; implementing it is no longer implied by `Environment` (ADR 0013).

### `rlevo-environments`

**Changed**

- Render types (`StyledFrame`, `AsciiRenderable`, etc.) re-exported from `rlevo-core::render`; originals removed (ADR 0009).
- `Environment::new` removed; each environment exposes its own `new` constructor and may opt into `ConstructableEnv` (ADR 0011).

### `rlevo-evolution`

**Changed**

- All EA algorithms and shared ops (`selection`, `crossover`, `mutation`, `replacement`) now draw random values through `seed_stream` on the host CPU rather than calling `B::seed + Tensor::random`, eliminating the process-wide RNG mutex contention that caused non-determinism in parallel tests.
- `SharedPopulationObserver` unified to `parking_lot::Mutex` (was split between `std::sync` and `parking_lot` lock types, causing type mismatches in recording examples) (ADR 0010).

### `rlevo-reinforcement-learning`

**Added**

- `polyak_update` hoisted as a shared utility function available to all RL algorithm crates.

### `rlevo-benchmarks`

**Added**

- Record schema **v6** (`FORMAT_VERSION` bumped `5 → 6`, ADR 0014):
  - Expanded `CANONICAL_METRICS` (explained variance, per-iteration episode-return statistics, DQN/SAC loss terms) — list now owned by `rlevo-metrics-registry`.
  - Typed run-provenance fields on `RunManifest`: algorithm name, crate versions, git ref, device, seed count, success threshold.
  - `EpisodeKind { Training, Evaluation }` field in episode headers.
  - Episode wall-clock duration as a terminal metric.
  - `checkpoints: Vec<CheckpointRef>` seam for deep-RL Burn-`Recorder` model files (EA runs unaffected).
- Metrics-only live `ratatui` TUI replaces the earlier three-tier visualisation plan (ADR 0013 supersedes ADR 0008); no environment render panel in the TUI.

**Changed**

- `CANONICAL_METRICS` constant moved to `rlevo-metrics-registry`; `rlevo-benchmarks` re-exports it for back-compat.

### `rlevo-benchmarks-report-client`

**Added**

- Interactive post-run static HTML report (Leptos + WASM):
  - Min/max downsampling for long metric series (ADR 0013 / M8.2).
  - Multi-seed mean ± std band aggregation.
  - Hover crosshair with exact raw-value tooltip.
  - Per-panel SVG export buttons.
  - Step / episode / wall-clock x-axis toggle for episode panels.
  - Eval/training split via `EpisodeKind` in the episode index and table badge.
  - Landscape heatmap background for EA optimisation landscape runs.
  - Diversity-threshold guideline line with breach-pulse highlight.
  - Strip-plot overlay toggle on the population box-plot panel.

### `rlevo` (umbrella)

**Changed**

- Lightweight examples retained; heavy viz/record/report examples migrated to `rlevo-examples` (ADR 0012).

### Infrastructure

**Added**

- GitHub Actions CI: integration-test matrix (Linux × stable toolchain) and weekly full-workspace test run.
- `BACKEND_LOCK` per-binary synchronisation for wgpu-backed integration tests; removes the previous `--test-threads=1` requirement.

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

[0.2.0]: https://github.com/anthonytorlucci/rlevo/releases/tag/v0.2.0
[0.1.0]: https://github.com/anthonytorlucci/rlevo/releases/tag/v0.1.0
