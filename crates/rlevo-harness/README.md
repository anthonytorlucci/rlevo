# rlevo-harness

![Alt Text](https://raw.githubusercontent.com/anthonytorlucci/rlevo/main/rlevo-logo.png)

Reproducible evaluation harness for the `rlevo` workspace, reached from the
umbrella as `rlevo::harness`. Through 0.5.0 this crate was named `rlevo-benchmarks`.

This crate provides a deterministic, parallelisable evaluation framework for testing reinforcement-learning and evolutionary-optimisation agents across one or more environments. A single `base_seed` fans out to per-trial seeds via splitmix64 so results are reproducible regardless of thread scheduling. Completed trials can be checkpointed and resumed without recomputation.

---

## Architecture

```
Suite<E>
  └── EnvFactory<E>  (Arc closure: seed → fresh env)
        │
        ▼
  Evaluator::run_suite  ──►  Evaluator::run_trials
        │  (rayon parallel)
        ├── Trial::run
        │     ├── EpisodicTrial   ──► Environment::reset / Environment::step
        │     └── GenerationTrial ──► GenerationProbe::begin / ::advance
        │         │
        │    BenchableAgent::act   (episodic path only)
        │         │
        │    EpisodeSummary (return, length)
        │         │
        │    TrialReport  ◄── core_metrics + agent emit_metrics     (episodic)
        │                 ◄── generations + wall clock
        │                     + probe MetricsProvider::emit         (generation)
        │
        ├── checkpoint::save  (atomic JSON write)
        └── Reporter events  (LoggingReporter / JsonReporter / TuiReporter)
```

---

## Modules

### Environments and trials

`Evaluator::run_suite` binds directly on `rlevo_core::environment::Environment`, inferring its rank parameters from the `Suite<E>` it is given, so environments are registered as themselves with no adapter (ADR 0077). Evolutionary generation loops, which are not `Environment`s, go through `GenerationProbe` and `GenerationTrial` instead. This crate has no environment module of its own; the trial types live in `evaluator`.

| Item | Description |
|------|-------------|
| `Trial` | `run(self, cfg, info, reporter) → TrialReport` — the seam that lets one machinery drive several trial shapes |
| `EpisodicTrial<E, A, D, SD, AD>` | Rolls an `Environment` out against a `BenchableAgent` for `num_episodes` episodes |
| `GenerationTrial<P>` | Drives a `GenerationProbe` to its budget; reports no episodes |

### Agent and fitness traits

The traits a trial drives are part of the contract, so they live in `rlevo-core`, in `rlevo_core::fitness` (`rlevo::core::fitness` through the umbrella). This crate has no module of its own for them.

| Trait | Description |
|-------|-------------|
| `BenchableAgent<Obs, Act>` | Inference-only (frozen policy): `act(&mut self, obs, rng) → Act`; optional `emit_metrics()` hook |
| `Landscape` | For optimizer-on-landscape scenarios; `evaluate(x) → f64`, the landscape being the fitness function |

### `evaluator` — Suite Runner

`Evaluator::run_suite` executes all trials in parallel via Rayon, catches per-trial panics, skips already-completed trials loaded from a checkpoint, and forwards events to any `Reporter`.

| Field | Default | Purpose |
|-------|---------|---------|
| `num_episodes` | `100` | Episodes per trial |
| `num_trials_per_env` | `1` | Trials per environment |
| `max_steps` | `1000` | Step ceiling per episode; for a `GenerationTrial`, a cap on generations that warns when it ends a trial early |
| `base_seed` | `0` | The suite's seed: the root of `SeedStream`, from which every trial's seeds are derived. Not the seed of any one run |
| `num_threads` | `None` (Rayon default) | Thread pool size |
| `checkpoint_dir` | `None` | Directory for checkpoint files |
| `fail_fast` | `false` | Abort suite on first panic |
| `success_threshold` | `None` | Enables `success_rate` metric |

### `suite` — Suite Definition

| Item | Description |
|------|-------------|
| `Suite<E>` | Named collection of `EnvFactory<E>` entries with a default `EvaluatorConfig` |
| `EnvFactory<E>` | `Arc`-wrapped `Fn(u64) -> E` — constructs a fresh env from a seed |
| `TrialKey` | `(env_idx, trial_idx)` — uniquely identifies a trial |

### Deterministic seeding

`SeedStream`, in `rlevo_core::util::seed`, derives independent seeds for every trial, environment, and agent from a single `base_seed` using the splitmix64 finaliser. Because each derivation is a pure function of `(base_seed, env_idx, trial_idx)`, outputs are identical regardless of thread scheduling.

| Method | Returns |
|--------|---------|
| `trial_seed(env_idx, trial_idx)` | Unique u64 per trial |
| `env_seed(trial_seed)` | Environment-specific seed |
| `agent_seed(trial_seed)` | Agent-specific seed |

### `metrics` — Metric Collection

Three metric variants are represented by the `Metric` enum (`Scalar`, `Histogram`, `Counter`). Agents report method-specific metrics by implementing `MetricsProvider` and returning values from `BenchableAgent::emit_metrics`.

| Sub-module | Contents |
|------------|----------|
| `metrics::core` | `return/{mean,median,std,min,max}`, `episode_length/mean`, `throughput/steps_per_sec`, `wall_clock_seconds`, `success_rate` |
| `metrics::ea` | `ea/population_diversity`, `ea/best_fitness`, `ea/generations_to_converge`, `ea/fitness_variance` |
| `metrics::rl` | Metric name constants for RL agents: `rl/policy_loss`, `rl/value_loss`, `rl/approx_kl`, `rl/entropy`, `rl/epsilon`, `rl/learning_rate` |

### `metrics_registry` — Canonical Metric Names

Always-on re-export of the `rlevo-metrics-registry` leaf crate (ADR 0015). Supplies the single source of truth for canonical metric field names — `CANONICAL_METRICS`, `is_canonical_metric`, `MetricDescriptor`/`MetricKind`, `Cadence`, `descriptor`, `is_per_generation`, `title_for` — so the recorder (`RecordingLayer`) and live TUI (`tui::log_layer`) agree on which `tracing` fields are promoted to metrics.

### `report` — Result Types

| Item | Description |
|------|-------------|
| `EpisodeSummary` | Per-episode `(episode_idx, return_value, length)` |
| `TrialReport` | Aggregated per-trial result with scalar/histogram/counter BTreeMaps and error state |
| `BenchmarkReport` | Suite-level result; `finalize()` marks the run complete |

### `reporter` — Event Sink

The `Reporter` trait receives lifecycle events (`on_suite_start`, `on_trial_start`, `on_episode_end`, `on_trial_end`, `on_suite_end`) with no metric-computation responsibility. Five built-in implementations are provided:

| Reporter | Feature gate | Behaviour |
|----------|-------------|-----------|
| `LoggingReporter` | always | Emits structured `tracing::info!` events under the target `rlevo_harness` |
| `JsonReporter` | `json` | Buffers events; writes a single JSON document atomically at suite end |
| `TuiReporter` | `tui` | Sends `TuiEvent` values over an mpsc channel for terminal-UI consumption |
| `RecordingReporter` | `record` | Drives the on-disk `EpisodeRecord` writer; pairs with `RecordingTap` + `RecordingLayer` |
| `MultiReporter` | always | Fans the event stream out to a `Vec<Box<dyn Reporter>>` in insertion order |

### `record` — On-Disk Per-Episode Recording

The `record` module emits per-episode files plus a run manifest under `runs/<run_id>/`:

```
runs/<run_id>/
  ├── episode_000000.rec   ← 16-byte preamble + EpisodeRecordHeader + length-prefixed RecordChunks
  ├── episode_000001.rec
  ├── ...
  └── run.toml             ← RunManifest (seed, env_family, hyperparameters, …)
```

Four producers share one `Arc<Mutex<dyn RecordSink>>`:

| Producer | Role |
|----------|------|
| `RecordingTap<E>` | Wraps `Environment`; emits per-step `FrameRecord` (subject to `frame_stride`) + closes the episode when the snapshot's status is done |
| `RecordingReporter` | Drives the suite lifecycle; finalises `run.toml` at `on_suite_end`. Two modes: with or without per-episode signals |
| `PopulationReporter` | `PopulationObserver` attached with `EvolutionaryLoop::with_observer`; writes one `PopulationSample` (fitnesses, diversity, best) per generation |
| `RecordingLayer` | `tracing_subscriber::Layer` that captures canonical metric fields (matches the `tui::log_layer` registry) |

Encoding: bincode 2.x with `bincode::config::standard()`. `FORMAT_VERSION` is stamped into every `EpisodeRecordHeader`; loaders refuse mismatched versions.

### `report` — Static-HTML Report Emitter

The `report` module loads a recording emitted by `record` and serialises it into a single self-contained `index.html`:

```
runs/<run_id>/index.html
  ├── <style>...</style>                                ← inlined CSS
  ├── placeholder body (manifest header + episode table)
  ├── <script id="rlevo-manifest">{...}</script>        ← JSON manifest
  ├── <script id="rlevo-warnings">[...]</script>        ← non-fatal load warnings
  ├── <script id="rlevo-episode-NNNNNN">BASE64</script> ← raw .rec bytes per episode
  └── <script id="rlevo-episode-index">[...]</script>   ← episode summary metadata
```

| Item | Role |
|------|------|
| `RecordedRun::open(dir)` | Loads `run.toml` + every `episode_*.rec`. Synthesises a manifest if missing. Its `OpenWarning`s flag a synthesised manifest and an episode count that disagrees with the manifest; a truncated episode file loads its whole frames with no warning. |
| `EpisodeIndex` | Per-episode summary: number, source path, frame count, episode reward, length, metric samples. Frames are not kept after load; `RecordedRun::frame_at_or_before` re-reads them from the source `.rec` file. |
| `emit_static_html(&run, &out, &cfg)` | Writes the single-file report atomically (tmp + fsync + rename). Returns episode count, bytes written, and a `size_warning` flag. |
| `export-report` (binary) | CLI front-end: `cargo run -p rlevo-harness --features report --bin export-report -- <run-dir> <out.html>`. |

Without client assets the emitter writes the **data payloads** under a static placeholder body (manifest header and episode table); playback and charts come from the report client below, which reads those payloads. The data contract — the four `<script>` block ids above — is keyed to the record `FORMAT_VERSION` (currently `7`, per ADR 0023); the loader rejects inlined payloads stamped with any other version.

### Optional Leptos/WASM client

The [`rlevo-benchmarks-report-client`](../rlevo-benchmarks-report-client/) sibling crate compiles to a Leptos/WASM client that decodes the inlined payloads and renders the manifest, load warnings and episode table, convergence and population charts, and per-episode playback (a scrubber with play/pause and speed, through a per-family adapter). Building it requires the `wasm32-unknown-unknown` rustup target and the `trunk` CLI; see the sibling crate's README for the build flow.

When the client artefacts are available, pass them through `EmitConfig`:

```rust
use rlevo_harness::report::{ClientAssets, EmitConfig, RecordedRun, emit_static_html};

let run = RecordedRun::open("runs/<run_id>")?;
let assets = ClientAssets::from_trunk_dist(
    std::path::Path::new("crates/rlevo-benchmarks-report-client/dist")
)?;
emit_static_html(
    &run,
    std::path::Path::new("runs/<run_id>/index.html"),
    &EmitConfig {
        client_assets: Some(assets),
        ..EmitConfig::default()
    },
)?;
```

The emitter replaces the placeholder body with a `<div id="rlevo-app">` mount point and inlines the WASM blob + JS shim + bundled CSS. When `client_assets` is `None`, the placeholder body ships unchanged.

### `env_wrappers` — Live-TUI Env Taps (feature `tui`)

Composable `Environment` wrappers that feed the metrics-only live TUI (ADR 0013). `TuiEnvTap` wraps an env and emits per-episode returns into the TUI's metric stream without the dashboard ever owning an environment panel. Gated behind the `tui` feature.

### `checkpoint` — Resume Support

When `checkpoint_dir` is set and the `json` feature is enabled, `Evaluator` saves an atomic checkpoint after each trial and skips already-completed `TrialKey`s on resume. Without the `json` feature the checkpoint functions are zero-cost stubs.

---

## Cargo Features

| Feature | Default | Enables |
|---------|---------|---------|
| `json` | yes | `JsonReporter`, checkpoint load/save (`serde` + `serde_json`) |
| `tui` | no | `TuiReporter`, `TuiEvent` (`ratatui` + `crossterm`), `tracing_subscriber` integration |
| `record` | no | `record` module: per-episode files + `RunManifest` (`bincode` + `toml` + `time`) |
| `report` | no | Random-access loader + static-HTML emitter in the `report` module (`base64` + `serde_json`). Implies `record`. |

---

## Usage

The umbrella crate carries the harness and re-exports it as
`rlevo::harness` (ADR 0084), so a project that depends on `rlevo` needs
no entry for this crate and writes `rlevo::harness::evaluator::Evaluator`.
The re-export is on `main` and ships with the next release.

To use the harness without the rest of `rlevo`, add to your `Cargo.toml`:

```toml
[dependencies]
rlevo-harness = { path = "../rlevo-harness" }
```

### Moving from `rlevo-benchmarks`

Through 0.5.0 this crate was published as `rlevo-benchmarks` (ADR 0088). A project that depends on it directly changes three things: the dependency line, every feature that names the crate (`rlevo-benchmarks/tui`, `rlevo-benchmarks/record`, `dep:rlevo-benchmarks`), and the first segment of its paths, from `rlevo_benchmarks::` to `rlevo_harness::`. Cargo's `package` key keeps the old local name, so the paths and feature entries go on working until they are changed, except the paths through `agent` and `seed`: those two modules were aliases for traits and a type that moved to `rlevo-core` under ADR 0004, and the rename removed them (ADR 0088). Import `BenchableAgent` and `Landscape` from `rlevo_core::fitness` (an evaluator written against `FitnessEvaluable` implements `Landscape` instead; ADR 0091), and `SeedStream` from `rlevo_core::util::seed`.

```toml
[dependencies]
rlevo-benchmarks = { package = "rlevo-harness", path = "../rlevo-harness" }
```

The harness's `tracing` events carry the target `rlevo_harness` where they carried `rlevo_benchmarks`. A log filter on the old target matches nothing and no build fails, so change it with the rest.

### Minimal example — $\epsilon$-greedy bandit

```rust
use rlevo_harness::{
    evaluator::{Evaluator, EvaluatorConfig},
    reporter::logging::LoggingReporter,
    suite::Suite,
};
use rlevo_core::environment::Environment;
use rlevo_core::fitness::BenchableAgent;

// 1. Implement `rlevo_core::environment::Environment` for your env. The
//    evaluator infers the rank parameters
//    from the `Suite<MyEnv>` you hand it — no adapter, no turbofish.
struct MyEnv { /* ... */ }
impl Environment<1, 1> for MyEnv {
    type ObservationType = MyObs;
    type ActionType = MyAction;

    fn reset(&mut self) -> Result<Snapshot<MyObs>, EnvironmentError> {
        Ok(Snapshot::running(MyObs, ScalarReward(0.0)))
    }
    fn step(&mut self, action: MyAction) -> Result<Snapshot<MyObs>, EnvironmentError> {
        Ok(Snapshot::terminated(MyObs, ScalarReward(1.0)))
    }
}

// 2. Wrap your agent as BenchableAgent
struct MyAgent;
impl BenchableAgent<MyObs, MyAction> for MyAgent {
    fn act(&mut self, _obs: &MyObs, _rng: &mut dyn rand::Rng) -> MyAction { MyAction }
}

// 3. Build and run
let cfg = EvaluatorConfig {
    num_episodes: 10,
    num_trials_per_env: 3,
    max_steps: 500,
    base_seed: 42,
    ..Default::default()
};
let suite: Suite<MyEnv> = Suite::new("my-suite", cfg.clone())
    .with_env("env-a", |_seed| MyEnv { /* ... */ });

let mut reporter = LoggingReporter::default();
let report = Evaluator::new(cfg).run_suite(&suite, |_seed| MyAgent, &mut reporter);
```

---

## Running Examples

```bash
# ε-greedy sample-average agent on the 10-armed bandit
# (the harness examples live in the umbrella crate, ADR 0084)
cargo run -p rlevo --example tabular_bandit

# Hand-rolled genetic algorithm minimising the 10-D Rastrigin function
cargo run -p rlevo --example ga_rastrigin
```

---

## Academic References

The following works directly informed the design and algorithms in this crate:

- Vigna, S. (2015). **Further scramblings of Marsaglia's xorshift generators**. *Journal of Computational and Applied Mathematics*, 315, 175–181. — basis for the `splitmix64` finaliser used in `SeedStream`. [arXiv:1402.6246](https://arxiv.org/abs/1402.6246)

- Steele, G. L., Lea, D., & Flood, C. H. (2014). **Fast splittable pseudorandom number generators**. *ACM SIGPLAN OOPSLA*, 453–472. — introduces the SplitMix family; the specific constants in `SeedStream` derive from this work.

- Rastrigin, L. A. (1974). *Systems of extreme control*. Nauka, Moscow. — original formulation of the Rastrigin function \(f(x)=A n + \sum x_i^2 - A \cos (2 \pi x_i ))\) used in the `ga_rastrigin` example and `rlevo-environments::landscapes::rastrigin`.

- Sutton, R. S., & Barto, A. G. (2018). **Reinforcement Learning: An Introduction** (2nd ed.). MIT Press. — sample-average and $\epsilon$-greedy methods demonstrated in `tabular_bandit` follow Chapter 2 of this text.

---

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or [MIT License](../../LICENSE-MIT) at your option.
