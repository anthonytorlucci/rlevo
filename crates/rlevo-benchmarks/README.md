# rlevo-benchmarks

Reproducible benchmarking harness for the `rlevo` workspace.

This crate provides a deterministic, parallelisable evaluation framework for testing reinforcement-learning and evolutionary-optimisation agents across one or more environments. A single `base_seed` fans out to per-trial seeds via splitmix64 so results are reproducible regardless of thread scheduling. Completed trials can be checkpointed and resumed without recomputation.

---

## Architecture

```
Suite<E>
  └── EnvFactory<E>  (Arc closure: seed → fresh env)
        │
        ▼
  Evaluator::run_suite
        │  (rayon parallel)
        ├── run_one_trial  ──► BenchEnv::reset / BenchEnv::step
        │         │
        │    BenchableAgent::act
        │         │
        │    EpisodeSummary (return, length)
        │         │
        │    TrialReport  ◄── core_metrics / ea_metrics / agent emit_metrics
        │
        ├── Checkpoint::save  (atomic JSON write)
        └── Reporter events  (LoggingReporter / JsonReporter / TuiReporter)
```

---

## Modules

### `env` — Environment Interface

`BenchEnv` is a narrow, object-safe environment trait intentionally lighter than `rlevo_core::Environment`. It avoids const-generic threading so that heterogeneous environments can be boxed and dispatched at runtime. The trait surface itself lives in `rlevo-core` (`rlevo_core::evaluation`, hoisted there per ADR 0004); this crate re-exports it under `rlevo_benchmarks::env` for convenience.

| Item | Description |
|------|-------------|
| `BenchEnv` | `reset() → Result<Obs, BenchError>`, `step(act) → Result<BenchStep<Obs>, BenchError>` |
| `BenchStep<Obs>` | Step result: `observation`, `reward: f64`, `done: bool` |
| `BenchError` | Recoverable error wrapping the upstream `EnvironmentError` |

### `agent` — Agent Interface

| Trait | Description |
|-------|-------------|
| `BenchableAgent<Obs, Act>` | Inference-only (frozen policy): `act(&mut self, obs, rng) → Act`; optional `emit_metrics()` hook |
| `FitnessEvaluable` | For optimizer-on-landscape scenarios; `evaluate(individual, landscape) → f64` |

### `evaluator` — Suite Runner

`Evaluator::run_suite` executes all trials in parallel via Rayon, catches per-trial panics, skips already-completed trials loaded from a checkpoint, and forwards events to any `Reporter`.

| Field | Default | Purpose |
|-------|---------|---------|
| `num_episodes` | — | Episodes per trial |
| `num_trials_per_env` | — | Trials per environment |
| `max_steps` | — | Step budget per episode |
| `base_seed` | — | Root for `SeedStream` |
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

### `seed` — Deterministic Seeding

`SeedStream` derives independent seeds for every trial, environment, and agent from a single `base_seed` using the splitmix64 finaliser. Because each derivation is a pure function of `(base_seed, env_idx, trial_idx)`, outputs are identical regardless of thread scheduling.

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
| `LoggingReporter` | always | Emits structured `tracing::info!` events |
| `JsonReporter` | `json` | Buffers events; writes a single JSON document atomically at suite end |
| `TuiReporter` | `tui` | Sends `TuiEvent` values over an mpsc channel for terminal-UI consumption |
| `RecordingReporter` | `record` | Drives the on-disk `EpisodeRecord` writer; pairs with `RecordingTap` + `RecordingLayer` |
| `MultiReporter` | always | Fans the event stream out to a `Vec<Box<dyn Reporter>>` in insertion order |

### `record` — On-Disk Per-Episode Recording (Milestone 4)

The `record` module emits per-episode files plus a run manifest under `runs/<run_id>/`:

```
runs/<run_id>/
  ├── episode_000000.rec   ← 16-byte preamble + EpisodeRecordHeader + length-prefixed RecordChunks
  ├── episode_000001.rec
  ├── ...
  └── run.toml             ← RunManifest (seed, env_family, hyperparameters, …)
```

Three producers share one `Arc<Mutex<dyn RecordSink>>`:

| Producer | Role |
|----------|------|
| `RecordingTap<E>` | Wraps `Environment`; emits per-step `FrameRecord` (subject to `frame_stride`) + closes the episode on `Snapshot::is_done` |
| `RecordingReporter` | Drives the suite lifecycle; finalises `run.toml` at `on_suite_end`. Two modes: with or without per-episode signals |
| `RecordingLayer` | `tracing_subscriber::Layer` that captures canonical metric fields (matches the `tui::log_layer` registry) |

Encoding: bincode 2.x with `bincode::config::standard()`. `FORMAT_VERSION` is stamped into every `EpisodeRecordHeader`; loaders refuse mismatched versions.

### `report` — Static-HTML Report Emitter (Milestone 5)

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
| `RecordedRun::open(dir)` | Loads `run.toml` + every `episode_*.rec`. Synthesises a manifest if missing; surfaces truncation as `OpenWarning`s. |
| `EpisodeIndex` | Per-episode summary: number, source path, frame count, episode reward, length, decoded frames + metric samples. |
| `emit_static_html(&run, &out, &cfg)` | Writes the single-file report atomically (tmp + fsync + rename). Returns episode count, bytes written, and a `size_warning` flag. |
| `export-report` (binary) | CLI front-end: `cargo run -p rlevo-benchmarks --features report --bin export-report -- <run-dir> <out.html>`. |

M5 ships the **data-transport skeleton**: per-family playback adapters and convergence plots land in subsequent milestones. The data contract — the four `<script>` block ids above — is keyed to the record `FORMAT_VERSION` (currently `6`, per ADR 0014); the loader rejects inlined payloads stamped with any other version.

### Optional Leptos/WASM client (Milestone 5.1)

The [`rlevo-benchmarks-report-client`](../rlevo-benchmarks-report-client/) sibling crate compiles to a Leptos/WASM client that decodes the inlined payloads and renders an interactive manifest header + episode table. Building it requires the `wasm32-unknown-unknown` rustup target and the `trunk` CLI; see the sibling crate's README for the build flow.

When the client artefacts are available, pass them through `EmitConfig`:

```rust
use rlevo_benchmarks::report::{ClientAssets, EmitConfig, RecordedRun, emit_static_html};

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

The emitter replaces the M5 placeholder body with a `<div id="rlevo-app">` mount point and inlines the WASM blob + JS shim + bundled CSS. When `client_assets` is `None`, the M5 placeholder body ships unchanged.

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
| `report` | no | `report` module: random-access loader + static-HTML emitter (`base64` + `serde_json`). Implies `record`. |

---

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
rlevo-benchmarks = { path = "../rlevo-benchmarks" }
```

### Minimal example — ε-greedy bandit

```rust
use rlevo_benchmarks::{
    agent::BenchableAgent,
    env::{BenchEnv, BenchError, BenchStep},
    evaluator::{Evaluator, EvaluatorConfig},
    reporter::logging::LoggingReporter,
    suite::Suite,
};

// 1. Wrap your environment as BenchEnv
struct MyEnv { /* ... */ }
impl BenchEnv for MyEnv {
    type Observation = f64;
    type Action = usize;

    fn reset(&mut self) -> Result<f64, BenchError> { Ok(0.0) }
    fn step(&mut self, action: usize) -> Result<BenchStep<f64>, BenchError> {
        Ok(BenchStep { observation: 0.0, reward: 1.0, done: true })
    }
}

// 2. Wrap your agent as BenchableAgent
struct MyAgent;
impl BenchableAgent<f64, usize> for MyAgent {
    fn act(&mut self, _obs: &f64, _rng: &mut dyn rand::RngCore) -> usize { 0 }
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
let report = Evaluator::new(cfg)
    .run_suite::<_, MyAgent, _, _>(
        &suite,
        |_seed| MyAgent,
        &mut reporter,
    );
```

---

## Running Examples

```bash
# ε-greedy sample-average agent on the 10-armed bandit
# (examples live in the rlevo-examples crate per ADR 0012)
cargo run -p rlevo-examples --example tabular_bandit

# Hand-rolled genetic algorithm minimising the 10-D Rastrigin function
cargo run -p rlevo-examples --example ga_rastrigin
```

---

## Academic References

The following works directly informed the design and algorithms in this crate:

- Vigna, S. (2015). **Further scramblings of Marsaglia's xorshift generators**. *Journal of Computational and Applied Mathematics*, 315, 175–181. — basis for the `splitmix64` finaliser used in `SeedStream`. [arXiv:1402.6246](https://arxiv.org/abs/1402.6246)

- Steele, G. L., Lea, D., & Flood, C. H. (2014). **Fast splittable pseudorandom number generators**. *ACM SIGPLAN OOPSLA*, 453–472. — introduces the SplitMix family; the specific constants in `SeedStream` derive from this work.

- Rastrigin, L. A. (1974). *Systems of extreme control*. Nauka, Moscow. — original formulation of the Rastrigin function \(f(x)=A n + \sum x_i^2 - A \cos (2 \pi x_i ))\) used in the `ga_rastrigin` example and `rlevo-environments::landscapes::rastrigin`.

- Sutton, R. S., & Barto, A. G. (2018). **Reinforcement Learning: An Introduction** (2nd ed.). MIT Press. — sample-average and ε-greedy methods demonstrated in `tabular_bandit` follow Chapter 2 of this text.

---

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
