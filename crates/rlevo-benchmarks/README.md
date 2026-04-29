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
        │    EpisodeRecord (return, length)
        │         │
        │    TrialReport  ◄── core_metrics / ea_metrics / agent emit_metrics
        │
        ├── Checkpoint::save  (atomic JSON write)
        └── Reporter events  (LoggingReporter / JsonReporter / TuiReporter)
```

---

## Modules

### `env` — Environment Interface

`BenchEnv` is a narrow, object-safe environment trait intentionally lighter than `rlevo_core::Environment`. It avoids const-generic threading so that heterogeneous environments can be boxed and dispatched at runtime.

| Item | Description |
|------|-------------|
| `BenchEnv` | `reset() → Obs`, `step(act) → BenchStep<Obs>` |
| `BenchStep<Obs>` | Step result: `observation`, `reward: f64`, `done: bool` |

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

### `report` — Result Types

| Item | Description |
|------|-------------|
| `EpisodeRecord` | Per-episode `(episode_idx, return_value, length)` |
| `TrialReport` | Aggregated per-trial result with scalar/histogram/counter BTreeMaps and error state |
| `BenchmarkReport` | Suite-level result; `finalize()` marks the run complete |

### `reporter` — Event Sink

The `Reporter` trait receives lifecycle events (`on_suite_start`, `on_trial_start`, `on_episode_end`, `on_trial_end`, `on_suite_end`) with no metric-computation responsibility. Three built-in implementations are provided:

| Reporter | Feature gate | Behaviour |
|----------|-------------|-----------|
| `LoggingReporter` | always | Emits structured `tracing::info!` events |
| `JsonReporter` | `json` | Buffers events; writes a single JSON document atomically at suite end |
| `TuiReporter` | `tui` | Sends `TuiEvent` values over an mpsc channel for terminal-UI consumption |

### `checkpoint` — Resume Support

When `checkpoint_dir` is set and the `json` feature is enabled, `Evaluator` saves an atomic checkpoint after each trial and skips already-completed `TrialKey`s on resume. Without the `json` feature the checkpoint functions are zero-cost stubs.

---

## Cargo Features

| Feature | Default | Enables |
|---------|---------|---------|
| `json` | yes | `JsonReporter`, checkpoint load/save (`serde` + `serde_json`) |
| `tui` | no | `TuiReporter`, `TuiEvent` (`ratatui` + `crossterm`) |

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
    env::{BenchEnv, BenchStep},
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

let report = Evaluator::new(cfg)
    .run_suite::<_, MyAgent, _, _>(
        &suite,
        |_seed| MyAgent,
        LoggingReporter::default(),
    );
```

---

## Running Examples

```bash
# ε-greedy sample-average agent on the 10-armed bandit
cargo run -p rlevo-benchmarks --example tabular_bandit

# Hand-rolled genetic algorithm minimising the 10-D Rastrigin function
cargo run -p rlevo-benchmarks --example ga_rastrigin
```

---

## Academic References

The following works directly informed the design and algorithms in this crate:

- Vigna, S. (2015). **Further scramblings of Marsaglia's xorshift generators**. *Journal of Computational and Applied Mathematics*, 315, 175–181. — basis for the `splitmix64` finaliser used in `SeedStream`. [arXiv:1402.6246](https://arxiv.org/abs/1402.6246)

- Steele, G. L., Lea, D., & Flood, C. H. (2014). **Fast splittable pseudorandom number generators**. *ACM SIGPLAN OOPSLA*, 453–472. — introduces the SplitMix family; the specific constants in `SeedStream` derive from this work.

- Rastrigin, L. A. (1974). *Systems of extreme control*. Nauka, Moscow. — original formulation of the Rastrigin function `f(x) = A·n + Σ(xᵢ² − A·cos(2πxᵢ))` used in the `ga_rastrigin` example and `rlevo-environments::landscapes::rastrigin`.

- Sutton, R. S., & Barto, A. G. (2018). **Reinforcement Learning: An Introduction** (2nd ed.). MIT Press. — sample-average and ε-greedy methods demonstrated in `tabular_bandit` follow Chapter 2 of this text.

---

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
