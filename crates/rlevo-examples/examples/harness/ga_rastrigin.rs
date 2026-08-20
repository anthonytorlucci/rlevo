//! Benchmark: hand-rolled GA on the Rastrigin landscape.
//!
//! Exercises two trait contracts in one binary:
//!
//! 1. `FitnessEvaluable` — the "optimizer-on-landscape" trait, implemented
//!    for a local wrapper around
//!    `rlevo_environments::landscapes::rastrigin::Rastrigin`.
//! 2. `GenerationProbe` — the GA implements the drive seam directly, so
//!    `Evaluator::run_trials` drives it as a `GenerationTrial` and produces a
//!    `BenchmarkReport` alongside RL trials.
//!
//! There is no agent and no episode axis: each `advance` runs one GA
//! generation and reports a typed `GaGenerationMetrics`, so the costs stay in
//! their natural minimise sense rather than being negated into a scalar
//! reward. This also shows a type outside `rlevo-evolution` implementing the
//! seam — nothing here is an `EvolutionaryHarness`.
//!
//! # Running
//!
//! ```text
//! cargo run -p rlevo-examples --example ga_rastrigin
//! ```
//!
//! No feature flags are required.
//!
//! # Output
//!
//! Structured log lines from `LoggingReporter` followed by a per-trial
//! summary printed to stdout:
//!
//! ```text
//! === rastrigin-ga ===
//! trial=0 seed=                   7 best_fitness≈0.9712  ea_metrics=1
//! trial=1 seed=  ...              8 best_fitness≈1.0431  ea_metrics=1
//! trial=2 seed=  ...              9 best_fitness≈0.8803  ea_metrics=1
//! ```
//!
//! `best_fitness` is the best (lowest) Rastrigin value reached after
//! `MAX_GENS` generations. The value is extracted by negating the
//! episode return, reversing the sign convention used internally.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use rlevo_benchmarks::agent::FitnessEvaluable;
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig, GenerationTrial};
use rlevo_benchmarks::metrics::Metric;
use rlevo_benchmarks::metrics::ea;
use rlevo_benchmarks::reporter::logging::LoggingReporter;
use rlevo_core::evaluation::GenerationProbe;
use rlevo_core::fitness::MetricsProvider;
use rlevo_environments::landscapes::rastrigin::Rastrigin;

// --- FitnessEvaluable wiring -------------------------------------------

struct Minimizer;

impl FitnessEvaluable for Minimizer {
    type Individual = Vec<f64>;
    type Landscape = Rastrigin;

    fn evaluate(&self, individual: &Self::Individual, landscape: &Self::Landscape) -> f64 {
        landscape.evaluate(individual)
    }
}

// --- GA state driven as a GenerationProbe ------------------------------

struct GaEnv {
    landscape: Rastrigin,
    population: Vec<Vec<f64>>,
    rng: StdRng,
    max_generations: usize,
    generation: usize,
    best_so_far: f64,
    sigma: f64,
}

impl GaEnv {
    fn new(seed: u64, dim: usize, pop_size: usize, max_generations: usize) -> Self {
        let landscape = Rastrigin::new(dim).expect("dim >= 1");
        let mut rng = StdRng::seed_from_u64(seed);
        let population = sample_population(&mut rng, &landscape, pop_size);
        Self {
            landscape,
            population,
            rng,
            max_generations,
            generation: 0,
            best_so_far: f64::INFINITY,
            sigma: 0.5,
        }
    }

    /// Tournament selection + Gaussian mutation. One-generation step.
    fn evolve(&mut self) -> f64 {
        let evaluator = Minimizer;
        let fitnesses: Vec<f64> = self
            .population
            .iter()
            .map(|x| evaluator.evaluate(x, &self.landscape))
            .collect();

        let pop_size = self.population.len();
        let pop_dist = Uniform::new(0_usize, pop_size).unwrap();
        let normal = Normal::new(0.0_f64, 1.0).unwrap();
        let (lo, hi) = self.landscape.bounds();

        let mut next: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let a = pop_dist.sample(&mut self.rng);
            let b = pop_dist.sample(&mut self.rng);
            let winner = if fitnesses[a] < fitnesses[b] { a } else { b };
            let mut child = self.population[winner].clone();
            for gene in &mut child {
                let delta: f64 = normal.sample(&mut self.rng);
                *gene = (*gene + self.sigma * delta).clamp(lo, hi);
            }
            next.push(child);
        }
        self.population = next;

        let best: f64 = fitnesses.iter().copied().fold(f64::INFINITY, f64::min);
        if best < self.best_so_far {
            self.best_so_far = best;
        }
        best
    }
}

fn sample_population(rng: &mut StdRng, landscape: &Rastrigin, pop_size: usize) -> Vec<Vec<f64>> {
    let (lo, hi) = landscape.bounds();
    let unit = Uniform::new(lo, hi).unwrap();
    (0..pop_size)
        .map(|_| (0..landscape.dim()).map(|_| unit.sample(rng)).collect())
        .collect()
}

/// Per-generation summary this example's hand-rolled GA reports.
///
/// A `GenerationProbe` names its own metrics type, so an implementor is not
/// limited to the single `f64` a `BenchStep` reward could carry.
struct GaGenerationMetrics {
    generation: usize,
    best_this_generation: f64,
    best_so_far: f64,
}

impl MetricsProvider for GaGenerationMetrics {
    #[allow(clippy::cast_precision_loss)]
    fn emit(&self) -> Vec<Metric> {
        vec![
            Metric::Scalar {
                name: "ga/generation".to_string(),
                value: self.generation as f64,
            },
            Metric::Scalar {
                name: "ga/best_this_generation".to_string(),
                value: self.best_this_generation,
            },
            Metric::Scalar {
                name: "ga/best_so_far".to_string(),
                value: self.best_so_far,
            },
        ]
    }
}

impl GenerationProbe for GaEnv {
    type Metrics = GaGenerationMetrics;

    fn begin(&mut self) {
        let pop_size = self.population.len();
        self.population = sample_population(&mut self.rng, &self.landscape, pop_size);
        self.best_so_far = f64::INFINITY;
        self.generation = 0;
    }

    fn advance(&mut self) -> Option<Self::Metrics> {
        if self.generation >= self.max_generations {
            return None;
        }
        let best = self.evolve();
        self.generation += 1;
        Some(GaGenerationMetrics {
            generation: self.generation,
            best_this_generation: best,
            best_so_far: self.best_so_far,
        })
    }
}

fn main() {
    const DIM: usize = 10;
    const POP: usize = 64;
    const MAX_GENS: usize = 80;

    tracing_subscriber::fmt().with_target(false).init();

    let cfg = EvaluatorConfig {
        num_episodes: 1,
        num_trials_per_env: 3,
        max_steps: MAX_GENS,
        base_seed: 7,
        num_threads: None,
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: None,
    };

    let evaluator = Evaluator::new(cfg);
    let mut reporter = LoggingReporter::new();
    let report = evaluator.run_trials(
        "rastrigin-ga",
        &["rastrigin-10d".to_string()],
        |_key, env_seed, _agent_seed| GenerationTrial {
            probe: GaEnv::new(env_seed, DIM, POP, MAX_GENS),
        },
        &mut reporter,
    );

    println!("=== {} ===", report.suite_name);
    for trial in &report.trials {
        // Read the best cost straight off the trial's scalars. The previous
        // `BenchEnv` version had to negate `episodes.last()`'s return, since a
        // `BenchStep` reward is maximise-space while this landscape is a cost.
        let best = trial
            .scalars
            .get("ga/best_so_far")
            .copied()
            .unwrap_or(f64::NAN);
        let ea_metrics = ea::ea_metrics(Some(best), None, None);
        println!(
            "trial={} seed={:>20} best_fitness≈{:.4}  ea_metrics={}",
            trial.key.trial_idx,
            trial.trial_seed,
            best,
            ea_metrics.len()
        );
    }
}
