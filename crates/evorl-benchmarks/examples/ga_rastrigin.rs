//! Benchmark: hand-rolled GA on the Rastrigin landscape.
//!
//! Exercises two spec contracts in one binary:
//!
//! 1. `FitnessEvaluable` — implemented for a local wrapper around
//!    `evorl_envs::benchmarks::rastrigin::Rastrigin`. This is the
//!    "optimizer-on-landscape" trait from spec §4.3.
//! 2. `BenchEnv` — the GA is itself wrapped as a `BenchEnv` so the full
//!    `run_suite` path drives it, producing a `BenchmarkReport` with the
//!    same shape as RL trials.
//!
//! The "action" is a unit type (the agent is passive); each "step" runs
//! one GA generation and the reward is the negative best-fitness (so the
//! harness's return-maximization framing lines up with minimization).

use evorl_benchmarks::agent::{BenchableAgent, FitnessEvaluable};
use evorl_benchmarks::env::{BenchEnv, BenchStep};
use evorl_benchmarks::evaluator::{Evaluator, EvaluatorConfig};
use evorl_benchmarks::metrics::ea;
use evorl_benchmarks::metrics::Metric;
use evorl_benchmarks::reporter::logging::LoggingReporter;
use evorl_benchmarks::suite::Suite;
use evorl_envs::benchmarks::rastrigin::Rastrigin;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

// --- FitnessEvaluable wiring -------------------------------------------

struct Minimizer;

impl FitnessEvaluable for Minimizer {
    type Individual = Vec<f64>;
    type Landscape = Rastrigin;

    fn evaluate(&self, individual: &Self::Individual, landscape: &Self::Landscape) -> f64 {
        landscape.evaluate(individual)
    }
}

// --- GA state wrapped as a BenchEnv ------------------------------------

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
        let landscape = Rastrigin::new(dim);
        let (lo, hi) = landscape.bounds();
        let mut rng = StdRng::seed_from_u64(seed);
        let unit = Uniform::new(lo, hi).unwrap();
        let population: Vec<Vec<f64>> = (0..pop_size)
            .map(|_| (0..dim).map(|_| unit.sample(&mut rng)).collect())
            .collect();
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

impl BenchEnv for GaEnv {
    type Observation = ();
    type Action = ();

    fn reset(&mut self) -> Self::Observation {
        self.generation = 0;
    }

    fn step(&mut self, _action: Self::Action) -> BenchStep<Self::Observation> {
        let best = self.evolve();
        self.generation += 1;
        BenchStep {
            observation: (),
            reward: -best,
            done: self.generation >= self.max_generations,
        }
    }
}

// --- Passive agent -----------------------------------------------------

struct Passive;

impl BenchableAgent<(), ()> for Passive {
    fn act(&mut self, _obs: &(), _rng: &mut dyn Rng) {}

    fn emit_metrics(&self) -> Vec<Metric> {
        Vec::new()
    }
}

fn main() {
    tracing_subscriber::fmt().with_target(false).init();

    const DIM: usize = 10;
    const POP: usize = 64;
    const MAX_GENS: usize = 80;

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

    let suite: Suite<GaEnv> = Suite::new("rastrigin-ga", cfg.clone())
        .with_env("rastrigin-10d", |seed| GaEnv::new(seed, DIM, POP, MAX_GENS));

    let evaluator = Evaluator::new(cfg);
    let mut reporter = LoggingReporter::new();
    let report = evaluator.run_suite(&suite, |_s| Passive, &mut reporter);

    println!("=== {} ===", report.suite_name);
    for trial in &report.trials {
        let last_return = trial
            .episodes
            .last()
            .map(|e| -e.return_value)
            .unwrap_or(f64::NAN);
        let ea_metrics = ea::ea_metrics(Some(last_return), None, None);
        println!(
            "trial={} seed={:>20} best_fitness≈{:.4}  ea_metrics={}",
            trial.key.trial_idx,
            trial.trial_seed,
            last_return,
            ea_metrics.len()
        );
    }
}
