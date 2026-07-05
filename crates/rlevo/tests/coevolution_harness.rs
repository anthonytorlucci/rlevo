//! `CoEvolutionaryHarness` is driven by the `rlevo-benchmarks`
//! evaluator without modification.
//!
//! A competitive predator–prey co-evolution is wrapped in a
//! [`CoEvolutionaryHarness`] and run through `Evaluator::run_suite` exactly
//! like any other `BenchEnv`. One `BenchEnv::step` is one simultaneous-update
//! generation; the test asserts the suite completes the expected number of
//! steps per trial with no errors and a finite return.

#![allow(clippy::cast_precision_loss)]

use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData};
use rand::Rng;

use rlevo_benchmarks::agent::BenchableAgent;
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_benchmarks::reporter::logging::LoggingReporter;
use rlevo_benchmarks::suite::Suite;
use rlevo_core::bounds::Bounds;
use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_evolution::coevolution::{CompetitiveCoEA, CompetitiveCoEAParams, CoupledFitness};
use rlevo_evolution::CoEvolutionaryHarness;

type B = Flex;

const DIM: usize = 2;
const POP: usize = 32;
const MAX_GENS: usize = 40;

/// Predator–prey coupled fitness on a separable quadratic: the predator (pop 0)
/// minimizes squared distance to the prey (chase); the prey (pop 1) minimizes
/// closeness (flee). Both are minimization, bounded for the prey.
struct PredatorPrey;

fn rows(pop: &Tensor<B, 2>) -> Vec<Vec<f32>> {
    let dims = pop.dims();
    let (n, d) = (dims[0], dims[1]);
    let flat = pop.clone().into_data().into_vec::<f32>().unwrap();
    (0..n).map(|i| flat[i * d..i * d + d].to_vec()).collect()
}

fn sqdist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

impl CoupledFitness<B> for PredatorPrey {
    fn evaluate_coupled(&self, populations: &[Tensor<B, 2>]) -> Vec<Tensor<B, 1>> {
        debug_assert_eq!(populations.len(), 2);
        let device = populations[0].device();
        let a = rows(&populations[0]);
        let b = rows(&populations[1]);

        // Predator: mean squared distance to prey (minimize -> approach).
        let predator: Vec<f32> = a
            .iter()
            .map(|ai| b.iter().map(|bj| sqdist(ai, bj)).sum::<f32>() / b.len().max(1) as f32)
            .collect();
        // Prey: mean closeness to predators (minimize -> flee).
        let prey: Vec<f32> = b
            .iter()
            .map(|bj| {
                a.iter().map(|ai| (-sqdist(bj, ai)).exp()).sum::<f32>() / a.len().max(1) as f32
            })
            .collect();

        vec![
            Tensor::<B, 1>::from_data(TensorData::new(predator.clone(), [predator.len()]), &device),
            Tensor::<B, 1>::from_data(TensorData::new(prey.clone(), [prey.len()]), &device),
        ]
    }
}

fn ga_config() -> GaConfig {
    GaConfig {
        pop_size: POP,
        genome_dim: DIM,
        bounds: Bounds::new(-5.0, 5.0),
        mutation_sigma: 0.2,
        selection: GaSelection::Tournament { size: 3 },
        crossover: GaCrossover::BlxAlpha { alpha: 0.5 },
        replacement: GaReplacement::Elitist { elitism_k: 1 },
    }
}

type Coea = CompetitiveCoEA<B, GeneticAlgorithm<B>, GeneticAlgorithm<B>, PredatorPrey>;

fn coea_factory(seed: u64) -> CoEvolutionaryHarness<B, Coea> {
    let device = Default::default();
    let algo = CompetitiveCoEA::new(
        GeneticAlgorithm::<B>::new(),
        GeneticAlgorithm::<B>::new(),
        PredatorPrey,
    );
    let params = CompetitiveCoEAParams {
        params_a: ga_config(),
        params_b: ga_config(),
    };
    CoEvolutionaryHarness::new(algo, params, seed, device, MAX_GENS).expect("valid params")
}

/// Passive agent: the harness drives the optimization; the agent only steps it.
struct Passive;
impl BenchableAgent<(), ()> for Passive {
    fn act(&mut self, (): &(), _: &mut dyn Rng) {}
}

#[test]
fn coevolutionary_harness_runs_through_evaluator() {
    let cfg = EvaluatorConfig {
        num_episodes: 1,
        num_trials_per_env: 2,
        max_steps: MAX_GENS,
        base_seed: 17,
        // Flex seeds via a process-global mutex; pin to one thread for
        // reproducibility (mirrors the other run-suite integration tests).
        num_threads: Some(1),
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: None,
    };

    let suite: Suite<CoEvolutionaryHarness<B, Coea>> =
        Suite::new("predator-prey-coea", cfg.clone()).with_env("predator-prey-2d", coea_factory);

    let evaluator = Evaluator::new(cfg);
    let mut reporter = LoggingReporter::new();
    let report = evaluator.run_suite(&suite, |_s| Passive, &mut reporter);

    assert_eq!(report.trials.len(), 2, "expected one report per trial");
    for trial in &report.trials {
        assert!(!trial.errored, "trial errored: {:?}", trial.error_message);
        assert_eq!(trial.episodes.len(), 1, "one episode per trial");
        let ep = &trial.episodes[0];
        assert_eq!(
            ep.length, MAX_GENS,
            "one BenchEnv::step should drive one generation"
        );
        assert!(
            ep.return_value.is_finite(),
            "return should be finite, got {}",
            ep.return_value
        );
    }
}
