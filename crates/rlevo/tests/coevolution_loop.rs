//! `CoEvolutionaryLoop` is driven by the `rlevo-harness`
//! evaluator without modification.
//!
//! A competitive predator–prey co-evolution is wrapped in a
//! [`CoEvolutionaryLoop`] and run through `Evaluator::run_trials` as a
//! `GenerationTrial`. One `GenerationProbe::advance` is one simultaneous-update
//! generation; the test asserts the suite completes the expected number of
//! steps per trial with no errors and a finite return.

#![allow(clippy::cast_precision_loss)]

use burn::tensor::{Device, Tensor, TensorData};

use rlevo_core::bounds::Bounds;
use rlevo_core::objective::ObjectiveSense;
use rlevo_core::rate::NonNegativeRate;
use rlevo_evolution::CoEvolutionaryLoop;
use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_evolution::coevolution::{CompetitiveCoEA, CompetitiveCoEAParams, CoupledFitness};
use rlevo_harness::evaluator::{Evaluator, EvaluatorConfig, GenerationTrial};
use rlevo_harness::reporter::logging::LoggingReporter;

const DIM: usize = 2;
const POP: usize = 32;
const MAX_GENS: usize = 40;

/// Predator–prey coupled fitness on a separable quadratic: the predator (pop 0)
/// minimizes squared distance to the prey (chase); the prey (pop 1) minimizes
/// closeness (flee). Both are minimization, bounded for the prey.
struct PredatorPrey;

fn rows(pop: &Tensor<2>) -> Vec<Vec<f32>> {
    let dims = pop.dims();
    let (n, d) = (dims[0], dims[1]);
    let flat = pop.clone().into_data().try_into_vec::<f32>().unwrap();
    (0..n).map(|i| flat[i * d..i * d + d].to_vec()).collect()
}

fn sqdist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

impl CoupledFitness for PredatorPrey {
    fn evaluate_coupled(&self, populations: &[Tensor<2>]) -> Vec<Tensor<1>> {
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
            Tensor::<1>::from_data(TensorData::new(predator.clone(), [predator.len()]), &device),
            Tensor::<1>::from_data(TensorData::new(prey.clone(), [prey.len()]), &device),
        ]
    }

    /// Both objectives are natural costs (see the struct doc), so the coupled
    /// fitness is a `Minimize`; the co-evolution algorithm canonicalises it.
    fn sense(&self) -> ObjectiveSense {
        ObjectiveSense::Minimize
    }
}

fn ga_config() -> GaConfig {
    GaConfig {
        pop_size: POP,
        genome_dim: DIM,
        bounds: Bounds::new(-5.0, 5.0),
        mutation_sigma: NonNegativeRate::new(0.2),
        selection: GaSelection::Tournament { size: 3 },
        crossover: GaCrossover::BlxAlpha {
            alpha: NonNegativeRate::new(0.5),
        },
        replacement: GaReplacement::Elitist { elitism_k: 1 },
    }
}

type Coea = CompetitiveCoEA<GeneticAlgorithm, GeneticAlgorithm, PredatorPrey>;

fn coea_factory(seed: u64) -> CoEvolutionaryLoop<Coea> {
    let device = Device::flex();
    let algo = CompetitiveCoEA::new(
        GeneticAlgorithm::new(),
        GeneticAlgorithm::new(),
        PredatorPrey,
    );
    let params = CompetitiveCoEAParams {
        params_a: ga_config(),
        params_b: ga_config(),
    };
    CoEvolutionaryLoop::new(algo, params, seed, device, MAX_GENS).expect("valid params")
}

#[test]
fn coevolutionary_loop_runs_through_evaluator() {
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

    let evaluator = Evaluator::new(cfg);
    let mut reporter = LoggingReporter::new();
    let report = evaluator.run_trials(
        "predator-prey-coea",
        &["predator-prey-2d".to_string()],
        |_key, env_seed, _agent_seed| GenerationTrial {
            probe: coea_factory(env_seed),
        },
        &mut reporter,
    );

    assert_eq!(report.trials.len(), 2, "expected one report per trial");
    for trial in &report.trials {
        assert!(!trial.errored, "trial errored: {:?}", trial.error_message);

        // A co-evolutionary run has no episode axis; the trial reports none.
        assert!(
            trial.episodes.is_empty(),
            "generation trial fabricates no episodes"
        );

        // One `advance` drives one simultaneous-update generation.
        assert_eq!(
            trial.scalars.get("generations").copied(),
            Some(MAX_GENS as f64),
            "one advance should drive one generation"
        );
        assert_eq!(
            trial.scalars.get("coea/generation").copied(),
            Some(MAX_GENS as f64),
            "the loop's counter agrees with the trial's own count"
        );

        // The binding fitness is the canonical value the loop reports.
        let binding = trial.scalars["coea/binding_fitness"];
        assert!(
            binding.is_finite(),
            "binding fitness should be finite, got {binding}"
        );
    }
}
