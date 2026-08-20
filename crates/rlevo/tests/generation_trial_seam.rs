//! Cross-crate proof that the `Trial` seam is shape-agnostic.
//!
//! Drives an `EvolutionaryHarness` through `Evaluator::run_trials` as a
//! `GenerationTrial` — no `BenchEnv`, no `BenchableAgent`, no episode axis —
//! and asserts it gets the same machinery the episodic path does: rayon
//! fan-out, per-trial seeding, checkpoint-shaped `TrialReport`s, and the
//! reporter lifecycle.

use burn::backend::Flex;
use burn::tensor::Tensor;
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig, GenerationTrial};
use rlevo_benchmarks::reporter::logging::LoggingReporter;
use rlevo_core::bounds::Bounds;
use rlevo_core::objective::ObjectiveSense;
use rlevo_core::rate::NonNegativeRate;
use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_evolution::fitness::BatchFitnessFn;
use rlevo_evolution::strategy::EvolutionaryHarness;

type B = Flex;

const GENS: usize = 12;

struct SphereCost;

impl BatchFitnessFn<B, Tensor<B, 2>> for SphereCost {
    fn evaluate_batch(
        &mut self,
        population: &Tensor<B, 2>,
        _device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        population
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .squeeze_dim::<1>(1)
    }

    fn sense(&self) -> ObjectiveSense {
        ObjectiveSense::Minimize
    }
}

fn harness(seed: u64) -> EvolutionaryHarness<B, GeneticAlgorithm<B>, SphereCost> {
    let params = GaConfig {
        pop_size: 16,
        genome_dim: 4,
        bounds: Bounds::new(-5.0, 5.0),
        mutation_sigma: NonNegativeRate::new(0.3),
        selection: GaSelection::Tournament { size: 2 },
        crossover: GaCrossover::BlxAlpha {
            alpha: NonNegativeRate::new(0.5),
        },
        replacement: GaReplacement::Elitist { elitism_k: 1 },
    };
    EvolutionaryHarness::new(
        GeneticAlgorithm::<B>::new(),
        params,
        SphereCost,
        seed,
        Default::default(),
        GENS,
    )
    .expect("valid params")
}

fn cfg() -> EvaluatorConfig {
    EvaluatorConfig {
        // Irrelevant to a generation trial — deliberately set to a value that
        // would be wrong if the probe path honoured it as an episode count.
        num_episodes: 7,
        num_trials_per_env: 2,
        // Backstop only; the probe's own budget (GENS) must end the loop.
        max_steps: GENS * 10,
        base_seed: 17,
        num_threads: Some(1),
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: None,
    }
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn generation_trial_runs_through_the_shared_machinery() {
    let cfg = cfg();
    let mut reporter = LoggingReporter::new();
    let report = Evaluator::new(cfg.clone()).run_trials(
        "sphere-ga",
        &["sphere-d4".to_string()],
        |_key, env_seed, _agent_seed| GenerationTrial {
            probe: harness(env_seed),
        },
        &mut reporter,
    );

    assert_eq!(report.trials.len(), 2, "num_trials_per_env trials ran");

    for t in &report.trials {
        assert!(!t.errored, "trial errored: {:?}", t.error_message);

        // The point of the seam: no episode axis is fabricated.
        assert!(
            t.episodes.is_empty(),
            "a generation trial reports no episodes, got {}",
            t.episodes.len()
        );

        // The probe's budget ended the loop, not `max_steps`, and `num_episodes`
        // did not multiply the work.
        assert_eq!(
            t.scalars.get("generations").copied(),
            Some(GENS as f64),
            "probe budget governs the loop"
        );

        // Typed EA metrics arrived, which `BenchStep`'s single f64 could not
        // carry. The harness's own post-increment counter must agree with the
        // trial's independently maintained `units` count — if these ever
        // diverge, one of the two loops is miscounting.
        assert_eq!(
            t.scalars.get("ea/generation").copied(),
            Some(GENS as f64),
            "harness generation counter agrees with the trial's own count"
        );
        assert!(t.scalars.contains_key("ea/best_fitness_ever"));
        assert!(t.scalars.contains_key("ea/population_size"));
    }
}

/// Distinct trial seeds must produce distinct runs — the shared seeding
/// machinery applies to probes exactly as it does to environments.
#[test]
fn distinct_trial_seeds_give_distinct_runs() {
    let cfg = cfg();
    let mut reporter = LoggingReporter::new();
    let report = Evaluator::new(cfg).run_trials(
        "sphere-ga",
        &["sphere-d4".to_string()],
        |_key, env_seed, _agent_seed| GenerationTrial {
            probe: harness(env_seed),
        },
        &mut reporter,
    );

    let bests: Vec<f64> = report
        .trials
        .iter()
        .map(|t| t.scalars["ea/best_fitness_ever"])
        .collect();

    assert_eq!(bests.len(), 2);
    assert!(
        (bests[0] - bests[1]).abs() > f64::EPSILON,
        "per-trial seeds must diverge, both were {}",
        bests[0]
    );
}
