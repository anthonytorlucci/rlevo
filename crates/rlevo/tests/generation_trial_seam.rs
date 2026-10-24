//! Cross-crate proof that the `Trial` seam is shape-agnostic.
//!
//! Drives an `EvolutionaryLoop` through `Evaluator::run_trials` as a
//! `GenerationTrial` — no `BenchEnv`, no `Agent`, no episode axis —
//! and asserts it gets the same machinery the episodic path does: rayon
//! fan-out, per-trial seeding, checkpoint-shaped `TrialReport`s, and the
//! reporter lifecycle.

use burn::tensor::{Device, Tensor};
use rlevo_core::bounds::Bounds;
use rlevo_core::objective::ObjectiveSense;
use rlevo_core::rate::NonNegativeRate;
use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_evolution::fitness::BatchFitnessFn;
use rlevo_evolution::strategy::EvolutionaryLoop;
use rlevo_harness::evaluator::{Evaluator, EvaluatorConfig, GenerationTrial};
use rlevo_harness::metrics::core::WALL_CLOCK_SECONDS;
use rlevo_harness::reporter::logging::LoggingReporter;

const GENS: usize = 12;

struct SphereCost;

impl BatchFitnessFn<Tensor<2>> for SphereCost {
    fn evaluate_batch(&mut self, population: &Tensor<2>, _device: &Device) -> Tensor<1> {
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

fn evo_loop(seed: u64) -> EvolutionaryLoop<GeneticAlgorithm, SphereCost> {
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
    EvolutionaryLoop::new(
        GeneticAlgorithm::new(),
        params,
        SphereCost,
        seed,
        Device::flex(),
        GENS,
    )
    .expect("valid params")
}

fn cfg() -> EvaluatorConfig {
    EvaluatorConfig {
        // Irrelevant to a generation trial — deliberately set to a value that
        // would be wrong if the probe path honoured it as an episode count.
        num_episodes: 7,
        train_steps: 0,
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
            probe: evo_loop(env_seed),
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
        // carry. The loop's own post-increment counter must agree with the
        // trial's independently maintained `units` count — if these ever
        // diverge, one of the two loops is miscounting.
        assert_eq!(
            t.scalars.get("ea/generation").copied(),
            Some(GENS as f64),
            "the loop's generation counter agrees with the trial's own count"
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
            probe: evo_loop(env_seed),
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

/// Seeded trials give the same result whether the evaluator runs them on one
/// thread or on a pool, scalar for scalar and bit for bit.
///
/// A strategy draws every random number from its loop's own host RNG and none
/// from the backend's process-global generator, so thread scheduling cannot
/// reach a trial's trajectory. This pins that: if a strategy or the loop ever
/// draws from the backend generator, the pooled runs stop matching the serial
/// one.
#[test]
fn parallel_trials_match_serial_trials() {
    const TRIALS: usize = 12;

    fn run(num_threads: Option<usize>) -> Vec<(usize, Vec<(String, u64)>)> {
        let cfg = EvaluatorConfig {
            num_trials_per_env: TRIALS,
            num_threads,
            ..cfg()
        };
        let mut reporter = LoggingReporter::new();
        let report = Evaluator::new(cfg).run_trials(
            "sphere-ga",
            &["sphere-d4".to_string()],
            |_key, env_seed, _agent_seed| GenerationTrial {
                probe: evo_loop(env_seed),
            },
            &mut reporter,
        );
        let mut rows: Vec<(usize, Vec<(String, u64)>)> = report
            .trials
            .iter()
            .map(|t| {
                let scalars = t
                    .scalars
                    .iter()
                    // Wall-clock time is the one scalar that is not a function
                    // of the seed.
                    .filter(|(name, _)| name.as_str() != WALL_CLOCK_SECONDS)
                    .map(|(name, value)| (name.clone(), value.to_bits()))
                    .collect();
                (t.key.trial_idx, scalars)
            })
            .collect();
        rows.sort();
        rows
    }

    let serial = run(Some(1));
    assert_eq!(serial.len(), TRIALS);
    assert!(
        serial.iter().all(|(_, scalars)| scalars.len() >= 8),
        "every trial reports its generation count and the ea/* scalars"
    );
    assert_eq!(serial, run(Some(4)), "four threads must match one");
    assert_eq!(serial, run(None), "the default pool must match one thread");
}
