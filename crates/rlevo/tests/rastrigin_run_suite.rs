//! Rastrigin-D10 convergence via the benchmark harness.
//!
//! This test verifies that each real-valued strategy plugs into
//! `rlevo_benchmarks::Evaluator::run_suite` end-to-end. We assert that
//! the final best-fitness across trials is well below a random-search
//! baseline; precise convergence targets belong to per-strategy tests
//! in the `algorithms` module.

use std::sync::atomic::AtomicU32;

use burn::backend::Flex;
use rlevo_benchmarks::agent::FitnessEvaluable;
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig, GenerationTrial};
use rlevo_benchmarks::reporter::logging::LoggingReporter;
use rlevo_core::evaluation::GenerationProbe;
use rlevo_core::fitness::MetricsProvider;

use rlevo_core::bounds::Bounds;
use rlevo_core::objective::ObjectiveSense;
use rlevo_core::rate::NonNegativeRate;
use rlevo_environments::landscapes::rastrigin::Rastrigin;
use rlevo_evolution::algorithms::de::{DeConfig, DeVariant, DifferentialEvolution};
use rlevo_evolution::algorithms::ep::{EpConfig, EvolutionaryProgramming};
use rlevo_evolution::algorithms::es_classical::{EsConfig, EsKind, EvolutionStrategy};
use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_evolution::fitness::FromFitnessEvaluable;
use rlevo_evolution::strategy::EvolutionaryHarness;

type B = Flex;

const DIM: usize = 10;
const MAX_GENS: usize = 80;

struct Minimizer;
impl FitnessEvaluable for Minimizer {
    type Individual = Vec<f64>;
    type Landscape = Rastrigin;
    fn evaluate(&self, x: &Self::Individual, l: &Self::Landscape) -> f64 {
        l.evaluate(x)
    }
}

/// Creates a GA harness on Rastrigin-D10 with tournament selection, BLX-α crossover, and elitist replacement.
fn ga_factory(
    seed: u64,
) -> EvolutionaryHarness<B, GeneticAlgorithm<B>, FromFitnessEvaluable<Minimizer, Rastrigin>> {
    let device = Default::default();
    let params = GaConfig {
        pop_size: 64,
        genome_dim: DIM,
        bounds: Bounds::new(-5.12, 5.12),
        mutation_sigma: NonNegativeRate::new(0.3),
        selection: GaSelection::Tournament { size: 3 },
        crossover: GaCrossover::BlxAlpha {
            alpha: NonNegativeRate::new(0.5),
        },
        replacement: GaReplacement::Elitist { elitism_k: 2 },
    };
    EvolutionaryHarness::new(
        GeneticAlgorithm::<B>::new(),
        params,
        FromFitnessEvaluable::with_sense(
            Minimizer,
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        device,
        MAX_GENS,
    )
    .expect("valid params")
}

/// Creates a (5+30)-ES harness on Rastrigin-D10 with self-adaptive step sizes.
fn es_factory(
    seed: u64,
) -> EvolutionaryHarness<B, EvolutionStrategy<B>, FromFitnessEvaluable<Minimizer, Rastrigin>> {
    let device = Default::default();
    let params = EsConfig::default_for(EsKind::MuPlusLambda { mu: 5, lambda: 30 }, DIM);
    EvolutionaryHarness::new(
        EvolutionStrategy::<B>::new(),
        params,
        FromFitnessEvaluable::with_sense(
            Minimizer,
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        device,
        MAX_GENS,
    )
    .expect("valid params")
}

/// Creates an EP harness on Rastrigin-D10 with population size 30.
fn ep_factory(
    seed: u64,
) -> EvolutionaryHarness<B, EvolutionaryProgramming<B>, FromFitnessEvaluable<Minimizer, Rastrigin>>
{
    let device = Default::default();
    let params = EpConfig::default_for(30, DIM);
    EvolutionaryHarness::new(
        EvolutionaryProgramming::<B>::new(),
        params,
        FromFitnessEvaluable::with_sense(
            Minimizer,
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        device,
        MAX_GENS,
    )
    .expect("valid params")
}

/// Creates a DE/Rand/1/bin harness on Rastrigin-D10 with F=0.5 and CR=0.9.
fn de_factory(
    seed: u64,
) -> EvolutionaryHarness<B, DifferentialEvolution<B>, FromFitnessEvaluable<Minimizer, Rastrigin>> {
    let device = Default::default();
    let mut params = DeConfig::default_for(40, DIM);
    params.variant = DeVariant::Rand1Bin;
    params.f = 0.5;
    params.cr = 0.9;
    EvolutionaryHarness::new(
        DifferentialEvolution::<B>::new(),
        params,
        FromFitnessEvaluable::with_sense(
            Minimizer,
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        device,
        MAX_GENS,
    )
    .expect("valid params")
}

/// Returns a minimal [`EvaluatorConfig`] for the Rastrigin run-suite tests.
///
/// `num_threads` is pinned to `Some(1)` because Burn Flex seeds via a
/// process-wide mutex — parallel trials race on seeding and produce
/// non-reproducible trajectories. Forcing one thread is the simplest
/// honest option.
///
/// Runs 2 trials per env over 80 generations (matching `MAX_GENS`).
fn cfg() -> EvaluatorConfig {
    EvaluatorConfig {
        num_episodes: 1,
        num_trials_per_env: 2,
        max_steps: MAX_GENS,
        base_seed: 17,
        num_threads: Some(1),
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: None,
    }
}

/// Runs the probe factory through the evaluator and returns one `f64` per
/// trial: the best cost each trial reached, in the objective's natural sense.
///
/// Reads `ea/best_fitness_ever` straight off the trial's scalars. The previous
/// `BenchEnv` version had to compute `-e.return_value / steps` off
/// `episodes.last()` — three corrections at once (negate for objective sense,
/// divide to un-integrate a summed best-so-far, and `.last()` because the
/// episode axis is degenerate for an evolutionary run). A `GenerationProbe`
/// reports the value directly, so none of them are needed.
fn collect_best_returns<P>(
    suite_name: &str,
    env_name: &str,
    factory: impl Fn(u64) -> P + Sync + Send,
) -> Vec<f64>
where
    P: GenerationProbe + Send,
    P::Metrics: MetricsProvider,
{
    let cfg = cfg();
    let evaluator = Evaluator::new(cfg);
    let mut reporter = LoggingReporter::new();
    let report = evaluator.run_trials(
        suite_name,
        &[env_name.to_string()],
        |_key, env_seed, _agent_seed| GenerationTrial {
            probe: factory(env_seed),
        },
        &mut reporter,
    );
    report
        .trials
        .iter()
        .map(|t| {
            *t.scalars
                .get("ea/best_fitness_ever")
                .expect("generation trial emits ea/best_fitness_ever")
        })
        .collect()
}

#[test]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::similar_names)]
fn all_strategies_improve_on_rastrigin_via_run_suite() {
    // Each value is a trial's terminal best cost (`ea/best_fitness_ever`),
    // not the mean best-so-far the `BenchEnv` version measured. The two are
    // not interchangeable: terminal best <= mean-so-far, so the previous
    // ceiling of 120.0 would be close to vacuous here.
    let ga_bests = collect_best_returns("ga-rastrigin", "rastrigin-10d", ga_factory);
    let es_bests = collect_best_returns("es-rastrigin", "rastrigin-10d", es_factory);
    let ep_bests = collect_best_returns("ep-rastrigin", "rastrigin-10d", ep_factory);
    let de_bests = collect_best_returns("de-rastrigin", "rastrigin-10d", de_factory);

    // Recalibrated for the terminal-best metric. Measured averages over 80
    // generations with `base_seed: 17` are GA 31.9, EP 31.6, DE 40.6, ES 48.3;
    // 70.0 leaves the worst of those ~45% headroom for RNG seed drift while
    // staying far tighter than the 120.0 the mean-so-far metric needed. It
    // also sits above ES's worst single trial (68.7), so an unlucky pair of
    // draws does not flake the suite.
    //
    // Caveat, stated rather than papered over: the uniform-random-search
    // baseline was re-derived for the OLD metric (~80-120 mean best-so-far),
    // not for terminal best. This bar is calibrated off observed optimizer
    // behaviour, so it catches "stopped optimizing entirely" but is not a
    // measured margin over random search.
    let max_acceptable = 70.0_f64;
    for (name, bests) in [
        ("GA", &ga_bests),
        ("ES", &es_bests),
        ("EP", &ep_bests),
        ("DE", &de_bests),
    ] {
        let avg: f64 = bests.iter().sum::<f64>() / (bests.len() as f64);
        assert!(
            avg < max_acceptable,
            "{name} avg best on Rastrigin-D10 = {avg}, expected < {max_acceptable}",
        );
    }
}

/// Sanity: the harness satisfies every bound `run_trials` needs, and is
/// `Send` so rayon can own one per worker. Doubles as a compile-time check
/// that the `GenerationProbe` / `MetricsProvider` bounds line up.
#[test]
fn harness_is_send_and_satisfies_the_probe_bounds() {
    fn assert_drivable<P>(_: &P)
    where
        P: GenerationProbe + Send,
        P::Metrics: MetricsProvider,
    {
    }

    let _send_check = AtomicU32::new(0);
    let mut harness = ga_factory(0);
    assert_drivable(&harness);

    // One generation runs without panicking, and reports metrics.
    harness.begin();
    let m = harness.advance().expect("first generation yields metrics");
    assert!(!m.emit().is_empty(), "metrics provider emits scalars");
}
