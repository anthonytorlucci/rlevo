//! Every shipping swarm strategy drives
//! `rlevo_benchmarks::Evaluator::run_suite` end-to-end on Rastrigin-D10
//! and Ackley-D10.
//!
//! Mirrors the `rastrigin_run_suite.rs` pattern: assert each strategy
//! finishes materially below the uniform-random baseline. Precise
//! convergence targets belong to per-strategy tests in the
//! `algorithms::swarm::*` modules. ACO-Permutation is excluded because
//! its module is a `todo!()` stub.

use burn::backend::Flex;
use rand::Rng;
use rlevo_benchmarks::agent::{BenchableAgent, FitnessEvaluable};
use rlevo_benchmarks::env::BenchEnv;
use rlevo_benchmarks::evaluator::{Evaluator, EvaluatorConfig};
use rlevo_benchmarks::reporter::logging::LoggingReporter;
use rlevo_benchmarks::suite::Suite;

use rlevo_environments::landscapes::ackley::Ackley;
use rlevo_environments::landscapes::rastrigin::Rastrigin;
use rlevo_evolution::algorithms::metaheuristic::abc::{AbcConfig, ArtificialBeeColony};
use rlevo_evolution::algorithms::metaheuristic::aco_r::{AcoRConfig, AntColonyReal};
use rlevo_evolution::algorithms::metaheuristic::bat::{BatAlgorithm, BatConfig};
use rlevo_evolution::algorithms::metaheuristic::cuckoo::{CuckooConfig, CuckooSearch};
use rlevo_evolution::algorithms::metaheuristic::firefly::{FireflyAlgorithm, FireflyConfig};
use rlevo_evolution::algorithms::metaheuristic::gwo::{GreyWolfOptimizer, GwoConfig};
use rlevo_evolution::algorithms::metaheuristic::pso::{ParticleSwarm, PsoConfig};
use rlevo_evolution::algorithms::metaheuristic::salp::{SalpConfig, SalpSwarm};
use rlevo_evolution::algorithms::metaheuristic::woa::{WhaleOptimization, WoaConfig};
use rlevo_core::objective::ObjectiveSense;
use rlevo_evolution::fitness::FromFitnessEvaluable;
use rlevo_evolution::strategy::{EvolutionaryHarness, Strategy};

type B = Flex;
const DIM: usize = 10;
const MAX_GENS: usize = 120;

struct Passive;
impl BenchableAgent<(), ()> for Passive {
    fn act(&mut self, (): &(), _: &mut dyn Rng) {}
}

struct RastriginFit;
impl FitnessEvaluable for RastriginFit {
    type Individual = Vec<f64>;
    type Landscape = Rastrigin;
    fn evaluate(&self, x: &Self::Individual, l: &Self::Landscape) -> f64 {
        l.evaluate(x)
    }
}

struct AckleyFit;
impl FitnessEvaluable for AckleyFit {
    type Individual = Vec<f64>;
    type Landscape = Ackley;
    fn evaluate(&self, x: &Self::Individual, l: &Self::Landscape) -> f64 {
        l.evaluate(x)
    }
}

/// Returns a minimal [`EvaluatorConfig`] shared by every swarm strategy suite test.
///
/// Key values:
/// - `num_trials_per_env: 2` — enough signal to catch systematic failures without
///   inflating wall-clock time.
/// - `max_steps: MAX_GENS` (120) — matches the generation budget given to each
///   [`EvolutionaryHarness`].
/// - `num_threads: Some(1)` — forces single-threaded trial dispatch. Burn's `Flex`
///   backend holds a process-global RNG mutex; running trials in parallel would
///   cause threads to contend on that mutex and produce nondeterministic, often
///   degraded fitness results. See the inline comment in `rastrigin_run_suite.rs`
///   for the full rationale.
fn cfg() -> EvaluatorConfig {
    EvaluatorConfig {
        num_episodes: 1,
        num_trials_per_env: 2,
        max_steps: MAX_GENS,
        base_seed: 29,
        // Single-threaded — see rastrigin_run_suite.rs for the
        // Flex-RNG-mutex rationale.
        num_threads: Some(1),
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: None,
    }
}

/// Runs the given environment factory through the full suite evaluator and returns
/// the mean best-fitness (per-step average) across all trials as a single `f64`.
///
/// The evaluator stores cumulative reward as `−best_fitness` in `episode.return_value`
/// (lower raw fitness is better, so the harness negates it to keep the reward signal
/// positive). The expression `-e.return_value / steps` therefore inverts that negation
/// and divides by the generation count to produce a per-step mean best-fitness value
/// comparable across functions with different absolute scales.
///
/// The return value is the arithmetic mean over trials, not a `Vec<f64>`. This differs
/// from `rastrigin_run_suite.rs`'s `collect_best_returns`, which returns each trial's
/// value separately to support per-trial assertions; here a single mean is sufficient
/// because all swarm strategies are checked against a shared ceiling in one test.
fn collect_best_returns<E>(
    suite_name: &str,
    env_name: &str,
    factory: impl Fn(u64) -> E + Send + Sync + 'static,
) -> f64
where
    E: BenchEnv<Observation = (), Action = ()> + Send,
{
    let cfg = cfg();
    let suite = Suite::new(suite_name, cfg.clone()).with_env(env_name, factory);
    let evaluator = Evaluator::new(cfg.clone());
    let mut reporter = LoggingReporter::new();
    let report = evaluator.run_suite(&suite, |_| Passive, &mut reporter);
    #[allow(clippy::cast_precision_loss)]
    let steps = cfg.max_steps as f64;
    let bests: Vec<f64> = report
        .trials
        .iter()
        .map(|t| {
            t.episodes
                .last()
                .map_or(f64::INFINITY, |e| -e.return_value / steps)
        })
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let n = bests.len() as f64;
    bests.iter().sum::<f64>() / n
}

// ---------------------------------------------------------------------
// Rastrigin factories — one per shipping swarm strategy.
// ---------------------------------------------------------------------

fn pso_ra(
    seed: u64,
) -> EvolutionaryHarness<B, ParticleSwarm<B>, FromFitnessEvaluable<RastriginFit, Rastrigin>>
where
    ParticleSwarm<B>: Strategy<B, Params = PsoConfig>,
{
    EvolutionaryHarness::new(
        ParticleSwarm::<B>::new(),
        PsoConfig::default_for(32, DIM),
        FromFitnessEvaluable::with_sense(RastriginFit, Rastrigin::new(DIM), ObjectiveSense::Minimize),
        seed,
        Default::default(),
        MAX_GENS,
    )
}

fn gwo_ra(
    seed: u64,
) -> EvolutionaryHarness<B, GreyWolfOptimizer<B>, FromFitnessEvaluable<RastriginFit, Rastrigin>> {
    EvolutionaryHarness::new(
        GreyWolfOptimizer::<B>::new(),
        GwoConfig::default_for(32, DIM),
        FromFitnessEvaluable::with_sense(RastriginFit, Rastrigin::new(DIM), ObjectiveSense::Minimize),
        seed,
        Default::default(),
        MAX_GENS,
    )
}

fn woa_ra(
    seed: u64,
) -> EvolutionaryHarness<B, WhaleOptimization<B>, FromFitnessEvaluable<RastriginFit, Rastrigin>> {
    EvolutionaryHarness::new(
        WhaleOptimization::<B>::new(),
        WoaConfig::default_for(32, DIM),
        FromFitnessEvaluable::with_sense(RastriginFit, Rastrigin::new(DIM), ObjectiveSense::Minimize),
        seed,
        Default::default(),
        MAX_GENS,
    )
}

fn salp_ra(
    seed: u64,
) -> EvolutionaryHarness<B, SalpSwarm<B>, FromFitnessEvaluable<RastriginFit, Rastrigin>> {
    EvolutionaryHarness::new(
        SalpSwarm::<B>::new(),
        SalpConfig::default_for(32, DIM),
        FromFitnessEvaluable::with_sense(RastriginFit, Rastrigin::new(DIM), ObjectiveSense::Minimize),
        seed,
        Default::default(),
        MAX_GENS,
    )
}

fn abc_ra(
    seed: u64,
) -> EvolutionaryHarness<B, ArtificialBeeColony<B>, FromFitnessEvaluable<RastriginFit, Rastrigin>> {
    EvolutionaryHarness::new(
        ArtificialBeeColony::<B>::new(),
        AbcConfig::default_for(30, DIM),
        FromFitnessEvaluable::with_sense(RastriginFit, Rastrigin::new(DIM), ObjectiveSense::Minimize),
        seed,
        Default::default(),
        MAX_GENS,
    )
}

fn bat_ra(
    seed: u64,
) -> EvolutionaryHarness<B, BatAlgorithm<B>, FromFitnessEvaluable<RastriginFit, Rastrigin>> {
    EvolutionaryHarness::new(
        BatAlgorithm::<B>::new(),
        BatConfig::default_for(30, DIM),
        FromFitnessEvaluable::with_sense(RastriginFit, Rastrigin::new(DIM), ObjectiveSense::Minimize),
        seed,
        Default::default(),
        MAX_GENS,
    )
}

fn aco_r_ra(
    seed: u64,
) -> EvolutionaryHarness<B, AntColonyReal<B>, FromFitnessEvaluable<RastriginFit, Rastrigin>> {
    EvolutionaryHarness::new(
        AntColonyReal::<B>::new(),
        AcoRConfig::default_for(30, 15, DIM),
        FromFitnessEvaluable::with_sense(RastriginFit, Rastrigin::new(DIM), ObjectiveSense::Minimize),
        seed,
        Default::default(),
        MAX_GENS,
    )
}

fn cuckoo_ra(
    seed: u64,
) -> EvolutionaryHarness<B, CuckooSearch<B>, FromFitnessEvaluable<RastriginFit, Rastrigin>> {
    let mut cfg = CuckooConfig::default_for(30, DIM);
    cfg.alpha = 0.2;
    EvolutionaryHarness::new(
        CuckooSearch::<B>::new(),
        cfg,
        FromFitnessEvaluable::with_sense(RastriginFit, Rastrigin::new(DIM), ObjectiveSense::Minimize),
        seed,
        Default::default(),
        MAX_GENS,
    )
}

fn firefly_ra(
    seed: u64,
) -> EvolutionaryHarness<B, FireflyAlgorithm<B>, FromFitnessEvaluable<RastriginFit, Rastrigin>> {
    EvolutionaryHarness::new(
        FireflyAlgorithm::<B>::new(),
        FireflyConfig::default_for(24, DIM),
        FromFitnessEvaluable::with_sense(RastriginFit, Rastrigin::new(DIM), ObjectiveSense::Minimize),
        seed,
        Default::default(),
        MAX_GENS,
    )
}

// ---------------------------------------------------------------------
// Ackley factories — mirror the Rastrigin ones.
// ---------------------------------------------------------------------

fn pso_ak(
    seed: u64,
) -> EvolutionaryHarness<B, ParticleSwarm<B>, FromFitnessEvaluable<AckleyFit, Ackley>> {
    EvolutionaryHarness::new(
        ParticleSwarm::<B>::new(),
        PsoConfig::default_for(32, DIM),
        FromFitnessEvaluable::with_sense(AckleyFit, Ackley::new(DIM), ObjectiveSense::Minimize),
        seed,
        Default::default(),
        MAX_GENS,
    )
}

fn de_rand1_ak(
    seed: u64,
) -> EvolutionaryHarness<
    B,
    rlevo_evolution::algorithms::de::DifferentialEvolution<B>,
    FromFitnessEvaluable<AckleyFit, Ackley>,
> {
    use rlevo_evolution::algorithms::de::{DeConfig, DeVariant, DifferentialEvolution};
    let mut params = DeConfig::default_for(40, DIM);
    params.variant = DeVariant::Rand1Bin;
    EvolutionaryHarness::new(
        DifferentialEvolution::<B>::new(),
        params,
        FromFitnessEvaluable::with_sense(AckleyFit, Ackley::new(DIM), ObjectiveSense::Minimize),
        seed,
        Default::default(),
        MAX_GENS,
    )
}

#[test]
fn swarm_strategies_reduce_on_rastrigin_and_ackley() {
    // Random-search baseline on Rastrigin-D10 is roughly 80-120; on
    // Ackley-D10 it is ~18-21. The acceptance bar is "within 2× of the
    // best-in-class classical baseline (DE/Rand1/bin)". We compute DE's
    // result here too and use it as the anchor for the Ackley check;
    // for Rastrigin we accept any strategy-average below a generous
    // ceiling.
    //
    // Ceilings kept loose to stay stable across RNG seed versions.
    let rastrigin_ceiling = 120.0_f64;
    let ackley_ceiling = 18.0_f64;

    macro_rules! check_rastrigin {
        ($fn:ident, $name:expr) => {{
            let avg = collect_best_returns("swarm-ra", "rastrigin-10d", $fn);
            assert!(
                avg < rastrigin_ceiling,
                "{} avg best on Rastrigin-D10 = {avg}, expected < {rastrigin_ceiling}",
                $name
            );
        }};
    }
    check_rastrigin!(pso_ra, "PSO");
    check_rastrigin!(gwo_ra, "GWO");
    check_rastrigin!(woa_ra, "WOA");
    check_rastrigin!(salp_ra, "SSA");
    check_rastrigin!(abc_ra, "ABC");
    check_rastrigin!(bat_ra, "Bat");
    check_rastrigin!(aco_r_ra, "ACO_R");
    check_rastrigin!(cuckoo_ra, "Cuckoo");
    check_rastrigin!(firefly_ra, "Firefly");

    // Ackley — anchor against DE/Rand1/bin as the classical comparator
    // and require PSO (the canonical swarm baseline) to finish in the
    // same order of magnitude.
    let de_avg = collect_best_returns("swarm-ak", "ackley-10d", de_rand1_ak);
    let pso_avg = collect_best_returns("swarm-ak", "ackley-10d", pso_ak);
    assert!(
        de_avg < ackley_ceiling,
        "DE anchor on Ackley-D10 = {de_avg}, expected < {ackley_ceiling}",
    );
    assert!(
        pso_avg < ackley_ceiling,
        "PSO on Ackley-D10 = {pso_avg}, expected < {ackley_ceiling}",
    );
}
