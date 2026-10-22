//! Every shipping swarm strategy drives
//! `rlevo_harness::Evaluator::run_suite` end-to-end on Rastrigin-D10
//! and Ackley-D10.
//!
//! Mirrors the `rastrigin_run_suite.rs` pattern: assert each strategy
//! finishes materially below the uniform-random baseline. Precise
//! convergence targets belong to per-strategy tests in the
//! `algorithms::swarm::*` modules. ACO-Permutation is excluded because
//! its module is a `todo!()` stub.

use burn::backend::Flex;
use rlevo_core::evaluation::GenerationProbe;
use rlevo_core::fitness::MetricsProvider;
use rlevo_harness::evaluator::{Evaluator, EvaluatorConfig, GenerationTrial};
use rlevo_harness::reporter::logging::LoggingReporter;

use rlevo_core::objective::ObjectiveSense;
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
use rlevo_evolution::fitness::FromLandscape;
use rlevo_evolution::strategy::{EvolutionaryLoop, Strategy};

type B = Flex;
const DIM: usize = 10;
const MAX_GENS: usize = 120;

/// Returns a minimal [`EvaluatorConfig`] shared by every swarm strategy suite test.
///
/// Key values:
/// - `num_trials_per_env: 2` — enough signal to catch systematic failures without
///   inflating wall-clock time.
/// - `max_steps: MAX_GENS` (120) — matches the generation budget given to each
///   [`EvolutionaryLoop`]. The same constant is the schedule horizon of the
///   three strategies that have one (GWO, WOA, SSA).
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

/// Runs the given loop factory through the evaluator as generation trials and returns the
/// mean, across all trials, of each trial's best fitness as a single `f64`.
///
/// Each trial reports `ea/best_fitness_ever` in the objective's natural sense, so for these cost
/// landscapes a lower value is better and no sign needs undoing.
///
/// The return value is the arithmetic mean over trials, not a `Vec<f64>`. This differs
/// from `rastrigin_run_suite.rs`'s `collect_best_returns`, which returns each trial's
/// value separately to support per-trial assertions; here a single mean is sufficient
/// because all swarm strategies are checked against a shared ceiling in one test.
fn collect_best_returns<P>(
    suite_name: &str,
    env_name: &str,
    factory: impl Fn(u64) -> P + Send + Sync,
) -> f64
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
    let bests: Vec<f64> = report
        .trials
        .iter()
        .map(|t| {
            *t.scalars
                .get("ea/best_fitness_ever")
                .expect("generation trial emits ea/best_fitness_ever")
        })
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let n = bests.len() as f64;
    bests.iter().sum::<f64>() / n
}

// ---------------------------------------------------------------------
// Rastrigin factories — one per shipping swarm strategy.
// ---------------------------------------------------------------------

fn pso_ra(seed: u64) -> EvolutionaryLoop<B, ParticleSwarm<B>, FromLandscape<Rastrigin>>
where
    ParticleSwarm<B>: Strategy<B, Params = PsoConfig>,
{
    EvolutionaryLoop::new(
        ParticleSwarm::<B>::new(),
        PsoConfig::default_for(32, DIM),
        FromLandscape::with_sense(
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        Default::default(),
        MAX_GENS,
    )
    .expect("valid params")
}

fn gwo_ra(seed: u64) -> EvolutionaryLoop<B, GreyWolfOptimizer<B>, FromLandscape<Rastrigin>> {
    EvolutionaryLoop::new(
        GreyWolfOptimizer::<B>::new(),
        GwoConfig::default_for(32, DIM, MAX_GENS),
        FromLandscape::with_sense(
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        Default::default(),
        MAX_GENS,
    )
    .expect("valid params")
}

fn woa_ra(seed: u64) -> EvolutionaryLoop<B, WhaleOptimization<B>, FromLandscape<Rastrigin>> {
    EvolutionaryLoop::new(
        WhaleOptimization::<B>::new(),
        WoaConfig::default_for(32, DIM, MAX_GENS),
        FromLandscape::with_sense(
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        Default::default(),
        MAX_GENS,
    )
    .expect("valid params")
}

fn salp_ra(seed: u64) -> EvolutionaryLoop<B, SalpSwarm<B>, FromLandscape<Rastrigin>> {
    EvolutionaryLoop::new(
        SalpSwarm::<B>::new(),
        SalpConfig::default_for(32, DIM, MAX_GENS),
        FromLandscape::with_sense(
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        Default::default(),
        MAX_GENS,
    )
    .expect("valid params")
}

fn abc_ra(seed: u64) -> EvolutionaryLoop<B, ArtificialBeeColony<B>, FromLandscape<Rastrigin>> {
    EvolutionaryLoop::new(
        ArtificialBeeColony::<B>::new(),
        AbcConfig::default_for(30, DIM),
        FromLandscape::with_sense(
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        Default::default(),
        MAX_GENS,
    )
    .expect("valid params")
}

fn bat_ra(seed: u64) -> EvolutionaryLoop<B, BatAlgorithm<B>, FromLandscape<Rastrigin>> {
    EvolutionaryLoop::new(
        BatAlgorithm::<B>::new(),
        BatConfig::default_for(30, DIM),
        FromLandscape::with_sense(
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        Default::default(),
        MAX_GENS,
    )
    .expect("valid params")
}

fn aco_r_ra(seed: u64) -> EvolutionaryLoop<B, AntColonyReal<B>, FromLandscape<Rastrigin>> {
    EvolutionaryLoop::new(
        AntColonyReal::<B>::new(),
        AcoRConfig::default_for(30, 15, DIM),
        FromLandscape::with_sense(
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        Default::default(),
        MAX_GENS,
    )
    .expect("valid params")
}

fn cuckoo_ra(seed: u64) -> EvolutionaryLoop<B, CuckooSearch<B>, FromLandscape<Rastrigin>> {
    let mut cfg = CuckooConfig::default_for(30, DIM);
    cfg.alpha = 0.2;
    EvolutionaryLoop::new(
        CuckooSearch::<B>::new(),
        cfg,
        FromLandscape::with_sense(
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        Default::default(),
        MAX_GENS,
    )
    .expect("valid params")
}

fn firefly_ra(seed: u64) -> EvolutionaryLoop<B, FireflyAlgorithm<B>, FromLandscape<Rastrigin>> {
    EvolutionaryLoop::new(
        FireflyAlgorithm::<B>::new(),
        FireflyConfig::default_for(24, DIM),
        FromLandscape::with_sense(
            Rastrigin::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        Default::default(),
        MAX_GENS,
    )
    .expect("valid params")
}

// ---------------------------------------------------------------------
// Ackley factories — mirror the Rastrigin ones.
// ---------------------------------------------------------------------

fn pso_ak(seed: u64) -> EvolutionaryLoop<B, ParticleSwarm<B>, FromLandscape<Ackley>> {
    EvolutionaryLoop::new(
        ParticleSwarm::<B>::new(),
        PsoConfig::default_for(32, DIM),
        FromLandscape::with_sense(
            Ackley::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        Default::default(),
        MAX_GENS,
    )
    .expect("valid params")
}

fn de_rand1_ak(
    seed: u64,
) -> EvolutionaryLoop<
    B,
    rlevo_evolution::algorithms::de::DifferentialEvolution<B>,
    FromLandscape<Ackley>,
> {
    use rlevo_evolution::algorithms::de::{DeConfig, DeVariant, DifferentialEvolution};
    let mut params = DeConfig::default_for(40, DIM);
    params.variant = DeVariant::Rand1Bin;
    EvolutionaryLoop::new(
        DifferentialEvolution::<B>::new(),
        params,
        FromLandscape::with_sense(
            Ackley::new(DIM).expect("dim >= 1"),
            ObjectiveSense::Minimize,
        ),
        seed,
        Default::default(),
        MAX_GENS,
    )
    .expect("valid params")
}

#[test]
fn swarm_strategies_reduce_on_rastrigin_and_ackley() {
    // Ceilings recalibrated for the terminal-best metric
    // (`ea/best_fitness_ever`). The previous values were tuned for the
    // mean-best-so-far the `BenchEnv` path measured; terminal best is
    // strictly lower, which left the Ackley ceiling of 18.0 vacuous against
    // observed values near 0.02.
    //
    // Measured with this file's `cfg()`. Rastrigin-D10 averages: WOA ~3e-14,
    // ABC 1.2, GWO 2.2, SSA 11.9, PSO 13.2, Firefly 19.0, ACO_R 33.4,
    // Bat 69.7, Cuckoo 72.5. Ackley-D10: DE 0.022, PSO 0.010.
    //
    // GWO, WOA and SSA pace a schedule over a horizon of their own, and are
    // given `MAX_GENS` for it, the loop's budget. Their constructors once
    // filled in 500, so the 120 generations here covered a quarter of the
    // schedule: SSA scored 42.1 and GWO exactly 0.0. The 0.0 was not search.
    // GWO is drawn to the origin, where this Rastrigin has its optimum, and
    // on a long schedule it is still being drawn there when the run ends.
    //
    // Bars are set to catch "stopped optimizing" without flaking: 100.0 on
    // Rastrigin leaves the worst (Cuckoo, 72.5) ~38% headroom, and 1.0 on
    // Ackley is ~50x the observed values while still far under the ~18-21
    // uniform-random baseline. Both are materially tighter than before.
    //
    // Caveat: the random-search baselines quoted above were derived for the
    // OLD metric. These bars are calibrated off observed optimizer behaviour,
    // not off a re-derived random-search margin for terminal best.
    let rastrigin_ceiling = 100.0_f64;
    let ackley_ceiling = 1.0_f64;

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

    // Ackley — DE/Rand1/bin is the classical comparator and PSO the canonical
    // swarm baseline; both must clear the same ceiling. (The surrounding prose
    // previously described a "within 2x of DE" rule that the code never
    // implemented; the code's actual contract is a shared ceiling.)
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
