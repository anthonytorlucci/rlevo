//! Characterization baseline for the maximise-native / `ObjectiveSense`
//! refactor (ADR 0023).
//!
//! These assertions read **user-space** `best_fitness_ever` directly from
//! `EvolutionaryHarness::latest_metrics()` and are written to survive the
//! engine flip from minimise-native to maximise-native **unchanged**. A
//! `Minimize` landscape must still report its natural cost (Sphere → 0)
//! regardless of the engine's internal direction, so this file is the
//! regression oracle that proves the flip is behaviour-preserving in user
//! space.
//!
//! Only the *construction* of the fitness adapter changes across the
//! refactor (`FromLandscape::new` → `FromLandscape::with_sense(..,
//! ObjectiveSense::Minimize)`); the convergence bounds below do not.
//!
//! Determinism: the Flex backend seeds through a process-global mutex, so
//! these tests are single-threaded by construction (one harness per test,
//! driven serially) — no cross-thread seeding race.

use burn::backend::Flex;

use rlevo_environments::landscapes::ackley::Ackley;
use rlevo_environments::landscapes::rastrigin::Rastrigin;
use rlevo_environments::landscapes::sphere::Sphere;
use rlevo_evolution::algorithms::de::{DeConfig, DeVariant, DifferentialEvolution};
use rlevo_evolution::algorithms::es_classical::{EsConfig, EsKind, EvolutionStrategy};
use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_core::bounds::Bounds;
use rlevo_core::objective::ObjectiveSense;
use rlevo_core::rate::NonNegativeRate;
use rlevo_evolution::fitness::FromLandscape;
use rlevo_evolution::strategy::{EvolutionaryHarness, Strategy};

type B = Flex;

const DIM: usize = 10;
const MAX_GENS: usize = 120;
const SEED: u64 = 17;

/// Drives a harness to completion and returns the final user-space
/// `best_fitness_ever` (the natural cost for a `Minimize` landscape).
fn run_to_best<S, L>(strategy: S, params: S::Params, landscape: L) -> f32
where
    S: Strategy<B>,
    S::Params: rlevo_core::config::Validate,
    L: rlevo_core::fitness::Landscape + 'static,
    EvolutionaryHarness<B, S, FromLandscape<L>>: Sized,
    FromLandscape<L>: rlevo_evolution::fitness::BatchFitnessFn<B, S::Genome>,
{
    let device = Default::default();
    let mut harness = EvolutionaryHarness::<B, S, _>::new(
        strategy,
        params,
        FromLandscape::with_sense(landscape, ObjectiveSense::Minimize),
        SEED,
        device,
        MAX_GENS,
    ).expect("valid params");
    harness.reset();
    while !harness.step(()).done {}
    harness
        .latest_metrics()
        .expect("at least one generation ran")
        .best_fitness_ever()
}

fn ga_config() -> GaConfig {
    GaConfig {
        pop_size: 64,
        genome_dim: DIM,
        bounds: Bounds::new(-5.12, 5.12),
        mutation_sigma: NonNegativeRate::new(0.3),
        selection: GaSelection::Tournament { size: 3 },
        crossover: GaCrossover::BlxAlpha { alpha: NonNegativeRate::new(0.5) },
        replacement: GaReplacement::Elitist { elitism_k: 2 },
    }
}

fn es_config() -> EsConfig {
    EsConfig::default_for(EsKind::MuPlusLambda { mu: 5, lambda: 30 }, DIM)
}

fn de_config() -> DeConfig {
    let mut params = DeConfig::default_for(40, DIM);
    params.variant = DeVariant::Rand1Bin;
    params.f = 0.5;
    params.cr = 0.9;
    params
}

#[test]
fn sphere_strategies_reach_near_optimum() {
    // Sphere is convex/unimodal with optimum 0; every real-valued strategy
    // should drive best_fitness_ever well below 1.0 in user space.
    let ga = run_to_best(GeneticAlgorithm::<B>::new(), ga_config(), Sphere::new(DIM));
    let es = run_to_best(EvolutionStrategy::<B>::new(), es_config(), Sphere::new(DIM));
    let de = run_to_best(DifferentialEvolution::<B>::new(), de_config(), Sphere::new(DIM));

    for (name, best) in [("GA", ga), ("ES", es), ("DE", de)] {
        assert!(
            best.is_finite() && best >= 0.0,
            "{name} Sphere best_fitness_ever should be a non-negative cost, got {best}",
        );
        assert!(
            best < 1.0,
            "{name} Sphere best_fitness_ever = {best}, expected < 1.0 (optimum 0)",
        );
    }
}

#[test]
fn rastrigin_strategies_improve_on_random_search() {
    // Rastrigin-D10 random search averages ~80-120; a working optimizer
    // finishes well below. Loose ceiling stays stable across RNG versions
    // while catching a strategy that fails to optimize.
    let ga = run_to_best(GeneticAlgorithm::<B>::new(), ga_config(), Rastrigin::new(DIM));
    let es = run_to_best(EvolutionStrategy::<B>::new(), es_config(), Rastrigin::new(DIM));
    let de = run_to_best(DifferentialEvolution::<B>::new(), de_config(), Rastrigin::new(DIM));

    for (name, best) in [("GA", ga), ("ES", es), ("DE", de)] {
        assert!(
            best.is_finite() && best >= 0.0,
            "{name} Rastrigin best_fitness_ever should be a non-negative cost, got {best}",
        );
        assert!(
            best < 60.0,
            "{name} Rastrigin best_fitness_ever = {best}, expected < 60.0",
        );
    }
}

#[test]
fn ackley_de_reaches_near_optimum() {
    // Ackley optimum is 0; DE is a reliable converger on it.
    let de = run_to_best(DifferentialEvolution::<B>::new(), de_config(), Ackley::new(DIM));
    assert!(
        de.is_finite() && de >= 0.0,
        "DE Ackley best_fitness_ever should be a non-negative cost, got {de}",
    );
    assert!(
        de < 5.0,
        "DE Ackley best_fitness_ever = {de}, expected < 5.0 (optimum 0)",
    );
}
