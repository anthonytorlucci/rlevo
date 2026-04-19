//! Determinism integration test.
//!
//! On the ndarray backend, running a strategy twice with the same
//! `base_seed` must produce a bit-identical fitness trajectory.
//!
//! # Why this file runs a single test
//!
//! The Burn ndarray backend seeds its RNG through a process-wide
//! `Mutex<Option<NdArrayRng>>`. Multiple tests that race on that mutex
//! will interleave `Tensor::random` draws and destroy bit-equality. This
//! file therefore contains exactly one `#[test]` function so the cargo
//! runner cannot execute anything in parallel with it — within the
//! single test body, the two runs are strictly sequential and compare
//! byte-for-byte.

use burn::backend::NdArray;
use evorl_benchmarks::agent::FitnessEvaluable;
use evorl_benchmarks::env::BenchEnv;

use evorl_evolution::algorithms::es_classical::{EsConfig, EsKind, EvolutionStrategy};
use evorl_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use evorl_evolution::algorithms::swarm::abc::{AbcConfig, ArtificialBeeColony};
use evorl_evolution::algorithms::swarm::aco_r::{AcoRConfig, AntColonyReal};
use evorl_evolution::algorithms::swarm::bat::{BatAlgorithm, BatConfig};
use evorl_evolution::algorithms::swarm::cuckoo::{CuckooConfig, CuckooSearch};
use evorl_evolution::algorithms::swarm::firefly::{FireflyAlgorithm, FireflyConfig};
use evorl_evolution::algorithms::swarm::gwo::{GreyWolfOptimizer, GwoConfig};
use evorl_evolution::algorithms::swarm::pso::{ParticleSwarm, PsoConfig};
use evorl_evolution::algorithms::swarm::salp::{SalpConfig, SalpSwarm};
use evorl_evolution::algorithms::swarm::woa::{WhaleOptimization, WoaConfig};
use evorl_evolution::fitness::FromFitnessEvaluable;
use evorl_evolution::strategy::{EvolutionaryHarness, Strategy};

type B = NdArray;

struct Sphere;
struct SphereFit;
impl FitnessEvaluable for SphereFit {
    type Individual = Vec<f64>;
    type Landscape = Sphere;
    fn evaluate(&self, x: &Self::Individual, _: &Self::Landscape) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
}

/// Drive a strategy through `gens` generations and collect the
/// per-generation best-fitness trajectory.
fn run<S>(
    strategy: S,
    params: S::Params,
    seed: u64,
    gens: usize,
) -> Vec<f32>
where
    S: Strategy<B>,
    <S as Strategy<B>>::Genome: 'static,
    FromFitnessEvaluable<SphereFit, Sphere>:
        evorl_evolution::fitness::BatchFitnessFn<B, S::Genome>,
{
    let device = Default::default();
    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        strategy,
        params,
        FromFitnessEvaluable::new(SphereFit, Sphere),
        seed,
        device,
        gens,
    );
    harness.reset();
    let mut trajectory = Vec::with_capacity(gens);
    loop {
        let step = harness.step(());
        trajectory.push(harness.latest_metrics().unwrap().best_fitness);
        if step.done {
            break;
        }
    }
    trajectory
}

fn run_ga(seed: u64, gens: usize) -> Vec<f32> {
    let params = GaConfig {
        pop_size: 32,
        genome_dim: 5,
        bounds: (-5.0, 5.0),
        mutation_sigma: 0.2,
        selection: GaSelection::Tournament { size: 2 },
        crossover: GaCrossover::BlxAlpha { alpha: 0.5 },
        replacement: GaReplacement::Elitist { elitism_k: 1 },
    };
    run(GeneticAlgorithm::<B>::new(), params, seed, gens)
}

fn run_es(seed: u64, gens: usize, kind: EsKind) -> Vec<f32> {
    let params = EsConfig::default_for(kind, 5);
    run(EvolutionStrategy::<B>::new(), params, seed, gens)
}

fn run_pso(seed: u64, gens: usize) -> Vec<f32> {
    let params = PsoConfig::default_for(16, 5);
    run(ParticleSwarm::<B>::new(), params, seed, gens)
}

fn run_gwo(seed: u64, gens: usize) -> Vec<f32> {
    let params = GwoConfig::default_for(16, 5);
    run(GreyWolfOptimizer::<B>::new(), params, seed, gens)
}

fn run_woa(seed: u64, gens: usize) -> Vec<f32> {
    let params = WoaConfig::default_for(16, 5);
    run(WhaleOptimization::<B>::new(), params, seed, gens)
}

fn run_salp(seed: u64, gens: usize) -> Vec<f32> {
    let params = SalpConfig::default_for(16, 5);
    run(SalpSwarm::<B>::new(), params, seed, gens)
}

fn run_abc(seed: u64, gens: usize) -> Vec<f32> {
    let params = AbcConfig::default_for(12, 5);
    run(ArtificialBeeColony::<B>::new(), params, seed, gens)
}

fn run_bat(seed: u64, gens: usize) -> Vec<f32> {
    let params = BatConfig::default_for(16, 5);
    run(BatAlgorithm::<B>::new(), params, seed, gens)
}

fn run_aco_r(seed: u64, gens: usize) -> Vec<f32> {
    let params = AcoRConfig::default_for(16, 8, 5);
    run(AntColonyReal::<B>::new(), params, seed, gens)
}

fn run_cuckoo(seed: u64, gens: usize) -> Vec<f32> {
    let params = CuckooConfig::default_for(16, 5);
    run(CuckooSearch::<B>::new(), params, seed, gens)
}

fn run_firefly(seed: u64, gens: usize) -> Vec<f32> {
    let params = FireflyConfig::default_for(16, 5);
    run(FireflyAlgorithm::<B>::new(), params, seed, gens)
}

#[test]
fn same_seed_same_generations() {
    const SEED: u64 = 1_234_567;
    const GENS: usize = 30;

    // Classical EAs.
    let ga_a = run_ga(SEED, GENS);
    let ga_b = run_ga(SEED, GENS);
    assert_eq!(ga_a, ga_b, "GA trajectories diverge under the same seed");

    for kind in [
        EsKind::OnePlusOne,
        EsKind::OnePlusLambda { lambda: 6 },
        EsKind::MuPlusLambda { mu: 3, lambda: 9 },
        EsKind::MuCommaLambda { mu: 3, lambda: 9 },
    ] {
        let a = run_es(SEED, GENS, kind);
        let b = run_es(SEED, GENS, kind);
        assert_eq!(
            a, b,
            "ES trajectories diverge under the same seed for {kind:?}"
        );
    }

    // Swarm strategies. ACO-Permutation excluded (stub).
    macro_rules! check {
        ($fn:ident, $name:expr) => {
            let a = $fn(SEED, GENS);
            let b = $fn(SEED, GENS);
            assert_eq!(
                a, b,
                "{} trajectories diverge under the same seed",
                $name
            );
        };
    }
    check!(run_pso, "PSO");
    check!(run_gwo, "GWO");
    check!(run_woa, "WOA");
    check!(run_salp, "SSA");
    check!(run_abc, "ABC");
    check!(run_bat, "Bat");
    check!(run_aco_r, "ACO_R");
    check!(run_cuckoo, "Cuckoo");
    check!(run_firefly, "Firefly");
}
