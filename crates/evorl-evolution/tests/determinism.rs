//! Determinism integration test.
//!
//! Spec §12.4: on the ndarray backend, running a strategy twice with the
//! same `base_seed` must produce a bit-identical fitness trajectory.
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
use evorl_evolution::fitness::FromFitnessEvaluable;
use evorl_evolution::strategy::EvolutionaryHarness;

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

fn run_ga(seed: u64, gens: usize) -> Vec<f32> {
    let device = Default::default();
    let params = GaConfig {
        pop_size: 32,
        genome_dim: 5,
        bounds: (-5.0, 5.0),
        mutation_sigma: 0.2,
        selection: GaSelection::Tournament { size: 2 },
        crossover: GaCrossover::BlxAlpha { alpha: 0.5 },
        replacement: GaReplacement::Elitist { elitism_k: 1 },
    };
    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        GeneticAlgorithm::<B>::new(),
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

fn run_es(seed: u64, gens: usize, kind: EsKind) -> Vec<f32> {
    let device = Default::default();
    let params = EsConfig::default_for(kind, 5);
    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        EvolutionStrategy::<B>::new(),
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

#[test]
fn same_seed_same_generations() {
    const SEED: u64 = 1_234_567;
    const GENS: usize = 50;

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
}
