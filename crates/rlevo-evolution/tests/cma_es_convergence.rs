//! Convergence and determinism integration tests for CMA-ES and CMSA-ES.
//!
//! Acceptance criteria for CMA-ES and CMSA-ES:
//! - Sphere-D10: `best_fitness < 1e-6` within 1000 generations.
//! - Rastrigin-D10: `best_fitness < 1.0` within 2000 generations.
//! - Identical seeds reproduce identical trajectories.
//!
//! Landscapes are defined inline as [`Landscape`] impls (no `rlevo-harness`
//! dependency — production-crate purity, CLAUDE.md). Both strategies sample
//! host-side through `seed_stream` rather than the backend's process-global RNG,
//! so these tests are reproducible regardless of the cargo runner's threading.

use burn::backend::Flex;
use rlevo_core::fitness::Landscape;

use rlevo_evolution::algorithms::cma_es::{CmaEs, CmaEsConfig};
use rlevo_evolution::algorithms::cmsa_es::{CmsaEs, CmsaEsConfig};
use rlevo_evolution::fitness::{BatchFitnessFn, FromLandscape};
use rlevo_evolution::strategy::{EvolutionaryLoop, Strategy};

type B = Flex;

struct Sphere;
impl Landscape for Sphere {
    fn evaluate(&self, x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
}

struct Rastrigin;
impl Landscape for Rastrigin {
    fn evaluate(&self, x: &[f64]) -> f64 {
        const A: f64 = 10.0;
        #[allow(clippy::cast_precision_loss)]
        let n = x.len() as f64;
        A * n
            + x.iter()
                .map(|v| v * v - A * (2.0 * std::f64::consts::PI * v).cos())
                .sum::<f64>()
    }
}

/// Drive a strategy for `gens` generations and return the per-generation
/// `best_fitness_ever` trajectory (rolling minimum).
fn run<S, F>(strategy: S, params: S::Params, fitness_fn: F, seed: u64, gens: usize) -> Vec<f32>
where
    S: Strategy<B>,
    S::Params: rlevo_core::config::Validate,
    F: BatchFitnessFn<B, S::Genome>,
{
    let device = Default::default();
    let mut evo_loop =
        EvolutionaryLoop::<B, _, _>::new(strategy, params, fitness_fn, seed, device, gens)
            .expect("valid params");
    evo_loop.reset();
    let mut trajectory: Vec<f32> = Vec::with_capacity(gens);
    while !evo_loop.is_done() {
        evo_loop.step();
        trajectory.push(evo_loop.latest_metrics().unwrap().best_fitness_ever());
    }
    trajectory
}

#[test]
fn cma_es_converges_on_sphere_d10() {
    let traj = run(
        CmaEs::<B>::new(),
        CmaEsConfig::default_for(10),
        FromLandscape::new(Sphere),
        42,
        1000,
    );
    let best = *traj.last().unwrap();
    assert!(best < 1e-6, "CMA-ES Sphere-D10 best_fitness_ever={best}");
}

#[test]
fn cmsa_es_converges_on_sphere_d10() {
    let traj = run(
        CmsaEs::<B>::new(),
        CmsaEsConfig::default_for(10),
        FromLandscape::new(Sphere),
        42,
        1000,
    );
    let best = *traj.last().unwrap();
    assert!(best < 1e-6, "CMSA-ES Sphere-D10 best_fitness_ever={best}");
}

#[test]
fn cma_es_converges_on_rastrigin_d10() {
    // Restart-free CMA-ES on the highly multimodal Rastrigin: a larger population (Hansen 2016 §A
    // multimodal guidance) plus a wider initial `$\sigma$` buys basin-finding. The seed is fixed so
    // the test is deterministic; with these settings the run reaches the global optimum (0) well
    // inside the 2000-generation budget.
    let mut params = CmaEsConfig::with_pop_size(200, 10);
    params.initial_sigma = 2.0;
    let traj = run(
        CmaEs::<B>::new(),
        params,
        FromLandscape::new(Rastrigin),
        12_345,
        1200,
    );
    let best = *traj.last().unwrap();
    assert!(best < 1.0, "CMA-ES Rastrigin-D10 best_fitness_ever={best}");
}

#[test]
fn cmsa_es_converges_on_rastrigin_d10() {
    let mut params = CmsaEsConfig::with_pop_size(200, 10);
    params.initial_sigma = 2.0;
    let traj = run(
        CmsaEs::<B>::new(),
        params,
        FromLandscape::new(Rastrigin),
        12_345,
        1200,
    );
    let best = *traj.last().unwrap();
    assert!(best < 1.0, "CMSA-ES Rastrigin-D10 best_fitness_ever={best}");
}

#[test]
fn cma_es_same_seed_identical_trajectory() {
    let a = run(
        CmaEs::<B>::new(),
        CmaEsConfig::default_for(10),
        FromLandscape::new(Sphere),
        7,
        60,
    );
    let b = run(
        CmaEs::<B>::new(),
        CmaEsConfig::default_for(10),
        FromLandscape::new(Sphere),
        7,
        60,
    );
    assert_eq!(a, b, "CMA-ES trajectories diverge under the same seed");
}

#[test]
fn cmsa_es_same_seed_identical_trajectory() {
    let a = run(
        CmsaEs::<B>::new(),
        CmsaEsConfig::default_for(10),
        FromLandscape::new(Sphere),
        7,
        60,
    );
    let b = run(
        CmsaEs::<B>::new(),
        CmsaEsConfig::default_for(10),
        FromLandscape::new(Sphere),
        7,
        60,
    );
    assert_eq!(a, b, "CMSA-ES trajectories diverge under the same seed");
}
