//! Bit-exact determinism for the EDA strategies (cross-crate).
//!
//! Runs each of the five `ProbabilityModel` implementations twice from
//! the same seed on the real [`Sphere`] landscape from
//! `rlevo-environments` and asserts the per-generation best-fitness
//! trajectories are bit-identical. EDA sampling draws exclusively from
//! host-side [`SeedPurpose::EdaSampling`] streams, so equal seeds must
//! produce equal trajectories on any backend.
//!
//! [`SeedPurpose::EdaSampling`]: rlevo_evolution::rng::SeedPurpose::EdaSampling

use burn::backend::Flex;

use rlevo_environments::landscapes::sphere::Sphere;
use rlevo_evolution::algorithms::eda::{
    BayesianNetworkParams, CompactGeneticParams, DependencyChainParams,
    UnivariateBernoulliParams, UnivariateGaussianParams,
};
use rlevo_core::bounds::Bounds;
use rlevo_core::objective::ObjectiveSense;
use rlevo_evolution::fitness::FromLandscape;
use rlevo_evolution::strategy::EvolutionaryHarness;
use rlevo_evolution::{
    BayesianNetwork, CompactGenetic, DependencyChain, EdaParams, EdaStrategy,
    ProbabilityModel, UnivariateBernoulli, UnivariateGaussian,
};

type B = Flex;

const DIM: usize = 10;
const SEED: u64 = 1_234_567;
const GENS: usize = 30;
const BOUNDS: Option<Bounds> = Some(Bounds::new(-5.12, 5.12));

/// Drives one EDA model through `GENS` generations on Sphere-D10 and
/// collects the per-generation best-fitness trajectory.
fn run<M>(model: M, model_params: M::Params, bounds: Option<Bounds>) -> Vec<f32>
where
    M: ProbabilityModel<B> + 'static,
{
    let device = Default::default();
    let params = EdaParams {
        pop_size: 32,
        selection_ratio: 0.5,
        bounds,
        model: model_params,
    };
    let mut harness = EvolutionaryHarness::<B, EdaStrategy<B, M>, _>::new(
        EdaStrategy::new(model),
        params,
        FromLandscape::with_sense(Sphere::new(DIM), ObjectiveSense::Minimize),
        SEED,
        device,
        GENS,
    ).expect("valid params");
    harness.reset();
    let mut trajectory = Vec::with_capacity(GENS);
    loop {
        let step = harness.step(());
        trajectory.push(harness.latest_metrics().unwrap().best_fitness());
        if step.done {
            break;
        }
    }
    trajectory
}

/// Single test function on purpose: determinism assertions must not
/// interleave with other tests that could perturb shared backend state
/// (same convention as `rlevo-evolution/tests/determinism.rs`).
#[test]
fn eda_same_seed_same_trajectories() {
    // UMDA — univariate Gaussian.
    let a = run(
        UnivariateGaussian,
        UnivariateGaussianParams::default_for(DIM),
        BOUNDS,
    );
    let b = run(
        UnivariateGaussian,
        UnivariateGaussianParams::default_for(DIM),
        BOUNDS,
    );
    assert!(a.iter().all(|f| f.is_finite()));
    assert_eq!(
        a, b,
        "UnivariateGaussian (UMDA) trajectories diverge under the same seed"
    );

    // PBIL — univariate Bernoulli; raw {0,1} genes, bounds unused.
    let a = run(
        UnivariateBernoulli,
        UnivariateBernoulliParams::default_for(DIM),
        None,
    );
    let b = run(
        UnivariateBernoulli,
        UnivariateBernoulliParams::default_for(DIM),
        None,
    );
    assert!(a.iter().all(|f| f.is_finite()));
    assert_eq!(
        a, b,
        "UnivariateBernoulli (PBIL) trajectories diverge under the same seed"
    );

    // cGA — compact genetic algorithm; raw {0,1} genes, bounds unused.
    let a = run(
        CompactGenetic,
        CompactGeneticParams::default_for(DIM),
        None,
    );
    let b = run(
        CompactGenetic,
        CompactGeneticParams::default_for(DIM),
        None,
    );
    assert!(a.iter().all(|f| f.is_finite()));
    assert_eq!(
        a, b,
        "CompactGenetic (cGA) trajectories diverge under the same seed"
    );

    // MIMIC — Gaussian dependency chain.
    let a = run(
        DependencyChain,
        DependencyChainParams::default_for(DIM),
        BOUNDS,
    );
    let b = run(
        DependencyChain,
        DependencyChainParams::default_for(DIM),
        BOUNDS,
    );
    assert!(a.iter().all(|f| f.is_finite()));
    assert_eq!(
        a, b,
        "DependencyChain (MIMIC) trajectories diverge under the same seed"
    );

    // BOA — Bayesian network; raw {0,1} genes, bounds unused.
    let a = run(
        BayesianNetwork,
        BayesianNetworkParams::default_for(DIM),
        None,
    );
    let b = run(
        BayesianNetwork,
        BayesianNetworkParams::default_for(DIM),
        None,
    );
    assert!(a.iter().all(|f| f.is_finite()));
    assert_eq!(
        a, b,
        "BayesianNetwork (BOA) trajectories diverge under the same seed"
    );
}
