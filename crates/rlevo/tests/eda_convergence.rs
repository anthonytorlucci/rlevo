//! EDA convergence gates (cross-crate).
//!
//! Two acceptance criteria from the Phase 3b spec (issue #31):
//!
//! 1. Every `ProbabilityModel` implementation drives `best_fitness_ever`
//!    to `<= 0.01` on Sphere-D10 within 500 generations
//!    (`pop_size = 50`, `selection_ratio = 0.5`).
//! 2. MIMIC ([`DependencyChain`]) achieves a strictly lower *median*
//!    final fitness than UMDA ([`UnivariateGaussian`]) on Rosenbrock-D10
//!    over nine fixed seeds — the dependency-chain model must exploit
//!    Rosenbrock's adjacent-variable coupling that a univariate model
//!    cannot represent.
//!
//! Runs are bit-deterministic per seed (host-RNG sampling only), so
//! these gates cannot flake run-to-run on a given platform.

use burn::backend::Flex;

use rlevo_core::fitness::Landscape;
use rlevo_environments::landscapes::rosenbrock::Rosenbrock;
use rlevo_environments::landscapes::sphere::Sphere;
use rlevo_evolution::algorithms::eda::{
    CompactGeneticParams, DependencyChainParams, UnivariateBernoulliParams,
    UnivariateGaussianParams,
};
use rlevo_evolution::fitness::FromLandscape;
use rlevo_evolution::strategy::EvolutionaryHarness;
use rlevo_evolution::{
    CompactGenetic, DependencyChain, EdaParams, EdaStrategy, ProbabilityModel,
    UnivariateBernoulli, UnivariateGaussian,
};

type B = Flex;

const DIM: usize = 10;
const POP_SIZE: usize = 50;
const SELECTION_RATIO: f32 = 0.5;
const GENS: usize = 500;

/// Drives one EDA model for `GENS` generations and returns the final
/// `best_fitness_ever`.
fn run<M, L>(
    model: M,
    model_params: M::Params,
    selection_ratio: f32,
    bounds: Option<(f32, f32)>,
    landscape: L,
    seed: u64,
) -> f32
where
    M: ProbabilityModel<B> + 'static,
    L: Landscape + 'static,
{
    let device = Default::default();
    let params = EdaParams {
        pop_size: POP_SIZE,
        selection_ratio,
        bounds,
        model: model_params,
    };
    let mut harness = EvolutionaryHarness::<B, EdaStrategy<B, M>, _>::new(
        EdaStrategy::new(model),
        params,
        FromLandscape::new(landscape),
        seed,
        device,
        GENS,
    );
    harness.reset();
    loop {
        if harness.step(()).done {
            break;
        }
    }
    harness.latest_metrics().unwrap().best_fitness_ever
}

/// Median of a small sample; NaN-free by construction (fitness is
/// sanitized inside `EdaStrategy::tell`).
fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

#[test]
fn all_four_models_reach_sphere_threshold() {
    const SEED: u64 = 42;
    const TARGET: f32 = 0.01;
    let bounds = Some((-5.12, 5.12));

    let umda = run(
        UnivariateGaussian,
        UnivariateGaussianParams::default_for(DIM),
        SELECTION_RATIO,
        bounds,
        Sphere::new(DIM),
        SEED,
    );
    assert!(
        umda <= TARGET,
        "UMDA best_fitness_ever on Sphere-D10 after {GENS} gens = {umda}"
    );

    // PBIL and cGA emit raw {0,1} genes; the all-zeros genome scores
    // exactly 0.0 on Sphere, so the gate stays meaningful without bounds.
    let pbil = run(
        UnivariateBernoulli,
        UnivariateBernoulliParams::default_for(DIM),
        SELECTION_RATIO,
        None,
        Sphere::new(DIM),
        SEED,
    );
    assert!(
        pbil <= TARGET,
        "PBIL best_fitness_ever on Sphere-D10 after {GENS} gens = {pbil}"
    );

    let cga = run(
        CompactGenetic,
        CompactGeneticParams::default_for(DIM),
        SELECTION_RATIO,
        None,
        Sphere::new(DIM),
        SEED,
    );
    assert!(
        cga <= TARGET,
        "cGA best_fitness_ever on Sphere-D10 after {GENS} gens = {cga}"
    );

    let mimic = run(
        DependencyChain,
        DependencyChainParams::default_for(DIM),
        SELECTION_RATIO,
        bounds,
        Sphere::new(DIM),
        SEED,
    );
    assert!(
        mimic <= TARGET,
        "MIMIC best_fitness_ever on Sphere-D10 after {GENS} gens = {mimic}"
    );
}

#[test]
fn mimic_beats_umda_median_on_rosenbrock() {
    const SEEDS: [u64; 9] = [101, 202, 303, 404, 505, 606, 707, 808, 909];
    // Calibrated configuration (identical for both models). Rationale:
    //
    // - De Jong domain ±2.048 rather than Rosenbrock's native ±30: the
    //   narrower box keeps early generations out of the 1e8 plateau so
    //   the model-quality difference, not the initial spread, drives the
    //   gap.
    // - `init_mean = 0.5`, `init_std = 0.5`: starts the prior on the limb
    //   of Rosenbrock's `x_{i+1} ≈ x_i²` valley, where the coupling is
    //   locally *linear*. Centered at the origin the parabola's tangent is
    //   flat, Pearson correlation vanishes, and a Gaussian chain has
    //   nothing to model — the comparison would measure noise.
    // - `min_variance = 1e-3` keeps both models exploring instead of
    //   collapsing onto the valley floor before traversing it.
    // - `selection_ratio = 0.3` sharpens selection so the elite set lies
    //   inside the valley, where the chain correlations are real.
    const RATIO: f32 = 0.3;
    const INIT_MEAN: f32 = 0.5;
    const INIT_STD: f32 = 0.5;
    const MIN_VARIANCE: f32 = 1e-3;
    let bounds = Some((-2.048, 2.048));

    let mut umda: Vec<f32> = SEEDS
        .iter()
        .map(|&seed| {
            let mut p = UnivariateGaussianParams::default_for(DIM);
            p.init_mean = INIT_MEAN;
            p.init_std = INIT_STD;
            p.min_variance = MIN_VARIANCE;
            run(
                UnivariateGaussian,
                p,
                RATIO,
                bounds,
                Rosenbrock::new(DIM),
                seed,
            )
        })
        .collect();
    let mut mimic: Vec<f32> = SEEDS
        .iter()
        .map(|&seed| {
            let mut p = DependencyChainParams::default_for(DIM);
            p.init_mean = INIT_MEAN;
            p.init_std = INIT_STD;
            p.min_variance = MIN_VARIANCE;
            run(
                DependencyChain,
                p,
                RATIO,
                bounds,
                Rosenbrock::new(DIM),
                seed,
            )
        })
        .collect();

    let umda_median = median(&mut umda);
    let mimic_median = median(&mut mimic);
    assert!(
        mimic_median < umda_median,
        "MIMIC median final fitness ({mimic_median}) must be strictly below \
         UMDA's ({umda_median}) on Rosenbrock-D10; per-seed UMDA {umda:?}, \
         MIMIC {mimic:?}"
    );
}
