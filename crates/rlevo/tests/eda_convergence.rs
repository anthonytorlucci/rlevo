//! EDA convergence gates (cross-crate).
//!
//! Two convergence criteria (issue #31), plus a follow-up discriminating
//! gate (issue #37):
//!
//! 1. Every `ProbabilityModel` implementation drives `best_fitness_ever`
//!    to `<= 0.01` on Sphere-D10 within 500 generations
//!    (`pop_size = 50`, `selection_ratio = 0.5`).
//! 2. MIMIC ([`DependencyChain`]) achieves a strictly lower *median*
//!    final fitness than UMDA ([`UnivariateGaussian`]) on Rosenbrock-D10
//!    over nine fixed seeds — the dependency-chain model must exploit
//!    Rosenbrock's adjacent-variable coupling that a univariate model
//!    cannot represent.
//! 3. BOA ([`BayesianNetwork`]) solves the deceptive
//!    [`ConcatenatedTrap`] trap-5 × 4 (cost 0, all-ones) while UMDA and
//!    MIMIC stall near the all-zeros deceptive basin (cost ≈ 4) — only a
//!    model capturing the order-5 intra-block linkage can sample and
//!    keep solved blocks.
//!
//! Runs are bit-deterministic per seed (host-RNG sampling only), so
//! these gates cannot flake run-to-run on a given platform.

use burn::backend::Flex;

use rlevo_core::fitness::Landscape;
use rlevo_environments::landscapes::concatenated_trap::ConcatenatedTrap;
use rlevo_environments::landscapes::rosenbrock::Rosenbrock;
use rlevo_environments::landscapes::sphere::Sphere;
use rlevo_evolution::algorithms::eda::{
    BayesianNetworkParams, CompactGeneticParams, DependencyChainParams,
    UnivariateBernoulliParams, UnivariateGaussianParams,
};
use rlevo_evolution::fitness::FromLandscape;
use rlevo_evolution::strategy::EvolutionaryHarness;
use rlevo_evolution::{
    BayesianNetwork, CompactGenetic, DependencyChain, EdaParams, EdaStrategy,
    ProbabilityModel, UnivariateBernoulli, UnivariateGaussian,
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
    run_config(
        model,
        model_params,
        POP_SIZE,
        selection_ratio,
        bounds,
        landscape,
        GENS,
        seed,
    )
}

/// Fully-parameterised runner: the trap gate needs a larger population
/// and a shorter budget than the file-level Sphere/Rosenbrock constants.
#[allow(clippy::too_many_arguments)] // mirrors the EdaParams + harness knobs 1:1
fn run_config<M, L>(
    model: M,
    model_params: M::Params,
    pop_size: usize,
    selection_ratio: f32,
    bounds: Option<(f32, f32)>,
    landscape: L,
    gens: usize,
    seed: u64,
) -> f32
where
    M: ProbabilityModel<B> + 'static,
    L: Landscape + 'static,
{
    let device = Default::default();
    let params = EdaParams {
        pop_size,
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
        gens,
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
fn all_five_models_reach_sphere_threshold() {
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

    // BOA also emits raw {0,1} genes; the all-zeros genome scores
    // exactly 0.0 on Sphere (degenerate but NaN-free sanity gate).
    let boa = run(
        BayesianNetwork,
        BayesianNetworkParams::default_for(DIM),
        SELECTION_RATIO,
        None,
        Sphere::new(DIM),
        SEED,
    );
    assert!(
        boa <= TARGET,
        "BOA best_fitness_ever on Sphere-D10 after {GENS} gens = {boa}"
    );
}

/// Issue #37 discriminating gate: BOA solves the deceptive trap-5 × 4
/// (`dim = 20`) to cost 0 (all-ones) while UMDA and MIMIC stall at/near
/// the all-zeros basin (cost ≈ 4 = `num_blocks`).
///
/// Configuration pinned by the calibration sweep recorded in ADR 0018:
///
/// - `pop_size = 2000`, `selection_ratio = 0.3`, 60 generations,
///   `max_parents = 3` (the `BayesianNetworkParams` default).
/// - Both knobs are load-bearing for BOA. The BIC edge gain grows with
///   the selected-row count `N` while the penalty grows like `ln N`, so
///   the intra-block edges only clear the penalty when thousands of rows
///   are selected; and `0.3` truncation enriches solved-block carriers
///   fast enough that the structure is learned before the deceptive
///   per-gene gradient collapses the population onto all-zeros. At
///   `pop = 600` (any ratio) or `ratio = 0.5` (any pop ≤ 4000) BOA
///   solved 0–4 of 10 calibration seeds; at this config it solved 10/10.
/// - UMDA/MIMIC run the *same* budget with a symmetric prior over the
///   binary box (`init_mean = 0.5`, `init_std = 0.5`, bounds `(0, 1)`);
///   their failure is structural — a univariate (or first-order-chain)
///   model cannot represent the order-5 intra-block linkage — not a
///   budget artifact. Calibration medians: UMDA 3.0, MIMIC 3.0.
#[test]
fn boa_solves_trap_where_umda_and_mimic_stall() {
    const SEEDS: [u64; 5] = [11, 22, 33, 44, 55];
    const TRAP_POP: usize = 2000;
    const TRAP_RATIO: f32 = 0.3;
    const TRAP_GENS: usize = 60;
    let trap = ConcatenatedTrap::new(4, 5); // trap-5 × 4 blocks, dim 20

    let mut boa: Vec<f32> = SEEDS
        .iter()
        .map(|&seed| {
            run_config(
                BayesianNetwork,
                BayesianNetworkParams::default_for(trap.dim()),
                TRAP_POP,
                TRAP_RATIO,
                None,
                trap,
                TRAP_GENS,
                seed,
            )
        })
        .collect();
    let mut umda: Vec<f32> = SEEDS
        .iter()
        .map(|&seed| {
            let mut p = UnivariateGaussianParams::default_for(trap.dim());
            p.init_mean = 0.5;
            p.init_std = 0.5;
            run_config(
                UnivariateGaussian,
                p,
                TRAP_POP,
                TRAP_RATIO,
                Some((0.0, 1.0)),
                trap,
                TRAP_GENS,
                seed,
            )
        })
        .collect();
    let mut mimic: Vec<f32> = SEEDS
        .iter()
        .map(|&seed| {
            let mut p = DependencyChainParams::default_for(trap.dim());
            p.init_mean = 0.5;
            p.init_std = 0.5;
            run_config(
                DependencyChain,
                p,
                TRAP_POP,
                TRAP_RATIO,
                Some((0.0, 1.0)),
                trap,
                TRAP_GENS,
                seed,
            )
        })
        .collect();

    let boa_median = median(&mut boa);
    let umda_median = median(&mut umda);
    let mimic_median = median(&mut mimic);

    // Trap cost is integer-valued, so 0.0 is exact in f32.
    assert!(
        boa_median == 0.0,
        "BOA must reach the all-ones optimum (median cost 0) on trap-5×4; \
         got median {boa_median}, per-seed {boa:?}"
    );
    // ">= 2.0" rather than "== 4.0": a lucky block may be solved, but the
    // univariate/pairwise models must remain deceived on most blocks.
    assert!(
        umda_median >= 2.0,
        "UMDA is expected to stall near the all-zeros basin (cost ≈ 4) on \
         trap-5×4; got median {umda_median}, per-seed {umda:?}"
    );
    assert!(
        mimic_median >= 2.0,
        "MIMIC's first-order chain cannot capture order-5 linkage and must \
         stall near cost 4 on trap-5×4; got median {mimic_median}, per-seed \
         {mimic:?}"
    );
    assert!(
        boa_median < umda_median && boa_median < mimic_median,
        "BOA (median {boa_median}) must beat both UMDA ({umda_median}) and \
         MIMIC ({mimic_median}) on trap-5×4"
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
