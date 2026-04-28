//! Runs every real-valued strategy in the crate on the Sphere-D10
//! landscape and prints a convergence summary.
//!
//! Run with `cargo run --release -p rlevo-evolution --example sphere_showcase`.

use burn::backend::NdArray;
use rlevo_environments::landscapes::sphere::Sphere;
use rlevo_evolution::algorithms::de::{DeConfig, DeVariant, DifferentialEvolution};
use rlevo_evolution::algorithms::ep::{EpConfig, EvolutionaryProgramming};
use rlevo_evolution::algorithms::es_classical::{EsConfig, EsKind, EvolutionStrategy};
use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_evolution::fitness::{BatchFitnessFn, FromLandscape};
use rlevo_evolution::strategy::{EvolutionaryHarness, Strategy};

type B = NdArray;

const DIM: usize = 10;
const GENS: usize = 500;
const SEED: u64 = 42;

fn run<S, F>(label: &str, strategy: S, params: S::Params, fitness_fn: F)
where
    S: Strategy<B>,
    S::Params: std::fmt::Debug,
    F: BatchFitnessFn<B, S::Genome>,
    B: burn::tensor::backend::Backend,
    <B as burn::tensor::backend::Backend>::Device: Clone + Default,
{
    let device = Default::default();
    let mut harness =
        EvolutionaryHarness::<B, S, F>::new(strategy, params, fitness_fn, SEED, device, GENS);
    harness.reset();
    loop {
        if harness.step(()).done {
            break;
        }
    }
    let m = harness
        .latest_metrics()
        .expect("at least one generation ran");
    println!(
        "{label:>30} | gens={:>4} | best={:>.6e} | mean={:>.6e}",
        m.generation, m.best_fitness_ever, m.mean_fitness,
    );
}

fn main() {
    let landscape = Sphere::new(DIM);
    let (lo, hi) = landscape.bounds();
    #[allow(clippy::cast_possible_truncation)]
    let bounds = (lo as f32, hi as f32);

    println!(
        "Sphere-D{DIM} showcase — {GENS} generations, ndarray backend, seed={SEED}\n\
         {:-<80}",
        "",
    );

    run(
        "GA/real/tournament-blx-elite",
        GeneticAlgorithm::<B>::new(),
        GaConfig {
            pop_size: 64,
            genome_dim: DIM,
            bounds,
            mutation_sigma: 0.2,
            selection: GaSelection::Tournament { size: 2 },
            crossover: GaCrossover::BlxAlpha { alpha: 0.5 },
            replacement: GaReplacement::Elitist { elitism_k: 2 },
        },
        FromLandscape::new(landscape),
    );

    for kind in [
        EsKind::OnePlusOne,
        EsKind::OnePlusLambda { lambda: 8 },
        EsKind::MuPlusLambda { mu: 5, lambda: 20 },
        EsKind::MuCommaLambda { mu: 5, lambda: 20 },
    ] {
        let mut params = EsConfig::default_for(kind, DIM);
        params.bounds = bounds;
        let label = format!("ES/{kind:?}");
        run(
            &label,
            EvolutionStrategy::<B>::new(),
            params,
            FromLandscape::new(landscape),
        );
    }

    let mut ep_params = EpConfig::default_for(20, DIM);
    ep_params.bounds = bounds;
    run(
        "EP/fogel/μ=20/q=10",
        EvolutionaryProgramming::<B>::new(),
        ep_params,
        FromLandscape::new(landscape),
    );

    for variant in [
        DeVariant::Rand1Bin,
        DeVariant::Best1Bin,
        DeVariant::CurrentToBest1Bin,
        DeVariant::Rand2Bin,
        DeVariant::Rand1Exp,
    ] {
        let mut params = DeConfig::default_for(30, DIM);
        params.variant = variant;
        params.bounds = bounds;
        let label = format!("DE/{variant:?}");
        run(
            &label,
            DifferentialEvolution::<B>::new(),
            params,
            FromLandscape::new(landscape),
        );
    }
}
