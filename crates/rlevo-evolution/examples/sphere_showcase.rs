//! Runs every real-valued strategy in the crate on the Sphere-D10
//! landscape and prints a convergence summary.
//!
//! Run with `cargo run --release -p evorl-evolution --example sphere_showcase`.

use burn::backend::NdArray;
use evorl_benchmarks::agent::FitnessEvaluable;
use evorl_benchmarks::env::BenchEnv;
use evorl_evolution::algorithms::de::{DeConfig, DeVariant, DifferentialEvolution};
use evorl_evolution::algorithms::ep::{EpConfig, EvolutionaryProgramming};
use evorl_evolution::algorithms::es_classical::{EsConfig, EsKind, EvolutionStrategy};
use evorl_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use evorl_evolution::fitness::FromFitnessEvaluable;
use evorl_evolution::strategy::EvolutionaryHarness;

type B = NdArray;

const DIM: usize = 10;
const GENS: usize = 500;
const SEED: u64 = 42;

#[derive(Debug)]
struct Sphere;
struct SphereFit;
impl FitnessEvaluable for SphereFit {
    type Individual = Vec<f64>;
    type Landscape = Sphere;
    fn evaluate(&self, x: &Self::Individual, _: &Self::Landscape) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
}

fn run<S, F>(label: &str, strategy: S, params: S::Params, fitness_fn: F)
where
    S: evorl_evolution::strategy::Strategy<B>,
    S::Params: std::fmt::Debug,
    F: evorl_evolution::fitness::BatchFitnessFn<B, S::Genome>,
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
            bounds: (-5.12, 5.12),
            mutation_sigma: 0.2,
            selection: GaSelection::Tournament { size: 2 },
            crossover: GaCrossover::BlxAlpha { alpha: 0.5 },
            replacement: GaReplacement::Elitist { elitism_k: 2 },
        },
        FromFitnessEvaluable::new(SphereFit, Sphere),
    );

    for kind in [
        EsKind::OnePlusOne,
        EsKind::OnePlusLambda { lambda: 8 },
        EsKind::MuPlusLambda { mu: 5, lambda: 20 },
        EsKind::MuCommaLambda { mu: 5, lambda: 20 },
    ] {
        let label = format!("ES/{kind:?}");
        run(
            &label,
            EvolutionStrategy::<B>::new(),
            EsConfig::default_for(kind, DIM),
            FromFitnessEvaluable::new(SphereFit, Sphere),
        );
    }

    run(
        "EP/fogel/μ=20/q=10",
        EvolutionaryProgramming::<B>::new(),
        EpConfig::default_for(20, DIM),
        FromFitnessEvaluable::new(SphereFit, Sphere),
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
        let label = format!("DE/{variant:?}");
        run(
            &label,
            DifferentialEvolution::<B>::new(),
            params,
            FromFitnessEvaluable::new(SphereFit, Sphere),
        );
    }
}
