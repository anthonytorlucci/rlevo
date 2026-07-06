//! Shared harness for the landscape evolution showcases.
//!
//! Each `*_showcase` example wires one landscape into [`showcase`], which runs
//! every real-valued strategy in `rlevo-evolution` (GA, the four classical ES
//! kinds, EP, and five DE variants) for a fixed generation budget and prints a
//! one-line convergence summary per strategy. Keeping the driver here means the
//! per-landscape files stay down to a few lines that differ only in the
//! landscape, its dimensionality, and the GA mutation scale.
//!
//! This file is a module shared by the example binaries, not an example itself.
//!
//! # Reading the output
//!
//! Every strategy prints one row in the form:
//!
//! ```text
//!          ES/OnePlusOne | gens= 500 | best=1.234567e-3 | mean=4.567890e-2
//! ```
//!
//! - **label** — the strategy and its configuration (e.g. `DE/Rand1Bin`).
//! - **gens** — generations actually run; always [`GENS`] here, since no
//!   strategy stops early.
//! - **best** — `best_fitness_ever`: the lowest cost seen by *any* individual
//!   across *all* generations (a rolling minimum). Fitness is the raw landscape
//!   cost under the minimization convention, so **lower is better** and this is
//!   the headline "how close did we get" number. Compare it against the
//!   landscape's global optimum `f*` (stated in each example's own docs).
//! - **mean** — `mean_fitness`: the average cost of the *final* generation. A
//!   small `best`–`mean` gap means the whole population converged near the
//!   optimum; a large gap means a few good individuals are dragging a still
//!   spread-out population, i.e. premature or partial convergence.
//!
//! Values print in scientific notation (`1.23e-4`), so an exponent like `e-6`
//! is near-converged and `e+2` is far off. Because every run shares the same
//! [`SEED`], the numbers are reproducible and directly comparable across
//! strategies.

use burn::backend::Flex;
use rlevo_core::bounds::Bounds;
use rlevo_core::fitness::Landscape;
use rlevo_evolution::algorithms::de::{DeConfig, DeVariant, DifferentialEvolution};
use rlevo_evolution::algorithms::ep::{EpConfig, EvolutionaryProgramming};
use rlevo_evolution::algorithms::es_classical::{EsConfig, EsKind, EvolutionStrategy};
use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_core::objective::ObjectiveSense;
use rlevo_core::rate::NonNegativeRate;
use rlevo_evolution::fitness::{BatchFitnessFn, FromLandscape};
use rlevo_evolution::strategy::{EvolutionaryHarness, Strategy};

type B = Flex;

/// Generation budget every strategy is given.
pub const GENS: usize = 500;
/// Fixed seed so runs are comparable across strategies and reproducible.
pub const SEED: u64 = 42;

/// Drives one strategy to completion and prints its convergence summary.
fn run<S, F>(label: &str, strategy: S, params: S::Params, fitness_fn: F)
where
    S: Strategy<B>,
    S::Params: std::fmt::Debug + rlevo_core::config::Validate,
    F: BatchFitnessFn<B, S::Genome>,
    B: burn::tensor::backend::Backend,
    <B as burn::tensor::backend::BackendTypes>::Device: Clone + Default,
{
    let device = Default::default();
    let mut harness =
        EvolutionaryHarness::<B, S, F>::new(strategy, params, fitness_fn, SEED, device, GENS).expect("valid params");
    harness.reset();
    loop {
        if harness.step(()).done {
            break;
        }
    }
    let m = harness
        .latest_metrics()
        .expect("at least one generation ran");
    // `best` = best_fitness_ever (rolling minimum across all generations);
    // `mean` = mean_fitness of the final generation. See the module docs'
    // "Reading the output" section for how to interpret the two together.
    println!(
        "{label:>30} | gens={:>4} | best={:>.6e} | mean={:>.6e}",
        m.generation, m.best_fitness_ever, m.mean_fitness,
    );
}

/// Runs the full real-valued strategy panel on `landscape`.
///
/// `dim` is the genome dimensionality (use the landscape's natural dimension —
/// 2 for the strictly 2-D landscapes), `bounds` is the per-coordinate search
/// box (typically `landscape.bounds()`), and `mutation_sigma` sets the GA
/// Gaussian mutation scale; the ES/EP/DE configs self-tune from `dim`.
pub fn showcase<L>(title: &str, dim: usize, bounds: (f64, f64), mutation_sigma: f32, landscape: L)
where
    L: Landscape + Copy,
{
    #[allow(clippy::cast_possible_truncation)]
    let bounds: Bounds = Bounds::new(bounds.0 as f32, bounds.1 as f32);

    println!(
        "{title}-D{dim} showcase — {GENS} generations, Flex backend, seed={SEED}\n{:-<80}",
        "",
    );

    run(
        "GA/real/tournament-blx-elite",
        GeneticAlgorithm::<B>::new(),
        GaConfig {
            pop_size: 64,
            genome_dim: dim,
            bounds,
            mutation_sigma: NonNegativeRate::new(mutation_sigma),
            selection: GaSelection::Tournament { size: 2 },
            crossover: GaCrossover::BlxAlpha { alpha: NonNegativeRate::new(0.5) },
            replacement: GaReplacement::Elitist { elitism_k: 2 },
        },
        FromLandscape::with_sense(landscape, ObjectiveSense::Minimize),
    );

    for kind in [
        EsKind::OnePlusOne,
        EsKind::OnePlusLambda { lambda: 8 },
        EsKind::MuPlusLambda { mu: 5, lambda: 20 },
        EsKind::MuCommaLambda { mu: 5, lambda: 20 },
    ] {
        let mut params = EsConfig::default_for(kind, dim);
        params.bounds = bounds;
        let label = format!("ES/{kind:?}");
        run(
            &label,
            EvolutionStrategy::<B>::new(),
            params,
            FromLandscape::with_sense(landscape, ObjectiveSense::Minimize),
        );
    }

    let mut ep_params = EpConfig::default_for(20, dim);
    ep_params.bounds = bounds;
    run(
        "EP/fogel/μ=20/q=10",
        EvolutionaryProgramming::<B>::new(),
        ep_params,
        FromLandscape::with_sense(landscape, ObjectiveSense::Minimize),
    );

    for variant in [
        DeVariant::Rand1Bin,
        DeVariant::Best1Bin,
        DeVariant::CurrentToBest1Bin,
        DeVariant::Rand2Bin,
        DeVariant::Rand1Exp,
    ] {
        let mut params = DeConfig::default_for(30, dim);
        params.variant = variant;
        params.bounds = bounds;
        let label = format!("DE/{variant:?}");
        run(
            &label,
            DifferentialEvolution::<B>::new(),
            params,
            FromLandscape::with_sense(landscape, ObjectiveSense::Minimize),
        );
    }
}
