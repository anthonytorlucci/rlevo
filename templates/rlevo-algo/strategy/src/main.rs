//! Runs your strategy through rlevo's evaluation harness on a shipped
//! landscape, the sphere, beside the shipped CMA-ES, and prints the result
//! as a Markdown table.
//!
//! Each row is an `EvolutionaryLoop`, which owns the run's generator, its
//! generation budget and the objective's direction, driven as a
//! `GenerationTrial` under the one configuration in `config`.

mod strategy;

use rlevo::burn::tensor::{Device, Tensor};
use rlevo::core::config::Validate;
use rlevo::envs::landscapes::sphere::Sphere;
use rlevo::evo::algorithms::cma_es::{CmaEs, CmaEsConfig};
use rlevo::evo::fitness::FromLandscape;
use rlevo::evo::strategy::{EvolutionaryLoop, Strategy};
use rlevo::harness::evaluator::{Evaluator, EvaluatorConfig, GenerationTrial};
use rlevo::harness::report::BenchmarkReport;
use rlevo::harness::reporter::logging::LoggingReporter;

use strategy::{MyParams, MyStrategy};

/// Coordinates in a point on the sphere.
const DIM: usize = 10;
/// Generations each loop runs.
const GENERATIONS: usize = 100;
/// Points each generation proposes: CMA-ES's default for ten coordinates,
/// so both rows score the same number of points.
const POPULATION: usize = 10;
/// The name a loop reports the best cost it has found under.
const BEST_EVER: &str = "ea/best_fitness_ever";

/// The configuration every row runs under.
fn config() -> EvaluatorConfig {
    EvaluatorConfig {
        // Independent trials per strategy, each from its own seed.
        num_trials_per_env: 3,
        // The harness's cap on generations: at the loops' own budget, so it
        // never cuts one short.
        max_steps: GENERATIONS,
        base_seed: 2026,
        num_threads: Some(1),
        ..EvaluatorConfig::default()
    }
}

fn sphere() -> Sphere {
    Sphere::new(DIM).expect("a positive dimension")
}

/// `strategy` on the sphere, one loop per trial, seeded by the trial.
fn on_sphere<S>(strategy: S, params: &S::Params) -> BenchmarkReport
where
    S: Strategy<Genome = Tensor<2>> + Clone,
    S::Params: Validate,
{
    let mut reporter = LoggingReporter::new();
    Evaluator::new(config()).run_trials(
        "sphere",
        &["sphere".to_string()],
        |_key, seed, _agent_seed| GenerationTrial {
            probe: EvolutionaryLoop::new(
                strategy.clone(),
                params.clone(),
                FromLandscape::new(sphere()),
                seed,
                Device::flex(),
                GENERATIONS,
            )
            .expect("the parameters are valid"),
        },
        &mut reporter,
    )
}

/// One row of the table: the best cost each strategy found, averaged over
/// its trials. The sphere's minimum is 0.
fn row(name: &str, report: &BenchmarkReport) -> Result<String, String> {
    let mut best = 0.0;
    for trial in &report.trials {
        if trial.errored {
            let why = trial.error_message.as_deref().unwrap_or("no message");
            return Err(format!("{name}: a trial errored: {why}"));
        }
        best += trial.scalars[BEST_EVER];
    }
    #[allow(clippy::cast_precision_loss)]
    let trials = report.trials.len() as f64;
    Ok(format!(
        "| {name} | {:.3e} | {GENERATIONS} | {} |",
        best / trials,
        report.trials.len()
    ))
}

fn main() -> Result<(), String> {
    let (low, high) = sphere().bounds();
    #[allow(clippy::cast_possible_truncation)]
    let params = MyParams {
        dim: DIM,
        low: low as f32,
        high: high as f32,
        population: POPULATION,
    };
    params.validate().map_err(|e| e.to_string())?;

    println!("| strategy | best cost | generations | trials |");
    println!("|---|---:|---:|---:|");
    println!("{}", row("MyStrategy", &on_sphere(MyStrategy, &params))?);
    let cma_es = CmaEsConfig::with_pop_size(POPULATION, DIM);
    println!("{}", row("CmaEs", &on_sphere(CmaEs::new(), &cma_es))?);
    Ok(())
}
