//! The harness runs: both strategies on a shipped landscape through
//! `Evaluator::run_trials`, as `GenerationTrial`s, then the comparisons the
//! walk's last three steps make.

use std::collections::BTreeMap;

use rlevo::burn::tensor::{Device, Tensor};
use rlevo::envs::landscapes::sphere::Sphere;
use rlevo::evo::fitness::FromLandscape;
use rlevo::evo::strategy::{EvolutionaryLoop, Strategy};
use rlevo::harness::evaluator::{Evaluator, EvaluatorConfig, GenerationTrial};
use rlevo::harness::metrics::core::{GENERATIONS, WALL_CLOCK_SECONDS};
use rlevo::harness::report::{BenchmarkReport, TrialReport};
use rlevo::harness::reporter::logging::LoggingReporter;

use crate::hill_climber::{HillClimber, HillClimberParams};
use crate::null::Null;

// FINDING: `rlevo::harness::metrics::ea` has `BEST_FITNESS` and no constant
// for the key the loop reports its best-ever fitness under. It is spelled
// here as `docs/ask-tell-contract.md` spells it.
const BEST_EVER: &str = "ea/best_fitness_ever";

/// The shipped landscape's dimension.
const DIM: usize = 4;
/// The largest change the hill-climber makes to one coordinate.
const STEP: f32 = 0.5;
const TRIALS: usize = 2;
const BUDGET: usize = 60;
const BASE_SEED: u64 = 2026;

/// One run's reports, by trial index.
type Trials = BTreeMap<usize, TrialReport>;

fn sphere() -> Sphere {
    Sphere::new(DIM).expect("a positive dimension")
}

fn params() -> HillClimberParams {
    let (low, high) = sphere().bounds();
    #[allow(clippy::cast_possible_truncation)]
    HillClimberParams {
        dim: DIM,
        low: low as f32,
        high: high as f32,
        step: STEP,
    }
}

/// Run `strategy` on the landscape with the loop's budget at `generations`,
/// and check each trial's report.
fn run_suite<S>(strategy: S, generations: usize) -> Result<Trials, String>
where
    S: Strategy<Params = HillClimberParams, Genome = Tensor<2>> + Copy,
{
    let config = EvaluatorConfig {
        num_trials_per_env: TRIALS,
        // The cap must not cut a trial short of the loop's own budget.
        max_steps: BUDGET,
        base_seed: BASE_SEED,
        num_threads: Some(2),
        ..EvaluatorConfig::default()
    };
    let mut reporter = LoggingReporter::new();
    let report: BenchmarkReport = Evaluator::new(config).run_trials(
        "byoa-1b",
        &["sphere-d4".to_string()],
        |_key, seed, _agent_seed| GenerationTrial {
            probe: EvolutionaryLoop::new(
                strategy,
                params(),
                FromLandscape::new(sphere()),
                seed,
                Device::flex(),
                generations,
            )
            .expect("valid parameters"),
        },
        &mut reporter,
    );

    if report.trials.len() != TRIALS {
        return Err(format!(
            "expected {TRIALS} trials, the report holds {}",
            report.trials.len()
        ));
    }
    let mut trials = Trials::new();
    for trial in report.trials {
        if trial.errored {
            return Err(format!(
                "a trial errored: {}",
                trial.error_message.as_deref().unwrap_or("no message")
            ));
        }
        if !trial.episodes.is_empty() {
            return Err(format!(
                "a generation trial reported {} episodes",
                trial.episodes.len()
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let budget = generations as f64;
        if trial.scalars.get(GENERATIONS) != Some(&budget) {
            return Err(format!(
                "a trial ran {:?} generations, not its budget of {generations}",
                trial.scalars.get(GENERATIONS)
            ));
        }
        match trial.scalars.get(BEST_EVER) {
            Some(value) if value.is_finite() => {}
            Some(value) => return Err(format!("`{BEST_EVER}` is {value}")),
            None => return Err(format!("a trial has no `{BEST_EVER}`")),
        }
        trials.insert(trial.key.trial_idx, trial);
    }
    Ok(trials)
}

fn best_ever(trial: &TrialReport) -> f64 {
    trial.scalars[BEST_EVER]
}

/// The four runs the ordering step compares.
struct Runs {
    null_first: Trials,
    null_last: Trials,
    climber_first: Trials,
    climber_last: Trials,
}

/// Run the walk's last three steps, reporting each.
pub fn run(report: impl Fn(&str, Result<String, String>)) {
    let runs = match run_all() {
        Ok(runs) => runs,
        Err(detail) => return report("run-trials", Err(detail)),
    };
    report(
        "run-trials",
        Ok(format!(
            "both strategies, {TRIALS} trials each, none errored, no episodes, \
             1 and {BUDGET} generations; {BEST_EVER} present and finite"
        )),
    );
    let ordering = ordering(&runs);
    let ordered = ordering.is_ok();
    report("ordering", ordering);
    if ordered {
        report("determinism", determinism(&runs));
    }
}

fn run_all() -> Result<Runs, String> {
    Ok(Runs {
        null_first: run_suite(Null, 1)?,
        null_last: run_suite(Null, BUDGET)?,
        climber_first: run_suite(HillClimber, 1)?,
        climber_last: run_suite(HillClimber, BUDGET)?,
    })
}

/// The null strategy must end where it started, the two must start at the
/// same point, and the hill-climber must end strictly lower.
fn ordering(runs: &Runs) -> Result<String, String> {
    let mut lines = Vec::with_capacity(TRIALS);
    for (trial, null_first) in &runs.null_first {
        let start = best_ever(null_first);
        let null_end = best_ever(&runs.null_last[trial]);
        let climber_start = best_ever(&runs.climber_first[trial]);
        let climber_end = best_ever(&runs.climber_last[trial]);
        if null_end.to_bits() != start.to_bits() {
            return Err(format!(
                "trial {trial}: the null strategy's {BEST_EVER} moved from {start} to \
                 {null_end}; a strategy that never moves cannot improve"
            ));
        }
        if climber_start.to_bits() != start.to_bits() {
            return Err(format!(
                "trial {trial}: after one generation the hill-climber reports {climber_start} \
                 and the null strategy {start}; they start from one point"
            ));
        }
        if climber_end >= null_end {
            return Err(format!(
                "trial {trial}: the hill-climber ends at {climber_end}, no better than the \
                 null strategy's {null_end}"
            ));
        }
        lines.push(format!("{start:.4} to {climber_end:.4}"));
    }
    Ok(format!(
        "per trial the null strategy ends where it starts; the hill-climber falls from {}",
        lines.join(" and ")
    ))
}

/// Both budget runs again: every scalar but the wall clock, bit for bit.
fn determinism(runs: &Runs) -> Result<String, String> {
    let again = [
        ("null", &runs.null_last, run_suite(Null, BUDGET)?),
        (
            "hill-climber",
            &runs.climber_last,
            run_suite(HillClimber, BUDGET)?,
        ),
    ];
    let mut compared = 0;
    for (name, first, second) in &again {
        for (trial, report) in *first {
            let repeat = &second[trial];
            let keys = |r: &TrialReport| -> Vec<String> {
                r.scalars
                    .keys()
                    .filter(|key| key.as_str() != WALL_CLOCK_SECONDS)
                    .cloned()
                    .collect()
            };
            if keys(report) != keys(repeat) {
                return Err(format!(
                    "{name}, trial {trial}: the second run reports other scalars"
                ));
            }
            for key in keys(report) {
                let (a, b) = (report.scalars[&key], repeat.scalars[&key]);
                if a.to_bits() != b.to_bits() {
                    return Err(format!(
                        "{name}, trial {trial}: `{key}` is {a} and then {b}"
                    ));
                }
                compared += 1;
            }
        }
    }
    Ok(format!(
        "a second run of each repeats all {compared} scalars bit for bit, the wall clock aside"
    ))
}
