//! The harness run: the loop, one per trial, through `Evaluator::run_trials`
//! as a `GenerationTrial`.

use std::collections::BTreeMap;

use rlevo::harness::evaluator::{Evaluator, EvaluatorConfig, GenerationTrial};
use rlevo::harness::metrics::core::GENERATIONS;
use rlevo::harness::report::BenchmarkReport;
use rlevo::harness::reporter::logging::LoggingReporter;

use crate::strategy::evo_loop;
use crate::well_siting::WellSiting;

// FINDING: `rlevo::harness::metrics::ea` has `BEST_FITNESS` and no constant
// for the key the loop reports its best-ever fitness under. It is spelled
// here as `docs/ask-tell-contract.md` spells it.
const BEST_EVER: &str = "ea/best_fitness_ever";

const TRIALS: usize = 2;
const BUDGET: usize = 30;
const BASE_SEED: u64 = 2026;

/// Run the suite with the loop's budget at `generations`.
fn run_suite(generations: usize) -> BenchmarkReport {
    let config = EvaluatorConfig {
        num_trials_per_env: TRIALS,
        // The cap must not cut a trial short of the loop's own budget.
        max_steps: BUDGET,
        base_seed: BASE_SEED,
        num_threads: Some(2),
        ..EvaluatorConfig::default()
    };
    let mut reporter = LoggingReporter::new();
    Evaluator::new(config).run_trials(
        "byoe-2",
        &["well-siting".to_string()],
        |_key, seed, _agent_seed| GenerationTrial {
            probe: evo_loop(seed, generations).expect("checked at build-loop"),
        },
        &mut reporter,
    )
}

/// Each trial's best-ever cost, by trial index, after checking the trial.
fn best_ever_by_trial(
    report: &BenchmarkReport,
    generations: usize,
) -> Result<BTreeMap<usize, f64>, String> {
    if report.trials.len() != TRIALS {
        return Err(format!(
            "expected {TRIALS} trials, the report holds {}",
            report.trials.len()
        ));
    }
    let mut best = BTreeMap::new();
    for trial in &report.trials {
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
        let Some(&value) = trial.scalars.get(BEST_EVER) else {
            return Err(format!("a trial has no `{BEST_EVER}`"));
        };
        if !value.is_finite() {
            return Err(format!("`{BEST_EVER}` is {value}"));
        }
        best.insert(trial.key.trial_idx, value);
    }
    Ok(best)
}

/// The best cost must fall over the run, from the same starting points,
/// and never below the surface's minimum.
pub fn run() -> Result<String, String> {
    let first = best_ever_by_trial(&run_suite(1), 1)?;
    let last = best_ever_by_trial(&run_suite(BUDGET), BUDGET)?;
    let minimum = WellSiting::minimum_cost();
    let mut lines = Vec::with_capacity(TRIALS);
    for (trial, &after_one) in &first {
        let Some(&after_budget) = last.get(trial) else {
            return Err(format!(
                "trial {trial} is missing from the {BUDGET}-generation run"
            ));
        };
        if after_budget >= after_one {
            return Err(format!(
                "trial {trial}: `{BEST_EVER}` is {after_budget} after {BUDGET} generations \
                 and {after_one} after one; it did not fall"
            ));
        }
        // The loop reports in single precision, so allow its rounding.
        if after_budget < minimum - 1e-4 {
            return Err(format!(
                "trial {trial}: `{BEST_EVER}` is {after_budget}, below the surface's \
                 minimum {minimum}"
            ));
        }
        lines.push(format!("{after_one:.4} to {after_budget:.4}"));
    }
    Ok(format!(
        "{TRIALS} trials, none errored, no episodes, {BUDGET} generations each; \
         {BEST_EVER} fell from {} (minimum {minimum:.4})",
        lines.join(" and ")
    ))
}
