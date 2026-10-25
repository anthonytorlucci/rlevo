//! Step 4: can the documented entry point train a learner? The learner is
//! trained and evaluated by `Evaluator::run_suite` alone, with a training
//! budget on its configuration: no `Trial` is written here, and no loop.

use rlevo::harness::evaluator::{Evaluator, EvaluatorConfig};
use rlevo::harness::report::BenchmarkReport;
use rlevo::harness::reporter::logging::LoggingReporter;
use rlevo::harness::suite::Suite;

use crate::frozen_lake::{ENV_ID, lake, learner};
use crate::q_learner::UPDATES;

/// Environment steps each trial trains its agent for before evaluating it.
pub const TRAIN_STEPS: usize = 20_000;

/// The one configuration every run in the probe uses, the baseline's
/// included.
pub fn config() -> EvaluatorConfig {
    EvaluatorConfig {
        num_episodes: 20,
        train_steps: TRAIN_STEPS,
        num_trials_per_env: 1,
        max_steps: 100,
        base_seed: 2026,
        num_threads: Some(1),
        ..EvaluatorConfig::default()
    }
}

/// The learner, trained and evaluated on FrozenLake.
pub fn run_learner() -> BenchmarkReport {
    let suite = Suite::new("byoa-1a").with_env(ENV_ID, lake);
    Evaluator::new(config()).run_suite(&suite, |_seed| learner(), &mut LoggingReporter::new())
}

/// Run step 4 and report it.
pub fn run(report: impl Fn(&str, Result<String, String>)) {
    report("learning-seam", trained(&run_learner()));
}

/// The learner was shown every training step: one update per transition.
fn trained(report: &BenchmarkReport) -> Result<String, String> {
    let [trial] = report.trials.as_slice() else {
        return Err(format!(
            "expected 1 trial, the report holds {}",
            report.trials.len()
        ));
    };
    if trial.errored {
        return Err(format!(
            "the trial errored: {}",
            trial.error_message.as_deref().unwrap_or("no message")
        ));
    }
    match trial.counters.get(UPDATES) {
        Some(&updates) if updates == TRAIN_STEPS as u64 => Ok(format!(
            "run_suite trained the learner through its Agent hooks, {updates} updates in \
             {TRAIN_STEPS} steps, then evaluated it; no Trial written"
        )),
        other => Err(format!(
            "run_suite with train_steps {TRAIN_STEPS} reported {UPDATES} {other:?}; \
             the learner updates once per transition it is shown"
        )),
    }
}
