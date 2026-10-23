//! The harness run: the tank farm behind `TimeLimit`, registered as itself,
//! driven through `Evaluator::run_suite` against the proportional policy.

use rlevo::envs::wrappers::TimeLimit;
use rlevo::harness::evaluator::{Evaluator, EvaluatorConfig};
use rlevo::harness::metrics::core::RETURN_MEAN;
use rlevo::harness::reporter::logging::LoggingReporter;
use rlevo::harness::suite::Suite;

use crate::TIME_LIMIT;
use crate::policy::Proportional;
use crate::tank_farm::{ENV_ID, TankFarm};

const EPISODES: usize = 3;
const TRIALS: usize = 2;

/// Run the suite and check the report it returns.
///
/// `run_suite` never returns an error: a trial that failed is a report with
/// `errored` set. So "a report came back" proves nothing, and the checks
/// here are on what the report holds.
pub fn run() -> Result<String, String> {
    let config = EvaluatorConfig {
        num_episodes: EPISODES,
        num_trials_per_env: TRIALS,
        // Four steps of slack over the time limit: an episode that reaches
        // this cap was not ended by the wrapper.
        max_steps: TIME_LIMIT + 4,
        base_seed: 2026,
        num_threads: Some(2),
        ..EvaluatorConfig::default()
    };
    let suite = Suite::new("byoe-1").with_env(ENV_ID, |seed| {
        TimeLimit::new(TankFarm::with_seed(seed), TIME_LIMIT)
    });
    let mut reporter = LoggingReporter::default();
    let report = Evaluator::new(config).run_suite(&suite, |_seed| Proportional, &mut reporter);

    if report.trials.len() != TRIALS {
        return Err(format!(
            "expected {TRIALS} trials, the report holds {}",
            report.trials.len()
        ));
    }
    let mut returns = Vec::with_capacity(TRIALS);
    for trial in &report.trials {
        if trial.errored {
            return Err(format!(
                "a trial on `{}` errored: {}",
                trial.env_name,
                trial.error_message.as_deref().unwrap_or("no message")
            ));
        }
        if trial.env_name != ENV_ID {
            return Err(format!(
                "a trial is named `{}`, not `{ENV_ID}`",
                trial.env_name
            ));
        }
        if trial.episodes.len() != EPISODES {
            return Err(format!(
                "expected {EPISODES} episodes per trial, a trial holds {}",
                trial.episodes.len()
            ));
        }
        // Shorter means the policy overflowed a tank; longer, that the
        // evaluator's cap ended the episode and not the wrapper.
        if let Some(episode) = trial.episodes.iter().find(|ep| ep.length != TIME_LIMIT) {
            return Err(format!(
                "an episode ran {} steps; the time limit ends it at {TIME_LIMIT}",
                episode.length
            ));
        }
        let Some(mean) = trial.scalars.get(RETURN_MEAN) else {
            return Err(format!("a trial has no `{RETURN_MEAN}`"));
        };
        if !mean.is_finite() {
            return Err(format!("`{RETURN_MEAN}` is {mean}"));
        }
        returns.push(format!("{mean:.3}"));
    }
    Ok(format!(
        "{TRIALS} trials, none errored, {EPISODES} episodes of {TIME_LIMIT} steps each, ended \
         by TimeLimit; {RETURN_MEAN} {}",
        returns.join(" and ")
    ))
}
