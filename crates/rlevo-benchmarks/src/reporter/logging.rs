//! Tracing-backed [`Reporter`] with no additional dependencies.
//!
//! [`LoggingReporter`] emits one `INFO`-level tracing event per lifecycle
//! callback under the `rlevo_benchmarks` target. It is always compiled; no
//! feature flag is required.
//!
//! [`Reporter`]: crate::reporter::Reporter

use tracing::info;

use crate::report::{BenchmarkReport, EpisodeSummary, TrialReport};
use crate::reporter::Reporter;
use crate::suite::{SuiteInfo, TrialInfo};

/// [`Reporter`] that emits a structured `INFO` tracing event at each lifecycle boundary.
///
/// Events are tagged with `target: "rlevo_benchmarks"` so they can be
/// filtered independently of other `rlevo` spans. The struct carries no
/// state; it is cheaply cloneable via its [`Default`] derivation.
///
/// [`Reporter`]: crate::reporter::Reporter
#[derive(Debug, Default)]
pub struct LoggingReporter;

impl LoggingReporter {
    /// Creates a new `LoggingReporter`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Reporter for LoggingReporter {
    fn on_suite_start(&mut self, suite: &SuiteInfo) {
        info!(
            target: "rlevo_benchmarks",
            suite = %suite.name,
            num_envs = suite.env_names.len(),
            trials_per_env = suite.num_trials_per_env,
            "suite start"
        );
    }

    fn on_trial_start(&mut self, trial: &TrialInfo) {
        info!(
            target: "rlevo_benchmarks",
            env = %trial.env_name,
            env_idx = trial.key.env_idx,
            trial_idx = trial.key.trial_idx,
            seed = trial.trial_seed,
            "trial start"
        );
    }

    fn on_episode_end(&mut self, trial: &TrialInfo, ep: &EpisodeSummary) {
        info!(
            target: "rlevo_benchmarks",
            env = %trial.env_name,
            trial_idx = trial.key.trial_idx,
            episode = ep.episode_idx,
            ret = ep.return_value,
            length = ep.length,
            "episode end"
        );
    }

    fn on_trial_end(&mut self, trial: &TrialInfo, report: &TrialReport) {
        info!(
            target: "rlevo_benchmarks",
            env = %trial.env_name,
            trial_idx = trial.key.trial_idx,
            episodes = report.episodes.len(),
            scalars = report.scalars.len(),
            errored = report.errored,
            "trial end"
        );
    }

    fn on_suite_end(&mut self, report: &BenchmarkReport) {
        info!(
            target: "rlevo_benchmarks",
            suite = %report.suite_name,
            trials = report.trials.len(),
            "suite end"
        );
    }
}
