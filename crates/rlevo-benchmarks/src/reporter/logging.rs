//! Zero-dep reporter that emits tracing events.

use tracing::info;

use crate::report::{BenchmarkReport, EpisodeRecord, TrialReport};
use crate::reporter::Reporter;
use crate::suite::{SuiteInfo, TrialInfo};

#[derive(Debug, Default)]
pub struct LoggingReporter;

impl LoggingReporter {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Reporter for LoggingReporter {
    fn on_suite_start(&mut self, suite: &SuiteInfo) {
        info!(
            target: "evorl_benchmarks",
            suite = %suite.name,
            num_envs = suite.env_names.len(),
            trials_per_env = suite.num_trials_per_env,
            "suite start"
        );
    }

    fn on_trial_start(&mut self, trial: &TrialInfo) {
        info!(
            target: "evorl_benchmarks",
            env = %trial.env_name,
            env_idx = trial.key.env_idx,
            trial_idx = trial.key.trial_idx,
            seed = trial.trial_seed,
            "trial start"
        );
    }

    fn on_episode_end(&mut self, trial: &TrialInfo, ep: &EpisodeRecord) {
        info!(
            target: "evorl_benchmarks",
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
            target: "evorl_benchmarks",
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
            target: "evorl_benchmarks",
            suite = %report.suite_name,
            trials = report.trials.len(),
            "suite end"
        );
    }
}
