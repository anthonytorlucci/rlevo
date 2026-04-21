//! Streaming reporter sink.
//!
//! Reporters receive events during suite execution. They are not
//! responsible for computing metrics — that happens in `metrics::core` —
//! only for emitting / displaying / buffering the events they observe.

pub mod logging;

#[cfg(feature = "json")]
pub mod json;

#[cfg(feature = "tui")]
pub mod tui;

use crate::report::{BenchmarkReport, EpisodeRecord, TrialReport};
use crate::suite::{SuiteInfo, TrialInfo};

pub trait Reporter: Send {
    fn on_suite_start(&mut self, suite: &SuiteInfo);
    fn on_trial_start(&mut self, trial: &TrialInfo);
    fn on_episode_end(&mut self, trial: &TrialInfo, ep: &EpisodeRecord);
    fn on_trial_end(&mut self, trial: &TrialInfo, report: &TrialReport);
    fn on_suite_end(&mut self, report: &BenchmarkReport);
}
