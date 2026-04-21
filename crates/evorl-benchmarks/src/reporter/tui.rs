//! TUI reporter — non-blocking, minimal.
//!
//! v1 intentionally avoids spinning a ratatui render loop on a dedicated
//! thread: it's enough for the harness contract to (a) exist behind the
//! feature flag, (b) never block rayon workers, and (c) provide an
//! observable channel for tests. A richer render loop can land later
//! without changing the `Reporter` surface.

use std::sync::mpsc::{self, Receiver, Sender};

use crate::report::{BenchmarkReport, EpisodeRecord, TrialReport};
use crate::reporter::Reporter;
use crate::suite::{SuiteInfo, TrialInfo};

#[derive(Debug, Clone)]
pub enum TuiEvent {
    SuiteStart(SuiteInfo),
    TrialStart(TrialInfo),
    EpisodeEnd {
        trial: TrialInfo,
        episode: EpisodeRecord,
    },
    TrialEnd {
        trial: TrialInfo,
        report: TrialReport,
    },
    SuiteEnd(BenchmarkReport),
}

#[derive(Debug)]
pub struct TuiReporter {
    tx: Sender<TuiEvent>,
}

impl TuiReporter {
    /// Construct a reporter plus a receiver that a consumer can drain to
    /// drive a ratatui render loop (or a test).
    #[must_use]
    pub fn channel() -> (Self, Receiver<TuiEvent>) {
        let (tx, rx) = mpsc::channel();
        (Self { tx }, rx)
    }
}

impl Reporter for TuiReporter {
    fn on_suite_start(&mut self, suite: &SuiteInfo) {
        let _ = self.tx.send(TuiEvent::SuiteStart(suite.clone()));
    }

    fn on_trial_start(&mut self, trial: &TrialInfo) {
        let _ = self.tx.send(TuiEvent::TrialStart(trial.clone()));
    }

    fn on_episode_end(&mut self, trial: &TrialInfo, ep: &EpisodeRecord) {
        let _ = self.tx.send(TuiEvent::EpisodeEnd {
            trial: trial.clone(),
            episode: ep.clone(),
        });
    }

    fn on_trial_end(&mut self, trial: &TrialInfo, report: &TrialReport) {
        let _ = self.tx.send(TuiEvent::TrialEnd {
            trial: trial.clone(),
            report: report.clone(),
        });
    }

    fn on_suite_end(&mut self, report: &BenchmarkReport) {
        let _ = self.tx.send(TuiEvent::SuiteEnd(report.clone()));
    }
}
