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

use crate::report::{BenchmarkReport, EpisodeSummary, TrialReport};
use crate::suite::{SuiteInfo, TrialInfo};

pub trait Reporter: Send {
    fn on_suite_start(&mut self, suite: &SuiteInfo);
    fn on_trial_start(&mut self, trial: &TrialInfo);
    fn on_episode_end(&mut self, trial: &TrialInfo, ep: &EpisodeSummary);
    fn on_trial_end(&mut self, trial: &TrialInfo, report: &TrialReport);
    fn on_suite_end(&mut self, report: &BenchmarkReport);
}

/// Fans a single stream of reporter events out to multiple sinks in
/// insertion order. Used when an execution must drive both a live
/// reporter (e.g. `TuiReporter`) and a recording reporter
/// (Milestone 4 `RecordingReporter`) from the same suite run.
pub struct MultiReporter {
    reporters: Vec<Box<dyn Reporter>>,
}

impl std::fmt::Debug for MultiReporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiReporter")
            .field("len", &self.reporters.len())
            .finish()
    }
}

impl MultiReporter {
    #[must_use]
    pub fn new(reporters: Vec<Box<dyn Reporter>>) -> Self {
        Self { reporters }
    }
}

impl Reporter for MultiReporter {
    fn on_suite_start(&mut self, suite: &SuiteInfo) {
        for r in &mut self.reporters {
            r.on_suite_start(suite);
        }
    }
    fn on_trial_start(&mut self, trial: &TrialInfo) {
        for r in &mut self.reporters {
            r.on_trial_start(trial);
        }
    }
    fn on_episode_end(&mut self, trial: &TrialInfo, ep: &EpisodeSummary) {
        for r in &mut self.reporters {
            r.on_episode_end(trial, ep);
        }
    }
    fn on_trial_end(&mut self, trial: &TrialInfo, report: &TrialReport) {
        for r in &mut self.reporters {
            r.on_trial_end(trial, report);
        }
    }
    fn on_suite_end(&mut self, report: &BenchmarkReport) {
        for r in &mut self.reporters {
            r.on_suite_end(report);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{MultiReporter, Reporter};
    use crate::report::{BenchmarkReport, EpisodeSummary, TrialReport};
    use crate::suite::{SuiteInfo, TrialInfo, TrialKey};

    #[derive(Debug, Clone)]
    enum Tag {
        SuiteStart(String),
        TrialStart(String),
        EpisodeEnd(String),
        TrialEnd(String),
        SuiteEnd(String),
    }

    struct TaggingReporter {
        name: String,
        log: Arc<Mutex<Vec<Tag>>>,
    }

    impl Reporter for TaggingReporter {
        fn on_suite_start(&mut self, suite: &SuiteInfo) {
            self.log
                .lock()
                .unwrap()
                .push(Tag::SuiteStart(format!("{}:{}", self.name, suite.name)));
        }
        fn on_trial_start(&mut self, trial: &TrialInfo) {
            self.log
                .lock()
                .unwrap()
                .push(Tag::TrialStart(format!("{}:{}", self.name, trial.env_name)));
        }
        fn on_episode_end(&mut self, _trial: &TrialInfo, _ep: &EpisodeSummary) {
            self.log
                .lock()
                .unwrap()
                .push(Tag::EpisodeEnd(self.name.clone()));
        }
        fn on_trial_end(&mut self, trial: &TrialInfo, _report: &TrialReport) {
            self.log
                .lock()
                .unwrap()
                .push(Tag::TrialEnd(format!("{}:{}", self.name, trial.env_name)));
        }
        fn on_suite_end(&mut self, report: &BenchmarkReport) {
            self.log
                .lock()
                .unwrap()
                .push(Tag::SuiteEnd(format!("{}:{}", self.name, report.suite_name)));
        }
    }

    #[test]
    fn multi_reporter_fans_out_in_insertion_order() {
        let log: Arc<Mutex<Vec<Tag>>> = Arc::new(Mutex::new(Vec::new()));
        let r1 = Box::new(TaggingReporter {
            name: "a".into(),
            log: Arc::clone(&log),
        });
        let r2 = Box::new(TaggingReporter {
            name: "b".into(),
            log: Arc::clone(&log),
        });
        let mut multi = MultiReporter::new(vec![r1, r2]);

        let suite = SuiteInfo {
            name: "S".into(),
            env_names: vec!["E".into()],
            num_trials_per_env: 1,
        };
        let trial = TrialInfo {
            key: TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            env_name: "E".into(),
            trial_seed: 0,
        };
        let ep = EpisodeSummary {
            episode_idx: 7,
            return_value: 1.0,
            length: 2,
        };
        let report = BenchmarkReport::new("S".into(), 0);
        let trep = TrialReport::new(trial.key, "E".into(), 0);

        multi.on_suite_start(&suite);
        multi.on_trial_start(&trial);
        multi.on_episode_end(&trial, &ep);
        multi.on_trial_end(&trial, &trep);
        multi.on_suite_end(&report);

        let entries = log.lock().unwrap().clone();
        let names: Vec<String> = entries
            .iter()
            .map(|t| match t {
                Tag::SuiteStart(s)
                | Tag::TrialStart(s)
                | Tag::TrialEnd(s)
                | Tag::SuiteEnd(s)
                | Tag::EpisodeEnd(s) => s.clone(),
            })
            .collect();
        // Each event fans out in [a, b] order before the next event fires.
        assert_eq!(
            names,
            vec![
                "a:S", "b:S", "a:E", "b:E", "a", "b", "a:E", "b:E", "a:S", "b:S",
            ]
        );
    }
}
