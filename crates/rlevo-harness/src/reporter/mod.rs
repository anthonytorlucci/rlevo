//! Streaming reporter sinks for benchmark suite events.
//!
//! The central abstraction is the [`Reporter`] trait. Implementations receive
//! lifecycle callbacks as a suite progresses and are free to emit, display, or
//! buffer events however they choose. Metric *computation* is not their
//! responsibility — that belongs to `metrics`.
//!
//! Built-in implementations live in the sub-modules:
//!
//! - `logging` — a simple tracing-backed sink (always available)
//! - `json` — serialises reports to JSON files (requires the `json` feature)
//!
//! When a run must drive several sinks simultaneously, wrap them in a
//! [`MultiReporter`].

pub mod logging;

#[cfg(feature = "json")]
pub mod json;

use std::path::PathBuf;

use crate::metrics::Metric;
use crate::report::{BenchmarkReport, EpisodeSummary, TrialReport};
use crate::suite::{SuiteInfo, TrialInfo};

/// Where in a trial a set of metrics was read.
///
/// The harness reads the agent's or probe's `emit_metrics` at each of these
/// and hands the values to [`Reporter::on_metrics`]. A read describes the
/// trial so far and changes nothing (ADR 0097), so a sink may keep every read
/// as a time series or only the latest.
///
/// `#[non_exhaustive]`: a later kind of boundary is not a breaking change, so
/// a `match` needs a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Boundary {
    /// A training episode of an episodic trial ended.
    TrainingEpisode {
        /// Zero-based index of the episode among the trial's training
        /// episodes.
        index: usize,
        /// Environment steps the trial has taken so far, training and
        /// evaluation together, this episode's included.
        step: u64,
    },
    /// An evaluation episode of an episodic trial ended.
    EvaluationEpisode {
        /// Zero-based index of the episode among the trial's evaluation
        /// episodes.
        index: usize,
        /// Environment steps the trial has taken so far, training and
        /// evaluation together, this episode's included.
        step: u64,
    },
    /// A generation of a generation trial ended.
    Generation {
        /// Generations run so far, this one included: 1 after the first. The
        /// same count as the report's `generations` scalar and an
        /// `EvolutionaryLoop`'s `ea/generation`.
        generation: usize,
    },
}

/// The file a recording sink wrote for one trial, as the sink reports it
/// through [`Reporter::recording`].
///
/// `#[non_exhaustive]`: build one with [`Recording::new`], so a later field
/// is not a breaking change.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Recording {
    /// Where the trial's recording is written.
    pub path: PathBuf,
    /// The version of the recording SDK that writes it. A viewer may open
    /// only files of versions near its own, so the report keeps this beside
    /// the path.
    pub sdk_version: String,
}

impl Recording {
    /// A recording at `path`, written by version `sdk_version` of its SDK.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>, sdk_version: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            sdk_version: sdk_version.into(),
        }
    }
}

/// Event sink called by the `Evaluator` throughout a benchmark run.
///
/// Implementors are notified at each lifecycle boundary — suite start/end,
/// trial start/end, each episode, and each point where the trial's metrics
/// are read — and may emit, log, or buffer the data however they need.
/// Implementors must be [`Send`] so they can be driven from the evaluator's
/// execution context.
///
/// For running multiple sinks in parallel, see [`MultiReporter`].
pub trait Reporter: Send {
    /// Called once before any trials begin, with metadata about the suite.
    fn on_suite_start(&mut self, suite: &SuiteInfo);

    /// Called at the start of each (env, seed) trial.
    fn on_trial_start(&mut self, trial: &TrialInfo);

    /// Called after each episode of an episodic trial, training and
    /// evaluation alike; [`EpisodeSummary::phase`] says which. A generation
    /// trial plays no episodes and never calls it.
    ///
    /// Named `on_episode_end` before 0.7.0, when it saw evaluation episodes
    /// only. A sink that wants the old behaviour returns early unless the
    /// phase is [`Phase::Evaluation`](crate::report::Phase::Evaluation).
    fn on_episode(&mut self, trial: &TrialInfo, ep: &EpisodeSummary);

    /// Called with the trial's metrics each time the harness reads them: after
    /// every episode of an episodic trial (the agent's
    /// [`emit_metrics`](rlevo_core::agent::Agent::emit_metrics)), and after
    /// every generation of a generation trial (the probe's). `at` says which
    /// boundary, and carries the index and, for episodes, the step count a
    /// time series is plotted against.
    ///
    /// For an episode, it follows that episode's
    /// [`on_episode`](Self::on_episode). It is called even when `metrics` is
    /// empty, and never when [`wants_metrics`](Self::wants_metrics) is
    /// `false`. The default does nothing.
    fn on_metrics(&mut self, trial: &TrialInfo, at: Boundary, metrics: &[Metric]) {
        let _ = (trial, at, metrics);
    }

    /// Whether this sink uses [`on_metrics`](Self::on_metrics). The harness
    /// asks once as each trial starts; when it is `false`, the trial neither
    /// reads its agent's or probe's metrics at each boundary nor calls
    /// `on_metrics`, which saves a read and a lock per episode. Defaults to
    /// `true`; a sink that leaves `on_metrics` at its default returns `false`.
    fn wants_metrics(&self) -> bool {
        true
    }

    /// The recording this sink is writing for `trial`, if it writes one.
    ///
    /// The harness asks once, after the trial has run and before
    /// [`on_trial_end`](Self::on_trial_end), and copies the answer into the
    /// trial's report ([`TrialReport::recording_path`] and
    /// [`TrialReport::recording_sdk_version`]), so the report a sink receives
    /// and the JSON a run leaves both say where the trial's recording is. A
    /// sink cannot write there itself: it is handed the report by shared
    /// reference. The default is `None`, for a sink that records nothing.
    fn recording(&self, trial: &TrialInfo) -> Option<Recording> {
        let _ = trial;
        None
    }

    /// Called when a trial finishes, with its full accumulated report.
    fn on_trial_end(&mut self, trial: &TrialInfo, report: &TrialReport);

    /// Called once after all trials complete, with the aggregate suite report.
    fn on_suite_end(&mut self, report: &BenchmarkReport);
}

/// Fans a single stream of reporter events out to multiple sinks in
/// insertion order. Used when an execution must drive both a progress
/// reporter (e.g. `LoggingReporter`) and a recording reporter
/// (e.g. `RecordingReporter`) from the same suite run.
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
    /// Creates a `MultiReporter` that forwards events to `reporters` in order.
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
    fn on_episode(&mut self, trial: &TrialInfo, ep: &EpisodeSummary) {
        for r in &mut self.reporters {
            r.on_episode(trial, ep);
        }
    }
    /// Forwards to the sinks that [want metrics](Reporter::wants_metrics).
    fn on_metrics(&mut self, trial: &TrialInfo, at: Boundary, metrics: &[Metric]) {
        for r in &mut self.reporters {
            if r.wants_metrics() {
                r.on_metrics(trial, at, metrics);
            }
        }
    }
    /// `true` when any sink wants them.
    fn wants_metrics(&self) -> bool {
        self.reporters.iter().any(|r| r.wants_metrics())
    }
    /// The first sink's that has one, in insertion order: a report names one
    /// recording per trial.
    fn recording(&self, trial: &TrialInfo) -> Option<Recording> {
        self.reporters.iter().find_map(|r| r.recording(trial))
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
    use crate::report::{BenchmarkReport, EpisodeSummary, Phase, TrialReport};
    use crate::suite::{SuiteInfo, TrialInfo, TrialKey};
    use rlevo_core::environment::EpisodeStatus;

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
                .push(Tag::TrialStart(format!("{}:{}", self.name, trial.env_id)));
        }
        fn on_episode(&mut self, _trial: &TrialInfo, _ep: &EpisodeSummary) {
            self.log
                .lock()
                .unwrap()
                .push(Tag::EpisodeEnd(self.name.clone()));
        }
        fn on_trial_end(&mut self, trial: &TrialInfo, _report: &TrialReport) {
            self.log
                .lock()
                .unwrap()
                .push(Tag::TrialEnd(format!("{}:{}", self.name, trial.env_id)));
        }
        fn on_suite_end(&mut self, report: &BenchmarkReport) {
            self.log.lock().unwrap().push(Tag::SuiteEnd(format!(
                "{}:{}",
                self.name, report.suite_name
            )));
        }
    }

    /// `MultiReporter` wants metrics when any sink does, and hands them only to
    /// the sinks that want them.
    #[test]
    fn multi_reporter_forwards_metrics_to_the_sinks_that_want_them() {
        use super::Boundary;
        use crate::metrics::Metric;

        struct Counts {
            wants: bool,
            calls: Arc<Mutex<usize>>,
        }
        impl Reporter for Counts {
            fn on_suite_start(&mut self, _: &SuiteInfo) {}
            fn on_trial_start(&mut self, _: &TrialInfo) {}
            fn on_episode(&mut self, _: &TrialInfo, _: &EpisodeSummary) {}
            fn on_metrics(&mut self, _: &TrialInfo, _: Boundary, _: &[Metric]) {
                *self.calls.lock().unwrap() += 1;
            }
            fn wants_metrics(&self) -> bool {
                self.wants
            }
            fn on_trial_end(&mut self, _: &TrialInfo, _: &TrialReport) {}
            fn on_suite_end(&mut self, _: &BenchmarkReport) {}
        }

        let calls = |wants| {
            let n = Arc::new(Mutex::new(0));
            (
                Box::new(Counts {
                    wants,
                    calls: Arc::clone(&n),
                }) as Box<dyn Reporter>,
                n,
            )
        };
        let trial = TrialInfo::new(
            TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            "E".into(),
            0,
        );

        let (no, no_calls) = calls(false);
        let mut quiet = MultiReporter::new(vec![no]);
        assert!(!quiet.wants_metrics());

        let (yes, yes_calls) = calls(true);
        let (no, no_calls_2) = calls(false);
        let mut mixed = MultiReporter::new(vec![yes, no]);
        assert!(mixed.wants_metrics());
        mixed.on_metrics(&trial, Boundary::Generation { generation: 1 }, &[]);
        quiet.on_metrics(&trial, Boundary::Generation { generation: 1 }, &[]);

        assert_eq!(*yes_calls.lock().unwrap(), 1);
        assert_eq!(*no_calls.lock().unwrap(), 0);
        assert_eq!(*no_calls_2.lock().unwrap(), 0);
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
            env_ids: vec!["E".into()],
            num_trials_per_env: 1,
            success_threshold: None,
            run_id: "run".into(),
        };
        let trial = TrialInfo {
            key: TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            env_id: "E".into(),
            trial_seed: 0,
        };
        let ep = EpisodeSummary::new(Phase::Evaluation, 7, 1.0, 2, EpisodeStatus::Terminated);
        let report = BenchmarkReport::new("S".into(), 0);
        let trep = TrialReport::new(trial.key, "E".into(), 0);

        multi.on_suite_start(&suite);
        multi.on_trial_start(&trial);
        multi.on_episode(&trial, &ep);
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

    /// A report names one recording per trial, so `MultiReporter` answers
    /// with the first sink's that has one, in insertion order.
    #[test]
    fn multi_reporter_reports_the_first_sinks_recording() {
        use super::Recording;

        struct Records(Option<&'static str>);
        impl Reporter for Records {
            fn on_suite_start(&mut self, _: &SuiteInfo) {}
            fn on_trial_start(&mut self, _: &TrialInfo) {}
            fn on_episode(&mut self, _: &TrialInfo, _: &EpisodeSummary) {}
            fn recording(&self, _: &TrialInfo) -> Option<Recording> {
                self.0.map(|path| Recording::new(path, "1.2.3"))
            }
            fn on_trial_end(&mut self, _: &TrialInfo, _: &TrialReport) {}
            fn on_suite_end(&mut self, _: &BenchmarkReport) {}
        }
        let trial = TrialInfo {
            key: TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            env_id: "E".into(),
            trial_seed: 0,
        };
        let multi = |sinks: &[Option<&'static str>]| {
            MultiReporter::new(
                sinks
                    .iter()
                    .map(|s| Box::new(Records(*s)) as Box<dyn Reporter>)
                    .collect(),
            )
        };

        assert_eq!(
            multi(&[None, Some("b.rrd"), Some("c.rrd")]).recording(&trial),
            Some(Recording::new("b.rrd", "1.2.3"))
        );
        assert_eq!(multi(&[None, None]).recording(&trial), None);
        assert_eq!(multi(&[]).recording(&trial), None);
    }
}
