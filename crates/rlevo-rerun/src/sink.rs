//! [`RerunSink`], its configuration and its error.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use rlevo_harness::metrics::Metric;
use rlevo_harness::report::{BenchmarkReport, EpisodeSummary, TrialReport};
use rlevo_harness::reporter::{Boundary, Recording, Reporter};
use rlevo_harness::suite::{SuiteInfo, TrialInfo, TrialKey};

use crate::trial::{TrialIdentity, TrialKind, TrialStream, sequence};

/// Where a [`RerunSink`] writes and what it calls the run.
///
/// `#[non_exhaustive]`: build one with [`RerunConfig::new`], so a later
/// option is not a breaking change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RerunConfig {
    /// The directory the trial files are written to, created if absent. Give
    /// each run a directory of its own: a file is named for its run label,
    /// environment and trial, and a second run with the same three replaces
    /// it. A run resumed from a checkpoint is given the same directory, and
    /// rewrites only the trials it runs again.
    pub output_dir: PathBuf,
    /// A short name for what is being run, usually the algorithm (`"ppo"`).
    /// It is part of every path in the recording, so the files of two runs
    /// opened together sit side by side instead of on top of one another.
    pub run_label: String,
}

impl RerunConfig {
    /// A configuration that writes to `output_dir` under `run_label`.
    #[must_use]
    pub fn new(output_dir: impl Into<PathBuf>, run_label: impl Into<String>) -> Self {
        Self {
            output_dir: output_dir.into(),
            run_label: run_label.into(),
        }
    }
}

/// Why a [`RerunSink`] could not be built.
///
/// A failure after that, while a run is under way, does not stop the run: it
/// is logged through `tracing`, and the trial it struck has no recording in
/// the report.
#[derive(Debug)]
#[non_exhaustive]
pub enum RerunError {
    /// The output directory does not exist and could not be created.
    OutputDir {
        /// The directory asked for.
        path: PathBuf,
        /// What the operating system said.
        source: std::io::Error,
    },
}

impl std::fmt::Display for RerunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutputDir { path, source } => write!(
                f,
                "could not create the output directory {}: {source}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for RerunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OutputDir { source, .. } => Some(source),
        }
    }
}

/// The open trials, by key. Shared, so that a tracing layer can find the
/// stream of the trial an event belongs to.
type Trials = Arc<Mutex<HashMap<TrialKey, Arc<TrialStream>>>>;

/// A [`Reporter`] that records a run as rerun `.rrd` files, one per trial.
///
/// See the [crate docs](crate) for what a run writes. Build one, hand it to
/// `Evaluator::run_suite` or `run_trials` (alone, or in a
/// [`MultiReporter`](rlevo_harness::reporter::MultiReporter) beside a
/// `JsonReporter`), and read [`viewer_instructions`](Self::viewer_instructions)
/// afterwards.
///
/// The harness records each trial's file in its report
/// ([`TrialReport::recording_path`]), so the JSON a run leaves says where the
/// recordings are.
#[derive(Debug)]
pub struct RerunSink {
    config: RerunConfig,
    sdk_version: String,
    run_id: String,
    suite: String,
    trials: Trials,
    /// The kind of the last trial seen to reach a boundary: what a trial that
    /// ended before its first is opened as.
    seen_kind: Option<TrialKind>,
    written: Vec<PathBuf>,
    metric_names: BTreeSet<String>,
    histograms_skipped: BTreeSet<String>,
    failures: u64,
}

impl RerunSink {
    /// A sink that writes where `config` says.
    ///
    /// # Errors
    ///
    /// Returns [`RerunError::OutputDir`] if the output directory does not
    /// exist and cannot be created.
    pub fn new(config: RerunConfig) -> Result<Self, RerunError> {
        std::fs::create_dir_all(&config.output_dir).map_err(|source| RerunError::OutputDir {
            path: config.output_dir.clone(),
            source,
        })?;
        Ok(Self {
            config,
            sdk_version: rerun::external::re_build_info::CrateVersion::LOCAL.to_string(),
            run_id: String::new(),
            suite: String::new(),
            trials: Trials::default(),
            seen_kind: None,
            written: Vec::new(),
            metric_names: BTreeSet::new(),
            histograms_skipped: BTreeSet::new(),
            failures: 0,
        })
    }

    /// The files written so far, in the order their trials ended.
    #[must_use]
    pub fn files(&self) -> &[PathBuf] {
        &self.written
    }

    /// How to open what was written: the command that installs the rerun
    /// viewer at the version that wrote the files, and the command that opens
    /// them. `None` before any trial has ended.
    ///
    /// A viewer is only sure to open recordings of versions next to its own,
    /// so the install commands name the version exactly. The viewer is told
    /// to listen on this machine alone.
    #[must_use]
    pub fn viewer_instructions(&self) -> Option<String> {
        if self.written.is_empty() {
            return None;
        }
        let version = &self.sdk_version;
        let dir = self.config.output_dir.display();
        Some(format!(
            "{} recordings in {dir}, written by rerun {version}\n\
             install the viewer:  cargo binstall rerun-cli@{version}\n\
             or build it:         cargo install rerun-cli --version ={version} --locked\n\
             open the run:        rerun --bind 127.0.0.1 {dir}/*.rrd",
            self.written.len()
        ))
    }

    fn open_trial(&self, key: TrialKey) -> Option<Arc<TrialStream>> {
        self.trials.lock().get(&key).cloned()
    }

    /// What a trial is opened as when it ends before its first boundary.
    fn fallback_kind(&self) -> TrialKind {
        self.seen_kind.unwrap_or(TrialKind::Episodic)
    }
}

impl Reporter for RerunSink {
    fn on_suite_start(&mut self, suite: &SuiteInfo) {
        self.run_id.clone_from(&suite.run_id);
        self.suite.clone_from(&suite.name);
    }

    fn on_trial_start(&mut self, trial: &TrialInfo) {
        let identity = TrialIdentity {
            run_id: self.run_id.clone(),
            suite: self.suite.clone(),
            label: self.config.run_label.clone(),
            env_id: trial.env_id.clone(),
            env_idx: trial.key.env_idx,
            trial_idx: trial.key.trial_idx,
            seed: trial.trial_seed,
            sdk_version: self.sdk_version.clone(),
        };
        let stream = Arc::new(TrialStream::new(identity, &self.config.output_dir));
        self.trials.lock().insert(trial.key, stream);
    }

    fn on_episode(&mut self, trial: &TrialInfo, ep: &EpisodeSummary) {
        self.seen_kind = Some(TrialKind::Episodic);
        let Some(entry) = self.open_trial(trial.key) else {
            return;
        };
        let Some(stream) = entry.stream(TrialKind::Episodic) else {
            return;
        };
        // The trial's step count at this episode's end comes with the
        // metrics read that follows it, and only the harness knows it: a
        // training episode the budget cut short adds steps and is never
        // delivered here. So the episode waits for that read. One still
        // waiting was followed by no read, and is logged without a step.
        if let Some(unread) = entry.hold(ep.clone()) {
            entry.log_episode(stream, &unread, None);
        }
    }

    fn on_metrics(&mut self, trial: &TrialInfo, at: Boundary, metrics: &[Metric]) {
        let (kind, times): (TrialKind, Vec<(&'static str, i64)>) = match at {
            Boundary::TrainingEpisode { index, step } => (
                TrialKind::Episodic,
                vec![
                    ("episode", sequence(index as u64)),
                    ("step", sequence(step)),
                ],
            ),
            Boundary::EvaluationEpisode { index, step } => (
                TrialKind::Episodic,
                vec![
                    ("eval_episode", sequence(index as u64)),
                    ("step", sequence(step)),
                ],
            ),
            Boundary::Generation { generation } => (
                TrialKind::Generation,
                vec![("generation", sequence(generation as u64))],
            ),
            // A boundary this version does not know has no timeline to log on.
            _ => return,
        };
        self.seen_kind = Some(kind);
        let Some(entry) = self.open_trial(trial.key) else {
            return;
        };
        let Some(stream) = entry.stream(kind) else {
            return;
        };
        if let Boundary::TrainingEpisode { step, .. } | Boundary::EvaluationEpisode { step, .. } =
            at
            && let Some(episode) = entry.take_held()
        {
            entry.log_episode(stream, &episode, Some(step));
        }
        for metric in metrics {
            let (name, value) = match metric {
                Metric::Scalar { name, value } => (name, *value),
                // Exact below 2^53.
                #[allow(clippy::cast_precision_loss)]
                Metric::Counter { name, count } => (name, *count as f64),
                // Every sample since the trial began, at every read: logging
                // it as it is would grow with the square of the run.
                Metric::Histogram { name, .. } => {
                    if self.histograms_skipped.insert(name.clone()) {
                        tracing::warn!(
                            target: "rlevo_rerun",
                            metric = %name,
                            "histogram metrics are not recorded"
                        );
                    }
                    continue;
                }
            };
            if !self.metric_names.contains(name) {
                self.metric_names.insert(name.clone());
            }
            entry.log_scalar(stream, entry.metric_path(name, None), &times, value);
        }
    }

    fn recording(&self, trial: &TrialInfo) -> Option<Recording> {
        let entry = self.open_trial(trial.key)?;
        // A trial that ended before its first boundary has no stream yet;
        // it still gets a file, for its status and properties.
        entry.stream(self.fallback_kind())?;
        Some(Recording::new(entry.path(), self.sdk_version.clone()))
    }

    fn on_trial_end(&mut self, trial: &TrialInfo, report: &TrialReport) {
        let Some(entry) = self.trials.lock().remove(&trial.key) else {
            return;
        };
        let Some(stream) = entry.stream(self.fallback_kind()) else {
            return;
        };
        if let Some(unread) = entry.take_held() {
            entry.log_episode(stream, &unread, None);
        }
        if report.errored {
            let message = report.error_message.as_deref().unwrap_or("unknown error");
            entry.log_status(stream, &format!("errored: {message}"));
        }
        let mut facts = vec![
            ("errored", report.errored.to_string()),
            // The report keeps evaluation episodes; training ones reach sinks
            // through `on_episode` alone.
            ("eval_episodes", report.episodes.len().to_string()),
        ];
        if let Some(hash) = &report.task_hash {
            facts.push(("task_hash", hash.clone()));
        }
        if let Some(build) = &report.provenance {
            facts.push(("rlevo_version", build.rlevo_version.clone()));
            facts.push(("platform", build.platform.clone()));
            if let Some(commit) = &build.git_commit {
                facts.push(("git_commit", commit.clone()));
            }
            if let Some(dirty) = build.git_dirty {
                facts.push(("git_dirty", dirty.to_string()));
            }
            if let Some(rustc) = &build.rustc_version {
                facts.push(("rustc_version", rustc.clone()));
            }
        }
        entry.send_properties(stream, &facts);
        entry.flush(stream);
        self.failures += entry.failures();
        self.written.push(entry.path().to_path_buf());
    }

    fn on_suite_end(&mut self, _report: &BenchmarkReport) {
        // A view rooted at a metric shows everything below it, so a metric
        // whose name is another's prefix would draw the other's trials too.
        for name in &self.metric_names {
            let below = format!("{name}/");
            if let Some(longer) = self.metric_names.iter().find(|n| n.starts_with(&below)) {
                tracing::warn!(
                    target: "rlevo_rerun",
                    metric = %name,
                    also = %longer,
                    "one metric's name is a prefix of another's; a view of the first shows both"
                );
            }
        }
        if self.failures > 0 {
            tracing::warn!(
                target: "rlevo_rerun",
                failures = self.failures,
                "some values could not be logged; see the earlier warnings"
            );
        }
        if let Some(how) = self.viewer_instructions() {
            tracing::info!(target: "rlevo_rerun", "{how}");
        }
    }
}
