//! [`RerunSink`], its configuration and its error.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use rlevo_harness::metrics::Metric;
use rlevo_harness::report::{BenchmarkReport, EpisodeSummary, TrialReport};
use rlevo_harness::reporter::{Boundary, Reporter};
use rlevo_harness::suite::{SuiteInfo, TrialInfo, TrialKey};

use crate::registry::{self, RunState};
use crate::trial::{TrialIdentity, TrialKind, TrialStream, sequence};

/// Where a [`RerunSink`] writes and what it calls the run.
///
/// `#[non_exhaustive]`: build one with [`RerunConfig::new`], so a later
/// option is not a breaking change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RerunConfig {
    /// The directory recordings are written under, created if absent. Each
    /// run gets a directory of its own inside it, named for the run's ID
    /// ([`BenchmarkReport::run_id`]), so one sink may record run after run,
    /// and a run resumed from a checkpoint finds the files it began.
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
/// is logged through `tracing`, and the trial it struck has no file in
/// [`RerunSink::files`].
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

/// `path` as one word of a POSIX shell command: as it is when every
/// character is safe unquoted, and in single quotes otherwise, so a
/// directory with a space in its name is still one argument.
fn shell_quoted(path: &str) -> String {
    let plain = |c: char| c.is_ascii_alphanumeric() || "/._-+:@%,=".contains(c);
    if !path.is_empty() && path.chars().all(plain) {
        path.to_string()
    } else {
        format!("'{}'", path.replace('\'', "'\\''"))
    }
}

/// The run a sink is recording.
#[derive(Debug)]
struct Run {
    id: String,
    dir: PathBuf,
    state: Arc<RunState>,
    /// The process's count of stray update events when the run began.
    strays_before: u64,
}

/// A trial that has ended: what `on_trial_end` writes last.
#[derive(Debug)]
struct Ended {
    stream: Arc<TrialStream>,
    /// The status entry of a trial that errored.
    status: Option<String>,
    /// Facts about the trial known only at its end.
    facts: Vec<(&'static str, String)>,
}

/// A [`Reporter`] that records a run as rerun `.rrd` files, one per trial.
///
/// See the [crate docs](crate) for what a run writes. Build one, hand it to
/// `Evaluator::run_suite` or `run_trials` (alone, or in a
/// [`MultiReporter`](rlevo_harness::reporter::MultiReporter) beside a
/// `JsonReporter`), and read [`viewer_instructions`](Self::viewer_instructions)
/// afterwards.
///
/// A run's files are in `<output_dir>/<run ID>/`, and the run ID is in the
/// harness's report ([`BenchmarkReport::run_id`]): that is what joins a
/// report's JSON to its recordings. One sink may record run after run. A
/// run is no longer being recorded when the next begins or the sink is
/// dropped.
#[derive(Debug)]
pub struct RerunSink {
    config: RerunConfig,
    sdk_version: String,
    suite: String,
    /// The run being recorded: its ID, its directory, and its state in the
    /// process's registry, where the tracing layer finds its trials. `None`
    /// before the first run, and for a run whose directory could not be made.
    run: Option<Run>,
    /// `false` when rerun is switched off (`RERUN=off`): nothing is recorded.
    enabled: bool,
    /// The kind of this run's trials, once one has reached an episode or a
    /// generation.
    seen_kind: Option<TrialKind>,
    /// Trials that ended before their first episode or generation, while no
    /// trial of the run had reached one. Nothing said what kind they were,
    /// so their files wait for the end of the suite, when the run's other
    /// trials have said.
    kindless: Vec<Ended>,
    written: Vec<PathBuf>,
    /// The environment IDs this run's suite has more than once.
    repeated_envs: BTreeSet<String>,
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
            suite: String::new(),
            run: None,
            // rerun's own reading of the `RERUN` environment variable. With
            // it off, a stream opens without error and writes nothing, so
            // the sink must not report a file for it.
            enabled: rerun::decide_logging_enabled(true),
            seen_kind: None,
            kindless: Vec::new(),
            written: Vec::new(),
            repeated_envs: BTreeSet::new(),
            metric_names: BTreeSet::new(),
            histograms_skipped: BTreeSet::new(),
            failures: 0,
        })
    }

    /// The files of the current run written so far, in the order their
    /// trials ended. A trial that ended before any trial of the run had
    /// reached an episode or a generation is written when the suite ends,
    /// and listed then.
    #[must_use]
    pub fn files(&self) -> &[PathBuf] {
        &self.written
    }

    /// How to open what was written: the command that installs the rerun
    /// viewer at the version that wrote the files, and the command that opens
    /// the run's directory, which for a resumed run holds the files of the
    /// earlier run too. `None` before any trial has ended.
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
        let dir = self.run.as_ref()?.dir.display().to_string();
        let quoted = shell_quoted(&dir);
        Some(format!(
            "wrote {} recordings to {dir} with rerun {version}\n\
             install the viewer:  cargo binstall rerun-cli@{version}\n\
             or build it:         cargo install rerun-cli --version ={version} --locked\n\
             open the run:        rerun --bind 127.0.0.1 {quoted}/*.rrd",
            self.written.len()
        ))
    }

    /// Update-rate events the [`UpdateLayer`](crate::UpdateLayer) saw during
    /// this run and could not record: events of this run whose trial was not
    /// open (it had ended), and events emitted outside any trial (by work a
    /// trial handed to another thread, say). The second kind names no run,
    /// so with several runs at once each sink counts them all. `0` in a run
    /// whose every event was recorded.
    #[must_use]
    pub fn unattributed_updates(&self) -> u64 {
        self.run.as_ref().map_or(0, |run| {
            run.state.late.load(Ordering::Relaxed) + (registry::strays() - run.strays_before)
        })
    }

    fn open_trial(&self, key: TrialKey) -> Option<Arc<TrialStream>> {
        self.run.as_ref()?.state.trials.lock().get(&key).cloned()
    }

    /// Stops recording the current run, if there is one.
    fn leave_run(&mut self) {
        if let Some(run) = self.run.take() {
            registry::unregister(&run.id, &run.state);
        }
    }

    /// Writes the end of a trial whose stream is open, and closes it.
    fn finish(&mut self, ended: &Ended, stream_kind: TrialKind) {
        let entry = &ended.stream;
        let Some(stream) = entry.stream(stream_kind) else {
            return;
        };
        if let Some(unread) = entry.take_held() {
            entry.log_episode(stream, &unread, None);
        }
        if let Some(status) = &ended.status {
            entry.log_status(stream, status);
        }
        entry.send_properties(stream, &ended.facts);
        entry.flush(stream);
        self.failures += entry.failures();
        self.written.push(entry.path().to_path_buf());
    }
}

impl Reporter for RerunSink {
    fn on_suite_start(&mut self, suite: &SuiteInfo) {
        self.suite.clone_from(&suite.name);
        // A sink may be handed to one run after another; what it knows of a
        // run is that run's.
        self.leave_run();
        self.seen_kind = None;
        self.kindless.clear();
        self.written.clear();
        self.metric_names.clear();
        self.histograms_skipped.clear();
        self.failures = 0;
        let mut seen = BTreeSet::new();
        self.repeated_envs = suite
            .env_ids
            .iter()
            .filter(|id| !seen.insert(id.as_str()))
            .cloned()
            .collect();
        if !self.enabled {
            tracing::warn!(
                target: "rlevo_rerun",
                "rerun is switched off (the RERUN environment variable); nothing is recorded"
            );
            return;
        }
        let dir = self.config.output_dir.join(&suite.run_id);
        if let Err(error) = std::fs::create_dir_all(&dir) {
            tracing::warn!(
                target: "rlevo_rerun",
                path = %dir.display(),
                %error,
                "could not create the run's directory; the run is not recorded"
            );
            return;
        }
        self.run = Some(Run {
            id: suite.run_id.clone(),
            dir,
            state: registry::register(&suite.run_id),
            strays_before: registry::strays(),
        });
    }

    fn on_trial_start(&mut self, trial: &TrialInfo) {
        let Some(run) = &self.run else {
            return;
        };
        let identity = TrialIdentity {
            run_id: run.id.clone(),
            suite: self.suite.clone(),
            label: self.config.run_label.clone(),
            env_name: if self.repeated_envs.contains(&trial.env_id) {
                format!("{}-{}", trial.key.env_idx, trial.env_id)
            } else {
                trial.env_id.clone()
            },
            env_id: trial.env_id.clone(),
            env_idx: trial.key.env_idx,
            trial_idx: trial.key.trial_idx,
            seed: trial.trial_seed,
            sdk_version: self.sdk_version.clone(),
        };
        let stream = Arc::new(TrialStream::new(identity, &run.dir));
        run.state.trials.lock().insert(trial.key, stream);
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
        {
            entry.note_step(step);
            if let Some(episode) = entry.take_held() {
                entry.log_episode(stream, &episode, Some(step));
            }
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

    fn on_trial_end(&mut self, trial: &TrialInfo, report: &TrialReport) {
        let Some(run) = &self.run else {
            return;
        };
        let Some(entry) = run.state.trials.lock().remove(&trial.key) else {
            return;
        };
        let status = report.errored.then(|| {
            let message = report.error_message.as_deref().unwrap_or("unknown error");
            format!("errored: {message}")
        });
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
        let ended = Ended {
            stream: entry,
            status,
            facts,
        };
        match (ended.stream.opened(), self.seen_kind) {
            // Opening failed: the trial is not recorded.
            (Some(false), _) => {}
            // Every trial of a run is of one kind, so a kind any of them
            // has shown is this one's too, whether or not it got as far as
            // an episode or a generation itself.
            (_, Some(kind)) => self.finish(&ended, kind),
            // No trial has yet: nothing says what kind this one was. The
            // application ID follows the kind, and a file under the wrong
            // one is outside the run's recording, so it waits for the
            // suite's end, when a sibling will have said.
            (_, None) => self.kindless.push(ended),
        }
    }

    fn on_suite_end(&mut self, _report: &BenchmarkReport) {
        // Every trial of a run is of one kind. If none reached an episode or
        // a generation, the run says nothing either, and its files are filed
        // as episodic: arbitrary, but together.
        let kind = self.seen_kind.unwrap_or(TrialKind::Episodic);
        for ended in std::mem::take(&mut self.kindless) {
            self.finish(&ended, kind);
        }
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
        let unattributed = self.unattributed_updates();
        if unattributed > 0 && self.enabled {
            tracing::warn!(
                target: "rlevo_rerun",
                unattributed,
                "update events were emitted outside any trial of this run and are not recorded"
            );
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

impl Drop for RerunSink {
    fn drop(&mut self) {
        self.leave_run();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suite(run_id: &str) -> SuiteInfo {
        SuiteInfo::new(
            "suite".to_string(),
            vec!["env".to_string()],
            1,
            None,
            run_id.to_string(),
        )
    }

    /// A run is in the process's registry, where the tracing layer looks,
    /// only while a sink is recording it: not once the sink has moved to
    /// the next run, and not once the sink is gone. Otherwise every run a
    /// long-lived process ever recorded would stay in memory.
    #[test]
    fn a_run_is_registered_only_while_it_is_recorded() {
        if !rerun::decide_logging_enabled(true) {
            return;
        }
        let dir = std::env::temp_dir().join(format!("rlevo-rerun-registry-{}", std::process::id()));
        let mut sink = RerunSink::new(RerunConfig::new(&dir, "x")).expect("a sink");
        assert!(registry::recording("00000000000000c1").is_empty());

        sink.on_suite_start(&suite("00000000000000c1"));
        assert_eq!(registry::recording("00000000000000c1").len(), 1);

        sink.on_suite_start(&suite("00000000000000c2"));
        assert!(registry::recording("00000000000000c1").is_empty());
        assert_eq!(registry::recording("00000000000000c2").len(), 1);

        drop(sink);
        assert!(registry::recording("00000000000000c2").is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
