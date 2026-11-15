//! One trial's recording: its file, its stream, and where its values go.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use rerun::log::ChunkBatcherConfig;
use rerun::{EntityPath, EntityPathPart, RecordingStream, RecordingStreamBuilder};
use rlevo_harness::report::{EpisodeSummary, Phase};

/// Which loop a trial drives. The application ID follows it, because the
/// viewer keeps a layout per application ID and the two kinds want different
/// ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrialKind {
    /// An agent playing episodes.
    Episodic,
    /// A generation loop.
    Generation,
}

impl TrialKind {
    // No dots: rerun turns them into hyphens.
    fn application_id(self) -> &'static str {
        match self {
            Self::Episodic => "rlevo-episodic",
            Self::Generation => "rlevo-generation",
        }
    }
}

/// What names a trial inside a run, and the run it belongs to.
#[derive(Debug, Clone)]
pub(crate) struct TrialIdentity {
    pub(crate) run_id: String,
    pub(crate) suite: String,
    pub(crate) label: String,
    pub(crate) env_id: String,
    pub(crate) env_idx: usize,
    pub(crate) trial_idx: usize,
    pub(crate) seed: u64,
    pub(crate) sdk_version: String,
}

/// A trial's file and the stream that writes it.
///
/// The stream opens on first use, since the application ID depends on the
/// trial's kind and nothing says which it is before its first episode or
/// generation. Every method takes `&self`: the handle is shared between the
/// sink and its tracing layer.
#[derive(Debug)]
pub(crate) struct TrialStream {
    identity: TrialIdentity,
    path: PathBuf,
    /// `Some(None)` once opening has failed: it is not tried again.
    stream: OnceLock<Option<RecordingStream>>,
    /// The episode `on_episode` delivered, held until the metrics read that
    /// follows it says how many steps the trial has taken.
    pending: Mutex<Option<EpisodeSummary>>,
    /// Updates logged before the stream was open, in order: each its index,
    /// its step and its statistics. The layer cannot open the stream, since
    /// it does not know the trial's kind, so they wait for the sink to.
    early_updates: Mutex<Vec<EarlyUpdate>>,
    /// The trial's step count at its last episode boundary.
    last_step: AtomicU64,
    /// Update events logged so far: the next one's index.
    updates: AtomicU64,
    /// Log calls rerun refused.
    failures: AtomicU64,
}

/// How rows are gathered into chunks before they are written.
///
/// rerun's default writes what it has every 200 ms, which suits a live
/// viewer. Here a trial logs a row or two per entity in that time, so nearly
/// every row became a chunk of its own, and a chunk's framing is several
/// times a scalar's size: a 20-second PPO trial wrote 1.2 MB that rerun's own
/// `rrd optimize` packs into 79 KB. Ten seconds between writes gathers the
/// rows an entity logs meanwhile into one chunk.
///
/// The cost is what a killed run loses: the rows of its last ten seconds,
/// where it was the last fifth of a second. A trial that ends is flushed
/// whole. rerun's `RERUN_FLUSH_TICK_SECS` overrides the interval.
const BATCHING: ChunkBatcherConfig = ChunkBatcherConfig {
    flush_tick: Duration::from_secs(10),
    ..ChunkBatcherConfig::DEFAULT
};

/// An update logged before its trial's stream was open.
#[derive(Debug)]
struct EarlyUpdate {
    update: u64,
    step: u64,
    metrics: Vec<(&'static str, f64)>,
}

/// A step or index as rerun's sequence type.
pub(crate) fn sequence(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// `name` with every character a file name cannot safely hold replaced.
fn file_safe(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

impl TrialStream {
    pub(crate) fn new(identity: TrialIdentity, output_dir: &std::path::Path) -> Self {
        // The indices keep two environments whose names clean to the same
        // text in two files.
        let file = format!(
            "{}.{}-{}.{}.rrd",
            file_safe(&identity.label),
            identity.env_idx,
            file_safe(&identity.env_id),
            identity.trial_idx
        );
        Self {
            path: output_dir.join(file),
            identity,
            stream: OnceLock::new(),
            pending: Mutex::new(None),
            early_updates: Mutex::new(Vec::new()),
            last_step: AtomicU64::new(0),
            updates: AtomicU64::new(0),
            failures: AtomicU64::new(0),
        }
    }

    pub(crate) fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Records the trial's step count at an episode boundary.
    pub(crate) fn note_step(&self, step: u64) {
        self.last_step.store(step, Ordering::Relaxed);
    }

    /// Logs one update's statistics, each on `update` (the update's index in
    /// this trial) and on `step`: the step the event named, or, for an event
    /// that named none, the trial's step count at its last episode boundary,
    /// so the values still draw on the timeline everything else is on.
    ///
    /// The caller is the tracing layer, which does not know the trial's
    /// kind and so never opens the stream. An update that comes before the
    /// sink has opened it (a learner may update before its first episode
    /// ends) is kept, and written when the stream opens.
    pub(crate) fn log_update(&self, step: Option<u64>, metrics: Vec<(&'static str, f64)>) {
        let update = EarlyUpdate {
            update: self.updates.fetch_add(1, Ordering::Relaxed),
            step: step.unwrap_or_else(|| self.last_step.load(Ordering::Relaxed)),
            metrics,
        };
        match self.stream.get() {
            Some(Some(stream)) => self.write_update(stream, &update),
            // Opening failed: the trial is not recorded.
            Some(None) => {}
            None => self.early_updates.lock().push(update),
        }
    }

    fn write_update(&self, stream: &RecordingStream, update: &EarlyUpdate) {
        let times = [
            ("update", sequence(update.update)),
            ("step", sequence(update.step)),
        ];
        for &(name, value) in &update.metrics {
            self.log_scalar(stream, self.metric_path(name, None), &times, value);
        }
    }

    /// Whether the stream has been opened, or opening it has failed: `None`
    /// while no one has tried.
    pub(crate) fn opened(&self) -> Option<bool> {
        self.stream.get().map(Option::is_some)
    }

    pub(crate) fn failures(&self) -> u64 {
        self.failures.load(Ordering::Relaxed)
    }

    /// Holds `episode` until its step count is known, and returns the one
    /// held before it, if that was never logged.
    pub(crate) fn hold(&self, episode: EpisodeSummary) -> Option<EpisodeSummary> {
        self.pending.lock().replace(episode)
    }

    /// The episode being held, if any.
    pub(crate) fn take_held(&self) -> Option<EpisodeSummary> {
        self.pending.lock().take()
    }

    /// Logs an episode's return and length on its episode timeline, and on
    /// `step` when the trial's step count at its end is known.
    pub(crate) fn log_episode(
        &self,
        stream: &RecordingStream,
        episode: &EpisodeSummary,
        step: Option<u64>,
    ) {
        let (phase, timeline) = match episode.phase {
            Phase::Training => ("train", "episode"),
            Phase::Evaluation => ("eval", "eval_episode"),
        };
        let mut times = vec![(timeline, sequence(episode.episode_idx as u64))];
        times.extend(step.map(|step| ("step", sequence(step))));
        self.log_scalar(
            stream,
            self.metric_path("episode_return", Some(phase)),
            &times,
            episode.return_value,
        );
        // An episode's length is far below 2^53, so the cast is exact.
        #[allow(clippy::cast_precision_loss)]
        let length = episode.length as f64;
        self.log_scalar(
            stream,
            self.metric_path("episode_length", Some(phase)),
            &times,
            length,
        );
    }

    /// The trial's stream, opened as `kind` if this is its first use. `None`
    /// if the file could not be opened, which is logged once.
    pub(crate) fn stream(&self, kind: TrialKind) -> Option<&RecordingStream> {
        let stream = self.stream.get_or_init(|| self.open(kind)).as_ref()?;
        // The trial's own thread logs its updates and makes this call, so
        // none can slip in between the open and this.
        let early = std::mem::take(&mut *self.early_updates.lock());
        for update in &early {
            self.write_update(stream, update);
        }
        Some(stream)
    }

    fn open(&self, kind: TrialKind) -> Option<RecordingStream> {
        // `save` replaces a file already there, which is what a resumed run's
        // pending trial wants.
        let stream = RecordingStreamBuilder::new(kind.application_id())
            .recording_id(self.identity.run_id.as_str())
            .batcher_config(BATCHING)
            .save(&self.path);
        let stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                tracing::warn!(
                    target: "rlevo_rerun",
                    path = %self.path.display(),
                    %error,
                    "could not open the trial's recording; the trial is not recorded"
                );
                return None;
            }
        };
        // The wall clock would make two runs of one seed differ.
        stream.set_log_time_enabled(false);
        self.check(stream.send_recording_name(self.identity.suite.as_str()));
        let id = &self.identity;
        self.send_properties(
            &stream,
            &[
                ("env_id", id.env_id.clone()),
                ("env_idx", id.env_idx.to_string()),
                ("trial_idx", id.trial_idx.to_string()),
                ("seed", id.seed.to_string()),
                ("run_label", id.label.clone()),
                ("run_id", id.run_id.clone()),
                ("rerun_version", id.sdk_version.clone()),
            ],
        );
        Some(stream)
    }

    fn check<E: std::fmt::Display>(&self, result: Result<(), E>) {
        if let Err(error) = result {
            // Once per trial: a stream that fails tends to fail every call.
            if self.failures.fetch_add(1, Ordering::Relaxed) == 0 {
                tracing::warn!(
                    target: "rlevo_rerun",
                    path = %self.path.display(),
                    %error,
                    "a value could not be logged to the trial's recording"
                );
            }
        }
    }

    /// The trial's own tail of a path: run label, environment, trial index.
    /// Free text is one part each, which rerun escapes; the index, not the
    /// seed, names the trial.
    fn tail(&self) -> [EntityPathPart; 3] {
        [
            EntityPathPart::new(self.identity.label.as_str()),
            EntityPathPart::new(self.identity.env_id.as_str()),
            EntityPathPart::new(self.identity.trial_idx.to_string()),
        ]
    }

    /// `/metrics/<metric>[/<phase>]/<label>/<env>/<idx>`. A `/` in the metric
    /// name stays hierarchy, so `rl/` and `ea/` metrics group as subtrees.
    pub(crate) fn metric_path(&self, metric: &str, phase: Option<&str>) -> EntityPath {
        let mut parts = vec![EntityPathPart::new("metrics")];
        let named = parts.len();
        parts.extend(
            metric
                .split('/')
                .filter(|part| !part.is_empty())
                .map(EntityPathPart::new),
        );
        if parts.len() == named {
            parts.push(EntityPathPart::new("unnamed"));
        }
        parts.extend(phase.map(EntityPathPart::new));
        parts.extend(self.tail());
        EntityPath::new(parts)
    }

    fn rooted(&self, root: &[&str]) -> EntityPath {
        let mut parts: Vec<EntityPathPart> =
            root.iter().copied().map(EntityPathPart::new).collect();
        parts.extend(self.tail());
        EntityPath::new(parts)
    }

    /// Logs `value` at `path`, at exactly the given points in time.
    ///
    /// rerun keeps the current time per thread and per recording ID, and
    /// every trial of a run shares one ID, so the time is cleared and set
    /// here, immediately before the call that reads it, with nothing between
    /// that could let another trial's work run on this thread.
    pub(crate) fn log_scalar(
        &self,
        stream: &RecordingStream,
        path: EntityPath,
        times: &[(&'static str, i64)],
        value: f64,
    ) {
        stream.reset_time();
        for &(timeline, at) in times {
            stream.set_time_sequence(timeline, at);
        }
        self.check(stream.log(path, &rerun::Scalars::single(value)));
    }

    /// Logs a timeless text entry at `/status/<label>/<env>/<idx>`.
    pub(crate) fn log_status(&self, stream: &RecordingStream, text: &str) {
        self.check(stream.log_static(self.rooted(&["status"]), &rerun::TextDocument::new(text)));
    }

    /// Sends facts about the trial as properties of the recording, under a
    /// name of the trial's own: the files of a run merge into one recording,
    /// and a recording-level property would keep one trial's value.
    pub(crate) fn send_properties(
        &self,
        stream: &RecordingStream,
        facts: &[(&'static str, String)],
    ) {
        let mut values = rerun::AnyValues::default();
        for (name, value) in facts {
            values = values.with_component::<rerun::components::Text>(*name, [value.clone()]);
        }
        let path = EntityPath::properties().join(&self.rooted(&["trial"]));
        self.check(stream.log_static(path, &values));
    }

    /// Writes everything buffered to the file.
    pub(crate) fn flush(&self, stream: &RecordingStream) {
        self.check(stream.flush_blocking());
    }
}
