//! Streaming writer for per-episode record files plus a slim decoder
//! sufficient for round-trip tests. The full report-tier loader lives
//! in [`super::super::report::replay`].
//!
//! Wire layout per `episode_<N>.rec`:
//!
//! ```text
//! [16-byte preamble: format_version u16 LE, then 14 reserved bytes]
//! [length-prefixed bincode EpisodeRecordHeader]
//! [length-prefixed bincode RecordChunk]* until EOF
//! ```
//!
//! `RecordChunk` is one of `Frame(FrameRecord)` or
//! `Metrics(Vec<MetricSample>)`. Frames are written as they arrive;
//! metrics buffer in-memory and flush once per episode at close. This
//! way a partially-written file truncated mid-frame is still
//! decodable up to the last whole chunk.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::thread::{self, ThreadId};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::error::RecordError;
use super::manifest::RunManifest;
use super::schema::{
    EnvFamily, EpisodeRecord, EpisodeRecordHeader, FORMAT_VERSION, FrameRecord, MetricSample,
    PopulationSample, RecordedEnvFamily, RunId, TrialRef, bincode_config, default_frame_stride,
};

/// Per-run configuration: the writer materialises this once at the
/// start and re-uses it for every episode. `frame_stride = None`
/// picks the per-family default.
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    /// Frame decimation factor override; `None` uses the per-family default.
    pub frame_stride: Option<u16>,
    /// Environment family recorded in this run.
    pub env_family: EnvFamily,
    /// RNG seed used by the env / agent, preserved in the manifest.
    pub seed: u64,
    /// Explicit run id, or `None` to generate a fresh timestamped id on open.
    pub run_id: Option<RunId>,
}

impl RecordingConfig {
    /// Creates a minimal config with per-family default stride and a
    /// generated run id.
    #[must_use]
    pub fn new(env_family: EnvFamily, seed: u64) -> Self {
        Self {
            frame_stride: None,
            env_family,
            seed,
            run_id: None,
        }
    }

    /// Like [`new`](Self::new), but derives the family from an environment
    /// type that opts into [`RecordedEnvFamily`] instead of taking it as a
    /// literal. This keeps the family a single source of truth shared with
    /// any TUI config (`with_env_family(E::FAMILY)`), so the two cannot
    /// silently disagree.
    #[must_use]
    pub fn for_env<E: RecordedEnvFamily>(seed: u64) -> Self {
        Self::new(E::FAMILY, seed)
    }

    fn resolved_stride(&self) -> u16 {
        self.frame_stride
            .unwrap_or_else(|| default_frame_stride(self.env_family))
    }
}

/// Sink the three producers (env wrapper, harness reporter, tracing
/// layer) all push into. Implementations buffer / write as they like.
///
/// `Send + 'static` so producers can hold the sink behind
/// `Arc<parking_lot::Mutex<dyn RecordSink>>` across rayon worker threads.
pub trait RecordSink: Send + 'static {
    /// Called when a new episode begins; `episode_idx` is zero-based.
    fn on_episode_start(&mut self, episode_idx: u32);
    /// Pushes one captured frame to the sink.
    fn on_frame(&mut self, frame: FrameRecord);
    /// Pushes one scalar metric sample to the sink.
    fn on_metric(&mut self, sample: MetricSample);
    /// Called when an episode terminates or truncates.
    fn on_episode_end(&mut self, return_value: f64, length: u32);
    /// Called once when the run concludes; `manifest` carries finalised run-level metadata.
    fn on_run_end(&mut self, manifest: RunManifest);
    /// Pushes one population snapshot. EA producers call this once per
    /// generation; RL producers never call it. Default no-op so impls
    /// outside the EA path don't need a stub.
    fn on_population_sample(&mut self, _sample: PopulationSample) {}

    /// Take the first error the sink encountered during the run, clearing
    /// it. Writes are best-effort and non-fatal *during* the run (they log
    /// and continue); a driver **must** call this **after** the run to fail
    /// loudly instead of shipping a silently-truncated recording. Omitting
    /// the post-run check on a durable run defeats the error-as-API
    /// design — write failures (`Io`, `ConcurrentUse`, `ActionEncode`)
    /// will then pass unnoticed.
    ///
    /// The default returns `None` — sinks that cannot fail (e.g.
    /// [`InMemoryRecordSink`]) need no override.
    fn take_error(&mut self) -> Option<RecordError> {
        None
    }

    /// Report an error a *producer* detected (e.g. the
    /// [`RecordingTap`](super::env_tap::RecordingTap) failing to encode an
    /// action) so it surfaces through [`take_error`](Self::take_error)
    /// alongside the writer's own IO failures. First-error-wins.
    ///
    /// Default no-op — sinks that only ever observe their own writes (e.g.
    /// [`InMemoryRecordSink`]) need no override.
    fn record_external_error(&mut self, _error: RecordError) {}

    /// Set the trial context stamped into subsequent episode headers.
    ///
    /// Harness producers call this at each trial start so recorded
    /// episodes carry their `(env_index, trial_index)` provenance.
    /// Default no-op — non-harness producers leave episodes with
    /// `trial = None`.
    fn set_trial_context(&mut self, _trial: Option<TrialRef>) {}
}

/// One framed chunk in the per-episode wire stream.
///
/// **Variant ordering is wire-format-stable** — new variants append at
/// the end so existing bincode tags keep decoding. `Population` is at tag 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum RecordChunk {
    Frame(FrameRecord),
    Metrics(Vec<MetricSample>),
    Population(PopulationSample),
}

/// On-disk implementation of [`RecordSink`]. Opens one file per
/// episode, fsyncs on close.
pub struct RecordWriter {
    dir: PathBuf,
    cfg: RecordingConfig,
    run_id: RunId,
    stride: u16,
    created_at: i64,
    episode_count: u32,
    current: Option<EpisodeFile>,
    pending_metrics: Vec<MetricSample>,
    frames_seen: u64,
    /// First non-fatal write error seen this run; surfaced via
    /// [`RecordSink::take_error`].
    first_error: Option<RecordError>,
    /// Thread that opened the currently-open episode file, if any. Used to
    /// detect concurrent (cross-thread) use of a single-stream writer —
    /// the corruption mode when a recording harness runs with
    /// `num_threads > 1`. `None` between episodes.
    owner: Option<ThreadId>,
    /// Trial context stamped into each episode header until changed.
    /// Set by [`RecordSink::set_trial_context`]; `None` for non-harness
    /// producers.
    current_trial: Option<TrialRef>,
}

impl std::fmt::Debug for RecordWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordWriter")
            .field("dir", &self.dir)
            .field("cfg", &self.cfg)
            .field("run_id", &self.run_id)
            .field("stride", &self.stride)
            .field("created_at", &self.created_at)
            .field("episode_count", &self.episode_count)
            .field("current", &self.current.is_some())
            .field("pending_metrics", &self.pending_metrics.len())
            .field("frames_seen", &self.frames_seen)
            .field("first_error", &self.first_error)
            .field("owner", &self.owner)
            .field("current_trial", &self.current_trial)
            .finish()
    }
}

struct EpisodeFile {
    writer: BufWriter<File>,
}

/// Default root directory for recordings, honoring the `RLEVO_RUNS_DIR`
/// environment override.
///
/// Returns `RLEVO_RUNS_DIR` if set and non-empty, otherwise `"runs"`
/// (relative to the working directory, preserving prior behaviour). Pass
/// it to [`RecordWriter::open`], or use [`RecordWriter::open_default`], so
/// recordings land in one configurable place rather than wherever the
/// binary was invoked from.
#[must_use]
pub fn default_runs_dir() -> PathBuf {
    resolve_runs_dir(std::env::var_os("RLEVO_RUNS_DIR"))
}

/// Pure resolution behind [`default_runs_dir`], split out so it can be
/// unit-tested without mutating process-global environment state.
fn resolve_runs_dir(override_var: Option<std::ffi::OsString>) -> PathBuf {
    match override_var {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => PathBuf::from("runs"),
    }
}

impl RecordWriter {
    /// Open a recorder rooted at `dir/<run_id>/`. Creates the
    /// directory if it does not exist; does not touch a pre-existing
    /// `run.toml` until [`Self::finalize_manifest`] is called.
    ///
    /// # Errors
    ///
    /// Returns any IO error from `create_dir_all`.
    pub fn open(dir: impl Into<PathBuf>, cfg: RecordingConfig) -> io::Result<Self> {
        let run_id = cfg.run_id.clone().unwrap_or_else(RunId::new_now);
        let stride = cfg.resolved_stride();
        let dir = dir.into().join(&run_id.0);
        fs::create_dir_all(&dir)?;
        let created_at = now_unix();
        Ok(Self {
            dir,
            cfg,
            run_id,
            stride,
            created_at,
            episode_count: 0,
            current: None,
            pending_metrics: Vec::new(),
            frames_seen: 0,
            first_error: None,
            owner: None,
            current_trial: None,
        })
    }

    /// Open a recorder under [`default_runs_dir`] — i.e. the
    /// `RLEVO_RUNS_DIR` environment override, or `"runs"` when unset.
    /// Convenience wrapper over [`open`](Self::open) for examples and
    /// binaries that want the standard, configurable output location
    /// instead of a path relative to wherever the binary happened to run.
    ///
    /// # Errors
    ///
    /// Returns any IO error from `create_dir_all`.
    pub fn open_default(cfg: RecordingConfig) -> io::Result<Self> {
        Self::open(default_runs_dir(), cfg)
    }

    /// Detect concurrent (cross-thread) use of this single-stream writer.
    ///
    /// Returns `true` — and records a [`RecordError::ConcurrentUse`] — when
    /// an episode is open and the calling thread is not the one that opened
    /// it. Callers must bail out without performing their write so the
    /// in-flight episode is not truncated. Single-threaded sequential use
    /// (including the abandoned-episode `reset` path, which re-enters from
    /// the same thread) never trips this.
    fn reject_concurrent_use(&mut self) -> bool {
        let foreign = self.current.is_some()
            && self.owner.is_some_and(|o| o != thread::current().id());
        if foreign {
            self.record_error(RecordError::ConcurrentUse);
        }
        foreign
    }

    /// Log a write failure and retain it if it is the first this run.
    ///
    /// Keeps today's `tracing::warn!` live visibility while making the
    /// failure recoverable after the run via [`RecordSink::take_error`].
    fn record_error(&mut self, error: RecordError) {
        tracing::warn!(target: "rlevo_benchmarks::record", error = %error, "recording error");
        if self.first_error.is_none() {
            self.first_error = Some(error);
        }
    }

    /// Directory the writer writes into, after the run-id suffix has
    /// been applied.
    #[must_use]
    pub fn run_dir(&self) -> &Path {
        &self.dir
    }

    /// Resolved frame stride.
    #[must_use]
    pub fn frame_stride(&self) -> u16 {
        self.stride
    }

    /// The run id used for filenames + the manifest.
    #[must_use]
    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    /// Build a [`RunManifest`] reflecting the writer's current state.
    /// Caller may augment hyperparameters before passing to
    /// [`Self::on_run_end`].
    #[must_use]
    pub fn manifest_template(&self) -> RunManifest {
        let mut m = RunManifest::new(
            self.run_id.clone(),
            self.cfg.seed,
            self.cfg.env_family,
            self.stride,
        );
        m.created_at = self.created_at;
        m
    }

    fn open_episode_file(&mut self, episode_idx: u32) -> io::Result<()> {
        let path = self.dir.join(format!("episode_{episode_idx:06}.rec"));
        let f = File::create(&path)?;
        let mut writer = BufWriter::new(f);

        // 16-byte preamble: format_version LE + 14 zero bytes.
        let mut preamble = [0u8; 16];
        preamble[0..2].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        writer.write_all(&preamble)?;

        let header = EpisodeRecordHeader {
            format_version: FORMAT_VERSION,
            run_id: self.run_id.clone(),
            seed: self.cfg.seed,
            env_family: self.cfg.env_family,
            created_at: self.created_at,
            trial: self.current_trial,
        };
        write_chunk_raw(&mut writer, &header)?;

        self.current = Some(EpisodeFile { writer });
        self.owner = Some(thread::current().id());
        Ok(())
    }

    /// Flush any buffered metrics into the currently-open episode file and
    /// close it, fsyncing on the way out. Returns `true` if a file was
    /// actually open.
    ///
    /// Shared by [`RecordSink::on_episode_end`] (normal close) and
    /// [`RecordSink::on_episode_start`] (an episode abandoned by an early
    /// `reset`, i.e. a new episode opened before the prior one ended).
    /// Metrics buffer in memory and are only written as a single chunk at
    /// episode close; without flushing here an abandoned episode's metrics
    /// — which belong with the frames already written to its file — would
    /// be silently dropped.
    fn flush_and_close_current(&mut self) -> bool {
        let Some(mut current) = self.current.take() else {
            return false;
        };
        self.owner = None;
        if !self.pending_metrics.is_empty() {
            let metrics = std::mem::take(&mut self.pending_metrics);
            if let Err(e) = write_chunk_raw(&mut current.writer, &RecordChunk::Metrics(metrics)) {
                self.record_error(RecordError::Io {
                    context: "write metric chunk",
                    source: e,
                });
            }
        }
        match current.writer.into_inner() {
            Ok(mut f) => {
                let _ = f.flush();
                if let Err(e) = f.sync_all() {
                    self.record_error(RecordError::Io {
                        context: "fsync episode on close",
                        source: e,
                    });
                }
            }
            Err(e) => {
                self.record_error(RecordError::Io {
                    context: "flush episode buffer on close",
                    source: e.into_error(),
                });
            }
        }
        true
    }
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(0))
}

fn write_chunk_raw<T: Serialize, W: Write>(w: &mut W, value: &T) -> io::Result<()> {
    let bytes = bincode::serde::encode_to_vec(value, bincode_config()).map_err(io::Error::other)?;
    let len = u32::try_from(bytes.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "chunk exceeds u32::MAX bytes"))?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&bytes)?;
    Ok(())
}

impl RecordSink for RecordWriter {
    fn on_episode_start(&mut self, episode_idx: u32) {
        if self.reject_concurrent_use() {
            return;
        }
        // Close any still-open episode first. If the prior episode was
        // abandoned (a new `reset` before its `on_episode_end`), this
        // flushes its buffered metrics into the file before closing it —
        // otherwise they would be lost. `clear()` afterward is a defensive
        // no-op for the normal path (the flush already drained them).
        self.flush_and_close_current();
        self.pending_metrics.clear();
        self.frames_seen = 0;
        if let Err(e) = self.open_episode_file(episode_idx) {
            self.record_error(RecordError::Io {
                context: "open episode file",
                source: e,
            });
        }
    }

    fn on_frame(&mut self, frame: FrameRecord) {
        if self.reject_concurrent_use() {
            return;
        }
        let stride = u64::from(self.stride.max(1));
        let idx = self.frames_seen;
        self.frames_seen = self.frames_seen.saturating_add(1);
        if !idx.is_multiple_of(stride) {
            return;
        }
        let Some(current) = self.current.as_mut() else {
            return;
        };
        // `current` borrow ends with this call; record_error then re-borrows self.
        let res = write_chunk_raw(&mut current.writer, &RecordChunk::Frame(frame));
        if let Err(e) = res {
            self.record_error(RecordError::Io {
                context: "write frame chunk",
                source: e,
            });
        }
    }

    fn on_metric(&mut self, sample: MetricSample) {
        self.pending_metrics.push(sample);
    }

    fn on_population_sample(&mut self, sample: PopulationSample) {
        if self.reject_concurrent_use() {
            return;
        }
        let Some(current) = self.current.as_mut() else {
            return;
        };
        let res = write_chunk_raw(&mut current.writer, &RecordChunk::Population(sample));
        if let Err(e) = res {
            self.record_error(RecordError::Io {
                context: "write population chunk",
                source: e,
            });
        }
    }

    fn on_episode_end(&mut self, _return_value: f64, _length: u32) {
        if self.reject_concurrent_use() {
            return;
        }
        // Only count episodes that were actually open; a stray
        // `on_episode_end` with no current file is a no-op.
        if self.flush_and_close_current() {
            self.episode_count = self.episode_count.saturating_add(1);
        }
    }

    fn on_run_end(&mut self, mut manifest: RunManifest) {
        manifest.episode_count = self.episode_count;
        manifest.finished_at = now_unix();
        if let Err(e) = manifest.write_atomic(&self.dir) {
            self.record_error(RecordError::Io {
                context: "write run manifest",
                source: e,
            });
        }
    }

    fn take_error(&mut self) -> Option<RecordError> {
        self.first_error.take()
    }

    fn record_external_error(&mut self, error: RecordError) {
        self.record_error(error);
    }

    fn set_trial_context(&mut self, trial: Option<TrialRef>) {
        self.current_trial = trial;
    }
}

/// Slim decoder used by tests and the in-process round-trip path. The
/// full random-access loader lives in [`super::super::report::replay`].
///
/// # Errors
///
/// Returns `InvalidData` if the format version is not exactly
/// `FORMAT_VERSION`, or any IO/bincode error encountered along the way.
/// Truncation past the last whole chunk returns `Ok(...)` with the
/// frames decoded up to the truncation boundary.
pub fn read_episode_record(path: &Path) -> io::Result<EpisodeRecord> {
    let mut f = File::open(path)?;
    let mut preamble = [0u8; 16];
    f.read_exact(&mut preamble)?;
    let version = u16::from_le_bytes([preamble[0], preamble[1]]);
    if version != FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("format version unsupported: file={version} expected={FORMAT_VERSION}"),
        ));
    }

    let header: EpisodeRecordHeader = read_chunk(&mut f)?
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "missing header chunk"))?;

    let mut frames = Vec::new();
    let mut metrics = Vec::new();
    let mut population_samples = Vec::new();
    while let Some(chunk) = read_chunk::<RecordChunk, _>(&mut f)? {
        match chunk {
            RecordChunk::Frame(fr) => frames.push(fr),
            RecordChunk::Metrics(ms) => metrics.extend(ms),
            RecordChunk::Population(ps) => population_samples.push(ps),
        }
    }
    Ok(EpisodeRecord {
        header,
        frames,
        metrics,
        population_samples,
    })
}

fn read_chunk<T: for<'de> Deserialize<'de>, R: Read>(r: &mut R) -> io::Result<Option<T>> {
    let mut len_buf = [0u8; 4];
    // `Read::read` may legally return fewer bytes than requested even when
    // not at EOF, so we must loop rather than trust a single read to fill
    // the buffer (the previous bug: a short read on a valid chunk was
    // misread as truncation, silently dropping the rest of the episode).
    // A full 4-byte prefix means another chunk follows; 0 bytes is a clean
    // EOF at a chunk boundary and 1–3 bytes is a truncated prefix — both
    // stop iteration without erroring.
    if read_full(r, &mut len_buf)? != 4 {
        return Ok(None);
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut bytes = vec![0u8; len];
    if read_full(r, &mut bytes)? < len {
        // Fewer than `len` bytes before EOF → genuine truncation.
        return Ok(None);
    }
    let (value, _): (T, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode_config()).map_err(io::Error::other)?;
    Ok(Some(value))
}

/// Read repeatedly until `buf` is full or EOF is reached, returning the
/// number of bytes actually read.
///
/// Differs from [`Read::read_exact`] in that a short fill at EOF is
/// reported through the return value (`< buf.len()`) rather than raising
/// [`io::ErrorKind::UnexpectedEof`] — the record loader treats a partial
/// trailing chunk as truncation, not an error, so a crash mid-write still
/// decodes up to the last whole chunk. `Interrupted` is retried.
fn read_full<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

/// In-memory [`RecordSink`] used by producer tests so they do not
/// have to touch the filesystem.
#[derive(Debug, Default)]
pub struct InMemoryRecordSink {
    /// All accumulated episode records keyed by episode index.
    pub episodes: BTreeMap<u32, EpisodeRecord>,
    /// Episode index of the currently open episode, or `None` between episodes.
    pub current: Option<u32>,
    /// Run manifests collected via [`RecordSink::on_run_end`].
    pub manifests: Vec<RunManifest>,
    /// Last trial context set via [`RecordSink::set_trial_context`];
    /// stamped into the header of each episode opened afterward.
    pub last_trial_context: Option<TrialRef>,
    /// First error reported by a producer via
    /// [`RecordSink::record_external_error`], surfaced through
    /// [`take_error`](RecordSink::take_error). First-error-wins, matching
    /// [`RecordWriter`].
    pub first_error: Option<RecordError>,
}

impl InMemoryRecordSink {
    /// Creates an empty in-memory sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl RecordSink for InMemoryRecordSink {
    fn on_episode_start(&mut self, episode_idx: u32) {
        self.current = Some(episode_idx);
        self.episodes.insert(
            episode_idx,
            EpisodeRecord {
                header: EpisodeRecordHeader {
                    format_version: FORMAT_VERSION,
                    run_id: RunId(String::new()),
                    seed: 0,
                    env_family: EnvFamily::Classic,
                    created_at: 0,
                    trial: self.last_trial_context,
                },
                frames: Vec::new(),
                metrics: Vec::new(),
                population_samples: Vec::new(),
            },
        );
    }
    fn on_frame(&mut self, frame: FrameRecord) {
        if let Some(idx) = self.current
            && let Some(ep) = self.episodes.get_mut(&idx)
        {
            ep.frames.push(frame);
        }
    }
    fn on_metric(&mut self, sample: MetricSample) {
        if let Some(idx) = self.current
            && let Some(ep) = self.episodes.get_mut(&idx)
        {
            ep.metrics.push(sample);
        }
    }
    fn on_population_sample(&mut self, sample: PopulationSample) {
        if let Some(idx) = self.current
            && let Some(ep) = self.episodes.get_mut(&idx)
        {
            ep.population_samples.push(sample);
        }
    }
    fn on_episode_end(&mut self, _return_value: f64, _length: u32) {
        self.current = None;
    }
    fn on_run_end(&mut self, manifest: RunManifest) {
        self.manifests.push(manifest);
    }
    fn record_external_error(&mut self, error: RecordError) {
        if self.first_error.is_none() {
            self.first_error = Some(error);
        }
    }
    fn take_error(&mut self) -> Option<RecordError> {
        self.first_error.take()
    }
    fn set_trial_context(&mut self, trial: Option<TrialRef>) {
        self.last_trial_context = trial;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::render::{StyledFrame, StyledLine, StyledSpan};
    use tempfile::tempdir;

    fn frame(step: u32, reward: f32) -> FrameRecord {
        FrameRecord {
            step,
            action: vec![u8::try_from(step & 0xFF).unwrap_or(0)],
            reward,
            ascii: Some(format!("step {step}")),
            styled: Some(StyledFrame {
                lines: vec![StyledLine {
                    spans: vec![StyledSpan::raw(format!("step {step}"))],
                }],
            }),
            family_payload: crate::record::FamilyPayload::Ascii,
        }
    }

    #[test]
    fn resolve_runs_dir_honors_override_and_defaults() {
        use std::ffi::OsString;
        assert_eq!(
            resolve_runs_dir(Some(OsString::from("/tmp/rlevo-runs"))),
            PathBuf::from("/tmp/rlevo-runs")
        );
        // Empty override falls back to the default.
        assert_eq!(resolve_runs_dir(Some(OsString::new())), PathBuf::from("runs"));
        assert_eq!(resolve_runs_dir(None), PathBuf::from("runs"));
    }

    #[test]
    fn record_external_error_sets_first_error_and_take_clears() {
        let dir = tempdir().unwrap();
        let cfg = RecordingConfig {
            frame_stride: Some(1),
            env_family: EnvFamily::Classic,
            seed: 0,
            run_id: Some(RunId("ext-err-run".into())),
        };
        let mut w = RecordWriter::open(dir.path(), cfg).unwrap();
        w.record_external_error(RecordError::ActionEncode {
            step: 3,
            message: "boom".into(),
        });
        // First-error-wins: a later error does not overwrite.
        w.record_external_error(RecordError::ConcurrentUse);
        match w.take_error() {
            Some(RecordError::ActionEncode { step, .. }) => assert_eq!(step, 3),
            other => panic!("expected the first ActionEncode, got {other:?}"),
        }
        assert!(w.take_error().is_none());
    }

    #[test]
    fn in_memory_sink_captures_external_error() {
        let mut sink = InMemoryRecordSink::new();
        sink.record_external_error(RecordError::ActionEncode {
            step: 1,
            message: "boom".into(),
        });
        assert!(matches!(
            sink.take_error(),
            Some(RecordError::ActionEncode { step: 1, .. })
        ));
        assert!(sink.take_error().is_none());
    }

    #[test]
    fn round_trip_single_episode() {
        let dir = tempdir().unwrap();
        let cfg = RecordingConfig {
            frame_stride: Some(1),
            env_family: EnvFamily::Classic,
            seed: 7,
            run_id: Some(RunId("test-run".into())),
        };
        let mut w = RecordWriter::open(dir.path(), cfg).unwrap();
        w.on_episode_start(0);
        w.on_frame(frame(0, 1.0));
        w.on_frame(frame(1, 0.5));
        w.on_metric(MetricSample {
            step: 100,
            name: "policy_loss".into(),
            value: 0.01,
        });
        w.on_episode_end(1.5, 2);
        w.on_run_end(w.manifest_template());

        let ep_path = w.run_dir().join("episode_000000.rec");
        let decoded = read_episode_record(&ep_path).unwrap();
        assert_eq!(decoded.header.seed, 7);
        assert_eq!(decoded.header.format_version, FORMAT_VERSION);
        assert_eq!(decoded.frames.len(), 2);
        assert_eq!(decoded.frames[0].step, 0);
        assert_eq!(decoded.frames[1].step, 1);
        assert_eq!(decoded.metrics.len(), 1);
        assert_eq!(decoded.metrics[0].name, "policy_loss");

        // Manifest written + decodable.
        let manifest_path = w.run_dir().join("run.toml");
        let body = std::fs::read_to_string(&manifest_path).unwrap();
        let m: RunManifest = toml::from_str(&body).unwrap();
        assert_eq!(m.episode_count, 1);
        assert_eq!(m.frame_stride, 1);
    }

    #[test]
    fn frame_stride_decimates() {
        let dir = tempdir().unwrap();
        let cfg = RecordingConfig {
            frame_stride: Some(3),
            env_family: EnvFamily::Classic,
            seed: 0,
            run_id: Some(RunId("stride-run".into())),
        };
        let mut w = RecordWriter::open(dir.path(), cfg).unwrap();
        w.on_episode_start(0);
        for i in 0..10 {
            w.on_frame(frame(i, 0.0));
        }
        w.on_episode_end(0.0, 10);
        w.on_run_end(w.manifest_template());

        let decoded = read_episode_record(&w.run_dir().join("episode_000000.rec")).unwrap();
        // 10 frames, stride 3 → indexes 0,3,6,9 kept = 4 frames.
        assert_eq!(decoded.frames.len(), 4);
        assert_eq!(
            decoded.frames.iter().map(|f| f.step).collect::<Vec<_>>(),
            vec![0, 3, 6, 9]
        );
    }

    #[test]
    fn multiple_episodes_get_separate_files() {
        let dir = tempdir().unwrap();
        let cfg = RecordingConfig {
            frame_stride: Some(1),
            env_family: EnvFamily::Classic,
            seed: 0,
            run_id: Some(RunId("multi-run".into())),
        };
        let mut w = RecordWriter::open(dir.path(), cfg).unwrap();
        for ep in 0..3 {
            w.on_episode_start(ep);
            w.on_frame(frame(0, 1.0));
            w.on_episode_end(1.0, 1);
        }
        w.on_run_end(w.manifest_template());

        for ep in 0..3 {
            let path = w.run_dir().join(format!("episode_{ep:06}.rec"));
            assert!(path.exists(), "episode file {} missing", path.display());
            let decoded = read_episode_record(&path).unwrap();
            assert_eq!(decoded.frames.len(), 1);
        }
        // Manifest count matches.
        let body = std::fs::read_to_string(w.run_dir().join("run.toml")).unwrap();
        let m: RunManifest = toml::from_str(&body).unwrap();
        assert_eq!(m.episode_count, 3);
    }

    #[test]
    fn truncated_file_decodes_up_to_last_whole_frame() {
        let dir = tempdir().unwrap();
        let cfg = RecordingConfig {
            frame_stride: Some(1),
            env_family: EnvFamily::Classic,
            seed: 0,
            run_id: Some(RunId("trunc-run".into())),
        };
        let mut w = RecordWriter::open(dir.path(), cfg).unwrap();
        w.on_episode_start(0);
        w.on_frame(frame(0, 1.0));
        w.on_frame(frame(1, 2.0));
        w.on_episode_end(3.0, 2);
        drop(w);

        // Append a partial third chunk: 4-byte length prefix claiming a
        // 9999-byte payload, plus only a handful of bytes. The reader
        // must stop cleanly without erroring, returning the two valid
        // frames it already saw.
        let path = dir.path().join("trunc-run/episode_000000.rec");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(&9999u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 6]);
        std::fs::write(&path, bytes).unwrap();

        let decoded = read_episode_record(&path).unwrap();
        assert_eq!(decoded.frames.len(), 2);
    }

    #[test]
    fn in_memory_sink_round_trip() {
        let mut sink = InMemoryRecordSink::new();
        sink.on_episode_start(7);
        sink.on_frame(frame(0, 1.0));
        sink.on_metric(MetricSample {
            step: 50,
            name: "entropy".into(),
            value: 0.7,
        });
        sink.on_episode_end(1.0, 1);
        assert!(sink.episodes.contains_key(&7));
        let ep = &sink.episodes[&7];
        assert_eq!(ep.frames.len(), 1);
        assert_eq!(ep.metrics.len(), 1);
        assert_eq!(ep.metrics[0].name, "entropy");
    }

    fn population_sample(generation: u32) -> PopulationSample {
        PopulationSample {
            generation,
            fitnesses: vec![0.1, 0.2, 0.3, 0.4],
            diversity: Some(0.5),
            best_index: 0,
            best_genome_digest: None,
            parents_of_best: Vec::new(),
            inner_rl_returns: None,
        }
    }

    #[test]
    fn record_writer_persists_population_samples() {
        let dir = tempdir().unwrap();
        let cfg = RecordingConfig {
            frame_stride: Some(1),
            env_family: EnvFamily::Landscapes,
            seed: 3,
            run_id: Some(RunId("pop-run".into())),
        };
        let mut w = RecordWriter::open(dir.path(), cfg).unwrap();
        w.on_episode_start(0);
        w.on_frame(frame(0, 0.0));
        w.on_population_sample(population_sample(0));
        w.on_population_sample(population_sample(1));
        w.on_episode_end(0.0, 1);
        w.on_run_end(w.manifest_template());

        let decoded =
            read_episode_record(&w.run_dir().join("episode_000000.rec")).unwrap();
        assert_eq!(decoded.population_samples.len(), 2);
        assert_eq!(decoded.population_samples[0].generation, 0);
        assert_eq!(decoded.population_samples[1].generation, 1);
        assert_eq!(decoded.population_samples[0].fitnesses.len(), 4);
    }

    #[test]
    fn in_memory_sink_collects_population_samples() {
        let mut sink = InMemoryRecordSink::new();
        sink.on_episode_start(0);
        sink.on_population_sample(population_sample(0));
        sink.on_population_sample(population_sample(1));
        sink.on_episode_end(0.0, 0);
        let ep = &sink.episodes[&0];
        assert_eq!(ep.population_samples.len(), 2);
        assert_eq!(ep.population_samples[1].generation, 1);
    }

    // ----------------------------------------------------------------
    // Regression: short reads must not be mistaken for truncation (#2).
    // ----------------------------------------------------------------

    /// A `Read` that hands back at most `max` bytes per call, emulating the
    /// legal-but-inconvenient short reads `Read::read` may produce. With
    /// `max = 1` it is the pathological worst case the old `read_chunk`
    /// mis-handled.
    struct ThrottledReader<'a> {
        data: &'a [u8],
        pos: usize,
        max: usize,
    }

    impl Read for ThrottledReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            let remaining = self.data.len() - self.pos;
            let n = remaining.min(buf.len()).min(self.max.max(1));
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    /// Frame a chunk exactly the way the writer does: 4-byte LE length
    /// prefix followed by the bincode payload.
    fn encode_chunk(chunk: &RecordChunk) -> Vec<u8> {
        let payload = bincode::serde::encode_to_vec(chunk, bincode_config()).unwrap();
        let mut out = Vec::with_capacity(4 + payload.len());
        out.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    #[test]
    fn read_chunk_reassembles_across_one_byte_reads() {
        // Two chunks back-to-back, fed one byte at a time. Both must decode
        // and the reader must then report clean EOF — never a spurious
        // truncation in the middle.
        let mut buf = encode_chunk(&RecordChunk::Frame(frame(3, 1.5)));
        buf.extend(encode_chunk(&RecordChunk::Frame(frame(4, -2.0))));

        let mut r = ThrottledReader {
            data: &buf,
            pos: 0,
            max: 1,
        };

        match read_chunk::<RecordChunk, _>(&mut r).unwrap() {
            Some(RecordChunk::Frame(f)) => assert_eq!(f.step, 3),
            other => panic!("expected first Frame chunk, got {other:?}"),
        }
        match read_chunk::<RecordChunk, _>(&mut r).unwrap() {
            Some(RecordChunk::Frame(f)) => assert_eq!(f.step, 4),
            other => panic!("expected second Frame chunk, got {other:?}"),
        }
        assert!(
            read_chunk::<RecordChunk, _>(&mut r).unwrap().is_none(),
            "reader should report clean EOF after the last whole chunk"
        );
    }

    #[test]
    fn read_chunk_treats_real_truncation_as_eof() {
        // A full length prefix but a payload cut short before `len` bytes
        // is genuine truncation and must return None, not error.
        let chunk = encode_chunk(&RecordChunk::Frame(frame(0, 1.0)));
        let cut = &chunk[..chunk.len() - 3]; // drop 3 payload bytes
        let mut r = ThrottledReader {
            data: cut,
            pos: 0,
            max: 1,
        };
        assert!(read_chunk::<RecordChunk, _>(&mut r).unwrap().is_none());
    }

    #[test]
    fn large_frame_round_trips_through_real_file() {
        // A frame far larger than any single OS read is the realistic
        // trigger for the short-read bug on actual files.
        let dir = tempdir().unwrap();
        let cfg = RecordingConfig {
            frame_stride: Some(1),
            env_family: EnvFamily::Classic,
            seed: 0,
            run_id: Some(RunId("big-frame-run".into())),
        };
        let big_ascii = "x".repeat(512 * 1024);
        let mut w = RecordWriter::open(dir.path(), cfg).unwrap();
        w.on_episode_start(0);
        let mut f = frame(0, 1.0);
        f.ascii = Some(big_ascii.clone());
        w.on_frame(f);
        w.on_episode_end(1.0, 1);
        w.on_run_end(w.manifest_template());

        let decoded = read_episode_record(&w.run_dir().join("episode_000000.rec")).unwrap();
        assert_eq!(decoded.frames.len(), 1);
        assert_eq!(decoded.frames[0].ascii.as_deref(), Some(big_ascii.as_str()));
    }

    // ----------------------------------------------------------------
    // Regression: an episode abandoned by an early reset must still flush
    // its buffered metrics (#3).
    // ----------------------------------------------------------------

    #[test]
    fn metrics_flush_when_episode_abandoned_by_early_reset() {
        let dir = tempdir().unwrap();
        let cfg = RecordingConfig {
            frame_stride: Some(1),
            env_family: EnvFamily::Classic,
            seed: 0,
            run_id: Some(RunId("abandon-run".into())),
        };
        let mut w = RecordWriter::open(dir.path(), cfg).unwrap();

        // Episode 0: a frame + a metric, but NO on_episode_end.
        w.on_episode_start(0);
        w.on_frame(frame(0, 1.0));
        w.on_metric(MetricSample {
            step: 5,
            name: "policy_loss".into(),
            value: 0.3,
        });
        // Early reset abandons episode 0 and opens episode 1.
        w.on_episode_start(1);
        w.on_frame(frame(0, 2.0));
        w.on_episode_end(2.0, 1);
        w.on_run_end(w.manifest_template());

        let ep0 = read_episode_record(&w.run_dir().join("episode_000000.rec")).unwrap();
        assert_eq!(
            ep0.metrics.len(),
            1,
            "abandoned episode's buffered metrics must still be flushed"
        );
        assert_eq!(ep0.metrics[0].name, "policy_loss");

        // The next episode must not inherit the prior episode's metrics.
        let ep1 = read_episode_record(&w.run_dir().join("episode_000001.rec")).unwrap();
        assert!(
            ep1.metrics.is_empty(),
            "metrics buffer must reset across the reset boundary"
        );
    }

    #[test]
    fn abandoned_episode_does_not_increment_episode_count() {
        // Abandoning via on_episode_start must not count the episode; only
        // a real on_episode_end does. Episode 0 is abandoned, episode 1
        // ends normally → manifest episode_count == 1.
        let dir = tempdir().unwrap();
        let cfg = RecordingConfig {
            frame_stride: Some(1),
            env_family: EnvFamily::Classic,
            seed: 0,
            run_id: Some(RunId("count-run".into())),
        };
        let mut w = RecordWriter::open(dir.path(), cfg).unwrap();
        w.on_episode_start(0);
        w.on_frame(frame(0, 1.0));
        w.on_episode_start(1); // abandon 0
        w.on_frame(frame(0, 1.0));
        w.on_episode_end(1.0, 1); // end 1 normally
        w.on_run_end(w.manifest_template());

        let body = std::fs::read_to_string(w.run_dir().join("run.toml")).unwrap();
        let m: RunManifest = toml::from_str(&body).unwrap();
        assert_eq!(m.episode_count, 1);
    }
}
