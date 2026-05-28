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
use std::io::{self, BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::manifest::RunManifest;
use super::schema::{
    EnvFamily, EpisodeRecord, EpisodeRecordHeader, FORMAT_VERSION, FrameRecord, MetricSample,
    PopulationSample, RunId, bincode_config, default_frame_stride,
};

/// Per-run configuration: the writer materialises this once at the
/// start and re-uses it for every episode. `frame_stride = None`
/// picks the per-family default.
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    pub frame_stride: Option<u16>,
    pub env_family: EnvFamily,
    pub seed: u64,
    pub run_id: Option<RunId>,
}

impl RecordingConfig {
    /// Minimum viable config: defaults `frame_stride` to the family
    /// default and `run_id` to a fresh timestamped id.
    #[must_use]
    pub fn new(env_family: EnvFamily, seed: u64) -> Self {
        Self {
            frame_stride: None,
            env_family,
            seed,
            run_id: None,
        }
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
/// `Arc<Mutex<dyn RecordSink>>` across rayon worker threads.
pub trait RecordSink: Send + 'static {
    fn on_episode_start(&mut self, episode_idx: u32);
    fn on_frame(&mut self, frame: FrameRecord);
    fn on_metric(&mut self, sample: MetricSample);
    fn on_episode_end(&mut self, return_value: f64, length: u32);
    fn on_run_end(&mut self, manifest: RunManifest);
    /// Push one population snapshot. EA producers call this once per
    /// generation; RL producers never call it. Default no-op so impls
    /// outside the EA path don't need a stub.
    fn on_population_sample(&mut self, _sample: PopulationSample) {}
}

/// One framed chunk in the per-episode wire stream.
///
/// **Variant ordering is wire-format-stable** — `Frame` and `Metrics`
/// keep tags 0 and 1 so v1/v2 records still decode under v3. The new
/// `Population` variant lands at tag 2.
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
            .finish()
    }
}

struct EpisodeFile {
    writer: BufWriter<File>,
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
        })
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
        };
        write_chunk_raw(&mut writer, &header)?;

        self.current = Some(EpisodeFile { writer });
        Ok(())
    }
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_secs()).unwrap_or(0))
        .unwrap_or(0)
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
        if let Some(prev) = self.current.take()
            && let Ok(f) = prev.writer.into_inner()
        {
            let _ = f.sync_all();
        }
        self.pending_metrics.clear();
        self.frames_seen = 0;
        if let Err(e) = self.open_episode_file(episode_idx) {
            tracing::warn!(target: "rlevo_benchmarks::record", error = %e, "failed to open episode file");
        }
    }

    fn on_frame(&mut self, frame: FrameRecord) {
        let stride = u64::from(self.stride.max(1));
        let idx = self.frames_seen;
        self.frames_seen = self.frames_seen.saturating_add(1);
        if !idx.is_multiple_of(stride) {
            return;
        }
        let Some(current) = self.current.as_mut() else {
            return;
        };
        if let Err(e) = write_chunk_raw(&mut current.writer, &RecordChunk::Frame(frame)) {
            tracing::warn!(target: "rlevo_benchmarks::record", error = %e, "failed to write frame chunk");
        }
    }

    fn on_metric(&mut self, sample: MetricSample) {
        self.pending_metrics.push(sample);
    }

    fn on_population_sample(&mut self, sample: PopulationSample) {
        let Some(current) = self.current.as_mut() else {
            return;
        };
        if let Err(e) =
            write_chunk_raw(&mut current.writer, &RecordChunk::Population(sample))
        {
            tracing::warn!(target: "rlevo_benchmarks::record", error = %e, "failed to write population chunk");
        }
    }

    fn on_episode_end(&mut self, _return_value: f64, _length: u32) {
        let Some(mut current) = self.current.take() else {
            return;
        };
        if !self.pending_metrics.is_empty() {
            let metrics = std::mem::take(&mut self.pending_metrics);
            if let Err(e) = write_chunk_raw(&mut current.writer, &RecordChunk::Metrics(metrics)) {
                tracing::warn!(target: "rlevo_benchmarks::record", error = %e, "failed to write metric chunk");
            }
        }
        match current.writer.into_inner() {
            Ok(mut f) => {
                let _ = f.flush();
                if let Err(e) = f.sync_all() {
                    tracing::warn!(target: "rlevo_benchmarks::record", error = %e, "fsync on episode close failed");
                }
            }
            Err(e) => {
                tracing::warn!(target: "rlevo_benchmarks::record", error = %e.error(), "buffered flush on close failed");
            }
        }
        self.episode_count = self.episode_count.saturating_add(1);
    }

    fn on_run_end(&mut self, mut manifest: RunManifest) {
        manifest.episode_count = self.episode_count;
        manifest.finished_at = now_unix();
        if let Err(e) = manifest.write_atomic(&self.dir) {
            tracing::warn!(target: "rlevo_benchmarks::record", error = %e, "manifest write failed");
        }
    }
}

/// Slim decoder used by tests and the in-process round-trip path. The
/// full random-access loader lives in [`super::super::report::replay`].
///
/// # Errors
///
/// Returns `InvalidData` if the format version doesn't match
/// `FORMAT_VERSION`, or any IO/bincode error encountered along the way.
/// Truncation past the last whole chunk returns `Ok(...)` with the
/// frames decoded up to the truncation boundary.
pub fn read_episode_record(path: &Path) -> io::Result<EpisodeRecord> {
    let mut f = File::open(path)?;
    let mut preamble = [0u8; 16];
    f.read_exact(&mut preamble)?;
    let version = u16::from_le_bytes([preamble[0], preamble[1]]);
    if !(crate::record::schema::MIN_SUPPORTED_VERSION..=FORMAT_VERSION).contains(&version) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "format version unsupported: file={version} \
                 supported={min}..={max}",
                min = crate::record::schema::MIN_SUPPORTED_VERSION,
                max = FORMAT_VERSION
            ),
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

fn read_chunk<T: for<'de> Deserialize<'de>, R: Read + Seek>(r: &mut R) -> io::Result<Option<T>> {
    let mut len_buf = [0u8; 4];
    match r.read(&mut len_buf)? {
        0 => return Ok(None),
        n if n < 4 => {
            // Partial length prefix — treat as truncation.
            return Ok(None);
        }
        _ => {}
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut bytes = vec![0u8; len];
    match r.read(&mut bytes)? {
        n if n < len => return Ok(None),
        _ => {}
    }
    let (value, _): (T, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode_config()).map_err(io::Error::other)?;
    Ok(Some(value))
}

/// In-memory [`RecordSink`] used by producer tests so they do not
/// have to touch the filesystem.
#[derive(Debug, Default)]
pub struct InMemoryRecordSink {
    pub episodes: BTreeMap<u32, EpisodeRecord>,
    pub current: Option<u32>,
    pub manifests: Vec<RunManifest>,
}

impl InMemoryRecordSink {
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
}
