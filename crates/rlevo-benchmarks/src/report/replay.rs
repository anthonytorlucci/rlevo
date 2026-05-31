//! Random-access loader over an on-disk recording.
//!
//! `RecordedRun::open(dir)` reads a `runs/<run_id>/` directory:
//!
//! * the `run.toml` manifest if present (synthesised from the first
//!   episode header when missing — incomplete-run tolerance per
//!   `rollout-and-replay` §2),
//! * every `episode_*.rec` file in numeric order, decoded via the
//!   [`read_episode_record`] helper (truncation-tolerant: a partially
//!   written trailing episode contributes whatever whole frames it
//!   held).
//!
//! Mismatches between the manifest's recorded `episode_count` and the
//! number of decodable files surface as [`OpenWarning::EpisodeCountMismatch`]
//! so the caller can render a banner without us treating the gap as a
//! load failure.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::record::{EnvFamily, MetricSample, RunId, RunManifest, read_episode_record};

/// Per-episode summary held in memory after a successful
/// [`RecordedRun::open`].
#[derive(Debug, Clone)]
pub struct EpisodeIndex {
    /// Episode number as parsed from the `episode_NNNNNN.rec` filename.
    pub episode: u32,
    /// Absolute path of the source `.rec` file. Frames are *not* retained
    /// in memory after load; [`RecordedRun::frame_at_or_before`] re-reads
    /// this file on demand so a long run does not hold every frame in RAM.
    pub path: PathBuf,
    /// Number of whole frames decoded.
    pub frame_count: u32,
    /// Sum of `FrameRecord::reward` across decoded frames. Reported as
    /// `f64` so a long episode does not accumulate rounding error.
    pub episode_reward: f64,
    /// Maximum step index seen in the frame stream, plus one. Mirrors
    /// the manifest's per-episode length semantics.
    pub length: u32,
    /// Per-episode metric samples (PPO loss series, EA fitness, etc.)
    /// already split out from the on-disk wire format. Small relative to
    /// the frame stream, so these are retained.
    pub metrics: Vec<MetricSample>,
}

/// A recorded benchmark run loaded from disk.
///
/// Construct with [`RecordedRun::open`]. Each episode is decoded once on
/// load to compute its summary (frame count, reward, length) and to merge
/// metrics into a per-series index; the decoded **frames are not retained**
/// — only one episode's frames are held transiently during the load pass,
/// and [`frame_at_or_before`](Self::frame_at_or_before) re-reads on demand.
/// Warnings produced during loading (synthesised manifest, episode count
/// mismatch) are accessible via [`warnings`](Self::warnings).
#[derive(Debug, Clone)]
pub struct RecordedRun {
    manifest: RunManifest,
    episodes: Vec<EpisodeIndex>,
    metrics_by_series: BTreeMap<String, Vec<MetricSample>>,
    warnings: Vec<OpenWarning>,
    dir: PathBuf,
}

impl RecordedRun {
    /// Open a recorded run rooted at `dir`. Tolerates a missing
    /// manifest (synthesises one from the first episode header) and
    /// truncated trailing episodes (decodes whole chunks up to the
    /// boundary).
    ///
    /// # Errors
    ///
    /// Returns [`OpenError::Io`] if the directory itself cannot be
    /// listed, [`OpenError::NoEpisodes`] if no `episode_*.rec` files
    /// were found, [`OpenError::ManifestParse`] when a `run.toml`
    /// exists but cannot be deserialised, or [`OpenError::EpisodeDecode`]
    /// when an episode file cannot even be opened.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, OpenError> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.is_dir() {
            return Err(OpenError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("run directory does not exist: {}", dir.display()),
            )));
        }

        let episode_paths = list_episode_files(&dir).map_err(OpenError::Io)?;
        if episode_paths.is_empty() {
            return Err(OpenError::NoEpisodes(dir.clone()));
        }

        let mut warnings = Vec::new();
        let mut episodes = Vec::with_capacity(episode_paths.len());
        let mut metrics_by_series: BTreeMap<String, Vec<MetricSample>> = BTreeMap::new();

        let mut first_header_seed: Option<u64> = None;
        let mut first_header_family: Option<EnvFamily> = None;
        let mut first_header_run_id: Option<RunId> = None;
        let mut first_header_created_at: Option<i64> = None;

        for (episode_num, path) in episode_paths {
            let decoded = read_episode_record(&path).map_err(|e| OpenError::EpisodeDecode {
                path: path.clone(),
                source: e,
            })?;

            if first_header_seed.is_none() {
                first_header_seed = Some(decoded.header.seed);
                first_header_family = Some(decoded.header.env_family);
                first_header_run_id = Some(decoded.header.run_id.clone());
                first_header_created_at = Some(decoded.header.created_at);
            }

            let episode_reward: f64 = decoded.frames.iter().map(|f| f64::from(f.reward)).sum();
            let length = decoded
                .frames
                .iter()
                .map(|f| f.step.saturating_add(1))
                .max()
                .unwrap_or(0);
            let frame_count = u32::try_from(decoded.frames.len()).unwrap_or(u32::MAX);

            for sample in &decoded.metrics {
                metrics_by_series
                    .entry(sample.name.clone())
                    .or_default()
                    .push(sample.clone());
            }

            // Summaries are computed from `decoded.frames` above; the frame
            // vector itself is intentionally dropped here rather than stored
            // so peak memory stays at one episode's frames during the pass,
            // not every episode's frames for the run's lifetime.
            episodes.push(EpisodeIndex {
                episode: episode_num,
                path,
                frame_count,
                episode_reward,
                length,
                metrics: decoded.metrics,
            });
        }

        let manifest = load_or_synthesise_manifest(
            &dir,
            first_header_run_id.unwrap_or(RunId(String::new())),
            first_header_seed.unwrap_or(0),
            first_header_family.unwrap_or(EnvFamily::Classic),
            first_header_created_at.unwrap_or(0),
            u32::try_from(episodes.len()).unwrap_or(u32::MAX),
            &mut warnings,
        )?;

        if u32::try_from(episodes.len()).unwrap_or(u32::MAX) != manifest.episode_count {
            warnings.push(OpenWarning::EpisodeCountMismatch {
                manifest_count: manifest.episode_count,
                found_count: u32::try_from(episodes.len()).unwrap_or(u32::MAX),
            });
        }

        Ok(Self {
            manifest,
            episodes,
            metrics_by_series,
            warnings,
            dir,
        })
    }

    /// Returns the run manifest, either loaded from `run.toml` or synthesised.
    #[must_use]
    pub fn manifest(&self) -> &RunManifest {
        &self.manifest
    }

    /// Returns the decoded episodes in ascending episode-number order.
    #[must_use]
    pub fn episodes(&self) -> &[EpisodeIndex] {
        &self.episodes
    }

    /// Returns per-series metric samples aggregated across all episodes.
    ///
    /// Keys are metric names (e.g. `"policy_loss"`); values are all samples
    /// for that series in the order they were emitted across episodes.
    #[must_use]
    pub fn metrics_by_series(&self) -> &BTreeMap<String, Vec<MetricSample>> {
        &self.metrics_by_series
    }

    /// Returns non-fatal conditions detected during loading, if any.
    ///
    /// An empty slice means the run was loaded without any anomalies.
    #[must_use]
    pub fn warnings(&self) -> &[OpenWarning] {
        &self.warnings
    }

    /// Returns the root directory from which this run was opened.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Find the frame whose recorded `step` is the largest value `<= step`
    /// inside `episode`. Useful for timeline scrubbing UIs (per
    /// `rollout-and-replay` §3).
    ///
    /// Frames are not held in memory after load, so this re-reads the
    /// episode's `.rec` file on demand and returns an owned frame. Returns
    /// `None` if the episode is unknown, its file cannot be re-read, or no
    /// frame satisfies the bound.
    #[must_use]
    pub fn frame_at_or_before(
        &self,
        episode: u32,
        step: u32,
    ) -> Option<crate::record::FrameRecord> {
        let ep = self.episodes.iter().find(|e| e.episode == episode)?;
        let decoded = read_episode_record(&ep.path).ok()?;
        decoded.frames.into_iter().rev().find(|f| f.step <= step)
    }
}

/// Errors that prevent a run directory from being opened.
///
/// Contrast with [`OpenWarning`], which covers non-fatal anomalies that
/// allow loading to succeed with a degraded or reconstructed manifest.
#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    /// An I/O failure while listing or reading files in the run directory.
    #[error("io error reading run directory: {0}")]
    Io(#[source] io::Error),
    /// The directory contained no `episode_*.rec` files.
    #[error("no episode_*.rec files found in {0}")]
    NoEpisodes(PathBuf),
    /// `run.toml` was present but could not be deserialized as a [`RunManifest`](crate::record::RunManifest).
    #[error("run.toml present but could not be parsed: {0}")]
    ManifestParse(#[source] toml::de::Error),
    /// An episode file could not be opened or decoded.
    #[error("could not decode {path}: {source}")]
    EpisodeDecode {
        /// Path of the episode file that triggered the error.
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// Non-fatal conditions surfaced after a successful load. UIs render
/// these as banners; the loader itself never errors on them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenWarning {
    /// `run.toml` was missing entirely. A synthetic manifest was
    /// reconstructed from the first episode header.
    ManifestSynthesised,
    /// Manifest reports a different `episode_count` than the number of
    /// files actually decoded.
    EpisodeCountMismatch {
        manifest_count: u32,
        found_count: u32,
    },
}

fn list_episode_files(dir: &Path) -> io::Result<Vec<(u32, PathBuf)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("episode_") || !name.to_ascii_lowercase().ends_with(".rec") {
            continue;
        }
        let middle = &name["episode_".len()..name.len() - ".rec".len()];
        let Ok(num) = middle.parse::<u32>() else {
            continue;
        };
        out.push((num, path));
    }
    out.sort_by_key(|(n, _)| *n);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn load_or_synthesise_manifest(
    dir: &Path,
    run_id: RunId,
    seed: u64,
    env_family: EnvFamily,
    created_at: i64,
    found_count: u32,
    warnings: &mut Vec<OpenWarning>,
) -> Result<RunManifest, OpenError> {
    let path = dir.join("run.toml");
    if path.exists() {
        let body = fs::read_to_string(&path).map_err(OpenError::Io)?;
        let m: RunManifest = toml::from_str(&body).map_err(OpenError::ManifestParse)?;
        return Ok(m);
    }
    warnings.push(OpenWarning::ManifestSynthesised);
    let mut m = RunManifest::new(
        run_id,
        seed,
        env_family,
        crate::record::default_frame_stride(env_family),
    );
    m.created_at = created_at;
    m.episode_count = found_count;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{
        FamilyPayload, FrameRecord, RecordSink, RecordWriter, RecordingConfig,
    };
    use tempfile::tempdir;

    fn frame(step: u32, reward: f32) -> FrameRecord {
        FrameRecord {
            step,
            action: vec![0],
            reward,
            ascii: Some(format!("step {step}")),
            styled: None,
            family_payload: FamilyPayload::Ascii,
        }
    }

    fn write_simple_run(root: &Path, episodes: u32, frames_per_episode: u32) -> PathBuf {
        let cfg = RecordingConfig {
            frame_stride: Some(1),
            env_family: EnvFamily::Classic,
            seed: 11,
            run_id: Some(RunId("test-run".into())),
        };
        let mut w = RecordWriter::open(root, cfg).unwrap();
        let run_dir = w.run_dir().to_path_buf();
        for ep in 0..episodes {
            w.on_episode_start(ep);
            for s in 0..frames_per_episode {
                w.on_frame(frame(s, 1.0));
            }
            w.on_metric(MetricSample {
                step: ep,
                name: "policy_loss".into(),
                value: 0.5 - f64::from(ep) * 0.1,
            });
            w.on_episode_end(f64::from(frames_per_episode), frames_per_episode);
        }
        w.on_run_end(w.manifest_template());
        run_dir
    }

    #[test]
    fn open_empty_dir_errors() {
        let dir = tempdir().unwrap();
        let res = RecordedRun::open(dir.path());
        assert!(matches!(res, Err(OpenError::NoEpisodes(_))));
    }

    #[test]
    fn open_missing_dir_errors() {
        let dir = tempdir().unwrap();
        let bogus = dir.path().join("does-not-exist");
        let res = RecordedRun::open(&bogus);
        assert!(matches!(res, Err(OpenError::Io(_))));
    }

    #[test]
    fn open_loads_episodes_and_manifest() {
        let dir = tempdir().unwrap();
        let run_dir = write_simple_run(dir.path(), 3, 4);

        let run = RecordedRun::open(&run_dir).unwrap();
        assert_eq!(run.episodes().len(), 3);
        assert_eq!(run.manifest().episode_count, 3);
        assert_eq!(run.manifest().env_family, EnvFamily::Classic);
        assert_eq!(run.manifest().seed, 11);
        assert!(run.warnings().is_empty());

        let ep0 = &run.episodes()[0];
        assert_eq!(ep0.episode, 0);
        assert_eq!(ep0.frame_count, 4);
        assert_eq!(ep0.length, 4);
        assert!((ep0.episode_reward - 4.0).abs() < 1e-9);

        assert!(run.metrics_by_series().contains_key("policy_loss"));
        assert_eq!(run.metrics_by_series()["policy_loss"].len(), 3);
    }

    #[test]
    fn missing_manifest_is_synthesised_with_warning() {
        let dir = tempdir().unwrap();
        let run_dir = write_simple_run(dir.path(), 2, 1);
        fs::remove_file(run_dir.join("run.toml")).unwrap();

        let run = RecordedRun::open(&run_dir).unwrap();
        assert_eq!(run.manifest().episode_count, 2);
        assert_eq!(run.manifest().seed, 11);
        assert!(
            run.warnings()
                .contains(&OpenWarning::ManifestSynthesised),
            "expected ManifestSynthesised warning"
        );
    }

    #[test]
    fn episode_count_mismatch_surfaces_warning() {
        let dir = tempdir().unwrap();
        let run_dir = write_simple_run(dir.path(), 3, 1);
        // Rewrite the manifest claiming 5 episodes — only 3 files exist.
        let path = run_dir.join("run.toml");
        let mut m: RunManifest = toml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        m.episode_count = 5;
        fs::write(&path, toml::to_string_pretty(&m).unwrap()).unwrap();

        let run = RecordedRun::open(&run_dir).unwrap();
        let mismatch = run.warnings().iter().any(|w| {
            matches!(
                w,
                OpenWarning::EpisodeCountMismatch {
                    manifest_count: 5,
                    found_count: 3
                }
            )
        });
        assert!(mismatch, "expected EpisodeCountMismatch warning");
    }

    #[test]
    fn malformed_manifest_errors() {
        let dir = tempdir().unwrap();
        let run_dir = write_simple_run(dir.path(), 1, 1);
        fs::write(run_dir.join("run.toml"), "not = ] toml").unwrap();

        let res = RecordedRun::open(&run_dir);
        assert!(matches!(res, Err(OpenError::ManifestParse(_))));
    }

    #[test]
    fn frame_at_or_before_finds_latest_le_step() {
        let dir = tempdir().unwrap();
        let run_dir = write_simple_run(dir.path(), 1, 10);

        let run = RecordedRun::open(&run_dir).unwrap();
        let f = run.frame_at_or_before(0, 5).expect("frame at step 5");
        assert_eq!(f.step, 5);
        let f = run.frame_at_or_before(0, 999).expect("clamp to last frame");
        assert_eq!(f.step, 9);
        assert!(run.frame_at_or_before(99, 0).is_none());
    }

    #[test]
    fn truncated_trailing_episode_loads() {
        let dir = tempdir().unwrap();
        let run_dir = write_simple_run(dir.path(), 2, 3);
        // Truncate the last episode's tail: append a bogus partial chunk.
        let ep_path = run_dir.join("episode_000001.rec");
        let mut bytes = fs::read(&ep_path).unwrap();
        bytes.extend_from_slice(&9999u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 6]);
        fs::write(&ep_path, bytes).unwrap();

        let run = RecordedRun::open(&run_dir).unwrap();
        assert_eq!(run.episodes().len(), 2);
        // Both episodes still parse — the partial chunk is dropped.
        assert!(run.episodes()[1].frame_count >= 1);
    }

    /// Frames are re-read from disk on demand rather than cached: the
    /// summary survives load, and `frame_at_or_before` still returns the
    /// correct frame content (#6).
    #[test]
    fn frame_at_or_before_rereads_from_disk_after_load() {
        let dir = tempdir().unwrap();
        let run_dir = write_simple_run(dir.path(), 1, 5);
        let run = RecordedRun::open(&run_dir).unwrap();

        // Summary is retained without holding the frame vector.
        assert_eq!(run.episodes()[0].frame_count, 5);

        // Frame content is fetched lazily from the file.
        let f = run.frame_at_or_before(0, 2).expect("frame re-read from disk");
        assert_eq!(f.step, 2);
        assert_eq!(f.ascii.as_deref(), Some("step 2"));
    }

    /// Regression lock for #6: removing an episode file after load makes
    /// `frame_at_or_before` return `None`. If frames were eagerly cached in
    /// RAM (the old behaviour) this would still return `Some`.
    #[test]
    fn frame_at_or_before_none_when_file_removed_after_load() {
        let dir = tempdir().unwrap();
        let run_dir = write_simple_run(dir.path(), 1, 3);
        let run = RecordedRun::open(&run_dir).unwrap();

        std::fs::remove_file(run_dir.join("episode_000000.rec")).unwrap();
        assert!(
            run.frame_at_or_before(0, 0).is_none(),
            "frames must be re-read from disk, not cached in memory"
        );
    }
}
