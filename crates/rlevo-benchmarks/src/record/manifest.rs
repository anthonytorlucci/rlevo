//! Run-level manifest written atomically as `run.toml` at run end.
//!
//! [`RunManifest`] carries the metadata a report-tier loader needs to
//! interpret a recording directory: the run id, seed, environment family,
//! timestamps, episode count, frame stride, and optional hyperparameters.
//!
//! [`RunManifest::write_atomic`] writes via a tmp-then-rename strategy so
//! the reader always observes either the previous `run.toml` or the new one,
//! never a partial write.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::schema::{CheckpointRef, EnvFamily, FORMAT_VERSION, Hyperparameters, RunId};

/// Run-level metadata written once per recording at suite end.
///
/// `Eq` is intentionally not derived: `success_threshold` and the metrics
/// carried by [`CheckpointRef`] are `f64`, which is not `Eq`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    /// Unique identifier for this recording run.
    pub run_id: RunId,
    /// RNG seed used for the run, preserved for reproducibility.
    pub seed: u64,
    /// Environment family recorded in this run.
    pub env_family: EnvFamily,
    /// Unix timestamp (seconds) when the run was opened.
    pub created_at: i64,
    /// Unix timestamp (seconds) when the run was finalised.
    pub finished_at: i64,
    /// Total number of episodes captured.
    pub episode_count: u32,
    /// Frame decimation factor applied by the writer.
    pub frame_stride: u16,
    /// Wire-format version used when writing this recording.
    pub format_version: u16,
    /// Agent / algorithm hyperparameters logged alongside the run.
    #[serde(default)]
    pub hyperparameters: Hyperparameters,
    /// Algorithm identity (e.g. `"ppo"`, `"dqn"`, `"ga"`). Lets the report
    /// tier choose loss panels without grepping `hyperparameters`. `None`
    /// if the producer did not declare it. Added in `FORMAT_VERSION = 6`.
    #[serde(default)]
    pub algorithm: Option<String>,
    /// `rlevo` crate version (`env!("CARGO_PKG_VERSION")`). Added in v6.
    #[serde(default)]
    pub rlevo_version: Option<String>,
    /// Rust toolchain version string (`rustc -V`). Added in v6.
    #[serde(default)]
    pub rustc_version: Option<String>,
    /// Resolved `burn` dependency version. Added in v6.
    #[serde(default)]
    pub burn_version: Option<String>,
    /// `std::env::consts::OS` + `ARCH`. Added in v6.
    #[serde(default)]
    pub platform: Option<String>,
    /// Git commit hash of the build, if known. Added in v6.
    #[serde(default)]
    pub git_commit: Option<String>,
    /// Whether the working tree had uncommitted changes at build time. Added in v6.
    #[serde(default)]
    pub git_dirty: Option<bool>,
    /// Backend device descriptor (CPU/GPU). Added in v6.
    #[serde(default)]
    pub device: Option<String>,
    /// Distinct seeds used across the trial suite (for IQM/CI aggregation). Added in v6.
    #[serde(default)]
    pub num_seeds: Option<u32>,
    /// Threshold that produced `success_rate`, lifted from `EvaluatorConfig`. Added in v6.
    #[serde(default)]
    pub success_threshold: Option<f64>,
    /// Deep-RL learner checkpoints (Burn-`Recorder` files referenced, never
    /// embedded). Empty for EA and un-wired RL. Added in v6.
    #[serde(default)]
    pub checkpoints: Vec<CheckpointRef>,
}

impl RunManifest {
    /// Constructs a fresh manifest. `frame_stride` is the resolved
    /// value (per-family default + per-run override) the writer
    /// actually applied.
    #[must_use]
    pub fn new(run_id: RunId, seed: u64, env_family: EnvFamily, frame_stride: u16) -> Self {
        Self {
            run_id,
            seed,
            env_family,
            created_at: 0,
            finished_at: 0,
            episode_count: 0,
            frame_stride,
            format_version: FORMAT_VERSION,
            hyperparameters: BTreeMap::new(),
            algorithm: None,
            rlevo_version: None,
            rustc_version: None,
            burn_version: None,
            platform: None,
            git_commit: None,
            git_dirty: None,
            device: None,
            num_seeds: None,
            success_threshold: None,
            checkpoints: Vec::new(),
        }
    }

    /// Atomically write the manifest to `dir/run.toml`. Writes to
    /// `dir/.run.toml.tmp` first, fsyncs, then renames — so a reader
    /// observes either the previous file or the new one, never a
    /// partial write.
    ///
    /// # Errors
    ///
    /// Returns any IO error from `create_dir_all`, `File::create`,
    /// `write_all`, `sync_all`, or `rename`. Returns
    /// `io::Error::other` on `toml::ser::Error`.
    pub fn write_atomic(&self, dir: &Path) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        let tmp = dir.join(".run.toml.tmp");
        let final_path = dir.join("run.toml");
        let body = toml::to_string_pretty(self).map_err(io::Error::other)?;
        {
            let mut f = File::create(&tmp)?;
            f.write_all(body.as_bytes())?;
            f.sync_all()?;
        }
        fs::rename(&tmp, &final_path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::schema::{CheckpointFormat, CheckpointKind};
    use tempfile::tempdir;

    fn sample_manifest() -> RunManifest {
        let mut hp = BTreeMap::new();
        hp.insert("lr".into(), "3e-4".into());
        hp.insert("clip_eps".into(), "0.2".into());
        RunManifest {
            created_at: 1_700_000_000,
            finished_at: 1_700_000_100,
            episode_count: 4,
            hyperparameters: hp,
            // Exercise the v6 provenance + checkpoint fields through round-trip.
            algorithm: Some("ppo".into()),
            rlevo_version: Some("0.1.0".into()),
            num_seeds: Some(10),
            success_threshold: Some(195.0),
            checkpoints: vec![CheckpointRef {
                step: 1000,
                kind: CheckpointKind::Final,
                format: CheckpointFormat::NamedMpk,
                path: "checkpoints/final.mpk".into(),
                metric: Some(200.0),
                digest: Some([2u8; 16]),
            }],
            ..RunManifest::new(RunId("20260527-120000-abc123".into()), 42, EnvFamily::Classic, 1)
        }
    }

    #[test]
    fn manifest_round_trips_with_all_provenance_none_and_empty_checkpoints() {
        let dir = tempdir().unwrap();
        let m = RunManifest::new(RunId("bare".into()), 1, EnvFamily::Grids, 1);
        m.write_atomic(dir.path()).unwrap();
        let body = std::fs::read_to_string(dir.path().join("run.toml")).unwrap();
        let parsed: RunManifest = toml::from_str(&body).unwrap();
        assert_eq!(m, parsed);
        assert!(parsed.algorithm.is_none());
        assert!(parsed.checkpoints.is_empty());
    }

    #[test]
    fn write_atomic_creates_run_toml() {
        let dir = tempdir().unwrap();
        let m = sample_manifest();
        m.write_atomic(dir.path()).unwrap();
        let body = std::fs::read_to_string(dir.path().join("run.toml")).unwrap();
        let parsed: RunManifest = toml::from_str(&body).unwrap();
        assert_eq!(m, parsed);
    }

    #[test]
    fn write_atomic_leaves_no_tmp_behind_on_success() {
        let dir = tempdir().unwrap();
        sample_manifest().write_atomic(dir.path()).unwrap();
        assert!(dir.path().join("run.toml").exists());
        assert!(
            !dir.path().join(".run.toml.tmp").exists(),
            ".run.toml.tmp should have been renamed away"
        );
    }

    #[test]
    fn write_atomic_overwrites_previous_run_toml() {
        let dir = tempdir().unwrap();
        let m1 = sample_manifest();
        m1.write_atomic(dir.path()).unwrap();
        let mut m2 = sample_manifest();
        m2.episode_count = 99;
        m2.write_atomic(dir.path()).unwrap();
        let body = std::fs::read_to_string(dir.path().join("run.toml")).unwrap();
        let parsed: RunManifest = toml::from_str(&body).unwrap();
        assert_eq!(parsed.episode_count, 99);
    }
}
