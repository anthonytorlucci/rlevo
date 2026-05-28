//! Per-run manifest written as `run.toml` at the root of the recording
//! directory. Atomic write: tmp + rename so a reader never observes a
//! partial file.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::schema::{EnvFamily, FORMAT_VERSION, Hyperparameters, RunId};

/// Run-level metadata written once per recording at suite end.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: RunId,
    pub seed: u64,
    pub env_family: EnvFamily,
    pub created_at: i64,
    pub finished_at: i64,
    pub episode_count: u32,
    pub frame_stride: u16,
    pub format_version: u16,
    #[serde(default)]
    pub hyperparameters: Hyperparameters,
}

impl RunManifest {
    /// Construct a fresh manifest. `frame_stride` is the resolved
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
    use tempfile::tempdir;

    fn sample_manifest() -> RunManifest {
        let mut hp = BTreeMap::new();
        hp.insert("lr".into(), "3e-4".into());
        hp.insert("clip_eps".into(), "0.2".into());
        RunManifest {
            run_id: RunId("20260527-120000-abc123".into()),
            seed: 42,
            env_family: EnvFamily::Classic,
            created_at: 1_700_000_000,
            finished_at: 1_700_000_100,
            episode_count: 4,
            frame_stride: 1,
            format_version: FORMAT_VERSION,
            hyperparameters: hp,
        }
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
