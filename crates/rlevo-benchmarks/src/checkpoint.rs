//! Trial-boundary checkpointing.
//!
//! After each completed trial the evaluator serializes the partial
//! `BenchmarkReport` to `<checkpoint_dir>/<suite_name>.ckpt.json`. On resume
//! the evaluator loads the checkpoint and skips any `TrialKey` already
//! present.
//!
//! Checkpointing requires the `json` feature because serialization uses
//! `serde_json`. Without it the module still compiles but all helpers are
//! no-ops returning `None`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::report::BenchmarkReport;
use crate::suite::TrialKey;

/// Returns the canonical checkpoint file path for a suite.
///
/// Combines `dir` and `suite_name` into `<dir>/<suite_name>.ckpt.json`.
#[must_use]
pub fn checkpoint_path(dir: &Path, suite_name: &str) -> PathBuf {
    dir.join(format!("{suite_name}.ckpt.json"))
}

/// Returns the set of [`TrialKey`]s that completed without error.
///
/// Used by the evaluator at startup to skip already-finished trials when
/// resuming from a checkpoint.
#[must_use]
pub fn completed_keys(report: &BenchmarkReport) -> HashSet<TrialKey> {
    report
        .trials
        .iter()
        .filter(|t| !t.errored)
        .map(|t| t.key)
        .collect()
}

/// Loads a [`BenchmarkReport`] from `path`, or returns `None` if the file does not exist.
///
/// # Errors
///
/// Returns an error if the file cannot be read or if deserialization fails.
#[cfg(feature = "json")]
pub fn load(path: &Path) -> std::io::Result<Option<BenchmarkReport>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path)?;
    let report: BenchmarkReport = serde_json::from_slice(&bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(Some(report))
}

/// Atomically writes `report` to `path` via a tmp-then-rename strategy.
///
/// The parent directory is created if it does not already exist. The write
/// is staged to a `.ckpt.json.tmp` sibling file and renamed into place so
/// that a concurrent reader never sees a partial file.
///
/// # Errors
///
/// Returns an error if the parent directory cannot be created, the temporary
/// file cannot be written, serialization fails, or the rename fails.
#[cfg(feature = "json")]
pub fn save(path: &Path, report: &BenchmarkReport) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("ckpt.json.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        let bytes = serde_json::to_vec(report).map_err(std::io::Error::other)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Loads a [`BenchmarkReport`] from `path`, or returns `None` if the file does not exist.
///
/// This is the no-op stub compiled when the `json` feature is disabled.
/// It always returns `Ok(None)`; enable the `json` feature for real
/// deserialization.
///
/// # Errors
///
/// This stub never returns an error.
#[cfg(not(feature = "json"))]
pub fn load(_path: &Path) -> std::io::Result<Option<BenchmarkReport>> {
    Ok(None)
}

/// Atomically writes `report` to `path` via a tmp-then-rename strategy.
///
/// This is the no-op stub compiled when the `json` feature is disabled.
/// It always returns `Ok(())`; enable the `json` feature for real
/// serialization.
///
/// # Errors
///
/// This stub never returns an error.
#[cfg(not(feature = "json"))]
pub fn save(_path: &Path, _report: &BenchmarkReport) -> std::io::Result<()> {
    Ok(())
}
