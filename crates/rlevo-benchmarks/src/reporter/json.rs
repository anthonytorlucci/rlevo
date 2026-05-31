//! JSON-file [`Reporter`] that buffers the suite and writes one document on completion.
//!
//! [`JsonReporter`] ignores trial-level and episode-level callbacks; it
//! only captures the [`BenchmarkReport`] delivered via `on_suite_end` and
//! pretty-prints it to a configurable output path. Parent directories are
//! created automatically.
//!
//! Write failures are surfaced through [`JsonReporter::write_error`] rather
//! than propagating an error from `on_suite_end`, because [`Reporter`] trait
//! methods are infallible. Check this accessor after suite completion if
//! confirmation of a successful write is important.
//!
//! Requires the `json` feature (enables `serde` derives on report types).
//!
//! [`Reporter`]: crate::reporter::Reporter
//! [`BenchmarkReport`]: crate::report::BenchmarkReport

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::report::{BenchmarkReport, EpisodeSummary, TrialReport};
use crate::reporter::Reporter;
use crate::suite::{SuiteInfo, TrialInfo};

/// [`Reporter`] that serialises the finished suite to a JSON file.
///
/// Only the final [`BenchmarkReport`] is captured; intermediate
/// trial/episode events are discarded. The file is written atomically
/// (create → flush) in [`on_suite_end`]; if the write fails the error is
/// stored internally and retrievable via [`write_error`].
///
/// [`Reporter`]: crate::reporter::Reporter
/// [`BenchmarkReport`]: crate::report::BenchmarkReport
/// [`on_suite_end`]: crate::reporter::Reporter::on_suite_end
/// [`write_error`]: JsonReporter::write_error
#[derive(Debug)]
pub struct JsonReporter {
    output_path: PathBuf,
    last_suite: Option<SuiteInfo>,
    last_report: Option<BenchmarkReport>,
    write_error: Option<String>,
}

impl JsonReporter {
    /// Creates a `JsonReporter` that will write output to `output_path`.
    ///
    /// Parent directories of `output_path` are created on first write if
    /// they do not already exist.
    #[must_use]
    pub fn new(output_path: impl Into<PathBuf>) -> Self {
        Self {
            output_path: output_path.into(),
            last_suite: None,
            last_report: None,
            write_error: None,
        }
    }

    /// Returns the buffered report (available after `on_suite_end`).
    #[must_use]
    pub fn report(&self) -> Option<&BenchmarkReport> {
        self.last_report.as_ref()
    }

    /// Returns the path the reporter will write (or has written) the JSON file to.
    #[must_use]
    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    /// Returns the write error from `on_suite_end`, if one occurred.
    ///
    /// `None` means either the suite has not ended yet or the write succeeded.
    #[must_use]
    pub fn write_error(&self) -> Option<&str> {
        self.write_error.as_deref()
    }

    fn write_document(&self) -> io::Result<()> {
        if let Some(parent) = self.output_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let report = self
            .last_report
            .as_ref()
            .ok_or_else(|| io::Error::other("no report buffered"))?;
        let file = File::create(&self.output_path)?;
        let mut w = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut w, report).map_err(io::Error::other)?;
        w.flush()?;
        Ok(())
    }
}

impl Reporter for JsonReporter {
    fn on_suite_start(&mut self, suite: &SuiteInfo) {
        self.last_suite = Some(suite.clone());
    }

    fn on_trial_start(&mut self, _trial: &TrialInfo) {}

    fn on_episode_end(&mut self, _trial: &TrialInfo, _ep: &EpisodeSummary) {}

    fn on_trial_end(&mut self, _trial: &TrialInfo, _report: &TrialReport) {}

    fn on_suite_end(&mut self, report: &BenchmarkReport) {
        self.last_report = Some(report.clone());
        if let Err(e) = self.write_document() {
            self.write_error = Some(e.to_string());
        }
    }
}
