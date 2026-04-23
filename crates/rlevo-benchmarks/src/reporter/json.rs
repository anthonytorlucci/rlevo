//! JSON reporter — buffers events and writes a single document at suite end.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::report::{BenchmarkReport, EpisodeRecord, TrialReport};
use crate::reporter::Reporter;
use crate::suite::{SuiteInfo, TrialInfo};

#[derive(Debug)]
pub struct JsonReporter {
    output_path: PathBuf,
    last_suite: Option<SuiteInfo>,
    last_report: Option<BenchmarkReport>,
    write_error: Option<String>,
}

impl JsonReporter {
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

    #[must_use]
    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    #[must_use]
    pub fn write_error(&self) -> Option<&str> {
        self.write_error.as_deref()
    }

    fn write_document(&self) -> io::Result<()> {
        if let Some(parent) = self.output_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
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

    fn on_episode_end(&mut self, _trial: &TrialInfo, _ep: &EpisodeRecord) {}

    fn on_trial_end(&mut self, _trial: &TrialInfo, _report: &TrialReport) {}

    fn on_suite_end(&mut self, report: &BenchmarkReport) {
        self.last_report = Some(report.clone());
        if let Err(e) = self.write_document() {
            self.write_error = Some(e.to_string());
        }
    }
}
