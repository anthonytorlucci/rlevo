//! Post-run benchmark reports and per-trial result records.
//!
//! This module defines the data structures written by the `Evaluator` during a
//! benchmark run. Each completed trial produces a [`TrialReport`], which is
//! collected into a [`BenchmarkReport`] for the whole suite.
//!
//! When the `report` feature is enabled, two additional sub-modules are
//! available:
//!
//! - `replay` — random-access loader for previously recorded runs
//! - `html` — static-HTML emitter for post-run visualisation
//!
//! [`Metric`](crate::metrics::Metric) is not directly serializable here; the
//! JSON reporter flattens metrics into plain key/value maps at serialization
//! time via [`TrialReport::absorb_metrics`].

#[cfg(feature = "report")]
pub mod html;
#[cfg(feature = "report")]
pub mod replay;

#[cfg(feature = "report")]
pub use html::{ClientAssets, EmitConfig, EmitError, EmitOutcome, emit_static_html};
#[cfg(feature = "report")]
pub use replay::{EpisodeIndex, OpenError, OpenWarning, RecordedRun};

use std::collections::BTreeMap;

use crate::suite::TrialKey;

/// Compact summary of a single episode within a trial.
///
/// Stored in [`TrialReport::episodes`] after each episode completes. The
/// `return_value` is the undiscounted sum of rewards over the episode.
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct EpisodeSummary {
    /// Zero-based index of this episode within the trial.
    pub episode_idx: usize,
    /// Undiscounted cumulative reward for the episode.
    pub return_value: f64,
    /// Number of steps taken before the episode terminated.
    pub length: usize,
}

/// All results collected for a single (env, seed) trial.
///
/// Created by the `Evaluator` for every entry in the benchmark suite and
/// populated incrementally: episodes are appended as they complete, then
/// metrics are absorbed in bulk via [`TrialReport::absorb_metrics`] when the
/// trial finishes.
///
/// Scalar, histogram, and counter metrics are stored in separate maps so that
/// downstream consumers (JSON serializer, HTML emitter) can handle each kind
/// without extra type dispatch.
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct TrialReport {
    /// Unique identifier for this trial within its suite.
    pub key: TrialKey,
    /// Human-readable name of the environment under test.
    pub env_name: String,
    /// Seed used to initialize this trial's environment and agent.
    pub trial_seed: u64,
    /// Per-episode summaries, in the order episodes were completed.
    pub episodes: Vec<EpisodeSummary>,
    /// Flattened scalar metrics (name → value).
    ///
    /// Histograms and counters are stored separately in [`histograms`](Self::histograms)
    /// and [`counters`](Self::counters).
    pub scalars: BTreeMap<String, f64>,
    /// Distribution metrics (name → sample values).
    pub histograms: BTreeMap<String, Vec<f64>>,
    /// Monotonically increasing counters (name → total count).
    pub counters: BTreeMap<String, u64>,
    /// `true` if the trial terminated due to an unrecoverable error.
    pub errored: bool,
    /// Human-readable error description when `errored` is `true`.
    pub error_message: Option<String>,
}

impl TrialReport {
    /// Creates an empty `TrialReport` for the given trial identity.
    ///
    /// All metric maps and the episode list start empty. Call
    /// [`absorb_metrics`](Self::absorb_metrics) and push to
    /// [`episodes`](Self::episodes) to populate the report during evaluation.
    #[must_use]
    pub fn new(key: TrialKey, env_name: String, trial_seed: u64) -> Self {
        Self {
            key,
            env_name,
            trial_seed,
            episodes: Vec::new(),
            scalars: BTreeMap::new(),
            histograms: BTreeMap::new(),
            counters: BTreeMap::new(),
            errored: false,
            error_message: None,
        }
    }

    /// Consumes a metric list and inserts each entry into the appropriate map.
    ///
    /// Existing entries with the same name are overwritten. Call this once
    /// after a trial finishes to transfer all collected metrics into the report.
    pub fn absorb_metrics(&mut self, metrics: Vec<crate::metrics::Metric>) {
        for m in metrics {
            match m {
                crate::metrics::Metric::Scalar { name, value } => {
                    self.scalars.insert(name, value);
                }
                crate::metrics::Metric::Histogram { name, values } => {
                    self.histograms.insert(name, values);
                }
                crate::metrics::Metric::Counter { name, count } => {
                    self.counters.insert(name, count);
                }
            }
        }
    }
}

/// Aggregated results for an entire benchmark suite run.
///
/// Constructed at the start of a run via [`BenchmarkReport::new`] and held
/// in memory while the `Evaluator` fills in individual [`TrialReport`]s.
/// Call [`finalize`](BenchmarkReport::finalize) when all trials complete to
/// mark the report as no longer in progress.
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Name of the benchmark suite, used as a display label in reports.
    pub suite_name: String,
    /// Root seed from which all trial seeds were derived.
    pub base_seed: u64,
    /// Ordered list of completed (or failed) trial results.
    pub trials: Vec<TrialReport>,
    /// `true` while trials are still running; `false` after [`finalize`](Self::finalize).
    pub in_progress: bool,
}

impl BenchmarkReport {
    /// Creates an empty, in-progress `BenchmarkReport` for a suite run.
    ///
    /// `in_progress` is set to `true` until [`finalize`](Self::finalize) is
    /// called. Push completed [`TrialReport`]s into [`trials`](Self::trials)
    /// as they become available.
    #[must_use]
    pub fn new(suite_name: String, base_seed: u64) -> Self {
        Self {
            suite_name,
            base_seed,
            trials: Vec::new(),
            in_progress: true,
        }
    }

    /// Marks the report as complete by setting `in_progress` to `false`.
    pub fn finalize(&mut self) {
        self.in_progress = false;
    }
}
