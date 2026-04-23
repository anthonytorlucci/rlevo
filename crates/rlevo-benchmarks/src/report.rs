//! Report records produced by the `Evaluator`.
//!
//! `Metric` is not directly serializable here; the JSON reporter flattens
//! metrics into plain key/value maps at serialization time.

use std::collections::BTreeMap;

use crate::suite::TrialKey;

#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct EpisodeRecord {
    pub episode_idx: usize,
    pub return_value: f64,
    pub length: usize,
}

#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct TrialReport {
    pub key: TrialKey,
    pub env_name: String,
    pub trial_seed: u64,
    pub episodes: Vec<EpisodeRecord>,
    /// Flattened scalar metrics (name -> value). Histograms and counters
    /// land in `histograms` / `counters`.
    pub scalars: BTreeMap<String, f64>,
    pub histograms: BTreeMap<String, Vec<f64>>,
    pub counters: BTreeMap<String, u64>,
    pub errored: bool,
    pub error_message: Option<String>,
}

impl TrialReport {
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

#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub suite_name: String,
    pub base_seed: u64,
    pub trials: Vec<TrialReport>,
    pub in_progress: bool,
}

impl BenchmarkReport {
    #[must_use]
    pub fn new(suite_name: String, base_seed: u64) -> Self {
        Self {
            suite_name,
            base_seed,
            trials: Vec::new(),
            in_progress: true,
        }
    }

    pub fn finalize(&mut self) {
        self.in_progress = false;
    }
}
