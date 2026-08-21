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
//! JSON reporter flattens metrics into plain key/value maps at absorption
//! time.
//!
//! Absorption is split **by provenance**, not by trust:
//!
//! - [`TrialReport::absorb_harness_metrics`] is the privileged path, for
//!   measurements the trial itself made. It inserts verbatim.
//! - [`TrialReport::absorb_metrics`] is the checked path, for metrics that came
//!   from an agent or a probe. A name the harness owns (see
//!   [`is_harness_reserved`](crate::metrics::core::is_harness_reserved)) is
//!   re-homed under `agent/<name>` and recorded in
//!   [`TrialReport::displaced_metrics`] instead of overwriting the harness's
//!   own value.
//!
//! The safe path carries the obvious name deliberately: [`Trial`](crate::evaluator::Trial)
//! is a public trait, so the next implementor reaches for `absorb_metrics` and
//! gets the checked behaviour by default.

#[cfg(feature = "report")]
pub mod html;
#[cfg(feature = "report")]
pub mod replay;

#[cfg(feature = "report")]
pub use html::{ClientAssets, EmitConfig, EmitError, EmitOutcome, emit_static_html};
#[cfg(feature = "report")]
pub use replay::{EpisodeIndex, OpenError, OpenWarning, RecordedRun};

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use crate::metrics::Metric;
use crate::metrics::core::is_harness_reserved;
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
/// metrics are absorbed in bulk when the trial finishes — its own
/// measurements through [`absorb_harness_metrics`](Self::absorb_harness_metrics),
/// everything an agent or probe produced through the checked
/// [`absorb_metrics`](Self::absorb_metrics).
///
/// Scalar, histogram, and counter metrics are stored in separate maps so that
/// downstream consumers (JSON serializer, HTML emitter) can handle each kind
/// without extra type dispatch.
///
/// # Invariants
///
/// - A key for which
///   [`is_harness_reserved`](crate::metrics::core::is_harness_reserved) holds
///   carries the harness's own measurement, never an agent's.
/// - The `agent/` namespace holds only re-homed agent metrics; the harness
///   never writes there directly.
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
    /// Agent-emitted names that were re-homed because the harness owns them.
    ///
    /// In emission order, one entry per collision (so a repeated collision on
    /// the same key appears more than once). The value itself is stored under
    /// `agent/<name>` in whichever map it would have gone into. An empty vec —
    /// the overwhelmingly common case — means no agent metric contested a
    /// harness key.
    ///
    /// `#[serde(default)]` is load-bearing: checkpoints written by binaries
    /// predating this field have no `displaced_metrics` key, and a resumed run
    /// must still load them.
    #[cfg_attr(feature = "json", serde(default))]
    pub displaced_metrics: Vec<String>,
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
            displaced_metrics: Vec::new(),
            errored: false,
            error_message: None,
        }
    }

    /// Absorbs metrics produced by an **agent or probe**, protecting
    /// harness-owned keys.
    ///
    /// This is the default path and the one a new [`Trial`](crate::evaluator::Trial)
    /// implementor should reach for. For each metric:
    ///
    /// - if the name is not harness-owned, it is inserted verbatim, overwriting
    ///   any earlier entry of the same name (unchanged behaviour);
    /// - if [`is_harness_reserved`](crate::metrics::core::is_harness_reserved)
    ///   holds for the name, the existing entry is left alone and the incoming
    ///   value is stored under `agent/<name>` instead. The original name is
    ///   pushed onto [`displaced_metrics`](Self::displaced_metrics) and exactly
    ///   one `tracing::warn!` is emitted for the collision.
    ///
    /// Because `agent/` is itself reserved, an agent that emits `agent/foo`
    /// directly is re-homed to `agent/agent/foo`: the landing zone is
    /// harness-managed, so a metric cannot place itself there and be mistaken
    /// for a displaced one.
    ///
    /// In the degenerate case where `agent/<name>` is already occupied — an
    /// emission carrying `<name>` twice, or carrying both `<name>` and
    /// `agent/<name>` — the value is dropped rather than overwriting the
    /// earlier one. The collision is still recorded and still warned about,
    /// which is the only trace a dropped value leaves.
    pub fn absorb_metrics(&mut self, metrics: Vec<Metric>) {
        self.absorb(metrics, Authority::Agent);
    }

    /// Absorbs measurements **this trial itself made**, inserting them verbatim.
    ///
    /// The privileged path: no name check, existing entries with the same name
    /// are overwritten, exactly as absorption behaved before the split. Calling
    /// it asserts that every metric in `metrics` is a measurement produced by
    /// the trial's own harness code — `core_metrics`, the non-finite-step
    /// counter, the generation count — and not something an agent, probe, or
    /// environment handed back.
    ///
    /// This is about **namespace authority**, not trust: an agent's metrics are
    /// not suspect, they simply do not own these names. Routing agent output
    /// through here would restore the silent-overwrite defect.
    pub fn absorb_harness_metrics(&mut self, metrics: Vec<Metric>) {
        self.absorb(metrics, Authority::Harness);
    }

    /// The single dispatch shared by both public paths.
    ///
    /// Kind dispatch and insertion live here (and in [`insert`]) exactly once,
    /// so the checked and unchecked paths cannot drift apart.
    fn absorb(&mut self, metrics: Vec<Metric>, authority: Authority) {
        for m in metrics {
            let displaced = match m {
                Metric::Scalar { name, value } => insert(&mut self.scalars, name, value, authority),
                Metric::Histogram { name, values } => {
                    insert(&mut self.histograms, name, values, authority)
                }
                Metric::Counter { name, count } => {
                    insert(&mut self.counters, name, count, authority)
                }
            };
            if let Some(name) = displaced {
                tracing::warn!(
                    target: "rlevo_benchmarks",
                    key = %name,
                    renamed_to = %format!("agent/{name}"),
                    "agent metric renamed: this key is harness-owned and its \
                     value is a harness measurement"
                );
                self.displaced_metrics.push(name);
            }
        }
    }
}

/// Who emitted a metric, and therefore whether its name is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Authority {
    /// The trial's own measurement — inserted verbatim.
    Harness,
    /// Emitted by an agent or probe — checked against the reserved set.
    Agent,
}

/// Inserts one metric value into `map`, applying the rename rule for
/// [`Authority::Agent`].
///
/// Returns `Some(original_name)` when the name was harness-owned and the value
/// was re-homed, so the caller can warn and record it once.
fn insert<V>(
    map: &mut BTreeMap<String, V>,
    name: String,
    value: V,
    authority: Authority,
) -> Option<String> {
    if authority == Authority::Harness || !is_harness_reserved(&name) {
        map.insert(name, value);
        return None;
    }
    // Degenerate case: `agent/<name>` already taken. Drop the value rather
    // than clobbering the earlier one; the collision is still reported.
    if let Entry::Vacant(slot) = map.entry(format!("agent/{name}")) {
        slot.insert(value);
    }
    Some(name)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> TrialReport {
        TrialReport::new(
            TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            "stub".to_string(),
            7,
        )
    }

    #[test]
    fn harness_metrics_are_inserted_verbatim() {
        let mut r = report();
        r.absorb_harness_metrics(vec![Metric::Scalar {
            name: "return/mean".to_string(),
            value: 1.0,
        }]);
        r.absorb_harness_metrics(vec![Metric::Scalar {
            name: "return/mean".to_string(),
            value: 2.0,
        }]);

        assert!((r.scalars["return/mean"] - 2.0).abs() < f64::EPSILON);
        assert!(
            !r.scalars.contains_key("agent/return/mean"),
            "the privileged path must not rename its own measurements"
        );
        assert!(r.displaced_metrics.is_empty());
    }

    #[test]
    fn unreserved_agent_names_keep_last_writer_wins() {
        let mut r = report();
        r.absorb_metrics(vec![Metric::Scalar {
            name: "rl/epsilon".to_string(),
            value: 0.1,
        }]);
        r.absorb_metrics(vec![Metric::Scalar {
            name: "rl/epsilon".to_string(),
            value: 0.2,
        }]);

        assert!((r.scalars["rl/epsilon"] - 0.2).abs() < f64::EPSILON);
        assert!(r.displaced_metrics.is_empty());
    }

    /// All three metric kinds go through the same rename rule — a fix that
    /// only guards scalars leaves counters and histograms clobberable.
    #[test]
    fn every_metric_kind_is_checked() {
        let mut r = report();
        r.absorb_harness_metrics(vec![
            Metric::Scalar {
                name: "return/mean".to_string(),
                value: 1.0,
            },
            Metric::Counter {
                name: "return/non_finite_steps".to_string(),
                count: 8,
            },
            Metric::Histogram {
                name: "episode_length/hist".to_string(),
                values: vec![1.0],
            },
        ]);
        r.absorb_metrics(vec![
            Metric::Scalar {
                name: "return/mean".to_string(),
                value: 999.0,
            },
            Metric::Counter {
                name: "return/non_finite_steps".to_string(),
                count: 999,
            },
            Metric::Histogram {
                name: "episode_length/hist".to_string(),
                values: vec![999.0],
            },
        ]);

        assert!((r.scalars["return/mean"] - 1.0).abs() < f64::EPSILON);
        assert_eq!(r.counters["return/non_finite_steps"], 8);
        assert_eq!(r.histograms["episode_length/hist"], vec![1.0]);

        assert!((r.scalars["agent/return/mean"] - 999.0).abs() < f64::EPSILON);
        assert_eq!(r.counters["agent/return/non_finite_steps"], 999);
        assert_eq!(r.histograms["agent/episode_length/hist"], vec![999.0]);

        assert_eq!(
            r.displaced_metrics,
            vec![
                "return/mean".to_string(),
                "return/non_finite_steps".to_string(),
                "episode_length/hist".to_string(),
            ],
            "recorded in emission order"
        );
    }

    /// The degenerate branch: `agent/<name>` already occupied. The earlier
    /// value stands, the later one is dropped, and the collision is still
    /// recorded. Must not panic.
    #[test]
    fn occupied_agent_slot_drops_the_value_but_still_records() {
        let mut r = report();
        r.absorb_harness_metrics(vec![Metric::Scalar {
            name: "success_rate".to_string(),
            value: 0.5,
        }]);
        r.absorb_metrics(vec![
            // Reserved by the `agent/` prefix → re-homed to `agent/agent/…`.
            Metric::Scalar {
                name: "agent/success_rate".to_string(),
                value: 1.0,
            },
            // Reserved exact key → wants `agent/success_rate`, which is free,
            // so it lands there.
            Metric::Scalar {
                name: "success_rate".to_string(),
                value: 2.0,
            },
            // Same again: `agent/success_rate` is now taken, so this is
            // dropped rather than clobbering the 2.0.
            Metric::Scalar {
                name: "success_rate".to_string(),
                value: 3.0,
            },
        ]);

        assert!((r.scalars["success_rate"] - 0.5).abs() < f64::EPSILON);
        assert!((r.scalars["agent/agent/success_rate"] - 1.0).abs() < f64::EPSILON);
        assert!(
            (r.scalars["agent/success_rate"] - 2.0).abs() < f64::EPSILON,
            "first writer into the landing zone keeps the slot"
        );
        assert_eq!(
            r.displaced_metrics.len(),
            3,
            "all three collisions recorded, including the dropped one"
        );
    }

    /// Checkpoints written before `displaced_metrics` existed must still load:
    /// `checkpoint::load` deserializes `BenchmarkReport` from
    /// `.ckpt.json` files produced by earlier binaries, so without
    /// `#[serde(default)]` every resume would fail. This deserializes the
    /// pre-change shape rather than asserting the attribute is present.
    #[cfg(feature = "json")]
    #[test]
    fn pre_change_checkpoint_json_deserializes_with_no_displaced_metrics() {
        let json = r#"{
            "suite_name": "s",
            "base_seed": 7,
            "in_progress": false,
            "trials": [{
                "key": { "env_idx": 0, "trial_idx": 0 },
                "env_name": "e",
                "trial_seed": 7,
                "episodes": [],
                "scalars": { "return/mean": 1.0 },
                "histograms": {},
                "counters": { "return/non_finite_steps": 0 },
                "errored": false,
                "error_message": null
            }]
        }"#;

        let decoded: BenchmarkReport =
            serde_json::from_str(json).expect("a pre-`displaced_metrics` checkpoint must load");

        assert_eq!(decoded.trials.len(), 1);
        assert!(
            decoded.trials[0].displaced_metrics.is_empty(),
            "the missing field defaults to an empty vec"
        );
        assert!((decoded.trials[0].scalars["return/mean"] - 1.0).abs() < f64::EPSILON);
    }
}
