//! [`RecordingReporter`] — harness-driven producer for the record
//! surface.
//!
//! The benchmarks harness (`Suite` + `Evaluator`) drives the
//! [`Reporter`] trait; [`RecordingReporter`] turns those lifecycle
//! events into [`RecordSink`] calls so that harness runs land
//! per-episode files even when the env is not wrapped in
//! [`RecordingTap`].
//!
//! Producer-disambiguation: when a run wraps its env in
//! [`RecordingTap`], the wrapper drives `on_episode_start` /
//! `on_frame` / `on_episode_end`. This reporter only emits
//! `on_episode_start` / `on_episode_end` when the wrapper isn't
//! present; the writer tolerates double-starts by silently rotating
//! the file, but explicit disambiguation via [`RecordingReporter::without_lifecycle`]
//! is preferred when composing.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::report::{BenchmarkReport, EpisodeSummary, TrialReport};
use crate::reporter::Reporter;
use crate::suite::{SuiteInfo, TrialInfo};

use super::manifest::RunManifest;
use super::schema::Hyperparameters;
use super::writer::RecordSink;

/// Routes harness lifecycle events into a shared [`RecordSink`].
pub struct RecordingReporter {
    sink: Arc<Mutex<dyn RecordSink>>,
    manifest: RunManifest,
    drive_episode_lifecycle: bool,
}

impl std::fmt::Debug for RecordingReporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingReporter")
            .field("sink", &"Arc<Mutex<dyn RecordSink>>")
            .field("manifest", &self.manifest)
            .field("drive_episode_lifecycle", &self.drive_episode_lifecycle)
            .finish()
    }
}

impl RecordingReporter {
    /// Build a reporter that owns the episode lifecycle. Use when the
    /// env is **not** wrapped in
    /// [`RecordingTap`](super::env_tap::RecordingTap).
    #[must_use]
    pub fn new(sink: Arc<Mutex<dyn RecordSink>>, manifest: RunManifest) -> Self {
        Self {
            sink,
            manifest,
            drive_episode_lifecycle: true,
        }
    }

    /// Build a reporter that suppresses `on_episode_start` /
    /// `on_episode_end`. Use when [`RecordingTap`] already drives
    /// those, so the reporter's role is limited to the run-level
    /// manifest at `on_suite_end`.
    ///
    /// [`RecordingTap`]: super::env_tap::RecordingTap
    #[must_use]
    pub fn without_lifecycle(sink: Arc<Mutex<dyn RecordSink>>, manifest: RunManifest) -> Self {
        Self {
            sink,
            manifest,
            drive_episode_lifecycle: false,
        }
    }

    /// Replace the hyperparameter map written to the manifest.
    #[must_use]
    pub fn with_hyperparameters(mut self, hp: Hyperparameters) -> Self {
        self.manifest.hyperparameters = hp;
        self
    }

    /// Set a single hyperparameter key/value pair.
    #[must_use]
    pub fn with_hyperparameter(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.manifest.hyperparameters.insert(key.into(), value.into());
        self
    }
}

impl Reporter for RecordingReporter {
    fn on_suite_start(&mut self, _suite: &SuiteInfo) {}

    fn on_trial_start(&mut self, _trial: &TrialInfo) {
        if self.drive_episode_lifecycle
            && let Ok(mut sink) = self.sink.lock()
        {
            sink.on_episode_start(0);
        }
    }

    fn on_episode_end(&mut self, _trial: &TrialInfo, ep: &EpisodeSummary) {
        if !self.drive_episode_lifecycle {
            return;
        }
        if let Ok(mut sink) = self.sink.lock() {
            sink.on_episode_end(
                ep.return_value,
                u32::try_from(ep.length).unwrap_or(u32::MAX),
            );
            // Open the next episode file straight away. The writer
            // tolerates a stray on_episode_start when the trial ends,
            // since the next call rotates the file cleanly.
            sink.on_episode_start(
                u32::try_from(ep.episode_idx.saturating_add(1)).unwrap_or(u32::MAX),
            );
        }
    }

    fn on_trial_end(&mut self, _trial: &TrialInfo, _report: &TrialReport) {}

    fn on_suite_end(&mut self, _report: &BenchmarkReport) {
        if let Ok(mut sink) = self.sink.lock() {
            sink.on_run_end(self.manifest.clone());
        }
    }
}

/// Returns an empty hyperparameter map — convenience to make the
/// builder API readable without bringing `BTreeMap` into call sites.
#[must_use]
pub fn empty_hyperparameters() -> Hyperparameters {
    BTreeMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::schema::{EnvFamily, RunId};
    use crate::record::writer::InMemoryRecordSink;
    use crate::reporter::MultiReporter;
    use crate::suite::TrialKey;

    fn sample_manifest() -> RunManifest {
        RunManifest::new(RunId("test-run".into()), 42, EnvFamily::Classic, 1)
    }

    fn trial() -> TrialInfo {
        TrialInfo {
            key: TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            env_name: "stub".into(),
            trial_seed: 1,
        }
    }

    fn suite() -> SuiteInfo {
        SuiteInfo {
            name: "S".into(),
            env_names: vec!["stub".into()],
            num_trials_per_env: 1,
        }
    }

    fn trep() -> TrialReport {
        TrialReport::new(
            TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            "stub".into(),
            1,
        )
    }

    #[test]
    fn reporter_with_lifecycle_drives_episode_starts_and_ends() {
        let probe: Arc<Mutex<InMemoryRecordSink>> = Arc::new(Mutex::new(InMemoryRecordSink::new()));
        let dyn_sink: Arc<Mutex<dyn RecordSink>> = probe.clone();
        let mut r = RecordingReporter::new(dyn_sink, sample_manifest());

        r.on_suite_start(&suite());
        r.on_trial_start(&trial());
        for i in 0..3 {
            let ep = EpisodeSummary {
                episode_idx: i,
                return_value: f64::from(i32::try_from(i).unwrap_or(0)),
                length: 10,
            };
            r.on_episode_end(&trial(), &ep);
        }
        r.on_trial_end(&trial(), &trep());
        r.on_suite_end(&BenchmarkReport::new("S".into(), 0));

        let probe = probe.lock().unwrap();
        // on_trial_start opens ep 0; on_episode_end (×3) closes and re-opens for
        // the next idx → episodes 0, 1, 2, 3 all reach on_episode_start.
        assert!(probe.episodes.contains_key(&0));
        assert!(probe.episodes.contains_key(&1));
        assert!(probe.episodes.contains_key(&2));
        assert!(probe.episodes.contains_key(&3));
        assert_eq!(probe.manifests.len(), 1);
        assert_eq!(probe.manifests[0].run_id.0, "test-run");
    }

    #[test]
    fn reporter_without_lifecycle_only_writes_manifest() {
        let probe: Arc<Mutex<InMemoryRecordSink>> = Arc::new(Mutex::new(InMemoryRecordSink::new()));
        let dyn_sink: Arc<Mutex<dyn RecordSink>> = probe.clone();
        let mut r = RecordingReporter::without_lifecycle(dyn_sink, sample_manifest());

        r.on_suite_start(&suite());
        r.on_trial_start(&trial());
        r.on_episode_end(
            &trial(),
            &EpisodeSummary {
                episode_idx: 0,
                return_value: 1.0,
                length: 4,
            },
        );
        r.on_suite_end(&BenchmarkReport::new("S".into(), 0));

        let probe = probe.lock().unwrap();
        assert!(probe.episodes.is_empty(), "no on_episode_start should fire");
        assert!(probe.current.is_none(), "no on_episode_end should fire");
        assert_eq!(probe.manifests.len(), 1);
    }

    #[test]
    fn hyperparameters_land_in_manifest() {
        let probe: Arc<Mutex<InMemoryRecordSink>> = Arc::new(Mutex::new(InMemoryRecordSink::new()));
        let dyn_sink: Arc<Mutex<dyn RecordSink>> = probe.clone();
        let mut r = RecordingReporter::new(dyn_sink, sample_manifest())
            .with_hyperparameter("lr", "3e-4")
            .with_hyperparameter("clip_eps", "0.2");

        r.on_suite_end(&BenchmarkReport::new("S".into(), 0));

        let probe = probe.lock().unwrap();
        assert_eq!(probe.manifests.len(), 1);
        assert_eq!(
            probe.manifests[0].hyperparameters.get("lr"),
            Some(&"3e-4".to_string())
        );
        assert_eq!(
            probe.manifests[0].hyperparameters.get("clip_eps"),
            Some(&"0.2".to_string())
        );
    }

    #[test]
    fn composes_with_multi_reporter() {
        // Both a counting reporter (sees lifecycle) and a recording reporter
        // (writes to the sink) share the suite lifecycle through MultiReporter.
        struct CountToShared {
            shared: Arc<Mutex<u32>>,
        }
        impl Reporter for CountToShared {
            fn on_suite_start(&mut self, _: &SuiteInfo) {}
            fn on_trial_start(&mut self, _: &TrialInfo) {}
            fn on_episode_end(&mut self, _: &TrialInfo, _: &EpisodeSummary) {
                *self.shared.lock().unwrap() += 1;
            }
            fn on_trial_end(&mut self, _: &TrialInfo, _: &TrialReport) {}
            fn on_suite_end(&mut self, _: &BenchmarkReport) {}
        }

        let probe: Arc<Mutex<InMemoryRecordSink>> = Arc::new(Mutex::new(InMemoryRecordSink::new()));
        let dyn_sink: Arc<Mutex<dyn RecordSink>> = probe.clone();
        let counter: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let counter_inner = Arc::clone(&counter);

        let mut multi = MultiReporter::new(vec![
            Box::new(RecordingReporter::new(dyn_sink, sample_manifest())),
            Box::new(CountToShared {
                shared: counter_inner,
            }),
        ]);

        multi.on_suite_start(&suite());
        multi.on_trial_start(&trial());
        multi.on_episode_end(
            &trial(),
            &EpisodeSummary {
                episode_idx: 0,
                return_value: 1.0,
                length: 5,
            },
        );
        multi.on_suite_end(&BenchmarkReport::new("S".into(), 0));

        let probe = probe.lock().unwrap();
        assert!(probe.episodes.contains_key(&0));
        assert!(probe.episodes.contains_key(&1));
        assert_eq!(*counter.lock().unwrap(), 1);
    }
}
