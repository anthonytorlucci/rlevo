//! TUI reporter and metric/episode-return producer handle.
//!
//! Two producer paths feed the same `mpsc` channel:
//!
//! - [`TuiReporter`] forwards harness-level [`Reporter`] callbacks (suite /
//!   trial / episode boundaries) as [`TuiEvent`]s. Rayon workers grab the
//!   reporter through a `Mutex`, so its send is brief and never blocks the
//!   render thread.
//! - [`TuiHandle`] is the lighter producer used by the metric tracing layer
//!   ([`TuiCaptureLayer`](crate::tui::log_layer::TuiCaptureLayer)) and by
//!   non-harness episode-return taps
//!   ([`TuiEnvTap`](crate::env_wrappers::TuiEnvTap)). It exposes only metric,
//!   log, and episode-return entry points so the rollout side cannot
//!   accidentally emit higher-level lifecycle events from inside the env.
//!
//! Both producers wrap the same `Sender<TuiEvent>`. The render thread holds
//! the single `Receiver` and drains it each tick. The channel is the
//! unbounded `std::sync::mpsc::channel`; under sustained pressure it can
//! queue, but the render thread folds events into `AppState` each tick so
//! the queue depth stays bounded.

use std::sync::mpsc::{self, Receiver, Sender};

use crate::report::{BenchmarkReport, EpisodeSummary, TrialReport};
use crate::reporter::Reporter;
use crate::suite::{SuiteInfo, TrialInfo};

/// Events delivered to the render thread.
#[derive(Debug, Clone)]
pub enum TuiEvent {
    /// Suite-level start, carrying the full [`SuiteInfo`].
    SuiteStart(SuiteInfo),
    /// Trial-level start. The render thread typically clears the env
    /// panel and updates the status line on this.
    TrialStart(TrialInfo),
    /// Episode terminated. The render thread pushes the return into the
    /// reward ring.
    EpisodeEnd {
        /// Identifies the trial the episode belongs to.
        trial: TrialInfo,
        /// Final episode record (return, length, index).
        episode: EpisodeSummary,
    },
    /// Episode terminated outside the benchmarks harness. Carries only the
    /// return and length — no `TrialInfo`, no `EpisodeSummary`. Emitted by
    /// non-harness env wrappers (e.g.,
    /// [`TuiEnvTap`](crate::env_wrappers::TuiEnvTap)) so a raw
    /// `Environment` driver (PPO's `train_discrete`, future EA loops) can
    /// populate the reward sparkline without synthesising harness-only
    /// types. The render thread folds the return into the same
    /// `reward_ring` that `EpisodeEnd` writes to.
    EpisodeReturn {
        /// Sum of per-step rewards across the just-terminated episode.
        return_value: f64,
        /// Step count of the just-terminated episode. Unused by the live
        /// tier today; carried for Milestone-4's `EpisodeRecord` writer.
        length: u32,
    },
    /// Trial terminated. Carries the aggregated [`TrialReport`].
    TrialEnd {
        /// Identifies the trial.
        trial: TrialInfo,
        /// Trial aggregate.
        report: TrialReport,
    },
    /// Suite-level termination with the full [`BenchmarkReport`].
    SuiteEnd(BenchmarkReport),
    /// One scalar metric sample extracted from a structured tracing event
    /// by [`TuiCaptureLayer`]. The render thread appends the value to the
    /// per-name ring in [`AppState`](crate::tui::state::AppState).
    ///
    /// [`TuiCaptureLayer`]: crate::tui::log_layer::TuiCaptureLayer
    MetricUpdate {
        /// Canonical metric name — see the registry in
        /// [`crate::tui::log_layer`] for the wiring contract.
        name: String,
        /// Sample value coerced to `f64` regardless of the source field's
        /// type (`f64` / `i64` / `u64`).
        value: f64,
    },
    /// One tracing event captured for the scrolling log panel. Carries
    /// level + target so the panel can style ERROR / WARN lines per the
    /// accessibility contract.
    LogLine {
        /// Severity. Drives panel styling.
        level: tracing::Level,
        /// Originating module / target string.
        target: String,
        /// Formatted message body (the literal `tracing::*!` argument).
        message: String,
    },
}

/// Lightweight producer the rollout side uses to emit metric, log, and
/// episode-return events.
///
/// Cloneable: the caller captures one handle and clones it into the metric
/// tracing layer and any [`TuiEnvTap`](crate::env_wrappers::TuiEnvTap) it
/// constructs. All clones feed the same underlying channel.
#[derive(Debug, Clone)]
pub struct TuiHandle {
    tx: Sender<TuiEvent>,
}

impl TuiHandle {
    /// Construct a fresh handle plus the receiver the render thread will
    /// drain. The reporter — if needed — is created via
    /// [`TuiHandle::as_reporter`] so it shares the same channel.
    #[must_use]
    pub fn channel() -> (Self, Receiver<TuiEvent>) {
        let (tx, rx) = mpsc::channel();
        (Self { tx }, rx)
    }

    /// Best-effort metric push. Used by [`TuiCaptureLayer`] when a known
    /// numeric field arrives on a tracing event. Returns `false` when the
    /// receiver has been dropped (the render thread has exited); never
    /// blocks the caller. The contract is intentionally lossy.
    ///
    /// [`TuiCaptureLayer`]: crate::tui::log_layer::TuiCaptureLayer
    #[must_use = "the return value indicates whether the render thread is still listening"]
    pub fn try_push_metric(&self, name: String, value: f64) -> bool {
        self.tx.send(TuiEvent::MetricUpdate { name, value }).is_ok()
    }

    /// Best-effort log push. Used by [`TuiCaptureLayer`] for every tracing
    /// event the subscriber sees. Same lossy semantics as
    /// [`Self::try_push_metric`].
    ///
    /// [`TuiCaptureLayer`]: crate::tui::log_layer::TuiCaptureLayer
    #[must_use = "the return value indicates whether the render thread is still listening"]
    pub fn try_push_log(
        &self,
        level: tracing::Level,
        target: String,
        message: String,
    ) -> bool {
        self.tx
            .send(TuiEvent::LogLine {
                level,
                target,
                message,
            })
            .is_ok()
    }

    /// Best-effort episode-return push. Used by non-harness env wrappers
    /// (e.g., [`TuiEnvTap`](crate::env_wrappers::TuiEnvTap)) on episode
    /// termination. Same lossy semantics as [`Self::try_push_metric`].
    #[must_use = "the return value indicates whether the render thread is still listening"]
    pub fn try_push_episode_return(&self, return_value: f64, length: u32) -> bool {
        self.tx
            .send(TuiEvent::EpisodeReturn {
                return_value,
                length,
            })
            .is_ok()
    }

    /// Build a [`TuiReporter`] sharing the same channel as this handle.
    /// Use this to plug the reporter into [`Evaluator::run_suite`] while
    /// retaining clones of the handle for the metric layer.
    ///
    /// [`Evaluator::run_suite`]: crate::evaluator::Evaluator::run_suite
    #[must_use]
    pub fn as_reporter(&self) -> TuiReporter {
        TuiReporter {
            tx: self.tx.clone(),
        }
    }
}

/// Channel-backed [`Reporter`] that forwards harness callbacks as
/// [`TuiEvent`]s.
///
/// Construct via [`TuiHandle::as_reporter`] for new code; the legacy
/// [`TuiReporter::channel`] convenience is preserved for callers that only
/// need the reporter and never tap frames.
#[derive(Debug)]
pub struct TuiReporter {
    tx: Sender<TuiEvent>,
}

impl TuiReporter {
    /// Construct a reporter plus a receiver. Backwards-compatible shortcut
    /// for callers that want only the harness event stream and don't need
    /// per-step frame capture.
    #[must_use]
    pub fn channel() -> (Self, Receiver<TuiEvent>) {
        let (handle, rx) = TuiHandle::channel();
        (handle.as_reporter(), rx)
    }
}

impl Reporter for TuiReporter {
    fn on_suite_start(&mut self, suite: &SuiteInfo) {
        let _ = self.tx.send(TuiEvent::SuiteStart(suite.clone()));
    }

    fn on_trial_start(&mut self, trial: &TrialInfo) {
        let _ = self.tx.send(TuiEvent::TrialStart(trial.clone()));
    }

    fn on_episode_end(&mut self, trial: &TrialInfo, ep: &EpisodeSummary) {
        let _ = self.tx.send(TuiEvent::EpisodeEnd {
            trial: trial.clone(),
            episode: ep.clone(),
        });
    }

    fn on_trial_end(&mut self, trial: &TrialInfo, report: &TrialReport) {
        let _ = self.tx.send(TuiEvent::TrialEnd {
            trial: trial.clone(),
            report: report.clone(),
        });
    }

    fn on_suite_end(&mut self, report: &BenchmarkReport) {
        let _ = self.tx.send(TuiEvent::SuiteEnd(report.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_pushes_metric_event() {
        let (handle, rx) = TuiHandle::channel();
        assert!(handle.try_push_metric("policy_loss".to_string(), 0.25));
        match rx.try_recv().expect("expected a queued event") {
            TuiEvent::MetricUpdate { name, value } => {
                assert_eq!(name, "policy_loss");
                assert!((value - 0.25).abs() < f64::EPSILON);
            }
            other => panic!("expected MetricUpdate, got {other:?}"),
        }
    }

    #[test]
    fn try_push_metric_returns_false_when_receiver_dropped() {
        let (handle, rx) = TuiHandle::channel();
        drop(rx);
        assert!(
            !handle.try_push_metric("x".to_string(), 0.0),
            "should fail cleanly after receiver drop"
        );
    }

    #[test]
    fn handle_pushes_episode_return_event() {
        let (handle, rx) = TuiHandle::channel();
        assert!(handle.try_push_episode_return(42.5, 17));
        match rx.try_recv().expect("expected a queued event") {
            TuiEvent::EpisodeReturn {
                return_value,
                length,
            } => {
                assert!((return_value - 42.5).abs() < f64::EPSILON);
                assert_eq!(length, 17);
            }
            other => panic!("expected EpisodeReturn, got {other:?}"),
        }
    }

    #[test]
    fn try_push_episode_return_returns_false_when_receiver_dropped() {
        let (handle, rx) = TuiHandle::channel();
        drop(rx);
        assert!(
            !handle.try_push_episode_return(1.0, 1),
            "should fail cleanly after receiver drop"
        );
    }

    /// Reporter callbacks land on the same channel the handle uses, so a
    /// caller can multiplex lifecycle events with metric updates through
    /// one render thread.
    #[test]
    fn reporter_and_handle_share_channel() {
        let (handle, rx) = TuiHandle::channel();
        let mut reporter = handle.as_reporter();
        let suite = SuiteInfo {
            name: "demo".to_string(),
            env_names: vec!["env-0".to_string()],
            num_trials_per_env: 1,
        };
        reporter.on_suite_start(&suite);
        assert!(handle.try_push_metric("entropy".to_string(), 1.0));

        let first = rx.recv().unwrap();
        let second = rx.recv().unwrap();
        assert!(matches!(first, TuiEvent::SuiteStart(_)));
        match second {
            TuiEvent::MetricUpdate { name, .. } => assert_eq!(name, "entropy"),
            other => panic!("expected MetricUpdate, got {other:?}"),
        }
    }

    /// `TuiReporter::channel()` remains a one-shot convenience for callers
    /// that only need the lifecycle stream. Backwards-compatible with v1 usage.
    #[test]
    fn legacy_channel_constructor_yields_working_reporter() {
        let (mut reporter, rx) = TuiReporter::channel();
        let suite = SuiteInfo {
            name: "legacy".to_string(),
            env_names: vec![],
            num_trials_per_env: 1,
        };
        reporter.on_suite_start(&suite);
        match rx.try_recv().expect("event") {
            TuiEvent::SuiteStart(s) => assert_eq!(s.name, "legacy"),
            other => panic!("expected SuiteStart, got {other:?}"),
        }
    }
}
