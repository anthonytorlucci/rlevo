//! TUI reporter and frame-producer handle.
//!
//! Two producer paths feed the same `mpsc` channel:
//!
//! - [`TuiReporter`] forwards harness-level [`Reporter`] callbacks (suite /
//!   trial / episode boundaries) as [`TuiEvent`]s. Rayon workers grab the
//!   reporter through a `Mutex`, so its send is brief and never blocks the
//!   render thread.
//! - [`TuiHandle`] is the lighter producer used by per-step frame capture
//!   (see `crate::env_wrappers::RenderTap`). It exposes only the
//!   [`TuiHandle::try_push_frame`] entry point so the rollout side cannot
//!   accidentally emit higher-level lifecycle events from inside the env.
//!
//! Both producers wrap the same `Sender<TuiEvent>`. The render thread holds
//! the single `Receiver` and drains it each tick. The channel is the
//! unbounded `std::sync::mpsc::channel`; under sustained rollout pressure
//! it can queue, but the render thread folds events into `AppState` each
//! tick (frame coalescing) so the queue depth is bounded by
//! `tick_ms × frame_rate`.

use std::sync::mpsc::{self, Receiver, Sender};

use rlevo_core::render::StyledFrame;

use crate::report::{BenchmarkReport, EpisodeRecord, TrialReport};
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
        episode: EpisodeRecord,
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
    /// Per-step environment frame captured by [`RenderTap`]. The render
    /// thread coalesces multiple `Frame` events between ticks into the
    /// most recent.
    ///
    /// [`RenderTap`]: crate::env_wrappers::RenderTap
    Frame {
        /// Monotonic step counter within the current episode. Resets on
        /// `reset` in the wrapping `RenderTap`.
        step: u32,
        /// Captured styled frame.
        frame: StyledFrame,
    },
}

/// Lightweight producer the rollout side uses to emit per-step frames.
///
/// Cloneable: the env factory captures one handle and clones it into every
/// [`RenderTap`](crate::env_wrappers::RenderTap) it constructs. All clones
/// feed the same underlying channel.
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

    /// Best-effort frame push. Returns `false` when the receiver has been
    /// dropped (i.e., the render thread has exited); never blocks the
    /// caller. Frames pushed into a live receiver always succeed because
    /// the underlying channel is unbounded.
    ///
    /// Callers in the rollout hot path typically ignore the return value
    /// (the contract is intentionally lossy); the `#[must_use]` is for
    /// tests and adapter code that want to assert delivery.
    #[must_use = "the return value indicates whether the render thread is still listening"]
    pub fn try_push_frame(&self, step: u32, frame: StyledFrame) -> bool {
        self.tx.send(TuiEvent::Frame { step, frame }).is_ok()
    }

    /// Build a [`TuiReporter`] sharing the same channel as this handle.
    /// Use this to plug the reporter into [`Evaluator::run_suite`] while
    /// retaining clones of the handle for [`RenderTap`].
    ///
    /// [`Evaluator::run_suite`]: crate::evaluator::Evaluator::run_suite
    /// [`RenderTap`]: crate::env_wrappers::RenderTap
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

    fn on_episode_end(&mut self, trial: &TrialInfo, ep: &EpisodeRecord) {
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
    use rlevo_core::render::{StyledLine, StyledSpan};

    fn make_frame(text: &str) -> StyledFrame {
        StyledFrame {
            lines: vec![StyledLine::from_spans([StyledSpan::raw(text)])],
        }
    }

    #[test]
    fn handle_pushes_frame_event() {
        let (handle, rx) = TuiHandle::channel();
        assert!(handle.try_push_frame(0, make_frame("hello")));
        match rx.try_recv().expect("expected a queued event") {
            TuiEvent::Frame { step, frame } => {
                assert_eq!(step, 0);
                assert_eq!(frame.plain_text(), "hello");
            }
            other => panic!("expected Frame, got {other:?}"),
        }
    }

    #[test]
    fn try_push_frame_returns_false_when_receiver_dropped() {
        let (handle, rx) = TuiHandle::channel();
        drop(rx);
        assert!(
            !handle.try_push_frame(0, make_frame("x")),
            "should fail cleanly after receiver drop"
        );
    }

    /// Reporter callbacks land on the same channel the handle uses, so a
    /// caller can multiplex lifecycle events with per-step frames through
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
        assert!(handle.try_push_frame(7, make_frame("step7")));

        let first = rx.recv().unwrap();
        let second = rx.recv().unwrap();
        assert!(matches!(first, TuiEvent::SuiteStart(_)));
        match second {
            TuiEvent::Frame { step, .. } => assert_eq!(step, 7),
            other => panic!("expected Frame, got {other:?}"),
        }
    }

    /// `TuiReporter::channel()` remains a one-shot convenience for callers
    /// that don't tap frames. Backwards-compatible with v1 usage.
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
