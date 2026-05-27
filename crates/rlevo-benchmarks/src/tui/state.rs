//! Shared application state read by the render thread.
//!
//! The live TUI has two writers and one reader:
//!
//! - The **rollout side** ([`RenderTap`], the existing [`TuiReporter`])
//!   pushes [`TuiEvent`]s through an `mpsc` channel.
//! - The **render thread** drains that channel each tick, folds the events
//!   into an [`AppState`], and draws panels from the resulting snapshot.
//!
//! [`AppState`] therefore lives entirely on the render thread; no `Mutex`,
//! no `Arc<RwLock<_>>`. Frame coalescing falls out of the design naturally:
//! [`AppState::push_frame`] *replaces* the held frame, so several `Frame`
//! events arriving between ticks collapse to the most recent.
//!
//! Returns are stored as raw `f64`. The sparkline's u64 representation is
//! a presentation concern owned by the panel widget that will land in
//! `panels/reward_sparkline.rs` — keeping the conversion out of the state
//! lets the panel choose a baseline dynamically from the visible window
//! (essential for envs with negative returns like `MountainCar`).
//!
//! [`RenderTap`]: super
//! [`TuiReporter`]: crate::reporter::tui::TuiReporter
//! [`TuiEvent`]: crate::reporter::tui::TuiEvent

use std::collections::VecDeque;

use rlevo_core::render::StyledFrame;

/// Default cap on the reward ring buffer. Chosen to match the spec's
/// default sparkline width allowance; widgets crop further as needed.
pub const DEFAULT_REWARD_HISTORY: usize = 256;

/// How the env panel should be rendered.
///
/// Locomotion envs do not implement `AsciiRenderable` (per ADR 0008), so
/// the live TUI cannot draw them. The placeholder mode is caller-set this
/// milestone; automatic family detection waits for the `EnvFamily` enum
/// that will land alongside `EpisodeRecord` in Milestone 4.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PanelMode {
    /// Render the latest captured [`StyledFrame`]; show a "waiting for
    /// frame" hint when none has arrived yet.
    #[default]
    Auto,
    /// Render the fixed locomotion placeholder regardless of incoming
    /// frames. Use for runs whose env family has no library-tier render.
    LocomotionPlaceholder,
}

/// Compact status-line summary surfaced under the panels.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct StatusLine {
    /// Human-readable suite name, set on `SuiteStart`.
    pub suite_name: Option<String>,
    /// Name of the currently-running env (one suite may host several).
    pub env_name: Option<String>,
    /// `(current_index, total)`. `current_index` is 0-based.
    pub episode: Option<(usize, usize)>,
    /// Return from the most recently completed episode.
    pub last_return: Option<f64>,
    /// `true` once `SuiteEnd` has been observed.
    pub finished: bool,
}

/// Render-thread-local snapshot the panels draw from.
#[derive(Debug, Clone)]
pub struct AppState {
    /// Latest captured environment frame, or `None` before the first push.
    pub frame: Option<StyledFrame>,
    /// Bounded ring of episode returns, oldest-first. Cap is
    /// [`Self::reward_history`].
    pub reward_ring: VecDeque<f64>,
    /// Maximum number of returns retained in `reward_ring`.
    pub reward_history: usize,
    /// Selected env-panel render mode.
    pub panel_mode: PanelMode,
    /// Summary surfaced in the bottom status row.
    pub status: StatusLine,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new(DEFAULT_REWARD_HISTORY, PanelMode::default())
    }
}

impl AppState {
    /// Construct a fresh state with the supplied reward-ring cap and panel
    /// mode. A zero `reward_history` is silently clamped to 1 so the ring
    /// can still hold the most recent return.
    #[must_use]
    pub fn new(reward_history: usize, panel_mode: PanelMode) -> Self {
        let cap = reward_history.max(1);
        Self {
            frame: None,
            reward_ring: VecDeque::with_capacity(cap),
            reward_history: cap,
            panel_mode,
            status: StatusLine::default(),
        }
    }

    /// Install the most recent captured frame, replacing any prior value.
    ///
    /// Frame coalescing happens here: multiple frames arriving between
    /// render ticks collapse to the last one without intermediate work.
    pub fn push_frame(&mut self, frame: StyledFrame) {
        self.frame = Some(frame);
    }

    /// Record an episode return and update the status line.
    ///
    /// Drops the oldest sample when the ring is at capacity, so the FIFO
    /// invariant is preserved for sparkline rendering.
    pub fn record_episode_end(&mut self, episode_idx: usize, return_value: f64) {
        if self.reward_ring.len() == self.reward_history {
            self.reward_ring.pop_front();
        }
        self.reward_ring.push_back(return_value);
        self.status.last_return = Some(return_value);

        // Update the episode counter while preserving an existing `total`.
        let total = self.status.episode.map_or(episode_idx + 1, |(_, t)| t);
        self.status.episode = Some((episode_idx, total));
    }

    /// Record a trial-start boundary: clears the frame slot and records the
    /// env name. Does not clear the reward ring — the user wants to see the
    /// reward trajectory continue across consecutive trials of the same
    /// suite. Reset between suites is the caller's job.
    pub fn mark_trial_start(&mut self, env_name: impl Into<String>) {
        self.frame = None;
        self.status.env_name = Some(env_name.into());
    }

    /// Record suite-level metadata at the start of a run.
    pub fn mark_suite_start(&mut self, suite_name: impl Into<String>, total_episodes: usize) {
        self.status.suite_name = Some(suite_name.into());
        self.status.episode = Some((0, total_episodes));
        self.status.finished = false;
    }

    /// Mark the suite as finished. Panels can read this flag to swap in
    /// a "done" state without tearing the render loop down immediately.
    pub fn mark_suite_end(&mut self) {
        self.status.finished = true;
    }

    /// `true` once the placeholder mode is active. Panels use this to
    /// decide which env-panel widget to draw.
    #[must_use]
    pub fn use_locomotion_placeholder(&self) -> bool {
        matches!(self.panel_mode, PanelMode::LocomotionPlaceholder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::render::{StyledLine, StyledSpan};

    fn frame_with_text(s: &str) -> StyledFrame {
        StyledFrame {
            lines: vec![StyledLine::from_spans([StyledSpan::raw(s)])],
        }
    }

    #[test]
    fn default_state_is_empty_with_default_history() {
        let state = AppState::default();
        assert!(state.frame.is_none());
        assert!(state.reward_ring.is_empty());
        assert_eq!(state.reward_history, DEFAULT_REWARD_HISTORY);
        assert_eq!(state.panel_mode, PanelMode::Auto);
        assert_eq!(state.status, StatusLine::default());
    }

    #[test]
    fn zero_history_is_clamped_to_one() {
        let state = AppState::new(0, PanelMode::Auto);
        assert_eq!(state.reward_history, 1);
    }

    #[test]
    fn push_frame_replaces_rather_than_accumulates() {
        let mut state = AppState::default();
        state.push_frame(frame_with_text("first"));
        state.push_frame(frame_with_text("second"));
        let held = state.frame.unwrap();
        assert_eq!(held.plain_text(), "second");
    }

    #[test]
    fn record_episode_end_appends_and_updates_status() {
        let mut state = AppState::new(4, PanelMode::Auto);
        state.record_episode_end(0, 1.5);
        state.record_episode_end(1, -2.0);

        assert_eq!(state.reward_ring.len(), 2);
        assert_eq!(state.reward_ring.front().copied(), Some(1.5));
        assert_eq!(state.reward_ring.back().copied(), Some(-2.0));
        assert_eq!(state.status.last_return, Some(-2.0));
        assert!(matches!(state.status.episode, Some((1, _))));
    }

    /// Ring boundedness: once at capacity, the oldest sample is evicted.
    #[test]
    fn reward_ring_is_bounded_by_history() {
        let mut state = AppState::new(3, PanelMode::Auto);
        for i in 0..6 {
            state.record_episode_end(i, f64::from(i32::try_from(i).unwrap()));
        }
        assert_eq!(state.reward_ring.len(), 3);
        // Oldest retained sample should be the 4th push (i = 3).
        let v: Vec<f64> = state.reward_ring.iter().copied().collect();
        assert_eq!(v, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn mark_trial_start_clears_frame_but_not_rewards() {
        let mut state = AppState::default();
        state.push_frame(frame_with_text("rendered"));
        state.record_episode_end(0, 7.0);

        state.mark_trial_start("cartpole");

        assert!(state.frame.is_none());
        assert_eq!(state.reward_ring.len(), 1, "reward ring must survive");
        assert_eq!(state.status.env_name.as_deref(), Some("cartpole"));
    }

    #[test]
    fn suite_lifecycle_flags_track() {
        let mut state = AppState::default();
        state.mark_suite_start("smoke", 10);
        assert_eq!(state.status.suite_name.as_deref(), Some("smoke"));
        assert_eq!(state.status.episode, Some((0, 10)));
        assert!(!state.status.finished);

        state.mark_suite_end();
        assert!(state.status.finished);
    }

    #[test]
    fn use_locomotion_placeholder_reflects_mode() {
        let auto = AppState::new(8, PanelMode::Auto);
        assert!(!auto.use_locomotion_placeholder());

        let loco = AppState::new(8, PanelMode::LocomotionPlaceholder);
        assert!(loco.use_locomotion_placeholder());
    }

    #[test]
    fn record_episode_preserves_known_total_after_suite_start() {
        let mut state = AppState::default();
        state.mark_suite_start("smoke", 50);
        state.record_episode_end(3, 1.0);
        let (idx, total) = state.status.episode.unwrap();
        assert_eq!(idx, 3);
        assert_eq!(total, 50, "suite total must not be overwritten by per-episode updates");
    }
}
