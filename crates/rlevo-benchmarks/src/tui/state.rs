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

use std::collections::{HashMap, VecDeque};

use rlevo_core::render::StyledFrame;

/// Default cap on the reward ring buffer. Chosen to match the spec's
/// default sparkline width allowance; widgets crop further as needed.
pub const DEFAULT_REWARD_HISTORY: usize = 256;

/// Default cap on the scrolling log ring buffer. Sized for ~10 KB of
/// recent log lines, generous for a typical 100-step PPO run while still
/// bounded so a runaway log source can't grow memory without limit.
pub const DEFAULT_LOG_HISTORY: usize = 100;

/// One captured tracing event held by the log panel.
///
/// Distinct from the [`crate::reporter::tui::TuiEvent::LogLine`] enum
/// variant because the variant is a wire-format payload, while this is
/// the stored-on-the-render-thread representation. Identical fields
/// today; kept separate so the storage shape can evolve (e.g., add a
/// `timestamp`) without touching the channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedLogLine {
    /// Severity. Drives panel styling (ERROR → HAZARD; WARN → Yellow).
    pub level: tracing::Level,
    /// Originating module / target string.
    pub target: String,
    /// Formatted message body.
    pub message: String,
}

/// How the env panel should be rendered.
///
/// Locomotion envs do not implement `AsciiRenderable`, so
/// the live TUI cannot draw them. Set manually for tests, or via
/// [`TuiConfig::with_env_family`](crate::tui::runner::TuiConfig::with_env_family)
/// for runs with the `record` feature enabled.
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
    /// One bounded ring per named metric, keyed by the metric name from
    /// [`crate::tui::log_layer::CANONICAL_METRICS`]. Each ring honours
    /// the same [`Self::reward_history`] cap so all sparklines share a
    /// visible window length.
    pub metric_rings: HashMap<String, VecDeque<f64>>,
    /// Bounded ring of recent log lines, oldest-first. Cap is
    /// [`Self::log_history`].
    pub log_ring: VecDeque<CapturedLogLine>,
    /// Maximum number of returns retained in `reward_ring` and in each
    /// `metric_rings` entry.
    pub reward_history: usize,
    /// Maximum number of log lines retained in `log_ring`.
    pub log_history: usize,
    /// Selected env-panel render mode.
    pub panel_mode: PanelMode,
    /// Summary surfaced in the bottom status row.
    pub status: StatusLine,
}

impl Default for AppState {
    fn default() -> Self {
        Self::with_history(
            DEFAULT_REWARD_HISTORY,
            DEFAULT_LOG_HISTORY,
            PanelMode::default(),
        )
    }
}

impl AppState {
    /// Construct a fresh state with the supplied reward-ring cap and panel
    /// mode. Log history defaults to [`DEFAULT_LOG_HISTORY`]; use
    /// [`Self::with_history`] for full control.
    ///
    /// A zero `reward_history` is silently clamped to 1 so the ring can
    /// still hold the most recent return.
    #[must_use]
    pub fn new(reward_history: usize, panel_mode: PanelMode) -> Self {
        Self::with_history(reward_history, DEFAULT_LOG_HISTORY, panel_mode)
    }

    /// Construct a fresh state with explicit caps for both rings. A zero
    /// `reward_history` or `log_history` is silently clamped to 1.
    #[must_use]
    pub fn with_history(
        reward_history: usize,
        log_history: usize,
        panel_mode: PanelMode,
    ) -> Self {
        let reward_cap = reward_history.max(1);
        let log_cap = log_history.max(1);
        Self {
            frame: None,
            reward_ring: VecDeque::with_capacity(reward_cap),
            metric_rings: HashMap::new(),
            log_ring: VecDeque::with_capacity(log_cap),
            reward_history: reward_cap,
            log_history: log_cap,
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

    /// Record one named metric sample. Creates the per-name ring on first
    /// call; subsequent calls evict the oldest sample at capacity.
    ///
    /// The cap is [`Self::reward_history`] — shared across all metric
    /// rings so every sparkline shows the same visible window length.
    pub fn record_metric(&mut self, name: impl Into<String>, value: f64) {
        let cap = self.reward_history;
        let ring = self
            .metric_rings
            .entry(name.into())
            .or_insert_with(|| VecDeque::with_capacity(cap));
        if ring.len() == cap {
            ring.pop_front();
        }
        ring.push_back(value);
    }

    /// Append one captured log line, evicting the oldest at capacity.
    pub fn record_log(&mut self, line: CapturedLogLine) {
        if self.log_ring.len() == self.log_history {
            self.log_ring.pop_front();
        }
        self.log_ring.push_back(line);
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

    /// Record an episode return from a non-harness producer.
    ///
    /// Equivalent to [`Self::record_episode_end`] except `status.episode`
    /// is left untouched — non-harness producers (e.g.
    /// [`TuiEnvTap`](crate::env_wrappers::TuiEnvTap)) have no notion of
    /// "episode N of total", and writing `(N, N+1)` per call would create a
    /// running counter that contradicts the harness-supplied total when the
    /// two paths coexist.
    pub fn record_episode_return(&mut self, return_value: f64) {
        if self.reward_ring.len() == self.reward_history {
            self.reward_ring.pop_front();
        }
        self.reward_ring.push_back(return_value);
        self.status.last_return = Some(return_value);
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
        assert!(state.metric_rings.is_empty());
        assert!(state.log_ring.is_empty());
        assert_eq!(state.reward_history, DEFAULT_REWARD_HISTORY);
        assert_eq!(state.log_history, DEFAULT_LOG_HISTORY);
        assert_eq!(state.panel_mode, PanelMode::Auto);
        assert_eq!(state.status, StatusLine::default());
    }

    #[test]
    fn zero_history_is_clamped_to_one() {
        let state = AppState::new(0, PanelMode::Auto);
        assert_eq!(state.reward_history, 1);
    }

    #[test]
    fn zero_log_history_is_clamped_to_one() {
        let state = AppState::with_history(8, 0, PanelMode::Auto);
        assert_eq!(state.log_history, 1);
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
    fn record_episode_return_appends_without_touching_episode_counter() {
        let mut state = AppState::new(4, PanelMode::Auto);
        state.record_episode_return(1.5);
        state.record_episode_return(-2.0);

        assert_eq!(state.reward_ring.len(), 2);
        assert_eq!(state.reward_ring.front().copied(), Some(1.5));
        assert_eq!(state.reward_ring.back().copied(), Some(-2.0));
        assert_eq!(state.status.last_return, Some(-2.0));
        assert!(
            state.status.episode.is_none(),
            "non-harness writer must not invent an episode counter"
        );
    }

    /// Ring boundedness on the non-harness writer: same eviction policy
    /// as `record_episode_end` so sparkline rendering is identical.
    #[test]
    fn record_episode_return_evicts_oldest_at_capacity() {
        let mut state = AppState::new(3, PanelMode::Auto);
        for i in 0..6_i32 {
            state.record_episode_return(f64::from(i));
        }
        let v: Vec<f64> = state.reward_ring.iter().copied().collect();
        assert_eq!(v, vec![3.0, 4.0, 5.0]);
    }

    /// When the harness has already populated `status.episode`, a
    /// non-harness writer must not clobber it. Confirms the harness and
    /// non-harness paths can coexist without one rewriting the other's
    /// counter (relevant when an EA loop fans out to both paths).
    #[test]
    fn record_episode_return_preserves_harness_episode_counter() {
        let mut state = AppState::default();
        state.mark_suite_start("smoke", 10);
        state.record_episode_end(3, 1.0);
        state.record_episode_return(2.0);

        let (idx, total) = state.status.episode.expect("harness counter set");
        assert_eq!(idx, 3, "harness episode index must not change");
        assert_eq!(total, 10, "harness episode total must not change");
        assert_eq!(state.status.last_return, Some(2.0));
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

    #[test]
    fn record_metric_creates_ring_on_first_sample() {
        let mut state = AppState::new(4, PanelMode::Auto);
        state.record_metric("policy_loss", 0.5);
        let ring = state.metric_rings.get("policy_loss").expect("ring created");
        assert_eq!(ring.iter().copied().collect::<Vec<_>>(), vec![0.5]);
    }

    #[test]
    fn record_metric_appends_in_order() {
        let mut state = AppState::new(4, PanelMode::Auto);
        for v in [0.5, 0.4, 0.3] {
            state.record_metric("policy_loss", v);
        }
        let ring = &state.metric_rings["policy_loss"];
        assert_eq!(ring.iter().copied().collect::<Vec<_>>(), vec![0.5, 0.4, 0.3]);
    }

    /// Each metric name gets its own ring, all bounded by the same
    /// `reward_history` cap.
    #[test]
    fn record_metric_rings_are_independent_per_name() {
        let mut state = AppState::new(8, PanelMode::Auto);
        state.record_metric("policy_loss", 0.5);
        state.record_metric("entropy", 1.2);
        state.record_metric("policy_loss", 0.4);

        let losses = &state.metric_rings["policy_loss"];
        let entropies = &state.metric_rings["entropy"];
        assert_eq!(losses.iter().copied().collect::<Vec<_>>(), vec![0.5, 0.4]);
        assert_eq!(entropies.iter().copied().collect::<Vec<_>>(), vec![1.2]);
    }

    /// Each metric ring honours `reward_history` independently; once at
    /// capacity, the oldest sample for *that name* evicts.
    #[test]
    fn record_metric_evicts_oldest_at_capacity_per_name() {
        let mut state = AppState::new(3, PanelMode::Auto);
        for i in 0..6_i32 {
            state.record_metric("loss", f64::from(i));
        }
        let ring = &state.metric_rings["loss"];
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.iter().copied().collect::<Vec<_>>(), vec![3.0, 4.0, 5.0]);
    }

    fn log_line(level: tracing::Level, msg: &str) -> CapturedLogLine {
        CapturedLogLine {
            level,
            target: "test".to_string(),
            message: msg.to_string(),
        }
    }

    #[test]
    fn record_log_appends_in_order() {
        let mut state = AppState::with_history(8, 4, PanelMode::Auto);
        state.record_log(log_line(tracing::Level::INFO, "first"));
        state.record_log(log_line(tracing::Level::WARN, "second"));
        let collected: Vec<_> = state
            .log_ring
            .iter()
            .map(|l| l.message.clone())
            .collect();
        assert_eq!(collected, vec!["first", "second"]);
    }

    /// Once the log ring is full, the oldest line is evicted.
    #[test]
    fn record_log_evicts_oldest_at_capacity() {
        let mut state = AppState::with_history(8, 3, PanelMode::Auto);
        for i in 0..6 {
            state.record_log(log_line(tracing::Level::INFO, &format!("line {i}")));
        }
        assert_eq!(state.log_ring.len(), 3);
        let collected: Vec<_> = state
            .log_ring
            .iter()
            .map(|l| l.message.clone())
            .collect();
        assert_eq!(collected, vec!["line 3", "line 4", "line 5"]);
    }

    #[test]
    fn record_log_preserves_level_and_target() {
        let mut state = AppState::default();
        state.record_log(CapturedLogLine {
            level: tracing::Level::ERROR,
            target: "rlevo_rl::ppo".to_string(),
            message: "exploded".to_string(),
        });
        let line = state.log_ring.front().unwrap();
        assert_eq!(line.level, tracing::Level::ERROR);
        assert_eq!(line.target, "rlevo_rl::ppo");
        assert_eq!(line.message, "exploded");
    }
}
