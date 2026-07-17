//! Terminal lifecycle and render-thread driver for the live TUI.
//!
//! [`TuiRunner::start`] enters raw mode + alt screen via
//! [`ratatui::try_init`], which also installs a panic hook that restores
//! the terminal before propagating the panic. A dedicated thread takes
//! ownership of the [`DefaultTerminal`](ratatui::DefaultTerminal) and
//! drains a [`mpsc::Receiver<TuiEvent>`] each tick, folding events into
//! the shared [`AppState`] and redrawing the dashboard.
//!
//! Rollout-side producers ([`TuiReporter`] for suite/trial/episode
//! lifecycle, [`TuiEnvTap`] for non-harness episode returns) feed the
//! channel through a cloneable [`TuiHandle`].
//!
//! # Shutdown
//!
//! Two paths cleanly tear the runner down:
//!
//! 1. **Explicit:** [`TuiRunner::shutdown`] sets the cancellation flag,
//!    joins the render thread, and calls [`ratatui::try_restore`].
//! 2. **Implicit:** [`TuiRunner::drop`] does the same, best-effort.
//!    Use the explicit form when you need to surface I/O errors.
//!
//! # Panic recovery
//!
//! The hook installed by `try_init` restores raw mode + alt screen before
//! the original hook runs, so an uncaught panic anywhere in the process
//! still leaves the terminal usable. Caught panics inside rayon workers
//! don't fire the hook (handled by `Evaluator`'s `catch_unwind`); the
//! render thread keeps drawing.
//!
//! [`TuiEnvTap`]: crate::env_wrappers::TuiEnvTap
//! [`TuiReporter`]: crate::reporter::tui::TuiReporter
//! [`AppState`]: crate::tui::state::AppState
//! [`TuiHandle`]: crate::reporter::tui::TuiHandle

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use ratatui::Frame;
use ratatui::Terminal;
use ratatui::backend::Backend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier as RatModifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::metrics_registry::{hint_for, trend_for};
use crate::reporter::tui::{TuiEvent, TuiHandle};
use crate::tui::panels::{LogPanel, MetricSparkline, RewardSparkline};
use crate::tui::state::{
    AppState, CapturedLogLine, DEFAULT_LOG_HISTORY, DEFAULT_REWARD_HISTORY, MetricsLayout,
    StatusLine,
};

/// Default render tick. 60 ms ≈ 16 fps — slow enough that the render
/// thread spends most of its time blocked on the channel rather than
/// redrawing, fast enough that the sparklines update fluidly.
pub const DEFAULT_TICK_MS: u64 = 60;

/// Configuration for [`TuiRunner`].
#[derive(Debug, Clone, Copy)]
pub struct TuiConfig {
    /// Maximum delay between forced redraws.
    pub tick_ms: u64,
    /// Cap on the reward ring and every per-name metric ring.
    pub reward_history: usize,
    /// Cap on the scrolling log ring.
    pub log_history: usize,
    /// Metric names shown in the metric column, in display order.
    /// Defaults to [`DASHBOARD_METRICS`]; narrow it to drop signals a run
    /// never emits (e.g. EA-only `best_fitness` in a pure RL run).
    pub metrics: &'static [&'static str],
    /// Geometry of the metric panels. See [`MetricsLayout`].
    pub metrics_layout: MetricsLayout,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            tick_ms: DEFAULT_TICK_MS,
            reward_history: DEFAULT_REWARD_HISTORY,
            log_history: DEFAULT_LOG_HISTORY,
            metrics: DASHBOARD_METRICS,
            metrics_layout: MetricsLayout::default(),
        }
    }
}

/// Errors emitted by the runner lifecycle.
#[derive(Debug, thiserror::Error)]
pub enum TuiError {
    /// Terminal init or shutdown I/O failure.
    #[error("terminal io: {0}")]
    Io(#[from] io::Error),
    /// Render thread panicked before clean shutdown.
    #[error("render thread panicked")]
    ThreadPanicked,
}

/// Live-TUI driver. Owns the render thread + cancellation signal; produces
/// cloneable [`TuiHandle`]s for the rollout side.
#[derive(Debug)]
pub struct TuiRunner {
    handle: TuiHandle,
    shutdown: Arc<AtomicBool>,
    join: Option<JoinHandle<io::Result<()>>>,
}

impl TuiRunner {
    /// Enter raw mode, install the panic hook, and spawn the render
    /// thread. The returned runner owns the thread + shutdown signal;
    /// drop it (or call [`shutdown`](Self::shutdown)) to restore the
    /// terminal.
    ///
    /// # Errors
    ///
    /// Returns [`TuiError::Io`] if `ratatui::try_init` fails (e.g.,
    /// stdout is not a tty, raw mode cannot be enabled).
    pub fn start(cfg: TuiConfig) -> Result<Self, TuiError> {
        let terminal = ratatui::try_init()?;
        let (handle, rx) = TuiHandle::channel();
        let shutdown = Arc::new(AtomicBool::new(false));

        let thread_shutdown = Arc::clone(&shutdown);
        let join = thread::Builder::new()
            .name("rlevo-tui-render".to_string())
            .spawn(move || {
                let mut term = terminal;
                let mut state = AppState::with_history(cfg.reward_history, cfg.log_history)
                    .with_layout(cfg.metrics, cfg.metrics_layout);
                let result = render_loop(&mut term, &mut state, &rx, &thread_shutdown, cfg.tick_ms);
                // Always restore — even if render_loop returned an error
                // we want the terminal usable for the user's last words.
                let _ = ratatui::try_restore();
                result
            })?;

        Ok(Self {
            handle,
            shutdown,
            join: Some(join),
        })
    }

    /// Clone a [`TuiHandle`] for use in env factories and reporter chains.
    #[must_use]
    pub fn handle(&self) -> TuiHandle {
        self.handle.clone()
    }

    /// Block the current thread until the user presses any key.
    ///
    /// Use after the run completes so the final dashboard state stays
    /// visible while the user reviews it; pair with [`Self::shutdown`]
    /// to restore the terminal afterward.
    ///
    /// The render thread keeps drawing in the background, so the
    /// dashboard continues to refresh while this waits. Non-key
    /// crossterm events (resize, paste, mouse) are consumed silently
    /// and the wait continues.
    ///
    /// # Errors
    ///
    /// Returns [`TuiError::Io`] if crossterm's event source fails.
    pub fn wait_for_keypress(&self) -> Result<(), TuiError> {
        loop {
            // Short poll so the loop stays responsive to a manual
            // shutdown signal (e.g. another thread flipping the flag)
            // without burning CPU.
            if crossterm::event::poll(Duration::from_millis(100))?
                && let crossterm::event::Event::Key(_) = crossterm::event::read()?
            {
                return Ok(());
            }
            if self.shutdown.load(Ordering::Acquire) {
                return Ok(());
            }
        }
    }

    /// Signal the render thread to exit, join it, and surface any
    /// terminal I/O error that occurred during the loop.
    ///
    /// # Errors
    ///
    /// Returns [`TuiError::Io`] if the render thread encountered an I/O
    /// failure while drawing, or [`TuiError::ThreadPanicked`] if the
    /// thread itself panicked (rare — the panic hook should have already
    /// restored the terminal).
    pub fn shutdown(mut self) -> Result<(), TuiError> {
        self.shutdown_inner()
    }

    fn shutdown_inner(&mut self) -> Result<(), TuiError> {
        self.shutdown.store(true, Ordering::Release);
        match self.join.take() {
            Some(j) => match j.join() {
                Ok(io_result) => io_result.map_err(TuiError::from),
                Err(_) => Err(TuiError::ThreadPanicked),
            },
            None => Ok(()),
        }
    }
}

impl Drop for TuiRunner {
    fn drop(&mut self) {
        // Best-effort: signal shutdown and join. Errors are swallowed
        // because Drop has no return path.
        let _ = self.shutdown_inner();
    }
}

/// Render-thread main loop. Generic over [`Backend`] so unit tests can
/// drive it against [`ratatui::backend::TestBackend`].
fn render_loop<B>(
    terminal: &mut Terminal<B>,
    state: &mut AppState,
    rx: &Receiver<TuiEvent>,
    shutdown: &Arc<AtomicBool>,
    tick_ms: u64,
) -> io::Result<()>
where
    B: Backend,
    B::Error: Send + Sync + 'static,
{
    let tick = Duration::from_millis(tick_ms);

    while !shutdown.load(Ordering::Acquire) {
        match rx.recv_timeout(tick) {
            Ok(event) => apply_event(state, event),
            Err(RecvTimeoutError::Timeout) => {}
            // All senders dropped — exit cleanly.
            Err(RecvTimeoutError::Disconnected) => break,
        }

        // Drain any further events buffered between the previous recv and
        // now, so frame coalescing happens in one place per tick.
        while let Ok(extra) = rx.try_recv() {
            apply_event(state, extra);
        }

        terminal
            .draw(|frame| draw_dashboard(frame, state))
            .map_err(io::Error::other)?;
    }

    Ok(())
}

/// Fold a single event into [`AppState`]. Pure; testable without a
/// receiver or a terminal.
pub fn apply_event(state: &mut AppState, event: TuiEvent) {
    match event {
        TuiEvent::SuiteStart(info) => {
            // num_episodes isn't on SuiteInfo today, so default total to
            // zero — the per-episode update will adjust as samples land.
            let total_envs = info.env_names.len() * info.num_trials_per_env;
            state.mark_suite_start(info.name, total_envs);
        }
        TuiEvent::TrialStart(trial) => {
            state.mark_trial_start(trial.env_name);
        }
        TuiEvent::EpisodeEnd { trial: _, episode } => {
            state.record_episode_end(episode.episode_idx, episode.return_value);
        }
        TuiEvent::EpisodeReturn {
            return_value,
            length: _,
        } => {
            state.record_episode_return(return_value);
        }
        TuiEvent::TrialEnd { .. } => {
            // No-op: the trial aggregate is summarised at suite end.
        }
        TuiEvent::SuiteEnd(_) => {
            state.mark_suite_end();
        }
        TuiEvent::MetricUpdate { name, value } => {
            state.record_metric(name, value);
        }
        TuiEvent::LogLine {
            level,
            target,
            message,
        } => {
            state.record_log(CapturedLogLine {
                level,
                target,
                message,
            });
        }
    }
}

/// Canonical metric names rendered in the right-hand column under the
/// reward sparkline, in display order. The reward panel always sits on
/// top; the four entries here fill the rows beneath it.
///
/// Editing this constant rearranges the live dashboard without touching
/// layout math or widget construction — the loop in [`draw_dashboard`]
/// reads it verbatim.
pub const DASHBOARD_METRICS: &[&str] = &["policy_loss", "entropy", "approx_kl", "best_fitness"];

/// Total height (in cells) of the log block, including its border.
const LOG_BLOCK_HEIGHT: u16 = 10;

/// Compose the dashboard layout for one tick.
///
/// The live TUI is metrics-only (ADR-0013): the metric column fills the
/// full width above the log strip and status line. Env playback lives in
/// the post-run report, not here.
pub fn draw_dashboard(frame: &mut Frame<'_>, state: &AppState) {
    let area = frame.area();

    // Outer vertical: metrics above, log strip, single-line status.
    let [metrics_area, logs_area, status_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(LOG_BLOCK_HEIGHT),
        Constraint::Length(1),
    ])
    .areas::<3>(area);

    render_metrics_column(frame, state, metrics_area);
    render_log_block(frame, state, logs_area);
    frame.render_widget(status_paragraph(&state.status), status_area);
}

/// Render the metric column according to `state.metrics_layout`, drawing
/// the metrics named in `state.metrics` (reward sparkline always first).
fn render_metrics_column(frame: &mut Frame<'_>, state: &AppState, area: Rect) {
    match state.metrics_layout {
        MetricsLayout::Combined => render_metrics_combined(frame, state, area),
        MetricsLayout::Separate => render_metrics_separate(frame, state, area),
    }
}

/// Combined layout: one bordered "Metrics" block holding the reward
/// sparkline plus one `MetricSparkline` per name in `state.metrics`. Each
/// occupies a single row; remaining vertical space is left blank.
///
/// Each metric's label is prefixed with its [`Trend`](crate::metrics_registry::Trend)
/// glyph (`↑`/`↓`/`•`) so the compact view still signals which direction is
/// good news. The full interpretation hint is a Separate-layout feature —
/// a single row leaves no room for it here.
fn render_metrics_combined(frame: &mut Frame<'_>, state: &AppState, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title("Metrics");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // One Length(1) per sparkline plus a Min(0) filler so the column
    // remains stable when metrics arrive out of order.
    let constraints: Vec<Constraint> =
        std::iter::repeat_n(Constraint::Length(1), 1 + state.metrics.len())
            .chain(std::iter::once(Constraint::Min(0)))
            .collect();
    let rects = Layout::vertical(constraints).split(inner);

    if let Some(reward_row) = rects.first() {
        frame.render_widget(RewardSparkline::new(state), *reward_row);
    }
    for (i, name) in state.metrics.iter().enumerate() {
        if let Some(row) = rects.get(i + 1) {
            let label = format!("{} {name}", trend_for(name).glyph());
            frame.render_widget(MetricSparkline::new(state, name, &label), *row);
        }
    }
}

/// Separate layout: the reward sparkline and each metric in `state.metrics`
/// get their own bordered, titled panel, splitting the column evenly so
/// every chart has room for taller bars.
///
/// Each panel title carries a trend glyph and a one-line interpretation hint
/// from the metric registry (see [`panel_title`]) so a reader can answer *"is
/// a rising sparkline good news?"* without leaving the dashboard.
fn render_metrics_separate(frame: &mut Frame<'_>, state: &AppState, area: Rect) {
    let panel_count = u32::try_from(1 + state.metrics.len()).unwrap_or(u32::MAX);
    let constraints: Vec<Constraint> =
        std::iter::repeat_n(Constraint::Ratio(1, panel_count), panel_count as usize).collect();
    let rects = Layout::vertical(constraints).split(area);

    if let Some(&rect) = rects.first() {
        // The reward panel plots episode returns, so it borrows that row's
        // registry metadata (higher = learning) for its glyph and hint.
        let block = Block::default()
            .borders(Borders::ALL)
            .title(panel_title("Reward", "episode_return"));
        let inner = block.inner(rect);
        frame.render_widget(block, rect);
        frame.render_widget(RewardSparkline::new(state), inner);
    }
    for (i, name) in state.metrics.iter().enumerate() {
        if let Some(&rect) = rects.get(i + 1) {
            let block = Block::default()
                .borders(Borders::ALL)
                .title(panel_title(name, name));
            let inner = block.inner(rect);
            frame.render_widget(block, rect);
            frame.render_widget(MetricSparkline::bars_only(state, name), inner);
        }
    }
}

/// Compose a metric panel's title line: `↑ display  —  hint`.
///
/// The trend glyph + `display` name are emphasised; the interpretation hint is
/// dimmed and dropped entirely when the registry carries none. `meta_key` is
/// the registry field name supplying the glyph and hint — usually the same as
/// `display`, but the reward panel titles itself "Reward" while reading
/// `episode_return`'s metadata.
fn panel_title(display: &str, meta_key: &str) -> Line<'static> {
    let glyph = trend_for(meta_key).glyph();
    let hint = hint_for(meta_key);

    let mut spans = vec![Span::styled(
        format!("{glyph} {display}"),
        Style::default().add_modifier(RatModifier::BOLD),
    )];
    if !hint.is_empty() {
        spans.push(Span::styled(
            format!("  —  {hint}"),
            Style::default().add_modifier(RatModifier::DIM),
        ));
    }
    Line::from(spans)
}

/// Bordered log block + its inner [`LogPanel`].
fn render_log_block(frame: &mut Frame<'_>, state: &AppState, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title("Logs");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(LogPanel::new(state), inner);
}

/// Wrap [`format_status`] in a dimmed [`Paragraph`] widget for the bottom
/// status row.
fn status_paragraph(status: &StatusLine) -> Paragraph<'static> {
    Paragraph::new(format_status(status)).style(Style::default().add_modifier(RatModifier::DIM))
}

/// Format the bottom status line. Pure — split out so the formatting is
/// trivial to unit-test.
#[must_use]
pub fn format_status(status: &StatusLine) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(4);
    if let Some(name) = status.suite_name.as_ref() {
        parts.push(format!("suite: {name}"));
    }
    if let Some(env) = status.env_name.as_ref() {
        parts.push(format!("env: {env}"));
    }
    if let Some((idx, total)) = status.episode {
        if total > 0 {
            parts.push(format!("episode {}/{total}", idx + 1));
        } else {
            parts.push(format!("episode {}", idx + 1));
        }
    }
    if let Some(r) = status.last_return {
        parts.push(format!("last return: {r:.2}"));
    }
    if status.finished {
        parts.push("finished — press any key to exit".to_string());
    }
    if parts.is_empty() {
        "waiting…".to_string()
    } else {
        parts.join(" | ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Cell;

    use crate::report::EpisodeSummary;
    use crate::suite::{SuiteInfo, TrialInfo, TrialKey};

    fn trial_info(env: &str) -> TrialInfo {
        TrialInfo {
            key: TrialKey {
                env_idx: 0,
                trial_idx: 0,
            },
            env_name: env.to_string(),
            trial_seed: 7,
        }
    }

    #[test]
    fn apply_event_records_episode_return_into_reward_ring() {
        let mut state = AppState::default();
        apply_event(
            &mut state,
            TuiEvent::EpisodeReturn {
                return_value: 12.5,
                length: 64,
            },
        );
        assert_eq!(state.reward_ring.back().copied(), Some(12.5));
        assert_eq!(state.status.last_return, Some(12.5));
        assert!(
            state.status.episode.is_none(),
            "non-harness writer must not populate the episode counter"
        );
    }

    #[test]
    fn apply_event_records_episode_into_reward_ring() {
        let mut state = AppState::default();
        apply_event(
            &mut state,
            TuiEvent::EpisodeEnd {
                trial: trial_info("cartpole"),
                episode: EpisodeSummary {
                    episode_idx: 4,
                    return_value: 195.0,
                    length: 200,
                },
            },
        );
        assert_eq!(state.reward_ring.back().copied(), Some(195.0));
        assert_eq!(state.status.last_return, Some(195.0));
    }

    #[test]
    fn apply_event_suite_start_sets_status() {
        let mut state = AppState::default();
        apply_event(
            &mut state,
            TuiEvent::SuiteStart(SuiteInfo {
                name: "smoke".to_string(),
                env_names: vec!["e1".to_string(), "e2".to_string()],
                num_trials_per_env: 3,
                success_threshold: None,
            }),
        );
        assert_eq!(state.status.suite_name.as_deref(), Some("smoke"));
        // total = env_names.len() * num_trials_per_env = 2 * 3 = 6
        assert_eq!(state.status.episode, Some((0, 6)));
    }

    #[test]
    fn apply_event_metric_update_lands_in_named_ring() {
        let mut state = AppState::default();
        apply_event(
            &mut state,
            TuiEvent::MetricUpdate {
                name: "policy_loss".to_string(),
                value: 0.25,
            },
        );
        let ring = state.metric_rings.get("policy_loss").expect("ring created");
        assert_eq!(ring.iter().copied().collect::<Vec<_>>(), vec![0.25]);
    }

    #[test]
    fn apply_event_log_line_appends_to_log_ring() {
        let mut state = AppState::default();
        apply_event(
            &mut state,
            TuiEvent::LogLine {
                level: tracing::Level::WARN,
                target: "rlevo_rl::ppo".to_string(),
                message: "kl spike".to_string(),
            },
        );
        assert_eq!(state.log_ring.len(), 1);
        let line = state.log_ring.front().unwrap();
        assert_eq!(line.level, tracing::Level::WARN);
        assert_eq!(line.target, "rlevo_rl::ppo");
        assert_eq!(line.message, "kl spike");
    }

    #[test]
    fn apply_event_trial_start_sets_env_keeps_rewards() {
        let mut state = AppState::default();
        state.record_episode_end(0, 1.0);

        apply_event(&mut state, TuiEvent::TrialStart(trial_info("cartpole")));
        assert_eq!(state.reward_ring.len(), 1);
        assert_eq!(state.status.env_name.as_deref(), Some("cartpole"));
    }

    #[test]
    fn format_status_empty_yields_waiting() {
        let s = StatusLine::default();
        assert_eq!(format_status(&s), "waiting…");
    }

    #[test]
    fn format_status_joins_present_fields() {
        let s = StatusLine {
            suite_name: Some("smoke".to_string()),
            env_name: Some("cartpole".to_string()),
            episode: Some((4, 10)),
            last_return: Some(195.5),
            finished: false,
        };
        let out = format_status(&s);
        assert!(out.contains("suite: smoke"));
        assert!(out.contains("env: cartpole"));
        // episode field is 1-based for display ((idx + 1) = 5)
        assert!(
            out.contains("episode 5/10"),
            "missing episode counter in {out:?}"
        );
        assert!(out.contains("last return: 195.50"));
        assert!(!out.contains("finished"));
    }

    #[test]
    fn format_status_marks_finished() {
        let s = StatusLine {
            finished: true,
            ..Default::default()
        };
        let out = format_status(&s);
        assert!(out.contains("finished"));
        assert!(
            out.contains("press any key to exit"),
            "finished status should hint at the dismissal key: {out:?}"
        );
    }

    /// End-to-end-ish: drive a render against a `TestBackend`, confirm the
    /// metric + log panels write content and the status line surfaces the
    /// run metadata. The live TUI is metrics-only (ADR-0013) — there is no
    /// env panel.
    #[test]
    fn draw_dashboard_populates_all_panels() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut state = AppState::default();
        state.mark_suite_start("smoke", 1);
        state.mark_trial_start("cartpole");
        state.record_episode_end(0, 1.0);
        state.record_episode_end(1, 5.0);

        // Feed each metric panel a sample so its sparkline lights up
        // rather than the "no data yet" placeholder.
        for name in DASHBOARD_METRICS {
            state.record_metric(*name, 0.5);
            state.record_metric(*name, 0.4);
        }
        state.record_log(CapturedLogLine {
            level: tracing::Level::INFO,
            target: "rlevo_rl".to_string(),
            message: "training step".to_string(),
        });

        terminal
            .draw(|f| draw_dashboard(f, &state))
            .expect("draw failed");

        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(Cell::symbol)
            .collect();

        // No env panel under ADR-0013.
        assert!(!text.contains("Env"), "env block must not render: {text:?}");
        assert!(text.contains("smoke"), "status suite missing");
        assert!(text.contains("cartpole"), "status env missing");

        // Metric + log panels.
        assert!(text.contains("Metrics"), "metrics block title missing");
        assert!(text.contains("Logs"), "logs block title missing");
        for name in DASHBOARD_METRICS {
            assert!(
                text.contains(name),
                "metric label {name:?} missing in dashboard"
            );
        }
        assert!(
            text.contains("training step"),
            "log message missing from log panel"
        );
    }

    /// Separate layout draws a bordered, titled panel per metric (plus a
    /// "Reward" panel) instead of stacking them in one "Metrics" block.
    #[test]
    fn draw_dashboard_separate_layout_titles_each_metric() {
        const M: &[&str] = &["policy_loss", "entropy", "approx_kl"];
        let backend = TestBackend::new(80, 32);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut state = AppState::default().with_layout(M, MetricsLayout::Separate);
        for name in M {
            state.record_metric(*name, 0.5);
            state.record_metric(*name, 0.4);
        }

        terminal
            .draw(|f| draw_dashboard(f, &state))
            .expect("draw failed");
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(Cell::symbol)
            .collect();

        // Each metric titles its own panel; the dedicated reward panel
        // replaces the combined "Metrics" block.
        assert!(text.contains("Reward"), "reward panel title missing");
        assert!(!text.contains("Metrics"), "combined block leaked through");
        for name in M {
            assert!(text.contains(name), "panel title {name:?} missing");
        }
        // best_fitness was not in the metric set — no empty panel for it.
        assert!(
            !text.contains("best_fitness"),
            "dropped metric should not appear"
        );
    }

    /// The combined-layout labels are prefixed with the metric's trend
    /// glyph so the compact view still cues the good-news direction.
    #[test]
    fn combined_layout_labels_carry_trend_glyph() {
        const M: &[&str] = &["episode_return_mean", "value_loss", "approx_kl"];
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut state = AppState::default().with_layout(M, MetricsLayout::Combined);
        for name in M {
            state.record_metric(*name, 0.5);
        }

        terminal
            .draw(|f| draw_dashboard(f, &state))
            .expect("draw failed");
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(Cell::symbol)
            .collect();

        // One combined "Metrics" block, each metric labelled with its glyph.
        assert!(text.contains("Metrics"), "combined block title missing");
        assert!(
            text.contains('↑'),
            "higher-is-better glyph missing: {text:?}"
        );
        assert!(
            text.contains('↓'),
            "lower-is-better glyph missing: {text:?}"
        );
        assert!(text.contains('•'), "diagnostic glyph missing: {text:?}");
        // Labels remain the raw metric names alongside the glyph.
        assert!(
            text.contains("value_loss"),
            "metric label missing: {text:?}"
        );
    }

    /// The separate-layout panel titles carry a trend glyph and the
    /// registry interpretation hint so the reader knows which direction is
    /// good news.
    #[test]
    fn separate_layout_titles_carry_trend_glyph_and_hint() {
        const M: &[&str] = &["episode_return", "approx_kl"];
        let backend = TestBackend::new(90, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut state = AppState::default().with_layout(M, MetricsLayout::Separate);
        for name in M {
            state.record_metric(*name, 0.5);
        }

        terminal
            .draw(|f| draw_dashboard(f, &state))
            .expect("draw failed");
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(Cell::symbol)
            .collect();

        // Reward panel: "higher is better" glyph + its hint.
        assert!(text.contains('↑'), "up glyph missing: {text:?}");
        assert!(
            text.contains("higher = agent learning"),
            "reward hint missing: {text:?}"
        );
        // A diagnostic metric: neutral glyph + its healthy-range hint.
        assert!(text.contains('•'), "diagnostic glyph missing: {text:?}");
        assert!(
            text.contains("keep small & stable"),
            "approx_kl hint missing: {text:?}"
        );
    }

    /// `render_loop` exits when the shutdown flag flips, regardless of
    /// whether events have stopped arriving. Bounded by `tick_ms`.
    #[test]
    fn render_loop_honours_shutdown_flag() {
        use std::sync::mpsc;
        use std::time::Instant;

        let (tx, rx) = mpsc::channel::<TuiEvent>();
        let backend = TestBackend::new(40, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = AppState::default();
        let shutdown = Arc::new(AtomicBool::new(false));

        let s = Arc::clone(&shutdown);
        let join = thread::spawn(move || render_loop(&mut terminal, &mut state, &rx, &s, 20));

        // Push one event so the loop isn't entirely idle.
        let _ = tx.send(TuiEvent::MetricUpdate {
            name: "policy_loss".to_string(),
            value: 0.1,
        });

        // Now signal shutdown and observe the join.
        let start = Instant::now();
        shutdown.store(true, Ordering::Release);
        let result = join.join().expect("thread join");
        let elapsed = start.elapsed();

        result.expect("render_loop io");
        // Within a small multiple of tick_ms — bound generously for CI flakiness.
        assert!(
            elapsed < Duration::from_millis(500),
            "render_loop exited too slowly: {elapsed:?}"
        );

        // Keep the sender alive until after the join to avoid the
        // disconnected branch being hit accidentally.
        drop(tx);
    }

    /// `render_loop` returns cleanly when all senders drop, even if the
    /// shutdown flag was never set. This is the fallback path when the
    /// caller forgets to call `shutdown()`.
    #[test]
    fn render_loop_exits_on_disconnect() {
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel::<TuiEvent>();
        let backend = TestBackend::new(40, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = AppState::default();
        let shutdown = Arc::new(AtomicBool::new(false));

        let join =
            thread::spawn(move || render_loop(&mut terminal, &mut state, &rx, &shutdown, 20));

        drop(tx);
        join.join().unwrap().expect("render_loop io");
    }
}
