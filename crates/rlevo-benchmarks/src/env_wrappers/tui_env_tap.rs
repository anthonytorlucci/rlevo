//! [`TuiEnvTap`] — an [`Environment`] wrapper that captures styled frames
//! and episode returns for the live TUI.
//!
//! Sibling of [`RenderTap`](crate::env_wrappers::RenderTap). The two differ
//! only in the trait they wrap:
//!
//! - [`RenderTap`] wraps a [`BenchEnv`](rlevo_core::evaluation::BenchEnv) —
//!   the harness-facing trait. It pairs with [`Suite`](crate::suite::Suite)
//!   + [`Evaluator`](crate::evaluator::Evaluator) +
//!     [`TuiReporter`](crate::reporter::tui::TuiReporter), which together
//!     surface lifecycle (`EpisodeEnd`) events.
//! - [`TuiEnvTap`] wraps a raw [`Environment`] — the trait every RL/EA
//!   algorithm crate drives directly. Used by training loops that bypass
//!   the benchmarks harness (PPO's
//!   [`train_discrete`](https://docs.rs/rlevo-reinforcement-learning),
//!   future evolutionary loops). Since no `Reporter` is involved, the
//!   wrapper itself accumulates per-episode reward + step count and
//!   emits a [`TuiEvent::EpisodeReturn`] on termination.
//!
//! Frame capture is best-effort and lossy via
//! [`TuiHandle::try_push_frame`]; episode-return emission is similarly
//! lossy via [`TuiHandle::try_push_episode_return`]. The wrapped env never
//! stalls on render-thread state.
//!
//! [`RenderTap`]: crate::env_wrappers::RenderTap
//! [`TuiEvent::EpisodeReturn`]: crate::reporter::tui::TuiEvent::EpisodeReturn

use rlevo_core::environment::{Environment, EnvironmentError, Snapshot};
use rlevo_core::render::{AsciiRenderable, StyledFrame};

use crate::reporter::tui::TuiHandle;

/// Transparent [`Environment`] wrapper that emits frames and episode returns to the TUI.
///
/// After each successful `reset` / `step` a [`StyledFrame`] is forwarded to
/// the render thread. When `step` returns a done snapshot an
/// [`EpisodeReturn`](crate::reporter::tui::TuiEvent::EpisodeReturn) event is
/// also emitted with the summed reward and step count.
///
/// Requires `E: Environment<D, SD, AD> + AsciiRenderable`. Any env that
/// ships the styled projection drops in without further plumbing.
///
/// # Examples
///
/// ```no_run
/// # use rlevo_benchmarks::env_wrappers::tui_env_tap::TuiEnvTap;
/// # use rlevo_benchmarks::reporter::tui::TuiHandle;
/// # fn make_env() -> impl rlevo_core::environment::Environment<1, 1, 1> + rlevo_core::render::AsciiRenderable { todo!() }
/// let (handle, _rx) = TuiHandle::channel();
/// let mut tap: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(make_env(), handle);
/// tap.reset().unwrap();
/// ```
pub struct TuiEnvTap<E, const D: usize, const SD: usize, const AD: usize> {
    inner: E,
    handle: TuiHandle,
    step: u32,
    episode_return: f64,
    episode_length: u32,
}

impl<E, const D: usize, const SD: usize, const AD: usize> TuiEnvTap<E, D, SD, AD> {
    /// Wrap `inner`, forwarding rendered frames and episode-return events
    /// through `handle`.
    pub const fn new(inner: E, handle: TuiHandle) -> Self {
        Self {
            inner,
            handle,
            step: 0,
            episode_return: 0.0,
            episode_length: 0,
        }
    }

    /// Borrow the wrapped env. Useful for tests that need to inspect
    /// the underlying state directly.
    pub const fn inner(&self) -> &E {
        &self.inner
    }

    /// Mutably borrow the wrapped env.
    pub const fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    /// Consume the tap and return the wrapped env.
    #[must_use]
    pub fn into_inner(self) -> E {
        self.inner
    }

    /// Current step counter (0 immediately after `reset`).
    #[must_use]
    pub const fn step_count(&self) -> u32 {
        self.step
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> std::fmt::Debug
    for TuiEnvTap<E, D, SD, AD>
where
    E: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuiEnvTap")
            .field("step", &self.step)
            .field("episode_return", &self.episode_return)
            .field("episode_length", &self.episode_length)
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> AsciiRenderable
    for TuiEnvTap<E, D, SD, AD>
where
    E: AsciiRenderable,
{
    fn render_ascii(&self) -> String {
        self.inner.render_ascii()
    }
    fn render_styled(&self) -> StyledFrame {
        self.inner.render_styled()
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> Environment<D, SD, AD>
    for TuiEnvTap<E, D, SD, AD>
where
    E: Environment<D, SD, AD> + AsciiRenderable,
{
    type StateType = E::StateType;
    type ObservationType = E::ObservationType;
    type ActionType = E::ActionType;
    type RewardType = E::RewardType;
    type SnapshotType = E::SnapshotType;

    /// Satisfies the [`Environment`] trait bound; prefer [`TuiEnvTap::new`] instead.
    ///
    /// The synthesised [`TuiHandle`] receiver is dropped immediately, so every
    /// frame and episode-return push silently returns `false`.
    fn new(render: bool) -> Self {
        let (handle, _rx) = TuiHandle::channel();
        Self::new(E::new(render), handle)
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let snap = self.inner.reset()?;
        self.step = 0;
        self.episode_return = 0.0;
        self.episode_length = 0;
        let _ = self.handle.try_push_frame(self.step, self.inner.render_styled());
        Ok(snap)
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        let snap = self.inner.step(action)?;
        self.step = self.step.saturating_add(1);
        // `Reward: Into<f32>` is part of the trait definition in `rlevo-core`;
        // call it as a method to avoid the inverse `From` bound on `f32`.
        let r: f32 = snap.reward().clone().into();
        self.episode_return += f64::from(r);
        self.episode_length = self.episode_length.saturating_add(1);
        let _ = self.handle.try_push_frame(self.step, self.inner.render_styled());
        if snap.is_done() {
            let _ = self
                .handle
                .try_push_episode_return(self.episode_return, self.episode_length);
            // Accumulators zero on next `reset`; the caller is expected
            // to call `reset` before resuming stepping.
        }
        Ok(snap)
    }
}

#[cfg(test)]
mod tests {
    //! Tests use an in-file stub `Environment + AsciiRenderable` rather
    //! than depending on `rlevo-environments` to avoid pulling the full
    //! env crate (physics families, hundreds of tests) into the benchmarks
    //! test build. The wrapper's contract is pure delegation + bookkeeping,
    //! so a stub gives equivalent coverage.

    use std::sync::mpsc::Receiver;

    use rlevo_core::base::{Action, Observation, State};
    use rlevo_core::environment::{Environment, EnvironmentError, EpisodeStatus, SnapshotBase};
    use rlevo_core::render::{AsciiRenderable, StyledFrame};
    use rlevo_core::reward::ScalarReward;
    use serde::{Deserialize, Serialize};

    use super::TuiEnvTap;
    use crate::reporter::tui::{TuiEvent, TuiHandle};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    struct StubObs {
        pos: i32,
    }

    impl Observation<1> for StubObs {
        fn shape() -> [usize; 1] {
            [1]
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct StubState {
        pos: i32,
    }

    impl State<1> for StubState {
        type Observation = StubObs;
        fn shape() -> [usize; 1] {
            [1]
        }
        fn is_valid(&self) -> bool {
            true
        }
        fn numel(&self) -> usize {
            1
        }
        fn observe(&self) -> StubObs {
            StubObs { pos: self.pos }
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct StubAction;

    impl Action<1> for StubAction {
        fn shape() -> [usize; 1] {
            [1]
        }
        fn is_valid(&self) -> bool {
            true
        }
    }

    /// Configurable termination behaviour so each test drives a controlled
    /// trajectory.
    #[derive(Debug, Clone, Copy)]
    enum Termination {
        /// Always returns `Running`. Used for the lossy-on-drop check
        /// where we just need a few step events.
        Never,
        /// Returns `Terminated` on step `n`, `Running` otherwise.
        TerminateAt(u32),
        /// Returns `Truncated` on step `n`, `Running` otherwise.
        TruncateAt(u32),
    }

    struct StubEnv {
        pos: i32,
        step: u32,
        reward: f32,
        termination: Termination,
    }

    impl StubEnv {
        const fn with_termination(reward: f32, termination: Termination) -> Self {
            Self {
                pos: 0,
                step: 0,
                reward,
                termination,
            }
        }
    }

    impl AsciiRenderable for StubEnv {
        fn render_ascii(&self) -> String {
            format!("pos={}", self.pos)
        }
    }

    impl Environment<1, 1, 1> for StubEnv {
        type StateType = StubState;
        type ObservationType = StubObs;
        type ActionType = StubAction;
        type RewardType = ScalarReward;
        type SnapshotType = SnapshotBase<1, StubObs, ScalarReward>;

        fn new(_render: bool) -> Self {
            Self::with_termination(0.0, Termination::Never)
        }

        fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
            self.pos = 0;
            self.step = 0;
            Ok(SnapshotBase::running(StubObs { pos: 0 }, ScalarReward(0.0)))
        }

        fn step(&mut self, _action: StubAction) -> Result<Self::SnapshotType, EnvironmentError> {
            self.pos += 1;
            self.step += 1;
            let status = match self.termination {
                Termination::TerminateAt(n) if self.step == n => EpisodeStatus::Terminated,
                Termination::TruncateAt(n) if self.step == n => EpisodeStatus::Truncated,
                _ => EpisodeStatus::Running,
            };
            let obs = StubObs { pos: self.pos };
            let r = ScalarReward(self.reward);
            Ok(match status {
                EpisodeStatus::Running => SnapshotBase::running(obs, r),
                EpisodeStatus::Terminated => SnapshotBase::terminated(obs, r),
                EpisodeStatus::Truncated => SnapshotBase::truncated(obs, r),
            })
        }
    }

    fn collect_frames(rx: &Receiver<TuiEvent>) -> Vec<(u32, StyledFrame)> {
        let mut out = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let TuiEvent::Frame { step, frame } = event {
                out.push((step, frame));
            }
        }
        out
    }

    fn collect_episode_returns(rx: &Receiver<TuiEvent>) -> Vec<(f64, u32)> {
        let mut out = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let TuiEvent::EpisodeReturn {
                return_value,
                length,
            } = event
            {
                out.push((return_value, length));
            }
        }
        out
    }

    /// Drain every event of any variant. Used when a test needs to inspect
    /// the full interleaved order of frames + episode returns.
    fn drain_all(rx: &Receiver<TuiEvent>) -> Vec<TuiEvent> {
        let mut out = Vec::new();
        while let Ok(event) = rx.try_recv() {
            out.push(event);
        }
        out
    }

    #[test]
    fn reset_emits_initial_frame_at_step_zero() {
        let (handle, rx) = TuiHandle::channel();
        let env = StubEnv::with_termination(0.0, Termination::Never);
        let mut tap: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(env, handle);
        tap.reset().unwrap();

        let frames = collect_frames(&rx);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].0, 0);
        assert_eq!(frames[0].1.plain_text(), "pos=0");
        assert_eq!(tap.step_count(), 0);
    }

    #[test]
    fn step_emits_frame_with_incremented_counter() {
        let (handle, rx) = TuiHandle::channel();
        let env = StubEnv::with_termination(0.0, Termination::Never);
        let mut tap: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(env, handle);
        tap.reset().unwrap();
        tap.step(StubAction).unwrap();
        tap.step(StubAction).unwrap();

        let frames = collect_frames(&rx);
        let steps: Vec<u32> = frames.iter().map(|(s, _)| *s).collect();
        assert_eq!(steps, vec![0, 1, 2]);
        assert_eq!(frames[2].1.plain_text(), "pos=2");
        assert_eq!(tap.step_count(), 2);
    }

    /// On `Terminated`, the wrapper emits one `EpisodeReturn` carrying the
    /// summed per-step reward and the step count.
    #[test]
    fn episode_return_fires_on_terminated() {
        let (handle, rx) = TuiHandle::channel();
        let env = StubEnv::with_termination(1.0, Termination::TerminateAt(3));
        let mut tap: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(env, handle);
        tap.reset().unwrap();
        tap.step(StubAction).unwrap();
        tap.step(StubAction).unwrap();
        tap.step(StubAction).unwrap();

        let returns = collect_episode_returns(&rx);
        assert_eq!(returns.len(), 1);
        let (ret, length) = returns[0];
        assert!((ret - 3.0).abs() < f64::EPSILON);
        assert_eq!(length, 3);
    }

    /// `Truncated` is treated symmetrically with `Terminated` — both are
    /// `is_done`, and both must surface a return for the sparkline.
    #[test]
    fn episode_return_fires_on_truncated() {
        let (handle, rx) = TuiHandle::channel();
        let env = StubEnv::with_termination(0.5, Termination::TruncateAt(4));
        let mut tap: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(env, handle);
        tap.reset().unwrap();
        for _ in 0..4 {
            tap.step(StubAction).unwrap();
        }

        let returns = collect_episode_returns(&rx);
        assert_eq!(returns.len(), 1);
        let (ret, length) = returns[0];
        assert!((ret - 2.0).abs() < f64::EPSILON);
        assert_eq!(length, 4);
    }

    /// Across two episodes the accumulators must reset on `reset` so each
    /// episode reports its own return, not a running total.
    #[test]
    fn accumulators_reset_across_episodes() {
        let (handle, rx) = TuiHandle::channel();
        let env = StubEnv::with_termination(1.0, Termination::TerminateAt(2));
        let mut tap: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(env, handle);

        tap.reset().unwrap();
        tap.step(StubAction).unwrap();
        tap.step(StubAction).unwrap();
        tap.reset().unwrap();
        tap.step(StubAction).unwrap();
        tap.step(StubAction).unwrap();

        let returns = collect_episode_returns(&rx);
        assert_eq!(returns.len(), 2);
        let totals: Vec<f64> = returns.iter().map(|(r, _)| *r).collect();
        let lengths: Vec<u32> = returns.iter().map(|(_, l)| *l).collect();
        assert!(totals.iter().all(|r| (r - 2.0).abs() < f64::EPSILON));
        assert_eq!(lengths, vec![2, 2]);
    }

    /// Within an episode the frame for the terminating step must still be
    /// emitted before the `EpisodeReturn` event, so the env panel shows
    /// the final state at the moment the reward sparkline updates.
    #[test]
    fn final_frame_precedes_episode_return() {
        let (handle, rx) = TuiHandle::channel();
        let env = StubEnv::with_termination(1.0, Termination::TerminateAt(2));
        let mut tap: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(env, handle);
        tap.reset().unwrap();
        tap.step(StubAction).unwrap();
        tap.step(StubAction).unwrap();

        let events = drain_all(&rx);
        // reset emits one Frame at step 0; each step emits one Frame; the
        // terminating step also emits an EpisodeReturn. Order must be:
        // Frame(0), Frame(1), Frame(2), EpisodeReturn.
        assert_eq!(events.len(), 4);
        assert!(matches!(events[0], TuiEvent::Frame { step: 0, .. }));
        assert!(matches!(events[1], TuiEvent::Frame { step: 1, .. }));
        assert!(matches!(events[2], TuiEvent::Frame { step: 2, .. }));
        assert!(matches!(events[3], TuiEvent::EpisodeReturn { .. }));
    }

    /// A dropped receiver disables every push; the rollout loop must
    /// continue regardless.
    #[test]
    fn lossy_on_dropped_receiver() {
        let (handle, rx) = TuiHandle::channel();
        let env = StubEnv::with_termination(1.0, Termination::TerminateAt(3));
        let mut tap: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(env, handle);
        drop(rx);

        // None of these calls should panic or block.
        tap.reset().unwrap();
        tap.step(StubAction).unwrap();
        tap.step(StubAction).unwrap();
        tap.step(StubAction).unwrap();
        assert_eq!(tap.step_count(), 3);
    }

    /// Consuming `into_inner` yields the original env with its state intact.
    #[test]
    fn into_inner_returns_wrapped_env() {
        let (handle, _rx) = TuiHandle::channel();
        let env = StubEnv::with_termination(0.0, Termination::Never);
        let tap: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(env, handle);
        let inner = tap.into_inner();
        assert_eq!(inner.render_ascii(), "pos=0");
    }
}
