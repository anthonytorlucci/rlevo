//! [`TuiEnvTap`] — an [`Environment`] wrapper that emits episode returns to
//! the live (metrics-only) TUI.
//!
//! [`TuiEnvTap`] wraps a raw [`Environment`] — the trait every RL/EA
//! algorithm crate drives directly. It is used by training loops that bypass
//! the benchmarks harness (PPO's
//! [`train_discrete`](https://docs.rs/rlevo-reinforcement-learning), future
//! evolutionary loops). Since no `Reporter` is involved, the wrapper itself
//! accumulates per-episode reward + step count and emits a
//! [`TuiEvent::EpisodeReturn`] on termination, feeding the reward sparkline.
//!
//! The live TUI is metrics-only (ADR-0013): the tap no longer captures env
//! frames, only the per-episode return. Emission is best-effort and lossy
//! via [`TuiHandle::try_push_episode_return`]; the wrapped env never stalls
//! on render-thread state.
//!
//! [`TuiEvent::EpisodeReturn`]: crate::reporter::tui::TuiEvent::EpisodeReturn

use rlevo_core::environment::{Environment, EnvironmentError, Snapshot};
use rlevo_core::render::{AsciiRenderable, StyledFrame};

use crate::reporter::tui::TuiHandle;

/// Transparent [`Environment`] wrapper that emits episode returns to the TUI.
///
/// When `step` returns a done snapshot an
/// [`EpisodeReturn`](crate::reporter::tui::TuiEvent::EpisodeReturn) event is
/// emitted with the summed reward and step count.
///
/// Requires only `E: Environment<D, SD, AD>`. A separate
/// [`AsciiRenderable`] forwarding impl is provided when the inner env is
/// renderable, so the tap composes under a
/// [`RecordingTap`](crate::record::RecordingTap) that records env frames to
/// disk.
///
/// # Examples
///
/// ```no_run
/// # use rlevo_benchmarks::env_wrappers::tui_env_tap::TuiEnvTap;
/// # use rlevo_benchmarks::reporter::tui::TuiHandle;
/// # use rlevo_core::environment::Environment;
/// # fn demo<E: Environment<1, 1, 1>>(env: E) {
/// let (handle, _rx) = TuiHandle::channel();
/// let mut tap: TuiEnvTap<E, 1, 1, 1> = TuiEnvTap::new(env, handle);
/// tap.reset().unwrap();
/// # }
/// ```
pub struct TuiEnvTap<E, const D: usize, const SD: usize, const AD: usize> {
    inner: E,
    handle: TuiHandle,
    step: u32,
    episode_return: f64,
    episode_length: u32,
}

impl<E, const D: usize, const SD: usize, const AD: usize> TuiEnvTap<E, D, SD, AD> {
    /// Wrap `inner`, emitting per-episode return events through `handle`.
    /// The live TUI is metrics-only (ADR-0013): no env frames are
    /// captured, only the summed reward + step count on episode end.
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

/// Forward the optional `Classic2DPayloadSource` through to the wrapped env,
/// so a `TuiEnvTap` over a classic-control env composes under a
/// `RecordingTap::with_classic2d_payload` (ADR-0013 structured recording).
impl<E, const D: usize, const SD: usize, const AD: usize>
    rlevo_core::render::payload::Classic2DPayloadSource for TuiEnvTap<E, D, SD, AD>
where
    E: rlevo_core::render::payload::Classic2DPayloadSource,
{
    fn classic2d_snapshot(&self) -> rlevo_core::render::payload::Classic2DSnapshot {
        self.inner.classic2d_snapshot()
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> Environment<D, SD, AD>
    for TuiEnvTap<E, D, SD, AD>
where
    E: Environment<D, SD, AD>,
{
    type StateType = E::StateType;
    type ObservationType = E::ObservationType;
    type ActionType = E::ActionType;
    type RewardType = E::RewardType;
    type SnapshotType = E::SnapshotType;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let snap = self.inner.reset()?;
        self.step = 0;
        self.episode_return = 0.0;
        self.episode_length = 0;
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
    use rlevo_core::render::AsciiRenderable;
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

    /// Stepping advances the wrapper's step counter even though no frames
    /// are emitted (the live TUI is metrics-only, ADR-0013).
    #[test]
    fn step_advances_counter() {
        let (handle, _rx) = TuiHandle::channel();
        let env = StubEnv::with_termination(0.0, Termination::Never);
        let mut tap: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(env, handle);
        tap.reset().unwrap();
        tap.step(StubAction).unwrap();
        tap.step(StubAction).unwrap();
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

    /// Exactly one `EpisodeReturn` is emitted per terminating episode and
    /// nothing else lands on the channel (no per-step frames).
    #[test]
    fn only_episode_return_is_emitted() {
        let (handle, rx) = TuiHandle::channel();
        let env = StubEnv::with_termination(1.0, Termination::TerminateAt(2));
        let mut tap: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(env, handle);
        tap.reset().unwrap();
        tap.step(StubAction).unwrap();
        tap.step(StubAction).unwrap();

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], TuiEvent::EpisodeReturn { .. }));
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
