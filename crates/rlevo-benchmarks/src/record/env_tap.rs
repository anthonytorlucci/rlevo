//! [`RecordingTap`] — an [`Environment`] wrapper that drives the
//! [`RecordSink`] surface from a raw env trajectory.
//!
//! Sibling of [`TuiEnvTap`](crate::env_wrappers::TuiEnvTap). The two
//! differ in the sink they feed:
//!
//! - [`TuiEnvTap`] pushes through a `TuiHandle` channel for the live
//!   TUI panels (best-effort, lossy on a slow render thread).
//! - [`RecordingTap`] pushes through an
//!   [`Arc<Mutex<dyn RecordSink>>`](RecordSink), which the
//!   [`RecordWriter`](crate::record::RecordWriter) drains to disk. The
//!   wrapper still owns episode-return + length bookkeeping since no
//!   harness is involved.
//!
//! The two taps compose: `RecordingTap::new(TuiEnvTap::new(env, ...))`
//! drives the live TUI and the on-disk record from one env without
//! double-stepping the inner.
//!
//! [`TuiEnvTap`]: crate::env_wrappers::TuiEnvTap

use std::sync::{Arc, Mutex};

use rlevo_core::environment::{Environment, EnvironmentError, Snapshot};
use rlevo_core::render::AsciiRenderable;
use serde::Serialize;

use super::schema::{FamilyPayload, FrameRecord};
use super::writer::RecordSink;

/// Wraps an [`Environment`] and pushes a [`FrameRecord`] after every
/// successful `reset` / `step`, calling [`RecordSink::on_episode_start`]
/// at each new episode and [`RecordSink::on_episode_end`] on termination
/// or truncation.
///
/// Action encoding requires `Environment::ActionType: Serialize`. Most
/// env action types either already derive [`Serialize`] or only need a
/// one-line `#[derive(...)]` addition to opt in to recording — see the
/// example wired in `crates/rlevo/examples/viz/record_cartpole.rs`.
pub struct RecordingTap<E, const D: usize, const SD: usize, const AD: usize> {
    inner: E,
    sink: Arc<Mutex<dyn RecordSink>>,
    step: u32,
    episode_idx: u32,
    episode_return: f64,
    episode_length: u32,
    started: bool,
}

impl<E, const D: usize, const SD: usize, const AD: usize> RecordingTap<E, D, SD, AD> {
    /// Wrap `inner`, routing every per-step record to `sink`.
    pub fn new(inner: E, sink: Arc<Mutex<dyn RecordSink>>) -> Self {
        Self {
            inner,
            sink,
            step: 0,
            episode_idx: 0,
            episode_return: 0.0,
            episode_length: 0,
            started: false,
        }
    }

    /// Borrow the wrapped env.
    pub const fn inner(&self) -> &E {
        &self.inner
    }

    /// Mutably borrow the wrapped env.
    pub const fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    /// Consume the tap and return the wrapped env.
    pub fn into_inner(self) -> E {
        self.inner
    }

    /// Index of the next episode the wrapper will open on `reset`.
    #[must_use]
    pub const fn episode_idx(&self) -> u32 {
        self.episode_idx
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> std::fmt::Debug
    for RecordingTap<E, D, SD, AD>
where
    E: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingTap")
            .field("inner", &self.inner)
            .field("sink", &"Arc<Mutex<dyn RecordSink>>")
            .field("step", &self.step)
            .field("episode_idx", &self.episode_idx)
            .field("episode_return", &self.episode_return)
            .field("episode_length", &self.episode_length)
            .field("started", &self.started)
            .finish()
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> RecordingTap<E, D, SD, AD>
where
    E: Environment<D, SD, AD> + AsciiRenderable,
    E::ActionType: Serialize,
{
    fn capture_frame(&mut self, action_bytes: Vec<u8>, reward: f32) {
        let record = FrameRecord {
            step: self.step,
            action: action_bytes,
            reward,
            ascii: Some(self.inner.render_ascii()),
            styled: Some(self.inner.render_styled()),
            family_payload: FamilyPayload::Ascii,
        };
        if let Ok(mut sink) = self.sink.lock() {
            sink.on_frame(record);
        }
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> Environment<D, SD, AD>
    for RecordingTap<E, D, SD, AD>
where
    E: Environment<D, SD, AD> + AsciiRenderable,
    E::ActionType: Serialize + Clone,
{
    type StateType = E::StateType;
    type ObservationType = E::ObservationType;
    type ActionType = E::ActionType;
    type RewardType = E::RewardType;
    type SnapshotType = E::SnapshotType;

    fn new(render: bool) -> Self {
        // No meaningful standalone constructor — callers use
        // `RecordingTap::new(env, sink)`. This impl exists only to
        // satisfy the trait shape.
        struct NullSink;
        impl RecordSink for NullSink {
            fn on_episode_start(&mut self, _: u32) {}
            fn on_frame(&mut self, _: FrameRecord) {}
            fn on_metric(&mut self, _: super::schema::MetricSample) {}
            fn on_episode_end(&mut self, _: f64, _: u32) {}
            fn on_run_end(&mut self, _: super::manifest::RunManifest) {}
        }
        Self::new(E::new(render), Arc::new(Mutex::new(NullSink)))
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let snap = self.inner.reset()?;
        if self.started {
            self.episode_idx = self.episode_idx.saturating_add(1);
        }
        self.started = true;
        self.step = 0;
        self.episode_return = 0.0;
        self.episode_length = 0;
        if let Ok(mut sink) = self.sink.lock() {
            sink.on_episode_start(self.episode_idx);
        }
        self.capture_frame(Vec::new(), 0.0);
        Ok(snap)
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        let action_bytes =
            bincode::serde::encode_to_vec(&action, super::schema::bincode_config()).unwrap_or_default();
        let snap = self.inner.step(action)?;
        self.step = self.step.saturating_add(1);
        let r: f32 = snap.reward().clone().into();
        self.episode_return += f64::from(r);
        self.episode_length = self.episode_length.saturating_add(1);
        self.capture_frame(action_bytes, r);
        if snap.is_done()
            && let Ok(mut sink) = self.sink.lock()
        {
            sink.on_episode_end(self.episode_return, self.episode_length);
        }
        Ok(snap)
    }
}

#[cfg(test)]
mod tests {
    //! Tests use an in-file stub `Environment + AsciiRenderable` matching
    //! the `tui_env_tap` pattern: a stub gives the same coverage as a
    //! real env without pulling the env crate's full test surface.

    use std::sync::{Arc, Mutex};

    use rlevo_core::base::{Action, Observation, State};
    use rlevo_core::environment::{Environment, EnvironmentError, EpisodeStatus, SnapshotBase};
    use rlevo_core::render::AsciiRenderable;
    use rlevo_core::reward::ScalarReward;
    use serde::{Deserialize, Serialize};

    use super::RecordingTap;
    use crate::record::writer::InMemoryRecordSink;

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

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    struct StubAction(u8);

    impl Action<1> for StubAction {
        fn shape() -> [usize; 1] {
            [1]
        }
        fn is_valid(&self) -> bool {
            true
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum Termination {
        TerminateAt(u32),
        TruncateAt(u32),
    }

    struct StubEnv {
        pos: i32,
        step: u32,
        reward: f32,
        termination: Termination,
    }

    impl StubEnv {
        const fn new(reward: f32, termination: Termination) -> Self {
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
            Self::new(0.0, Termination::TerminateAt(u32::MAX))
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

    fn sink_handle() -> (Arc<Mutex<InMemoryRecordSink>>, Arc<Mutex<dyn super::RecordSink>>) {
        let s: Arc<Mutex<InMemoryRecordSink>> = Arc::new(Mutex::new(InMemoryRecordSink::new()));
        let dyn_sink: Arc<Mutex<dyn super::RecordSink>> = s.clone();
        (s, dyn_sink)
    }

    #[test]
    fn reset_starts_episode_and_emits_initial_frame() {
        let (probe, sink) = sink_handle();
        let env = StubEnv::new(0.0, Termination::TerminateAt(u32::MAX));
        let mut tap: RecordingTap<_, 1, 1, 1> = RecordingTap::new(env, sink);
        tap.reset().unwrap();

        let probe = probe.lock().unwrap();
        assert_eq!(probe.episodes.len(), 1);
        let ep = &probe.episodes[&0];
        assert_eq!(ep.frames.len(), 1);
        assert_eq!(ep.frames[0].step, 0);
        assert!(ep.frames[0].action.is_empty());
        assert_eq!(ep.frames[0].ascii.as_deref(), Some("pos=0"));
    }

    #[test]
    fn step_appends_frame_with_action_bytes_and_reward() {
        let (probe, sink) = sink_handle();
        let env = StubEnv::new(0.5, Termination::TerminateAt(u32::MAX));
        let mut tap: RecordingTap<_, 1, 1, 1> = RecordingTap::new(env, sink);
        tap.reset().unwrap();
        tap.step(StubAction(7)).unwrap();
        tap.step(StubAction(9)).unwrap();

        let probe = probe.lock().unwrap();
        let ep = &probe.episodes[&0];
        assert_eq!(ep.frames.len(), 3);
        assert_eq!(ep.frames[1].step, 1);
        assert!((ep.frames[1].reward - 0.5).abs() < f32::EPSILON);
        // bincode-standard encoding for a 1-byte tuple struct over u8 is
        // a varint length + payload.
        assert!(!ep.frames[1].action.is_empty());
        assert!(!ep.frames[2].action.is_empty());
    }

    #[test]
    fn termination_closes_episode_with_return_and_length() {
        let (probe, sink) = sink_handle();
        let env = StubEnv::new(1.0, Termination::TerminateAt(3));
        let mut tap: RecordingTap<_, 1, 1, 1> = RecordingTap::new(env, sink);
        tap.reset().unwrap();
        tap.step(StubAction(1)).unwrap();
        tap.step(StubAction(1)).unwrap();
        tap.step(StubAction(1)).unwrap();

        let probe = probe.lock().unwrap();
        // current cleared on on_episode_end → record was finalised
        assert!(probe.current.is_none());
        let ep = &probe.episodes[&0];
        assert_eq!(ep.frames.len(), 4); // reset + 3 steps
    }

    #[test]
    fn truncation_closes_episode_symmetrically() {
        let (probe, sink) = sink_handle();
        let env = StubEnv::new(0.5, Termination::TruncateAt(2));
        let mut tap: RecordingTap<_, 1, 1, 1> = RecordingTap::new(env, sink);
        tap.reset().unwrap();
        tap.step(StubAction(0)).unwrap();
        tap.step(StubAction(0)).unwrap();

        let probe = probe.lock().unwrap();
        assert!(probe.current.is_none());
    }

    #[test]
    fn second_episode_increments_episode_idx() {
        let (probe, sink) = sink_handle();
        let env = StubEnv::new(1.0, Termination::TerminateAt(2));
        let mut tap: RecordingTap<_, 1, 1, 1> = RecordingTap::new(env, sink);
        tap.reset().unwrap();
        tap.step(StubAction(1)).unwrap();
        tap.step(StubAction(1)).unwrap();
        tap.reset().unwrap();
        tap.step(StubAction(2)).unwrap();
        tap.step(StubAction(2)).unwrap();

        let probe = probe.lock().unwrap();
        assert!(probe.episodes.contains_key(&0));
        assert!(probe.episodes.contains_key(&1));
        assert_eq!(probe.episodes[&0].frames.len(), 3);
        assert_eq!(probe.episodes[&1].frames.len(), 3);
    }

    #[test]
    fn into_inner_returns_wrapped_env() {
        let (_probe, sink) = sink_handle();
        let env = StubEnv::new(0.0, Termination::TerminateAt(u32::MAX));
        let tap: RecordingTap<_, 1, 1, 1> = RecordingTap::new(env, sink);
        let inner = tap.into_inner();
        assert_eq!(inner.render_ascii(), "pos=0");
    }
}
