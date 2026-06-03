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

use std::sync::Arc;

use parking_lot::Mutex;
use rlevo_core::environment::{Environment, EnvironmentError, Snapshot};
use rlevo_core::render::{
    AsciiRenderable, Box2dPayloadSource, Landscape2DPayloadSource, Locomotion2DPayloadSource,
    StyledFrame,
};
use serde::Serialize;

use super::schema::{
    Box2dPayload, FamilyPayload, FrameRecord, Landscape2DPayload, Locomotion2DPayload,
};
use super::writer::RecordSink;

/// Boxed extractor that turns the wrapped env into a [`FamilyPayload`]
/// at every frame. Default constructions return [`FamilyPayload::Ascii`]
/// — callers opt in to richer payloads via
/// [`RecordingTap::with_payload_extractor`] or one of the per-family
/// convenience constructors.
pub type PayloadExtractor<E> = Box<dyn Fn(&E) -> FamilyPayload + Send + Sync>;

/// Boxed extractor for the optional ASCII rendering surface.
///
/// Returns `None` for envs that do not implement [`AsciiRenderable`] —
/// the `locomotion` family is the canonical example.
pub type AsciiExtractor<E> = Box<dyn Fn(&E) -> Option<String> + Send + Sync>;

/// Boxed extractor for the optional styled rendering surface.
///
/// Returns `None` for envs that do not implement [`AsciiRenderable`] —
/// the `locomotion` family is the canonical example.
pub type StyledExtractor<E> = Box<dyn Fn(&E) -> Option<StyledFrame> + Send + Sync>;

/// Environment wrapper that records every reset and step to a [`RecordSink`].
///
/// Pushes a [`FrameRecord`] after every successful `reset` / `step`,
/// calling [`RecordSink::on_episode_start`] at each new episode and
/// [`RecordSink::on_episode_end`] on termination or truncation.
///
/// Action encoding requires `Environment::ActionType: Serialize`. Most
/// env action types either already derive [`Serialize`] or only need a
/// one-line `#[derive(...)]` addition to opt in to recording — see the
/// example wired in `crates/rlevo/examples/viz/record_ppo_cartpole.rs`.
pub struct RecordingTap<E, const D: usize, const SD: usize, const AD: usize> {
    inner: E,
    sink: Arc<Mutex<dyn RecordSink>>,
    payload_extractor: PayloadExtractor<E>,
    ascii_extractor: AsciiExtractor<E>,
    styled_extractor: StyledExtractor<E>,
    step: u32,
    episode_idx: u32,
    episode_return: f64,
    episode_length: u32,
    started: bool,
}

impl<E, const D: usize, const SD: usize, const AD: usize> RecordingTap<E, D, SD, AD>
where
    E: AsciiRenderable,
{
    /// Wrap `inner`, routing every per-step record to `sink`. Frames
    /// ship with `FamilyPayload::Ascii` plus an `ascii` + `styled`
    /// projection from the env's [`AsciiRenderable`] surface — call
    /// [`with_payload_extractor`](Self::with_payload_extractor) or one
    /// of the per-family `with_*_payload` constructors to capture
    /// richer payloads instead. For locomotion envs (which deliberately
    /// have no [`AsciiRenderable`] impl) use
    /// [`with_locomotion_payload`](Self::with_locomotion_payload) — it
    /// goes through [`new_headless`](Self::new_headless) and ships
    /// `ascii = None` / `styled = None`.
    pub fn new(inner: E, sink: Arc<Mutex<dyn RecordSink>>) -> Self {
        Self::with_payload_extractor(inner, sink, |_| FamilyPayload::Ascii)
    }

    /// Wrap `inner` with a custom payload extractor. The closure runs
    /// once per captured frame and is responsible for projecting the
    /// env's state into a [`FamilyPayload`] variant. ASCII / styled
    /// extraction defaults to the env's [`AsciiRenderable`] impl.
    pub fn with_payload_extractor<F>(inner: E, sink: Arc<Mutex<dyn RecordSink>>, extractor: F) -> Self
    where
        F: Fn(&E) -> FamilyPayload + Send + Sync + 'static,
    {
        Self {
            inner,
            sink,
            payload_extractor: Box::new(extractor),
            ascii_extractor: Box::new(|e: &E| Some(e.render_ascii())),
            styled_extractor: Box::new(|e: &E| Some(e.render_styled())),
            step: 0,
            episode_idx: 0,
            episode_return: 0.0,
            episode_length: 0,
            started: false,
        }
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> RecordingTap<E, D, SD, AD> {
    /// Headless constructor — does **not** require [`AsciiRenderable`]
    /// on `E`. Frames ship with `ascii = None` and `styled = None`;
    /// the family payload is the only rendering surface. Intended for
    /// the `locomotion` family (no `AsciiRenderable` impl).
    pub fn new_headless<F>(inner: E, sink: Arc<Mutex<dyn RecordSink>>, payload: F) -> Self
    where
        F: Fn(&E) -> FamilyPayload + Send + Sync + 'static,
    {
        Self {
            inner,
            sink,
            payload_extractor: Box::new(payload),
            ascii_extractor: Box::new(|_: &E| None),
            styled_extractor: Box::new(|_: &E| None),
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

impl<E, const D: usize, const SD: usize, const AD: usize> RecordingTap<E, D, SD, AD>
where
    E: AsciiRenderable + Landscape2DPayloadSource + 'static,
{
    /// Convenience constructor for `landscapes` envs: extracts a
    /// [`Landscape2DPayload`] per frame via [`Landscape2DPayloadSource`].
    /// Landscape envs implement [`AsciiRenderable`], so the static-frame
    /// ascii / styled projection is captured alongside the rich payload.
    pub fn with_landscape_payload(inner: E, sink: Arc<Mutex<dyn RecordSink>>) -> Self {
        Self::with_payload_extractor(inner, sink, |e| {
            FamilyPayload::Landscape2D(Landscape2DPayload::from(e.landscape2d_snapshot()))
        })
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> RecordingTap<E, D, SD, AD>
where
    E: AsciiRenderable + Box2dPayloadSource + 'static,
{
    /// Convenience constructor for `box2d` envs: extracts a
    /// [`Box2dPayload`] per frame via [`Box2dPayloadSource`]. `Box2D` envs
    /// implement [`AsciiRenderable`], so the static-frame ascii / styled
    /// projection is captured alongside the rich payload.
    pub fn with_box2d_payload(inner: E, sink: Arc<Mutex<dyn RecordSink>>) -> Self {
        Self::with_payload_extractor(inner, sink, |e| {
            FamilyPayload::Box2dBodies(Box2dPayload::from(e.box2d_snapshot()))
        })
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> RecordingTap<E, D, SD, AD>
where
    E: Locomotion2DPayloadSource + 'static,
{
    /// Convenience constructor for `locomotion` envs: extracts a
    /// [`Locomotion2DPayload`] per frame via [`Locomotion2DPayloadSource`].
    /// This is locomotion's only rendering pathway in the report tier.
    /// Uses [`new_headless`](Self::new_headless) because locomotion envs
    /// do not implement [`AsciiRenderable`].
    pub fn with_locomotion_payload(inner: E, sink: Arc<Mutex<dyn RecordSink>>) -> Self {
        Self::new_headless(inner, sink, |e| {
            FamilyPayload::Locomotion2D(Locomotion2DPayload::from(e.locomotion2d_snapshot()))
        })
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
            .field("payload_extractor", &"Box<dyn Fn>")
            .field("ascii_extractor", &"Box<dyn Fn>")
            .field("styled_extractor", &"Box<dyn Fn>")
            .field("step", &self.step)
            .field("episode_idx", &self.episode_idx)
            .field("episode_return", &self.episode_return)
            .field("episode_length", &self.episode_length)
            .field("started", &self.started)
            .finish()
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> AsciiRenderable
    for RecordingTap<E, D, SD, AD>
where
    E: AsciiRenderable,
{
    fn render_ascii(&self) -> String {
        self.inner.render_ascii()
    }

    fn render_styled(&self) -> rlevo_core::render::StyledFrame {
        self.inner.render_styled()
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> RecordingTap<E, D, SD, AD>
where
    E: Environment<D, SD, AD>,
    E::ActionType: Serialize,
{
    fn capture_frame(&mut self, action_bytes: Vec<u8>, reward: f32) {
        let payload = (self.payload_extractor)(&self.inner);
        let ascii = (self.ascii_extractor)(&self.inner);
        let styled = (self.styled_extractor)(&self.inner);
        let record = FrameRecord {
            step: self.step,
            action: action_bytes,
            reward,
            ascii,
            styled,
            family_payload: payload,
        };
        self.sink.lock().on_frame(record);
    }
}

impl<E, const D: usize, const SD: usize, const AD: usize> Environment<D, SD, AD>
    for RecordingTap<E, D, SD, AD>
where
    E: Environment<D, SD, AD>,
    E::ActionType: Serialize + Clone,
{
    type StateType = E::StateType;
    type ObservationType = E::ObservationType;
    type ActionType = E::ActionType;
    type RewardType = E::RewardType;
    type SnapshotType = E::SnapshotType;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let snap = self.inner.reset()?;
        if self.started {
            self.episode_idx = self.episode_idx.saturating_add(1);
        }
        self.started = true;
        self.step = 0;
        self.episode_return = 0.0;
        self.episode_length = 0;
        self.sink.lock().on_episode_start(self.episode_idx);
        self.capture_frame(Vec::new(), 0.0);
        Ok(snap)
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        // On encode failure, surface it through `take_error` rather than
        // letting the action silently collapse to the same empty-bytes
        // sentinel a reset frame uses. The frame is still emitted (with an
        // empty action) so the trajectory stays decodable; `self.step + 1`
        // is the frame's eventual index since `self.step` bumps below.
        let action_bytes =
            match bincode::serde::encode_to_vec(&action, super::schema::bincode_config()) {
                Ok(bytes) => bytes,
                Err(e) => {
                    self.sink
                        .lock()
                        .record_external_error(super::error::RecordError::ActionEncode {
                            step: self.step.saturating_add(1),
                            message: e.to_string(),
                        });
                    Vec::new()
                }
            };
        let snap = self.inner.step(action)?;
        self.step = self.step.saturating_add(1);
        let r: f32 = snap.reward().clone().into();
        self.episode_return += f64::from(r);
        self.episode_length = self.episode_length.saturating_add(1);
        self.capture_frame(action_bytes, r);
        if snap.is_done() {
            self.sink
                .lock()
                .on_episode_end(self.episode_return, self.episode_length);
        }
        Ok(snap)
    }
}

#[cfg(test)]
mod tests {
    //! Tests use an in-file stub `Environment + AsciiRenderable` matching
    //! the `tui_env_tap` pattern: a stub gives the same coverage as a
    //! real env without pulling the env crate's full test surface.

    use std::sync::Arc;

    use parking_lot::Mutex;
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

        let probe = probe.lock();
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

        let probe = probe.lock();
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

        let probe = probe.lock();
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

        let probe = probe.lock();
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

        let probe = probe.lock();
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

    #[test]
    fn default_constructor_emits_ascii_payload() {
        let (probe, sink) = sink_handle();
        let env = StubEnv::new(0.0, Termination::TerminateAt(u32::MAX));
        let mut tap: RecordingTap<_, 1, 1, 1> = RecordingTap::new(env, sink);
        tap.reset().unwrap();
        let probe = probe.lock();
        let ep = &probe.episodes[&0];
        assert!(matches!(
            ep.frames[0].family_payload,
            crate::record::FamilyPayload::Ascii
        ));
    }

    #[test]
    fn custom_payload_extractor_lands_on_each_frame() {
        use rlevo_core::render::{Landscape2DSnapshot, Point2};

        use crate::record::{FamilyPayload, Landscape2DPayload};

        let (probe, sink) = sink_handle();
        let env = StubEnv::new(0.5, Termination::TerminateAt(2));
        let mut tap: RecordingTap<_, 1, 1, 1> =
            RecordingTap::with_payload_extractor(env, sink, |stub| {
                FamilyPayload::Landscape2D(Landscape2DPayload::from(Landscape2DSnapshot {
                    bounds_x: (-1.0, 1.0),
                    bounds_y: (-1.0, 1.0),
                    current: Point2::new(f32::from(i16::try_from(stub.pos).unwrap_or(0)), 0.0),
                    best: None,
                    trail: vec![],
                    label: "stub".into(),
                }))
            });
        tap.reset().unwrap();
        tap.step(StubAction(0)).unwrap();
        tap.step(StubAction(0)).unwrap();

        let probe = probe.lock();
        let ep = &probe.episodes[&0];
        for frame in &ep.frames {
            match &frame.family_payload {
                FamilyPayload::Landscape2D(p) => {
                    assert_eq!(p.label, "stub");
                }
                other => panic!("expected Landscape2D, got {other:?}"),
            }
        }
    }

    /// An action whose `Serialize` impl always fails, to exercise the
    /// encode-failure path in `RecordingTap::step`.
    #[derive(Debug, Clone, Copy)]
    struct FailAction;

    impl Serialize for FailAction {
        fn serialize<S: serde::Serializer>(&self, _s: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("intentional encode failure"))
        }
    }

    impl Action<1> for FailAction {
        fn shape() -> [usize; 1] {
            [1]
        }
        fn is_valid(&self) -> bool {
            true
        }
    }

    /// Minimal env over [`FailAction`]; reuses the stub state/obs and never
    /// terminates. Headless (no `AsciiRenderable`) — the tap is built with
    /// `new_headless`.
    struct FailEnv {
        pos: i32,
    }

    impl Environment<1, 1, 1> for FailEnv {
        type StateType = StubState;
        type ObservationType = StubObs;
        type ActionType = FailAction;
        type RewardType = ScalarReward;
        type SnapshotType = SnapshotBase<1, StubObs, ScalarReward>;

        fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
            self.pos = 0;
            Ok(SnapshotBase::running(StubObs { pos: 0 }, ScalarReward(0.0)))
        }

        fn step(&mut self, _action: FailAction) -> Result<Self::SnapshotType, EnvironmentError> {
            self.pos += 1;
            Ok(SnapshotBase::running(
                StubObs { pos: self.pos },
                ScalarReward(0.0),
            ))
        }
    }

    #[test]
    fn step_encode_failure_surfaces_via_take_error_and_still_emits_frame() {
        use crate::record::{FamilyPayload, RecordError};
        use crate::record::writer::RecordSink;

        let (probe, sink) = sink_handle();
        let env = FailEnv { pos: 0 };
        let mut tap: RecordingTap<_, 1, 1, 1> =
            RecordingTap::new_headless(env, sink, |_| FamilyPayload::Ascii);
        tap.reset().unwrap();
        // The inner step still succeeds; only the action encode fails.
        tap.step(FailAction).unwrap();
        tap.step(FailAction).unwrap();

        let mut guard = probe.lock();
        let ep = &guard.episodes[&0];
        // reset frame + two step frames are all present — the encode failure
        // did not desync bookkeeping.
        assert_eq!(ep.frames.len(), 3);
        // The step frames ship with an empty action, distinguishable from a
        // legitimate reset frame only via take_error.
        assert_eq!(ep.frames[1].step, 1);
        assert!(ep.frames[1].action.is_empty());

        // First-error-wins: the surfaced error is the *first* failing step.
        match guard.take_error() {
            Some(RecordError::ActionEncode { step, .. }) => assert_eq!(step, 1),
            other => panic!("expected ActionEncode at step 1, got {other:?}"),
        }
        // take_error cleared it.
        assert!(guard.take_error().is_none());
    }
}
