//! [`RenderTap`] — a [`BenchEnv`] wrapper that captures styled frames.
//!
//! `RenderTap<E>` is constructed inside the env factory closure of a
//! [`Suite`](crate::suite::Suite). After each successful `reset` and
//! `step` on the wrapped env, the wrapper calls
//! [`AsciiRenderable::render_styled`] and forwards the frame through a
//! cloned [`TuiHandle`] to the render thread.
//!
//! Frame capture is **best-effort and lossy**: the send goes through
//! [`TuiHandle::try_push_frame`], which never blocks. If the render thread
//! has exited (handle's receiver dropped), the frame is silently
//! discarded. The rollout loop is therefore guaranteed never to stall on
//! render-thread state — a stalled or crashed terminal cannot drag the
//! training run down with it.
//!
//! The wrapper does not own [`TrialInfo`]: the trial context is supplied
//! to the render thread by [`TuiReporter`] via the `TrialStart` event on
//! the same channel. Frames are tagged only with a monotonic step counter
//! (reset on each `reset`).

use rlevo_core::evaluation::{BenchEnv, BenchError, BenchStep};
use rlevo_core::render::AsciiRenderable;

use crate::reporter::tui::TuiHandle;

/// Transparent [`BenchEnv`] wrapper that emits a [`StyledFrame`] after each successful step.
///
/// Requires `E: BenchEnv + AsciiRenderable`. Any env that provides the
/// styled projection drops in without further plumbing.
///
/// # Examples
///
/// ```no_run
/// # use rlevo_benchmarks::env_wrappers::render_tap::RenderTap;
/// # use rlevo_benchmarks::reporter::tui::TuiHandle;
/// # fn make_env() -> impl rlevo_core::evaluation::BenchEnv + rlevo_core::render::AsciiRenderable { todo!() }
/// let (handle, _rx) = TuiHandle::channel();
/// let mut tap = RenderTap::new(make_env(), handle);
/// tap.reset().unwrap();
/// ```
#[derive(Debug)]
pub struct RenderTap<E> {
    inner: E,
    handle: TuiHandle,
    step: u32,
}

impl<E> RenderTap<E>
where
    E: BenchEnv + AsciiRenderable,
{
    /// Wrap `inner`, forwarding rendered frames through `handle`.
    pub const fn new(inner: E, handle: TuiHandle) -> Self {
        Self {
            inner,
            handle,
            step: 0,
        }
    }

    /// Borrow the wrapped env. Useful for tests that need to inspect the
    /// underlying state directly.
    pub const fn inner(&self) -> &E {
        &self.inner
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

    /// Render and push a frame. Pulled out of `reset`/`step` so the
    /// behaviour is identical on both paths and exercisable from tests.
    fn capture(&self) {
        let frame = self.inner.render_styled();
        let _ = self.handle.try_push_frame(self.step, frame);
    }
}

impl<E> BenchEnv for RenderTap<E>
where
    E: BenchEnv + AsciiRenderable,
{
    type Observation = E::Observation;
    type Action = E::Action;

    fn reset(&mut self) -> Result<Self::Observation, BenchError> {
        let obs = self.inner.reset()?;
        self.step = 0;
        self.capture();
        Ok(obs)
    }

    fn step(&mut self, action: Self::Action) -> Result<BenchStep<Self::Observation>, BenchError> {
        let result = self.inner.step(action)?;
        self.step = self.step.saturating_add(1);
        self.capture();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    //! Tests use a minimal stub env rather than pulling `rlevo-environments`
    //! in as a dev-dependency. The trade-off: importing the full env crate
    //! would compile ~600 tests (including the `box2d`/`locomotion`
    //! physics families) for one wrapper test. The wrapper's contract is
    //! pure delegation, so a stub gives equivalent coverage.

    use std::sync::mpsc::Receiver;

    use rlevo_core::evaluation::{BenchEnv, BenchError, BenchStep};
    use rlevo_core::render::{AsciiRenderable, StyledFrame};

    use super::RenderTap;
    use crate::reporter::tui::{TuiEvent, TuiHandle};

    /// Minimal `BenchEnv + AsciiRenderable` that lets us drive the wrapper.
    struct StubEnv {
        /// Steps observed so far; used to vary the rendered frame so test
        /// assertions can distinguish them.
        steps: u32,
        /// When true, the next `step` returns a `BenchError::Step` — used
        /// to verify the wrapper bails out without capturing on failure.
        fail_next_step: bool,
    }

    impl StubEnv {
        const fn new() -> Self {
            Self {
                steps: 0,
                fail_next_step: false,
            }
        }
    }

    impl BenchEnv for StubEnv {
        type Observation = u32;
        type Action = ();

        fn reset(&mut self) -> Result<Self::Observation, BenchError> {
            self.steps = 0;
            Ok(0)
        }

        fn step(&mut self, _action: ()) -> Result<BenchStep<Self::Observation>, BenchError> {
            if self.fail_next_step {
                use rlevo_core::environment::EnvironmentError;
                return Err(BenchError::Step(EnvironmentError::InvalidAction(
                    "stub forced failure".into(),
                )));
            }
            self.steps += 1;
            Ok(BenchStep {
                observation: self.steps,
                reward: 1.0,
                done: false,
            })
        }
    }

    impl AsciiRenderable for StubEnv {
        fn render_ascii(&self) -> String {
            format!("step={}", self.steps)
        }
    }

    /// Drain the channel collecting only `Frame` events.
    fn collect_frames(rx: &Receiver<TuiEvent>) -> Vec<(u32, StyledFrame)> {
        let mut out = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let TuiEvent::Frame { step, frame } = event {
                out.push((step, frame));
            }
        }
        out
    }

    #[test]
    fn reset_emits_initial_frame_at_step_zero() {
        let (handle, rx) = TuiHandle::channel();
        let mut tap = RenderTap::new(StubEnv::new(), handle);
        tap.reset().unwrap();

        let frames = collect_frames(&rx);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].0, 0);
        assert_eq!(frames[0].1.plain_text(), "step=0");
        assert_eq!(tap.step_count(), 0);
    }

    #[test]
    fn step_emits_frame_with_incremented_counter() {
        let (handle, rx) = TuiHandle::channel();
        let mut tap = RenderTap::new(StubEnv::new(), handle);
        tap.reset().unwrap();
        tap.step(()).unwrap();
        tap.step(()).unwrap();

        let frames = collect_frames(&rx);
        let steps: Vec<u32> = frames.iter().map(|(s, _)| *s).collect();
        assert_eq!(steps, vec![0, 1, 2]);
        assert_eq!(frames[2].1.plain_text(), "step=2");
        assert_eq!(tap.step_count(), 2);
    }

    /// Reset mid-episode rewinds the step counter so consecutive episodes
    /// each emit a step-0 initial frame.
    #[test]
    fn reset_after_steps_rewinds_counter() {
        let (handle, rx) = TuiHandle::channel();
        let mut tap = RenderTap::new(StubEnv::new(), handle);
        tap.reset().unwrap();
        tap.step(()).unwrap();
        tap.step(()).unwrap();
        tap.reset().unwrap();
        tap.step(()).unwrap();

        let steps: Vec<u32> = collect_frames(&rx).into_iter().map(|(s, _)| s).collect();
        assert_eq!(steps, vec![0, 1, 2, 0, 1]);
        assert_eq!(tap.step_count(), 1);
    }

    /// A failed `step` must not advance the counter or emit a frame.
    #[test]
    fn step_failure_skips_capture() {
        let (handle, rx) = TuiHandle::channel();
        let mut env = StubEnv::new();
        env.fail_next_step = true;
        let mut tap = RenderTap::new(env, handle);
        tap.reset().unwrap();
        let _ = tap.step(()).expect_err("step should fail");

        // Only the reset's frame should have landed.
        let frames = collect_frames(&rx);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].0, 0);
        assert_eq!(tap.step_count(), 0);
    }

    /// A dropped receiver makes `try_push_frame` return `false`, but the
    /// rollout loop must continue regardless.
    #[test]
    fn lossy_on_dropped_receiver() {
        let (handle, rx) = TuiHandle::channel();
        let mut tap = RenderTap::new(StubEnv::new(), handle);
        drop(rx);

        // None of these calls should panic or block.
        tap.reset().unwrap();
        tap.step(()).unwrap();
        tap.step(()).unwrap();
        assert_eq!(tap.step_count(), 2);
    }

    /// Consuming `into_inner` yields the original env with its state intact.
    #[test]
    fn into_inner_returns_wrapped_env() {
        let (handle, _rx) = TuiHandle::channel();
        let tap = RenderTap::new(StubEnv::new(), handle);
        let inner = tap.into_inner();
        assert_eq!(inner.render_ascii(), "step=0");
    }
}
