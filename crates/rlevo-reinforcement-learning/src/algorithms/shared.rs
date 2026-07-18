//! Shared infrastructure for the algorithm implementations in this module.
//!
//! Hosts [`Slot`], the network-ownership newtype every agent uses to hold a
//! trainable network across a Burn optimizer step, and [`LogWatermark`], the
//! progress-logging trigger shared by the on-policy training loops.
//!
//! # Why a `Slot` exists
//!
//! Burn's [`Optimizer::step`] consumes the module **by value** and returns the
//! updated module:
//!
//! ```text
//! fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M;
//! ```
//!
//! An agent that owns its network in a plain field therefore cannot call
//! `step` through `&mut self` — the module must first be moved out. The
//! historical idiom in this crate was to store the network as `Option<M>` and
//! `take()` it for the duration of an entire learn step:
//!
//! ```text
//! let net = self.net.take().expect("...");   // field is now None
//! let loss = net.forward(batch);             // ← any panic here
//! let grads = loss.backward();               // ← or here
//! let grads = GradientsParams::from_grads(grads, &net);
//! self.net = Some(self.optimizer.step(lr, net, grads));
//! ```
//!
//! Every line between the `take()` and the write-back is a window in which a
//! panic leaves the field permanently `None`, bricking the agent: every later
//! `act` / `learn_step` hits the `expect` and panics again, with a message
//! pointing at the *wrong* method. `Slot` closes that window by construction —
//! the module is only ever out of the field for the duration of the `step`
//! call itself, inside [`Slot::step_with`], which nothing else can widen.
//!
//! # The residual window is irreducible — do not try to close it
//!
//! A panic *inside* [`Optimizer::step`] still poisons the slot, and this is
//! accepted by design. It cannot be fixed, only paid for:
//!
//! - **An RAII drop guard cannot help.** A guard restores a value it still
//!   holds. Once `module` has been moved into `step`, this code no longer owns
//!   it — there is nothing left to restore. The value is in `step`'s frame and
//!   is dropped during unwinding.
//! - **`catch_unwind` cannot help either.** Catching the unwind tells us the
//!   step failed; it does not hand the moved-in module back. `step` returns `M`
//!   only on the success path.
//! - **The only real fix is a clone**, i.e. keeping a spare copy of the weights
//!   to restore from. That is a full copy of every parameter tensor on *every*
//!   step of the hot training loop, to defend against a panic that only occurs
//!   on an already-fatal bug (shape mismatch, device OOM, NaN assertion). The
//!   cost is rejected.
//!
//! So the contract is: do all fallible work — `forward`, loss, `backward`,
//! [`GradientsParams::from_grads`] — on a borrow from [`Slot::get`] *before*
//! calling [`Slot::step_with`], and accept that a panic in `step` itself is
//! terminal for that agent.
//!
//! [`Optimizer::step`]: burn::optim::Optimizer::step

use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, LearningRate, Optimizer};
use burn::tensor::backend::AutodiffBackend;

/// Panic message for a read against a poisoned slot.
///
/// Deliberately names the *cause* (a panic inside the optimizer step, the one
/// irreducible window — see the module docs) and the *remedy* (rebuild the
/// agent), rather than naming a method. Agents disagree on what their learn
/// step is called (`learn_step` for the DQN family, `update` for PPO/PPG), so a
/// method name here would be wrong for half the crate.
const POISONED: &str = "network slot is empty: the agent was poisoned by a panic inside an earlier \
                        optimizer step and cannot recover (the module was moved into `step` and \
                        lost during unwinding); rebuild the agent from a fresh network";

/// Owns a trainable network across a Burn optimizer step.
///
/// `Slot` is a newtype over `Option<M>` whose entire purpose is to make the
/// wide "module is temporarily nowhere" window **unrepresentable**. The module
/// is `None` only for the duration of the [`Optimizer::step`] call inside
/// [`step_with`](Self::step_with); at every point a caller can observe, it is
/// `Some`.
///
/// The type is never empty at birth — there is no `Default` and no dangling
/// constructor. A `Slot` is [`new`](Self::new)'d from a module and stays
/// populated unless [`step_with`](Self::step_with) panics.
///
/// # Usage
///
/// Do every fallible operation on a borrow from [`get`](Self::get) — the
/// forward pass, the loss, `backward`, and [`GradientsParams::from_grads`]
/// (which takes `&M` and carries no lifetime, so NLL ends the borrow at its
/// last use). Only then call [`step_with`](Self::step_with):
///
/// ```ignore
/// let loss = self.net.get().forward(batch);
/// let grads = loss.backward();
/// let grads = GradientsParams::from_grads(grads, self.net.get());
/// self.net.step_with(&mut self.optimizer, lr, grads);
/// // Post-step reads (e.g. refreshing a target net) go through `get` too:
/// self.target = self.net.get().valid();
/// ```
///
/// # There is deliberately no closure-taking variant
///
/// An `update_with(f: impl FnOnce(M) -> M)` would be the "flexible" API, and it
/// would reintroduce exactly the bug this type exists to prevent: a caller
/// could put the forward pass, the loss, and `backward` back *inside* the
/// closure — i.e. back inside the window where the module is out of the slot —
/// and every panic in that region would poison the agent again. The absence of
/// the closure is the design. Keep [`step_with`](Self::step_with) closure-free.
///
/// # Invariants
///
/// - `Slot` is `Some` immediately after [`new`](Self::new).
/// - `Slot` is `Some` after a [`step_with`](Self::step_with) that returns
///   normally.
/// - `Slot` is `None` (poisoned, permanently) only if
///   [`step_with`](Self::step_with) unwound. See the module docs for why this
///   case is irreducible.
pub(crate) struct Slot<M>(Option<M>);

// A network's `Debug` would dump every parameter tensor, so report the slot's
// state and the module's type instead. This also keeps `Slot<M>: Debug` free of
// an `M: Debug` bound, mirroring the hand-written `DqnAgent` impl's summary
// style.
impl<M> std::fmt::Debug for Slot<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slot")
            .field("module", &std::any::type_name::<M>())
            .field("poisoned", &self.is_poisoned())
            .finish()
    }
}

impl<M> Slot<M> {
    /// Wraps `module`, producing a populated (never poisoned) slot.
    ///
    /// # Arguments
    ///
    /// - `module` — the network this slot takes ownership of.
    ///
    /// # Returns
    ///
    /// A `Slot` for which [`is_poisoned`](Self::is_poisoned) is `false`.
    pub(crate) fn new(module: M) -> Self {
        Self(Some(module))
    }

    /// Borrows the network.
    ///
    /// This is the single read path for a `Slot`, and the only read-side
    /// `expect` in the crate. Use it for the forward pass, for
    /// [`GradientsParams::from_grads`], and for post-step reads such as
    /// `slot.get().valid()` — an owned module is never needed on the read side,
    /// because every consumer of the trained network (`AutodiffModule::valid`,
    /// `forward`, `soft_update`) takes `&self`.
    ///
    /// # Returns
    ///
    /// A shared reference to the owned network.
    ///
    /// # Panics
    ///
    /// Panics if the slot is poisoned — i.e. a previous
    /// [`step_with`](Self::step_with) unwound and the module was lost inside
    /// [`Optimizer::step`]. The agent cannot be recovered and must be rebuilt;
    /// see the module docs for why. Check [`is_poisoned`](Self::is_poisoned)
    /// first if a caller needs to handle this without unwinding.
    pub(crate) fn get(&self) -> &M {
        self.0.as_ref().expect(POISONED)
    }

    /// Reports whether the network was lost to a panic inside a previous
    /// optimizer step.
    ///
    /// # Returns
    ///
    /// `true` if the slot is empty, in which case [`get`](Self::get) and
    /// [`step_with`](Self::step_with) will panic and the owning agent must be
    /// rebuilt. `false` in every other case.
    pub(crate) fn is_poisoned(&self) -> bool {
        self.0.is_none()
    }

    /// Applies one optimizer step to the owned network, in place.
    ///
    /// Moves the module out, hands it to [`Optimizer::step`] (which consumes
    /// it), and writes the returned module back. This method is intentionally
    /// closure-free: the module is out of the slot for this call and nothing
    /// else, so no caller can widen the poison window — see the type-level docs.
    ///
    /// All fallible preparation (`forward`, `backward`,
    /// [`GradientsParams::from_grads`]) must already have happened against
    /// [`get`](Self::get) before this is called.
    ///
    /// # Arguments
    ///
    /// - `opt` — the optimizer paired with this network.
    /// - `lr` — learning rate for this step.
    /// - `grads` — gradients, already reduced to per-parameter form by
    ///   [`GradientsParams::from_grads`].
    ///
    /// # Panics
    ///
    /// Panics if the slot is already poisoned (see [`get`](Self::get)).
    ///
    /// Additionally, if [`Optimizer::step`] itself panics, the slot is left
    /// poisoned: the module was already moved into `step` and is dropped during
    /// unwinding, so there is nothing left to restore. This residual window is
    /// accepted by design and is not fixable with a drop guard or
    /// `catch_unwind` — see the module docs before proposing either.
    pub(crate) fn step_with<B, O>(&mut self, opt: &mut O, lr: LearningRate, grads: GradientsParams)
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
        O: Optimizer<M, B>,
    {
        let module = self.0.take().expect(POISONED);
        self.0 = Some(opt.step(lr, module, grads));
    }
}

/// Periodic progress-logging trigger for a loop that advances in strides.
///
/// Decides whether a training loop should emit its periodic `tracing` progress
/// event at the current `global_step`, using a **last-logged watermark** rather
/// than a divisibility test.
///
/// # Why a watermark and not `global_step % log_every == 0`
///
/// The on-policy loops (PPO, PPG) can only log at a **rollout boundary**: the
/// payload reads the update statistics, which do not exist until after
/// `update` / `policy_phase_update` has run at the end of an iteration. So the
/// step counter is never observed at every value — it is only ever observed at
/// multiples of `num_steps` (128 by default).
///
/// A divisibility test against a counter sampled on a stride fires on the
/// *intersection* of the two schedules, i.e. every `lcm(num_steps, log_every)`
/// steps — not every `log_every` steps. With `num_steps = 128` that silently
/// turned `log_every = 100` into an effective cadence of 3200 (32× too sparse)
/// and `log_every = 500` into 16 000 — so a 10 000-step run emitted **zero**
/// progress lines despite asking for 20 (issue #321).
///
/// The watermark form has no such coupling: it fires as soon as at least
/// `log_every` steps have elapsed since the previous log, whatever the stride,
/// and then re-anchors on the step it actually fired at. The cadence degrades
/// gracefully to "once per rollout" when `log_every` is smaller than the
/// stride, which is the best any boundary-gated logger can do.
///
/// # Contract
///
/// - `every == 0` disables logging entirely — [`should_log`](Self::should_log)
///   always returns `false`. This is documented public API on the `log_every`
///   parameter of the training entry points (issue #174).
/// - The watermark is monotonic: a given `global_step` fires at most once, so
///   re-querying the same step (or a step that went backwards) cannot produce a
///   duplicate log line.
#[derive(Debug)]
pub(crate) struct LogWatermark {
    /// Requested cadence in global steps; `0` disables logging.
    every: usize,
    /// Global step at which the last log fired (`0` before the first).
    last: usize,
}

impl LogWatermark {
    /// Creates a watermark for a cadence of `every` global steps.
    ///
    /// Pass `0` to disable logging; see the type-level contract.
    pub(crate) const fn new(every: usize) -> Self {
        Self { every, last: 0 }
    }

    /// Returns `true` if a progress line is due at `global_step`, and if so
    /// re-anchors the watermark on that step.
    ///
    /// Fires when at least `every` steps have elapsed since the previous log
    /// (or since the start of the run). Uses a saturating difference so a
    /// non-monotonic `global_step` can only under-fire, never panic.
    pub(crate) fn should_log(&mut self, global_step: usize) -> bool {
        if self.every == 0 {
            return false;
        }
        if global_step.saturating_sub(self.last) >= self.every {
            self.last = global_step;
            true
        } else {
            false
        }
    }

    /// Returns `true` if the run's terminal step still owes a progress line,
    /// and if so re-anchors the watermark so it cannot fire again.
    ///
    /// The final rollout is usually **partial** — the collection loop breaks as
    /// soon as `global_step >= total_timesteps`, so the last boundary lands on
    /// an arbitrary step rather than a multiple of the stride. That boundary
    /// can therefore sit less than `every` past the watermark and never trigger
    /// [`should_log`](Self::should_log), silently dropping the final update's
    /// statistics — the most interesting line of the run.
    ///
    /// Concretely, with `num_steps = 128` and `log_every = 500`, a
    /// 10 000-step run logs at 9728 and then finishes at 10 000 with nothing
    /// reported. Call this once after the loop to close that gap.
    ///
    /// Returns `false` when logging is disabled (`every == 0`) and when
    /// `global_step` was already logged by the in-loop trigger, so the terminal
    /// line is never a duplicate.
    pub(crate) fn should_log_final(&mut self, global_step: usize) -> bool {
        if self.every == 0 || global_step == self.last {
            return false;
        }
        self.last = global_step;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::AssertUnwindSafe;

    use burn::backend::{Autodiff, Flex};
    use burn::module::Module;
    use burn::nn::{Linear, LinearConfig};
    use burn::optim::adaptor::OptimizerAdaptor;
    use burn::optim::{Adam, AdamConfig};
    use burn::tensor::backend::Backend;
    use burn::tensor::{Device, Tensor, TensorData};

    type B = Autodiff<Flex>;

    /// Minimal trainable network for slot tests.
    ///
    /// The backend generic **must** be named `B`: Burn's `Module` derive keys
    /// off the identifier, and any other name makes the derive silently treat
    /// the struct as a parameter-free module — the optimizer would then step
    /// nothing and the "weights changed" assertions below would pass
    /// vacuously.
    #[derive(Module, Debug)]
    struct TestNet<B: Backend> {
        linear: Linear<B>,
    }

    impl<B: Backend> TestNet<B> {
        fn init(device: &B::Device) -> Self {
            Self {
                linear: LinearConfig::new(2, 2).init(device),
            }
        }

        fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
            self.linear.forward(x)
        }
    }

    fn test_optimizer() -> OptimizerAdaptor<Adam, TestNet<B>, B> {
        AdamConfig::new().init::<B, TestNet<B>>()
    }

    fn test_batch(device: &Device<B>) -> Tensor<B, 2> {
        Tensor::from_data(TensorData::new(vec![0.5_f32, -0.25], vec![1, 2]), device)
    }

    /// Runs one full learn step against `slot`: forward / backward on a borrow,
    /// then the closure-free optimizer step.
    fn learn_step(slot: &mut Slot<TestNet<B>>, opt: &mut OptimizerAdaptor<Adam, TestNet<B>, B>) {
        let device = Device::<B>::default();
        let loss = slot.get().forward(test_batch(&device)).sum();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, slot.get());
        slot.step_with(opt, 1e-2, grads);
    }

    /// Reads the linear layer's weights back to the host, element-wise.
    ///
    /// Deliberately not a `sum()` proxy: Adam's first step moves each weight by
    /// roughly `±lr` in the direction of its gradient's sign, and for this
    /// network the per-element steps very nearly cancel in the sum — a summed
    /// proxy reports a ~5e-7 delta for an update that actually moved every
    /// weight by ~1e-2, and would make "did the optimizer run" untestable.
    fn weights(slot: &Slot<TestNet<B>>) -> Vec<f32> {
        slot.get()
            .linear
            .weight
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("weight tensor host read")
    }

    /// Largest element-wise absolute difference between two weight snapshots.
    fn max_abs_diff(before: &[f32], after: &[f32]) -> f32 {
        assert_eq!(
            before.len(),
            after.len(),
            "weight snapshots must have equal length"
        );
        before
            .iter()
            .zip(after)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn test_slot_new_populates_and_get_returns_module() {
        let device = Device::<B>::default();
        let slot = Slot::new(TestNet::<B>::init(&device));

        assert!(
            !slot.is_poisoned(),
            "a freshly constructed Slot must hold its module"
        );
        let out = slot.get().forward(test_batch(&device));
        assert_eq!(
            out.dims(),
            [1, 2],
            "get() must return a usable module: forward through the borrow yields [batch, out]"
        );
    }

    #[test]
    fn test_slot_get_supports_repeated_borrows() {
        let device = Device::<B>::default();
        let slot = Slot::new(TestNet::<B>::init(&device));

        let first = weights(&slot);
        let second = weights(&slot);
        assert!(
            max_abs_diff(&first, &second) < 1e-6,
            "get() is a pure borrow: repeated reads must observe identical weights"
        );
    }

    #[test]
    fn test_slot_get_supports_valid_snapshot_post_step() {
        // Agents refresh their target network with `self.net.get().valid()`
        // after a step; `valid()` takes `&self`, so the borrow suffices and no
        // owned module is ever needed on the read side.
        let device = Device::<B>::default();
        let mut slot = Slot::new(TestNet::<B>::init(&device));
        let mut opt = test_optimizer();

        learn_step(&mut slot, &mut opt);

        let inner = slot.get().valid();
        let out = inner.forward(Tensor::from_data(
            TensorData::new(vec![0.5_f32, -0.25], vec![1, 2]),
            &device,
        ));
        assert_eq!(
            out.dims(),
            [1, 2],
            "valid() snapshot taken through get() after a step must be usable for inference"
        );
    }

    #[test]
    fn test_slot_step_with_leaves_slot_populated() {
        let device = Device::<B>::default();
        let mut slot = Slot::new(TestNet::<B>::init(&device));
        let mut opt = test_optimizer();

        learn_step(&mut slot, &mut opt);

        assert!(
            !slot.is_poisoned(),
            "a successful step_with must write the updated module back into the slot"
        );
        // `get()` must not panic — this is the assertion that a naive
        // take()-without-restore would fail.
        let _ = slot.get();
    }

    #[test]
    fn test_slot_is_poisoned_false_after_successful_step() {
        let device = Device::<B>::default();
        let mut slot = Slot::new(TestNet::<B>::init(&device));
        let mut opt = test_optimizer();

        for _ in 0..3 {
            learn_step(&mut slot, &mut opt);
            assert!(
                !slot.is_poisoned(),
                "is_poisoned must stay false across repeated successful steps"
            );
        }
    }

    #[test]
    fn test_slot_step_with_updates_weights() {
        let device = Device::<B>::default();
        let mut slot = Slot::new(TestNet::<B>::init(&device));
        let mut opt = test_optimizer();

        let before = weights(&slot);
        learn_step(&mut slot, &mut opt);
        let after = weights(&slot);

        // Guards against the `Module` derive silently treating `TestNet` as a
        // parameter-free module, which would make every other assertion here
        // vacuous. Adam's first step is ~lr per weight, so 1e-4 is a wide
        // margin under the 1e-2 learning rate `learn_step` uses.
        let delta = max_abs_diff(&before, &after);
        assert!(
            delta > 1e-4,
            "step_with must actually apply the optimizer update: \
             max weight delta was {delta}"
        );
    }

    #[test]
    fn test_slot_panic_in_borrow_region_leaves_slot_intact() {
        // This is the bug #167 fixes. Under the old `Option<M>` + `take()`
        // idiom the module was out of the field for the whole forward/backward
        // region, so a panic there — a shape mismatch, a device error, a
        // failed host read — left the field `None` forever and every
        // subsequent call panicked with a misleading message. With `Slot` the
        // module never leaves the field during that region, so the agent
        // survives a caught unwind unharmed.
        let device = Device::<B>::default();
        let mut slot = Slot::new(TestNet::<B>::init(&device));
        let mut opt = test_optimizer();

        let before = weights(&slot);

        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let loss = slot.get().forward(test_batch(&device)).sum();
            let _grads = loss.backward();
            panic!("simulated failure in the forward/backward region");
        }));

        assert!(result.is_err(), "the simulated failure must have unwound");
        assert!(
            !slot.is_poisoned(),
            "a panic before step_with must not poison the slot — the module never left it"
        );
        let after = weights(&slot);
        assert!(
            max_abs_diff(&before, &after) < 1e-6,
            "the surviving module must be the original, unmodified network"
        );

        // The agent is fully usable afterwards: a real learn step still works.
        learn_step(&mut slot, &mut opt);
        assert!(
            !slot.is_poisoned(),
            "the slot must remain usable for training after a caught panic in the borrow region"
        );
    }

    #[test]
    fn test_slot_poisoned_read_panics_with_actionable_message() {
        // A poisoned slot can only arise from a panic *inside*
        // `Optimizer::step` — which is irreducible by design (the module is
        // moved into `step` and dropped during unwinding; see the module
        // docs), and cannot be provoked here without a deliberately broken
        // optimizer. Construct the state directly to pin the read behaviour:
        // `is_poisoned` reports it, and `get` panics with a message that names
        // the cause and the remedy rather than a per-agent method name.
        let slot: Slot<TestNet<B>> = Slot(None);

        assert!(
            slot.is_poisoned(),
            "an empty slot must report itself as poisoned"
        );

        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = slot.get();
        }));
        let payload = result.expect_err("get() on a poisoned slot must panic");
        let message = payload
            .downcast_ref::<String>()
            .map_or_else(String::new, Clone::clone);
        assert!(
            message.contains("optimizer step"),
            "the panic message must name the cause (a panic inside the optimizer step), got: \
             {message}"
        );
        assert!(
            message.contains("rebuild the agent"),
            "the panic message must name the remedy (rebuild the agent), got: {message}"
        );
    }

    #[test]
    fn test_slot_debug_reports_state_without_dumping_weights() {
        let device = Device::<B>::default();
        let slot = Slot::new(TestNet::<B>::init(&device));

        let rendered = format!("{slot:?}");
        assert!(
            rendered.contains("poisoned: false"),
            "Debug must surface the slot's poison state, got: {rendered}"
        );
        assert!(
            rendered.contains("TestNet"),
            "Debug must name the module type, got: {rendered}"
        );
    }

    // -------- LogWatermark --------

    /// Global steps a PPO/PPG loop actually observes: one sample per rollout
    /// boundary, i.e. multiples of `stride`, up to and including `total`.
    fn boundaries(stride: usize, total: usize) -> impl Iterator<Item = usize> {
        (stride..=total).step_by(stride)
    }

    /// Global steps at which `watermark` fires over a strided run.
    fn fire_steps(log_every: usize, stride: usize, total: usize) -> Vec<usize> {
        let mut wm = LogWatermark::new(log_every);
        boundaries(stride, total)
            .filter(|&step| wm.should_log(step))
            .collect()
    }

    #[test]
    fn test_log_watermark_zero_never_fires() {
        let fired = fire_steps(0, 128, 10_000);
        assert!(
            fired.is_empty(),
            "log_every == 0 must disable logging entirely, but fired at {fired:?}"
        );
    }

    #[test]
    fn test_log_watermark_fires_when_every_is_coprime_with_stride() {
        // Regression for #321: with the old `global_step % log_every == 0`
        // test this fired only at lcm(128, 100) = 3200, 6400, ... — 32x too
        // sparse. The watermark must fire at the first boundary at or past
        // each 100-step mark.
        let fired = fire_steps(100, 128, 1000);
        assert_eq!(
            fired,
            vec![128, 256, 384, 512, 640, 768, 896],
            "log_every = 100 on a 128 stride must fire once per rollout boundary"
        );
    }

    #[test]
    fn test_log_watermark_fires_within_ten_thousand_steps() {
        // Regression for #321: log_every = 500 with a 128 stride produced ZERO
        // lines in a 10k run under divisibility (first hit at lcm = 16 000).
        let fired = fire_steps(500, 128, 10_000);
        assert_eq!(
            fired.first().copied(),
            Some(512),
            "the first log must land at the first boundary at or past step 500"
        );
        // 500 rounds up to the next boundary, so the realised spacing is 512
        // and a 10 000-step run yields floor(10_000 / 512) = 19 lines — the
        // requested ~20, versus zero before the fix.
        assert_eq!(
            fired.len(),
            19,
            "a 10k-step run at log_every = 500 must emit ~20 lines, got {fired:?}"
        );
    }

    #[test]
    fn test_log_watermark_smaller_than_stride_fires_once_per_boundary() {
        // `log_every` below the rollout length cannot log more often than the
        // loop reaches a boundary; it must not fire twice for one boundary.
        let fired = fire_steps(10, 128, 640);
        assert_eq!(
            fired,
            vec![128, 256, 384, 512, 640],
            "a sub-stride cadence must saturate at one line per rollout boundary"
        );
    }

    #[test]
    fn test_log_watermark_divisor_of_stride_preserves_old_cadence() {
        // `log_every` dividing the stride is the one case the old divisibility
        // test got right; the watermark must not change it.
        let fired = fire_steps(64, 128, 640);
        let old_behaviour: Vec<usize> = boundaries(128, 640)
            .filter(|step| step.is_multiple_of(64))
            .collect();
        assert_eq!(
            fired, old_behaviour,
            "an exact divisor of the stride must keep its pre-fix cadence"
        );
    }

    /// Boundaries a loop actually reaches, including the **partial** final
    /// rollout: collection breaks at `global_step >= total`, so the run always
    /// ends exactly on `total`.
    fn boundaries_with_partial_tail(stride: usize, total: usize) -> Vec<usize> {
        let mut steps: Vec<usize> = boundaries(stride, total).collect();
        if steps.last() != Some(&total) {
            steps.push(total);
        }
        steps
    }

    /// Replays a whole run — in-loop triggers plus the one terminal check —
    /// and returns every step that emitted a line.
    fn fire_steps_full_run(log_every: usize, stride: usize, total: usize) -> Vec<usize> {
        let mut wm = LogWatermark::new(log_every);
        let steps = boundaries_with_partial_tail(stride, total);
        let mut fired: Vec<usize> = steps
            .iter()
            .copied()
            .filter(|&step| wm.should_log(step))
            .collect();
        if wm.should_log_final(total) {
            fired.push(total);
        }
        fired
    }

    #[test]
    fn test_log_watermark_final_zero_never_fires() {
        let mut wm = LogWatermark::new(0);
        assert!(
            !wm.should_log_final(10_000),
            "log_every == 0 must disable the terminal line too"
        );
    }

    #[test]
    fn test_log_watermark_final_skips_already_logged_terminal_step() {
        let mut wm = LogWatermark::new(100);
        assert!(wm.should_log(128), "the in-loop trigger must fire first");
        assert!(
            !wm.should_log_final(128),
            "a terminal step the loop already logged must not be logged twice"
        );
    }

    #[test]
    fn test_log_watermark_final_recovers_dropped_terminal_line() {
        // Both regressions from the follow-up review: the partial final
        // rollout lands short of `every` past the watermark, so `should_log`
        // declines and the run's last update stats would vanish.
        let mut wm = LogWatermark::new(500);
        for step in boundaries_with_partial_tail(128, 10_000) {
            wm.should_log(step);
        }
        assert!(
            wm.should_log_final(10_000),
            "tt=10000 / le=500 must still report the terminal step (last in-loop line was 9728)"
        );

        let mut wm = LogWatermark::new(100);
        for step in boundaries_with_partial_tail(128, 300) {
            wm.should_log(step);
        }
        assert!(
            wm.should_log_final(300),
            "tt=300 / le=100 must still report the terminal step (last in-loop line was 256)"
        );
    }

    #[test]
    fn test_log_watermark_full_run_adds_at_most_one_final_line() {
        // A full run emits floor(total / realised_spacing) in-loop lines plus
        // at most one terminal line — never two for the same step.
        for &(log_every, stride, total) in &[
            (500_usize, 128_usize, 10_000_usize),
            (100, 128, 300),
            (128, 128, 1024), // `every` == stride: terminal step already logged
            (64, 128, 640),   // exact divisor of the stride
            (10, 128, 1000),  // sub-stride cadence
        ] {
            let fired = fire_steps_full_run(log_every, stride, total);
            let mut deduped = fired.clone();
            deduped.dedup();
            assert_eq!(
                fired, deduped,
                "no step may be logged twice (le={log_every}, stride={stride}, tt={total})"
            );
            assert_eq!(
                fired.last().copied(),
                Some(total),
                "every run must report its terminal step \
                 (le={log_every}, stride={stride}, tt={total}), got {fired:?}"
            );

            let in_loop_only = fire_steps(log_every, stride, total).len();
            assert!(
                fired.len() <= in_loop_only + 1,
                "the terminal line may add at most one entry \
                 (le={log_every}, stride={stride}, tt={total}), got {fired:?}"
            );
        }
    }

    #[test]
    fn test_log_watermark_never_fires_twice_for_same_step() {
        // `every` deliberately *divides* the boundary so this test isolates the
        // dedup/monotonicity property. With a non-dividing `every` the opening
        // `should_log` would already be exercising the #321 cadence bug, and a
        // failure here would not tell us which property broke.
        let mut wm = LogWatermark::new(64);
        assert!(wm.should_log(128), "the first qualifying step must fire");
        assert!(
            !wm.should_log(128),
            "re-querying the same global step must not emit a duplicate line"
        );
        assert!(
            !wm.should_log(64),
            "a step behind the watermark must not fire"
        );
        assert!(
            wm.should_log(192),
            "the watermark must re-arm once `every` steps have elapsed"
        );
    }
}
