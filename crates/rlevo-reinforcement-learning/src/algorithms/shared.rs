//! Shared infrastructure for the algorithm implementations in this module.
//!
//! Hosts [`Slot`], the network-ownership newtype every agent uses to hold a
//! trainable network across a Burn optimizer step; [`FiniteLossGuard`], the
//! non-finite-loss skip-and-warn guard every learn step consults before
//! `backward()` (ADR 0056, issue #318); and [`LogWatermark`], the
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
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

use rlevo_core::action::BoundedAction;

use crate::replay::{ImportanceExponent, SampledBatch};

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

/// The importance-sampling exponent (β) the off-policy agents pass to
/// [`ReplayStrategy::sample`].
///
/// Every agent draws from a [`UniformReplay`], which emits no IS weights and
/// therefore ignores β entirely (ADR 0050 §3). The value is
/// [`ImportanceExponent::ONE`] — the fully-annealed, no-correction end of
/// Schaul's schedule — so that it stays correct as a fallback rather than merely
/// inert. When prioritized replay is wired in (ADR 0050 step 4) the agents that
/// adopt it replace this constant with `beta(self.step)` off their own config
/// schedule; the ones that keep uniform replay keep this.
///
/// The type is [`ImportanceExponent`], not `f32`: β is `finite && [0, 1]` by
/// construction so that a bad value cannot reach `powf` and poison a batch's
/// importance weights (ADR 0051 §3).
///
/// [`ReplayStrategy::sample`]: crate::replay::ReplayStrategy::sample
/// [`UniformReplay`]: crate::replay::UniformReplay
/// [`ImportanceExponent`]: crate::replay::ImportanceExponent
/// [`ImportanceExponent::ONE`]: crate::replay::ImportanceExponent::ONE
pub(crate) const UNIFORM_REPLAY_BETA: ImportanceExponent = ImportanceExponent::ONE;

/// Asserts that `A`'s action bounds satisfy the whole [`BoundedAction`]
/// contract: `low().len() == high().len() == COMPONENTS`, and
/// `low()[i] < high()[i]` for every component.
///
/// [`BoundedAction::low`] and [`BoundedAction::high`] return `&'static [f32]`,
/// which cannot encode its own length, so neither half of that contract is a
/// compile-time guarantee — both are obligations on the implementor
/// (ADR 0053 §4). The three continuous-control agents index those slices per
/// component on every `act`, so a short slice would surface as an
/// out-of-bounds panic mid-episode, pointing at the agent rather than at the
/// offending impl.
///
/// The **ordering** half fails more quietly still. Both DDPG's and SAC's
/// warm-up samplers build `self.low[i]..=self.high[i]` and hand it to
/// [`Rng::random_range`](rand::Rng::random_range), which panics on an empty
/// range — but only for `low > high`; an *equal* pair yields a degenerate
/// action space that samples one value forever and never reports anything.
/// The per-component `clamp` on the policy path is worse: `f32::clamp` panics
/// on `min > max`, again mid-episode and again naming the agent. Checking here
/// means a nonconforming impl is rejected at agent-build time, by a message
/// that names the component.
///
/// # Panics
///
/// Panics if `A::low().len()` or `A::high().len()` differs from
/// `A::COMPONENTS`, or if `A::low()[i] >= A::high()[i]` for any component `i`.
pub(crate) fn assert_bounds_match_components<const DA: usize, A: BoundedAction<DA>>() {
    assert_eq!(
        A::low().len(),
        A::COMPONENTS,
        "BoundedAction contract violated: low() has {} elements but COMPONENTS is {}",
        A::low().len(),
        A::COMPONENTS,
    );
    assert_eq!(
        A::high().len(),
        A::COMPONENTS,
        "BoundedAction contract violated: high() has {} elements but COMPONENTS is {}",
        A::high().len(),
        A::COMPONENTS,
    );
    for (i, (low, high)) in A::low().iter().zip(A::high()).enumerate() {
        assert!(
            low < high,
            "BoundedAction contract violated: component {i} has low() = {low} but \
             high() = {high}; every component needs low < high",
        );
    }
}

/// Builds the `[1, ..action_shape]` per-component lower/upper bound tensors used
/// by the DDPG and TD3 target-action clip.
///
/// TD3 Eq. 14 (Fujimoto et al. 2018) clips the target action against the
/// `Box(low, high)` **vectors**, one limit per component. Burn 0.21's `clamp`
/// family takes a scalar and cannot express that, so the clip goes through
/// `max_pair`/`min_pair`, which need a tensor operand broadcastable to the
/// `[batch, ..action_shape]` action tensor. Leading dim `1` broadcasts across
/// the batch.
///
/// Built once per agent, at construction: the bounds are `&'static` (ADR 0053
/// §2) and the device is already in hand, so nothing here belongs on the
/// per-`learn_step` path.
///
/// # Panics
///
/// Panics if `A::shape()` does not describe exactly `A::COMPONENTS` scalars.
/// The learn path already assumes this — it lays out the batched action tensor
/// as `[batch, ..A::shape()]` and fills it from `A::as_slice()`, which is
/// `COMPONENTS` long — so an action using ADR 0053 §8's Convention B (rank > 1
/// with a non-matching `shape()` product) is unusable with these agents, and
/// says so here rather than as a `TensorData` shape error.
pub(crate) fn action_bound_tensors<BI, A, const DA: usize, const DAB: usize>(
    device: &BI::Device,
) -> (Tensor<BI, DAB>, Tensor<BI, DAB>)
where
    BI: Backend,
    A: BoundedAction<DA>,
{
    let action_shape = A::shape();
    let numel: usize = action_shape.iter().product();
    assert_eq!(
        numel,
        A::COMPONENTS,
        "action shape {action_shape:?} describes {numel} scalars but COMPONENTS is {}; \
         the continuous-control agents require shape().product() == COMPONENTS",
        A::COMPONENTS,
    );

    let mut shape: Vec<usize> = Vec::with_capacity(DAB);
    shape.push(1);
    shape.extend_from_slice(&action_shape);

    let low = Tensor::from_data(TensorData::new(A::low().to_vec(), shape.clone()), device);
    let high = Tensor::from_data(TensorData::new(A::high().to_vec(), shape), device);
    (low, high)
}

/// Clips a batch of actions to `[low, high]` **per component**.
///
/// This is TD3 Eq. 14's outer clip, shared by DDPG's target path (which is the
/// same clip without the smoothing noise). `low` and `high` come from
/// [`action_bound_tensors`] and are shaped `[1, ..action_shape]`, so they
/// broadcast across the batch dimension of `actions`.
///
/// Burn 0.21's `clamp` / `clamp_min` / `clamp_max` take a scalar
/// `E: ElementConversion` and cannot express one limit per component, so this
/// is `max_pair` followed by `min_pair` — two broadcast elementwise ops rather
/// than one fused scalar clamp, over the same `[batch, ..action_shape]` data.
pub(crate) fn clip_to_action_bounds<BI, const DAB: usize>(
    actions: Tensor<BI, DAB>,
    low: Tensor<BI, DAB>,
    high: Tensor<BI, DAB>,
) -> Tensor<BI, DAB>
where
    BI: Backend,
{
    actions.max_pair(low).min_pair(high)
}

/// Reduces a per-sample `[batch]` loss to a scalar, first scaling each sample by
/// its importance-sampling weight when the batch carries any.
///
/// This is where Schaul Algorithm 1 line 13's `w_j · δ_j` enters an autodiff
/// framework: multiplying the per-sample **loss** by `w_j` before reduction
/// scales that sample's gradient contribution by `w_j`, exactly as the paper
/// scales the per-sample update. The weight touches only the loss — never the
/// TD-target computation and never `δ` itself (ADR 0050 §10, §14).
///
/// - **Uniform replay** emits [`SampledBatch::weights`] of `None`, so this is a
///   plain `.mean()` — **bit-identical** to the pre-PER reduction, which is what
///   keeps the uniform path a behavioural no-op.
/// - **Prioritized replay** emits max-normalized weights in `(0, 1]`; a weight of
///   `0` (unreachable under Schaul's construction, but well-defined here) zeroes
///   that sample's contribution to both loss and gradient.
///
/// The largest weight in a prioritized batch is exactly `1.0`, so a batch drawn
/// with `β = 0` (all weights `1.0`) also reduces to a plain mean.
pub(crate) fn reduce_weighted_loss<B: Backend>(
    per_sample: Tensor<B, 1>,
    batch: &SampledBatch,
    device: &B::Device,
) -> Tensor<B, 1> {
    match batch.weights() {
        None => per_sample.mean(),
        Some(weights) => {
            let n = weights.len();
            let w: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(weights.to_vec(), vec![n]), device);
            (per_sample * w).mean()
        }
    }
}

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

/// Non-finite-loss skip-and-warn guard for one loss site of one agent
/// (ADR 0056, issue #318).
///
/// Burn does not panic on `NaN` — it propagates it silently, folding a poisoned
/// loss into the weights on the next optimizer step while training keeps
/// reporting finite-looking metrics. Every agent already reads its loss scalar
/// host-side (`into_scalar().elem::<f32>()`) *before* `loss.backward()` for its
/// metrics, so a finiteness check on that `f32` costs no extra device→host
/// sync. [`check`](Self::check) rides that existing read: on a non-finite value
/// it tells the caller to skip both `backward()` and the optimizer step, so the
/// poison never reaches the parameters and the next minibatch can recover.
///
/// This generalizes the SAC-α optimizer guard (#184,
/// [`sac_alpha`](super::sac::sac_alpha)) from one hand-rolled optimizer to
/// every agent's learn step. The canonical precedent outside RL is `PyTorch`
/// AMP's `GradScaler`, which skips `optimizer.step()` when the unscaled
/// gradients contain `inf`/`NaN` so the params stay uncorrupted, and continues.
///
/// One guard instance owns one loss site: each distinct failure mode keeps its
/// own `warn!` latch so one site firing cannot silence another's diagnostic.
pub(crate) struct FiniteLossGuard {
    /// One-shot latch for the non-finite-loss warning. Latches the `warn!`
    /// only — never the skip.
    warned: bool,
    /// Static site label (e.g. `"ppo/policy_loss"`) used in the warning's
    /// structured `site` field and message.
    label: &'static str,
}

impl FiniteLossGuard {
    /// Creates a guard for the loss site named `label` (e.g.
    /// `"ppo/policy_loss"`), with its warning latch un-armed.
    pub(crate) const fn new(label: &'static str) -> Self {
        Self {
            warned: false,
            label,
        }
    }

    /// Returns `true` if `loss` is finite — the caller proceeds to `backward()`
    /// and the optimizer step. Returns `false` if non-finite (NaN/±Inf) — the
    /// caller MUST skip both. On the FIRST non-finite value only, emits a
    /// one-shot `tracing::warn!`.
    ///
    /// CRITICAL: the skip (returning `false`) fires on EVERY non-finite
    /// occurrence; only the `warn!` is latched. Never gate the return value
    /// on `self.warned` — a run that emits NaN every step must be protected
    /// every step (see ADR 0056 §3 and the mixed-precision `GradScaler`
    /// precedent).
    pub(crate) fn check(&mut self, loss: f32) -> bool {
        if loss.is_finite() {
            return true;
        }
        if !self.warned {
            self.warned = true;
            tracing::warn!(
                loss = loss,
                site = self.label,
                "Non-finite loss ({loss}) at {} — the backward pass and \
                 optimizer step were SKIPPED to keep weights uncorrupted \
                 (Burn propagates NaN silently rather than panicking). This \
                 step's value is excluded from the reported mean. One isolated \
                 occurrence is usually harmless; a persistent source is almost \
                 always a diverging policy/critic, a bad learning rate, or a \
                 degenerate log/div (e.g. PPO ratio exp-overflow). Check loss \
                 magnitudes and lower the learning rate or rescale rewards if \
                 they grow without bound.",
                self.label,
            );
        }
        false
    }

    /// Whether the one-shot non-finite-loss warning has already fired for this
    /// instance.
    ///
    /// Exposed for tests: the crate has no `tracing` capture dependency, so the
    /// once-only latch is asserted directly rather than by scraping log output
    /// — the same approach as
    /// [`LogAlpha::nonfinite_grad_warning_fired`](super::sac::sac_alpha).
    #[cfg(test)]
    pub(crate) fn warning_fired(&self) -> bool {
        self.warned
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
    use burn::tensor::Device;

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

    fn test_batch(device: Device<B>) -> Tensor<B, 2> {
        Tensor::from_data(TensorData::new(vec![0.5_f32, -0.25], vec![1, 2]), &device)
    }

    /// Runs one full learn step against `slot`: forward / backward on a borrow,
    /// then the closure-free optimizer step.
    fn learn_step(slot: &mut Slot<TestNet<B>>, opt: &mut OptimizerAdaptor<Adam, TestNet<B>, B>) {
        let device = Device::<B>::default();
        let loss = slot.get().forward(test_batch(device)).sum();
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
        let out = slot.get().forward(test_batch(device));
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
            let loss = slot.get().forward(test_batch(device)).sum();
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

    // -------- assert_bounds_match_components --------

    use rlevo_core::action::ContinuousAction;
    use rlevo_core::base::Action;

    /// Declares a two-component [`BoundedAction`] fixture with the given
    /// bounds, so each way the contract can break has a witness.
    ///
    /// Hand-written types rather than one const-generic fixture: `low`/`high`
    /// return `&'static [f32]`, so every variant needs its own literal anyway.
    macro_rules! bounds_fixture {
        ($name:ident, $low:expr, $high:expr) => {
            #[derive(Debug, Clone, Copy)]
            struct $name([f32; 2]);

            impl Action<1> for $name {
                fn shape() -> [usize; 1] {
                    [2]
                }
                fn is_valid(&self) -> bool {
                    self.0.iter().all(|v| v.is_finite())
                }
            }

            impl ContinuousAction<1> for $name {
                const COMPONENTS: usize = 2;

                fn as_slice(&self) -> &[f32] {
                    &self.0
                }
                fn clip(&self, min: f32, max: f32) -> Self {
                    Self([self.0[0].clamp(min, max), self.0[1].clamp(min, max)])
                }
                fn from_slice(values: &[f32]) -> Self {
                    assert_eq!(values.len(), Self::COMPONENTS);
                    Self([values[0], values[1]])
                }
            }

            impl BoundedAction<1> for $name {
                fn low() -> &'static [f32] {
                    $low
                }
                fn high() -> &'static [f32] {
                    $high
                }
            }
        };
    }

    bounds_fixture!(ConformingBounds, &[-1.0, 0.0], &[1.0, 1.0]);
    bounds_fixture!(OverLongLowBounds, &[-1.0, 0.0, 0.0], &[1.0, 1.0]);
    bounds_fixture!(InvertedBounds, &[1.0, 0.0], &[-1.0, 1.0]);
    bounds_fixture!(DegenerateBounds, &[1.0, 0.0], &[1.0, 1.0]);

    #[test]
    fn test_assert_bounds_match_components_accepts_a_conforming_impl() {
        assert_bounds_match_components::<1, ConformingBounds>();
    }

    #[test]
    #[should_panic(expected = "low() has 3 elements but COMPONENTS is 2")]
    fn test_assert_bounds_match_components_rejects_a_wrong_length() {
        assert_bounds_match_components::<1, OverLongLowBounds>();
    }

    #[test]
    #[should_panic(expected = "component 0 has low() = 1 but high() = -1")]
    fn test_assert_bounds_match_components_rejects_inverted_bounds() {
        // ADR 0053's trait docs and docs/rules.md §3 both state
        // `low()[i] < high()[i]`. Left unchecked, an inverted pair reaches
        // `rng.random_range(low..=high)` (empty-range panic) and `f32::clamp`
        // (`min > max` panic) mid-episode, naming the agent instead of the impl.
        assert_bounds_match_components::<1, InvertedBounds>();
    }

    #[test]
    #[should_panic(expected = "component 0 has low() = 1 but high() = 1")]
    fn test_assert_bounds_match_components_rejects_a_degenerate_component() {
        // `low == high` is the quieter half: `random_range` is happy with a
        // one-point inclusive range, so a degenerate action space would train
        // silently forever rather than reporting anything.
        assert_bounds_match_components::<1, DegenerateBounds>();
    }

    // -------- reduce_weighted_loss --------

    use crate::replay::TransitionId;

    fn per_sample(values: Vec<f32>) -> Tensor<Flex, 1> {
        let device = <Flex as burn::tensor::backend::BackendTypes>::Device::default();
        let n = values.len();
        Tensor::from_data(TensorData::new(values, vec![n]), &device)
    }

    fn scalar(t: Tensor<Flex, 1>) -> f32 {
        t.into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 host read")[0]
    }

    #[test]
    fn test_reduce_weighted_loss_unweighted_is_a_plain_mean() {
        let device = <Flex as burn::tensor::backend::BackendTypes>::Device::default();
        let batch = SampledBatch::unweighted(vec![TransitionId::new(0); 4]);
        let reduced =
            reduce_weighted_loss::<Flex>(per_sample(vec![1.0, 2.0, 3.0, 4.0]), &batch, &device);
        assert!(
            (scalar(reduced) - 2.5).abs() < 1e-6,
            "an unweighted batch must reduce to the plain mean, preserving the uniform path"
        );
    }

    #[test]
    fn test_reduce_weighted_loss_zero_weight_zeroes_that_sample() {
        // Schaul Alg. 1 line 13: w_j scales sample j's contribution. A weight of
        // 0 must remove that sample from both loss and gradient — the mean is
        // taken over the batch size, so the zeroed sample drags the mean down.
        let device = <Flex as burn::tensor::backend::BackendTypes>::Device::default();
        let batch = SampledBatch::weighted(vec![TransitionId::new(0); 4], vec![1.0, 0.0, 1.0, 1.0]);
        let reduced =
            reduce_weighted_loss::<Flex>(per_sample(vec![1.0, 1.0, 1.0, 1.0]), &batch, &device);
        assert!(
            (scalar(reduced) - 0.75).abs() < 1e-6,
            "a zero weight must zero that sample's loss contribution: (1+0+1+1)/4 = 0.75"
        );
    }

    #[test]
    fn test_reduce_weighted_loss_scales_before_reducing() {
        let device = <Flex as burn::tensor::backend::BackendTypes>::Device::default();
        let batch = SampledBatch::weighted(vec![TransitionId::new(0); 3], vec![1.0, 0.5, 0.25]);
        // (2*1.0 + 4*0.5 + 8*0.25) / 3 = (2 + 2 + 2) / 3 = 2.0
        let reduced =
            reduce_weighted_loss::<Flex>(per_sample(vec![2.0, 4.0, 8.0]), &batch, &device);
        assert!(
            (scalar(reduced) - 2.0).abs() < 1e-6,
            "each sample must be scaled by its own weight before the batch mean"
        );
    }

    // -------- clip_to_action_bounds --------

    fn bound_row(values: &[f32]) -> Tensor<Flex, 2> {
        let device = <Flex as burn::tensor::backend::BackendTypes>::Device::default();
        Tensor::from_data(
            TensorData::new(values.to_vec(), vec![1, values.len()]),
            &device,
        )
    }

    #[test]
    fn test_clip_to_action_bounds_respects_asymmetric_components() {
        // ADR 0053 §6's regression witness, at the level DDPG's target path
        // uses directly. CarRacing is `Box([-1,0,0], [1,1,1])`: steering is
        // signed, gas and brake are not. The pre-fix scalar clip used
        // `low[0]`/`high[0]` = `-1`/`1` for all three components, so a target
        // actor emitting -5 everywhere produced a target action of
        // `[-1, -1, -1]` — negative gas and negative brake, actions the
        // environment cannot execute, whose Q-values TD3 Eq. 14 exists to keep
        // out of the target. Per-component clipping floors both at 0.
        let device = <Flex as burn::tensor::backend::BackendTypes>::Device::default();
        let low = bound_row(&[-1.0, 0.0, 0.0]);
        let high = bound_row(&[1.0, 1.0, 1.0]);
        // Two rows: the `[1, 3]` bounds must broadcast over the batch dim.
        let actions: Tensor<Flex, 2> = Tensor::from_data(
            TensorData::new(vec![-5.0_f32, -5.0, -5.0, 5.0, 5.0, 5.0], vec![2, 3]),
            &device,
        );

        let out = clip_to_action_bounds(actions, low, high);
        let data = out.into_data().convert::<f32>();
        let got = data.as_slice::<f32>().expect("f32 host read");

        let expected = [-1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (got[i] - want).abs() < 1e-6,
                "component {i} must clip to its own bound {want}, got {} (full row: {got:?})",
                got[i]
            );
        }
        assert!(
            got[1] >= -1e-6 && got[2] >= -1e-6,
            "gas and brake have a lower bound of 0 and must never go negative, got {got:?}"
        );
    }

    #[test]
    fn test_clip_to_action_bounds_is_a_noop_inside_the_box() {
        let device = <Flex as burn::tensor::backend::BackendTypes>::Device::default();
        let low = bound_row(&[-1.0, 0.0, 0.0]);
        let high = bound_row(&[1.0, 1.0, 1.0]);
        let actions: Tensor<Flex, 2> = Tensor::from_data(
            TensorData::new(vec![-0.5_f32, 0.25, 0.75], vec![1, 3]),
            &device,
        );

        let out = clip_to_action_bounds(actions, low, high);
        let data = out.into_data().convert::<f32>();
        let got = data.as_slice::<f32>().expect("f32 host read");

        for (i, want) in [-0.5_f32, 0.25, 0.75].iter().enumerate() {
            assert!(
                (got[i] - want).abs() < 1e-6,
                "an in-bounds action must pass through unchanged at component {i}: \
                 wanted {want}, got {}",
                got[i]
            );
        }
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

    // -------- FiniteLossGuard --------

    #[test]
    fn finite_loss_guard_passes_finite_and_does_not_arm() {
        let mut guard = FiniteLossGuard::new("test/site");
        assert!(guard.check(1.5), "a finite positive loss must proceed");
        assert!(guard.check(0.0), "zero is finite and must proceed");
        assert!(guard.check(-3.25), "a finite negative loss must proceed");
        assert!(
            !guard.warning_fired(),
            "a finite loss must not arm the warning latch"
        );
    }

    #[test]
    fn finite_loss_guard_skips_and_arms_on_nan() {
        let mut guard = FiniteLossGuard::new("test/site");
        assert!(
            !guard.check(f32::NAN),
            "a NaN loss must return false (skip)"
        );
        assert!(
            guard.warning_fired(),
            "the first NaN must arm the one-shot warning latch"
        );
    }

    #[test]
    fn finite_loss_guard_skips_and_arms_on_infinities() {
        for inf in [f32::INFINITY, f32::NEG_INFINITY] {
            let mut guard = FiniteLossGuard::new("test/site");
            assert!(!guard.check(inf), "±inf ({inf}) must return false (skip)");
            assert!(
                guard.warning_fired(),
                "an infinite loss ({inf}) must arm the warning latch"
            );
        }
    }

    #[test]
    fn finite_loss_guard_skip_refires_but_warning_latches_once() {
        // ADR 0056 §3: the skip must fire on EVERY non-finite occurrence — only
        // the `warn!` is one-shot. A run that emits NaN every step must be
        // protected every step, so `check` returning `false` can never be
        // gated on the latch.
        let mut guard = FiniteLossGuard::new("test/site");

        assert!(!guard.check(f32::NAN), "the first NaN must skip");
        assert!(guard.warning_fired(), "and arm the latch");

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                !guard.check(bad),
                "a subsequent non-finite value ({bad}) must STILL skip"
            );
            assert!(
                guard.warning_fired(),
                "the latch stays armed but cannot re-warn"
            );
        }

        assert!(
            guard.check(2.0),
            "a healthy loss after a latched failure must proceed again"
        );
    }
}
