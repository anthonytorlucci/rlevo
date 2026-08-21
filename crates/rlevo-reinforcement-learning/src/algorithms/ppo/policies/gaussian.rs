//! Continuous (tanh-squashed Gaussian) policy head for PPO.
//!
//! Two-layer MLP with `tanh` activations, a linear mean head, and a
//! state-independent `log_std` parameter (length `action_dim`). Sampling:
//! `$z = \mu + \sigma \cdot \epsilon$` with `$\epsilon \sim \mathcal{N}(0, 1)$` drawn on CPU; the env receives
//! `a = scale · tanh(z)`, keeping it within the bounded action space.
//!
//! # z vs a in the rollout
//!
//! The buffer stores the **pre-squash** sample `z`, and
//! [`evaluate`](PpoPolicy::evaluate) computes log-probability on `z` under
//! the current policy. The tanh Jacobian term `$\sum \log(1 - \tanh^2(z))$` is the
//! same under old and new policies and therefore cancels in the PPO
//! importance ratio — so we work entirely in Gaussian-on-`z` space, which is
//! numerically stabler than the `atanh`-from-squashed-action path.
//!
//! # Entropy
//!
//! Returns the Gaussian entropy `$\sum (\log \sigma + \frac{1}{2} \log(2\pi e))$`, summed across
//! action dims. This matches `CleanRL`'s `probs.entropy().sum(1)` — the tanh
//! Jacobian's entropy contribution is omitted, consistent with PPO practice.
//!
//! # Bounding `log_std`: a precondition for `log_prob` being total
//!
//! `log_std` is clamped to `[log_std_min, log_std_max]` everywhere it is read.
//! The justification is **f32 totality**, not parity with SAC.
//!
//! `log_prob` evaluates `$((z - \mu)/\sigma)^2$` with `$\sigma = \exp(\log \sigma)$`. As `$\log \sigma$`
//! drifts down, `$\sigma$` underflows toward zero and the squared, normalized
//! residual grows without bound: for a `k`-sigma sample drawn under `$\sigma_{old}$`
//! and evaluated under `$\sigma_{new}$`, the magnitude is `$O(k \cdot \sigma_{old}/\sigma_{new})$`, so
//! `$\text{scaled}^2$` leaves f32 range once
//! `$2 \cdot (\log \sigma_{old} - \log \sigma_{new}) + 2 \ln k > \ln(3.4\mathrm{e}38) \approx 88.7$`. Past that the
//! ratio, the surrogate loss, and every gradient downstream are `inf`/`NaN`.
//! Clamping makes the domain of `log_prob` finite by construction, so the
//! function is total on f32.
//!
//! The bound is deliberately **non-binding for healthy runs**. A converging
//! continuous-control policy lives around `$\log \sigma \in [-3, 1]$`; the floor used
//! throughout this repo, `−20`, is `$\sigma \approx 2 \times 10^{-9}$`. The only runs whose numbers
//! change are runs that were already producing garbage.
//!
//! ## Deliberate deviation from reference PPO
//!
//! Reference implementations leave PPO's `log_std` **unclamped**: `CleanRL`'s
//! `ppo_continuous_action.py`, Stable-Baselines3's `DiagGaussianDistribution`,
//! and `OpenAI` Spinning Up all do. SB3 clamps only in
//! `SquashedDiagGaussianDistribution` (its SAC path). Schulman et al. (2017)
//! specifies no bound. We diverge knowingly.
//!
//! The empirical cover comes from Andrychowicz et al. (2021), *What Matters In
//! On-Policy Reinforcement Learning?*: "the minimum action standard deviation
//! seems to matter little, if it is not set too large" — which is exactly why a
//! non-binding floor is safe — and the same study observed that exponentiating
//! an unbounded parameter "occasionally produced NaN values", which is the
//! precise defect this bound removes.
//!
//! ## Trap door: this is *not* the SAC behaviour
//!
//! The SAC analogy actively misleads here, so do not reason from it.
//! [SAC's head](crate::algorithms::sac::sac_policy) clamps a per-step
//! **network output**: its `log_std` `Linear` keeps receiving gradient from
//! every in-range observation in the batch, so a saturated policy still has a
//! path back.
//!
//! Here `log_std` is a state-independent [`Param`] shared by all states. Once
//! it crosses a bound, `clamp` zeroes its gradient — **permanently**, and that
//! includes the entropy bonus's restoring force, which would otherwise push
//! `$\log \sigma$` back up. The parameter is stuck at the bound for the remainder of
//! training with no route back. The clamp therefore converts a run-destroying
//! `NaN` into a silently frozen `$\sigma$`: a strictly better failure mode, but still
//! a failure mode, and one worth watching for.
//!
//! # Telemetry: making the trap door audible
//!
//! A silent failure is worse than a loud one. Before the clamp existed, a
//! collapsing `log_std` announced itself with a `NaN`; with the clamp it would
//! otherwise present as flat returns and no signal at all. Two mechanisms
//! restore observability (ADR 0049 §4), and both are driven from
//! [`read_log_std_extrema_and_warn`](TanhGaussianPolicyHead::read_log_std_extrema_and_warn),
//! which [`min_log_std`](PpoPolicy::min_log_std) and
//! [`max_log_std`](PpoPolicy::max_log_std) each project one half of:
//!
//! 1. **A one-shot `tracing::warn!` per bound** the first time the raw
//!    parameter leaves `[log_std_min, log_std_max]`, naming the bound, the
//!    action dims that crossed it, and the fact that those dims are now
//!    permanently frozen. The floor and the ceiling latch **independently**:
//!    they are opposite failures (σ collapsing to a point mass vs σ exploding
//!    into pure noise) with different repairs, so silencing one because the
//!    other already fired would hide the second diagnosis entirely.
//! 2. **Two per-update metrics** — the minimum and the maximum clamped `$\log \sigma$`
//!    across action dims — surfaced on
//!    [`PpoUpdateStats`](crate::algorithms::ppo::ppo_agent::PpoUpdateStats) so
//!    the drift toward *either* bound is visible *before* it pins. A minimum
//!    alone is structurally blind to the ceiling: with `[-20, 2]` bounds and a
//!    dim pinned at `2`, `min_log_std` can read a perfectly healthy `0.0`
//!    while σ on that dim is frozen at `$e^2 \approx 7.4$`.
//!
//! ## Why the check is not in the forward pass
//!
//! Deciding "did the clamp bind" means comparing the raw parameter against the
//! bounds, which is inherently a **host-side** predicate: on a GPU backend
//! (wgpu) it costs a device→host sync. Putting that in
//! [`clamped_log_std`](TanhGaussianPolicyHead::clamped_log_std) would pay a
//! sync on *every* forward pass — and gating it behind "stop checking once it
//! fires" does not help, because the healthy runs we care most about never
//! fire and would sync forever.
//!
//! So the check is deliberately *not* in the hot path. It rides along with the
//! stats read in [`min_log_std`](PpoPolicy::min_log_std) /
//! [`max_log_std`](PpoPolicy::max_log_std), which
//! [`PpoAgent::update`](crate::algorithms::ppo::ppo_agent::PpoAgent::update)
//! calls **once per update** — a cost the metrics already pay. A bound that
//! binds will be reported at the end of the very update in which it binds, and
//! since the parameter can never leave the bound again, no crossing is ever
//! missed.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};
use rand::Rng;
use rand_distr::{Distribution as RandDistribution, StandardNormal};
use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};

use crate::algorithms::ppo::ppo_policy::{LogProbEntropy, PolicyOutput, PpoPolicy};

/// Construction-time knobs for [`TanhGaussianPolicyHead`].
#[derive(Debug, Clone)]
pub struct TanhGaussianPolicyHeadConfig {
    /// Observation feature count.
    pub obs_dim: usize,
    /// Hidden layer width (applied to both hidden layers).
    pub hidden: usize,
    /// Number of continuous action dimensions.
    pub action_dim: usize,
    /// Initial value for every entry of the state-independent `log_std`.
    ///
    /// Must lie within [`log_std`](Self::log_std); otherwise the head starts
    /// out already clamped (and, being state-independent, immediately gradient
    /// -frozen), which is always a configuration mistake.
    pub log_std_init: f32,
    /// Clamp applied to the learned `$\log \sigma$` parameter. `$[-20, 2]$` is the range
    /// used throughout this repo: the floor is `$\sigma \approx 2 \times 10^{-9}$`, far below any
    /// healthy policy, so the bound never binds on a run that is working; the
    /// ceiling `2` (`$\sigma \approx 7.4$`) matches the SAC head's and sits well above any
    /// converged continuous-control policy.
    ///
    /// Unlike SAC's clamp this bound is a **trap door** — see the
    /// [module docs](self) — because `log_std` here is a single shared
    /// parameter rather than a per-step network output.
    ///
    /// A [`Bounds`] rather than a `(min, max)` pair: an inverted range reaches
    /// [`Tensor::clamp`], which silently pins every `$\log \sigma$` to `min` on the
    /// autodiff path but **panics** on raw `Flex` (its `f32::clamp` delegate
    /// asserts `min <= max`). `Bounds` makes that backend-divergent failure
    /// unrepresentable. It does **not** discharge the two numerical invariants
    /// below — the absolute floor (`$\log \sigma \geq -35$`) and the maximum span
    /// (`< 40`), both checked in `validate()`, are not
    /// expressible as an ordering, and `(-120, -100)` is a perfectly
    /// well-ordered range that still reaches `NaN` (ADR 0027, ADR 0049).
    pub log_std: Bounds,
    /// Multiplier applied to `tanh(z)` before the env sees the action.
    pub action_scale: f32,
}

/// Smallest permitted `log_std_min`, bounding `$\sigma$` itself in absolute terms.
///
/// This guards a **different** failure from [`MAX_LOG_STD_SPAN`]: that one
/// bounds the *ratio* `$\sigma_{old}/\sigma_{new}$`, this one bounds `$\sigma$` on its own. Neither
/// implies the other — `(-120, -100)` is correctly ordered and spans only
/// `20`, yet `exp(-110)` is exactly `0.0` in f32, so `$(z - \mu)/\sigma$` is `$\pm\infty$`
/// (or `0/0 = NaN`) and the `NaN` reaches `backward()`.
///
/// `$\text{scaled\_sq} = ((z - \mu)/\sigma)^2$` leaves f32 range once
/// `|scaled| > √f32::MAX ≈ 1.8447·10¹⁹`, so the admissible floor scales with
/// the residual that must stay representable:
/// `$\text{log\_std\_min} \geq \ln|z - \mu| - 44.36$`. Over a worst-case residual sweep of
/// `$10^{-3} \ldots 10^{2}$` the binding case is `$|z - \mu| = 10^2$`, which requires
/// `$\geq -39.75$`; `-35` is that rounded up with margin. At `-35` the admissible
/// residual is `√f32::MAX · exp(-35) ≈ 1.16·10⁴` — two decades beyond what the
/// derivation assumes.
///
/// The floor constrains no usable configuration: `$\sigma = \exp(-35) \approx 6.3 \times 10^{-16}$`,
/// six orders of magnitude below this repo's default floor of `-20`. (For
/// scale: `exp` leaves f32's **normal** range around `-87` but does not reach
/// exactly `0.0` until `≈ -104`.)
///
/// Together with [`MAX_LOG_STD_SPAN`] this also bounds `log_std_max` from
/// above, to `< 5`. That is intended and free — a converged
/// continuous-control policy sits near `$\log \sigma \in [-3, 1]$`.
const MIN_LOG_STD_FLOOR: f32 = -35.0;

/// Largest permitted `log_std_max − log_std_min`.
///
/// `log_prob` evaluates `$\text{scaled} = (z - \mu)/\sigma_{new}$` where `z` was sampled under
/// `$\sigma_{old}$`; for a `k`-sigma sample the magnitude is `$O(k \cdot \sigma_{old}/\sigma_{new})$`, so
/// `$\text{scaled}^2$` overflows f32 once
/// `2·(log_std_max − log_std_min) + 2·ln k > ln(3.4e38) ≈ 88.7`. A span below
/// `40` keeps `$((z - \mu)/\sigma)^2$` inside f32 range with headroom for large `k`.
const MAX_LOG_STD_SPAN: f32 = 40.0;

/// Per-head, per-bound one-shot latches for the "clamp has bound" warning.
///
/// Two flags, not one. The floor and the ceiling are **opposite** failures —
/// σ collapsing to a point mass versus σ exploding into pure noise — and they
/// carry different repairs, so a single shared latch would let whichever bound
/// crossed first permanently silence the diagnosis of the other. That is
/// exactly the failure a single shared latch produced before this split: a
/// 2-dim head crossing the floor on dim 0 and the ceiling on dim 1 emitted
/// one warning naming only the floor.
///
/// Both fields use [`Ordering::Relaxed`]: each flag guards nothing but itself,
/// and the only requirement is that exactly one `swap` per flag observes
/// `false`. The two flags are independent, so no ordering between them is
/// needed either.
///
/// `Default` starts both unlatched, which is the correct state for a freshly
/// built head and for one restored from a record (see the field docs on
/// [`TanhGaussianPolicyHead::clamp_warned`]).
#[derive(Debug, Default)]
struct ClampWarnLatch {
    /// Set once the raw `$\log \sigma$` has been observed below `log_std_min`.
    below: AtomicBool,
    /// Set once the raw `$\log \sigma$` has been observed above `log_std_max`.
    above: AtomicBool,
}

impl TanhGaussianPolicyHeadConfig {
    /// Validates the config, then constructs the module on `device`.
    ///
    /// [`validate`](Validate::validate) runs first, so a config that violates
    /// the absolute `$\log \sigma$` floor, the maximum span, or the `log_std_init`
    /// range can never reach a built head. This is the *only* constructor:
    /// there is deliberately no infallible `init`, because an unchecked path
    /// would simply reinstate the bypass this method exists to close.
    /// The `try_` prefix marks the departure from Burn's own infallible
    /// `*Config::init` idiom.
    ///
    /// # Errors
    ///
    /// Returns the first [`ConfigError`] reported by
    /// [`validate`](Validate::validate).
    pub fn try_init<B: Backend>(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Result<TanhGaussianPolicyHead<B>, ConfigError> {
        self.validate()?;
        let log_std_vec: Vec<f32> = vec![self.log_std_init; self.action_dim];
        let log_std: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(log_std_vec, vec![self.action_dim]), device);
        Ok(TanhGaussianPolicyHead {
            fc1: LinearConfig::new(self.obs_dim, self.hidden).init(device),
            fc2: LinearConfig::new(self.hidden, self.hidden).init(device),
            mean: LinearConfig::new(self.hidden, self.action_dim).init(device),
            log_std: Param::from_tensor(log_std),
            action_dim: self.action_dim,
            log_std_min: self.log_std.lo(),
            log_std_max: self.log_std.hi(),
            action_scale: self.action_scale,
            clamp_warned: Arc::new(ClampWarnLatch::default()),
        })
    }
}

impl Validate for TanhGaussianPolicyHeadConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "TanhGaussianPolicyHeadConfig";
        config::nonzero(C, "obs_dim", self.obs_dim)?;
        config::nonzero(C, "hidden", self.hidden)?;
        config::nonzero(C, "action_dim", self.action_dim)?;
        // Ordering is *not* checked here: `Bounds` cannot be constructed
        // inverted, so `lo <= hi` holds by type (ADR 0027). What `Bounds` does
        // permit and `config::ordered`'s strict `<` did not is the degenerate
        // `lo == hi`, so that case is re-checked explicitly. A zero-width
        // `log σ` range is not merely useless: `log_std` is a single shared
        // parameter, so pinning it to a constant freezes sigma — and its
        // gradient — from step 0 with no path back (see the module docs). That
        // is the trap door, entered deliberately at construction.
        config::nondegenerate_bounds(C, "log_std", self.log_std)?;
        // The *absolute* floor, checked before the span because the span check
        // says nothing about either bound's magnitude: `(-120, -100)` is
        // ordered and spans only 20, yet `exp(-110)` is exactly 0.0 in f32, so
        // `(z - mu)/sigma` is +-inf and NaN reaches `backward()`. No
        // `config::` helper fits — `in_range` is a two-sided closed interval
        // and there is no float `at_least` — so the error is built here to
        // carry the derivation.
        if self.log_std.lo() < MIN_LOG_STD_FLOOR {
            return Err(ConfigError {
                config: C,
                field: "log_std",
                kind: ConstraintKind::Custom(
                    "log_std_min must be >= -35: sigma = exp(log_std_min) must not \
                     underflow, since ((z-mu)/sigma)^2 overflows f32 once \
                     |z-mu|/sigma exceeds sqrt(f32::MAX) = 1.8447e19, i.e. \
                     log_std_min >= ln|z-mu| - 44.36; -35 admits |z-mu| up to \
                     1.16e4 and is six orders below the default floor of -20",
                ),
            });
        }
        // Beyond mere orderedness, the *span* is bounded: `scaled =
        // (z − μ)/σ_new` with `z` drawn under `σ_old` has magnitude
        // `O(k · σ_old/σ_new)` for a k-sigma sample, so `scaled²` overflows f32
        // once `2·(log_std_max − log_std_min) + 2·ln k > ln(3.4e38) ≈ 88.7`.
        // `Bounds::span()` supplies the width; nothing in `Bounds` bounds it,
        // and no range helper carries the derivation, so the error is built
        // here.
        if self.log_std.span() >= MAX_LOG_STD_SPAN {
            return Err(ConfigError {
                config: C,
                field: "log_std",
                kind: ConstraintKind::Custom(
                    "log_std_max - log_std_min must be < 40: ((z-mu)/sigma)^2 \
                     overflows f32 once 2*(log_std_max - log_std_min) + 2*ln(k) \
                     exceeds ln(3.4e38) = 88.7 for a k-sigma sample",
                ),
            });
        }
        // A `log_std_init` outside the bounds would start the head already
        // clamped; because `log_std` is state-independent, its gradient would
        // be zero from step 0 with no path back (see the module docs).
        config::in_range(
            C,
            "log_std_init",
            f64::from(self.log_std.lo()),
            f64::from(self.log_std.hi()),
            f64::from(self.log_std_init),
        )?;
        config::positive(C, "action_scale", f64::from(self.action_scale))?;
        Ok(())
    }
}

/// MLP → Gaussian mean, with state-independent `log_std`, squashed via
/// `scale · tanh(z)` at the env boundary.
///
/// `log_std_min` / `log_std_max` / `action_scale` are constants captured at
/// construction time. They are **not** learnable and travel with the module
/// only because Burn's `#[derive(Module)]` requires fields to be either
/// `Param`s, sub-modules, or plain data.
///
/// The `log_std` bounds are applied on every read (see
/// [`clamped_log_std`](Self::clamped_log_std)); the [module docs](self)
/// explain why the bound exists and why it is *not* the same as SAC's.
///
/// They are stored as two `f32`s rather than as the config's
/// [`Bounds`] because the clamp site is
/// [`Tensor::clamp`], which takes two scalars, and plain `f32` fields keep the
/// `#[derive(Module)]` plain-data classification and the module record
/// untouched. Both fields are private and written only by
/// [`try_init`](TanhGaussianPolicyHeadConfig::try_init) from an
/// already-validated `Bounds`, so `log_std_min <= log_std_max` holds for every
/// observable head.
#[derive(Module, Debug)]
pub struct TanhGaussianPolicyHead<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    mean: Linear<B>,
    log_std: Param<Tensor<B, 1>>,
    action_dim: usize,
    log_std_min: f32,
    log_std_max: f32,
    action_scale: f32,
    /// One-shot latches — one per bound — for the "clamp has bound" warning
    /// (see the [module docs](self) and ADR 0049 §4).
    ///
    /// # Why `Arc<ClampWarnLatch>` and not `Once` / a module-level `static`
    ///
    /// [`min_log_std`](PpoPolicy::min_log_std) and
    /// [`max_log_std`](PpoPolicy::max_log_std) take `&self`, so the latches
    /// need interior mutability. Burn's `#[derive(Module)]` classifies a field
    /// as a sub-module only if its type mentions the backend `B`, is a `Param`,
    /// or is a `Tensor`; everything else — including this one — is carried as
    /// plain constant data, cloned by the generated `map`/`valid` code and
    /// excluded from the record. `Arc<ClampWarnLatch>` satisfies the
    /// `Clone + Debug + Send + Sync` that entails (the `Arc` supplies `Clone`
    /// over a struct of `AtomicBool`s, which is not `Clone` itself), whereas a
    /// bare `AtomicBool` is not `Clone` and `std::sync::Once` is neither
    /// `Clone` nor `Debug`.
    ///
    /// A module-level `static` was rejected: it would latch **process-wide**,
    /// so a second head (a fresh run in the same process, an evolutionary
    /// population of policies, or simply another test in the same binary)
    /// would be silenced by the first one's warning. Per-head state is the
    /// correct granularity — and, within a head, per-*bound* state is the
    /// correct granularity below that.
    ///
    /// Two consequences follow from `Arc` being *shared* on clone, and both are
    /// wanted: [`valid()`](burn::module::AutodiffModule::valid) snapshots share
    /// the autodiff head's latches, so the inner module cannot re-warn on
    /// either bound; and because the field is not persisted, a head restored
    /// from a record starts fully unlatched and will warn once more per bound
    /// that still binds — which is the right behaviour for a fresh process
    /// reading a fresh log.
    ///
    /// [`Ordering::Relaxed`] is sufficient for both flags; see
    /// [`ClampWarnLatch`].
    clamp_warned: Arc<ClampWarnLatch>,
}

impl<B: Backend> TanhGaussianPolicyHead<B> {
    /// Forward pass to the Gaussian mean of shape `(batch, action_dim)`.
    pub fn mean(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = tanh(self.fc1.forward(obs));
        let h = tanh(self.fc2.forward(h));
        self.mean.forward(h)
    }

    /// The current **raw** `$\log \sigma$` parameter vector of length `action_dim`.
    ///
    /// Deliberately unclamped: this is the learnable parameter as stored, so
    /// callers can observe drift past the bounds. The values actually used by
    /// [`sample_with_logprob`](PpoPolicy::sample_with_logprob) and
    /// [`evaluate`](PpoPolicy::evaluate) come from
    /// [`clamped_log_std`](Self::clamped_log_std).
    pub fn log_std_vec(&self) -> Tensor<B, 1> {
        self.log_std.val()
    }

    /// Action dimension `A`.
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Tanh scale applied to the pre-squash sample.
    pub fn action_scale(&self) -> f32 {
        self.action_scale
    }

    /// Clamp lower bound applied to `$\log \sigma$`.
    pub fn log_std_min(&self) -> f32 {
        self.log_std_min
    }

    /// Clamp upper bound applied to `$\log \sigma$`.
    pub fn log_std_max(&self) -> f32 {
        self.log_std_max
    }

    /// The `$\log \sigma$` used by *every* density computation: the learned parameter
    /// clamped to `[log_std_min, log_std_max]` and broadcast from `(A,)` to
    /// `(batch, A)`.
    ///
    /// Sampling and log-prob evaluation must both go through this. If one path
    /// clamped and the other did not, the sample's stored log-probability and
    /// its re-evaluation under the same weights would disagree, silently
    /// biasing the PPO importance ratio from the very first minibatch.
    fn clamped_log_std(&self, batch: usize) -> Tensor<B, 2> {
        let row: Tensor<B, 2> = self.log_std.val().unsqueeze_dim::<2>(0);
        row.repeat_dim(0, batch)
            .clamp(self.log_std_min, self.log_std_max)
    }

    /// Reads the raw `$\log \sigma$` vector to the host, warns **once per bound** if it
    /// has left `[log_std_min, log_std_max]`, and returns the
    /// `(minimum, maximum)` *clamped* `$\log \sigma$` across action dims.
    ///
    /// This is the single place that pays a device→host sync for `log_std`, and
    /// both trait-level metrics project from it — so a PPO update pays **two
    /// 4-byte readbacks, one per trait method
    /// ([`min_log_std`](PpoPolicy::min_log_std),
    /// [`max_log_std`](PpoPolicy::max_log_std)), both off the hot path**. They
    /// are not deduplicated: the trait surface is two independent `&self`
    /// getters with no shared per-update scratch, and a second read of an
    /// `action_dim`-length vector once per update is far below the noise floor
    /// of the update it accompanies. Neither is ever called from the forward
    /// pass — see the [module docs](self) for why the check cannot live in
    /// [`clamped_log_std`](Self::clamped_log_std).
    ///
    /// Clamping commutes with both extrema (`clamp` is monotone
    /// non-decreasing), so `min(clamp(x)) == clamp(min(x))` and likewise for
    /// `max` — one pass over the host slice yields both metrics, both bound
    /// checks, and the crossing dims that name them.
    fn read_log_std_extrema_and_warn(&self) -> (f32, f32) {
        let data = self.log_std.val().into_data().convert::<f32>();
        let raw = data.as_slice::<f32>().expect("log_std is f32");

        // One pass yields the two extrema *and* the crossing dims. The dims are
        // what make the warning actionable on a many-dim head: "dim 7 of 17 has
        // pinned" is a repair, "log_std has pinned" is an obituary.
        let mut raw_min = f32::INFINITY;
        let mut raw_max = f32::NEG_INFINITY;
        let mut below_dims: Vec<usize> = Vec::new();
        let mut above_dims: Vec<usize> = Vec::new();
        for (dim, &v) in raw.iter().enumerate() {
            raw_min = raw_min.min(v);
            raw_max = raw_max.max(v);
            if v < self.log_std_min {
                below_dims.push(dim);
            } else if v > self.log_std_max {
                above_dims.push(dim);
            }
        }

        // Two independently latched blocks, not an `if below { .. } else { .. }`
        // behind one flag. The floor and the ceiling are distinct diagnoses with
        // distinct repairs and they can bind in the same call (dim 0 collapsed,
        // dim 1 exploded), so each gets its own `swap`: exactly one caller per
        // bound observes `false`, and neither bound can silence the other
        // (ADR 0049 §4). Emitting the crossing dims keeps the claim scoped
        // to the dims that actually froze.
        let action_dim = raw.len();
        if !below_dims.is_empty() && !self.clamp_warned.below.swap(true, Ordering::Relaxed) {
            let unaffected = action_dim - below_dims.len();
            tracing::warn!(
                bound = "log_std_min",
                configured = self.log_std_min,
                observed = raw_min,
                sigma = raw_min.exp(),
                dims = ?below_dims,
                action_dim = action_dim,
                "PPO Gaussian log_std has hit its log_std_min clamp on action dims \
                 {below_dims:?} of {action_dim} (configured {}, raw parameter \
                 reached {raw_min}). log_std is a single state-independent \
                 parameter, so the clamp now zeroes the gradient of those dims \
                 PERMANENTLY — including the entropy bonus that would otherwise \
                 push them back up. Their sigma is frozen at exp({}) = {} for the \
                 rest of training with no path back, so those action dims are now \
                 effectively deterministic; the remaining {unaffected} dim(s) are \
                 unaffected and still learning. Restart with a smaller learning \
                 rate, a larger entropy_coef, or a corrected reward scale.",
                self.log_std_min,
                self.log_std_min,
                self.log_std_min.exp(),
            );
        }
        if !above_dims.is_empty() && !self.clamp_warned.above.swap(true, Ordering::Relaxed) {
            let unaffected = action_dim - above_dims.len();
            tracing::warn!(
                bound = "log_std_max",
                configured = self.log_std_max,
                observed = raw_max,
                sigma = raw_max.exp(),
                dims = ?above_dims,
                action_dim = action_dim,
                "PPO Gaussian log_std has hit its log_std_max clamp on action dims \
                 {above_dims:?} of {action_dim} (configured {}, raw parameter \
                 reached {raw_max}). This is the opposite failure to a collapse: \
                 sigma is *diverging*, so those dims are sampling near-uniform \
                 noise across the whole squashed range and carry almost no policy \
                 signal. log_std is a single state-independent parameter, so the \
                 clamp now zeroes their gradient PERMANENTLY — the entropy bonus \
                 pushes only upward, so nothing remains to pull them back down. \
                 Their sigma is frozen at exp({}) = {} for the rest of training; \
                 the remaining {unaffected} dim(s) are unaffected and still \
                 learning. This usually means the advantage scale or the reward \
                 scale is too large, or entropy_coef is set too high.",
                self.log_std_max,
                self.log_std_max,
                self.log_std_max.exp(),
            );
        }

        (
            raw_min.clamp(self.log_std_min, self.log_std_max),
            raw_max.clamp(self.log_std_min, self.log_std_max),
        )
    }

    /// Whether the one-shot **floor** clamp warning has already fired for this
    /// head.
    ///
    /// Exposed for tests: the crate has no `tracing` capture dependency, so the
    /// once-only latch is asserted directly rather than by scraping log output.
    /// Split per bound because the property under test is that the two latches
    /// are *independent* — a single combined accessor could not distinguish
    /// "both fired" from "one fired", which is precisely the defect a shared
    /// latch reintroduces.
    #[cfg(test)]
    pub(crate) fn clamp_warning_below_fired(&self) -> bool {
        self.clamp_warned.below.load(Ordering::Relaxed)
    }

    /// Whether the one-shot **ceiling** clamp warning has already fired for
    /// this head. See [`clamp_warning_below_fired`](Self::clamp_warning_below_fired).
    #[cfg(test)]
    pub(crate) fn clamp_warning_above_fired(&self) -> bool {
        self.clamp_warned.above.load(Ordering::Relaxed)
    }

    /// Computes per-row Gaussian log-prob and entropy of `z` under the
    /// current policy (no tanh Jacobian; it cancels in the PPO ratio).
    fn log_prob_entropy(&self, obs: Tensor<B, 2>, z: Tensor<B, 2>) -> LogProbEntropy<B> {
        let [batch, _] = z.dims();
        let mean = self.mean(obs);
        let log_std = self.clamped_log_std(batch);
        let std = log_std.clone().exp();

        // log N(z | μ, σ) = -0.5·((z-μ)/σ)² - log σ - 0.5 log 2π
        let centered = z - mean;
        let scaled = centered.clone() / std.clone();
        let scaled_sq = scaled.clone() * scaled;
        let log_2pi = (2.0_f32 * std::f32::consts::PI).ln();
        let per_dim: Tensor<B, 2> = scaled_sq.mul_scalar(-0.5) - log_std.clone() - log_2pi * 0.5;
        // Sum over action dim → (batch,).
        let log_prob = per_dim.sum_dim(1).squeeze_dim::<1>(1);

        // Gaussian entropy per dim: 0.5·log(2πe) + log σ.
        let log_2pi_e = log_2pi + 1.0;
        let entropy_per_dim = log_std + log_2pi_e * 0.5;
        let entropy = entropy_per_dim.sum_dim(1).squeeze_dim::<1>(1);
        LogProbEntropy { log_prob, entropy }
    }
}

impl<B: AutodiffBackend> PpoPolicy<B, 2> for TanhGaussianPolicyHead<B> {
    type ActionTensor = Tensor<B, 2>;

    /// Number of continuous action dimensions `A` this head was built for.
    fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Samples actions via reparameterisation `$z = \mu + \sigma \cdot \epsilon$` with `$\epsilon \sim \mathcal{N}(0, 1)$`
    /// drawn on CPU, then returns `z` (the pre-squash sample), its Gaussian
    /// log-probability, and the per-row Gaussian entropy.
    ///
    /// `z` is stored in the rollout buffer, not the tanh-squashed env action
    /// `a = scale · tanh(z)`. See the module-level note on why the tanh
    /// Jacobian is omitted.
    // `rand`'s standard-normal sampler yields f64; the tensor being filled is f32.
    // Narrowing to the tensor's own dtype is the intent, and the sample is finite
    // by construction.
    #[allow(clippy::cast_possible_truncation)]
    fn sample_with_logprob(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut (impl Rng + ?Sized),
    ) -> PolicyOutput<B, Self::ActionTensor> {
        let device = obs.device();
        let [batch, _] = obs.dims();
        let action_dim = self.action_dim;

        // Draw ε ~ N(0, 1) on CPU for reproducibility.
        let mut eps_vec: Vec<f32> = Vec::with_capacity(batch * action_dim);
        let normal = StandardNormal;
        for _ in 0..(batch * action_dim) {
            let x: f64 = normal.sample(rng);
            eps_vec.push(x as f32);
        }
        let eps: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(eps_vec, vec![batch, action_dim]), &device);

        let mean = self.mean(obs.clone());
        let std = self.clamped_log_std(batch).exp();
        // z = μ + σ·ε
        let z = mean + std * eps;

        let lp_ent = self.log_prob_entropy(obs, z.clone());
        PolicyOutput {
            action: z,
            log_prob: lp_ent.log_prob,
            entropy: lp_ent.entropy,
        }
    }

    /// Evaluates log-probability and entropy of the pre-squash samples
    /// `actions` (shape `(batch, action_dim)`) under the current Gaussian
    /// policy given `obs`. The tanh Jacobian cancels in the PPO importance
    /// ratio and is not included.
    fn evaluate(&self, obs: Tensor<B, 2>, actions: Self::ActionTensor) -> LogProbEntropy<B> {
        self.log_prob_entropy(obs, actions)
    }

    /// Extracts the pre-squash `z` row at `row` as `action_dim` `f32` values.
    ///
    /// This is the buffer representation for continuous actions. The
    /// environment-facing value (`scale · tanh(z)`) is produced separately by
    /// [`raw_to_env_row`](Self::raw_to_env_row).
    fn action_row_from_tensor(action: &Self::ActionTensor, row: usize) -> Vec<f32> {
        let data = action.clone().into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().expect("gaussian action is f32");
        let [_, action_dim] = action.dims();
        let start = row * action_dim;
        slice[start..start + action_dim].to_vec()
    }

    /// Rebuilds a `(n_rows, action_dim)` tensor from row-major pre-squash
    /// `f32` data. `action_dim` is inferred as `flat.len() / n_rows`.
    fn action_tensor_from_flat(
        flat: &[f32],
        n_rows: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::ActionTensor {
        let action_dim = flat.len() / n_rows.max(1);
        Tensor::<B, 2>::from_data(
            TensorData::new(flat.to_vec(), vec![n_rows, action_dim]),
            device,
        )
    }

    /// Converts a pre-squash `z` row to the env-facing action
    /// `scale · tanh(z)`, squashing each component into
    /// `(−action_scale, +action_scale)`.
    fn raw_to_env_row(&self, raw_row: &[f32]) -> Vec<f32> {
        raw_row
            .iter()
            .map(|z| self.action_scale * z.tanh())
            .collect()
    }

    /// Minimum **clamped** `$\log \sigma$` across action dims, and one of the two
    /// points at which the per-bound one-shot clamp warnings are evaluated.
    ///
    /// Costs one device→host sync; call once per update, never per step.
    fn min_log_std(&self) -> Option<f32> {
        Some(self.read_log_std_extrema_and_warn().0)
    }

    /// Maximum **clamped** `$\log \sigma$` across action dims — the ceiling-side
    /// counterpart to [`min_log_std`](Self::min_log_std), and the other point
    /// at which the per-bound one-shot clamp warnings are evaluated.
    ///
    /// A minimum cannot see a diverging dim: with `[-20, 2]` bounds and one dim
    /// pinned at `2`, `min_log_std` reports whatever the *healthiest* dim is
    /// doing (ADR 0049 §4). Costs one device→host sync; call once per
    /// update, never per step.
    fn max_log_std(&self) -> Option<f32> {
        Some(self.read_log_std_extrema_and_warn().1)
    }

    /// Returns `scale · tanh(μ(obs))` as the deterministic (noise-free) action
    /// for the first batch row, evaluated on the frozen inner backend.
    ///
    /// No σ·ε noise is added — appropriate for evaluation and benchmarking.
    fn deterministic_env_row_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Vec<f32> {
        // Deterministic action = squashed policy mean (no σ·ε noise).
        let action_dim = inner.action_dim();
        let env = tanh(inner.mean(obs)).mul_scalar(inner.action_scale());
        let data = env.into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().expect("gaussian mean is f32");
        slice[0..action_dim].to_vec()
    }
}

/// Converts the env-action row produced by
/// [`TanhGaussianPolicyHead::raw_to_env_row`] (the tanh-squashed, scaled
/// values) into a typed [`ContinuousAction`](rlevo_core::action::ContinuousAction)
/// via [`from_slice`](rlevo_core::action::ContinuousAction::from_slice).
///
/// Use this as the `action_from_row` closure in custom train loops; the
/// built-in [`train_continuous`](crate::algorithms::ppo::train::train_continuous)
/// calls it automatically.
#[must_use]
pub fn continuous_action_from_row<const AR: usize, A: rlevo_core::action::ContinuousAction<AR>>(
    row: &[f32],
) -> A {
    A::from_slice(row)
}

#[cfg(test)]
mod tests {
    // Exact comparison is intentional throughout this test module: the values are
    // config literals read back unchanged, or a computed result whose bit-exactness
    // is itself the property under test (that an anneal lands exactly on its
    // endpoint, that `-0.0` is accepted as the no-correction setting). A tolerance
    // would let a real regression pass. Reviewed as a class, not site-by-site.
    #![allow(clippy::float_cmp)]
    use std::sync::Mutex;

    use super::*;
    use burn::backend::{Autodiff, Flex};
    use burn::tensor::ElementConversion;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type TestAdBackend = Autodiff<Flex>;

    #[test]
    fn test_ppo_gaussian_policy_representative_head_config_is_valid() {
        let cfg = TanhGaussianPolicyHeadConfig {
            obs_dim: 3,
            hidden: 64,
            action_dim: 1,
            log_std_init: 0.0,
            log_std: Bounds::new(-20.0, 2.0),
            action_scale: 2.0,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_ppo_gaussian_policy_gaussian_logprob_consistency_between_sample_and_evaluate() {
        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            obs_dim: 3,
            hidden: 8,
            action_dim: 1,
            log_std_init: 0.0,
            log_std: Bounds::new(-20.0, 2.0),
            action_scale: 2.0,
        };
        let head: TanhGaussianPolicyHead<TestAdBackend> = cfg
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");
        let obs: Tensor<TestAdBackend, 2> = Tensor::from_data(
            TensorData::new(vec![0.1_f32, -0.2, 0.3], vec![1, 3]),
            &device,
        );
        let mut rng = StdRng::seed_from_u64(17);
        let out = head.sample_with_logprob(obs.clone(), &mut rng);
        let eval = head.evaluate(obs, out.action.clone());
        let a = out.log_prob.into_scalar().elem::<f32>();
        let b = eval.log_prob.into_scalar().elem::<f32>();
        assert!((a - b).abs() < 1e-5, "sample {a} vs evaluate {b}");
    }

    #[test]
    fn test_ppo_gaussian_policy_gaussian_logprob_at_mean_matches_reference() {
        // With μ=0, σ=1, z=0, per-dim log N(0|0,1) = -0.5·log(2π).
        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            obs_dim: 1,
            hidden: 2,
            action_dim: 2,
            log_std_init: 0.0,
            log_std: Bounds::new(-20.0, 2.0),
            action_scale: 1.0,
        };
        let head: TanhGaussianPolicyHead<TestAdBackend> = cfg
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");
        // Zero-out the mean MLP by constructing obs that yields a nonzero
        // mean — skip: use z = mean(obs). We simply check sample log_prob
        // equals evaluate on that z.
        let obs: Tensor<TestAdBackend, 2> =
            Tensor::from_data(TensorData::new(vec![0.0_f32], vec![1, 1]), &device);
        let mean = head.mean(obs.clone());
        let eval = head.evaluate(obs, mean.clone());
        // Expected per dim: −0.5·log(2π), sum across dim=2 → −log(2π) ≈ −1.8379.
        let expected = -(2.0_f32 * std::f32::consts::PI).ln();
        let got = eval.log_prob.into_scalar().elem::<f32>();
        assert!(
            (got - expected).abs() < 1e-4,
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn test_ppo_gaussian_policy_deterministic_env_row_inner_is_scaled_tanh_mean() {
        use burn::module::AutodiffModule;
        use burn::tensor::backend::AutodiffBackend;

        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            obs_dim: 3,
            hidden: 8,
            action_dim: 1,
            log_std_init: 0.0,
            log_std: Bounds::new(-20.0, 2.0),
            action_scale: 2.0,
        };
        let head: TanhGaussianPolicyHead<TestAdBackend> = cfg
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");
        let obs_vals = vec![0.1_f32, -0.2, 0.3];

        // Reference: scale · tanh(μ) computed on the autodiff head.
        let obs: Tensor<TestAdBackend, 2> =
            Tensor::from_data(TensorData::new(obs_vals.clone(), vec![1, 3]), &device);
        let mean = head.mean(obs).into_scalar().elem::<f32>();
        let expected = 2.0 * mean.tanh();

        // The inner deterministic path must agree (no sampling noise).
        let inner = head.valid();
        let obs_inner: Tensor<<TestAdBackend as AutodiffBackend>::InnerBackend, 2> =
            Tensor::from_data(TensorData::new(obs_vals, vec![1, 3]), &device);
        let row = <TanhGaussianPolicyHead<TestAdBackend> as PpoPolicy<TestAdBackend, 2>>::deterministic_env_row_inner(
            &inner, obs_inner,
        );
        assert_eq!(row.len(), 1);
        assert!(
            (row[0] - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            row[0]
        );
    }

    #[test]
    fn test_ppo_gaussian_policy_raw_to_env_row_applies_tanh_scale() {
        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            obs_dim: 1,
            hidden: 2,
            action_dim: 1,
            log_std_init: 0.0,
            log_std: Bounds::new(-20.0, 2.0),
            action_scale: 2.0,
        };
        let head: TanhGaussianPolicyHead<TestAdBackend> = cfg
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");
        let env_row = head.raw_to_env_row(&[0.5_f32]);
        let expected = 2.0 * 0.5_f32.tanh();
        assert!((env_row[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_gaussian_policy_gaussian_entropy_matches_gaussian_formula_at_sigma_one() {
        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            obs_dim: 1,
            hidden: 2,
            action_dim: 2,
            log_std_init: 0.0, // σ = 1
            log_std: Bounds::new(-20.0, 2.0),
            action_scale: 1.0,
        };
        let head: TanhGaussianPolicyHead<TestAdBackend> = cfg
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");
        let obs: Tensor<TestAdBackend, 2> =
            Tensor::from_data(TensorData::new(vec![0.0_f32], vec![1, 1]), &device);
        let mean = head.mean(obs.clone());
        let eval = head.evaluate(obs, mean);
        // Per dim entropy at σ=1 is 0 + 0.5·log(2πe); two dims summed.
        let expected = 2.0 * 0.5 * ((2.0_f32 * std::f32::consts::PI).ln() + 1.0);
        let got = eval.entropy.into_scalar().elem::<f32>();
        assert!(
            (got - expected).abs() < 1e-5,
            "expected {expected}, got {got}"
        );
    }

    // -----------------------------------------------------------------
    // log_std bounds: validation
    // -----------------------------------------------------------------

    /// A config template with valid bounds; individual tests perturb one field.
    fn bounded_cfg() -> TanhGaussianPolicyHeadConfig {
        TanhGaussianPolicyHeadConfig {
            obs_dim: 3,
            hidden: 8,
            action_dim: 1,
            log_std_init: 0.0,
            log_std: Bounds::new(-20.0, 2.0),
            action_scale: 2.0,
        }
    }

    /// Inverted bounds are no longer a *validation* failure, because they are
    /// no longer a constructible config: `Bounds` rejects `lo > hi` at its own
    /// boundary, so the head is unreachable.
    ///
    /// This replaces the `config::ordered` check `validate()` used to carry,
    /// and is asserted here rather than left to `rlevo-core`'s tests because
    /// this rejection is precisely *why* the ordering check was deleted — if
    /// `Bounds` ever started accepting an inverted range, PPO would silently
    /// lose the guard.
    #[test]
    fn test_ppo_gaussian_policy_inverted_log_std_bounds_are_unrepresentable() {
        assert!(
            Bounds::try_new(2.0, -20.0).is_err(),
            "an inverted log_std range must not be constructible"
        );
        assert!(Bounds::try_new(-20.0, 2.0).is_ok());
    }

    /// `Bounds` permits the degenerate `lo == hi` (clamping to a constant is
    /// well-defined), but PPO does not: `log_std` is a single shared parameter,
    /// so a zero-width range freezes σ *and its gradient* from step 0 with no
    /// path back. The old strict-`<` `config::ordered` rejected this as a side
    /// effect; the explicit `config::nondegenerate_bounds` check preserves it.
    #[test]
    fn test_ppo_gaussian_policy_validate_rejects_equal_log_std_bounds() {
        let cfg = TanhGaussianPolicyHeadConfig {
            log_std: Bounds::new(0.0, 0.0),
            ..bounded_cfg()
        };
        let err = cfg.validate().unwrap_err();
        assert_eq!(err.field, "log_std");
        assert_eq!(err.kind, ConstraintKind::DegenerateInterval { value: 0.0 });
    }

    /// A span of 40 or more lets `$((z-\mu)/\sigma)^2$` leave f32 range even though the
    /// bounds are correctly ordered — orderedness alone (all `Bounds`
    /// guarantees) would accept it.
    ///
    /// The lower bound is pinned to the absolute floor `−35` so that the *span*
    /// is the only invariant in play: a lower bound of `−38` would trip
    /// [`MIN_LOG_STD_FLOOR`] first and this test would silently assert the
    /// wrong guard.
    #[test]
    fn test_ppo_gaussian_policy_validate_rejects_log_std_span_of_forty_or_more() {
        let cfg = TanhGaussianPolicyHeadConfig {
            log_std: Bounds::new(-35.0, 5.0),
            ..bounded_cfg()
        };
        let err = cfg.validate().unwrap_err();
        assert_eq!(err.field, "log_std");
        match err.kind {
            ConstraintKind::Custom(msg) => {
                assert!(msg.contains("f32"), "message must state the reason: {msg}");
                assert!(
                    msg.contains("log_std_max - log_std_min"),
                    "the span guard must be the one that fired, not the floor: {msg}"
                );
            }
            other => panic!("expected a Custom span violation, got {other:?}"),
        }

        // Just inside the limit is accepted.
        let ok = TanhGaussianPolicyHeadConfig {
            log_std: Bounds::new(-35.0, 4.0),
            ..bounded_cfg()
        };
        assert!(ok.validate().is_ok());
    }

    /// The absolute floor is **not** implied by the other three invariants.
    ///
    /// `(-120, -100)` with `log_std_init = -110` is correctly ordered, spans
    /// only `20`, and initializes inside its own bounds — so ordering, span and
    /// range all pass — yet `exp(-110)` is exactly `0.0` in f32. The test
    /// asserts the other three genuinely pass first so that it pins *which*
    /// guard fires, rather than accidentally re-testing the span.
    #[test]
    fn test_ppo_gaussian_policy_validate_rejects_log_std_min_below_the_absolute_floor() {
        let (min, max, init) = (-120.0_f32, -100.0_f32, -110.0_f32);

        // The three pre-existing invariants are all satisfied by this triple.
        assert!(min < max, "ordering holds");
        assert!(max - min < MAX_LOG_STD_SPAN, "span {} < 40", max - min);
        assert!(min <= init && init <= max, "init lies within the bounds");
        // And yet sigma is exactly zero, which is the whole point.
        assert_eq!(init.exp(), 0.0, "exp(-110) must be exactly 0.0 in f32");

        let cfg = TanhGaussianPolicyHeadConfig {
            log_std: Bounds::new(min, max),
            log_std_init: init,
            ..bounded_cfg()
        };
        let err = cfg.validate().unwrap_err();
        assert_eq!(err.field, "log_std");
        match err.kind {
            ConstraintKind::Custom(msg) => {
                assert!(
                    msg.contains("log_std_min must be >= -35"),
                    "message must pin the floor: {msg}"
                );
            }
            other => panic!("expected a Custom floor violation, got {other:?}"),
        }

        // One ulp below the floor is still rejected; the floor itself is not.
        let just_below = TanhGaussianPolicyHeadConfig {
            log_std: Bounds::new(-35.000_004, 2.0),
            ..bounded_cfg()
        };
        assert_eq!(
            just_below.validate().unwrap_err().field,
            "log_std",
            "the floor is closed on -35 and rejects anything below it"
        );
    }

    /// The floor is non-binding: every configuration a user would plausibly
    /// write — including the repo default `[-20, 2]` — still validates.
    #[test]
    fn test_ppo_gaussian_policy_validate_accepts_sane_log_std_bounds_including_the_default() {
        for (min, max, init) in [
            (-20.0_f32, 2.0_f32, 0.0_f32), // the repo-wide default
            (-35.0, 2.0, 0.0),             // exactly at the floor: accepted
            (-35.0, 4.0, -0.5),            // widest usable interval (span 39)
            (-10.0, 2.0, 0.0),
            (-5.0, 1.0, -0.5),
            (-3.0, 1.0, 0.0),
        ] {
            let cfg = TanhGaussianPolicyHeadConfig {
                log_std: Bounds::new(min, max),
                log_std_init: init,
                ..bounded_cfg()
            };
            assert!(
                cfg.validate().is_ok(),
                "({min}, {max}, {init}) must be accepted: {:?}",
                cfg.validate().unwrap_err()
            );
        }
    }

    /// The numerical property the floor exists to buy, and the proof that
    /// rejecting the config is what prevents the `NaN`.
    ///
    /// [`try_init`](TanhGaussianPolicyHeadConfig::try_init) runs `validate()`
    /// first and is the only constructor, so the rejected `-110` head is no
    /// longer reachable through the public API at all — which is the whole
    /// point of validating on the construction path rather than leaving an
    /// infallible bypass. The below-floor case is therefore reconstructed here by
    /// writing the head's private bound fields directly, a bypass available
    /// only to this in-module test. When it is reached, `log_prob` is
    /// non-finite exactly as the ADR predicts; at the floor (`-35`) with the
    /// same residual it is finite.
    #[test]
    fn test_ppo_gaussian_policy_log_prob_is_finite_at_the_floor_and_not_below_it() {
        let device = Default::default();
        let obs_vals = vec![0.1_f32, -0.2, 0.3];

        // `max` is passed explicitly rather than derived: the below-floor case
        // must keep its span under 40 so that the *floor* is the only invariant
        // it violates. Deriving `max` from `init` would widen the span to 122
        // and the span guard, not the floor, would be the thing under test.
        let floor_cfg = |min: f32, max: f32, init: f32| TanhGaussianPolicyHeadConfig {
            log_std: Bounds::new(min, max),
            log_std_init: init,
            ..bounded_cfg()
        };

        // Evaluates `log_prob` for a head whose bounds and initial `log σ` are
        // `(min, max, init)`, *without* consulting `validate()` — the head is
        // built from an accepted config and then forced into the state under
        // test. This is deliberately the bypass `try_init` closed: the point of
        // the test is the numerical fact, not the construction path.
        let eval_at = |min: f32, max: f32, init: f32| -> f32 {
            let mut head: TanhGaussianPolicyHead<TestAdBackend> = bounded_cfg()
                .try_init::<TestAdBackend>(&device)
                .expect("the template config is valid");
            head.log_std_min = min;
            head.log_std_max = max;
            head.log_std = Param::from_tensor(Tensor::<TestAdBackend, 1>::from_data(
                TensorData::new(vec![init], vec![1]),
                &device,
            ));
            let obs: Tensor<TestAdBackend, 2> =
                Tensor::from_data(TensorData::new(obs_vals.clone(), vec![1, 3]), &device);
            // A residual of order 1 — far inside the 1.16e4 budget the floor buys.
            let z = head.mean(obs.clone()).add_scalar(1.0);
            head.evaluate(obs, z).log_prob.into_scalar().elem::<f32>()
        };

        // At the floor the log-prob is finite, and the config validates — the
        // floor is admissible, not merely small.
        let at_floor = eval_at(MIN_LOG_STD_FLOOR, 2.0, MIN_LOG_STD_FLOOR);
        assert!(
            at_floor.is_finite(),
            "log_prob at the floor must be finite, got {at_floor}"
        );
        assert!(
            floor_cfg(MIN_LOG_STD_FLOOR, 2.0, MIN_LOG_STD_FLOOR)
                .try_init::<TestAdBackend>(&device)
                .is_ok(),
            "the floor itself must be an accepted configuration"
        );

        // Below it the log-prob blows up...
        let below = eval_at(-120.0, -100.0, -110.0);
        assert!(
            !below.is_finite(),
            "log_prob below the floor must blow up; if this is finite the floor \
             is no longer load-bearing, got {below}"
        );
        // ...and *that* is why the guard must reject it at construction. This
        // assertion is what ties the numerical fact above to the validation
        // rule, and it now exercises the real construction path: `try_init`
        // itself refuses, so no caller outside this module can obtain the head
        // whose `log_prob` was just shown to be non-finite.
        let err = floor_cfg(-120.0, -100.0, -110.0)
            .try_init::<TestAdBackend>(&device)
            .expect_err("a config whose log_prob is non-finite must not build a head");
        assert_eq!(err.field, "log_std");
    }

    #[test]
    fn test_ppo_gaussian_policy_validate_rejects_log_std_init_outside_bounds() {
        for init in [-25.0_f32, 5.0] {
            let cfg = TanhGaussianPolicyHeadConfig {
                log_std_init: init,
                ..bounded_cfg()
            };
            let err = cfg.validate().unwrap_err();
            assert_eq!(err.field, "log_std_init", "init {init} must be rejected");
            assert_eq!(
                err.kind,
                ConstraintKind::OutOfRange {
                    lo: -20.0,
                    hi: 2.0,
                    got: f64::from(init),
                }
            );
        }
    }

    /// The regression this test locks down: `validate()` was correct all
    /// along, but nothing on the construction path called it. Every invariant
    /// it checks must now be reachable through `try_init`, which is the only
    /// constructor.
    #[test]
    fn test_ppo_gaussian_policy_try_init_rejects_every_invalid_config() {
        let device = Default::default();
        let cases: [(TanhGaussianPolicyHeadConfig, &str); 7] = [
            (
                TanhGaussianPolicyHeadConfig {
                    obs_dim: 0,
                    ..bounded_cfg()
                },
                "obs_dim",
            ),
            (
                TanhGaussianPolicyHeadConfig {
                    hidden: 0,
                    ..bounded_cfg()
                },
                "hidden",
            ),
            (
                TanhGaussianPolicyHeadConfig {
                    action_dim: 0,
                    ..bounded_cfg()
                },
                "action_dim",
            ),
            // Degenerate range: permitted by `Bounds`, rejected here.
            (
                TanhGaussianPolicyHeadConfig {
                    log_std: Bounds::new(0.0, 0.0),
                    ..bounded_cfg()
                },
                "log_std",
            ),
            // Below the absolute floor (span kept under 40 so the floor fires).
            (
                TanhGaussianPolicyHeadConfig {
                    log_std: Bounds::new(-120.0, -100.0),
                    log_std_init: -110.0,
                    ..bounded_cfg()
                },
                "log_std",
            ),
            // Span of exactly 40, at the rejection boundary.
            (
                TanhGaussianPolicyHeadConfig {
                    log_std: Bounds::new(-35.0, 5.0),
                    ..bounded_cfg()
                },
                "log_std",
            ),
            (
                TanhGaussianPolicyHeadConfig {
                    log_std_init: 5.0,
                    ..bounded_cfg()
                },
                "log_std_init",
            ),
        ];
        for (cfg, field) in cases {
            let err = cfg
                .try_init::<TestAdBackend>(&device)
                .expect_err("an invalid config must not build a head");
            assert_eq!(err.config, "TanhGaussianPolicyHeadConfig");
            assert_eq!(err.field, field);
        }
        // `action_scale` is checked last, so it needs an otherwise-valid config.
        let err = TanhGaussianPolicyHeadConfig {
            action_scale: 0.0,
            ..bounded_cfg()
        }
        .try_init::<TestAdBackend>(&device)
        .expect_err("a non-positive action_scale must not build a head");
        assert_eq!(err.field, "action_scale");
    }

    /// The bounds are inclusive: initializing exactly at a bound is legal.
    #[test]
    fn test_ppo_gaussian_policy_validate_accepts_log_std_init_on_the_bounds() {
        for init in [-20.0_f32, 2.0] {
            let cfg = TanhGaussianPolicyHeadConfig {
                log_std_init: init,
                ..bounded_cfg()
            };
            assert!(cfg.validate().is_ok(), "init {init} must be accepted");
        }
    }

    // -----------------------------------------------------------------
    // log_std bounds: the clamp actually binds
    // -----------------------------------------------------------------

    /// Regression test for the NaN defect: a `log_std` that has drifted far
    /// below the floor must still yield a finite `log_prob`.
    ///
    /// Without the clamp, `$\sigma = \exp(-60) \approx 8.8 \times 10^{-27}$`, so `$((z-\mu)/\sigma)^2$` for a
    /// residual of order 1 is ~`$10^{52}$` — well past f32's `$3.4 \times 10^{38}$` — and the
    /// log-prob comes back `-inf`, which poisons the ratio and every gradient
    /// downstream. With the clamp, `$\sigma = \exp(-20) \approx 2 \times 10^{-9}$` and the square stays
    /// around `$2.5 \times 10^{17}$`, comfortably inside range.
    #[test]
    fn test_ppo_gaussian_policy_clamp_keeps_log_prob_finite_when_log_std_collapses() {
        let device = Default::default();
        let mut head: TanhGaussianPolicyHead<TestAdBackend> = bounded_cfg()
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");

        // Force the learned parameter far below the floor.
        let collapsed: Tensor<TestAdBackend, 1> =
            Tensor::from_data(TensorData::new(vec![-60.0_f32], vec![1]), &device);
        head.log_std = Param::from_tensor(collapsed);

        let obs: Tensor<TestAdBackend, 2> = Tensor::from_data(
            TensorData::new(vec![0.1_f32, -0.2, 0.3], vec![1, 3]),
            &device,
        );
        // A residual of order 1 away from the mean is what blows up.
        let z = head.mean(obs.clone()).add_scalar(1.0);
        let eval = head.evaluate(obs, z);

        let lp = eval.log_prob.into_scalar().elem::<f32>();
        assert!(lp.is_finite(), "log_prob must stay finite, got {lp}");
        let ent = eval.entropy.into_scalar().elem::<f32>();
        assert!(ent.is_finite(), "entropy must stay finite, got {ent}");
    }

    /// The sampling path must draw `$z = \mu + \sigma \cdot \epsilon$` from the **clamped** `$\sigma$`, not
    /// the raw parameter.
    ///
    /// Log-prob agreement alone cannot catch a one-sided clamp here, because
    /// `sample_with_logprob` scores its own sample through `log_prob_entropy` —
    /// the same function `evaluate` uses — so the two always agree by
    /// construction. What a one-sided clamp *does* corrupt is the sample's
    /// spread: with `log_std` forced to `−60`, an unclamped draw would sit
    /// `$\exp(-60) \approx 9 \times 10^{-27}$` from the mean while every density downstream scored
    /// it under `$\exp(-20) \approx 2 \times 10^{-9}$` — an on-policy rollout collecting samples
    /// from a distribution the loss does not believe in. So this test pins the
    /// realized `$|z - \mu|$` to the clamped scale, and checks agreement on top.
    #[test]
    fn test_ppo_gaussian_policy_clamped_sample_draws_at_the_floor_scale_and_agrees_with_evaluate() {
        let device = Default::default();
        // A floor of −10 (σ ≈ 4.5·10⁻⁵) rather than the usual −20: at −20 the
        // increment σ·ε is below one f32 ulp of a mean of order 0.1, so `$z - \mu$`
        // would round to exactly zero and the assertion could not distinguish
        // the two cases. The clamp under test is identical either way.
        let cfg = TanhGaussianPolicyHeadConfig {
            log_std: Bounds::new(-10.0, 2.0),
            ..bounded_cfg()
        };
        let mut head: TanhGaussianPolicyHead<TestAdBackend> = cfg
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");
        let collapsed: Tensor<TestAdBackend, 1> =
            Tensor::from_data(TensorData::new(vec![-60.0_f32], vec![1]), &device);
        head.log_std = Param::from_tensor(collapsed);

        let obs: Tensor<TestAdBackend, 2> = Tensor::from_data(
            TensorData::new(vec![0.1_f32, -0.2, 0.3], vec![1, 3]),
            &device,
        );
        let mu = head.mean(obs.clone()).into_scalar().elem::<f32>();

        let mut rng = StdRng::seed_from_u64(23);
        let out = head.sample_with_logprob(obs.clone(), &mut rng);
        let z = out.action.clone().into_scalar().elem::<f32>();

        // |z − μ| = σ·|ε| with σ = exp(log_std_min); a fixed seed keeps |ε| in a
        // sane band, so a two-decade window around σ is a generous bound that
        // still rules out the exp(−60) draw by ~22 orders of magnitude.
        let sigma_floor = head.log_std_min().exp();
        let dev = (z - mu).abs();
        assert!(
            dev > sigma_floor * 0.01 && dev < sigma_floor * 100.0,
            "|z − μ| = {dev} must sit at the clamped scale σ = {sigma_floor}"
        );

        let eval = head.evaluate(obs, out.action.clone());
        let a = out.log_prob.into_scalar().elem::<f32>();
        let b = eval.log_prob.into_scalar().elem::<f32>();
        assert!(a.is_finite(), "sampled log_prob must be finite, got {a}");
        assert!((a - b).abs() < 1e-3, "sample {a} vs evaluate {b}");
    }

    /// The bounds survive the `Module` round-trip through `.valid()` — they are
    /// plain-data constants, not `Param`s, so the derive must carry them.
    #[test]
    fn test_ppo_gaussian_policy_log_std_bounds_survive_valid_roundtrip() {
        use burn::module::AutodiffModule;

        let device = Default::default();
        let head: TanhGaussianPolicyHead<TestAdBackend> = bounded_cfg()
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");
        assert!((head.log_std_min() - (-20.0)).abs() < f32::EPSILON);
        assert!((head.log_std_max() - 2.0).abs() < f32::EPSILON);

        let inner = head.valid();
        assert!((inner.log_std_min() - (-20.0)).abs() < f32::EPSILON);
        assert!((inner.log_std_max() - 2.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------
    // log_std telemetry: the one-shot warning and the min_log_std metric
    // -----------------------------------------------------------------

    /// Overwrites `log_std` with a single-dim value, preserving the head's
    /// bounds and its warning latch.
    fn set_log_std(head: &mut TanhGaussianPolicyHead<TestAdBackend>, value: f32) {
        let device = Default::default();
        let t: Tensor<TestAdBackend, 1> =
            Tensor::from_data(TensorData::new(vec![value], vec![1]), &device);
        head.log_std = Param::from_tensor(t);
    }

    // -----------------------------------------------------------------
    // Capturing the warnings themselves
    //
    // The latch accessors above prove a *bit flipped*, which is strictly
    // weaker than proving a warning was *emitted*: an early return, a
    // `warn!` downgraded to `debug!`, or a renamed field in the structured
    // payload all leave the latches correct and the telemetry dead. Worse,
    // two opaque booleans cannot distinguish two atomics from one atomic
    // behind two accessor names, so a mutation collapsing `ClampWarnLatch`
    // back to a single shared flag passes every latch-only assertion. The
    // tests below therefore assert on the events.
    //
    // Hand-rolled rather than pulled in as a dev-dependency: `tracing` is
    // already a direct dependency of this crate, and `with_default` plus a
    // seven-method `Subscriber` is all that is needed. Adding
    // `tracing-test`/`tracing-subscriber` to this crate's dev-deps for one
    // test module would be a workspace dependency change (rules §8) for no
    // capability gain.
    // -----------------------------------------------------------------

    /// The fields of one captured `tracing` event that the warnings are
    /// asserted on.
    #[derive(Debug, Clone, Default)]
    struct CapturedEvent {
        level: Option<tracing::Level>,
        /// The structured `bound` field: `"log_std_min"` or `"log_std_max"`.
        bound: Option<String>,
        /// The structured `dims` field, rendered as its `Debug` form (`"[0, 2]"`).
        dims: Option<String>,
        /// The structured `action_dim` field.
        action_dim: Option<u64>,
        /// The rendered human-readable message.
        message: Option<String>,
    }

    /// Pulls the asserted fields out of an event's payload.
    struct FieldVisitor<'a>(&'a mut CapturedEvent);

    impl tracing::field::Visit for FieldVisitor<'_> {
        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            if field.name() == "bound" {
                self.0.bound = Some(value.to_owned());
            }
        }

        fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
            if field.name() == "action_dim" {
                self.0.action_dim = Some(value);
            }
        }

        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            let rendered = format!("{value:?}");
            match field.name() {
                "dims" => self.0.dims = Some(rendered),
                "message" => self.0.message = Some(rendered),
                // Only reached if `bound` ever stops being a `&'static str`;
                // strip the `Debug` quotes so assertions stay stable either way.
                "bound" => self.0.bound = Some(rendered.trim_matches('"').to_owned()),
                _ => {}
            }
        }
    }

    /// Minimal `Subscriber` that records every event it is handed.
    #[derive(Debug, Default)]
    struct CaptureSubscriber {
        events: Arc<Mutex<Vec<CapturedEvent>>>,
    }

    impl tracing::Subscriber for CaptureSubscriber {
        fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            // No spans are opened by the code under test; a constant id
            // satisfies the contract without any bookkeeping.
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

        fn event(&self, event: &tracing::Event<'_>) {
            let mut captured = CapturedEvent {
                level: Some(*event.metadata().level()),
                ..CapturedEvent::default()
            };
            event.record(&mut FieldVisitor(&mut captured));
            self.events
                .lock()
                .expect("capture mutex is never poisoned")
                .push(captured);
        }

        fn enter(&self, _span: &tracing::span::Id) {}

        fn exit(&self, _span: &tracing::span::Id) {}
    }

    /// Runs `f` with a capturing subscriber installed on this thread and
    /// returns the `WARN`-level events it emitted, in order.
    ///
    /// `with_default` is thread-local, so this is safe under the parallel test
    /// runner: a concurrently running test cannot observe or pollute these
    /// events.
    fn capture_warnings(f: impl FnOnce()) -> Vec<CapturedEvent> {
        let events = Arc::new(Mutex::new(Vec::new()));
        let subscriber = CaptureSubscriber {
            events: Arc::clone(&events),
        };
        tracing::subscriber::with_default(subscriber, f);
        let captured = events.lock().expect("capture mutex is never poisoned");
        captured
            .iter()
            .filter(|e| e.level == Some(tracing::Level::WARN))
            .cloned()
            .collect()
    }

    /// Finds the single captured event naming `bound`, asserting there is
    /// exactly one.
    fn event_for_bound<'a>(events: &'a [CapturedEvent], bound: &str) -> &'a CapturedEvent {
        let matching: Vec<&CapturedEvent> = events
            .iter()
            .filter(|e| e.bound.as_deref() == Some(bound))
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "expected exactly one WARN naming bound={bound}, got {matching:?} \
             (all captured: {events:?})"
        );
        matching[0]
    }

    /// The crate carries no `tracing` capture dependency, so the once-only
    /// property is asserted on the latch itself: after the first binding call
    /// it is set, and no later call can un-set or re-trigger it. `swap` is what
    /// makes this exactly-once, and the latch is the sole gate on `warn!`.
    #[test]
    fn test_ppo_gaussian_policy_clamp_warning_latches_after_first_binding_call() {
        let device = Default::default();
        let mut head: TanhGaussianPolicyHead<TestAdBackend> = bounded_cfg()
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");
        assert!(
            !head.clamp_warning_below_fired() && !head.clamp_warning_above_fired(),
            "a freshly built head must not have warned on either bound"
        );

        set_log_std(&mut head, -60.0);
        assert_eq!(head.min_log_std(), Some(-20.0));
        assert!(
            head.clamp_warning_below_fired(),
            "the first binding read must latch the floor warning"
        );
        assert!(
            !head.clamp_warning_above_fired(),
            "a floor crossing must leave the ceiling latch armed"
        );

        // Many further binding reads must not re-arm the latch: it is already
        // `true`, so `swap` never again observes `false` and `warn!` is
        // unreachable for the rest of this head's life.
        for _ in 0..64 {
            assert_eq!(head.min_log_std(), Some(-20.0));
            assert!(head.clamp_warning_below_fired());
        }
    }

    /// A `log_std` inside the bounds must stay silent — the warning is reserved
    /// for the run-is-dead case, so a false positive would train users to
    /// ignore it.
    #[test]
    fn test_ppo_gaussian_policy_clamp_warning_does_not_fire_while_log_std_is_in_bounds() {
        let device = Default::default();
        let mut head: TanhGaussianPolicyHead<TestAdBackend> = bounded_cfg()
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");

        // Interior values, plus both bounds exactly: the clamp is inclusive, so
        // sitting *on* a bound is not "outside" it and must not warn.
        for v in [0.0_f32, -5.0, -19.9, 1.9, -20.0, 2.0] {
            set_log_std(&mut head, v);
            let mut got = f32::NAN;
            let events = capture_warnings(|| {
                got = head.min_log_std().expect("gaussian head reports log_std");
            });
            assert!(
                (got - v).abs() < 1e-6,
                "in-bounds log_std {v} must pass through unclamped, got {got}"
            );
            // Silence asserted on the event stream, not just the latches: a
            // false positive here would train users to ignore the warning.
            assert!(
                events.is_empty(),
                "in-bounds log_std {v} must not warn, got {events:?}"
            );
            assert!(
                !head.clamp_warning_below_fired() && !head.clamp_warning_above_fired(),
                "in-bounds log_std {v} must not latch"
            );
        }
    }

    /// Drifting *above* `log_std_max` is the other trap door and must warn too
    /// — on its own latch, not the floor's.
    #[test]
    fn test_ppo_gaussian_policy_clamp_warning_fires_on_the_upper_bound() {
        let device = Default::default();
        let mut head: TanhGaussianPolicyHead<TestAdBackend> = bounded_cfg()
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");
        set_log_std(&mut head, 9.0);
        assert_eq!(head.min_log_std(), Some(2.0));
        assert!(head.clamp_warning_above_fired());
        assert!(
            !head.clamp_warning_below_fired(),
            "a ceiling crossing must not latch the floor's warning"
        );
    }

    /// The regression this test guards against: a single call in which one
    /// dim is below the floor **and** another is above the ceiling must emit
    /// **two** `WARN` events, one naming each bound, each carrying the dims
    /// that crossed it.
    ///
    /// Under the old single-`AtomicBool` latch this call emitted exactly one
    /// event, and — because the emitting branch was `if below { .. } else
    /// { .. }` — that event named only `log_std_min`. The ceiling violation
    /// (`$\log \sigma = 9$`, σ ≈ 8103, a policy sampling pure noise on dim 1) was
    /// dropped from the very first warning and, the latch now being set, from
    /// every subsequent one too.
    ///
    /// Asserted on the captured events rather than the latches, and that is
    /// load-bearing in two ways. Counting events is what makes this test able
    /// to see the *latch split itself*: two opaque booleans cannot distinguish
    /// two atomics from one atomic behind two accessor names, so a mutation
    /// collapsing `ClampWarnLatch` back to one shared flag passes any
    /// latch-only version of this test and fails this one at `len() == 2`.
    /// Reading the payload is what pins the second half of the fix — that each
    /// event names its own bound and its own dims — which no boolean can
    /// express at all.
    #[test]
    fn test_ppo_gaussian_policy_both_bounds_crossed_in_one_call_emit_two_distinct_warnings() {
        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            action_dim: 2,
            ..bounded_cfg()
        };
        let mut head: TanhGaussianPolicyHead<TestAdBackend> = cfg
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");

        // dim 0 collapsed below −20, dim 1 exploded above +2, in one parameter.
        let both: Tensor<TestAdBackend, 1> =
            Tensor::from_data(TensorData::new(vec![-60.0_f32, 9.0], vec![2]), &device);
        head.log_std = Param::from_tensor(both);

        let mut extrema = (0.0_f32, 0.0_f32);
        let events = capture_warnings(|| {
            extrema = head.read_log_std_extrema_and_warn();
        });

        let (lo, hi) = extrema;
        assert!(
            (lo - (-20.0)).abs() < 1e-6,
            "the minimum must saturate at the floor, got {lo}"
        );
        assert!(
            (hi - 2.0).abs() < 1e-6,
            "the maximum must saturate at the ceiling, got {hi}"
        );

        // Two events, not one. A single shared latch yields one and fails here.
        assert_eq!(
            events.len(),
            2,
            "one crossing per bound must produce one WARN per bound, got {events:?}"
        );

        let floor = event_for_bound(&events, "log_std_min");
        assert_eq!(
            floor.dims.as_deref(),
            Some("[0]"),
            "the floor warning must name the dim that collapsed"
        );
        assert_eq!(floor.action_dim, Some(2));

        let ceiling = event_for_bound(&events, "log_std_max");
        assert_eq!(
            ceiling.dims.as_deref(),
            Some("[1]"),
            "the ceiling warning must name the dim that exploded"
        );
        assert_eq!(ceiling.action_dim, Some(2));
        // The two messages must be genuinely different prose, not the floor's
        // text with a substituted bound name: a diverging σ and a collapsing σ
        // have opposite repairs.
        assert!(
            ceiling
                .message
                .as_deref()
                .is_some_and(|m| m.contains("diverging")),
            "the ceiling message must describe divergence, got {:?}",
            ceiling.message
        );
        assert!(
            floor
                .message
                .as_deref()
                .is_some_and(|m| m.contains("deterministic")),
            "the floor message must describe collapse, got {:?}",
            floor.message
        );

        // The latches agree with the events; kept as a cheap consistency check
        // now that the events are the primary assertion.
        assert!(head.clamp_warning_below_fired() && head.clamp_warning_above_fired());
    }

    /// The `dims` payload must name **every** crossing dim, not just the one
    /// that happened to set the extremum.
    ///
    /// The module docs justify the clamp warning's per-dim scoping by claiming
    /// it is what makes the event actionable on a many-dim head — "dim 7 of 17
    /// has pinned" rather than an obituary for the whole run. That claim is
    /// only true if the list is complete, and `raw_min` / `raw_max` alone
    /// cannot express it: they collapse any number of crossing dims to one
    /// number. Three of five dims cross the floor here, and one crosses the
    /// ceiling, so a payload built from the extrema instead of the collected
    /// dims fails.
    #[test]
    fn test_ppo_gaussian_policy_warning_payload_names_every_crossing_dim() {
        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            action_dim: 5,
            ..bounded_cfg()
        };
        let mut head: TanhGaussianPolicyHead<TestAdBackend> = cfg
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");

        // dims 0, 2, 4 below the floor; dim 3 above the ceiling; dim 1 healthy.
        let mixed: Tensor<TestAdBackend, 1> = Tensor::from_data(
            TensorData::new(vec![-60.0_f32, 0.0, -25.0, 9.0, -40.0], vec![5]),
            &device,
        );
        head.log_std = Param::from_tensor(mixed);

        let events = capture_warnings(|| {
            let _ = head.read_log_std_extrema_and_warn();
        });
        assert_eq!(events.len(), 2, "both bounds crossed, got {events:?}");

        let floor = event_for_bound(&events, "log_std_min");
        assert_eq!(
            floor.dims.as_deref(),
            Some("[0, 2, 4]"),
            "every collapsed dim must be named, not only the minimum"
        );
        assert_eq!(floor.action_dim, Some(5));
        // Scoped claim, not a blanket one: 5 dims, 3 collapsed, 2 unaffected.
        assert!(
            floor
                .message
                .as_deref()
                .is_some_and(|m| m.contains("2 dim(s) are unaffected")),
            "the message must scope its claim to the affected dims, got {:?}",
            floor.message
        );

        let ceiling = event_for_bound(&events, "log_std_max");
        assert_eq!(ceiling.dims.as_deref(), Some("[3]"));
        assert!(
            ceiling
                .message
                .as_deref()
                .is_some_and(|m| m.contains("4 dim(s) are unaffected")),
            "the ceiling message must scope its claim too, got {:?}",
            ceiling.message
        );
    }

    /// A ceiling crossing that happens *after* a floor crossing must still
    /// warn, and a second ceiling crossing must not warn again.
    ///
    /// This is the filed sequence: the old shared latch was consumed by the
    /// floor event, so every later ceiling crossing found it already set and
    /// was silent. The fix must restore the second diagnosis **without**
    /// weakening the once-per-bound property that keeps the warning
    /// trustworthy — so both halves are asserted here, in order, by **counting
    /// emitted events**. A latch flag can only say "has fired at some point";
    /// only a count can say "fired exactly once", which is the actual property.
    #[test]
    fn test_ppo_gaussian_policy_later_ceiling_crossing_warns_once_after_an_earlier_floor_crossing()
    {
        let device = Default::default();
        let mut head: TanhGaussianPolicyHead<TestAdBackend> = bounded_cfg()
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");

        // 1. Floor crossing: exactly one event, naming the floor.
        set_log_std(&mut head, -60.0);
        let first = capture_warnings(|| {
            assert_eq!(head.min_log_std(), Some(-20.0));
        });
        assert_eq!(first.len(), 1, "one floor crossing, one WARN: {first:?}");
        assert_eq!(first[0].bound.as_deref(), Some("log_std_min"));

        // 2. A later ceiling crossing must still be able to warn — under the
        // old shared latch this captured zero events.
        set_log_std(&mut head, 9.0);
        let second = capture_warnings(|| {
            assert_eq!(head.max_log_std(), Some(2.0));
        });
        assert_eq!(
            second.len(),
            1,
            "the ceiling warning must not have been consumed by the earlier \
             floor event: {second:?}"
        );
        assert_eq!(second[0].bound.as_deref(), Some("log_std_max"));
        assert_eq!(second[0].dims.as_deref(), Some("[0]"));

        // 3. ...and only once. The latch is `true`, so no further `swap`
        // observes `false` and `warn!` is unreachable for this bound: 16 more
        // binding reads must emit nothing at all.
        let rest = capture_warnings(|| {
            for _ in 0..16 {
                assert_eq!(head.max_log_std(), Some(2.0));
            }
        });
        assert!(
            rest.is_empty(),
            "the once-per-bound property must survive: {rest:?}"
        );
    }

    /// `min_log_std` is structurally blind to a dim pinned at the ceiling, and
    /// `max_log_std` is the metric that sees it.
    ///
    /// With `[-20, 2]` bounds, a head whose clamped `$\log \sigma$` is `[-20, 2]`
    /// reports `min_log_std = -20.0`... but a head whose *raw* parameter is
    /// `[-20.0, 9.0]` — one healthy dim, one dim frozen at σ = e² with a dead
    /// gradient — reports `min_log_std = -20.0` too, and a head at
    /// `[0.0, 9.0]` reports a completely healthy-looking `0.0`. A minimum can
    /// never fall when a dim rises, so no threshold on it detects divergence.
    /// That measured blindness is the reason `max_log_std` exists.
    #[test]
    fn test_ppo_gaussian_policy_max_log_std_sees_the_ceiling_that_min_log_std_cannot() {
        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            action_dim: 2,
            ..bounded_cfg()
        };
        let mut head: TanhGaussianPolicyHead<TestAdBackend> = cfg
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");

        // dim 0 healthy at σ = 1; dim 1 pinned well above the ceiling.
        let pinned: Tensor<TestAdBackend, 1> =
            Tensor::from_data(TensorData::new(vec![0.0_f32, 9.0], vec![2]), &device);
        head.log_std = Param::from_tensor(pinned);

        let (mut min, mut max) = (f32::NAN, f32::NAN);
        let events = capture_warnings(|| {
            min = head.min_log_std().expect("gaussian head reports log_std");
            max = head.max_log_std().expect("gaussian head reports log_std");
        });
        assert!(
            (min - 0.0).abs() < 1e-6,
            "min_log_std reads healthy while a dim is pinned — got {min}; this is \
             the blindness, not a bug"
        );
        assert!(
            (max - 2.0).abs() < 1e-6,
            "max_log_std must saturate at the ceiling and expose the pinned dim, got {max}"
        );
        // Exactly one WARN across both reads: the ceiling's. The floor was
        // never crossed, and the second read must not re-warn.
        assert_eq!(events.len(), 1, "expected one ceiling WARN, got {events:?}");
        assert_eq!(events[0].bound.as_deref(), Some("log_std_max"));
        assert_eq!(events[0].dims.as_deref(), Some("[1]"));
    }

    /// `min_log_std` is a minimum across action dims, and it reports the
    /// **clamped** value — the σ actually used by every density computation,
    /// not the raw parameter.
    #[test]
    fn test_ppo_gaussian_policy_min_log_std_is_the_clamped_minimum_across_action_dims() {
        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            action_dim: 3,
            ..bounded_cfg()
        };
        let mut head: TanhGaussianPolicyHead<TestAdBackend> = cfg
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");

        // Mixed dims, all in bounds: the smallest wins.
        let mixed: Tensor<TestAdBackend, 1> =
            Tensor::from_data(TensorData::new(vec![1.5_f32, -3.25, 0.0], vec![3]), &device);
        head.log_std = Param::from_tensor(mixed);
        let got = head.min_log_std().expect("gaussian head reports log_std");
        assert!((got - (-3.25)).abs() < 1e-6, "expected -3.25, got {got}");
        assert!(!head.clamp_warning_below_fired() && !head.clamp_warning_above_fired());

        // One dim collapsed below the floor: the metric saturates at the floor
        // rather than reporting the raw value, because the floor is the σ the
        // policy is actually sampling and scoring with.
        let collapsed: Tensor<TestAdBackend, 1> =
            Tensor::from_data(TensorData::new(vec![1.5_f32, -60.0, 0.0], vec![3]), &device);
        head.log_std = Param::from_tensor(collapsed);
        let got = head.min_log_std().expect("gaussian head reports log_std");
        assert!((got - (-20.0)).abs() < 1e-6, "expected -20.0, got {got}");
        assert!(head.clamp_warning_below_fired());
    }

    /// `.valid()` shares the latches rather than copying them, so an inner
    /// snapshot taken after a warning cannot emit a duplicate. This is a
    /// property of `Arc` being cloned by the `Module` derive's plain-data path;
    /// a bare `AtomicBool` would not even compile there, and a copied one would
    /// double-warn.
    ///
    /// Asserted for **both** latches, and for both directions of sharing: an
    /// already-fired flag must be inherited, and a flag fired *after* the
    /// snapshot was taken must be visible through it. A `ClampWarnLatch` cloned
    /// field-by-field instead of behind the `Arc` would pass the first check on
    /// the day it was written and fail the second.
    #[test]
    fn test_ppo_gaussian_policy_clamp_warning_latch_is_shared_across_valid_snapshots() {
        use burn::module::AutodiffModule;

        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            action_dim: 2,
            ..bounded_cfg()
        };
        let mut head: TanhGaussianPolicyHead<TestAdBackend> = cfg
            .try_init::<TestAdBackend>(&device)
            .expect("valid head config");

        // Fire the floor latch only, then snapshot.
        let collapsed: Tensor<TestAdBackend, 1> =
            Tensor::from_data(TensorData::new(vec![-60.0_f32, 0.0], vec![2]), &device);
        head.log_std = Param::from_tensor(collapsed);
        let _ = head.min_log_std();
        assert!(head.clamp_warning_below_fired());
        assert!(!head.clamp_warning_above_fired());

        let inner = head.valid();
        assert!(
            inner.clamp_warning_below_fired(),
            "the inner snapshot must inherit the already-fired floor latch"
        );
        assert!(
            !inner.clamp_warning_above_fired(),
            "the un-fired ceiling latch must be inherited un-fired, not conflated \
             with the floor's"
        );

        // Fire the ceiling latch on the outer head *after* the snapshot: the
        // `Arc` is shared, so the inner snapshot sees it too.
        let exploded: Tensor<TestAdBackend, 1> =
            Tensor::from_data(TensorData::new(vec![0.0_f32, 9.0], vec![2]), &device);
        head.log_std = Param::from_tensor(exploded);
        let _ = head.max_log_std();
        assert!(
            inner.clamp_warning_above_fired(),
            "the latch is shared, not copied: a warning fired after the snapshot \
             must still silence it"
        );
    }
}
