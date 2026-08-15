//! Hyperparameter configuration for the QR-DQN (Quantile Regression DQN)
//! algorithm.
//!
//! Mirrors [`crate::algorithms::c51::c51_config::C51TrainingConfig`] for all
//! shared training hyperparameters, but replaces the categorical
//! distributional fields (`num_atoms`, `v_min`, `v_max`) with
//! [`num_quantiles`](QrDqnTrainingConfig::num_quantiles) and the Huber
//! threshold [`kappa`](QrDqnTrainingConfig::kappa). QR-DQN learns a
//! fixed-cardinality quantile function rather than a categorical
//! distribution over a fixed support, so no `[v_min, v_max]` range is
//! required.

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};

use crate::replay::{PrioritizedReplaySettings, UniformReplayConfig};
use crate::target::TargetUpdate;

/// Configuration for training a Quantile Regression DQN (QR-DQN) agent.
///
/// Holds all hyperparameters required to initialise and train a
/// [`crate::algorithms::qrdqn::qrdqn_agent::QrDqnAgent`]. The distribution-
/// specific fields are [`num_quantiles`](Self::num_quantiles) and
/// [`kappa`](Self::kappa); the rest are the standard DQN knobs (learning
/// rate, γ, τ, ε schedule, replay capacity, …).
#[derive(Clone, Debug)]
pub struct QrDqnTrainingConfig {
    /// Minibatch size sampled from the replay buffer each learn step.
    pub batch_size: usize,

    /// Discount factor γ in `[0, 1]`.
    pub gamma: f64,

    /// Optimizer learning rate.
    pub learning_rate: f64,

    /// Initial ε value for the ε-greedy exploration schedule.
    pub epsilon_start: f64,

    /// Floor ε value for the exploration schedule.
    pub epsilon_end: f64,

    /// Multiplicative decay applied to ε each env step.
    pub epsilon_decay: f64,

    /// How the target network tracks the policy network: one cadence, one τ.
    ///
    /// [`TargetUpdate`] is a single mechanism, not two (ADR 0058). Its cadence
    /// [`every`](TargetUpdate::every) decides *when* an update fires; its
    /// coefficient [`tau`](TargetUpdate::tau) decides *how far* the target
    /// moves when it does — `$\text{target} \leftarrow (1 - \tau) \cdot \text{target} + \tau \cdot \text{policy}$`. A
    /// periodic hard copy is not a second mechanism but the degenerate
    /// `$\tau = 1.0$`, spelled [`TargetUpdate::hard`]. The update is applied inside
    /// [`QrDqnAgent::learn_step`] and nowhere else, so no train loop can forget
    /// to drive it.
    ///
    /// # The cadence counts gradient updates — its neighbours count env steps
    ///
    /// [`every`](TargetUpdate::every) is a **gradient (optimizer) update**
    /// count (ADR 0059), matching Mnih et al. (2015), whose `C` is "measured in
    /// the number of *parameter updates*" (Extended Data Table 1).
    /// [`learning_starts`](Self::learning_starts) and
    /// [`train_frequency`](Self::train_frequency) stay in **environment
    /// steps**, because they gate whether a gradient step is taken at all.
    /// This config therefore carries two units deliberately, and only this note
    /// distinguishes them: at the default `train_frequency = 4`, a cadence of
    /// `10_000` gradient updates is 40 000 environment steps.
    ///
    /// The counter fed to the cadence advances once per attempted optimizer
    /// step — including one skipped by the non-finite-loss guard (ADR 0056), so
    /// a diverging run cannot silently stretch the target-update rhythm.
    ///
    /// [`QrDqnAgent::learn_step`]: crate::algorithms::qrdqn::qrdqn_agent::QrDqnAgent::learn_step
    pub target_update: TargetUpdate,

    /// Upper bound on steps allowed per episode (for bookkeeping only —
    /// environments are responsible for enforcing their own step limits).
    pub steps_per_episode: usize,

    /// Maximum number of transitions retained in the replay buffer.
    ///
    /// Must be non-zero and at most
    /// [`MAX_BUFFER_CAPACITY`](crate::MAX_BUFFER_CAPACITY). The ceiling is not
    /// a taste judgement: a misconfigured value (a unit slip, a `usize::MAX`
    /// sentinel) either aborts on a failed allocation or wraps the `SumTree`'s
    /// node arithmetic. Both bounds are checked by [`UniformReplayConfig`],
    /// which `validate` delegates to, and which shares its ceiling with
    /// `PrioritizedReplayConfig` so toggling
    /// [`prioritized_replay`](Self::prioritized_replay) cannot move the
    /// accept/reject boundary. See
    /// [`MAX_BUFFER_CAPACITY`](crate::MAX_BUFFER_CAPACITY) for why `$2^{32}$`
    /// is the bound.
    pub replay_buffer_capacity: usize,

    /// Number of env steps collected before the first gradient update.
    pub learning_starts: usize,

    /// Period, in env steps, between gradient updates.
    pub train_frequency: usize,

    /// Number of quantiles `N` used to represent the return distribution.
    ///
    /// The default is 200 — the value from Dabney et al. (2018). Quantile
    /// midpoints `$\tau_i = (i + 0.5) / N$` are implied by this field alone.
    pub num_quantiles: usize,

    /// Huber threshold `$\kappa$` used by the quantile Huber loss. Values below
    /// `$\kappa$` are treated quadratically, above `$\kappa$` linearly. Default `1.0`.
    ///
    /// # Valid domain: finite, strictly positive, with `$0.5 \cdot \kappa^2$` finite in `f32`
    ///
    /// [`validate`](Validate::validate) rejects `$\kappa \leq 0$`; every non-finite κ
    /// (`NaN` and `$\pm\infty$` alike, as
    /// [`ConstraintKind::NotFinite`](rlevo_core::config::ConstraintKind::NotFinite));
    /// and every *finite* κ large enough that `$0.5 \cdot \kappa^2$` overflows `f32` — the
    /// last accepted value is `≈ 2.6087635e19` (`√(2 · f32::MAX)`).
    ///
    /// Two independent `NaN` sources motivate the bounds. Dabney et al. (2018)
    /// Eq. (10) is `$\rho^\kappa_\tau(u) = |\tau - \mathbb{1}\{u<0\}| \cdot L_\kappa(u) / \kappa$`, so κ is a
    /// **divisor** and κ = 0 evaluates `$0/0$`. Separately, `huber` evaluates
    /// *both* branches eagerly and selects with a multiplicative mask:
    /// `$\text{linear} = (|u| - 0.5\cdot\kappa)\cdot\kappa \approx -0.5\kappa^2$` for the small-residual elements
    /// the mask is about to discard. Once `$0.5 \cdot \kappa^2$` overflows to `$-\infty$`, the
    /// discard is `0 · (−∞) = NaN` — so a huge-but-finite κ poisons every
    /// element even though the mask selects the quadratic branch everywhere.
    /// The upper bound is therefore the precondition of that eager branch, not
    /// a magic constant; it is verified against the measured boundary in
    /// `rejects_overflowing_kappa`.
    ///
    /// # What an invalid κ actually costs
    ///
    /// It does **not** corrupt the weights. Every QR-DQN update reads its loss
    /// scalar host-side and passes it through `FiniteLossGuard` (ADR 0056,
    /// issue #318) *before* `backward()`, so a `NaN` loss skips the backward
    /// pass, the optimizer step, the target update, and the PER writeback. The
    /// failure mode is quieter than corruption and harder to spot: every
    /// update becomes a no-op while `gradient_updates` keeps advancing, so
    /// training runs to completion, reports its cadence faithfully, and learns
    /// nothing — discoverable only from a single one-shot `tracing::warn!`.
    /// Rejecting κ here fails fast with a named-field [`ConfigError`] instead
    /// of relying on a downstream generic guard and a silent stall.
    ///
    /// # Tiny κ is accepted and is not a `NaN`, but is not useful
    ///
    /// Subnormal κ (~`1e-45`) validates and produces a finite loss, because
    /// `$L_\kappa(u)$` underflows to `0.0` before the division does any damage. The
    /// measured loss is then exactly `0.0` for ordinary residuals: the
    /// gradient signal is flushed away by `f32` precision loss rather than by
    /// any modelled behaviour. This is documented, not rejected — the boundary
    /// is unrepresentable as a clean threshold and depends on the residual
    /// magnitudes. Values below ~`1e-20` should be treated as a modelling
    /// mistake.
    ///
    /// # κ = 0 (QR-DQN-0) is not representable here
    ///
    /// The paper notes that "as κ → 0 the quantile Huber loss reverts to the
    /// quantile regression loss" — a **limit** statement, not an evaluation.
    /// The paper's own κ = 0 variant, QR-DQN-0, is the strict quantile loss of
    /// Eq. (8), `$\rho_\tau(u) = u(\tau - \mathbb{1}\{u<0\})$`, a separate unsmoothed formula
    /// substituted for Eq. (10) rather than Eq. (10) at κ = 0. That unsmoothed
    /// path is not implemented in this crate, so QR-DQN-0 cannot be selected by
    /// setting this field to `0.0`. The default `1.0` is the paper's
    /// recommended QR-DQN-1, which outperformed QR-DQN-0 on Atari-57.
    pub kappa: f32,

    /// Optional gradient-norm / gradient-value clipping.
    pub clip_grad: Option<GradientClippingConfig>,

    /// Optimizer configuration (Adam β's, ε, etc.).
    pub optimizer: AdamConfig,

    /// Opt-in prioritized experience replay (Schaul et al. 2016), `None` by
    /// default (uniform replay).
    ///
    /// The priority signal for QR-DQN is the per-sample **quantile Huber loss
    /// magnitude**.
    ///
    /// # This combination is uncited — a design choice by analogy, not a result
    ///
    /// Dabney et al. (2018) explicitly **decline** to combine QR-DQN with
    /// prioritized replay: "we expect both to benefit from … prioritized replay
    /// (Schaul et al. 2016). However, in our evaluations we compare the pure
    /// versions of C51 and QR-DQN **without these additions**." Using the
    /// quantile Huber loss as the priority signal extrapolates Rainbow's stated
    /// *principle* — prioritize transitions "by what the algorithm is
    /// minimizing" — to a case **no primary source ablates**. It is offered
    /// here as an opt-in, and this note is the whole of its provenance: there is
    /// no citation for it, and none should be added. Buffer capacity comes from
    /// [`replay_buffer_capacity`](Self::replay_buffer_capacity).
    pub prioritized_replay: Option<PrioritizedReplaySettings>,
}

impl QrDqnTrainingConfig {
    /// Builds the quantile-midpoint tensor `$[(i + 0.5) / N]_{i=0..N-1}$` on
    /// the requested backend and device. Shape: `(num_quantiles,)`.
    // Structural size of the distributional support (atom / quantile count). The
    // configs cap these in the low hundreds, so the value is exact in f32; a
    // non-finite or zero spacing is rejected by assertion before use.
    #[allow(clippy::cast_precision_loss)]
    pub fn quantile_taus<B: Backend>(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        let n = self.num_quantiles;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5) / n as f32).collect();
        Tensor::from_data(TensorData::new(data, vec![n]), device)
    }
}

impl Default for QrDqnTrainingConfig {
    /// Returns defaults consistent with Dabney et al. 2018 QR-DQN
    /// reference hyperparameters.
    ///
    /// [`target_update`](Self::target_update) is a τ = 0.005 Polyak step on
    /// **every** gradient update. That is bit-for-bit the pre-[`TargetUpdate`]
    /// behaviour: the old `tau = 0.005` soft update ran ungated inside every
    /// learn step, which in gradient-update units is exactly `every = 1`. The
    /// old `target_update_frequency = 10_000` is deliberately not carried over
    /// — it was inert under `tau > 0` and, read as a cadence under the unified
    /// rule, would collapse the Polyak schedule 10 000×
    /// (ADR 0059 §Consequences).
    fn default() -> Self {
        Self {
            batch_size: 32,
            gamma: 0.99,
            learning_rate: 0.001,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            target_update: TargetUpdate::polyak(0.005, 1),
            steps_per_episode: 1000,
            replay_buffer_capacity: 10_000,
            learning_starts: 1_000,
            train_frequency: 4,
            num_quantiles: 200,
            kappa: 1.0,
            clip_grad: Some(GradientClippingConfig::Value(100.0)),
            optimizer: AdamConfig::new(),
            prioritized_replay: None,
        }
    }
}

impl Validate for QrDqnTrainingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "QrDqnTrainingConfig";
        config::nonzero(C, "batch_size", self.batch_size)?;
        config::in_range(C, "gamma", 0.0, 1.0, self.gamma)?;
        config::positive(C, "learning_rate", self.learning_rate)?;
        config::in_range(C, "epsilon_start", 0.0, 1.0, self.epsilon_start)?;
        config::in_range(C, "epsilon_end", 0.0, 1.0, self.epsilon_end)?;
        config::in_range(C, "epsilon_decay", 0.0, 1.0, self.epsilon_decay)?;
        // `replay_buffer_capacity` carries no `config::` lines of its own: the
        // floor-and-ceiling predicate is single-sourced in
        // `UniformReplayConfig::validate`, which is the same check the buffer
        // this field feeds runs at construction. The error is relabelled to
        // name *this* config's field, so what the caller sees is byte-identical
        // to the two hand-written lines it replaces — only `kind` passes
        // through, and only the predicate's home moved (ADR 0050).
        UniformReplayConfig {
            capacity: self.replay_buffer_capacity,
        }
        .validate()
        .map_err(|e| ConfigError {
            config: C,
            field: "replay_buffer_capacity",
            ..e
        })?;
        config::nonzero(C, "train_frequency", self.train_frequency)?;
        config::nonzero(C, "steps_per_episode", self.steps_per_episode)?;
        config::nonzero(C, "num_quantiles", self.num_quantiles)?;
        // κ is the *divisor* of Dabney et al. (2018) Eq. (10), so the whole
        // open interval is required — not the closed `[0, ∞]` an `in_range`
        // would accept. κ = 0 makes the loss evaluate `0/0` → NaN, reported as
        // `NotPositive`. This half also catches every non-finite κ — `NaN`,
        // `−∞`, and `+∞` alike — which `config::positive` reports as
        // `NotFinite` (issue #353), a distinct kind from κ = 0's.
        config::positive(C, "kappa", f64::from(self.kappa))?;
        // The upper bound is not a range check but the precondition of
        // `huber`'s eagerly-evaluated linear branch: it computes
        // `(|u| − 0.5·κ)·κ` for *every* element and then discards the
        // small-residual ones by multiplying by a 0.0 mask. Once `0.5·κ²`
        // overflows f32 to `−∞`, that discard is `0 · (−∞) = NaN`, so a
        // huge-but-finite κ poisons the whole loss even though the mask
        // selects the quadratic branch everywhere. Expressing the guard as
        // "the branch must not overflow" rather than a hardcoded threshold
        // keeps it correct if `huber` changes; the resulting cutoff is
        // `√(2 · f32::MAX) ≈ 2.6087635e19`, measured to be exact to the ULP
        // (see `rejects_overflowing_kappa`). Its remaining job is the
        // **huge-but-finite** κ: `+∞` is now rejected one line above by
        // `config::positive` as `NotFinite`, but a finite κ whose `0.5·κ²`
        // overflows f32 passes every generic predicate and is caught only
        // here. Do not remove this guard.
        if !(0.5 * self.kappa * self.kappa).is_finite() {
            return Err(ConfigError {
                config: C,
                field: "kappa",
                kind: ConstraintKind::Custom(
                    "the Huber threshold is too large: the quantile Huber \
                     loss evaluates its linear branch `(|u| − 0.5·κ)·κ` for \
                     every element before masking, so `0.5·κ²` must stay \
                     finite in f32 (κ ≤ √(2·f32::MAX) ≈ 2.6e19) or the \
                     masked-out elements become `0 · (−∞)` = NaN",
                ),
            });
        }
        if let Some(per) = &self.prioritized_replay {
            per.validate()?;
        }
        // `target_update` carries no check here, deliberately: `TargetUpdate`
        // is valid by construction (ADR 0027 §3 — a validated newtype *removes*
        // its paired `config::` line). Its `PolyakTau` excludes τ = 0.0 and its
        // `NonZeroUsize` cadence excludes 0, so the frozen target the old
        // cross-field check rejected is now unrepresentable rather than merely
        // rejected — including through `..Default::default()` struct update,
        // which `validate` never saw.
        Ok(())
    }
}

/// Fluent builder for [`QrDqnTrainingConfig`]. All unset fields fall back to
/// [`QrDqnTrainingConfig::default`].
#[derive(Debug)]
pub struct QrDqnTrainingConfigBuilder {
    config: QrDqnTrainingConfig,
}

impl Default for QrDqnTrainingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QrDqnTrainingConfigBuilder {
    /// Creates a builder pre-populated with [`QrDqnTrainingConfig::default`] values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: QrDqnTrainingConfig::default(),
        }
    }

    /// Sets [`QrDqnTrainingConfig::batch_size`].
    #[must_use]
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Sets [`QrDqnTrainingConfig::gamma`] (discount factor γ).
    #[must_use]
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.config.gamma = gamma;
        self
    }

    /// Sets [`QrDqnTrainingConfig::learning_rate`].
    #[must_use]
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Sets [`QrDqnTrainingConfig::epsilon_start`] — initial ε for ε-greedy.
    #[must_use]
    pub fn epsilon_start(mut self, epsilon_start: f64) -> Self {
        self.config.epsilon_start = epsilon_start;
        self
    }

    /// Sets [`QrDqnTrainingConfig::epsilon_end`] — floor ε for ε-greedy.
    #[must_use]
    pub fn epsilon_end(mut self, epsilon_end: f64) -> Self {
        self.config.epsilon_end = epsilon_end;
        self
    }

    /// Sets [`QrDqnTrainingConfig::epsilon_decay`] — per-step multiplicative decay.
    #[must_use]
    pub fn epsilon_decay(mut self, epsilon_decay: f64) -> Self {
        self.config.epsilon_decay = epsilon_decay;
        self
    }

    /// Sets [`QrDqnTrainingConfig::target_update`] — cadence and τ together.
    ///
    /// One setter, because there is one mechanism (ADR 0058). The cadence is in
    /// **gradient updates**, unlike the env-step
    /// [`train_frequency`](Self::train_frequency) beside it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_config::QrDqnTrainingConfigBuilder;
    /// use rlevo_reinforcement_learning::target::TargetUpdate;
    ///
    /// let cfg = QrDqnTrainingConfigBuilder::new()
    ///     .target_update(TargetUpdate::polyak(0.01, 2))
    ///     .build()
    ///     .expect("valid config");
    /// assert_eq!(cfg.target_update.every(), 2);
    /// ```
    #[must_use]
    pub fn target_update(mut self, target_update: TargetUpdate) -> Self {
        self.config.target_update = target_update;
        self
    }

    /// Sets [`QrDqnTrainingConfig::steps_per_episode`].
    #[must_use]
    pub fn steps_per_episode(mut self, steps: usize) -> Self {
        self.config.steps_per_episode = steps;
        self
    }

    /// Sets [`QrDqnTrainingConfig::replay_buffer_capacity`].
    #[must_use]
    pub fn replay_buffer_capacity(mut self, capacity: usize) -> Self {
        self.config.replay_buffer_capacity = capacity;
        self
    }

    /// Sets [`QrDqnTrainingConfig::learning_starts`].
    #[must_use]
    pub fn learning_starts(mut self, learning_starts: usize) -> Self {
        self.config.learning_starts = learning_starts;
        self
    }

    /// Sets [`QrDqnTrainingConfig::train_frequency`].
    #[must_use]
    pub fn train_frequency(mut self, train_frequency: usize) -> Self {
        self.config.train_frequency = train_frequency;
        self
    }

    /// Sets [`QrDqnTrainingConfig::num_quantiles`] (`N` in Dabney et al. 2018).
    #[must_use]
    pub fn num_quantiles(mut self, num_quantiles: usize) -> Self {
        self.config.num_quantiles = num_quantiles;
        self
    }

    /// Sets [`QrDqnTrainingConfig::kappa`] (Huber loss threshold κ).
    ///
    /// κ must be strictly positive and small enough that `$0.5 \cdot \kappa^2$` stays
    /// finite in `f32`; [`build`](Self::build) rejects anything else, because
    /// the quantile Huber loss divides by κ and evaluates a `$-0.5\kappa^2$` branch
    /// eagerly. See [`QrDqnTrainingConfig::kappa`] for the derivation and for
    /// why κ = 0 is not QR-DQN-0.
    #[must_use]
    pub fn kappa(mut self, kappa: f32) -> Self {
        self.config.kappa = kappa;
        self
    }

    /// Sets [`QrDqnTrainingConfig::clip_grad`].
    ///
    /// Pass `None` to disable gradient clipping entirely.
    #[must_use]
    pub fn clip_grad(mut self, config: Option<GradientClippingConfig>) -> Self {
        self.config.clip_grad = config;
        self
    }

    /// Replaces the default [`AdamConfig`] with a custom optimizer configuration.
    #[must_use]
    pub fn optimizer(mut self, optimizer: AdamConfig) -> Self {
        self.config.optimizer = optimizer;
        self
    }

    /// Enables prioritized experience replay with the given settings.
    ///
    /// The QR-DQN priority (quantile Huber loss) is an uncited extrapolation —
    /// see [`QrDqnTrainingConfig::prioritized_replay`]. Leave unset for uniform
    /// replay.
    #[must_use]
    pub fn prioritized_replay(mut self, settings: PrioritizedReplaySettings) -> Self {
        self.config.prioritized_replay = Some(settings);
        self
    }

    /// Consumes the builder and returns the assembled [`QrDqnTrainingConfig`].
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if the assembled config violates any invariant
    /// checked by [`QrDqnTrainingConfig::validate`] — e.g. zero
    /// `num_quantiles`, or a `kappa` that is not strictly positive or is large
    /// enough to overflow the Huber loss's linear branch (see
    /// [`QrDqnTrainingConfig::kappa`]).
    pub fn build(self) -> Result<QrDqnTrainingConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MAX_BUFFER_CAPACITY;
    use burn::backend::Flex;

    type TestBackend = Flex;

    #[test]
    fn test_qrdqn_config_defaults_are_200_quantiles_with_kappa_1() {
        let cfg = QrDqnTrainingConfig::default();
        assert_eq!(cfg.num_quantiles, 200);
        assert!((cfg.kappa - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_qrdqn_config_builder_round_trips_distributional_fields() {
        let cfg = QrDqnTrainingConfigBuilder::new()
            .num_quantiles(51)
            .kappa(0.5)
            .batch_size(128)
            .build()
            .expect("valid config");
        assert_eq!(cfg.num_quantiles, 51);
        assert!((cfg.kappa - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.batch_size, 128);
    }

    #[test]
    fn test_qrdqn_config_default_config_is_valid() {
        assert!(QrDqnTrainingConfig::default().validate().is_ok());
    }

    /// `replay_buffer_capacity` had a floor and no ceiling, so a
    /// `usize::MAX` sentinel — a unit slip, a deserialized garbage field —
    /// validated and went straight to the buffer constructor, where it aborts
    /// on a failed allocation or wraps the `SumTree` arithmetic. The rejection
    /// must be a **recoverable `Err`**, which is why this asserts on the
    /// returned error rather than using `#[should_panic]`: a panic here would
    /// be the bug, not the fix.
    #[test]
    fn test_qrdqn_config_rejects_replay_buffer_capacity_above_ceiling() {
        let err = QrDqnTrainingConfigBuilder::new()
            .replay_buffer_capacity(usize::MAX)
            .build()
            .expect_err("usize::MAX capacity must be rejected, not allocated");
        assert_eq!(err.field, "replay_buffer_capacity");
        let max = u64::try_from(MAX_BUFFER_CAPACITY).expect("the ceiling fits in u64");
        let got = u64::try_from(usize::MAX).expect("usize::MAX fits in u64 on this target");
        assert_eq!(err.kind, ConstraintKind::TooLarge { max, got });
    }

    /// The boundary is inclusive: the ceiling itself is a legal (if
    /// unallocatable) capacity, so the guard cannot be off by one.
    #[test]
    fn test_qrdqn_config_accepts_replay_buffer_capacity_at_ceiling() {
        let cfg = QrDqnTrainingConfigBuilder::new()
            .replay_buffer_capacity(MAX_BUFFER_CAPACITY)
            .build()
            .expect("the ceiling itself must validate");
        assert_eq!(cfg.replay_buffer_capacity, MAX_BUFFER_CAPACITY);
    }

    #[test]
    fn test_qrdqn_config_rejects_zero_quantiles() {
        let err = QrDqnTrainingConfigBuilder::new()
            .num_quantiles(0)
            .build()
            .unwrap_err();
        assert_eq!(err.field, "num_quantiles");
    }

    /// The default is behaviour-preserving: the old `tau = 0.005` soft update
    /// ran inside *every* learn step with the hard path inert, which in
    /// gradient-update units is `every = 1` (ADR 0059 §Consequences).
    #[test]
    fn test_qrdqn_config_default_target_update_is_polyak_every_gradient_update() {
        let cfg = QrDqnTrainingConfig::default();
        assert_eq!(cfg.target_update, TargetUpdate::polyak(0.005, 1));
        assert_eq!(cfg.target_update.every(), 1);
    }

    /// κ = 0 makes Dabney et al. (2018) Eq. (10) evaluate `0/0`: the loss
    /// divides by κ, so the config boundary — not the loss — must reject it.
    #[test]
    fn test_qrdqn_config_rejects_zero_kappa() {
        let err = QrDqnTrainingConfigBuilder::new()
            .kappa(0.0)
            .build()
            .expect_err("kappa = 0 divides by zero in the quantile Huber loss");
        assert_eq!(err.field, "kappa", "the error must name the kappa field");
        assert_eq!(
            err.kind,
            ConstraintKind::NotPositive { got: 0.0 },
            "κ = 0 must be rejected by the strict-positivity half of the guard, \
             not incidentally by the overflow half"
        );
    }

    /// Regression guard: a negative Huber threshold was already rejected and
    /// must stay rejected.
    #[test]
    fn test_qrdqn_config_rejects_negative_kappa() {
        let err = QrDqnTrainingConfigBuilder::new()
            .kappa(-1.0)
            .build()
            .expect_err("a negative Huber threshold is meaningless");
        assert_eq!(err.field, "kappa", "the error must name the kappa field");
        assert_eq!(
            err.kind,
            ConstraintKind::NotPositive { got: -1.0 },
            "a negative κ must be rejected as NotPositive"
        );
    }

    /// `NaN` and `$-\infty$` are caught by `config::positive`, but as `NotFinite`
    /// (issue #353) — a *different* kind from κ = 0's `NotPositive`, because
    /// "not a number" and "not above zero" are different complaints. Asserted
    /// per value rather than in one loop with `$+\infty$`: `$+\infty$` is also `NotFinite`
    /// now, but it reaches that verdict having previously been the overflow
    /// guard's job (see `rejects_overflowing_kappa`).
    #[test]
    fn test_qrdqn_config_rejects_non_finite_kappa() {
        for bad in [f32::NAN, f32::NEG_INFINITY] {
            let err = QrDqnTrainingConfigBuilder::new()
                .kappa(bad)
                .build()
                .expect_err("kappa must be finite");
            assert_eq!(err.field, "kappa", "the error must name the kappa field");
            match err.kind {
                ConstraintKind::NotFinite { got } => assert_eq!(
                    got.is_nan(),
                    bad.is_nan(),
                    "NotFinite must carry the offending value verbatim, got {got}"
                ),
                other => panic!("{bad} must be rejected as NotFinite, got {other:?}"),
            }
        }
    }

    /// A huge-but-*finite* κ is as fatal as κ = 0 and was the hole the first
    /// `is_finite` guard left open. `huber` evaluates its linear branch
    /// `$(|u| - 0.5\cdot\kappa)\cdot\kappa$` for every element before masking, so once `$0.5\cdot\kappa^2$`
    /// overflows f32 to `$-\infty$` the masked-out elements become `0 · (−∞)` = NaN,
    /// even though the mask selects the quadratic branch everywhere.
    ///
    /// The rejected values below bracket the measured NaN boundary: on the
    /// `Flex` backend `quantile_huber_loss_per_sample` returns a finite loss at
    /// κ = `2.608_763_5e19` and `NaN` at the very next `f32`,
    /// `2.608_763_7e19` — i.e. the guard `!(0.5·κ²).is_finite()` is exact to
    /// the ULP, not merely conservative.
    #[test]
    fn test_qrdqn_config_rejects_overflowing_kappa() {
        // Bit-search the exact f32 boundary rather than transcribing a
        // literal: `√(2 · f32::MAX)` is not representable by a computation
        // that stays in f32, and a transcribed decimal could land on either
        // side of the true cutoff as the rounding of the literal drifts.
        // f32 bit patterns of positive finite values are monotonic, so a
        // plain binary search over `to_bits` finds it.
        let (mut lo, mut hi) = (1.0_f32.to_bits(), f32::MAX.to_bits());
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            let k = f32::from_bits(mid);
            if (0.5 * k * k).is_finite() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let last_ok = f32::from_bits(lo);
        assert!(
            (2.608_763e19..2.608_764e19).contains(&last_ok),
            "the cutoff must be √(2·f32::MAX) ≈ 2.6087635e19, got {last_ok:e}"
        );

        for bad in [f32::from_bits(lo + 1), 2.7e19, 1e20, f32::MAX] {
            let err = QrDqnTrainingConfigBuilder::new()
                .kappa(bad)
                .build()
                .expect_err("kappa whose 0.5·κ² overflows f32 makes the loss NaN");
            assert_eq!(err.field, "kappa", "the error must name the kappa field");
            assert!(
                matches!(err.kind, ConstraintKind::Custom(_)),
                "{bad} overflows the Huber linear branch and must be rejected by \
                 the Custom overflow guard, not as NotPositive; got {:?}",
                err.kind
            );
        }

        // `+∞` used to be listed above, because `config::positive` waved it
        // through (`f64::INFINITY > 0.0`) and this overflow guard was the only
        // thing that stopped it. Issue #353 moved that verdict one line
        // earlier: `+∞` is not a config *value* at all, so it is now
        // `NotFinite`. Still rejected, still at the same chokepoint — but the
        // overflow guard is no longer load-bearing for it, and the four
        // huge-but-*finite* κ above are the cases only it catches.
        let err = QrDqnTrainingConfigBuilder::new()
            .kappa(f32::INFINITY)
            .build()
            .expect_err("an infinite Huber threshold is not a config value");
        assert_eq!(err.field, "kappa", "the error must name the kappa field");
        assert_eq!(
            err.kind,
            ConstraintKind::NotFinite { got: f64::INFINITY },
            "+∞ must be rejected as NotFinite by config::positive, ahead of the \
             overflow guard"
        );

        // The bound must not be over-tight: κ far above any sane setting, but
        // whose linear branch still fits in f32, stays accepted.
        for good in [1.0_f32, 1e9, 1e18, last_ok] {
            QrDqnTrainingConfigBuilder::new()
                .kappa(good)
                .build()
                .expect("a κ whose 0.5·κ² is finite must validate");
        }
    }

    /// Replaces `rejects_frozen_target_when_tau_and_frequency_are_both_zero`.
    /// The frozen-target state is no longer *rejected* by `validate` — it is
    /// unrepresentable, so the assertion moves to the constructor (ADR 0058
    /// §Consequences). That is strictly stronger: the old check could be
    /// bypassed by struct-update syntax on the two `pub` scalar fields.
    #[test]
    fn test_qrdqn_config_frozen_target_is_unreachable_through_the_type() {
        assert!(
            TargetUpdate::try_polyak(0.0, 1).is_err(),
            "τ = 0 fires on schedule and moves nothing — a frozen target"
        );
        assert!(
            TargetUpdate::try_polyak(0.005, 0).is_err(),
            "a cadence of 0 never fires — a frozen target"
        );
        assert!(
            TargetUpdate::try_polyak(0.0, 0).is_err(),
            "the pair the deleted cross-field check rejected"
        );
    }

    /// Replaces the three `accepts_*_configuration` tests, which enumerated the
    /// legal (τ, frequency) combinations of the two-mechanism era. There is one
    /// mechanism now, so the property is simply: anything constructible is
    /// valid — hard included, since hard is `$\tau = 1.0$` on the same rule.
    #[test]
    fn test_qrdqn_config_every_constructible_target_update_yields_a_valid_config() {
        for rule in [
            TargetUpdate::polyak(0.005, 1),
            TargetUpdate::polyak(0.01, 100),
            TargetUpdate::hard(1),
            TargetUpdate::hard(500),
        ] {
            let cfg = QrDqnTrainingConfigBuilder::new()
                .target_update(rule)
                .build()
                .expect("a constructible TargetUpdate is always a valid config");
            assert_eq!(cfg.target_update, rule);
        }
    }

    #[test]
    fn test_qrdqn_config_taus_midpoints_match_dabney_eq_9_for_n_4() {
        // N = 4 ⇒ τ_i = (i + 0.5) / 4 ⇒ {0.125, 0.375, 0.625, 0.875}.
        let device: <TestBackend as burn::tensor::backend::BackendTypes>::Device =
            Default::default();
        let cfg = QrDqnTrainingConfigBuilder::new()
            .num_quantiles(4)
            .build()
            .expect("valid config");
        let taus: Tensor<TestBackend, 1> = cfg.quantile_taus::<TestBackend>(&device);
        let v: Vec<f32> = taus
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        let expected = [0.125_f32, 0.375, 0.625, 0.875];
        for (got, want) in v.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }
}
