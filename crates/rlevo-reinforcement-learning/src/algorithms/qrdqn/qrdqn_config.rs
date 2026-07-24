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
use rlevo_core::config::{self, ConfigError, Validate};

use crate::replay::PrioritizedReplaySettings;
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
    /// moves when it does — `target ← (1 − τ) · target + τ · policy`. A
    /// periodic hard copy is not a second mechanism but the degenerate
    /// `τ = 1.0`, spelled [`TargetUpdate::hard`]. The update is applied inside
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
    pub replay_buffer_capacity: usize,

    /// Number of env steps collected before the first gradient update.
    pub learning_starts: usize,

    /// Period, in env steps, between gradient updates.
    pub train_frequency: usize,

    /// Number of quantiles `N` used to represent the return distribution.
    ///
    /// The default is 200 — the value from Dabney et al. (2018). Quantile
    /// midpoints `τ_i = (i + 0.5) / N` are implied by this field alone.
    pub num_quantiles: usize,

    /// Huber threshold `κ` used by the quantile Huber loss. Values below
    /// `κ` are treated quadratically, above `κ` linearly. Default `1.0`.
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
    /// Builds the quantile-midpoint tensor `[(i + 0.5) / N]_{i=0..N-1}` on
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
        config::nonzero(C, "replay_buffer_capacity", self.replay_buffer_capacity)?;
        config::nonzero(C, "train_frequency", self.train_frequency)?;
        config::nonzero(C, "steps_per_episode", self.steps_per_episode)?;
        config::nonzero(C, "num_quantiles", self.num_quantiles)?;
        config::in_range(C, "kappa", 0.0, f64::INFINITY, f64::from(self.kappa))?;
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
    /// checked by [`QrDqnTrainingConfig::validate`] (e.g. zero `num_quantiles`
    /// or a negative `kappa`).
    pub fn build(self) -> Result<QrDqnTrainingConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;

    type B = Flex;

    #[test]
    fn defaults_are_200_quantiles_with_kappa_1() {
        let cfg = QrDqnTrainingConfig::default();
        assert_eq!(cfg.num_quantiles, 200);
        assert!((cfg.kappa - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn builder_round_trips_distributional_fields() {
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
    fn default_config_is_valid() {
        assert!(QrDqnTrainingConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_quantiles() {
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
    fn default_target_update_is_polyak_every_gradient_update() {
        let cfg = QrDqnTrainingConfig::default();
        assert_eq!(cfg.target_update, TargetUpdate::polyak(0.005, 1));
        assert_eq!(cfg.target_update.every(), 1);
    }

    /// Replaces `rejects_frozen_target_when_tau_and_frequency_are_both_zero`.
    /// The frozen-target state is no longer *rejected* by `validate` — it is
    /// unrepresentable, so the assertion moves to the constructor (ADR 0058
    /// §Consequences). That is strictly stronger: the old check could be
    /// bypassed by struct-update syntax on the two `pub` scalar fields.
    #[test]
    fn frozen_target_is_unreachable_through_the_type() {
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
    /// valid — hard included, since hard is `τ = 1.0` on the same rule.
    #[test]
    fn every_constructible_target_update_yields_a_valid_config() {
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
    fn taus_midpoints_match_dabney_eq_9_for_n_4() {
        // N = 4 ⇒ τ_i = (i + 0.5) / 4 ⇒ {0.125, 0.375, 0.625, 0.875}.
        let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let cfg = QrDqnTrainingConfigBuilder::new()
            .num_quantiles(4)
            .build()
            .expect("valid config");
        let taus: Tensor<B, 1> = cfg.quantile_taus::<B>(&device);
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
