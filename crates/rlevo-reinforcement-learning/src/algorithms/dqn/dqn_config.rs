//! Hyperparameter configuration for the DQN algorithm.

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use rlevo_core::config::{self, ConfigError, Validate};

use crate::replay::PrioritizedReplaySettings;
use crate::target::TargetUpdate;

/// Configuration structure for training a Deep Q-Network (DQN).
///
/// This struct holds all hyperparameters and configuration settings required
/// to initialize and train a DQN agent, including learning rates,
/// epsilon-greedy parameters, and optimization settings.
#[derive(Clone, Debug)]
pub struct DqnTrainingConfig {
    /// The number of samples processed before the model is updated.
    pub batch_size: usize,

    /// The discount factor ($\gamma$).
    ///
    /// A value between 0 and 1 that balances the importance of immediate rewards
    /// versus future rewards. A value close to 0 makes the agent short-sighted,
    /// while a value close to 1 makes it strive for long-term high reward.
    pub gamma: f64,

    /// The learning rate used by the optimizer.
    pub learning_rate: f64,

    /// The starting value for epsilon in the epsilon-greedy exploration strategy.
    ///
    /// Represents the initial probability of choosing a random action over the greedy action.
    pub epsilon_start: f64,

    /// The minimum value that epsilon can decay to.
    ///
    /// Ensures that there is always a small chance of exploration.
    pub epsilon_end: f64,

    /// The decay rate for epsilon.
    ///
    /// Multiplicative factor applied to epsilon after each step or episode to reduce exploration over time.
    pub epsilon_decay: f64,

    /// How the target network tracks the policy network: one cadence, one τ.
    ///
    /// [`TargetUpdate`] is a single mechanism, not two (ADR 0058). Its cadence
    /// [`every`](TargetUpdate::every) decides *when* an update fires; its
    /// coefficient [`tau`](TargetUpdate::tau) decides *how far* the target
    /// moves when it does — `target ← (1 − τ)·target + τ·policy`. A periodic
    /// hard copy is not a second mechanism but the degenerate `τ = 1.0`,
    /// spelled [`TargetUpdate::hard`]. The update is applied inside
    /// [`DqnAgent::learn_step`] and nowhere else, so no train loop can forget
    /// to drive it.
    ///
    /// # The cadence counts gradient updates — its neighbours count env steps
    ///
    /// [`every`](TargetUpdate::every) is a **gradient (optimizer) update**
    /// count (ADR 0059), matching Mnih et al. (2015), whose `C` is "measured in
    /// the number of *parameter updates*" (Extended Data Table 1), and Haarnoja
    /// et al. (2018a), whose target update sits inside "for each gradient step
    /// do". [`learning_starts`](Self::learning_starts) and
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
    /// [`DqnAgent::learn_step`]: crate::algorithms::dqn::dqn_agent::DqnAgent::learn_step
    pub target_update: TargetUpdate,

    /// The maximum number of steps allowed per episode.
    pub steps_per_episode: usize,

    /// The maximum number of transitions to store in the replay buffer.
    pub replay_buffer_capacity: usize,

    /// Number of environment steps collected before learning starts.
    ///
    /// Acts as a warm-up period that fills the replay buffer with diverse
    /// transitions before the first gradient update, stabilising early
    /// training.
    pub learning_starts: usize,

    /// How often (in environment steps) a learning update is performed.
    ///
    /// `train_frequency = 4` means one gradient step every four env steps,
    /// matching the Nature-DQN setting.
    pub train_frequency: usize,

    /// If `true`, compute bootstrap targets using Double-DQN
    /// (`a* = argmax_a Q_online(s', a)`, then `y = Q_target(s', a*)`).
    ///
    /// Leave `false` for vanilla DQN.
    pub double_q: bool,

    /// Configuration for gradient clipping.
    ///
    /// Prevents exploding gradients by scaling the gradient vector if its norm exceeds a threshold.
    /// Set to `None` to disable clipping.
    pub clip_grad: Option<GradientClippingConfig>,

    /// Configuration for the optimizer (e.g., Adam).
    ///
    /// This defines the optimization algorithm used to update the network weights.
    pub optimizer: AdamConfig,

    /// Opt-in prioritized experience replay (Schaul et al. 2016).
    ///
    /// `None` (the default) uses uniform replay — the pre-PER behaviour, byte
    /// for byte. `Some` enables PER: transitions are drawn in proportion to
    /// `(|δ| + ε)^α`, and the per-sample loss is scaled by max-normalized
    /// importance weights annealed over `beta_anneal_steps`. The priority signal
    /// is the absolute Huber TD error `|δ|` (Schaul §3.3, direct).
    ///
    /// Buffer capacity comes from [`replay_buffer_capacity`](Self::replay_buffer_capacity);
    /// the remaining knobs (α, ε, β schedule) live on
    /// [`PrioritizedReplaySettings`]. Rainbow's ablation puts prioritized replay
    /// among the two most crucial of its seven components on the value-based
    /// side, which is why it is offered here (ADR 0050 §Context).
    pub prioritized_replay: Option<PrioritizedReplaySettings>,
}

impl Default for DqnTrainingConfig {
    /// Returns sensible defaults suited to small discrete-action environments.
    ///
    /// Key values: `batch_size = 32`, `gamma = 0.99`,
    /// `target_update = polyak(0.005, 1)`, `learning_rate = 1e-3`,
    /// `epsilon_start = 1.0` decaying to `0.01` at rate `0.995`,
    /// `replay_buffer_capacity = 10_000`, `learning_starts = 1_000`,
    /// `train_frequency = 4`, `double_q = false`, gradient clipping at norm
    /// 100.
    ///
    /// The target rule is a τ = 0.005 Polyak step on **every** gradient update.
    /// That is bit-for-bit the pre-[`TargetUpdate`] behaviour: the old
    /// `tau = 0.005` soft update ran ungated inside every learn step, which in
    /// gradient-update units is exactly `every = 1`. The old
    /// `target_update_frequency = 10_000` is deliberately not carried over — it
    /// was inert under `tau > 0` and, read as a cadence under the unified rule,
    /// would collapse the Polyak schedule 10 000× (ADR 0059 §Consequences).
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
            replay_buffer_capacity: 10000,
            learning_starts: 1000,
            train_frequency: 4,
            double_q: false,
            clip_grad: Some(GradientClippingConfig::Value(100.0)),
            optimizer: AdamConfig::new(),
            prioritized_replay: None,
        }
    }
}

impl Validate for DqnTrainingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "DqnTrainingConfig";
        config::nonzero(C, "batch_size", self.batch_size)?;
        config::in_range(C, "gamma", 0.0, 1.0, self.gamma)?;
        config::positive(C, "learning_rate", self.learning_rate)?;
        config::in_range(C, "epsilon_start", 0.0, 1.0, self.epsilon_start)?;
        config::in_range(C, "epsilon_end", 0.0, 1.0, self.epsilon_end)?;
        config::in_range(C, "epsilon_decay", 0.0, 1.0, self.epsilon_decay)?;
        config::nonzero(C, "replay_buffer_capacity", self.replay_buffer_capacity)?;
        config::nonzero(C, "train_frequency", self.train_frequency)?;
        config::nonzero(C, "steps_per_episode", self.steps_per_episode)?;
        if let Some(per) = &self.prioritized_replay {
            per.validate()?;
        }
        // `target_update` carries no check here, deliberately: `TargetUpdate`
        // is valid by construction (ADR 0027 §3 — a validated newtype *removes*
        // its paired `config::` line). Its `PolyakTau` excludes τ = 0.0 and its
        // `NonZeroUsize` cadence excludes 0, so the frozen-target combination
        // the old cross-field check rejected is now unrepresentable rather than
        // merely rejected — including through `..Default::default()` struct
        // update, which `validate` never saw.
        Ok(())
    }
}

/// Builder for [`DqnTrainingConfig`] with fluent setters.
///
/// All unset fields default to the values from [`DqnTrainingConfig::default`].
///
/// # Examples
///
/// ```rust
/// use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
///
/// // Default configuration.
/// let cfg = DqnTrainingConfigBuilder::new().build().expect("valid config");
///
/// // Custom learning rate and batch size.
/// let cfg = DqnTrainingConfigBuilder::new()
///     .learning_rate(0.0005)
///     .batch_size(64)
///     .build()
///     .expect("valid config");
/// ```
#[derive(Debug)]
pub struct DqnTrainingConfigBuilder {
    config: DqnTrainingConfig,
}

impl Default for DqnTrainingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DqnTrainingConfigBuilder {
    /// Creates a new builder initialized with default configuration values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DqnTrainingConfig::default(),
        }
    }

    /// Sets the batch size.
    #[must_use]
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Sets the discount factor (gamma).
    #[must_use]
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.config.gamma = gamma;
        self
    }

    /// Sets the learning rate.
    #[must_use]
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Sets the starting epsilon value for exploration.
    #[must_use]
    pub fn epsilon_start(mut self, epsilon_start: f64) -> Self {
        self.config.epsilon_start = epsilon_start;
        self
    }

    /// Sets the minimum epsilon value.
    #[must_use]
    pub fn epsilon_end(mut self, epsilon_end: f64) -> Self {
        self.config.epsilon_end = epsilon_end;
        self
    }

    /// Sets the epsilon decay rate.
    #[must_use]
    pub fn epsilon_decay(mut self, epsilon_decay: f64) -> Self {
        self.config.epsilon_decay = epsilon_decay;
        self
    }

    /// Sets the target-network update rule — cadence and τ together.
    ///
    /// One setter, because there is one mechanism (ADR 0058). The cadence is in
    /// **gradient updates**, unlike the env-step
    /// [`train_frequency`](Self::train_frequency) beside it; see
    /// [`DqnTrainingConfig::target_update`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
    /// use rlevo_reinforcement_learning::target::TargetUpdate;
    ///
    /// // Polyak, every gradient update (the shipped default).
    /// let soft = DqnTrainingConfigBuilder::new()
    ///     .target_update(TargetUpdate::polyak(0.005, 1))
    ///     .build()
    ///     .expect("valid config");
    ///
    /// // Nature-DQN style: a full copy every 10 000 gradient updates.
    /// let hard = DqnTrainingConfigBuilder::new()
    ///     .target_update(TargetUpdate::hard(10_000))
    ///     .build()
    ///     .expect("valid config");
    /// ```
    ///
    /// That `10_000` is the Atari-derived figure: at the default
    /// `train_frequency: 4` it is about 40 000 environment steps, so a
    /// classic-control run wants a much smaller cadence (issue #337).
    #[must_use]
    pub fn target_update(mut self, target_update: TargetUpdate) -> Self {
        self.config.target_update = target_update;
        self
    }

    /// Sets the maximum steps per episode.
    #[must_use]
    pub fn steps_per_episode(mut self, steps: usize) -> Self {
        self.config.steps_per_episode = steps;
        self
    }

    /// Sets the capacity of the replay buffer.
    #[must_use]
    pub fn replay_buffer_capacity(mut self, capacity: usize) -> Self {
        self.config.replay_buffer_capacity = capacity;
        self
    }

    /// Sets the number of warm-up steps before learning begins.
    #[must_use]
    pub fn learning_starts(mut self, learning_starts: usize) -> Self {
        self.config.learning_starts = learning_starts;
        self
    }

    /// Sets how often a learning update runs, in environment steps.
    #[must_use]
    pub fn train_frequency(mut self, train_frequency: usize) -> Self {
        self.config.train_frequency = train_frequency;
        self
    }

    /// Enables or disables Double-DQN bootstrap targets.
    #[must_use]
    pub fn double_q(mut self, double_q: bool) -> Self {
        self.config.double_q = double_q;
        self
    }

    /// Sets the gradient clipping configuration.
    #[must_use]
    pub fn clip_grad(mut self, config: Option<GradientClippingConfig>) -> Self {
        self.config.clip_grad = config;
        self
    }

    /// Sets the optimizer configuration (e.g., specific Adam beta values).
    #[must_use]
    pub fn optimizer(mut self, optimizer: AdamConfig) -> Self {
        self.config.optimizer = optimizer;
        self
    }

    /// Enables prioritized experience replay with the given settings.
    ///
    /// Pass [`PrioritizedReplaySettings::default`] for Schaul's proportional
    /// defaults (α = 0.6, β 0.4 → 1.0). Leave unset for uniform replay.
    #[must_use]
    pub fn prioritized_replay(mut self, settings: PrioritizedReplaySettings) -> Self {
        self.config.prioritized_replay = Some(settings);
        self
    }

    /// Consumes the builder and returns the final `DqnTrainingConfig`.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if the assembled config violates any invariant
    /// checked by [`DqnTrainingConfig::validate`] (e.g. a zero `batch_size` or a
    /// `gamma` outside `[0, 1]`).
    pub fn build(self) -> Result<DqnTrainingConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        assert!(DqnTrainingConfig::default().validate().is_ok());
    }

    /// The default is the behaviour-preserving one: the pre-[`TargetUpdate`]
    /// config ran a `tau = 0.005` Polyak step inside *every* learn step with
    /// the hard path inert, which in gradient-update units is `every = 1`.
    /// Pinned literally, because ADR 0059 §Consequences turns on this pair
    /// being `(0.005, 1)` and not the old `(0.005, 10_000)` transcription.
    #[test]
    fn default_target_update_is_polyak_every_gradient_update() {
        let cfg = DqnTrainingConfig::default();
        assert_eq!(cfg.target_update, TargetUpdate::polyak(0.005, 1));
        assert!(!cfg.target_update.is_hard());
        assert_eq!(cfg.target_update.every(), 1);
    }

    /// Replaces `rejects_frozen_target_network`, which asserted `validate`
    /// rejected `tau = 0.0` with `target_update_frequency = 0`. That state is
    /// no longer *rejected* — it is unrepresentable, so the assertion moves
    /// from `validate()` to the constructor (ADR 0058 §Consequences). This is
    /// strictly stronger: the old check could be bypassed by struct-update
    /// syntax on the two `pub` scalar fields, whereas no `DqnTrainingConfig`
    /// value can hold a frozen target at all now.
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

    /// Replaces `accepts_soft_updates_with_inert_hard_sync_frequency`. There is
    /// no longer an "inert" second knob to accept: every constructible
    /// `TargetUpdate` is live, and a config built from one validates.
    #[test]
    fn every_constructible_target_update_yields_a_valid_config() {
        for rule in [
            TargetUpdate::polyak(0.005, 1),
            TargetUpdate::polyak(0.5, 100),
            TargetUpdate::hard(1),
            TargetUpdate::hard(10_000),
        ] {
            let cfg = DqnTrainingConfigBuilder::new()
                .target_update(rule)
                .build()
                .expect("a constructible TargetUpdate is always a valid config");
            assert_eq!(cfg.target_update, rule);
        }
    }

    #[test]
    fn rejects_gamma_out_of_range() {
        let err = DqnTrainingConfigBuilder::new()
            .gamma(1.5)
            .build()
            .unwrap_err();
        assert_eq!(err.field, "gamma");
    }

    /// Replaces `rejects_nan_tau_from_struct_update_syntax`. `tau` used to be a
    /// `pub f64`, so struct-update syntax could hand an agent a `NaN` Polyak
    /// coefficient that would poison every target weight on the first soft
    /// update; `validate` was the only thing standing in the way.
    /// `PolyakTau` now rejects `NaN` at construction, so the struct-update
    /// route cannot even build the value to place in the field.
    #[test]
    fn nan_tau_cannot_be_constructed_for_struct_update_syntax() {
        assert!(TargetUpdate::try_polyak(f32::NAN, 1).is_err());
        // And a config built the struct-update way is necessarily valid,
        // because the only τ it can carry came through `PolyakTau`.
        let config = DqnTrainingConfig {
            target_update: TargetUpdate::polyak(0.005, 1),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }
}
