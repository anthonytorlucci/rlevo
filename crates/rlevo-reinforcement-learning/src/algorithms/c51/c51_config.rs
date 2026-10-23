//! Hyperparameter configuration for the C51 (Categorical DQN) algorithm.
//!
//! Mirrors [`crate::algorithms::dqn::dqn_config::DqnTrainingConfig`] for all
//! shared training hyperparameters and adds the three categorical
//! distributional parameters — `num_atoms`, `v_min`, and `v_max` — that
//! define the fixed support over which return distributions are represented.

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use rlevo_core::config::{self, ConfigError, Validate};

use crate::algorithms::c51::projection::atom_spacing;
use crate::replay::{PrioritizedReplaySettings, UniformReplayConfig};
use crate::target::TargetUpdate;

/// Configuration for training a Categorical DQN (C51) agent.
///
/// Holds all hyperparameters required to initialise and train a
/// [`crate::algorithms::c51::c51_agent::C51Agent`]. The distribution-specific
/// fields are [`num_atoms`](Self::num_atoms), [`v_min`](Self::v_min), and
/// [`v_max`](Self::v_max); the rest are the standard DQN knobs (learning
/// rate, `$\gamma$`, `$\tau$`, `$\epsilon$` schedule, replay capacity, …).
#[derive(Clone, Debug)]
pub struct C51TrainingConfig {
    /// Minibatch size sampled from the replay buffer each learn step.
    pub batch_size: usize,

    /// Discount factor `$\gamma$` in `[0, 1]`.
    pub gamma: f64,

    /// Optimizer learning rate.
    pub learning_rate: f64,

    /// Initial `$\epsilon$` value for the `$\epsilon$`-greedy exploration schedule.
    pub epsilon_start: f64,

    /// Floor `$\epsilon$` value for the exploration schedule.
    pub epsilon_end: f64,

    /// Multiplicative decay applied to `$\epsilon$` each env step.
    pub epsilon_decay: f64,

    /// How the target network tracks the policy network: one cadence, one `$\tau$`.
    ///
    /// [`TargetUpdate`] is a single mechanism, not two (ADR 0058). Its cadence
    /// [`every`](TargetUpdate::every) decides *when* an update fires; its
    /// coefficient [`tau`](TargetUpdate::tau) decides *how far* the target
    /// moves when it does — `$\text{target} \leftarrow (1 - \tau) \cdot \text{target} + \tau \cdot \text{policy}$`. A
    /// periodic hard copy is not a second mechanism but the degenerate
    /// `$\tau = 1.0$`, spelled [`TargetUpdate::hard`]. The update is applied inside
    /// [`C51Agent::learn_step`] and nowhere else, so no train loop can forget
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
    /// [`C51Agent::learn_step`]: crate::algorithms::c51::c51_agent::C51Agent::learn_step
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

    /// Number of atoms `N` in the categorical return distribution.
    ///
    /// The default is 51 — the value from the original C51 paper, from which
    /// the algorithm takes its name.
    pub num_atoms: usize,

    /// Lower bound of the atom support `$z_0$`.
    pub v_min: f32,

    /// Upper bound of the atom support `$z_{N-1}$`.
    pub v_max: f32,

    /// Optional gradient-norm / gradient-value clipping.
    pub clip_grad: Option<GradientClippingConfig>,

    /// Optimizer configuration (Adam `$\beta$`'s, `$\epsilon$`, etc.).
    pub optimizer: AdamConfig,

    /// Opt-in prioritized experience replay (Schaul et al. 2016), `None` by
    /// default (uniform replay).
    ///
    /// The priority signal for C51 is the **KL divergence**
    /// `$D_{KL}(\text{target} \parallel \text{pred})$`, *not* the cross-entropy the gradient uses — the
    /// two differ by the per-sample target entropy `$H(\text{target})$` (Rainbow,
    /// "prioritize transitions by the KL loss, since this is what the algorithm
    /// is minimizing"; see
    /// [`categorical_kl_per_sample`](crate::algorithms::c51::loss::categorical_kl_per_sample)).
    ///
    /// **Literature-recommended exponent.** Rainbow pairs the KL priority with
    /// `$\omega = 0.5$` (its `priority_exponent` here), and reports performance is "very
    /// robust to the choice of `$\omega$`" under the KL priority. The shipped default is
    /// Schaul's `0.6`; set `priority_exponent` to `0.5` on
    /// [`PrioritizedReplaySettings`] to match Rainbow. Buffer capacity comes from
    /// [`replay_buffer_capacity`](Self::replay_buffer_capacity).
    pub prioritized_replay: Option<PrioritizedReplaySettings>,
}

impl C51TrainingConfig {
    /// Spacing between adjacent atoms: `$\left (v_{\max} - v_{\min} \right ) / \left ( \text{num_atoms} - 1 \right )$`.
    ///
    /// Delegates to [`atom_spacing`] so the support this config builds and the
    /// bin indices the categorical projection computes share one definition of
    /// `$\Delta z$` — an ULP of disagreement between the two is enough to push a
    /// scatter index off the end of the support.
    ///
    /// # Returns
    ///
    /// The uniform atom spacing, or [`f32::NAN`] for a degenerate
    /// `num_atoms < 2` (a spacing is undefined with fewer than two atoms).
    /// [`Validate`] rejects such a config, but `delta_z` can be called on a
    /// config no constructor has validated, so it stays total rather than
    /// panicking.
    #[must_use]
    pub fn delta_z(&self) -> f32 {
        atom_spacing(self.v_min, self.v_max, self.num_atoms)
    }
}

impl Default for C51TrainingConfig {
    /// Returns defaults consistent with `CleanRL`'s reference C51 hyperparameters.
    ///
    /// [`target_update`](Self::target_update) is a `$\tau$` = 0.005 Polyak step on **every**
    /// gradient update. That is bit-for-bit the pre-[`TargetUpdate`] behaviour: the old
    /// `tau = 0.005` soft update ran ungated inside every learn step, which in gradient-update
    /// units is exactly `every = 1`. The old `target_update_frequency = 10_000` is deliberately not
    /// carried over — it was inert under `tau > 0` and, read as a cadence under the unified rule,
    /// would collapse the Polyak schedule `$10\,000\times$` (ADR 0059 section on Consequences).
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
            num_atoms: 51,
            v_min: -10.0,
            v_max: 10.0,
            clip_grad: Some(GradientClippingConfig::Value(100.0)),
            optimizer: AdamConfig::new(),
            prioritized_replay: None,
        }
    }
}

impl Validate for C51TrainingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "C51TrainingConfig";
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
        config::at_least(C, "num_atoms", self.num_atoms, 2)?;
        config::distinct(C, "v_max", f64::from(self.v_min), f64::from(self.v_max))?;
        config::ordered(C, "v_max", f64::from(self.v_min), f64::from(self.v_max))?;
        if let Some(per) = &self.prioritized_replay {
            per.validate()?;
        }
        // `target_update` carries no check here, deliberately: `TargetUpdate` is valid by
        // construction (ADR 0027 section 3 — a validated newtype *removes* its paired `config::`
        // line). Its `PolyakTau` excludes `$\tau$` = 0.0 and its `NonZeroUsize` cadence excludes 0,
        // so the frozen target the old cross-field check rejected is now unrepresentable rather
        // than merely rejected — including through `..Default::default()` struct update, which
        // `validate` never saw.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Exact comparison is intentional throughout this test module: the values are
    // config literals read back unchanged, or a computed result whose bit-exactness
    // is itself the property under test (that an anneal lands exactly on its
    // endpoint, that `-0.0` is accepted as the no-correction setting). A tolerance
    // would let a real regression pass. Reviewed as a class, not site-by-site.
    #![allow(clippy::float_cmp)]
    use super::*;
    use crate::MAX_BUFFER_CAPACITY;
    use rlevo_core::config::ConstraintKind;

    #[test]
    fn defaults_are_51_atoms_on_symmetric_support() {
        let cfg = C51TrainingConfig::default();
        assert_eq!(cfg.num_atoms, 51);
        assert_eq!(cfg.v_min, -10.0);
        assert_eq!(cfg.v_max, 10.0);
    }

    #[test]
    fn delta_z_matches_uniform_spacing() {
        let cfg = C51TrainingConfig {
            num_atoms: 51,
            v_min: -10.0,
            v_max: 10.0,
            ..Default::default()
        };
        // 50 gaps spanning 20.0 → 0.4 per atom.
        assert!((cfg.delta_z() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn default_config_is_valid() {
        assert!(C51TrainingConfig::default().validate().is_ok());
    }

    /// `replay_buffer_capacity` had a floor and no ceiling, so a
    /// `usize::MAX` sentinel — a unit slip, a deserialized garbage field —
    /// validated and went straight to the buffer constructor, where it aborts
    /// on a failed allocation or wraps the `SumTree` arithmetic. The rejection
    /// must be a **recoverable `Err`**, which is why this asserts on the
    /// returned error rather than using `#[should_panic]`: a panic here would
    /// be the bug, not the fix.
    #[test]
    fn rejects_replay_buffer_capacity_above_ceiling() {
        let err = C51TrainingConfig {
            replay_buffer_capacity: usize::MAX,
            ..Default::default()
        }
        .validate()
        .expect_err("usize::MAX capacity must be rejected, not allocated");
        assert_eq!(err.field, "replay_buffer_capacity");
        let max = u64::try_from(MAX_BUFFER_CAPACITY).expect("the ceiling fits in u64");
        let got = u64::try_from(usize::MAX).expect("usize::MAX fits in u64 on this target");
        assert_eq!(err.kind, ConstraintKind::TooLarge { max, got });
    }

    /// The boundary is inclusive: the ceiling itself is a legal (if
    /// unallocatable) capacity, so the guard cannot be off by one.
    #[test]
    fn accepts_replay_buffer_capacity_at_ceiling() {
        let cfg = C51TrainingConfig {
            replay_buffer_capacity: MAX_BUFFER_CAPACITY,
            ..Default::default()
        };
        cfg.validate().expect("the ceiling itself must validate");
    }

    /// The default is behaviour-preserving: the old `tau = 0.005` soft update
    /// ran inside *every* learn step with the hard path inert, which in
    /// gradient-update units is `every = 1` (ADR 0059 §Consequences).
    #[test]
    fn default_target_update_is_polyak_every_gradient_update() {
        let cfg = C51TrainingConfig::default();
        assert_eq!(cfg.target_update, TargetUpdate::polyak(0.005, 1));
        assert_eq!(cfg.target_update.every(), 1);
    }

    /// Replaces `rejects_config_where_target_never_updates`. The frozen-target
    /// state is no longer *rejected* by `validate` — it is unrepresentable, so
    /// the assertion moves to the constructor (ADR 0058 §Consequences). That is
    /// strictly stronger: the old check could be bypassed by struct-update
    /// syntax on the two `pub` scalar fields.
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

    /// Replaces `accepts_either_target_update_mechanism_alone`. There are no
    /// longer two mechanisms to accept "alone": hard is `$\tau = 1.0$` on the one
    /// rule, and every constructible rule yields a valid config.
    #[test]
    fn every_constructible_target_update_yields_a_valid_config() {
        for rule in [
            TargetUpdate::polyak(0.005, 1),
            TargetUpdate::polyak(0.5, 100),
            TargetUpdate::hard(1),
            TargetUpdate::hard(10_000),
        ] {
            let cfg = C51TrainingConfig {
                target_update: rule,
                ..Default::default()
            };
            cfg.validate()
                .expect("a constructible TargetUpdate is always a valid config");
        }
    }

    #[test]
    fn rejects_degenerate_support() {
        let err = C51TrainingConfig {
            v_min: 5.0,
            v_max: 5.0,
            ..Default::default()
        }
        .validate()
        .unwrap_err();
        assert_eq!(err.field, "v_max");
    }
}
