//! Model traits implemented by every actor and critic used with [`DdpgAgent`].
//!
//! DDPG pairs a deterministic actor — mapping observations to continuous
//! actions — with a Q-critic that takes an `(obs, action)` pair and returns a
//! scalar value. Both networks have a Polyak-averaged target twin, a
//! [`valid`](Module::valid) snapshot updated by
//! [`polyak_update`](crate::utils::polyak_update). These traits mirror
//! [`crate::algorithms::dqn::dqn_model::DqnModel`]: one [`forward`] each, run
//! on the training network and on its target alike.
//!
//! [`MlpActor`] and [`MlpCritic`] are the default networks for the two
//! traits; TD3 takes both traits, and SAC the critic.
//!
//! [`DdpgAgent`]: super::ddpg_agent::DdpgAgent
//! [`forward`]: DeterministicPolicy::forward

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, tanh};
use burn::tensor::{Device, Tensor};
use rlevo_core::config::{self, ConfigError, Validate};

use crate::algorithms::shared::{flatten_batch, hidden_layers, nonzero_widths, run_hidden};

/// Contract implemented by any network usable as a DDPG actor or target
/// actor.
///
/// The actor maps a batch of observations of rank `BOR` (including the batch
/// dimension) to a batch of continuous actions of rank `BAR`. Implementors
/// are expected to bake any tanh-squash + action-scale/bias into
/// [`forward`](Self::forward) so that the emitted tensor is already in the
/// env-visible action range.
///
/// The agent runs [`forward`](Self::forward) on the training actor, where the
/// autodiff graph is active, and on the frozen target actor during target-Q
/// computation, where it is not.
pub trait DeterministicPolicy<const BOR: usize, const BAR: usize>: Module {
    /// `(batch, ...obs)` → `(batch, ...action)`.
    fn forward(&self, obs: Tensor<BOR>) -> Tensor<BAR>;
}

/// Contract implemented by any network usable as a DDPG continuous Q-critic
/// or target critic.
///
/// The critic scores an `(observation, action)` pair and returns a scalar
/// value per batch row (shape `(batch,)`). Implementors typically concatenate
/// the observation and action along the feature axis before running an MLP,
/// but the trait does not prescribe the internal layout.
///
/// The agent runs [`forward`](Self::forward) on the training critic during
/// the critic and actor gradient steps, where the autodiff graph is active,
/// and on the target critic for target-Q computation, where no gradients
/// flow.
pub trait ContinuousQ<const BOR: usize, const BAR: usize>: Module {
    /// A `(batch,)` Q-value tensor.
    fn forward(&self, obs: Tensor<BOR>, act: Tensor<BAR>) -> Tensor<1>;
}

/// Construction-time knobs for [`MlpActor`].
#[derive(Debug, Clone)]
pub struct MlpActorConfig {
    /// Observation feature count, flattened.
    pub obs_dim: usize,
    /// The hidden layers' widths, first to last. Empty is a linear model.
    pub hidden: Vec<usize>,
    /// Action components.
    pub action_dim: usize,
    /// The bound of a symmetric action range: the actor emits
    /// `tanh(..) * action_scale`, in `[-action_scale, action_scale]`.
    /// `1.0` leaves `tanh`'s range; the agent then clips each component to
    /// the action's own bounds.
    pub action_scale: f32,
}

impl MlpActorConfig {
    /// Validates the config, then builds the network on `device` with Burn's
    /// default initializer, the hidden layers first to last and then the
    /// head.
    ///
    /// # Errors
    ///
    /// Returns the first [`ConfigError`] reported by
    /// [`validate`](Validate::validate).
    pub fn try_init(&self, device: &Device) -> Result<MlpActor, ConfigError> {
        self.validate()?;
        let (hidden, width) = hidden_layers(self.obs_dim, &self.hidden, device);
        Ok(MlpActor {
            hidden,
            head: LinearConfig::new(width, self.action_dim).init(device),
            action_scale: self.action_scale,
        })
    }
}

impl Validate for MlpActorConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "MlpActorConfig";
        config::nonzero(C, "obs_dim", self.obs_dim)?;
        nonzero_widths(C, &self.hidden)?;
        config::nonzero(C, "action_dim", self.action_dim)?;
        config::positive(C, "action_scale", f64::from(self.action_scale))?;
        Ok(())
    }
}

/// The default deterministic actor: an MLP from a batch of observations to
/// actions, `(batch, action_dim)`, in `[-action_scale, action_scale]`.
///
/// `ReLU` after each hidden layer, a linear head, then `tanh` times
/// `action_scale`. The range is symmetric about zero; for any other range,
/// write an actor, or use a scale of `1.0` and let the agent clip. It
/// implements [`DeterministicPolicy`] at every observation rank `BOR`,
/// flattening the observation first. Build it with
/// [`MlpActorConfig::try_init`].
#[derive(Module, Debug)]
pub struct MlpActor {
    hidden: Vec<Linear>,
    head: Linear,
    action_scale: f32,
}

impl<const BOR: usize> DeterministicPolicy<BOR, 2> for MlpActor {
    fn forward(&self, obs: Tensor<BOR>) -> Tensor<2> {
        let x = run_hidden(&self.hidden, flatten_batch(obs), relu);
        tanh(self.head.forward(x)).mul_scalar(self.action_scale)
    }
}

/// Construction-time knobs for [`MlpCritic`].
#[derive(Debug, Clone)]
pub struct MlpCriticConfig {
    /// Observation feature count, flattened.
    pub obs_dim: usize,
    /// Action components.
    pub action_dim: usize,
    /// The hidden layers' widths, first to last. Empty is a linear model.
    pub hidden: Vec<usize>,
}

impl MlpCriticConfig {
    /// Validates the config, then builds the network on `device` with Burn's
    /// default initializer, the hidden layers first to last and then the
    /// head. The first layer reads `obs_dim + action_dim` features.
    ///
    /// # Errors
    ///
    /// Returns the first [`ConfigError`] reported by
    /// [`validate`](Validate::validate).
    pub fn try_init(&self, device: &Device) -> Result<MlpCritic, ConfigError> {
        self.validate()?;
        let (hidden, width) = hidden_layers(self.obs_dim + self.action_dim, &self.hidden, device);
        Ok(MlpCritic {
            hidden,
            head: LinearConfig::new(width, 1).init(device),
        })
    }
}

impl Validate for MlpCriticConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "MlpCriticConfig";
        config::nonzero(C, "obs_dim", self.obs_dim)?;
        config::nonzero(C, "action_dim", self.action_dim)?;
        nonzero_widths(C, &self.hidden)?;
        Ok(())
    }
}

/// The default continuous Q-critic: an MLP from a batch of observations and
/// actions to `Q(s, a)`, `(batch,)`.
///
/// The flattened observation and the action are concatenated on the
/// feature axis, observation first; `ReLU` after each hidden layer; a linear
/// head one unit wide, squeezed. It implements [`ContinuousQ`] at every
/// observation rank `BOR`, for DDPG, TD3 and SAC. Build it with
/// [`MlpCriticConfig::try_init`].
#[derive(Module, Debug)]
pub struct MlpCritic {
    hidden: Vec<Linear>,
    head: Linear,
}

impl<const BOR: usize> ContinuousQ<BOR, 2> for MlpCritic {
    fn forward(&self, obs: Tensor<BOR>, act: Tensor<2>) -> Tensor<1> {
        let x = Tensor::cat(vec![flatten_batch(obs), act], 1);
        let x = run_hidden(&self.hidden, x, relu);
        self.head.forward(x).squeeze_dim::<1>(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::config::ConstraintKind;

    fn actor_config() -> MlpActorConfig {
        MlpActorConfig {
            obs_dim: 3,
            hidden: vec![8, 8],
            action_dim: 2,
            action_scale: 2.0,
        }
    }

    #[test]
    fn actor_stays_inside_its_scale() {
        let device = Device::flex();
        let net = actor_config().try_init(&device).unwrap();
        let obs = Tensor::<2>::ones([4, 3], &device).mul_scalar(100.0);
        let act: Tensor<2> = DeterministicPolicy::<2, 2>::forward(&net, obs);
        assert_eq!(act.dims(), [4, 2]);
        let max = act.abs().max().into_scalar::<f32>();
        assert!(max <= 2.0, "max |action| {max} exceeds the scale");
    }

    #[test]
    fn actor_config_rejects_a_scale_that_is_not_positive() {
        let device = Device::flex();
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let err = MlpActorConfig {
                action_scale: scale,
                ..actor_config()
            }
            .try_init(&device)
            .unwrap_err();
            assert_eq!(err.field, "action_scale");
            assert!(matches!(
                err.kind,
                ConstraintKind::NotPositive { .. } | ConstraintKind::NotFinite { .. }
            ));
        }
    }

    #[test]
    fn critic_scores_each_row_at_rank_2_and_rank_4() {
        let device = Device::flex();
        let config = MlpCriticConfig {
            obs_dim: 12,
            action_dim: 2,
            hidden: vec![8],
        };
        let net = config.try_init(&device).unwrap();
        let act = Tensor::<2>::zeros([5, 2], &device);
        let flat: Tensor<1> =
            ContinuousQ::<2, 2>::forward(&net, Tensor::zeros([5, 12], &device), act.clone());
        let grid: Tensor<1> =
            ContinuousQ::<4, 2>::forward(&net, Tensor::zeros([5, 2, 3, 2], &device), act);
        assert_eq!(flat.dims(), [5]);
        assert_eq!(grid.dims(), [5]);
        let err = MlpCriticConfig {
            hidden: vec![0],
            ..config
        }
        .try_init(&device)
        .unwrap_err();
        assert_eq!(err.field, "hidden");
    }
}
