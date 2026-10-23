//! The [`DqnModel`] trait implemented by every network used with [`DqnAgent`],
//! [`C51Agent`] or [`QrDqnAgent`], and its two default networks:
//! [`MlpQNetwork`] for DQN and [`MlpDistributionalQ`] for C51 and QR-DQN.
//!
//! [`DqnAgent`]: crate::algorithms::dqn::dqn_agent::DqnAgent
//! [`C51Agent`]: crate::algorithms::c51::c51_agent::C51Agent
//! [`QrDqnAgent`]: crate::algorithms::qrdqn::qrdqn_agent::QrDqnAgent

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::{Device, Tensor};
use rlevo_core::config::{self, ConfigError, Validate};

use crate::algorithms::shared::{flatten_batch, hidden_layers, nonzero_widths, run_hidden};

/// Contract implemented by any network usable as a DQN, C51 or QR-DQN
/// policy or target.
///
/// One method, [`forward`](Self::forward), from a batch of observations to a
/// value per action. The output rank `OUT` says which:
///
/// - **`OUT = 2`, the default:** Q-values, a `(batch, num_actions)` tensor.
///   [`DqnAgent`] takes this, so `DqnModel<BOR>` means a Q-network.
/// - **`OUT = 3`:** a return distribution per action, a `(batch,
///   num_actions, num_atoms)` tensor. [`C51Agent`] reads it as
///   **unnormalised logits** over a fixed atom support: it takes a `softmax`
///   along the atom axis and the expected Q-value `$\sum_i z_i \cdot p_i$`.
///   [`QrDqnAgent`] reads it as **raw quantile values** at the midpoints
///   `$\tau_i = (i + 0.5)/N$`, so a network applies no activation at its
///   head. The network is the same for both; the agent decides what the
///   numbers mean.
///
/// The agent runs `forward` on the policy network, which trains, and on the
/// target network, a [`valid`](Module::valid) snapshot of the policy that
/// does not. The target's update is
/// [`polyak_update`](crate::utils::polyak_update), the same for every
/// network, so it is not part of this trait.
///
/// The `BOR` const generic is the observation tensor rank *including* the
/// leading batch dimension (e.g. `BOR = 2` for vector observations of shape
/// `[batch_size, features]`; `BOR = 4` for image observations of shape
/// `[batch_size, channels, height, width]`).
///
/// [`DqnAgent`]: crate::algorithms::dqn::dqn_agent::DqnAgent
/// [`C51Agent`]: crate::algorithms::c51::c51_agent::C51Agent
/// [`QrDqnAgent`]: crate::algorithms::qrdqn::qrdqn_agent::QrDqnAgent
pub trait DqnModel<const BOR: usize, const OUT: usize = 2>: Module {
    /// A value per action for a batch of observations: Q-values when
    /// `OUT = 2`, a distribution per action when `OUT = 3`.
    fn forward(&self, observations: Tensor<BOR>) -> Tensor<OUT>;
}

/// Construction-time knobs for [`MlpQNetwork`].
#[derive(Debug, Clone)]
pub struct MlpQNetworkConfig {
    /// Observation feature count, flattened: `h * w * c` for observations
    /// batched as `[batch, h, w, c]`.
    pub obs_dim: usize,
    /// The hidden layers' widths, first to last. Empty is a linear model.
    pub hidden: Vec<usize>,
    /// Number of discrete actions.
    pub num_actions: usize,
}

impl MlpQNetworkConfig {
    /// Validates the config, then builds the network on `device` with Burn's
    /// default initializer, the hidden layers first to last and then the
    /// head.
    ///
    /// There is no infallible `init`, as for the policy heads: a zero width
    /// would otherwise build a zero-width layer silently.
    ///
    /// # Errors
    ///
    /// Returns the first [`ConfigError`] reported by
    /// [`validate`](Validate::validate).
    pub fn try_init(&self, device: &Device) -> Result<MlpQNetwork, ConfigError> {
        self.validate()?;
        let (hidden, width) = hidden_layers(self.obs_dim, &self.hidden, device);
        Ok(MlpQNetwork {
            hidden,
            head: LinearConfig::new(width, self.num_actions).init(device),
        })
    }
}

impl Validate for MlpQNetworkConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "MlpQNetworkConfig";
        config::nonzero(C, "obs_dim", self.obs_dim)?;
        nonzero_widths(C, &self.hidden)?;
        config::nonzero(C, "num_actions", self.num_actions)?;
        Ok(())
    }
}

/// The default Q-network: an MLP from a batch of observations to Q-values,
/// `(batch, num_actions)`.
///
/// `ReLU` after each hidden layer, a linear head. It implements
/// [`DqnModel`] at every observation rank `BOR`, flattening the observation
/// to `(batch, obs_dim)` first, so one network serves vector and grid
/// observations. Build it with [`MlpQNetworkConfig::try_init`].
#[derive(Module, Debug)]
pub struct MlpQNetwork {
    hidden: Vec<Linear>,
    head: Linear,
}

impl<const BOR: usize> DqnModel<BOR> for MlpQNetwork {
    fn forward(&self, observations: Tensor<BOR>) -> Tensor<2> {
        let x = run_hidden(&self.hidden, flatten_batch(observations), relu);
        self.head.forward(x)
    }
}

/// Construction-time knobs for [`MlpDistributionalQ`].
#[derive(Debug, Clone)]
pub struct MlpDistributionalQConfig {
    /// Observation feature count, flattened.
    pub obs_dim: usize,
    /// The hidden layers' widths, first to last. Empty is a linear model.
    pub hidden: Vec<usize>,
    /// Number of discrete actions.
    pub num_actions: usize,
    /// Atoms per action for C51, quantiles per action for QR-DQN.
    pub num_atoms: usize,
}

impl MlpDistributionalQConfig {
    /// Validates the config, then builds the network on `device` with Burn's
    /// default initializer, the hidden layers first to last and then the
    /// head.
    ///
    /// # Errors
    ///
    /// Returns the first [`ConfigError`] reported by
    /// [`validate`](Validate::validate).
    pub fn try_init(&self, device: &Device) -> Result<MlpDistributionalQ, ConfigError> {
        self.validate()?;
        let (hidden, width) = hidden_layers(self.obs_dim, &self.hidden, device);
        Ok(MlpDistributionalQ {
            hidden,
            head: LinearConfig::new(width, self.num_actions * self.num_atoms).init(device),
            num_actions: self.num_actions,
            num_atoms: self.num_atoms,
        })
    }
}

impl Validate for MlpDistributionalQConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "MlpDistributionalQConfig";
        config::nonzero(C, "obs_dim", self.obs_dim)?;
        nonzero_widths(C, &self.hidden)?;
        config::nonzero(C, "num_actions", self.num_actions)?;
        config::nonzero(C, "num_atoms", self.num_atoms)?;
        Ok(())
    }
}

/// The default distributional Q-network: an MLP from a batch of
/// observations to a `(batch, num_actions, num_atoms)` tensor, for C51 and
/// QR-DQN alike.
///
/// `ReLU` after each hidden layer, a linear head `num_actions * num_atoms`
/// wide, reshaped. No activation at the head: C51 reads the output as
/// logits and QR-DQN as quantile values ([`DqnModel`] says how each does).
/// It implements [`DqnModel`] at output rank 3 and every observation rank
/// `BOR`. Build it with [`MlpDistributionalQConfig::try_init`].
#[derive(Module, Debug)]
pub struct MlpDistributionalQ {
    hidden: Vec<Linear>,
    head: Linear,
    num_actions: usize,
    num_atoms: usize,
}

impl<const BOR: usize> DqnModel<BOR, 3> for MlpDistributionalQ {
    fn forward(&self, observations: Tensor<BOR>) -> Tensor<3> {
        let batch = observations.dims()[0];
        let x = run_hidden(&self.hidden, flatten_batch(observations), relu);
        self.head
            .forward(x)
            .reshape([batch, self.num_actions, self.num_atoms])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::config::ConstraintKind;

    fn q_config(hidden: Vec<usize>) -> MlpQNetworkConfig {
        MlpQNetworkConfig {
            obs_dim: 12,
            hidden,
            num_actions: 3,
        }
    }

    #[test]
    fn q_network_shapes_at_rank_2_and_rank_4() {
        let device = Device::flex();
        let net = q_config(vec![8, 8]).try_init(&device).unwrap();
        let flat: Tensor<2> = DqnModel::<2>::forward(&net, Tensor::zeros([5, 12], &device));
        let grid: Tensor<2> = DqnModel::<4>::forward(&net, Tensor::zeros([5, 2, 3, 2], &device));
        assert_eq!(flat.dims(), [5, 3]);
        assert_eq!(grid.dims(), [5, 3]);
    }

    #[test]
    fn q_network_reads_a_grid_as_its_flattened_row() {
        let device = Device::flex();
        let net = q_config(vec![8]).try_init(&device).unwrap();
        let values: Vec<f32> = (0..24_u8).map(|i| f32::from(i) / 10.0).collect();
        let grid = Tensor::<1>::from_floats(values.as_slice(), &device).reshape([2, 2, 3, 2]);
        let flat = Tensor::<1>::from_floats(values.as_slice(), &device).reshape([2, 12]);
        let a = DqnModel::<4>::forward(&net, grid).into_data();
        let b = DqnModel::<2>::forward(&net, flat).into_data();
        assert_eq!(a, b);
    }

    #[test]
    fn empty_hidden_is_a_linear_model() {
        let device = Device::flex();
        let net = q_config(Vec::new()).try_init(&device).unwrap();
        let out: Tensor<2> = DqnModel::<2>::forward(&net, Tensor::zeros([4, 12], &device));
        assert_eq!(out.dims(), [4, 3]);
    }

    #[test]
    fn q_network_config_rejects_each_zero() {
        let device = Device::flex();
        for (config, field) in [
            (
                MlpQNetworkConfig {
                    obs_dim: 0,
                    ..q_config(vec![8])
                },
                "obs_dim",
            ),
            (q_config(vec![8, 0]), "hidden"),
            (
                MlpQNetworkConfig {
                    num_actions: 0,
                    ..q_config(vec![8])
                },
                "num_actions",
            ),
        ] {
            let err = config.try_init(&device).unwrap_err();
            assert_eq!(err.field, field);
            assert_eq!(err.kind, ConstraintKind::Zero);
        }
    }

    #[test]
    fn distributional_shape_and_zero_atoms() {
        let device = Device::flex();
        let config = MlpDistributionalQConfig {
            obs_dim: 4,
            hidden: vec![8, 8],
            num_actions: 2,
            num_atoms: 5,
        };
        let net = config.try_init(&device).unwrap();
        let out: Tensor<3> = DqnModel::<2, 3>::forward(&net, Tensor::zeros([3, 4], &device));
        assert_eq!(out.dims(), [3, 2, 5]);
        let err = MlpDistributionalQConfig {
            num_atoms: 0,
            ..config
        }
        .try_init(&device)
        .unwrap_err();
        assert_eq!(err.field, "num_atoms");
    }
}
