//! The [`PpoValue`] trait implemented by every value network used with
//! [`PpoAgent`](crate::algorithms::ppo::ppo_agent::PpoAgent).
//!
//! Parallels [`PpoPolicy`](crate::algorithms::ppo::ppo_policy::PpoPolicy).
//! Value networks output one scalar `V(s)` per batch row (return shape
//! `(batch_size,)`). [`MlpValue`] is the default: an MLP with `tanh` hidden
//! layers and a single linear output, the structure of the built-in policy
//! heads. PPG takes the same trait.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::tanh;
use burn::tensor::{Device, Tensor};
use rlevo_core::config::{self, ConfigError, Validate};

use crate::algorithms::shared::{flatten_batch, hidden_layers, nonzero_widths, run_hidden};

/// Contract implemented by any network usable as a PPO value function.
///
/// The `BOR` const generic is the batched observation tensor rank (usually
/// `2` for vector observations of shape `[batch, features]`). The output is
/// shape `(batch_size,)`.
pub trait PpoValue<const BOR: usize>: Module {
    /// Forward pass: predicts `V(s)` per batch row. Return shape: `(batch_size,)`.
    fn forward(&self, obs: Tensor<BOR>) -> Tensor<1>;
}

/// Construction-time knobs for [`MlpValue`].
#[derive(Debug, Clone)]
pub struct MlpValueConfig {
    /// Observation feature count, flattened.
    pub obs_dim: usize,
    /// The hidden layers' widths, first to last. Empty is a linear model.
    pub hidden: Vec<usize>,
}

impl MlpValueConfig {
    /// Validates the config, then builds the network on `device` with Burn's
    /// default initializer, the hidden layers first to last and then the
    /// head.
    ///
    /// # Errors
    ///
    /// Returns the first [`ConfigError`] reported by
    /// [`validate`](Validate::validate).
    pub fn try_init(&self, device: &Device) -> Result<MlpValue, ConfigError> {
        self.validate()?;
        let (hidden, width) = hidden_layers(self.obs_dim, &self.hidden, device);
        Ok(MlpValue {
            hidden,
            head: LinearConfig::new(width, 1).init(device),
        })
    }
}

impl Validate for MlpValueConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "MlpValueConfig";
        config::nonzero(C, "obs_dim", self.obs_dim)?;
        nonzero_widths(C, &self.hidden)?;
        Ok(())
    }
}

/// The default value network: an MLP from a batch of observations to
/// `V(s)`, `(batch,)`.
///
/// `tanh` after each hidden layer (`CleanRL`'s PPO), a linear head one unit
/// wide, squeezed. It implements [`PpoValue`] at every observation rank
/// `BOR`, flattening the observation to `(batch, obs_dim)` first. Build it
/// with [`MlpValueConfig::try_init`].
#[derive(Module, Debug)]
pub struct MlpValue {
    hidden: Vec<Linear>,
    head: Linear,
}

impl<const BOR: usize> PpoValue<BOR> for MlpValue {
    fn forward(&self, obs: Tensor<BOR>) -> Tensor<1> {
        let x = run_hidden(&self.hidden, flatten_batch(obs), tanh);
        self.head.forward(x).squeeze_dim::<1>(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_value_per_row_at_rank_2_and_rank_3() {
        let device = Device::flex();
        let config = MlpValueConfig {
            obs_dim: 6,
            hidden: vec![8, 8],
        };
        let net = config.try_init(&device).unwrap();
        let flat: Tensor<1> = PpoValue::<2>::forward(&net, Tensor::zeros([4, 6], &device));
        let image: Tensor<1> = PpoValue::<3>::forward(&net, Tensor::zeros([4, 2, 3], &device));
        assert_eq!(flat.dims(), [4]);
        assert_eq!(image.dims(), [4]);
        let err = MlpValueConfig {
            obs_dim: 0,
            ..config
        }
        .try_init(&device)
        .unwrap_err();
        assert_eq!(err.field, "obs_dim");
    }
}
