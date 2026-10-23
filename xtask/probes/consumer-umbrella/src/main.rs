//! Measurement, and one check that it builds: a network an outside author
//! writes, derived through `rlevo::burn`, beside one of the RL crate's
//! default networks. Burn's `Module` and `Config` derives expand to `burn::`
//! paths, so the module that uses them keeps `use rlevo::burn;` in scope (the
//! umbrella's crate docs, Backends). A doctest cannot check that: rustdoc
//! compiles `rlevo`'s doctests with `rlevo`'s own `burn` in scope.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use rlevo::burn;
use rlevo::rl::algorithms::dqn::dqn_model::{DqnModel, MlpQNetworkConfig};

#[derive(Config, Debug)]
struct CriticConfig {
    obs_dim: usize,
}

#[derive(Module, Debug)]
struct Critic {
    head: Linear,
}

impl CriticConfig {
    fn init(&self, device: &Device) -> Critic {
        Critic {
            head: LinearConfig::new(self.obs_dim, 1).init(device),
        }
    }
}

fn main() {
    let device = Device::flex();
    let critic = CriticConfig::new(4).init(&device);
    let network = MlpQNetworkConfig {
        obs_dim: 4,
        hidden: vec![64, 64],
        num_actions: 2,
    }
    .try_init(&device)
    .expect("valid Q-network config");
    let obs: Tensor<2> = Tensor::zeros([1, 4], &device);
    let value = critic.head.forward(obs.clone());
    let q_values = DqnModel::<2>::forward(&network, obs);
    assert_eq!((value.dims(), q_values.dims()), ([1, 1], [1, 2]));
}
