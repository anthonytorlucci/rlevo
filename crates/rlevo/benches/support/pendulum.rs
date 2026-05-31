//! Shared actor/critic scaffolding for the `pendulum_rl` bench.
//!
//! Pulled in per-bench via `#[path = "support/pendulum.rs"] mod pendulum_support;`.
//! Lives in a subdirectory so Cargo's bench autodiscovery does not treat it as a
//! standalone target.
//!
//! Provides:
//! - [`ActorMlp`] — deterministic tanh-scaled actor for DDPG and TD3
//! - [`CriticMlp`] — concat(obs, action) Q-value head shared by DDPG, TD3, SAC
//! - [`StochasticActor`] — squashed-Gaussian reparameterized actor for SAC
//! - Polyak averaging helpers (identical to `support/dqn.rs`)

#![allow(dead_code)]

use std::collections::HashMap;
use std::marker::PhantomData;

use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};
use burn::tensor::{Tensor, TensorData};

// TD3 and SAC re-export these from ddpg_model, so one impl per type covers all three algorithms.
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_model::{ContinuousQ, DeterministicPolicy};
use rlevo_reinforcement_learning::algorithms::sac::sac_model::{SampleOutput, SquashedGaussianPolicy};

// ---------------------------------------------------------------------------
// ActorMlp — deterministic tanh-scaled actor for DDPG and TD3
// (batch, 3) → (batch, 1) ∈ [-2, 2]
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ActorMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
    action_scale: f32,
    action_bias: f32,
}

impl<B: Backend> ActorMlp<B> {
    pub fn new(
        obs_dim: usize,
        hidden: usize,
        action_dim: usize,
        device: &<B as BackendTypes>::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            head: LinearConfig::new(hidden, action_dim).init(device),
            action_scale: 2.0,
            action_bias: 0.0,
        }
    }

    pub fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(self.fc1.forward(obs));
        let h = relu(self.fc2.forward(h));
        tanh(self.head.forward(h))
            .mul_scalar(self.action_scale)
            .add_scalar(self.action_bias)
    }
}

// TD3 re-exports DeterministicPolicy from ddpg_model, so this single impl covers both.
impl<B: AutodiffBackend> DeterministicPolicy<B, 2, 2> for ActorMlp<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_impl(obs)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.forward_impl(obs)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, ActorMlp<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

// ---------------------------------------------------------------------------
// CriticMlp — concat(obs, action) → scalar Q-value
// Shared by DDPG, TD3, and SAC via the respective `ContinuousQ` trait impls.
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct CriticMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> CriticMlp<B> {
    pub fn new(
        obs_dim: usize,
        action_dim: usize,
        hidden: usize,
        device: &<B as BackendTypes>::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim + action_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            head: LinearConfig::new(hidden, 1).init(device),
        }
    }

    pub fn forward_impl(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = Tensor::cat(vec![obs, act], 1);
        let h = relu(self.fc1.forward(x));
        let h = relu(self.fc2.forward(h));
        self.head.forward(h).squeeze_dim::<1>(1)
    }
}

// TD3 and SAC re-export ContinuousQ from ddpg_model, so this single impl covers all three.
impl<B: AutodiffBackend> ContinuousQ<B, 2, 2> for CriticMlp<B> {
    fn forward(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_impl(obs, act)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        act: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 1> {
        inner.forward_impl(obs, act)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, CriticMlp<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

// ---------------------------------------------------------------------------
// StochasticActor — squashed-Gaussian reparameterized actor for SAC
// (batch, 3) → (batch, 1) ∈ [-2, 2]
// ---------------------------------------------------------------------------

const LOG_STD_MIN: f32 = -5.0;
const LOG_STD_MAX: f32 = 2.0;

#[derive(Module, Debug)]
pub struct StochasticActor<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    mean: Linear<B>,
    log_std: Linear<B>,
    action_dim: usize,
    action_scale: f32,
    action_bias: f32,
}

impl<B: Backend> StochasticActor<B> {
    pub fn new(
        obs_dim: usize,
        hidden: usize,
        action_dim: usize,
        device: &<B as BackendTypes>::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            mean: LinearConfig::new(hidden, action_dim).init(device),
            log_std: LinearConfig::new(hidden, action_dim).init(device),
            action_dim,
            action_scale: 2.0,
            action_bias: 0.0,
        }
    }

    fn features(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        relu(self.fc2.forward(relu(self.fc1.forward(obs))))
    }

    fn mean_and_log_std(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.features(obs);
        let mean = self.mean.forward(h.clone());
        let log_std = self.log_std.forward(h).clamp(LOG_STD_MIN, LOG_STD_MAX);
        (mean, log_std)
    }

    #[allow(clippy::cast_precision_loss)]
    fn squashed_sample(
        &self,
        obs: Tensor<B, 2>,
        eps: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let (mean, log_std) = self.mean_and_log_std(obs);
        let action_dim = mean.dims()[1];
        let std = log_std.clone().exp();
        let z = mean.clone() + std * eps;

        let diff = z.clone() - mean;
        let scaled = diff / log_std.clone().exp();
        let scaled_sq = scaled.clone() * scaled;
        let log_2pi = (2.0_f32 * std::f32::consts::PI).ln();
        let per_dim_gauss: Tensor<B, 2> = scaled_sq.mul_scalar(-0.5) - log_std - log_2pi * 0.5;

        let ln_2 = std::f32::consts::LN_2;
        let neg_two_z = z.clone().mul_scalar(-2.0);
        let sp = burn::tensor::activation::softplus(neg_two_z, 1.0);
        let per_dim_jac: Tensor<B, 2> = (z.clone().neg() - sp + ln_2).mul_scalar(2.0);

        let log_prob_z = (per_dim_gauss - per_dim_jac)
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let log_scale_abs = self.action_scale.abs().ln();
        let log_prob = log_prob_z.sub_scalar(log_scale_abs * action_dim as f32);

        let action = tanh(z)
            .mul_scalar(self.action_scale)
            .add_scalar(self.action_bias);
        (action, log_prob)
    }
}

impl<B: AutodiffBackend> SquashedGaussianPolicy<B, 2, 2> for StochasticActor<B> {
    fn action_dim(&self) -> usize {
        self.action_dim
    }

    fn forward_sample(&self, obs: Tensor<B, 2>, eps: Tensor<B, 2>) -> SampleOutput<B, 2> {
        let (action, log_prob) = self.squashed_sample(obs, eps);
        SampleOutput { action, log_prob }
    }

    fn forward_sample_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        eps: Tensor<B::InnerBackend, 2>,
    ) -> SampleOutput<B::InnerBackend, 2> {
        let (action, log_prob) = inner.squashed_sample(obs, eps);
        SampleOutput { action, log_prob }
    }

    fn deterministic_action(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let (mean, _) = self.mean_and_log_std(obs);
        tanh(mean)
            .mul_scalar(self.action_scale)
            .add_scalar(self.action_bias)
    }
}

// ---------------------------------------------------------------------------
// Polyak averaging — identical to support/dqn.rs
// ---------------------------------------------------------------------------

struct ParamCollector<B: Backend> {
    tensors: HashMap<ParamId, TensorData>,
    _marker: PhantomData<B>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.tensors.insert(param.id, param.val().to_data());
    }
}

struct PolyakMapper<B: Backend> {
    active: HashMap<ParamId, TensorData>,
    tau: f32,
    _marker: PhantomData<B>,
}

impl<B: Backend> ModuleMapper<B> for PolyakMapper<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let id = param.id;
        let active = self
            .active
            .remove(&id)
            .expect("param not collected from active network");
        let tau = self.tau;
        param.map(move |target_tensor| {
            let device = target_tensor.device();
            let active_tensor = Tensor::<B, D>::from_data(active, &device);
            target_tensor.mul_scalar(1.0 - tau) + active_tensor.mul_scalar(tau)
        })
    }
}

pub fn polyak_update<B: Backend, M: Module<B>>(active: &M, target: M, tau: f32) -> M {
    let mut collector = ParamCollector::<B> {
        tensors: HashMap::new(),
        _marker: PhantomData,
    };
    active.visit(&mut collector);
    let mut mapper = PolyakMapper::<B> {
        active: collector.tensors,
        tau,
        _marker: PhantomData,
    };
    target.map(&mut mapper)
}
