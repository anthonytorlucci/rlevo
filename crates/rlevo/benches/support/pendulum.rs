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
//!
//! The Polyak target-network update is shared across all actor-critic benches via
//! [`rlevo_reinforcement_learning::utils::polyak_update`] rather than re-implemented here.

#![allow(dead_code)]

use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::{relu, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};

// TD3 and SAC re-export these from ddpg_model, so one impl per type covers all three algorithms.
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_model::{
    ContinuousQ, DeterministicPolicy,
};
use rlevo_reinforcement_learning::algorithms::sac::sac_model::{
    SampleOutput, SquashedGaussianPolicy,
};

use rlevo_reinforcement_learning::utils::{PolyakError, polyak_update};

// ---------------------------------------------------------------------------
// ActorMlp — deterministic tanh-scaled actor for DDPG and TD3
// (batch, 3) → (batch, 1) ∈ [-2, 2]
// ---------------------------------------------------------------------------

/// Deterministic tanh-scaled actor for DDPG and TD3.
///
/// A two-hidden-layer `ReLU` MLP whose output is passed through `tanh` and
/// then linearly scaled to the Pendulum action range `[-2, 2]`. Implements
/// [`DeterministicPolicy`] so the same struct serves both DDPG and TD3 (TD3
/// re-exports the trait from `ddpg_model`).
#[derive(Module, Debug)]
pub struct ActorMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
    action_scale: f32,
    action_bias: f32,
}

impl<B: Backend> ActorMlp<B> {
    /// Builds an `ActorMlp` with two `hidden`-wide rectified-linear layers.
    ///
    /// `action_scale` is fixed at `2.0` and `action_bias` at `0.0`, matching
    /// the Pendulum-v1 torque range `[-2, 2]`.
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

    /// Runs the forward pass: `ReLU → ReLU → tanh → scale → bias`.
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
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
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

/// Continuous Q-value critic shared by DDPG, TD3, and SAC.
///
/// Concatenates the observation and action tensors along the feature axis and
/// passes the result through a two-hidden-layer `ReLU` MLP, producing a scalar
/// Q-value per batch row. Implements [`ContinuousQ`] — TD3 and SAC re-export
/// that trait from `ddpg_model`, so this single struct covers all three
/// algorithms.
#[derive(Module, Debug)]
pub struct CriticMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> CriticMlp<B> {
    /// Builds a `CriticMlp` with two `hidden`-wide rectified-linear layers.
    ///
    /// The first layer input width is `obs_dim + action_dim` to accommodate the
    /// concatenated observation–action pair.
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

    /// Concatenates `obs` and `act`, then runs `ReLU → ReLU → squeeze` to a
    /// rank-1 scalar tensor.
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
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
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

/// Lower clamp on the log-standard-deviation output, preventing near-zero std
/// from destabilizing the log-probability computation.
const LOG_STD_MIN: f32 = -5.0;

/// Upper clamp on the log-standard-deviation output, preventing excessively
/// large std from making the policy nearly uniform.
const LOG_STD_MAX: f32 = 2.0;

/// Squashed-Gaussian reparameterized actor for SAC.
///
/// A two-hidden-layer `ReLU` MLP with two parallel heads — one for the
/// Gaussian mean and one for the log-standard-deviation (clamped to
/// `[LOG_STD_MIN, LOG_STD_MAX]`). Actions are drawn via the reparameterization
/// trick, squashed through `tanh`, and scaled to `[-2, 2]`. The corresponding
/// log-probabilities account for the tanh Jacobian correction.
///
/// Implements [`SquashedGaussianPolicy`] so the SAC trainer can drive both
/// stochastic sampling (with entropy) and deterministic evaluation.
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
    /// Builds a `StochasticActor` with two `hidden`-wide rectified-linear
    /// layers and parallel mean/log-std heads of width `action_dim`.
    ///
    /// `action_scale` is fixed at `2.0` and `action_bias` at `0.0`, matching
    /// the Pendulum-v1 torque range `[-2, 2]`.
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

    /// Shared trunk: `obs → ReLU(fc1) → ReLU(fc2)`.
    fn features(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        relu(self.fc2.forward(relu(self.fc1.forward(obs))))
    }

    /// Returns the Gaussian `(mean, log_std)` for the given observation batch.
    /// `log_std` is clamped to `[LOG_STD_MIN, LOG_STD_MAX]`.
    fn mean_and_log_std(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.features(obs);
        let mean = self.mean.forward(h.clone());
        let log_std = self.log_std.forward(h).clamp(LOG_STD_MIN, LOG_STD_MAX);
        (mean, log_std)
    }

    /// Draws one squashed-Gaussian sample per row using the pre-sampled noise
    /// `eps ~ N(0, I)` (reparameterization trick) and returns the action
    /// together with its log-probability.
    ///
    /// The log-probability is computed as the Gaussian log-density of the
    /// pre-squash sample `z`, minus the tanh Jacobian correction
    /// `sum_j 2(log 2 - z_j - softplus(-2 z_j))`, minus the log of
    /// `|action_scale|` per action dimension. This matches the numerically
    /// stable form from the SAC paper appendix.
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

        let log_prob_z = (per_dim_gauss - per_dim_jac).sum_dim(1).squeeze_dim::<1>(1);
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
