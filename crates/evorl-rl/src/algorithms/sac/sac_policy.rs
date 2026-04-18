//! Built-in squashed-Gaussian policy head for SAC.
//!
//! Two-hidden-layer MLP emitting a state-conditional mean and `log σ` per
//! action dimension. Samples via the reparameterization trick
//! `z = μ + σ·ε`, squashes to `a = action_scale · tanh(z) + action_bias`,
//! and returns the log-probability of `a` under the true squashed-Gaussian
//! density — Jacobian correction included.
//!
//! # Why state-conditional `log σ`?
//!
//! Unlike [PPO's continuous head](crate::algorithms::ppo::policies::gaussian),
//! which uses a free `log_std` parameter and deliberately skips the tanh
//! Jacobian (it cancels in the PPO ratio), SAC needs (a) a policy whose
//! entropy can shrink in certainty regions and grow in uncertain ones, which
//! requires `log σ` to depend on the observation, and (b) the exact log-density
//! of the squashed sample because the entropy appears directly in both the
//! Bellman target and the actor loss.
//!
//! # Numerical stability
//!
//! The tanh Jacobian `log(1 − tanh²(z))` is computed as
//! `2·(ln 2 − z − softplus(−2·z))` — the identity used throughout the SAC
//! literature to avoid catastrophic cancellation near the saturation region.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, softplus, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

use crate::algorithms::sac::sac_model::{SampleOutput, SquashedGaussianPolicy};

/// Construction-time knobs for [`SquashedGaussianPolicyHead`].
#[derive(Debug, Clone)]
pub struct SquashedGaussianPolicyHeadConfig {
    /// Observation feature count.
    pub obs_dim: usize,
    /// Hidden layer width (shared across both hidden layers).
    pub hidden: usize,
    /// Number of continuous action dimensions.
    pub action_dim: usize,
    /// Lower clamp applied to the `log σ` head output. CleanRL uses `-5`.
    pub log_std_min: f32,
    /// Upper clamp applied to the `log σ` head output. CleanRL uses `2`.
    pub log_std_max: f32,
    /// Multiplier applied to `tanh(z)` before the env sees the action.
    pub action_scale: f32,
    /// Bias added after the tanh-scale so asymmetric ranges (e.g., `[0, 1]`)
    /// can be encoded.
    pub action_bias: f32,
}

impl SquashedGaussianPolicyHeadConfig {
    /// Constructs the module on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SquashedGaussianPolicyHead<B> {
        SquashedGaussianPolicyHead {
            fc1: LinearConfig::new(self.obs_dim, self.hidden).init(device),
            fc2: LinearConfig::new(self.hidden, self.hidden).init(device),
            mean: LinearConfig::new(self.hidden, self.action_dim).init(device),
            log_std: LinearConfig::new(self.hidden, self.action_dim).init(device),
            action_dim: self.action_dim,
            log_std_min: self.log_std_min,
            log_std_max: self.log_std_max,
            action_scale: self.action_scale,
            action_bias: self.action_bias,
        }
    }
}

/// MLP → `(μ, log σ)` squashed-Gaussian head with tanh output.
///
/// `log_std_min` / `log_std_max` / `action_scale` / `action_bias` are
/// constants captured at construction time. They are **not** learnable and
/// travel with the module only because Burn's `#[derive(Module)]` requires
/// fields to be either `Param`s, sub-modules, or plain data.
#[derive(Module, Debug)]
pub struct SquashedGaussianPolicyHead<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    mean: Linear<B>,
    log_std: Linear<B>,
    action_dim: usize,
    log_std_min: f32,
    log_std_max: f32,
    action_scale: f32,
    action_bias: f32,
}

impl<B: Backend> SquashedGaussianPolicyHead<B> {
    /// Shared hidden-feature extractor.
    fn features(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(self.fc1.forward(obs));
        relu(self.fc2.forward(h))
    }

    /// `(μ, log σ)` with `log σ` clamped to `[log_std_min, log_std_max]`.
    /// Exposed at crate visibility so the agent can pull the mean for
    /// deterministic evaluation without re-running the feature extractor.
    pub(crate) fn mean_and_log_std(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.features(obs);
        let mean = self.mean.forward(h.clone());
        let raw_log_std = self.log_std.forward(h);
        let log_std = raw_log_std.clamp(self.log_std_min, self.log_std_max);
        (mean, log_std)
    }

    /// Squashed policy mean (deterministic action): `scale·tanh(μ) + bias`.
    pub fn mean_action(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let (mean, _) = self.mean_and_log_std(obs);
        tanh(mean)
            .mul_scalar(self.action_scale)
            .add_scalar(self.action_bias)
    }

    /// Scale multiplier applied after tanh.
    pub fn action_scale(&self) -> f32 {
        self.action_scale
    }

    /// Bias added after `scale·tanh(z)`.
    pub fn action_bias(&self) -> f32 {
        self.action_bias
    }

    /// Clamp lower bound applied to `log σ`.
    pub fn log_std_min(&self) -> f32 {
        self.log_std_min
    }

    /// Clamp upper bound applied to `log σ`.
    pub fn log_std_max(&self) -> f32 {
        self.log_std_max
    }
}

/// Shared math: given `(μ, log σ, ε)`, returns the squashed sample and its
/// log-probability (Jacobian-corrected) as a `(batch, action_dim)` /
/// `(batch,)` pair. Generic over the backend so the agent can call it on
/// both `B` (autodiff) and `B::InnerBackend` via the SAC policy trait.
fn squashed_sample_log_prob<BB: Backend>(
    mean: Tensor<BB, 2>,
    log_std: Tensor<BB, 2>,
    eps: Tensor<BB, 2>,
    action_scale: f32,
    action_bias: f32,
) -> (Tensor<BB, 2>, Tensor<BB, 1>) {
    let action_dim = mean.dims()[1];
    // z = μ + σ·ε
    let std = log_std.clone().exp();
    let z = mean.clone() + std * eps;

    // log N(z | μ, σ) per dim, summed across action dim.
    let diff = z.clone() - mean;
    let scaled = diff / log_std.clone().exp();
    let scaled_sq = scaled.clone() * scaled;
    let log_2pi = (2.0_f32 * std::f32::consts::PI).ln();
    let per_dim_gauss: Tensor<BB, 2> =
        scaled_sq.mul_scalar(-0.5) - log_std - log_2pi * 0.5;

    // Tanh Jacobian per dim: log(1 − tanh²(z)) = 2·(ln 2 − z − softplus(−2z))
    // Additionally, `action = scale·tanh(z) + bias` introduces a `log|scale|`
    // term per action dim (the `+bias` shift has unit Jacobian).
    let ln_2 = std::f32::consts::LN_2;
    let neg_two_z = z.clone().mul_scalar(-2.0);
    let sp = softplus(neg_two_z, 1.0);
    let per_dim_jac: Tensor<BB, 2> = (z.clone().neg() - sp + ln_2).mul_scalar(2.0);

    let per_dim = per_dim_gauss - per_dim_jac;
    let log_prob_z = per_dim.sum_dim(1).squeeze_dim::<1>(1);
    let log_scale_abs = action_scale.abs().ln();
    let log_prob = log_prob_z.sub_scalar(log_scale_abs * action_dim as f32);

    let action = tanh(z).mul_scalar(action_scale).add_scalar(action_bias);
    (action, log_prob)
}

impl<B: AutodiffBackend> SquashedGaussianPolicy<B, 2, 2> for SquashedGaussianPolicyHead<B> {
    fn action_dim(&self) -> usize {
        self.action_dim
    }

    fn forward_sample(
        &self,
        obs: Tensor<B, 2>,
        eps: Tensor<B, 2>,
    ) -> SampleOutput<B, 2> {
        let (mean, log_std) = self.mean_and_log_std(obs);
        let (action, log_prob) = squashed_sample_log_prob::<B>(
            mean,
            log_std,
            eps,
            self.action_scale,
            self.action_bias,
        );
        SampleOutput { action, log_prob }
    }

    fn forward_sample_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        eps: Tensor<B::InnerBackend, 2>,
    ) -> SampleOutput<B::InnerBackend, 2> {
        let (mean, log_std) = inner.mean_and_log_std(obs);
        let (action, log_prob) = squashed_sample_log_prob::<B::InnerBackend>(
            mean,
            log_std,
            eps,
            inner.action_scale,
            inner.action_bias,
        );
        SampleOutput { action, log_prob }
    }

    fn deterministic_action(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.mean_action(obs)
    }
}

/// Draws `rows × cols` iid standard-normal samples on CPU and stacks them
/// into a `(rows, cols)` tensor on `device`. Callers own the RNG so the
/// sampled noise stays reproducible under a seeded `rand::Rng`.
pub fn standard_normal_tensor<B: Backend, R: rand::Rng + ?Sized>(
    rows: usize,
    cols: usize,
    device: &B::Device,
    rng: &mut R,
) -> Tensor<B, 2> {
    use rand_distr::{Distribution, StandardNormal};
    let mut data: Vec<f32> = Vec::with_capacity(rows * cols);
    let normal = StandardNormal;
    for _ in 0..(rows * cols) {
        let x: f64 = normal.sample(rng);
        data.push(x as f32);
    }
    Tensor::from_data(TensorData::new(data, vec![rows, cols]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::ElementConversion;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type B = Autodiff<NdArray>;
    type BI = NdArray;

    /// Pin μ=0, log_std=0, ε=0.5 (so z=0.5, σ=1) with scale=1, bias=0.
    /// Hand-rolled reference:
    ///   log N(0.5 | 0, 1)  = −0.5·log(2π) − 0.5·0.25
    ///   − log|1 − tanh²(0.5)| (two terms since action_dim=2 and we feed the
    ///   same values twice → just double the single-dim result).
    #[test]
    fn squashed_gaussian_logprob_matches_hand_roll_at_pinned_inputs() {
        let device = Default::default();
        let mean = Tensor::<BI, 2>::from_data(
            TensorData::new(vec![0.0_f32, 0.0], vec![1, 2]),
            &device,
        );
        let log_std = Tensor::<BI, 2>::from_data(
            TensorData::new(vec![0.0_f32, 0.0], vec![1, 2]),
            &device,
        );
        let eps = Tensor::<BI, 2>::from_data(
            TensorData::new(vec![0.5_f32, 0.5], vec![1, 2]),
            &device,
        );
        let (action, log_prob) = squashed_sample_log_prob::<BI>(
            mean, log_std, eps, 1.0, 0.0,
        );

        let z = 0.5_f32;
        let gauss_per_dim = -0.5 * (2.0_f32 * std::f32::consts::PI).ln()
            - 0.5 * z * z;
        let jac_per_dim = (1.0_f32 - z.tanh().powi(2)).ln();
        let expected = 2.0 * (gauss_per_dim - jac_per_dim);
        let got = log_prob.into_scalar().elem::<f32>();
        assert!(
            (got - expected).abs() < 1e-4,
            "expected {expected}, got {got}"
        );

        // Sanity-check the squashed action equals tanh(z) in both dims.
        let a = action.into_data().convert::<f32>();
        let slice = a.as_slice::<f32>().unwrap();
        let expected_a = z.tanh();
        assert!((slice[0] - expected_a).abs() < 1e-6);
        assert!((slice[1] - expected_a).abs() < 1e-6);
    }

    /// Evaluating `deterministic_action` yields `scale·tanh(μ) + bias` and
    /// ignores any ε / log_std contribution.
    #[test]
    fn deterministic_action_applies_scale_and_bias() {
        let device = Default::default();
        let cfg = SquashedGaussianPolicyHeadConfig {
            obs_dim: 1,
            hidden: 2,
            action_dim: 1,
            log_std_min: -5.0,
            log_std_max: 2.0,
            action_scale: 2.0,
            action_bias: 0.5,
        };
        let head: SquashedGaussianPolicyHead<B> = cfg.init::<B>(&device);
        let obs = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.1_f32], vec![1, 1]),
            &device,
        );
        let det = head.deterministic_action(obs.clone());
        let (mean, _) = head.mean_and_log_std(obs);
        let expected = tanh(mean).mul_scalar(2.0).add_scalar(0.5);
        let a = det.into_data().convert::<f32>();
        let b = expected.into_data().convert::<f32>();
        assert!(
            (a.as_slice::<f32>().unwrap()[0] - b.as_slice::<f32>().unwrap()[0]).abs() < 1e-6
        );
    }

    /// Two calls with the same ε produce identical samples and log-probs.
    #[test]
    fn forward_sample_is_deterministic_under_same_eps() {
        let device = Default::default();
        let cfg = SquashedGaussianPolicyHeadConfig {
            obs_dim: 2,
            hidden: 4,
            action_dim: 1,
            log_std_min: -5.0,
            log_std_max: 2.0,
            action_scale: 1.0,
            action_bias: 0.0,
        };
        let head: SquashedGaussianPolicyHead<B> = cfg.init::<B>(&device);
        let obs = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.2_f32, -0.3], vec![1, 2]),
            &device,
        );
        let mut rng = StdRng::seed_from_u64(9);
        let eps1: Tensor<B, 2> = standard_normal_tensor(1, 1, &device, &mut rng);
        let eps2 = eps1.clone();
        let o1 = head.forward_sample(obs.clone(), eps1);
        let o2 = head.forward_sample(obs, eps2);
        let a1 = o1.action.into_scalar().elem::<f32>();
        let a2 = o2.action.into_scalar().elem::<f32>();
        let lp1 = o1.log_prob.into_scalar().elem::<f32>();
        let lp2 = o2.log_prob.into_scalar().elem::<f32>();
        assert!((a1 - a2).abs() < 1e-6);
        assert!((lp1 - lp2).abs() < 1e-6);
    }
}
