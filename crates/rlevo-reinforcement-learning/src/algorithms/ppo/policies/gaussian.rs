//! Continuous (tanh-squashed Gaussian) policy head for PPO.
//!
//! Two-layer MLP with `tanh` activations, a linear mean head, and a
//! state-independent `log_std` parameter (length `action_dim`). Sampling:
//! `z = μ + σ·ε` with `ε ∼ N(0, 1)` drawn on CPU; the env receives
//! `a = scale · tanh(z)`, keeping it within the bounded action space.
//!
//! # z vs a in the rollout
//!
//! The buffer stores the **pre-squash** sample `z`, and
//! [`evaluate`](PpoPolicy::evaluate) computes log-probability on `z` under
//! the current policy. The tanh Jacobian term `Σ log(1 − tanh²(z))` is the
//! same under old and new policies and therefore cancels in the PPO
//! importance ratio — so we work entirely in Gaussian-on-`z` space, which is
//! numerically stabler than the `atanh`-from-squashed-action path.
//!
//! # Entropy
//!
//! Returns the Gaussian entropy `Σ (log σ + ½ log(2πe))`, summed across
//! action dims. This matches CleanRL's `probs.entropy().sum(1)` — the tanh
//! Jacobian's entropy contribution is omitted, consistent with PPO practice.

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};
use rand::Rng;
use rand_distr::{Distribution as RandDistribution, StandardNormal};

use crate::algorithms::ppo::ppo_policy::{LogProbEntropy, PolicyOutput, PpoPolicy};

/// Construction-time knobs for [`TanhGaussianPolicyHead`].
#[derive(Debug, Clone)]
pub struct TanhGaussianPolicyHeadConfig {
    /// Observation feature count.
    pub obs_dim: usize,
    /// Hidden layer width (applied to both hidden layers).
    pub hidden: usize,
    /// Number of continuous action dimensions.
    pub action_dim: usize,
    /// Initial value for every entry of the state-independent `log_std`.
    pub log_std_init: f32,
    /// Multiplier applied to `tanh(z)` before the env sees the action.
    pub action_scale: f32,
}

impl TanhGaussianPolicyHeadConfig {
    /// Constructs the module on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TanhGaussianPolicyHead<B> {
        let log_std_vec: Vec<f32> = vec![self.log_std_init; self.action_dim];
        let log_std: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(log_std_vec, vec![self.action_dim]), device);
        TanhGaussianPolicyHead {
            fc1: LinearConfig::new(self.obs_dim, self.hidden).init(device),
            fc2: LinearConfig::new(self.hidden, self.hidden).init(device),
            mean: LinearConfig::new(self.hidden, self.action_dim).init(device),
            log_std: Param::from_tensor(log_std),
            action_dim: self.action_dim,
            action_scale: self.action_scale,
        }
    }
}

/// MLP → Gaussian mean, with state-independent `log_std`, squashed via
/// `scale · tanh(z)` at the env boundary.
#[derive(Module, Debug)]
pub struct TanhGaussianPolicyHead<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    mean: Linear<B>,
    log_std: Param<Tensor<B, 1>>,
    action_dim: usize,
    action_scale: f32,
}

impl<B: Backend> TanhGaussianPolicyHead<B> {
    /// Forward pass to the Gaussian mean of shape `(batch, action_dim)`.
    pub fn mean(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = tanh(self.fc1.forward(obs));
        let h = tanh(self.fc2.forward(h));
        self.mean.forward(h)
    }

    /// The current `log σ` vector of length `action_dim`.
    pub fn log_std_vec(&self) -> Tensor<B, 1> {
        self.log_std.val()
    }

    /// Action dimension `A`.
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Tanh scale applied to the pre-squash sample.
    pub fn action_scale(&self) -> f32 {
        self.action_scale
    }

    /// Computes per-row Gaussian log-prob and entropy of `z` under the
    /// current policy (no tanh Jacobian; it cancels in the PPO ratio).
    fn log_prob_entropy(&self, obs: Tensor<B, 2>, z: Tensor<B, 2>) -> LogProbEntropy<B> {
        let [batch, _] = z.dims();
        let mean = self.mean(obs);
        // Broadcast log_std of shape (A,) to (batch, A) via repeat_dim.
        let log_std_row: Tensor<B, 2> = self.log_std.val().unsqueeze_dim::<2>(0);
        let log_std: Tensor<B, 2> = log_std_row.repeat_dim(0, batch);
        let std = log_std.clone().exp();

        // log N(z | μ, σ) = -0.5·((z-μ)/σ)² - log σ - 0.5 log 2π
        let centered = z - mean;
        let scaled = centered.clone() / std.clone();
        let scaled_sq = scaled.clone() * scaled;
        let log_2pi = (2.0_f32 * std::f32::consts::PI).ln();
        let per_dim: Tensor<B, 2> = scaled_sq.mul_scalar(-0.5) - log_std.clone() - log_2pi * 0.5;
        // Sum over action dim → (batch,).
        let log_prob = per_dim.sum_dim(1).squeeze_dim::<1>(1);

        // Gaussian entropy per dim: 0.5·log(2πe) + log σ.
        let log_2pi_e = log_2pi + 1.0;
        let entropy_per_dim = log_std + log_2pi_e * 0.5;
        let entropy = entropy_per_dim.sum_dim(1).squeeze_dim::<1>(1);
        LogProbEntropy { log_prob, entropy }
    }
}

impl<B: AutodiffBackend> PpoPolicy<B, 2> for TanhGaussianPolicyHead<B> {
    type ActionTensor = Tensor<B, 2>;

    fn action_dim(&self) -> usize {
        self.action_dim
    }

    fn sample_with_logprob(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut dyn Rng,
    ) -> PolicyOutput<B, Self::ActionTensor> {
        let device = obs.device();
        let [batch, _] = obs.dims();
        let action_dim = self.action_dim;

        // Draw ε ~ N(0, 1) on CPU for reproducibility.
        let mut eps_vec: Vec<f32> = Vec::with_capacity(batch * action_dim);
        let normal = StandardNormal;
        for _ in 0..(batch * action_dim) {
            let x: f64 = normal.sample(rng);
            eps_vec.push(x as f32);
        }
        let eps: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(eps_vec, vec![batch, action_dim]), &device);

        let mean = self.mean(obs.clone());
        let log_std_row: Tensor<B, 2> = self.log_std.val().unsqueeze_dim::<2>(0);
        let log_std: Tensor<B, 2> = log_std_row.repeat_dim(0, batch);
        let std = log_std.exp();
        // z = μ + σ·ε
        let z = mean + std * eps;

        let lp_ent = self.log_prob_entropy(obs, z.clone());
        PolicyOutput {
            action: z,
            log_prob: lp_ent.log_prob,
            entropy: lp_ent.entropy,
        }
    }

    fn evaluate(&self, obs: Tensor<B, 2>, actions: Self::ActionTensor) -> LogProbEntropy<B> {
        self.log_prob_entropy(obs, actions)
    }

    fn action_row_from_tensor(action: &Self::ActionTensor, row: usize) -> Vec<f32> {
        let data = action.clone().into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().expect("gaussian action is f32");
        let [_, action_dim] = action.dims();
        let start = row * action_dim;
        slice[start..start + action_dim].to_vec()
    }

    fn action_tensor_from_flat(
        flat: &[f32],
        n_rows: usize,
        device: &B::Device,
    ) -> Self::ActionTensor {
        let action_dim = flat.len() / n_rows.max(1);
        Tensor::<B, 2>::from_data(
            TensorData::new(flat.to_vec(), vec![n_rows, action_dim]),
            device,
        )
    }

    fn raw_to_env_row(&self, raw_row: &[f32]) -> Vec<f32> {
        raw_row
            .iter()
            .map(|z| self.action_scale * z.tanh())
            .collect()
    }
}

/// Convenience: build a `ContinuousAction` from the env-action row produced
/// by [`TanhGaussianPolicyHead::raw_to_env_row`].
pub fn continuous_action_from_row<const AD: usize, A: rlevo_core::action::ContinuousAction<AD>>(
    row: &[f32],
) -> A {
    A::from_slice(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::ElementConversion;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type B = Autodiff<NdArray>;

    #[test]
    fn gaussian_logprob_consistency_between_sample_and_evaluate() {
        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            obs_dim: 3,
            hidden: 8,
            action_dim: 1,
            log_std_init: 0.0,
            action_scale: 2.0,
        };
        let head: TanhGaussianPolicyHead<B> = cfg.init::<B>(&device);
        let obs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(vec![0.1_f32, -0.2, 0.3], vec![1, 3]),
            &device,
        );
        let mut rng = StdRng::seed_from_u64(17);
        let out = head.sample_with_logprob(obs.clone(), &mut rng);
        let eval = head.evaluate(obs, out.action.clone());
        let a = out.log_prob.into_scalar().elem::<f32>();
        let b = eval.log_prob.into_scalar().elem::<f32>();
        assert!((a - b).abs() < 1e-5, "sample {a} vs evaluate {b}");
    }

    #[test]
    fn gaussian_logprob_at_mean_matches_reference() {
        // With μ=0, σ=1, z=0, per-dim log N(0|0,1) = -0.5·log(2π).
        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            obs_dim: 1,
            hidden: 2,
            action_dim: 2,
            log_std_init: 0.0,
            action_scale: 1.0,
        };
        let head: TanhGaussianPolicyHead<B> = cfg.init::<B>(&device);
        // Zero-out the mean MLP by constructing obs that yields a nonzero
        // mean — skip: use z = mean(obs). We simply check sample log_prob
        // equals evaluate on that z.
        let obs: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(vec![0.0_f32], vec![1, 1]), &device);
        let mean = head.mean(obs.clone());
        let eval = head.evaluate(obs, mean.clone());
        // Expected per dim: −0.5·log(2π), sum across dim=2 → −log(2π) ≈ −1.8379.
        let expected = -(2.0_f32 * std::f32::consts::PI).ln();
        let got = eval.log_prob.into_scalar().elem::<f32>();
        assert!(
            (got - expected).abs() < 1e-4,
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn raw_to_env_row_applies_tanh_scale() {
        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            obs_dim: 1,
            hidden: 2,
            action_dim: 1,
            log_std_init: 0.0,
            action_scale: 2.0,
        };
        let head: TanhGaussianPolicyHead<B> = cfg.init::<B>(&device);
        let env_row = head.raw_to_env_row(&[0.5_f32]);
        let expected = 2.0 * 0.5_f32.tanh();
        assert!((env_row[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn gaussian_entropy_matches_gaussian_formula_at_sigma_one() {
        let device = Default::default();
        let cfg = TanhGaussianPolicyHeadConfig {
            obs_dim: 1,
            hidden: 2,
            action_dim: 2,
            log_std_init: 0.0, // σ = 1
            action_scale: 1.0,
        };
        let head: TanhGaussianPolicyHead<B> = cfg.init::<B>(&device);
        let obs: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(vec![0.0_f32], vec![1, 1]), &device);
        let mean = head.mean(obs.clone());
        let eval = head.evaluate(obs, mean);
        // Per dim entropy at σ=1 is 0 + 0.5·log(2πe); two dims summed.
        let expected = 2.0 * 0.5 * ((2.0_f32 * std::f32::consts::PI).ln() + 1.0);
        let got = eval.entropy.into_scalar().elem::<f32>();
        assert!(
            (got - expected).abs() < 1e-5,
            "expected {expected}, got {got}"
        );
    }
}
