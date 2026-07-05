//! Discrete (categorical) policy head for Phasic Policy Gradient.
//!
//! Structurally identical to the PPO
//! [`CategoricalPolicyHead`](crate::algorithms::ppo::policies::CategoricalPolicyHead):
//! a two-layer `tanh` MLP feeding a softmax-logits head. PPG adds an
//! **auxiliary value head** `aux_value` fed off the same trunk; it is trained
//! only in the auxiliary phase and carries no gradient during the policy
//! phase.
//!
//! Sampling uses CPU-side Gumbel-max for the same bitwise-reproducibility
//! reason as the PPO categorical head.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{log_softmax, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use rand::{Rng, RngExt};
use rlevo_core::config::{self, ConfigError, Validate};

use crate::algorithms::ppg::ppg_policy::PpgAuxValueHead;
use crate::algorithms::ppo::ppo_policy::{LogProbEntropy, PolicyOutput, PpoPolicy};

/// Construction-time configuration for [`PpgCategoricalPolicyHead`].
///
/// Specifies the observation dimensionality, hidden layer width, and action
/// count. All three values are needed before the Burn module can be
/// initialised because Burn's linear layers require static shapes.
#[derive(Debug, Clone)]
pub struct PpgCategoricalPolicyHeadConfig {
    /// Observation feature count (flattened).
    pub obs_dim: usize,
    /// Hidden layer width. Applied to both hidden layers.
    pub hidden: usize,
    /// Number of discrete actions.
    pub num_actions: usize,
}

impl PpgCategoricalPolicyHeadConfig {
    /// Constructs the module on `device` using Burn's default initializer.
    pub fn init<B: Backend>(&self, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> PpgCategoricalPolicyHead<B> {
        PpgCategoricalPolicyHead {
            fc1: LinearConfig::new(self.obs_dim, self.hidden).init(device),
            fc2: LinearConfig::new(self.hidden, self.hidden).init(device),
            logits_head: LinearConfig::new(self.hidden, self.num_actions).init(device),
            aux_value_head: LinearConfig::new(self.hidden, 1).init(device),
            num_actions: self.num_actions,
        }
    }
}

impl Validate for PpgCategoricalPolicyHeadConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "PpgCategoricalPolicyHeadConfig";
        config::nonzero(C, "obs_dim", self.obs_dim)?;
        config::nonzero(C, "hidden", self.hidden)?;
        config::nonzero(C, "num_actions", self.num_actions)?;
        Ok(())
    }
}

/// Two-hidden-layer MLP with a softmax policy head and a scalar auxiliary
/// value head sharing the trunk.
#[derive(Module, Debug)]
pub struct PpgCategoricalPolicyHead<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    logits_head: Linear<B>,
    aux_value_head: Linear<B>,
    num_actions: usize,
}

impl<B: Backend> PpgCategoricalPolicyHead<B> {
    /// Number of discrete actions this head was built for.
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Shared two-layer `tanh` trunk: `tanh(W2 · tanh(W1 · obs))`.
    ///
    /// Both the logits head and the auxiliary value head branch off this
    /// intermediate representation.
    fn trunk(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = tanh(self.fc1.forward(obs));
        tanh(self.fc2.forward(h))
    }
}

impl<B: AutodiffBackend> PpgAuxValueHead<B, 2> for PpgCategoricalPolicyHead<B> {
    fn aux_value(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        let h = self.trunk(obs);
        self.aux_value_head.forward(h).squeeze_dim::<1>(1)
    }

    fn logits(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.trunk(obs);
        self.logits_head.forward(h)
    }
}

impl<B: AutodiffBackend> PpoPolicy<B, 2> for PpgCategoricalPolicyHead<B> {
    type ActionTensor = Tensor<B, 2, Int>;

    /// Always `1`: one discrete action index per step.
    fn action_dim(&self) -> usize {
        1
    }

    /// Samples one action per batch row using the Gumbel-max trick.
    ///
    /// Gumbel-max sampling (`argmax(logits + Gumbel noise)`) is equivalent to
    /// categorical sampling from `softmax(logits)` but avoids materialising
    /// a full probability vector on the accelerator. Noise is drawn from the
    /// caller-supplied `rng` so action selection is fully reproducible given
    /// the same seed — consistent with the host-RNG convention used across
    /// `rlevo-evolution` and `rlevo-reinforcement-learning`.
    ///
    /// Returns a [`PolicyOutput`] containing:
    /// - `action` — shape `(batch, 1)`, integer action index.
    /// - `log_prob` — shape `(batch,)`, log-probability of the sampled action.
    /// - `entropy` — shape `(batch,)`, per-row categorical entropy.
    fn sample_with_logprob<R: Rng + ?Sized>(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut R,
    ) -> PolicyOutput<B, Self::ActionTensor> {
        let device = obs.device();
        let [batch, _] = obs.dims();
        let num_actions = self.num_actions;

        let logits = PpgAuxValueHead::logits(self, obs);
        let log_probs = log_softmax(logits.clone(), 1);

        let logits_data = logits.into_data().convert::<f32>();
        let logits_slice = logits_data
            .as_slice::<f32>()
            .expect("ppg categorical head: logits are f32");

        let mut sampled = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut best_i = 0_usize;
            let mut best_v = f32::NEG_INFINITY;
            let row = &logits_slice[b * num_actions..(b + 1) * num_actions];
            for (i, &l) in row.iter().enumerate() {
                let u: f32 = rng.random_range(1e-20_f32..1.0);
                let g = -(-u.ln()).ln();
                let v = l + g;
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            sampled.push(best_i as i64);
        }

        let action_1d: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(sampled, vec![batch]), &device);
        let action_2d: Tensor<B, 2, Int> = action_1d.unsqueeze_dim::<2>(1);

        let gathered = log_probs.clone().gather(1, action_2d.clone());
        let log_prob = gathered.squeeze_dim::<1>(1);

        let probs = log_probs.clone().exp();
        let entropy = (probs * log_probs).sum_dim(1).squeeze_dim::<1>(1).neg();

        PolicyOutput {
            action: action_2d,
            log_prob,
            entropy,
        }
    }

    /// Evaluates stored `actions` against the current policy to produce
    /// log-probabilities and entropy used by the clipped surrogate loss.
    ///
    /// `actions` has shape `(batch, 1)` and contains integer action indices;
    /// the function gathers the corresponding log-prob column from
    /// `log_softmax(logits)`.
    fn evaluate(&self, obs: Tensor<B, 2>, actions: Self::ActionTensor) -> LogProbEntropy<B> {
        let logits = PpgAuxValueHead::logits(self, obs);
        let log_probs = log_softmax(logits, 1);
        let gathered = log_probs.clone().gather(1, actions);
        let log_prob = gathered.squeeze_dim::<1>(1);
        let probs = log_probs.clone().exp();
        let entropy = (probs * log_probs).sum_dim(1).squeeze_dim::<1>(1).neg();
        LogProbEntropy { log_prob, entropy }
    }

    /// Extracts one row from an `(N, 1)` integer action tensor as a `Vec<f32>`.
    ///
    /// The returned vector always has length `1` (one action index). The `f32`
    /// cast is lossless for action indices within the 24-bit float mantissa
    /// range, which covers all practical discrete action spaces.
    fn action_row_from_tensor(action: &Self::ActionTensor, row: usize) -> Vec<f32> {
        let data = action.clone().into_data().convert::<i64>();
        let slice = data
            .as_slice::<i64>()
            .expect("ppg categorical action tensor is i64");
        vec![slice[row] as f32]
    }

    /// Rebuilds an `(n_rows, 1)` integer action tensor from a flat `f32` slice.
    ///
    /// `flat` must have exactly `n_rows` elements (one index per row).
    ///
    /// # Panics
    ///
    /// Panics if `flat.len() != n_rows`.
    fn action_tensor_from_flat(
        flat: &[f32],
        n_rows: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::ActionTensor {
        assert_eq!(
            flat.len(),
            n_rows,
            "ppg categorical: action_dim=1, so len must equal n_rows"
        );
        let as_i64: Vec<i64> = flat.iter().map(|&x| x as i64).collect();
        let t: Tensor<B, 1, Int> = Tensor::from_data(TensorData::new(as_i64, vec![n_rows]), device);
        t.unsqueeze_dim::<2>(1)
    }

    /// Deterministic action on the inner (non-autodiff) backend — the argmax
    /// over the policy logits.
    ///
    /// Operates on `B::InnerBackend` to avoid building an autodiff graph during
    /// evaluation. Because [`PpgAuxValueHead::logits`] is only defined over
    /// `AutodiffBackend`, the trunk and logits head are accessed directly
    /// through the inner module's fields rather than via the trait method.
    fn deterministic_env_row_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Vec<f32> {
        // Deterministic action = argmax over logits (the categorical mode).
        // `PpgAuxValueHead::logits` is autodiff-only, so recompute logits on the
        // inner backend from the shared trunk + logits head directly.
        let logits = inner.logits_head.forward(inner.trunk(obs));
        let idx: Tensor<B::InnerBackend, 2, Int> = logits.argmax(1);
        let data = idx.into_data().convert::<i64>();
        let slice = data.as_slice::<i64>().expect("argmax index is i64");
        vec![slice[0] as f32]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Flex};
    use burn::tensor::ElementConversion;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type B = Autodiff<Flex>;

    #[test]
    fn representative_head_config_is_valid() {
        let cfg = PpgCategoricalPolicyHeadConfig { obs_dim: 4, hidden: 64, num_actions: 2 };
        assert!(cfg.validate().is_ok());
    }

    fn head() -> PpgCategoricalPolicyHead<B> {
        let device = Default::default();
        PpgCategoricalPolicyHeadConfig {
            obs_dim: 4,
            hidden: 8,
            num_actions: 3,
        }
        .init::<B>(&device)
    }

    fn obs(batch: usize) -> Tensor<B, 2> {
        let device = Default::default();
        let data: Vec<f32> = (0..batch * 4).map(|i| 0.01 * (i as f32)).collect();
        Tensor::<B, 2>::from_data(TensorData::new(data, vec![batch, 4]), &device)
    }

    #[test]
    fn aux_value_forward_shape_is_batch() {
        let h = head();
        let out = h.aux_value(obs(5));
        assert_eq!(out.dims(), [5]);
    }

    #[test]
    fn logits_shape_matches_num_actions() {
        let h = head();
        let out = PpgAuxValueHead::logits(&h, obs(5));
        assert_eq!(out.dims(), [5, 3]);
    }

    #[test]
    fn ppg_categorical_logprob_consistency() {
        let h = head();
        let o = obs(1);
        let mut rng = StdRng::seed_from_u64(11);
        let out = h.sample_with_logprob(o.clone(), &mut rng);
        let eval = h.evaluate(o, out.action.clone());
        let a = out.log_prob.into_scalar().elem::<f32>();
        let b = eval.log_prob.into_scalar().elem::<f32>();
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }

    #[test]
    fn ppg_categorical_round_trips_action_rows() {
        let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let a1d: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(vec![0_i64, 2, 1], vec![3]), &device);
        let a2d: Tensor<B, 2, Int> = a1d.unsqueeze_dim::<2>(1);
        let row0 =
            <PpgCategoricalPolicyHead<B> as PpoPolicy<B, 2>>::action_row_from_tensor(&a2d, 0);
        let row2 =
            <PpgCategoricalPolicyHead<B> as PpoPolicy<B, 2>>::action_row_from_tensor(&a2d, 2);
        assert_eq!(row0, vec![0.0]);
        assert_eq!(row2, vec![1.0]);
        let rebuilt = <PpgCategoricalPolicyHead<B> as PpoPolicy<B, 2>>::action_tensor_from_flat(
            &[0.0, 2.0, 1.0],
            3,
            &device,
        );
        let data = rebuilt.into_data().convert::<i64>();
        let slice = data.as_slice::<i64>().unwrap();
        assert_eq!(slice, &[0, 2, 1]);
    }
}
