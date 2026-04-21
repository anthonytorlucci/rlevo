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

use crate::algorithms::ppg::ppg_policy::PpgAuxValueHead;
use crate::algorithms::ppo::ppo_policy::{LogProbEntropy, PolicyOutput, PpoPolicy};

/// Construction-time knobs for [`PpgCategoricalPolicyHead`].
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> PpgCategoricalPolicyHead<B> {
        PpgCategoricalPolicyHead {
            fc1: LinearConfig::new(self.obs_dim, self.hidden).init(device),
            fc2: LinearConfig::new(self.hidden, self.hidden).init(device),
            logits_head: LinearConfig::new(self.hidden, self.num_actions).init(device),
            aux_value_head: LinearConfig::new(self.hidden, 1).init(device),
            num_actions: self.num_actions,
        }
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

    fn action_dim(&self) -> usize {
        1
    }

    fn sample_with_logprob(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut dyn Rng,
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

    fn evaluate(&self, obs: Tensor<B, 2>, actions: Self::ActionTensor) -> LogProbEntropy<B> {
        let logits = PpgAuxValueHead::logits(self, obs);
        let log_probs = log_softmax(logits, 1);
        let gathered = log_probs.clone().gather(1, actions);
        let log_prob = gathered.squeeze_dim::<1>(1);
        let probs = log_probs.clone().exp();
        let entropy = (probs * log_probs).sum_dim(1).squeeze_dim::<1>(1).neg();
        LogProbEntropy { log_prob, entropy }
    }

    fn action_row_from_tensor(action: &Self::ActionTensor, row: usize) -> Vec<f32> {
        let data = action.clone().into_data().convert::<i64>();
        let slice = data
            .as_slice::<i64>()
            .expect("ppg categorical action tensor is i64");
        vec![slice[row] as f32]
    }

    fn action_tensor_from_flat(
        flat: &[f32],
        n_rows: usize,
        device: &B::Device,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::ElementConversion;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type B = Autodiff<NdArray>;

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
        let device: <B as Backend>::Device = Default::default();
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
