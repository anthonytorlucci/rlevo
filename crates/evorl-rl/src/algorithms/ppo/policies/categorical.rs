//! Discrete (categorical) policy head for PPO.
//!
//! Two-layer MLP with `tanh` activations (matching CleanRL's discrete PPO
//! default) followed by a softmax over `num_actions` logits. Sampling is
//! done via **Gumbel-max on CPU** so the RNG is explicitly threaded and
//! bitwise-reproducible under the ndarray backend.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{log_softmax, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use rand::{Rng, RngExt};

use crate::algorithms::ppo::ppo_policy::{LogProbEntropy, PolicyOutput, PpoPolicy};

/// Construction-time knobs for [`CategoricalPolicyHead`].
#[derive(Debug, Clone)]
pub struct CategoricalPolicyHeadConfig {
    /// Observation feature count (flattened).
    pub obs_dim: usize,
    /// Hidden layer width. Applied to both hidden layers.
    pub hidden: usize,
    /// Number of discrete actions.
    pub num_actions: usize,
}

impl CategoricalPolicyHeadConfig {
    /// Constructs the module on `device` using Burn's default initializer.
    /// CleanRL's orthogonal-init detail is a deferred follow-up; users who
    /// want it can post-process the module via a `ModuleMapper`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> CategoricalPolicyHead<B> {
        CategoricalPolicyHead {
            fc1: LinearConfig::new(self.obs_dim, self.hidden).init(device),
            fc2: LinearConfig::new(self.hidden, self.hidden).init(device),
            logits: LinearConfig::new(self.hidden, self.num_actions).init(device),
            num_actions: self.num_actions,
        }
    }
}

/// Two-hidden-layer MLP whose final head produces softmax logits over a
/// discrete action space.
#[derive(Module, Debug)]
pub struct CategoricalPolicyHead<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    logits: Linear<B>,
    num_actions: usize,
}

impl<B: Backend> CategoricalPolicyHead<B> {
    /// Forward pass producing raw logits of shape `(batch, num_actions)`.
    pub fn logits(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = tanh(self.fc1.forward(obs));
        let h = tanh(self.fc2.forward(h));
        self.logits.forward(h)
    }

    /// Number of discrete actions this head was built for.
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }
}

impl<B: AutodiffBackend> PpoPolicy<B, 2> for CategoricalPolicyHead<B> {
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

        // Forward through the autodiff module; we materialise logits on CPU to
        // sample, but also keep the autodiff tensor around so that the returned
        // log_prob / entropy stay differentiable (PPO uses these at update
        // time via `evaluate`, not directly from `sample_with_logprob`, so the
        // graph isn't actually needed — but keeping the tensors as
        // `Tensor<B, 1>` keeps the trait surface consistent).
        let logits = self.logits(obs);
        let log_probs = log_softmax(logits.clone(), 1);

        // CPU-side Gumbel-max sampling for bitwise reproducibility under
        // ndarray. We read the logits data (cheap clone: ref-counted in Burn),
        // add Gumbel(0,1) noise, argmax per row.
        let logits_data = logits.clone().into_data().convert::<f32>();
        let logits_slice = logits_data
            .as_slice::<f32>()
            .expect("categorical head: logits are f32");

        let mut sampled = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut best_i = 0_usize;
            let mut best_v = f32::NEG_INFINITY;
            let row = &logits_slice[b * num_actions..(b + 1) * num_actions];
            for (i, &l) in row.iter().enumerate() {
                // Gumbel(0, 1) = -ln(-ln U) where U ~ Uniform(0, 1).
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

        let action_1d: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(sampled, vec![batch]),
            &device,
        );
        let action_2d: Tensor<B, 2, Int> = action_1d.unsqueeze_dim::<2>(1);

        // log_prob of the sampled action per row.
        let gathered = log_probs.clone().gather(1, action_2d.clone());
        let log_prob = gathered.squeeze_dim::<1>(1);

        // Entropy = -Σ p · log p per row.
        let probs = log_probs.clone().exp();
        let entropy = (probs * log_probs).sum_dim(1).squeeze_dim::<1>(1).neg();

        PolicyOutput {
            action: action_2d,
            log_prob,
            entropy,
        }
    }

    fn evaluate(&self, obs: Tensor<B, 2>, actions: Self::ActionTensor) -> LogProbEntropy<B> {
        let logits = self.logits(obs);
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
            .expect("categorical action tensor is i64");
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
            "categorical: action_dim=1, so len must equal n_rows"
        );
        let as_i64: Vec<i64> = flat.iter().map(|&x| x as i64).collect();
        let t: Tensor<B, 1, Int> = Tensor::from_data(TensorData::new(as_i64, vec![n_rows]), device);
        t.unsqueeze_dim::<2>(1)
    }
}

/// Convenience converter: extract a `DiscreteAction` value from a single
/// sampled action row produced by [`CategoricalPolicyHead`].
pub fn discrete_action_from_row<const AD: usize, A: evorl_core::action::DiscreteAction<AD>>(
    row: &[f32],
) -> A {
    assert_eq!(row.len(), 1, "discrete action row must have exactly 1 component");
    let idx = row[0] as usize;
    A::from_index(idx)
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
    fn categorical_logprob_consistency() {
        let device = Default::default();
        let cfg = CategoricalPolicyHeadConfig {
            obs_dim: 4,
            hidden: 8,
            num_actions: 3,
        };
        let head: CategoricalPolicyHead<B> = cfg.init::<B>(&device);

        // Deterministic obs.
        let obs: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(vec![0.1_f32, -0.2, 0.3, 0.4], vec![1, 4]),
            &device,
        );
        let mut rng = StdRng::seed_from_u64(7);
        let out = head.sample_with_logprob(obs.clone(), &mut rng);

        // Re-evaluate on the sampled action; log_prob should match.
        let eval = head.evaluate(obs, out.action.clone());
        let a = out.log_prob.into_scalar().elem::<f32>();
        let b = eval.log_prob.into_scalar().elem::<f32>();
        assert!(
            (a - b).abs() < 1e-5,
            "sample log_prob {a} vs evaluate log_prob {b}"
        );
        assert!(
            (out.entropy.into_scalar().elem::<f32>()
                - eval.entropy.into_scalar().elem::<f32>())
            .abs()
                < 1e-5
        );
    }

    #[test]
    fn categorical_round_trips_action_rows() {
        let device = Default::default();
        // Build a toy action tensor of shape (3, 1).
        let a1d: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(vec![0_i64, 2, 1], vec![3]), &device);
        let a2d: Tensor<B, 2, Int> = a1d.unsqueeze_dim::<2>(1);

        let row0 = <CategoricalPolicyHead<B> as PpoPolicy<B, 2>>::action_row_from_tensor(&a2d, 0);
        let row2 = <CategoricalPolicyHead<B> as PpoPolicy<B, 2>>::action_row_from_tensor(&a2d, 2);
        assert_eq!(row0, vec![0.0]);
        assert_eq!(row2, vec![1.0]);

        let rebuilt = <CategoricalPolicyHead<B> as PpoPolicy<B, 2>>::action_tensor_from_flat(
            &[0.0, 2.0, 1.0],
            3,
            &device,
        );
        let data = rebuilt.into_data().convert::<i64>();
        let slice = data.as_slice::<i64>().unwrap();
        assert_eq!(slice, &[0, 2, 1]);
    }
}
