//! On-policy rollout buffer and Generalized Advantage Estimation (GAE).
//!
//! The buffer is CPU-resident — observations stay owned, action components and
//! scalar bookkeeping are plain `Vec<f32>` / `Vec<bool>`. At minibatch time the
//! agent rebuilds device tensors via [`RolloutBuffer::indices_shuffled`] and
//! the typed accessors. This matches the C51 replay buffer's
//! "store-on-CPU, materialise-on-device" pattern and avoids holding a live
//! autodiff graph on the GPU across multiple update epochs.
//!
//! # GAE
//!
//! [`compute_gae`] follows Schulman et al. 2016 with CleanRL's convention that
//! a single `done` flag (terminated OR truncated) zeros the bootstrap at
//! reset boundaries within the rollout. See the doc-comment on that function
//! for caveats on truncation.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use std::marker::PhantomData;

use evorl_core::environment::EpisodeStatus;

/// Fixed-capacity on-policy rollout buffer.
///
/// Storage layout is struct-of-arrays: `obs[i]`, `action_flat[i · action_dim + j]`,
/// `log_probs[i]`, etc. all index the same step `i`. Advantages and returns
/// are only populated after [`RolloutBuffer::finish`].
#[derive(Debug)]
pub struct RolloutBuffer<B: Backend, O> {
    capacity: usize,
    action_dim: usize,
    obs: Vec<O>,
    action_flat: Vec<f32>,
    log_probs: Vec<f32>,
    values: Vec<f32>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
    _marker: PhantomData<B>,
}

impl<B: Backend, O: Clone> RolloutBuffer<B, O> {
    /// Allocates a buffer for `capacity` steps, with `action_dim` action
    /// components per step (1 for discrete, A for continuous-A).
    pub fn new(capacity: usize, action_dim: usize) -> Self {
        Self {
            capacity,
            action_dim,
            obs: Vec::with_capacity(capacity),
            action_flat: Vec::with_capacity(capacity * action_dim),
            log_probs: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            terminated: Vec::with_capacity(capacity),
            truncated: Vec::with_capacity(capacity),
            advantages: Vec::new(),
            returns: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Appends one step. `action_row.len()` must equal `action_dim`.
    pub fn push_step(
        &mut self,
        obs: O,
        action_row: &[f32],
        log_prob: f32,
        value: f32,
        reward: f32,
        status: EpisodeStatus,
    ) {
        assert_eq!(
            action_row.len(),
            self.action_dim,
            "action_row len {} ≠ action_dim {}",
            action_row.len(),
            self.action_dim,
        );
        self.obs.push(obs);
        self.action_flat.extend_from_slice(action_row);
        self.log_probs.push(log_prob);
        self.values.push(value);
        self.rewards.push(reward);
        self.terminated.push(status.is_terminated());
        self.truncated.push(status.is_truncated());
    }

    /// Finalises the rollout: computes GAE advantages and returns.
    ///
    /// `last_value` is `V(s_final)` bootstrap (typically the value-network's
    /// prediction on the last observation); `last_done` is whether the final
    /// step's snapshot was done.
    pub fn finish(&mut self, last_value: f32, last_done: bool, gamma: f32, gae_lambda: f32) {
        let (advs, rets) = compute_gae(
            &self.rewards,
            &self.values,
            &self.terminated,
            &self.truncated,
            last_value,
            last_done,
            gamma,
            gae_lambda,
        );
        self.advantages = advs;
        self.returns = rets;
    }

    /// Current number of stored steps.
    pub fn len(&self) -> usize {
        self.obs.len()
    }

    /// `true` if no steps have been pushed.
    pub fn is_empty(&self) -> bool {
        self.obs.is_empty()
    }

    /// Capacity, i.e. `num_envs · num_steps`.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Action components per step.
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Shuffled step indices for one epoch. Uses the provided RNG.
    pub fn indices_shuffled<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec<usize> {
        use rand::seq::SliceRandom;
        let mut idx: Vec<usize> = (0..self.len()).collect();
        idx.shuffle(rng);
        idx
    }

    /// Typed observations in step order.
    pub fn obs(&self) -> &[O] {
        &self.obs
    }

    /// Raw action components, shape `(num_steps, action_dim)` in row-major order.
    pub fn action_flat(&self) -> &[f32] {
        &self.action_flat
    }

    /// Log-probabilities captured at rollout-time under the sampling policy.
    pub fn log_probs(&self) -> &[f32] {
        &self.log_probs
    }

    /// Value-network predictions captured at rollout time.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// GAE advantages. Empty until [`Self::finish`] has been called.
    pub fn advantages(&self) -> &[f32] {
        &self.advantages
    }

    /// Bootstrap returns: advantages + values. Empty until
    /// [`Self::finish`] has been called.
    pub fn returns(&self) -> &[f32] {
        &self.returns
    }

    /// Builds a 1-D float tensor indexed by `indices`, on `device`.
    pub fn gather_f32(&self, data: &[f32], indices: &[usize], device: &B::Device) -> Tensor<B, 1> {
        let gathered: Vec<f32> = indices.iter().map(|&i| data[i]).collect();
        let n = gathered.len();
        Tensor::<B, 1>::from_data(TensorData::new(gathered, vec![n]), device)
    }

    /// Extracts the action rows for `indices` as a flat `Vec<f32>` of length
    /// `indices.len() * action_dim`.
    pub fn gather_action_flat(&self, indices: &[usize]) -> Vec<f32> {
        let mut out = Vec::with_capacity(indices.len() * self.action_dim);
        for &i in indices {
            let start = i * self.action_dim;
            let end = start + self.action_dim;
            out.extend_from_slice(&self.action_flat[start..end]);
        }
        out
    }

    /// Clears all stored steps and derived advantages/returns. Retains the
    /// underlying Vec capacity for the next iteration.
    pub fn clear(&mut self) {
        self.obs.clear();
        self.action_flat.clear();
        self.log_probs.clear();
        self.values.clear();
        self.rewards.clear();
        self.terminated.clear();
        self.truncated.clear();
        self.advantages.clear();
        self.returns.clear();
    }
}

/// Generalized Advantage Estimation.
///
/// Returns `(advantages, returns)` where `returns[t] = advantages[t] + values[t]`.
///
/// # Done-handling caveat
///
/// Within the rollout a step's `done` flag is `terminated || truncated`; both
/// zero the bootstrap at reset boundaries. Strictly, a *truncated* step should
/// bootstrap from `V(s_continuation)`, which isn't available once the env has
/// reset. Matching CleanRL's default PPO, we accept this small bias on
/// truncation-heavy envs. The terminated/truncated arrays are kept separate
/// so a future revision can rework this without changing the buffer API.
#[allow(clippy::too_many_arguments)]
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    terminated: &[bool],
    truncated: &[bool],
    last_value: f32,
    last_done: bool,
    gamma: f32,
    gae_lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    assert_eq!(values.len(), n);
    assert_eq!(terminated.len(), n);
    assert_eq!(truncated.len(), n);
    let mut advantages = vec![0.0_f32; n];
    let mut returns = vec![0.0_f32; n];

    let mut last_gae_lam = 0.0_f32;
    for t in (0..n).rev() {
        let (next_nonterminal, next_value) = if t == n - 1 {
            (if last_done { 0.0 } else { 1.0 }, last_value)
        } else {
            let done_next = terminated[t + 1] || truncated[t + 1];
            (if done_next { 0.0 } else { 1.0 }, values[t + 1])
        };
        let delta = rewards[t] + gamma * next_value * next_nonterminal - values[t];
        last_gae_lam = delta + gamma * gae_lambda * next_nonterminal * last_gae_lam;
        advantages[t] = last_gae_lam;
        returns[t] = advantages[t] + values[t];
    }
    (advantages, returns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gae_matches_reference_trajectory_no_termination() {
        // 5-step rollout, no mid-rollout episode boundary.
        let rewards = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let term = vec![false; 5];
        let trunc = vec![false; 5];
        let last_value = 0.5_f32;
        let last_done = false;
        let gamma = 0.99_f32;
        let lam = 0.95_f32;
        let (advs, rets) = compute_gae(
            &rewards, &values, &term, &trunc, last_value, last_done, gamma, lam,
        );

        // Hand-compute: delta = r + γ · V' · nonterm − V = 1 + 0.99·0.5·1 − 0.5 = 0.995
        // A[4] = delta[4] = 0.995
        // A[3] = delta[3] + γλ·A[4]·1 = 0.995 + 0.99·0.95·0.995 = 0.995 + 0.9356..
        let delta = 1.0 + gamma * 0.5 * 1.0 - 0.5;
        let mut expected = [0.0_f32; 5];
        let mut last = 0.0_f32;
        for t in (0..5).rev() {
            last = delta + gamma * lam * 1.0 * last;
            expected[t] = last;
        }
        for (i, (a, e)) in advs.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-6, "step {i}: {a} vs {e}");
        }
        for (a, v) in advs.iter().zip(values.iter()) {
            // returns = adv + V
            let r = a + v;
            assert!(rets.contains(&r) || (rets[0] - r).abs() < 1e-5);
        }
    }

    #[test]
    fn gae_handles_terminated_mid_rollout() {
        // 3-step rollout, episode terminates after step 1.
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5];
        let term = vec![false, true, false];
        let trunc = vec![false, false, false];
        let (advs, _) = compute_gae(&rewards, &values, &term, &trunc, 0.5, false, 0.99, 0.95);
        // At t=0: next is t=1 which is NOT marked done-at-next (done_next looks at term[1]||trunc[1] = true).
        // Wait — the convention is: done_next means t+1 is the reset boundary. We look at term[t+1].
        // So at t=0, term[1] = true → next_nonterminal = 0 → δ_0 = 1.0 + 0 − 0.5 = 0.5
        // A[0] = δ_0 + γλ·0·A[1] = 0.5
        // At t=1, not the last step — look at term[2] = false → nonterminal=1 → δ_1 = 1 + 0.99·0.5 − 0.5 = 0.995
        //                    A[1] = δ_1 + γλ·1·A[2]
        // At t=2, last_done=false so next_nonterminal=1, next_value=0.5 → δ_2 = 0.995, A[2]=0.995
        // A[1] = 0.995 + 0.99·0.95·0.995
        // A[0] = 0.5 (no propagation past terminal)
        assert!((advs[0] - 0.5).abs() < 1e-6);
        let a2_expected = 0.995_f32;
        let a1_expected = 0.995_f32 + 0.99 * 0.95 * a2_expected;
        assert!((advs[1] - a1_expected).abs() < 1e-5);
        assert!((advs[2] - a2_expected).abs() < 1e-5);
    }

    #[test]
    fn gae_last_done_zeros_final_bootstrap() {
        let rewards = vec![1.0];
        let values = vec![0.5];
        let (advs, _) = compute_gae(
            &rewards,
            &values,
            &[false],
            &[false],
            10.0, // would-be bootstrap
            true, // but last_done zeros it
            0.99,
            0.95,
        );
        // δ_0 = 1 + γ·10·0 − 0.5 = 0.5
        assert!((advs[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn buffer_push_and_indices() {
        type B = burn::backend::NdArray;
        let mut buf: RolloutBuffer<B, [f32; 2]> = RolloutBuffer::new(4, 1);
        buf.push_step([0.0, 0.0], &[0.0], 0.0, 0.0, 1.0, EpisodeStatus::Running);
        buf.push_step([1.0, 1.0], &[1.0], -0.1, 0.1, 1.0, EpisodeStatus::Running);
        buf.push_step(
            [2.0, 2.0],
            &[2.0],
            -0.2,
            0.2,
            1.0,
            EpisodeStatus::Terminated,
        );
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.action_flat().len(), 3);
        buf.finish(0.0, true, 0.99, 0.95);
        assert_eq!(buf.advantages().len(), 3);
        assert_eq!(buf.returns().len(), 3);
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn gather_action_flat_rows_concatenate() {
        type B = burn::backend::NdArray;
        let mut buf: RolloutBuffer<B, [f32; 1]> = RolloutBuffer::new(4, 2);
        buf.push_step([0.0], &[1.0, 2.0], 0.0, 0.0, 0.0, EpisodeStatus::Running);
        buf.push_step([0.0], &[3.0, 4.0], 0.0, 0.0, 0.0, EpisodeStatus::Running);
        buf.push_step([0.0], &[5.0, 6.0], 0.0, 0.0, 0.0, EpisodeStatus::Running);
        let flat = buf.gather_action_flat(&[0, 2]);
        assert_eq!(flat, vec![1.0, 2.0, 5.0, 6.0]);
    }
}
