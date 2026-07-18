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
//! [`compute_gae`] follows Schulman et al. 2016 with **partial-episode
//! bootstrapping** (Pardo et al. 2018, Eq. 6) at truncations, per ADR 0048.
//! A truncated step bootstraps its delta from `V(s_continuation)` while its
//! λ-recursion is cut; a terminated step does neither. This deliberately
//! diverges from CleanRL's default PPO, which ORs the two flags.
//!
//! # Indexing convention
//!
//! [`RolloutBuffer::push_step`] stores `obs[t]` together with the status of the
//! transition *out of* `obs[t]`. So `terminated[t]` means "transition `t` ended
//! the episode", which is exactly what decides whether the bootstrap after step
//! `t` is valid. Every done-ness read in [`compute_gae`] is therefore at index
//! `[t]` — never `[t + 1]`.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use std::marker::PhantomData;

/// How a recorded transition left its episode, carrying whatever datum GAE
/// needs to treat it correctly.
///
/// This is the buffer's replacement for a bare
/// [`EpisodeStatus`](rlevo_core::environment::EpisodeStatus) at push time. A
/// truncated step *requires* a bootstrap value (ADR 0048) and there is no way
/// to record one without it — the flag and its value cannot skew apart, and
/// there is no "unset" float that silently reads as a legitimate zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepEnd {
    /// The episode continues past this transition.
    Running,
    /// The environment reported an intrinsic terminal state. The MDP genuinely
    /// ends here, so there is no future to bootstrap: the target is just `r`.
    Terminated,
    /// An extrinsic cutoff (typically a `TimeLimit` wrapper) ended the episode
    /// while the MDP itself continued.
    ///
    /// Carries `V(s_continuation)` — the value network's estimate on the
    /// observation the environment produced *before* it was reset. Per Pardo
    /// et al. Eq. 6 the target is `r + γ·V(s_continuation)`, not `r`.
    Truncated {
        /// `V(s_continuation)`, the bootstrap for this step's delta.
        bootstrap_value: f32,
    },
}

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
    /// `Some(V(s_continuation))` at a truncated step, `None` otherwise — so
    /// "was truncated" is `truncation_value[t].is_some()` by construction.
    truncation_value: Vec<Option<f32>>,
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
            truncation_value: Vec::with_capacity(capacity),
            advantages: Vec::new(),
            returns: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Appends one step. `action_row.len()` must equal `action_dim`.
    ///
    /// `end` describes how the transition *out of* `obs` left the episode; see
    /// the module's indexing note. A [`StepEnd::Truncated`] must carry its
    /// `V(s_continuation)` bootstrap, which is why this takes a [`StepEnd`]
    /// rather than a bare `EpisodeStatus`.
    ///
    /// # Panics
    ///
    /// If `action_row.len() != action_dim`.
    pub fn push_step(
        &mut self,
        obs: O,
        action_row: &[f32],
        log_prob: f32,
        value: f32,
        reward: f32,
        end: StepEnd,
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
        self.terminated.push(end == StepEnd::Terminated);
        self.truncation_value.push(match end {
            StepEnd::Truncated { bootstrap_value } => Some(bootstrap_value),
            StepEnd::Running | StepEnd::Terminated => None,
        });
    }

    /// Finalises the rollout: computes GAE advantages and returns.
    ///
    /// `last_value` is the `V(s_final)` bootstrap (typically the value
    /// network's prediction on the observation the rollout stopped at). It is
    /// consulted **only** when the final step left the episode `Running` — see
    /// [`Self::last_step_ended`]; on any other path the final step's own stored
    /// status supplies the bootstrap, so `last_value` is ignored and callers
    /// may pass any value.
    pub fn finish(&mut self, last_value: f32, gamma: f32, gae_lambda: f32) {
        let (advs, rets) = compute_gae(
            &self.rewards,
            &self.values,
            &self.terminated,
            &self.truncation_value,
            last_value,
            gamma,
            gae_lambda,
        );
        self.advantages = advs;
        self.returns = rets;
    }

    /// Whether the most recently pushed step ended its episode.
    ///
    /// `true` when the final stored transition was terminated or truncated,
    /// i.e. exactly when [`Self::finish`]'s `last_value` argument is unused.
    /// An empty buffer reports `false`.
    pub fn last_step_ended(&self) -> bool {
        let n = self.len();
        n > 0 && (self.terminated[n - 1] || self.truncation_value[n - 1].is_some())
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
    pub fn gather_f32(
        &self,
        data: &[f32],
        indices: &[usize],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
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
        self.truncation_value.clear();
        self.advantages.clear();
        self.returns.clear();
    }
}

/// Generalized Advantage Estimation.
///
/// Returns `(advantages, returns)` where `returns[t] = advantages[t] + values[t]`.
///
/// # Indexing
///
/// `terminated[t]` / `truncated[t]` describe the transition *out of* step `t`
/// (see the module docs), so both the bootstrap mask and the λ-recursion mask
/// read index `[t]`. `last_value` supplies the bootstrap for the final step
/// only when that step left the episode `Running`; when it ended, its own flag
/// zeroes the bootstrap and `last_value` is unused.
///
/// # Truncation: two masks, not one
///
/// A truncated step is **not** a terminated one (Pardo et al. 2018, Eq. 6;
/// Towers et al. 2025 — the bootstrap mask is `¬terminated`, never `¬done`).
/// `rlevo`'s `TimeLimit` is an opt-in wrapper, never intrinsic to an
/// environment's MDP, so every environment here is Pardo's *time-unlimited*
/// case, where a timeout must bootstrap rather than terminate. The two effects
/// are distinct and a single `next_nonterminal` factor cannot express both:
///
/// - the **delta bootstraps** from `V(s_continuation)`, carried in
///   `truncation_value[t]` — the agent's future is real, only the clock
///   stopped;
/// - the **λ-recursion is cut** — the trajectory genuinely ended, so advantage
///   must not propagate across the boundary.
///
/// This deliberately diverges from CleanRL's default PPO, which ORs the flags.
/// See ADR 0048.
///
/// # Panics
///
/// If `values`, `terminated`, or `truncation_value` differ in length from
/// `rewards`.
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    terminated: &[bool],
    truncation_value: &[Option<f32>],
    last_value: f32,
    gamma: f32,
    gae_lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    assert_eq!(values.len(), n);
    assert_eq!(terminated.len(), n);
    assert_eq!(truncation_value.len(), n);
    let mut advantages = vec![0.0_f32; n];
    let mut returns = vec![0.0_f32; n];

    let mut last_gae_lam = 0.0_f32;
    for t in (0..n).rev() {
        let next_value = if t == n - 1 {
            last_value
        } else {
            values[t + 1]
        };
        // Mask 1: the episode boundary, which cuts the λ-recursion.
        let ended = terminated[t] || truncation_value[t].is_some();
        // Mask 2: the delta's bootstrap. Zero only at a true termination —
        // a truncation bootstraps V(s_continuation) instead (PEB).
        let boot = if terminated[t] {
            0.0
        } else {
            truncation_value[t].unwrap_or(next_value)
        };
        let delta = rewards[t] + gamma * boot - values[t];
        last_gae_lam = delta + gamma * gae_lambda * f32::from(!ended) * last_gae_lam;
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
        //
        // `values` is deliberately non-constant: a flat critic makes both the
        // advantage check and the `returns = A + V` check permutation-blind,
        // since every index carries the same V and any misalignment still
        // matches. Distinct per-step values let the positional assertions
        // below actually pin the ordering.
        let rewards = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let values = vec![0.1, 0.4, 0.2, 0.7, 0.5];
        let term = vec![false; 5];
        let trunc = vec![None; 5];
        let last_value = 0.5_f32;
        let gamma = 0.99_f32;
        let lam = 0.95_f32;
        let (advs, rets) = compute_gae(&rewards, &values, &term, &trunc, last_value, gamma, lam);

        // Independent reference recursion, general per-step form. With no
        // termination and no truncation every mask is 1, so
        //   δ[t] = r[t] + γ·V'[t] − V[t],  V'[t] = V[t+1] (V' = last_value at t = 4)
        //   A[t] = δ[t] + γλ·A[t+1],       A[5] ≡ 0,  γλ = 0.9405
        // Spot values for the tail:
        //   t=4: δ = 1 + 0.99·0.5 − 0.5 = 0.995        → A[4] = 0.995
        //   t=3: δ = 1 + 0.99·0.5 − 0.7 = 0.795        → A[3] = 0.795 + 0.9405·0.995
        //                                                     = 1.730_797_5
        let n = rewards.len();
        let mut expected = [0.0_f32; 5];
        let mut last = 0.0_f32;
        for t in (0..n).rev() {
            let next_value = if t == n - 1 {
                last_value
            } else {
                values[t + 1]
            };
            let delta = rewards[t] + gamma * next_value - values[t];
            last = delta + gamma * lam * last;
            expected[t] = last;
        }
        for (i, (a, e)) in advs.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-6, "step {i}: {a} vs {e}");
        }
        // Positional — not set-membership — check of the returns identity.
        // `rets[t]` must equal `advs[t] + values[t]` at the *same* index t, so
        // a reversed or otherwise misaligned `returns` vector fails here.
        for t in 0..n {
            let want = advs[t] + values[t];
            assert!(
                (rets[t] - want).abs() < 1e-6,
                "step {t}: returns {} != advantage {} + value {} = {want}",
                rets[t],
                advs[t],
                values[t]
            );
        }
    }

    #[test]
    fn gae_handles_terminated_mid_rollout() {
        // 3-step rollout. `term[1] == true` means the transition *out of* step 1
        // terminated the episode, so the bootstrap is cut after step 1 — not
        // after step 0.
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5];
        let term = vec![false, true, false];
        let trunc = vec![None, None, None];
        let (advs, rets) = compute_gae(&rewards, &values, &term, &trunc, 0.5, 0.99, 0.95);

        // Hand-computed, γ = 0.99, λ = 0.95, γλ = 0.9405:
        //   t=2: ended = term[2] = false → V' = last_value = 0.5
        //        δ₂ = 1 + 0.99·0.5 − 0.5 = 0.995
        //        A₂ = 0.995
        //   t=1: ended = term[1] = true → bootstrap and recursion both cut
        //        δ₁ = 1 + 0 − 0.5 = 0.5
        //        A₁ = 0.5 + 0.9405·0·A₂ = 0.5
        //   t=0: ended = term[0] = false → V' = values[1] = 0.5
        //        δ₀ = 1 + 0.99·0.5 − 0.5 = 0.995
        //        A₀ = 0.995 + 0.9405·0.5 = 0.995 + 0.47025 = 1.46525
        let a2 = 0.995_f32;
        let a1 = 0.5_f32;
        let a0 = 1.465_25_f32;
        assert!(
            (advs[2] - a2).abs() < 1e-5,
            "A₂ bootstraps last_value: {} vs {a2}",
            advs[2]
        );
        assert!(
            (advs[1] - a1).abs() < 1e-5,
            "A₁ is the terminal step: bootstrap zeroed, no propagation in: {} vs {a1}",
            advs[1]
        );
        assert!(
            (advs[0] - a0).abs() < 1e-5,
            "A₀ precedes the terminal step, so it bootstraps values[1] and \
             receives no λ-propagation from across the boundary: {} vs {a0}",
            advs[0]
        );
        for (t, (a, v)) in advs.iter().zip(values.iter()).enumerate() {
            assert!(
                (rets[t] - (a + v)).abs() < 1e-6,
                "returns[{t}] must equal advantage + value"
            );
        }
    }

    #[test]
    fn gae_bootstraps_truncation_value_mid_rollout() {
        // Same 3-step rollout as `gae_handles_terminated_mid_rollout`, but the
        // transition out of step 1 was *truncated* (a time limit fired) with
        // V(s_continuation) = 2.0. Partial-episode bootstrapping splits the
        // two masks: step 1's delta bootstraps 2.0, while its λ-recursion is
        // still cut. Contrast with the terminated case, where the delta gets
        // nothing — that difference is the whole point of ADR 0048.
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5];
        let term = vec![false, false, false];
        let trunc = vec![None, Some(2.0), None];
        let (advs, rets) = compute_gae(&rewards, &values, &term, &trunc, 0.5, 0.99, 0.95);

        // Hand-computed, γ = 0.99, λ = 0.95, γλ = 0.9405:
        //   t=2: not ended → V' = last_value = 0.5
        //        δ₂ = 1 + 0.99·0.5 − 0.5 = 0.995
        //        A₂ = 0.995
        //   t=1: truncated → boot = 2.0 (NOT 0.0), but ended = true
        //        δ₁ = 1 + 0.99·2.0 − 0.5 = 1 + 1.98 − 0.5 = 2.48
        //        A₁ = 2.48 + 0.9405·0·A₂ = 2.48
        //   t=0: not ended → V' = values[1] = 0.5
        //        δ₀ = 1 + 0.99·0.5 − 0.5 = 0.995
        //        A₀ = 0.995 + 0.9405·2.48 = 0.995 + 2.33244 = 3.32744
        let a2 = 0.995_f32;
        let a1 = 2.48_f32;
        let a0 = 3.327_44_f32;
        assert!(
            (advs[2] - a2).abs() < 1e-5,
            "A₂ is unaffected by the earlier truncation: {} vs {a2}",
            advs[2]
        );
        assert!(
            (advs[1] - a1).abs() < 1e-5,
            "A₁'s delta must bootstrap V(s_continuation) = 2.0, not 0.0: {} vs {a1}",
            advs[1]
        );
        assert!(
            (advs[0] - a0).abs() < 1e-4,
            "A₀ receives no λ-propagation across the truncation boundary, but \
             A₁ itself is larger because its delta bootstrapped: {} vs {a0}",
            advs[0]
        );
        for (t, (a, v)) in advs.iter().zip(values.iter()).enumerate() {
            assert!(
                (rets[t] - (a + v)).abs() < 1e-6,
                "returns[{t}] must equal advantage + value"
            );
        }
    }

    #[test]
    fn gae_truncation_differs_from_termination() {
        // The same boundary at the same index, differing only in *why* the
        // episode ended, must produce different advantages. Under the old
        // `terminated || truncated` collapse these were identical.
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5];
        let (term_advs, _) = compute_gae(
            &rewards,
            &values,
            &[false, true, false],
            &[None, None, None],
            0.5,
            0.99,
            0.95,
        );
        let (trunc_advs, _) = compute_gae(
            &rewards,
            &values,
            &[false, false, false],
            &[None, Some(2.0), None],
            0.5,
            0.99,
            0.95,
        );
        assert!(
            (trunc_advs[1] - term_advs[1]).abs() > 1.0,
            "truncation must not be treated as termination: {} vs {}",
            trunc_advs[1],
            term_advs[1]
        );
        // The λ-recursion is cut identically in both cases, so the *only*
        // difference at t=0 is the one that propagated in via A₁.
        assert!(
            trunc_advs[0] > term_advs[0],
            "the truncation bootstrap must raise the preceding advantage: {} vs {}",
            trunc_advs[0],
            term_advs[0]
        );
    }

    #[test]
    fn gae_terminal_final_step_zeros_bootstrap() {
        // Single-step rollout whose one transition terminated. The final step's
        // own stored status — not a separate `last_done` argument — is what
        // zeroes the bootstrap, so the supplied `last_value` is ignored.
        let (advs, _) = compute_gae(
            &[1.0],
            &[0.5],
            &[true],
            &[None],
            10.0, // would-be bootstrap; must not be used
            0.99,
            0.95,
        );
        // δ₀ = 1 + 0.99·10·0 − 0.5 = 0.5
        assert!(
            (advs[0] - 0.5).abs() < 1e-6,
            "terminated final step must ignore last_value: {} vs 0.5",
            advs[0]
        );
    }

    #[test]
    fn gae_running_final_step_uses_last_value() {
        // Same rollout, but the episode is still `Running`, so `last_value`
        // *is* the bootstrap: δ₀ = 1 + 0.99·10 − 0.5 = 10.4
        let (advs, _) = compute_gae(&[1.0], &[0.5], &[false], &[None], 10.0, 0.99, 0.95);
        assert!(
            (advs[0] - 10.4).abs() < 1e-4,
            "running final step must bootstrap last_value: {} vs 10.4",
            advs[0]
        );
    }

    #[test]
    fn last_step_ended_tracks_final_status() {
        type B = burn::backend::Flex;
        let mut buf: RolloutBuffer<B, [f32; 1]> = RolloutBuffer::new(4, 1);
        assert!(!buf.last_step_ended(), "empty buffer has no final step");
        buf.push_step([0.0], &[0.0], 0.0, 0.0, 0.0, StepEnd::Running);
        assert!(!buf.last_step_ended(), "running final step has not ended");
        buf.push_step(
            [0.0],
            &[0.0],
            0.0,
            0.0,
            0.0,
            StepEnd::Truncated {
                bootstrap_value: 1.0,
            },
        );
        assert!(buf.last_step_ended(), "truncated final step has ended");
        buf.push_step([0.0], &[0.0], 0.0, 0.0, 0.0, StepEnd::Terminated);
        assert!(buf.last_step_ended(), "terminated final step has ended");
    }

    #[test]
    fn buffer_push_and_indices() {
        type B = burn::backend::Flex;
        let mut buf: RolloutBuffer<B, [f32; 2]> = RolloutBuffer::new(4, 1);
        buf.push_step([0.0, 0.0], &[0.0], 0.0, 0.0, 1.0, StepEnd::Running);
        buf.push_step([1.0, 1.0], &[1.0], -0.1, 0.1, 1.0, StepEnd::Running);
        buf.push_step([2.0, 2.0], &[2.0], -0.2, 0.2, 1.0, StepEnd::Terminated);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.action_flat().len(), 3);
        buf.finish(0.0, 0.99, 0.95);
        assert_eq!(buf.advantages().len(), 3);
        assert_eq!(buf.returns().len(), 3);
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn gather_action_flat_rows_concatenate() {
        type B = burn::backend::Flex;
        let mut buf: RolloutBuffer<B, [f32; 1]> = RolloutBuffer::new(4, 2);
        buf.push_step([0.0], &[1.0, 2.0], 0.0, 0.0, 0.0, StepEnd::Running);
        buf.push_step([0.0], &[3.0, 4.0], 0.0, 0.0, 0.0, StepEnd::Running);
        buf.push_step([0.0], &[5.0, 6.0], 0.0, 0.0, 0.0, StepEnd::Running);
        let flat = buf.gather_action_flat(&[0, 2]);
        assert_eq!(flat, vec![1.0, 2.0, 5.0, 6.0]);
    }
}
