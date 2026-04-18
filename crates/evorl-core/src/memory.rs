//! Experience replay buffers for off-policy reinforcement learning.
//!
//! This module provides [`PrioritizedExperienceReplay`], an off-policy replay
//! buffer that samples transitions proportional to their temporal-difference
//! error, and [`TrainingBatch`], a GPU-ready tensor bundle consumed by
//! learning algorithms.
//!
//! Use [`PrioritizedExperienceReplayBuilder`] to configure and construct the
//! buffer.

use crate::base::{Action, Observation, Reward, TensorConvertible};
use crate::experience::{ExperienceTuple, History};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use rand::prelude::IteratorRandom;
use rand::RngExt;
use std::collections::VecDeque;

// todo! RolloutBuffer for on-policy algorithms)

/// Errors that can occur during replay buffer operations.
#[derive(Debug)]
pub enum ReplayBufferError {
    /// A general batch-assembly failure.
    BatchError(String),
    /// The buffer holds fewer experiences than the requested batch size.
    InsufficientData { requested: usize, available: usize },
    /// A domain type could not be converted to or from a tensor.
    TensorConversionError(String),
}

impl std::fmt::Display for ReplayBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayBufferError::BatchError(msg) => write!(f, "Batch error: {}", msg),
            ReplayBufferError::InsufficientData {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Insufficient data: requested {}, available {}",
                    requested, available
                )
            }
            ReplayBufferError::TensorConversionError(msg) => {
                write!(f, "Tensor conversion error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ReplayBufferError {}

/// A GPU-ready bundle of tensors sampled from a replay buffer for one training step.
///
/// `BD` and `BAD` are **batched** tensor ranks — they are one greater than
/// the unbatched ranks of the source [`Observation`] and [`Action`]:
/// `BD = Observation::DIM + 1`, `BAD = Action::DIM + 1`. Stacking a batch of
/// rank-`D` tensors produces rank-`D + 1` output, and Rust cannot express
/// `D + 1` in a generic position on stable, so the caller of
/// [`PrioritizedExperienceReplay::sample_batch`] supplies both ranks.
///
/// | Observation       | `O: Observation<D>` | `BD` |
/// |-------------------|---------------------|------|
/// | Scalar (bandit)   | `D = 1`, shape `[1]`| `2`  |
/// | Vector (CartPole) | `D = 1`, shape `[4]`| `2`  |
/// | Image (Atari)     | `D = 3`, shape `[C, H, W]` | `4` |
pub struct TrainingBatch<const BD: usize, const BAD: usize, B: Backend> {
    /// Stacked observations at time *t* with shape `[batch, ...obs_shape]`.
    pub observations: Tensor<B, BD>,
    /// Stacked actions with shape `[batch, ...action_shape]`.
    pub actions: Tensor<B, BAD>,
    /// Per-step scalar rewards with shape `[batch]`.
    pub rewards: Tensor<B, 1>,
    /// Stacked observations at time *t+1* with shape `[batch, ...obs_shape]`.
    pub next_observations: Tensor<B, BD>,
    /// Episode-done flags encoded as `0.0`/`1.0` with shape `[batch]`.
    pub dones: Tensor<B, 1>,
}

/// Off-policy replay buffer with priority-weighted sampling.
///
/// Transitions are sampled proportional to `priority^alpha` rather than
/// uniformly, so the agent replays "surprising" transitions (high TD error)
/// more often than mundane ones.
///
/// # Priority exponent (`alpha`)
///
/// - `alpha = 0.0` — uniform random sampling (no prioritization)
/// - `alpha = 1.0` — fully greedy prioritization (raw priorities)
/// - Typical values: `0.4`–`0.7`
///
/// Construct via [`PrioritizedExperienceReplayBuilder`].
pub struct PrioritizedExperienceReplay<
    const D: usize,
    const AD: usize,
    O: Observation<D>,
    A: Action<AD>,
    R: Reward,
> {
    buffer: History<D, AD, O, A, R>,
    priorities: VecDeque<f32>, // Sampling weights
    capacity: usize,
    alpha: f32, // Priority exponent
}

/// Builder pattern implementation for `PrioritizedExperienceReplay`.
/// Provides a convenient way to construct instances with custom configuration.
///
/// # Example
/// ```ignore
/// let replay = PrioritizedExperienceReplayBuilder::new()
///     .with_capacity(10000)
///     .with_alpha(0.6)
///     .build();
/// ```
pub struct PrioritizedExperienceReplayBuilder<
    const D: usize,
    const AD: usize,
    O: Observation<D>,
    A: Action<AD>,
    R: Reward,
> {
    capacity: usize,
    alpha: f32,
    _phantom: std::marker::PhantomData<(O, A, R)>,
}

impl<const D: usize, const AD: usize, O: Observation<D>, A: Action<AD>, R: Reward> Default
    for PrioritizedExperienceReplayBuilder<D, AD, O, A, R>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, const AD: usize, O: Observation<D>, A: Action<AD>, R: Reward>
    PrioritizedExperienceReplayBuilder<D, AD, O, A, R>
{
    /// Creates a new builder with default configuration.
    ///
    /// **Defaults:**
    /// - `capacity`: 100,000
    /// - `alpha`: 0.6 (balances between prioritization and uniform sampling)
    pub fn new() -> Self {
        Self {
            capacity: 100_000,
            alpha: 0.6,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Sets the maximum capacity of the replay buffer.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of experiences to store
    ///
    /// # Panics
    /// Panics if capacity is 0.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be greater than 0");
        self.capacity = capacity;
        self
    }

    /// Sets the priority exponent (alpha).
    ///
    /// # Arguments
    /// * `alpha` - Priority exponent controlling the degree of prioritization
    ///   - `alpha = 0.0`: Pure uniform sampling (no prioritization)
    ///   - `alpha = 1.0`: Full prioritization based on TD error
    ///   - Typical values: 0.4 - 0.7
    ///
    /// # Panics
    /// Panics if alpha is not in range [0.0, 1.0].
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "Alpha must be in range [0.0, 1.0], got {}",
            alpha
        );
        self.alpha = alpha;
        self
    }

    /// Builds the `PrioritizedExperienceReplay` instance.
    pub fn build(self) -> PrioritizedExperienceReplay<D, AD, O, A, R> {
        PrioritizedExperienceReplay {
            buffer: History::new(self.capacity),
            priorities: VecDeque::with_capacity(self.capacity),
            capacity: self.capacity,
            alpha: self.alpha,
        }
    }
}

impl<const D: usize, const AD: usize, O, A, R> PrioritizedExperienceReplay<D, AD, O, A, R>
where
    O: Observation<D>,
    A: Action<AD>,
    R: Reward,
{
    /// Returns the number of transitions currently stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` when the buffer contains no transitions.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns `true` when `len() >= capacity`.
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Appends a transition to the buffer and records its sampling priority.
    ///
    /// When the buffer is at capacity the oldest transition (and its priority)
    /// is evicted before the new one is inserted. New transitions are usually
    /// given the current maximum priority so they are sampled at least once
    /// before their TD error is known; callers can pass `None` to get that
    /// behaviour, or an explicit value to override it.
    pub fn add(
        &mut self,
        observation: O,
        action: A,
        reward: R,
        next_observation: O,
        is_done: bool,
        priority: Option<f32>,
    ) {
        if self.buffer.len() >= self.capacity {
            self.priorities.pop_front();
        }
        self.buffer.add(observation, action, reward, next_observation, is_done);
        let p = priority.unwrap_or_else(|| {
            self.priorities
                .iter()
                .copied()
                .fold(1.0_f32, f32::max)
        });
        self.priorities.push_back(p);
    }

    /// Samples a batch of experiences based on priorities for training.
    ///
    /// This method implements **Prioritized Experience Replay (PER)**, which samples
    /// experiences based on their priority values rather than uniformly at random.
    ///
    /// # Algorithm
    ///
    /// 1. **Compute Sampling Probabilities**: Each experience's priority is raised to
    ///    the power of `alpha` to control the degree of prioritization:
    ///    - `alpha = 0.0`: Uniform random sampling (no prioritization)
    ///    - `alpha = 1.0`: Full prioritization (sample proportional to raw priorities)
    ///    - Typical: `alpha ∈ [0.4, 0.7]` balances exploration and exploitation
    ///
    /// 2. **Weighted Sampling**: Experiences are sampled using weighted random selection
    ///    based on the computed probabilities.
    ///
    /// 3. **Batch Construction**: Selected experiences are converted to tensors for
    ///    efficient GPU-based training.
    ///
    /// # Parameters
    ///
    /// * `batch_size` - Number of experiences to sample
    /// * `device` - The device (CPU/GPU) where tensors should be allocated
    ///
    /// # Returns
    ///
    /// A `TrainingBatch` containing tensors for observations, actions, rewards,
    /// next observations, and done flags.
    ///
    /// # Errors
    ///
    /// Returns `ReplayBufferError::InsufficientData` if the buffer contains fewer
    /// experiences than requested batch size.
    pub fn sample_batch<const BD: usize, const BAD: usize, B: Backend>(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<TrainingBatch<BD, BAD, B>, ReplayBufferError>
    where
        O: TensorConvertible<D, B>,
        A: TensorConvertible<AD, B>,
        R: TensorConvertible<1, B>,
    {
        assert_eq!(
            BD,
            D + 1,
            "batched observation rank BD must equal Observation::DIM + 1"
        );
        assert_eq!(
            BAD,
            AD + 1,
            "batched action rank BAD must equal Action::DIM + 1"
        );
        if batch_size > self.buffer.len() {
            return Err(ReplayBufferError::InsufficientData {
                requested: batch_size,
                available: self.buffer.len(),
            });
        }

        let mut rng = rand::rng();

        // Compute sampling probabilities based on priorities
        let indices: Vec<usize> = if self.priorities.is_empty() || self.alpha == 0.0 {
            // Fallback to uniform random sampling if no priorities or alpha = 0
            (0..self.buffer.len())
                .sample(&mut rng, batch_size)
                .into_iter()
                .collect()
        } else {
            // Prioritized sampling
            // Step 1: Compute priority^alpha for each experience
            let priorities_alpha: Vec<f32> = self
                .priorities
                .iter()
                .map(|&p| p.powf(self.alpha))
                .collect();

            // Step 2: Compute the sum for normalization
            let sum_priorities: f32 = priorities_alpha.iter().sum();

            // Step 3: Compute sampling probabilities
            let probabilities: Vec<f32> = priorities_alpha
                .iter()
                .map(|&p| p / sum_priorities)
                .collect();

            // Step 4: Perform weighted sampling without replacement
            // We use a weighted reservoir sampling approach
            let mut selected_indices = Vec::with_capacity(batch_size);
            let mut available_indices: Vec<usize> = (0..self.buffer.len()).collect();
            let mut available_probs = probabilities.clone();

            for _ in 0..batch_size {
                if available_indices.is_empty() {
                    break;
                }

                // Normalize available probabilities
                let sum: f32 = available_probs.iter().sum();
                if sum <= 0.0 {
                    // Fallback to uniform if all probabilities are zero
                    let idx = rng.random_range(0..available_indices.len());
                    selected_indices.push(available_indices.swap_remove(idx));
                    available_probs.swap_remove(idx);
                    continue;
                }

                // Sample an index based on weighted probabilities
                let mut cumulative = 0.0;
                let random_val: f32 = rng.random_range(0.0..sum);

                let mut selected_pos = 0;
                for (i, &prob) in available_probs.iter().enumerate() {
                    cumulative += prob;
                    if random_val < cumulative {
                        selected_pos = i;
                        break;
                    }
                }

                // Add selected index and remove from available pool
                selected_indices.push(available_indices.swap_remove(selected_pos));
                available_probs.swap_remove(selected_pos);
            }

            selected_indices
        };

        // Pre-allocate vectors with exact capacity for efficient memory usage
        let mut observations_vec = Vec::with_capacity(batch_size);
        let mut actions_vec = Vec::with_capacity(batch_size);
        let mut rewards_vec = Vec::with_capacity(batch_size);
        let mut next_observations_vec = Vec::with_capacity(batch_size);
        let mut dones_vec = Vec::with_capacity(batch_size);

        // Collect data from sampled experiences
        for &idx in &indices {
            let exp: &ExperienceTuple<D, AD, O, A, R> = &self.buffer[idx];
            observations_vec.push(exp.observation.clone());
            actions_vec.push(exp.action.clone());
            rewards_vec.push(exp.reward.clone());
            next_observations_vec.push(exp.next_observation.clone());
            dones_vec.push(if exp.is_done { 1.0 } else { 0.0 });
        }

        // Convert individual observations to tensors and stack them into a batch
        let observations_tensors: Vec<Tensor<B, D>> = observations_vec
            .iter()
            .map(|obs| obs.to_tensor(device))
            .collect();

        let actions_tensors: Vec<Tensor<B, AD>> = actions_vec
            .iter()
            .map(|action| action.to_tensor(device))
            .collect();

        let rewards_tensors: Vec<Tensor<B, 1>> = rewards_vec
            .iter()
            .map(|reward| reward.to_tensor(device))
            .collect();

        let next_observations_tensors: Vec<Tensor<B, D>> = next_observations_vec
            .iter()
            .map(|obs| obs.to_tensor(device))
            .collect();

        // Stack individual tensors into batch tensors. Stacking bumps the
        // rank by 1 — `BD = D + 1`, `BAD = AD + 1` — enforced by the
        // `assert_eq!`s at the top of the function. Rewards come through as
        // rank-1 `[1]` tensors from `TensorConvertible<1, B>`, so we use
        // `cat` (rank-preserving) instead of `stack` to produce shape
        // `[batch]`.
        let observations: Tensor<B, BD> = Tensor::stack(observations_tensors, 0);
        let actions: Tensor<B, BAD> = Tensor::stack(actions_tensors, 0);
        let rewards: Tensor<B, 1> = Tensor::cat(rewards_tensors, 0);
        let next_observations: Tensor<B, BD> = Tensor::stack(next_observations_tensors, 0);
        let dones: Tensor<B, 1> = Tensor::from_floats(dones_vec.as_slice(), device);

        Ok(TrainingBatch {
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        })
    }
}

#[cfg(test)]
mod prioritized_experience_replay_builder_tests {
    use super::*;

    // Mock implementation for testing
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    struct TestObservation;

    impl Observation<2> for TestObservation {
        fn shape() -> [usize; 2] {
            [4, 4]
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    struct TestAction;

    impl Action<1> for TestAction {
        fn shape() -> [usize; 1] {
            [1]
        }

        fn is_valid(&self) -> bool {
            true
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    struct TestReward(f32);

    impl From<f32> for TestReward {
        fn from(value: f32) -> Self {
            TestReward(value)
        }
    }

    impl From<TestReward> for f32 {
        fn from(value: TestReward) -> Self {
            value.0
        }
    }

    impl std::ops::Add for TestReward {
        type Output = Self;

        fn add(self, other: Self) -> Self {
            TestReward(self.0 + other.0)
        }
    }

    impl Reward for TestReward {
        fn zero() -> Self {
            TestReward(0.0)
        }
    }

    #[test]
    fn test_builder_creates_with_defaults() {
        let builder = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new();
        assert_eq!(builder.capacity, 100_000);
        assert!((builder.alpha - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_builder_default_trait_impl() {
        let builder = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::default();
        assert_eq!(builder.capacity, 100_000);
        assert!((builder.alpha - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_builder_set_capacity() {
        let builder = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_capacity(5000);
        assert_eq!(builder.capacity, 5000);
    }

    #[test]
    fn test_builder_set_multiple_capacities() {
        let builder = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_capacity(2048)
        .with_capacity(8192);
        assert_eq!(builder.capacity, 8192); // Last set value wins
    }

    #[test]
    fn test_builder_set_alpha() {
        let builder = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_alpha(0.7);
        assert!((builder.alpha - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_builder_fluent_api() {
        let builder = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_capacity(4096)
        .with_alpha(0.4);
        assert_eq!(builder.capacity, 4096);
        assert!((builder.alpha - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_builder_build_returns_correct_values() {
        let replay = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_capacity(2048)
        .with_alpha(0.8)
        .build();
        assert_eq!(replay.capacity, 2048);
        assert!((replay.alpha - 0.8).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Capacity must be greater than 0")]
    fn test_builder_zero_capacity_panics() {
        let _ = PrioritizedExperienceReplayBuilder::<2, 1, TestObservation, TestAction, TestReward>::new()
            .with_capacity(0);
    }

    #[test]
    #[should_panic(expected = "Alpha must be in range")]
    fn test_builder_negative_alpha_panics() {
        let _ = PrioritizedExperienceReplayBuilder::<2, 1, TestObservation, TestAction, TestReward>::new()
            .with_alpha(-0.1);
    }

    #[test]
    #[should_panic(expected = "Alpha must be in range")]
    fn test_builder_alpha_above_one_panics() {
        let _ = PrioritizedExperienceReplayBuilder::<2, 1, TestObservation, TestAction, TestReward>::new()
            .with_alpha(1.5);
    }

    #[test]
    fn test_builder_alpha_boundary_zero() {
        let replay = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_alpha(0.0)
        .build();
        assert_eq!(replay.alpha, 0.0);
    }

    #[test]
    fn test_builder_alpha_boundary_one() {
        let replay = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_alpha(1.0)
        .build();
        assert_eq!(replay.alpha, 1.0);
    }

    #[test]
    fn test_builder_realistic_dqn_config() {
        let replay = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_capacity(1_000_000)
        .with_alpha(0.6)
        .build();
        assert_eq!(replay.capacity, 1_000_000);
        assert!((replay.alpha - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_builder_no_prioritization_config() {
        let replay = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_alpha(0.0)
        .build();
        assert_eq!(replay.alpha, 0.0); // Uniform sampling
    }

    #[test]
    fn test_builder_full_prioritization_config() {
        let replay = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_alpha(1.0)
        .build();
        assert_eq!(replay.alpha, 1.0); // Full prioritization
    }

    #[test]
    fn test_builder_large_capacity() {
        let replay = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_capacity(10_000_000)
        .build();
        assert_eq!(replay.capacity, 10_000_000);
    }

    #[test]
    fn test_builder_small_capacity() {
        let replay = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_capacity(1)
        .build();
        assert_eq!(replay.capacity, 1);
    }

    #[test]
    fn test_builder_chain_operations_order_doesnt_matter() {
        let replay1 = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_capacity(4096)
        .with_alpha(0.5)
        .build();

        let replay2 = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_alpha(0.5)
        .with_capacity(4096)
        .build();

        assert_eq!(replay1.capacity, replay2.capacity);
        assert!((replay1.alpha - replay2.alpha).abs() < 1e-6);
    }

    #[test]
    fn test_add_stores_transition_and_priority() {
        let mut per = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_capacity(3)
        .build();

        per.add(
            TestObservation,
            TestAction,
            TestReward(1.0),
            TestObservation,
            false,
            Some(0.5),
        );
        per.add(
            TestObservation,
            TestAction,
            TestReward(2.0),
            TestObservation,
            true,
            None, // should pick max existing priority (0.5 vs default 1.0 floor)
        );

        assert_eq!(per.len(), 2);
        assert_eq!(per.priorities.len(), 2);
        assert!((per.priorities[0] - 0.5).abs() < 1e-6);
        // `None` default picks max of (existing priorities ∪ {1.0}) so the
        // new item gets a priority of 1.0 here.
        assert!((per.priorities[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_evicts_oldest_when_full() {
        let mut per = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_capacity(2)
        .build();

        per.add(TestObservation, TestAction, TestReward(1.0), TestObservation, false, Some(0.1));
        per.add(TestObservation, TestAction, TestReward(2.0), TestObservation, false, Some(0.2));
        per.add(TestObservation, TestAction, TestReward(3.0), TestObservation, false, Some(0.3));

        // The oldest transition (priority 0.1) should have been evicted and
        // the buffer/priority queue must stay length-aligned at capacity.
        assert!(per.len() <= 2);
        assert_eq!(per.priorities.len(), per.len());
        assert!((per.priorities[0] - 0.2).abs() < 1e-6);
        assert!((per.priorities[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_builder_different_observation_dimensions() {
        // Test with different observation space dimensions
        // All still use TestAction which implements Action<1>
        let _replay_2d = PrioritizedExperienceReplayBuilder::<
            2,
            1,
            TestObservation,
            TestAction,
            TestReward,
        >::new()
        .with_capacity(512)
        .build();

        // Note: Can't test different AD dimensions without implementing
        // additional Action<N> for TestAction, which is not the focus of builder tests
    }
}

#[cfg(test)]
mod sample_batch_tests {
    //! End-to-end regression tests for [`PrioritizedExperienceReplay::sample_batch`].
    //!
    //! These tests actually call `sample_batch` against a concrete backend
    //! (Burn's ndarray) and verify the produced tensors have the batched
    //! shapes `[batch, ...]`. Before the BD/BAD generics landed, the
    //! function panicked at runtime because it tried to stack rank-`D`
    //! tensors into a rank-`D` output, so this suite guards against that
    //! regression.

    use super::*;
    use crate::base::{Action, Observation, Reward, TensorConvertible, TensorConversionError};
    use burn::backend::NdArray;
    use burn::tensor::Tensor;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    struct Obs(f32, f32, f32);

    impl Observation<1> for Obs {
        fn shape() -> [usize; 1] {
            [3]
        }
    }

    impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for Obs {
        fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
            Tensor::from_floats([self.0, self.1, self.2], device)
        }
        fn from_tensor(_t: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
            unimplemented!("not exercised by this test")
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    struct Act(u32); // one-hot index; shape [2]

    impl Action<1> for Act {
        fn shape() -> [usize; 1] {
            [2]
        }
        fn is_valid(&self) -> bool {
            self.0 < 2
        }
    }

    impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for Act {
        fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
            let mut one_hot = [0.0_f32; 2];
            one_hot[self.0 as usize] = 1.0;
            Tensor::from_floats(one_hot, device)
        }
        fn from_tensor(_t: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
            unimplemented!("not exercised by this test")
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    struct Rew(f32);

    impl Reward for Rew {
        fn zero() -> Self {
            Rew(0.0)
        }
    }

    impl std::ops::Add for Rew {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Rew(self.0 + rhs.0)
        }
    }

    impl From<Rew> for f32 {
        fn from(r: Rew) -> f32 {
            r.0
        }
    }

    impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for Rew {
        fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
            Tensor::from_floats([self.0], device)
        }
        fn from_tensor(_t: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
            unimplemented!("not exercised by this test")
        }
    }

    type Be = NdArray;

    fn populated(n: usize) -> PrioritizedExperienceReplay<1, 1, Obs, Act, Rew> {
        let mut per = PrioritizedExperienceReplayBuilder::<1, 1, Obs, Act, Rew>::new()
            .with_capacity(n.max(4))
            .with_alpha(0.0) // uniform sampling — deterministic path
            .build();
        for i in 0..n {
            let f = i as f32;
            per.add(
                Obs(f, f + 0.1, f + 0.2),
                Act((i % 2) as u32),
                Rew(f * 0.5),
                Obs(f + 1.0, f + 1.1, f + 1.2),
                i == n - 1,
                Some(1.0),
            );
        }
        per
    }

    #[test]
    fn sample_batch_returns_correctly_shaped_tensors() {
        let per = populated(16);
        let device = Default::default();
        let batch = per
            .sample_batch::<2, 2, Be>(8, &device)
            .expect("sample_batch");
        assert_eq!(batch.observations.shape().dims, [8, 3]);
        assert_eq!(batch.next_observations.shape().dims, [8, 3]);
        assert_eq!(batch.actions.shape().dims, [8, 2]);
        assert_eq!(batch.rewards.shape().dims, [8]);
        assert_eq!(batch.dones.shape().dims, [8]);
    }

    #[test]
    fn sample_batch_rejects_requests_larger_than_buffer() {
        let per = populated(4);
        let device = Default::default();
        let err = match per.sample_batch::<2, 2, Be>(8, &device) {
            Ok(_) => panic!("batch_size > len should fail"),
            Err(e) => e,
        };
        match err {
            ReplayBufferError::InsufficientData { requested, available } => {
                assert_eq!(requested, 8);
                assert_eq!(available, 4);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "batched observation rank BD")]
    fn sample_batch_panics_on_wrong_bd() {
        let per = populated(4);
        let device = Default::default();
        // BD = 1 is wrong (should be D + 1 = 2). Runtime assertion fires.
        let _ = per.sample_batch::<1, 2, Be>(2, &device);
    }

    #[test]
    #[should_panic(expected = "batched action rank BAD")]
    fn sample_batch_panics_on_wrong_bad() {
        let per = populated(4);
        let device = Default::default();
        // BAD = 1 is wrong (should be AD + 1 = 2).
        let _ = per.sample_batch::<2, 1, Be>(2, &device);
    }
}
