use crate::base::{Action, Observation, Reward, TensorConvertible};
use crate::experience::{ExperienceTuple, History};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use rand::prelude::IteratorRandom;
use rand::RngExt;
use std::collections::VecDeque;

// todo! RolloutBuffer for on-policy algorithms)

/// Errors that can occur during memory operations.
#[derive(Debug)]
pub enum ReplayBufferError {
    BatchError(String),
    InsufficientData { requested: usize, available: usize },
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

/// A batch of transitions ready for training. Contains tensors for efficient GPU processing.
/// D: The dimension of the Observation tensor (e.g., 4 for [Batch, Channel, Height, Width])
/// AD: The dimension of the Action tensor (e.g., 1 for [Batch])
pub struct TrainingBatch<const D: usize, const AD: usize, B: Backend> {
    // Observations are generic Rank 'D'
    // For Chess: [Batch, Channels(119), Height(8), Width(8)] -> Rank 4
    pub observations: Tensor<B, D>,

    // Rank 'A' is typically 1 for discrete actions [Batch]
    pub actions: Tensor<B, AD>,

    // Rewards are usually scalar floats per step
    pub rewards: Tensor<B, 1>,

    // Next states same rank as states
    pub next_observations: Tensor<B, D>,

    // Dones are boolean flags (0.0 or 1.0)
    pub dones: Tensor<B, 1>,
}

/// In modern reinforcement learning, the replay buffer is a standard component for almost any **off-policy** algorithm and several **model-based** approaches. It is essential for stabilizing training by:
///  * Storing and reusing past experiences so the agent can revisit and learn from them multiple times.
///  * Providing random, uncorrelated batches for efficient learning.
///  * Managing memory usage to avoid overflow.
///
/// A **Prioritized Experience Replay** is an improved version of a standard experience replay
/// buffer. Instead of sampling past experiences uniformly at random, it assigns a priority to
/// each experience to focus the agents learning on more "informative" or "surprising" transitions.
///
/// **Core Mechanism**
/// - **Priority Metric**: Most implementations use the **Temporal Difference (TD) error** as the
///   measure of priority. A high TD error indicates that the agent's current predictions
///   significantly differ from the actual outcome, suggesting there is more to "learn" from that
///   specific experience.
/// - **Stochastic Sampling**: Experiences are sampled based on their priority, but a "temperature"
///   hyperparameter ($\alpha$) is used to balance between purely greedy prioritization and
///   uniform random sampling.
/// - **Bias Correction**: Because prioritization changes the data distribution (leading to
///   potential overfitting), Importance Sampling (IS) weights are introduced to adjust the
///   gradient updates during training.
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
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    // pub fn clear(&mut self) {
    //     self.buffer.clear();
    // }

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
    pub fn sample_batch<B: Backend>(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<TrainingBatch<D, AD, B>, ReplayBufferError>
    where
        O: TensorConvertible<D, B>,
        A: TensorConvertible<AD, B>,
        R: TensorConvertible<1, B>,
    {
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

        // Stack individual tensors into batch tensors
        // The first dimension becomes the batch dimension
        let observations: Tensor<B, D> = Tensor::stack(observations_tensors, 0);
        let actions: Tensor<B, AD> = Tensor::stack(actions_tensors, 0);
        let rewards: Tensor<B, 1> = Tensor::stack(rewards_tensors, 0);
        let next_observations: Tensor<B, D> = Tensor::stack(next_observations_tensors, 0);
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
