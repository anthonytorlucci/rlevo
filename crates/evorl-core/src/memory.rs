use crate::base::TensorConvertible;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use rand::prelude::IteratorRandom;
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::Ordering;
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

// /// A single transition/experience in the replay memory.
// /// This is fundamentally different from supervised learning, where a batch is just input-label pairs. In RL, you need all five components to compute the Bellman update for Q-learning.
// #[derive(Debug, Clone)]
// pub struct Experience<E, const S: usize, const A: usize>
// where
//     E: Environment<S, A>,
// {
//     /// The environment state before the action
//     pub state: E::StateType,
//     /// The action taken
//     pub action: E::ActionType,
//     /// The immediate reward received
//     pub reward: E::RewardType,
//     /// The environment state after the action
//     pub next_state: E::StateType,
//     /// Whether the episode ended
//     pub done: bool,
// }

// /// A batch of transitions ready for training. Contains tensors for efficient GPU processing.
// /// S: The dimension of the State tensor (e.g., 4 for [Batch, Channel, Height, Width])
// /// A: The dimension of the Action tensor (e.g., 1 for [Batch])
// pub struct TrainingBatch<const S: usize, const A: usize, B: Backend> {
//     // States are now generic Rank 'S'
//     // For Chess: [Batch, Channels(119), Height(8), Width(8)] -> Rank 4
//     pub states: Tensor<B, S>,

//     // Actions are usually indices in Chess, so we use Int kind here.
//     // Rank 'A' is typically 1 for discrete actions [Batch]
//     pub actions: Tensor<B, A>,

//     // Rewards are usually scalar floats per step
//     pub rewards: Tensor<B, 1>,

//     // Next states same rank as states
//     pub next_states: Tensor<B, S>,

//     // Dones are boolean flags (0.0 or 1.0)
//     pub dones: Tensor<B, 1>,
// }

// /// In modern reinforcement learning, the replay buffer is a standard component for almost any **off-policy** algorithm and several **model-based** approaches. It is essential for stabilizing training by:
// ///  * Storing and reusing past experiences so the agent can revisit and learn from them multiple times.
// ///  * Providing random, uncorrelated batches for efficient learning.
// ///  * Managing memory usage to avoid overflow.
// ///
// /// A **Prioritized Experience Replay** is an improved version of a standard experience replay
// /// buffer. Instead of sampling past experiences uniformly at random, it assigns a priority to
// /// each experience to focus the agents learning on more "informative" or "surprising" transitions.
// ///
// /// **Core Mechanism**
// /// - **Priority Metric**: Most implementations use the **Temporal Difference (TD) error** as the
// ///   measure of priority. A high TD error indicates that the agent's current predictions
// ///   significantly differ from the actual outcome, suggesting there is more to "learn" from that
// ///   specific experience.
// /// - **Stochastic Sampling**: Experiences are sampled based on their priority, but a "temperature"
// ///   hyperparameter ($\alpha$) is used to balance between purely greedy prioritization and
// ///   uniform random sampling.
// /// - **Bias Correction**: Because prioritization changes the data distribution (leading to
// ///   potential overfitting), Importance Sampling (IS) weights are introduced to adjust the
// ///   gradient updates during training.
// pub struct PrioritizedExperienceReplay<E, const S: usize, const A: usize>
// where
//     E: Environment<S, A>,
// {
//     buffer: VecDeque<Experience<E, S, A>>,
//     priorities: VecDeque<f32>, // Sampling weights
//     capacity: usize,
//     alpha: f32, // Priority exponent
// }

// impl<E, const S: usize, const A: usize> PrioritizedExperienceReplay<E, S, A>
// where
//     E: Environment<S, A>,
// {
//     /// Creates a new replay buffer with the specified capacity and priority exponent.
//     pub fn new(capacity: usize, alpha: f32) -> Self {
//         Self {
//             buffer: VecDeque::with_capacity(capacity),
//             priorities: VecDeque::with_capacity(capacity),
//             capacity,
//             alpha,
//         }
//     }

//     /// Adds an experience to the buffer, maintaining circular buffer behavior.
//     pub fn push_pop(&mut self, experience: Experience<E, S, A>) {
//         if self.buffer.len() >= self.capacity {
//             self.buffer.pop_front();
//         }
//         self.buffer.push_back(experience);
//     }

//     /// Samples a batch of experiences based on priorities for training.
//     pub fn sample_batch<B: Backend>(
//         &self,
//         batch_size: usize,
//         device: &B::Device,
//     ) -> Result<TrainingBatch<S, A, B>, ReplayBufferError> {
//         // todo! Use priorites for weighted sampling
//         if batch_size > self.len() {
//             return Err(ReplayBufferError::InsufficientData {
//                 requested: batch_size,
//                 available: self.len(),
//             });
//         }

//         // Sample indices using a random number generator
//         let mut rng = rand::rng();
//         let indices: Vec<usize> = (0..self.len())
//             .choose_multiple(&mut rng, batch_size)
//             .into_iter()
//             .collect();

//         // Pre-allocate vectors with exact capacity - much more efficient
//         let mut states_vec = Vec::with_capacity(batch_size);
//         let mut actions_vec = Vec::with_capacity(batch_size);
//         let mut rewards_vec = Vec::with_capacity(batch_size);
//         let mut next_states_vec = Vec::with_capacity(batch_size);
//         let mut dones_vec = Vec::with_capacity(batch_size);

//         // Collect data with minimal allocations
//         for &idx in &indices {
//             let exp = &self.buffer[idx];
//             states_vec.push(exp.state.clone());
//             actions_vec.push(exp.action.clone());
//             rewards_vec.push(exp.reward.clone());
//             next_states_vec.push(exp.next_state.clone());
//             dones_vec.push(exp.done);
//         }

//         // Convert to tensors (this part depends on your tensor conversion traits)
//         let states_tensor = self.convert_states_to_tensor(states_vec, device)?;
//         let actions_tensor = self.convert_actions_to_tensor(actions_vec, device)?;
//         let rewards_tensor = self.convert_rewards_to_tensor(rewards_vec, device)?;
//         let next_states_tensor = self.convert_states_to_tensor(next_states_vec, device)?;
//         let dones_tensor = self.convert_dones_to_tensor(dones_vec, device)?;

//         Ok(TrainingBatch {
//             states: states_tensor,
//             actions: actions_tensor,
//             rewards: rewards_tensor,
//             next_states: next_states_tensor,
//             dones: dones_tensor,
//         })
//     }

//     pub fn len(&self) -> usize {
//         self.buffer.len()
//     }

//     pub fn is_empty(&self) -> bool {
//         self.buffer.is_empty()
//     }

//     pub fn is_full(&self) -> bool {
//         self.buffer.len() >= self.capacity
//     }

//     pub fn clear(&mut self) {
//         self.buffer.clear();
//     }

//     // Helper methods for tensor conversion (implement these based on your traits)
//     fn convert_states_to_tensor<B: Backend>(
//         &self,
//         states: Vec<E::StateType>,
//         device: &B::Device,
//     ) -> Result<Tensor<B, S>, ReplayBufferError>
//     where
//         E::StateType: StateTensorConvertible<S>,
//     {
//         if states.is_empty() {
//             return Err(ReplayBufferError::TensorConversionError(
//                 "Cannot create a tensor from an empty list of states.".to_string(),
//             ));
//         }

//         // Convert each state to a tensor. Each tensor is expected to have a batch dimension of 1.
//         let tensors: Vec<Tensor<B, S>> = states
//             .into_iter()
//             .map(|state| state.to_tensor(device))
//             .collect();

//         // Concatenate the list of tensors along the batch dimension (dim=0).
//         Ok(Tensor::cat(tensors, 0))
//     }

//     fn convert_actions_to_tensor<B: Backend>(
//         &self,
//         actions: Vec<E::ActionType>,
//         device: &B::Device,
//     ) -> Result<Tensor<B, A>, ReplayBufferError>
//     where
//         E::ActionType: ActionTensorConvertible<A>,
//     {
//         if actions.is_empty() {
//             return Err(ReplayBufferError::TensorConversionError(
//                 "Cannot create a tensor from an empty list of actions.".to_string(),
//             ));
//         }

//         // Convert each action to a tensor. Each tensor is expected to have a batch dimension of 1.
//         let tensors: Vec<Tensor<B, A>> = actions
//             .into_iter()
//             .map(|action| action.to_tensor(device))
//             .collect();

//         // Concatenate the list of tensors along the batch dimension (dim=0).
//         Ok(Tensor::cat(tensors, 0))
//     }

//     fn convert_rewards_to_tensor<B: Backend>(
//         &self,
//         rewards: Vec<E::RewardType>,
//         device: &B::Device,
//     ) -> Result<Tensor<B, 1>, ReplayBufferError>
//     where
//         E::RewardType: Into<f32>,
//     {
//         if rewards.is_empty() {
//             return Err(ReplayBufferError::TensorConversionError(
//                 "Cannot create a tensor from an empty list of rewards.".to_string(),
//             ));
//         }

//         let rewards_len = rewards.len();
//         let rewards_f32: Vec<f32> = rewards.into_iter().map(|r| r.into()).collect();
//         let tensor_data = TensorData::new(rewards_f32, [rewards_len]);

//         Ok(Tensor::<B, 1>::from_data(tensor_data, device))
//     }

//     fn convert_dones_to_tensor<B: Backend>(
//         &self,
//         dones: Vec<bool>,
//         device: &B::Device,
//     ) -> Result<Tensor<B, 1>, ReplayBufferError> {
//         if dones.is_empty() {
//             return Err(ReplayBufferError::TensorConversionError(
//                 "Cannot create a tensor from an empty list of dones.".to_string(),
//             ));
//         }

//         let dones_len = dones.len();
//         let dones_f32: Vec<f32> = dones
//             .into_iter()
//             .map(|done| if done { 1.0 } else { 0.0 })
//             .collect();
//         let tensor_data = TensorData::new(dones_f32, [dones_len]);

//         Ok(Tensor::<B, 1>::from_data(tensor_data, device))
//     }
// }
