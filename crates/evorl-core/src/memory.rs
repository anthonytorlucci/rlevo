use crate::environment::Environment;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use rand::Rng;
use ringbuffer::{ConstGenericRingBuffer, RingBuffer};
// use std::marker::PhantomData;

/// A single transition/experience in the replay memory.
/// This is fundamentally different from supervised learning, where a batch is just input-label pairs. In RL, you need all five components to compute the Bellman update for Q-learning.
pub struct Experience<E, const S: usize, const A: usize>
where
    E: Environment<S, A>,
{
    /// The environment state before the action
    pub state: E::StateType,
    /// The action taken
    pub action: E::ActionType,
    /// The immediate reward received
    pub reward: E::RewardType,
    /// The environment state after the action
    pub next_state: E::StateType,
    /// Whether the episode ended
    pub done: bool,
}

/// A batch of transitions ready for training. Contains tensors for efficient GPU processing.
/// S: The dimension of the State tensor (e.g., 4 for [Batch, Channel, Height, Width])
/// A: The dimension of the Action tensor (e.g., 1 for [Batch])
pub struct TrainingBatch<B: Backend, const S: usize, const A: usize> {
    // States are now generic Rank 'S'
    // For Chess: [Batch, Channels(119), Height(8), Width(8)] -> Rank 4
    pub states: Tensor<B, S>,

    // Actions are usually indices in Chess, so we use Int kind here.
    // Rank 'A' is typically 1 for discrete actions [Batch]
    pub actions: Tensor<B, A>,

    // Rewards are usually scalar floats per step
    pub rewards: Tensor<B, 1>,

    // Next states same rank as states
    pub next_states: Tensor<B, S>,

    // Dones are boolean flags (0.0 or 1.0)
    pub dones: Tensor<B, 1>,
}

// The `ReplayBuffer` struct is essential for stabilizing DQN training by:
//  * Storing and reusing past experiences.
//  * Providing random, uncorrelated batches for efficient learning.
//  * Managing memory usage to avoid overflow.
//
// This implementation efficiently manages experience storage and sampling, making it suitable for reinforcement learning algorithms like Q-learning or Deep Q-Networks (DQNs) that rely on experience replay to stabilize training. It directly addresses one of the major challenges in DQN: reducing temporal correlation in training data to improve convergence.
// CAP is the size or capacity of the buffer and must be known at compile time. For more information, see https://docs.rs/ringbuffer/latest/ringbuffer/struct.ConstGenericRingBuffer.html
pub struct ReplayBuffer<E, const S: usize, const A: usize, const CAP: usize>
where
    E: Environment<S, A>,
{
    states: ConstGenericRingBuffer<E::StateType, CAP>,
    actions: ConstGenericRingBuffer<E::ActionType, CAP>,
    rewards: ConstGenericRingBuffer<E::RewardType, CAP>,
    next_states: ConstGenericRingBuffer<E::StateType, CAP>,
    dones: ConstGenericRingBuffer<bool, CAP>,
}

impl<E, const S: usize, const A: usize, const CAP: usize> ReplayBuffer<E, S, A, CAP>
where
    E: Environment<S, A>,
{
    pub fn new() -> Self {
        Self {
            states: ConstGenericRingBuffer::new(),
            actions: ConstGenericRingBuffer::new(),
            rewards: ConstGenericRingBuffer::new(),
            next_states: ConstGenericRingBuffer::new(),
            dones: ConstGenericRingBuffer::new(),
        }
    }

    pub fn push(&mut self, experience: Experience<E, S, A>) {
        self.states.push(experience.state);
        self.actions.push(experience.action);
        self.rewards.push(experience.reward);
        self.next_states.push(experience.next_state);
        self.dones.push(experience.done);
    }

    pub fn sample_batch<B: Backend>(
        &self,
        batch_size: usize,
        device: &B::Device,
        rng: &mut impl Rng,
    ) -> Result<TrainingBatch<B, S, A>, ReplayBufferError> {
        if batch_size > self.len() {
            return Err(ReplayBufferError::InsufficientData {
                requested: batch_size,
                available: self.len(),
            });
        }

        // Sample random indices without replacement
        let indices: Vec<usize> = (0..self.len())
            .choose_multiple(rng, batch_size)
            .into_iter()
            .collect();

        // Collect data from ring buffers
        let mut states_vec = Vec::with_capacity(batch_size);
        let mut actions_vec = Vec::with_capacity(batch_size);
        let mut rewards_vec = Vec::with_capacity(batch_size);
        let mut next_states_vec = Vec::with_capacity(batch_size);
        let mut dones_vec = Vec::with_capacity(batch_size);

        for &idx in &indices {
            // Convert RewardType to f32
            let reward_f32: f32 = (*self.rewards.get(idx).unwrap()).into();

            states_vec.push(self.states.get(idx).unwrap().clone());
            actions_vec.push(self.actions.get(idx).unwrap().clone());
            rewards_vec.push(reward_f32);
            next_states_vec.push(self.next_states.get(idx).unwrap().clone());
            dones_vec.push(*self.dones.get(idx).unwrap());
        }

        // Convert a Vec<E::StateType> into a batch tensor. Implement this based on your state representation.
        // Note that StateType implements BurnRLState which implements to_tensor(), so concatenate the tensors for the batch.
        let states_tensor = self.states_to_tensor::<E::StateType::R1>(states_vec, device)?;
        let actions_tensor = self.actions_to_tensor::<E::ActionType::R1>(actions_vec, device)?;
        let rewards_tensor =
            Tensor::from_floats(TensorData::new(rewards_vec, [batch_size].into()), device);
        let next_states_tensor =
            self.states_to_tensor::<E::StateType::R1>(next_states_vec, device)?;
        let dones_tensor = Tensor::from_floats(
            TensorData::new(
                dones_vec
                    .iter()
                    .map(|&d| if d { 1.0f32 } else { 0.0f32 })
                    .collect(),
                [batch_size].into(),
            ),
            device,
        );

        Ok(TrainingBatch {
            states: states_tensor,
            actions: actions_tensor,
            rewards: rewards_tensor,
            next_states: next_states_tensor,
            dones: dones_tensor,
        })
    }

    fn states_to_tensor<B: Backend>(
        states: Vec<E::StateType>,
        device: &B::Device,
    ) -> Result<Tensor<B, S>, ReplayBufferError> {
        if states.is_empty() {
            return Err(ReplayBufferError::TensorConversionError(
                "Cannot convert empty state vector to tensor".to_string(),
            ));
        }

        // Convert each state to a tensor of rank STATE_RANK
        let tensors: Vec<Tensor<B, S>> = states
            .into_iter()
            .map(|state| state.to_tensor(device))
            .collect();

        // Validate shape at runtime to ensure all states have identical representations
        let first_shape = tensors[0].shape();
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape() != first_shape {
                return Err(ReplayBufferError::TensorConversionError(format!(
                    "State {} has shape {:?}, expected {:?}",
                    i,
                    tensor.shape(),
                    first_shape
                )));
            }
        }

        // Stack along a new batch dimension (dimension 0) to create rank BATCH_RANK
        Tensor::stack(tensors, 0)
            .map_err(|e| ReplayBufferError::TensorConversionError(e.to_string()))
    }

    fn actions_to_tensor<B: Backend>(
        actions: Vec<E::ActionType>,
        device: &B::Device,
    ) -> Result<Tensor<B, A>, ReplayBufferError> {
        if actions.is_empty() {
            return Err(ReplayBufferError::TensorConversionError(
                "Cannot convert empty action vector to tensor".to_string(),
            ));
        }

        // Convert each state to a tensor of rank ACTION_RANK
        let tensors: Vec<Tensor<B, A>> = actions
            .into_iter()
            .map(|action| action.to_tensor(device))
            .collect();

        // Validate shape at runtime to ensure all states have identical representations
        let first_shape = tensors[0].shape();
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape() != first_shape {
                return Err(ReplayBufferError::TensorConversionError(format!(
                    "State {} has shape {:?}, expected {:?}",
                    i,
                    tensor.shape(),
                    first_shape
                )));
            }
        }

        // Stack along a new batch dimension (dimension 0) to create rank BATCH_RANK
        Tensor::stack(tensors, 0)
            .map_err(|e| ReplayBufferError::TensorConversionError(e.to_string()))
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.states.is_full()
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.next_states.clear();
        self.dones.clear();
    }
}

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
