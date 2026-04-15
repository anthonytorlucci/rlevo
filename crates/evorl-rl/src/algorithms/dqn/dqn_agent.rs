use crate::algorithms::dqn::dqn_config::DqnTrainingConfig;
use crate::algorithms::dqn::dqn_model::DqnModel;
use burn::optim::Optimizer;
use burn::prelude::ToElement;
use burn::tensor::backend::AutodiffBackend;
use burn::Tensor;
use evorl_core::action::DiscreteAction;
// use evorl_core::agent::{Agent, NeuralAgent};
// use evorl_core::environment::Environment;
// use evorl_core::memory::{PrioritizedExperienceReplay, TrainingBatch};
use evorl_core::base::{Action, State, TensorConvertible};
use evorl_core::metrics::{AgentStats, PerformanceRecord};
use rand::RngExt;
use std::marker::PhantomData;

// Define error type for DQNAgent
// todo! implement From
#[derive(Debug)]
pub enum DqnAgentError {
    ModelNotInitialized,
    TensorConversionFailed(String),
    InvalidAction(String),
    IoError(std::io::Error),
}

impl std::fmt::Display for DqnAgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DqnAgentError::ModelNotInitialized => write!(f, "DQN model not initialized"),
            DqnAgentError::TensorConversionFailed(msg) => {
                write!(f, "Tensor conversion failed: {}", msg)
            }
            DqnAgentError::InvalidAction(msg) => {
                write!(f, "Invalid action: {}", msg)
            }
            DqnAgentError::IoError(err) => {
                write!(f, "IO error: {}", err)
            }
        }
    }
}

impl std::error::Error for DqnAgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DqnAgentError::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DqnAgentError {
    fn from(error: std::io::Error) -> Self {
        DqnAgentError::IoError(error)
    }
}

/// Deep Q-Network (DQN) agent training statistics
#[derive(Debug, Clone)]
struct DqnMetrics {
    reward: f32,
    steps: usize,
    policy_loss: f32,
    value_loss: f32,
}

impl PerformanceRecord for DqnMetrics {
    fn score(&self) -> f32 {
        self.reward
    }

    fn duration(&self) -> usize {
        self.steps
    }
}

// /// Deep Q-Network (DQN) agent
// pub struct DqnAgent<E, const S: usize, const A: usize, B: AutodiffBackend, M: DqnModel<B>>
// where
//     E: Environment<S, A>,
//     E::StateType: StateTensorConvertible<S> + State,
//     E::ActionType: ActionTensorConvertible<A> + Action + DiscreteAction,
// {
//     policy_net: Option<M>,
//     target_net: Option<M>,
//     epsilon: f64,
//     epsilon_min: f64,
//     epsilon_decay: f64,
//     state: PhantomData<E::StateType>,
//     action: PhantomData<E::ActionType>,
//     backend: PhantomData<B>,
//     device: B::Device,
//     stats: AgentStats<DqnMetrics>,
// }

// /// Builder for constructing a `DqnAgent` with configurable parameters.
// ///
// /// Provides sensible defaults aligned with the DQN reinforcement learning community standards:
// /// - `epsilon`: 1.0 (full exploration at start)
// /// - `epsilon_min`: 0.01 (minimum exploration rate)
// /// - `epsilon_decay`: 0.995 (per-step decay factor)
// /// - `window_size`: 10000 (replay buffer capacity)
// /// - `exploration_rate`: 0.0 (initial stat tracking rate)
// ///
// /// Key parameters and their significance:
// ///
// /// # Epsilon (ε): Exploration-Exploitation Trade-off
// /// - Controls the probability of taking a random action vs. greedy action
// /// - `epsilon=1.0`: 100% random (pure exploration)
// /// - `epsilon=0.0`: 0% random (pure exploitation/greedy)
// /// - Typical starting values: 0.1–1.0 depending on environment
// /// - Higher epsilon encourages discovering new state-action pairs
// ///
// /// # Epsilon-Min (ε_min): Exploration Floor
// /// - The lowest epsilon can decay to (prevents zero exploration)
// /// - Typical value: 0.01–0.05 (1–5% minimum exploration)
// /// - Ensures the agent occasionally tries unexpected actions
// /// - Helps escape local optima and discover better policies
// /// - Prevents the policy from becoming permanently suboptimal
// ///
// /// # Epsilon-Decay (decay_factor): Exploration Decay Rate
// /// - Multiplicative factor: new_epsilon = epsilon × decay_factor, each step
// /// - Value closer to 1.0 = slower decay (more gradual)
// /// - Value closer to 0.9 = faster decay (quick switch to exploitation)
// /// - Typical range: 0.99–0.9999
// /// - After N steps, epsilon ≈ ε_initial × (decay_factor)^N
// ///
// /// # Buffer Size (Replay Buffer Capacity)
// /// - Number of (state, action, reward, next_state, done) experiences stored
// /// - Larger buffers provide more diverse training data
// /// - Memory requirement: ~1KB–10KB per experience (depending on state size)
// /// - Typical values:
// ///   - Simple control: 1,000–10,000
// ///   - Atari: 1,000,000 (requires ~40GB RAM)
// ///   - Real-world robotics: 100,000–1,000,000
// /// - Larger buffers reduce correlation between training samples (improves learning)
// ///
// /// # Example
// /// ```ignore
// /// let agent = DQNAgent::builder(policy_net, target_net, device)
// ///     .epsilon(0.5)
// ///     .epsilon_min(0.05)
// ///     .buffer_size(5000)
// ///     .exploration_rate(0.1)
// ///     .build();
// /// ```
// #[derive(Debug)]
// pub struct DqnAgentBuilder<E, const S: usize, const A: usize, B: AutodiffBackend, M: DqnModel<B>>
// where
//     E: Environment<S, A>,
//     E::StateType: StateTensorConvertible<S> + State,
//     E::ActionType: ActionTensorConvertible<A> + Action,
// {
//     policy_net: M,
//     target_net: M,
//     epsilon: f64,
//     epsilon_min: f64,
//     epsilon_decay: f64,
//     device: B::Device,
//     exploration_rate: f64,
//     buffer_size: usize,
//     state: PhantomData<E::StateType>,
//     action: PhantomData<E::ActionType>,
//     backend: PhantomData<B>,
// }

// impl<E, const S: usize, const A: usize, B: AutodiffBackend, M: DqnModel<B>>
//     DqnAgentBuilder<E, S, A, B, M>
// where
//     E: Environment<S, A>,
//     E::StateType: StateTensorConvertible<S> + State,
//     E::ActionType: ActionTensorConvertible<A> + Action + DiscreteAction,
// {
//     /// Create a new builder with community-standard defaults.
//     ///
//     /// Community-standard defaults from the DQN literature:
//     /// - `epsilon`: 1.0 (start with 100% exploration)
//     /// - `epsilon_min`: 0.01 (explore at least 1% of the time)
//     /// - `epsilon_decay`: 0.995 (per-step multiplicative decay)
//     /// - `buffer_size`: 10000 (typical replay buffer capacity for Atari)
//     /// - `exploration_rate`: 0.0 (no initial stat tracking)
//     ///
//     /// # Arguments
//     /// - `policy_net` - The main Q-network for estimating action-values.
//     /// - `target_net` - The target Q-network for stable temporal-difference updates.
//     /// - `device` - Compute device (CPU/GPU) for tensor operations.
//     fn new(policy_net: M, target_net: M, device: B::Device) -> Self {
//         Self {
//             policy_net,
//             target_net,
//             // Community-standard defaults from Nature DQN paper and subsequent works
//             epsilon: 1.0,         // Full exploration at initialization
//             epsilon_min: 0.01,    // Minimum exploration (1%)
//             epsilon_decay: 0.995, // Multiplicative decay per step
//             device,
//             exploration_rate: 0.0, // Default stat tracking disabled
//             buffer_size: 10000,    // Typical replay buffer size (Atari games)
//             state: PhantomData,
//             action: PhantomData,
//             backend: PhantomData,
//         }
//     }

//     /// Set the initial exploration rate (epsilon).
//     ///
//     /// # Arguments
//     /// * `epsilon` - Exploration probability in the range (0.0, 1.0]. Typical values: 0.1–1.0.
//     ///
//     /// # Example
//     /// ```ignore
//     /// builder.epsilon(0.05) // 5% exploration
//     /// ```
//     pub fn epsilon(mut self, epsilon: f64) -> Self {
//         self.epsilon = epsilon;
//         self
//     }

//     /// Set the minimum exploration rate (epsilon_min).
//     ///
//     /// The agent will never explore less than this rate.
//     ///
//     /// # Arguments
//     /// * `epsilon_min` - Minimum exploration probability. Community standard: 0.01 (1%).
//     ///
//     /// # Example
//     /// ```ignore
//     /// builder.epsilon_min(0.05) // Never explore less than 5%
//     /// ```
//     pub fn epsilon_min(mut self, epsilon_min: f64) -> Self {
//         self.epsilon_min = epsilon_min;
//         self
//     }

//     /// Set the exploration decay factor (per-step multiplicative decay).
//     ///
//     /// # Arguments
//     /// * `epsilon_decay` - Multiplicative factor ∈ (0.99, 1.0). After each step, epsilon *= decay.
//     ///   Common values: 0.995 (gradual), 0.99 (slower), 0.9999 (very gradual).
//     ///
//     /// # Example
//     /// ```ignore
//     /// builder.epsilon_decay(0.999) // Very gradual exploration decay
//     /// ```
//     pub fn epsilon_decay(mut self, epsilon_decay: f64) -> Self {
//         self.epsilon_decay = epsilon_decay;
//         self
//     }

//     /// Set the replay buffer capacity (buffer_size).
//     ///
//     /// # Arguments
//     /// * `buffer_size` - Maximum experiences stored before oldest are discarded (cyclic buffer).
//     ///   Typical values: 5000–1000000 depending on environment/memory constraints.
//     ///
//     /// # Example
//     /// ```ignore
//     /// builder.buffer_size(50000) // Store up to 50k experiences
//     /// ```
//     pub fn buffer_size(mut self, buffer: usize) -> Self {
//         self.buffer_size = buffer;
//         self
//     }

//     /// Set the initial exploration rate for statistics tracking.
//     ///
//     /// This controls how stats module tracks initial behavior, typically left at 0.0.
//     ///
//     /// # Arguments
//     /// * `exploration_rate` - Initial rate for stat tracking. Default: 0.0.
//     pub fn exploration_rate(mut self, exploration_rate: f64) -> Self {
//         self.exploration_rate = exploration_rate;
//         self
//     }

//     /// Build the final `DqnAgent` with the configured parameters.
//     ///
//     /// # Returns
//     /// A fully initialized `DqnAgent` ready for interaction and training.
//     ///
//     /// # Example
//     /// ```ignore
//     /// let agent = DqnAgent::builder(policy_net, target_net, device)
//     ///     .epsilon(0.1)
//     ///     .buffer_size(5000)
//     ///     .build();
//     /// ```
//     pub fn build(self) -> DqnAgent<E, S, A, B, M> {
//         let stats: AgentStats<DqnMetrics> = AgentStats::new(self.buffer_size);
//         DqnAgent {
//             policy_net: Some(self.policy_net),
//             target_net: Some(self.target_net),
//             epsilon: self.epsilon,
//             epsilon_min: self.epsilon_min,
//             epsilon_decay: self.epsilon_decay,
//             state: PhantomData,
//             action: PhantomData,
//             backend: PhantomData,
//             device: self.device,
//             stats,
//         }
//     }
// }

// /// Implementation of the `NeuralAgent` trait for `DqnAgent`.
// ///
// /// # Trait Bounds Explanation
// ///
// /// This implementation requires several trait bounds to ensure type safety and correctness:
// ///
// /// 1. **`E: Environment<S, A>`**: The environment type must implement the core environment
// ///    interface with state dimension `S` and action dimension `A`.
// ///
// /// 2. **`E::StateType: StateTensorConvertible<S> + State`**: The state type must be:
// ///    - Convertible to tensors with shape `S` via `StateTensorConvertible`
// ///    - Implement the `State` marker trait for state semantics
// ///
// /// 3. **`E::ActionType: ActionTensorConvertible<A> + Action + DiscreteAction`**: The action
// ///    type must be:
// ///    - Convertible to/from tensors with shape `A` via `ActionTensorConvertible`
// ///    - Implement the `Action` marker trait for action semantics
// ///    - **Implement the `DiscreteAction` trait**
// ///
// /// # Why DiscreteAction is Required
// ///
// /// DQN is fundamentally designed for discrete action spaces. The algorithm:
// /// - Computes Q-values for each possible action: `Q(s, a)` where `a ∈ {0, 1, ..., A-1}`
// /// - Selects actions via `argmax`: `a* = argmax_a Q(s, a)`
// /// - Uses action indices in replay buffer updates
// ///
// /// Without the `DiscreteAction` bound, the type system cannot guarantee that:
// /// - Actions can be converted to/from numeric indices
// /// - The `from_index(index)` method is available for action reconstruction
// /// - The action type is semantically discrete (not continuous)
// ///
// /// # Relationship to Agent Trait
// ///
// /// The `NeuralAgent` trait itself doesn't require `DiscreteAction` because it's more
// /// general; it could apply to agents with any action representation. However, DQN
// /// specifically needs discrete actions, so this implementation adds the constraint.
// ///
// /// # Type Parameter M
// ///
// /// The model type `M` must implement `DqnModel<S, A, B>`, which provides:
// /// - Forward pass computation: `forward(state_tensor) -> q_values`
// /// - Model parameterization compatible with autodiff backend `B`
// ///
// /// # Type Parameter O
// ///
// /// The optimizer type `O` must implement `Optimizer<M, B>`, which enables:
// /// - Gradient-based optimization of the DQN model `M` using the autodiff backend `B`
// /// - Parameter updates during the training loop via `train_step`
// /// - Compatibility with the burn framework's optimizer ecosystem
// ///
// /// The constraint `O: Optimizer<M, B>` ensures type safety:
// /// - The optimizer understands how to optimize the specific model type `M`
// /// - The optimizer is backend-aware and uses the correct autodiff backend `B`
// /// - Training algorithms can abstract over different optimizer implementations
// ///   (e.g., SGD, Adam, RMSprop) without changing the core DQN logic
// impl<E, const S: usize, const A: usize, B: AutodiffBackend, M, O> NeuralAgent<E, S, A, B, O>
//     for DqnAgent<E, S, A, B, M>
// where
//     E: Environment<S, A>,
//     E::StateType: StateTensorConvertible<S> + State,
//     E::ActionType: ActionTensorConvertible<A> + Action + DiscreteAction,
//     M: DqnModel<B>,
//     O: Optimizer<M, B>,
// {
//     type Model = M;

//     fn model(&self) -> Option<&Self::Model> {
//         self.policy_net.as_ref()
//     }

//     fn model_mut(&mut self) -> Option<&mut Self::Model> {
//         self.policy_net.as_mut()
//     }

//     //     /// Neural Network/Q-Network
//     //     /// The specific process of using stored experiences (data) to update the neural network's weights to better approximate the optimal Q-function.
//     //     /// This method focuses purely on the Deep Learning aspect, which is often performed much less frequently than the interaction step, and only after enough experiences have been collected.
//     //     /// Role: Handles the supervised-like learning aspect (Q-function approximation):
//     //     /// - Samples a random mini-batch of experiences from the replay memory.
//     //     /// - Calculates the tredicted Q-values ($y_i$) using the Bellman Equation and the target network (for stability).
//     //     /// Calculates the Predicted Q-Values for the same actions using the Main Q-Network.
//     //     /// Computes the loss between the target and the predicted values.
//     //     /// Performs backpropagation (gradient descent) to update the weights of the Main Q-Network.
//     fn train_step(
//         &mut self,
//         batch: &TrainingBatch<S, A, B>,
//         optimizer: &mut O,
//     ) -> Result<f32, Self::Error> {
//         todo!("perform a single training step.")
//     }
// }

// // Implement Agent trait for DQNAgent
// impl<E, const S: usize, const A: usize, B: AutodiffBackend, M: DqnModel<B>> Agent<E, S, A, B>
//     for DqnAgent<E, S, A, B, M>
// where
//     E: Environment<S, A>,
//     E::StateType: StateTensorConvertible<S> + State,
//     E::ActionType: ActionTensorConvertible<A> + Action + DiscreteAction,
// {
//     type Error = DqnAgentError;

//     /// Choose an action using epsilon-greedy strategy
//     fn act(
//         &mut self,
//         state: &E::StateType,
//         device: &B::Device,
//     ) -> Result<E::ActionType, Self::Error> {
//         // Check if policy network is initialized
//         let policy_net = self
//             .policy_net
//             .as_ref()
//             .ok_or(DqnAgentError::ModelNotInitialized)?;

//         todo!()
//         // // Exploration vs exploitation
//         // let mut rng = rand::rng(); // uniform distribution between [0,1)
//         // if rng.random::<f64>() < self.epsilon {
//         //     // Exploration: choose random action
//         //     Ok(E::ActionType::random())
//         // } else {
//         //     // Exploitation: choose greedy action
//         //     let state_tensor: Tensor<B, S> = state.to_tensor(device);
//         //     let q_values: Tensor<B, A> = policy_net.forward(state_tensor.unsqueeze::<S>());

//         //     // Find the action with maximum Q-value
//         //     // For simplicity, assuming scalar output per action
//         //     let max_index = q_values.argmax(A - 1).into_scalar().to_usize();

//         //     // Convert index to action
//         //     Ok(E::ActionType::from_index(max_index))
//         // }
//     }

//     fn learn(&mut self, batch: &TrainingBatch<S, A, B>) -> Result<f32, Self::Error> {
//         // think about what learning actually means here. It would need to call self.train_step(batch, self.optimizer)
//         todo!("Implement the learning for the agent.")
//     }

//     fn update_exploration(&mut self, step: usize) -> Result<(), Self::Error> {
//         // Decay epsilon
//         self.epsilon = f64::max(self.epsilon_min, self.epsilon * self.epsilon_decay);
//         Ok(())
//     }

//     fn exploration_rate(&self) -> f64 {
//         self.epsilon
//     }

//     fn save(&self, path: &str) -> Result<(), Self::Error> {
//         // Check if policy network exists to save
//         let policy_net = self
//             .policy_net
//             .as_ref()
//             .ok_or(DqnAgentError::ModelNotInitialized)?;

//         // Serialize model weights
//         // Implementation depends on burn's serialization capabilities
//         todo!()
//     }

//     fn load(&mut self, path: &str) -> Result<(), Self::Error> {
//         // Load and deserialize model weights
//         todo!()
//     }

//     fn reset(&mut self) {
//         todo!("Clear the AgentStats")
//         // // Reset statistics, keep network weights
//         // self.stats.episode_rewards.clear();
//         // self.stats.episode_losses.clear();
//     }
// }

// impl<E, const S: usize, const A: usize, B: AutodiffBackend, M: DqnModel<B>> DqnAgent<E, S, A, B, M>
// where
//     E: Environment<S, A>,
//     E::StateType: StateTensorConvertible<S> + State,
//     E::ActionType: ActionTensorConvertible<A> + Action + DiscreteAction,
// {
//     /// Get a reference to the policy network
//     pub fn policy_net(&self) -> Option<&M> {
//         self.policy_net.as_ref()
//     }

//     /// Decay epsilon for exploration-exploitation tradeoff
//     pub fn decay_epsilon(&mut self) {
//         self.epsilon = f64::max(self.epsilon_min, self.epsilon * self.epsilon_decay);
//     }

//     /// Update the target network with polyak averaging
//     pub fn update_target_network(&mut self, tau: f32) -> Result<(), DqnAgentError> {
//         let policy_net = self
//             .policy_net
//             .as_ref()
//             .ok_or(DqnAgentError::ModelNotInitialized)?;
//         let target_net = self
//             .target_net
//             .as_ref()
//             .ok_or(DqnAgentError::ModelNotInitialized)?;

//         // Implement soft update: θ_target = τ*θ_policy + (1-τ)*θ_target
//         todo!("Implement soft_update method which is a required method for the trait DqnModel.")
//     }

//     /// Collect experience by interacting with the environment
//     pub fn collect_experience(
//         &mut self,
//         env: &mut E,
//         memory: &mut PrioritizedExperienceReplay<E, S, A>,
//         num_steps: usize,
//     ) -> Result<Vec<f32>, DqnAgentError> {
//         todo!()
//         // let mut rewards = Vec::new();

//         // for _ in 0..num_steps {
//         //     // Get current state
//         //     let snapshot = env.reset();
//         //     let state = snapshot.state;

//         //     // Select action using epsilon-greedy
//         //     let action = self.select_action_with_exploration(&state, self.epsilon)?;

//         //     // Take step in environment
//         //     let next_snapshot = env.step(action.clone())?;

//         //     // Store experience in replay memory
//         //     let experience = Experience {
//         //         state,
//         //         action,
//         //         reward: next_snapshot.reward,
//         //         next_state: next_snapshot.state.clone(),
//         //         done: next_snapshot.done,
//         //     };

//         //     memory.push(experience);
//         //     rewards.push(next_snapshot.reward.into());

//         //     if next_snapshot.done {
//         //         break;
//         //     }
//         // }

//         // Ok(rewards)
//     }

//     /// Agent-Environment Interaction
//     /// Complete learning step: collect experience and train
//     /// The overarching process of the agent discovering the optimal policy (maximizing cumulative reward) through trial-and-error in the environment.
//     /// Key Operations/Focus
//     /// Action selection (e.g., epsilon-greedy), environment stepping, experience collection, and storing the experience in Replay Memory
//     /// This method typically focuses on the agent's interaction with the environment, often corresponding to a single timestep or a few timesteps.
//     /// Role: Handles the reinforcement learning aspect.
//     /// - Takes the current state ($s$).
//     /// - Decides an action ($a$) using the current policy (e.g., epsilon-greedy with the Q-netowrk).
//     /// - Executes $a$ in the environment to get the next state ($s'$), reward ($r$), and whether the episode is done.
//     /// - Stores the experience in the replay memory buffer.
//     pub fn learn_step<O: Optimizer<M, B>>(
//         &mut self,
//         env: &mut E,
//         memory: &mut PrioritizedExperienceReplay<E, S, A>,
//         optimizer: &mut O,
//         config: &DqnTrainingConfig,
//         step_count: usize,
//     ) -> Result<DqnMetrics, DqnAgentError> {
//         todo!()
//         // // 1. Collect new experiences
//         // let collected_rewards = self.collect_experience(env, memory, config.steps_per_episode)?;

//         // // 2. Check if we have enough samples to train
//         // if memory.len() < config.batch_size {
//         //     return Ok(LearningStats {
//         //         loss: 0.0,
//         //         avg_reward: collected_rewards.iter().sum::<f32>() / collected_rewards.len() as f32,
//         //         epsilon: self.epsilon,
//         //     });
//         // }

//         // // 3. Sample batch from replay memory
//         // let mut rng = rand::rng();
//         // let batch = memory.sample_batch(config.batch_size, &B::Device::default(), &mut rng)?;

//         // // 4. Perform training step
//         // let loss = self.learn(&batch)?;

//         // // 5. Update target network periodically
//         // if step_count % config.target_update_frequency == 0 {
//         //     self.update_target_network(config.tau)?;
//         // }

//         // // 6. Decay epsilon
//         // self.decay_epsilon();

//         // Ok(LearningStats {
//         //     loss,
//         //     avg_reward: collected_rewards.iter().sum::<f32>() / collected_rewards.len() as f32,
//         //     epsilon: self.epsilon,
//         // })
//     }

//     fn stats(&self) -> AgentStats<DqnMetrics> {
//         self.stats.clone() // todo! would it better to return a reference? Would this pass the ownership to the caller?
//     }
// }
