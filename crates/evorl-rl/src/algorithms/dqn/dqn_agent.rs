use crate::algorithms::dqn::dqn_config::DQNTrainingConfig;
use crate::algorithms::dqn::dqn_model::DQNModel;
use burn::module::AutodiffModule;

use burn::optim::Optimizer;
use burn::tensor::backend::AutodiffBackend;
use evorl_core::action::{Action, ActionTensorConvertible};
use evorl_core::agent::{Agent, NeuralAgent, TrainableAgent};
use evorl_core::environment::Environment;
use evorl_core::memory::{ReplayBuffer, TrainingBatch};
use evorl_core::metrics::{AgentStats, PerformanceRecord};
use evorl_core::state::{State, StateTensorConvertible};
use std::marker::PhantomData;

// Define error type for DQNAgent
// todo! implement From
#[derive(Debug)]
pub enum DQNAgentError {
    ModelNotInitialized,
    TensorConversionFailed(String),
    InvalidAction(String),
    IoError(std::io::Error),
}

impl std::fmt::Display for DQNAgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DQNAgentError::ModelNotInitialized => write!(f, "DQN model not initialized"),
            DQNAgentError::TensorConversionFailed(msg) => {
                write!(f, "Tensor conversion failed: {}", msg)
            }
            DQNAgentError::InvalidAction(msg) => {
                write!(f, "Invalid action: {}", msg)
            }
            DQNAgentError::IoError(err) => {
                write!(f, "IO error: {}", err)
            }
        }
    }
}

impl std::error::Error for DQNAgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DQNAgentError::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DQNAgentError {
    fn from(error: std::io::Error) -> Self {
        DQNAgentError::IoError(error)
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

/// Deep Q-Network (DQN) agent
pub struct DQNAgent<E, const S: usize, const A: usize, B: AutodiffBackend, M: DQNModel<B, S>>
where
    E: Environment<S, A>,
    E::StateType: StateTensorConvertible<S> + State,
    E::ActionType: ActionTensorConvertible<A> + Action,
{
    policy_net: Option<M>,
    target_net: Option<M>,
    epsilon: f64,
    epsilon_min: f64,
    epsilon_decay: f64,
    state: PhantomData<E::StateType>,
    action: PhantomData<E::ActionType>,
    backend: PhantomData<B>,
    device: B::Device,
    stats: AgentStats<DqnMetrics>,
}

/// Builder for constructing a `DQNAgent` with configurable parameters.
///
/// Provides sensible defaults aligned with the DQN reinforcement learning community standards:
/// - `epsilon`: 1.0 (full exploration at start)
/// - `epsilon_min`: 0.01 (minimum exploration rate)
/// - `epsilon_decay`: 0.995 (per-step decay factor)
/// - `window_size`: 10000 (replay buffer capacity)
/// - `exploration_rate`: 0.0 (initial stat tracking rate)
///
/// Key parameters and their significance:
///
/// # Epsilon (ε): Exploration-Exploitation Trade-off
/// - Controls the probability of taking a random action vs. greedy action
/// - `epsilon=1.0`: 100% random (pure exploration)
/// - `epsilon=0.0`: 0% random (pure exploitation/greedy)
/// - Typical starting values: 0.1–1.0 depending on environment
/// - Higher epsilon encourages discovering new state-action pairs
///
/// # Epsilon-Min (ε_min): Exploration Floor
/// - The lowest epsilon can decay to (prevents zero exploration)
/// - Typical value: 0.01–0.05 (1–5% minimum exploration)
/// - Ensures the agent occasionally tries unexpected actions
/// - Helps escape local optima and discover better policies
/// - Prevents the policy from becoming permanently suboptimal
///
/// # Epsilon-Decay (decay_factor): Exploration Decay Rate
/// - Multiplicative factor: new_epsilon = epsilon × decay_factor, each step
/// - Value closer to 1.0 = slower decay (more gradual)
/// - Value closer to 0.9 = faster decay (quick switch to exploitation)
/// - Typical range: 0.99–0.9999
/// - After N steps, epsilon ≈ ε_initial × (decay_factor)^N
///
/// # Window Size (Replay Buffer Capacity)
/// - Number of (state, action, reward, next_state, done) experiences stored
/// - Larger buffers provide more diverse training data
/// - Memory requirement: ~1KB–10KB per experience (depending on state size)
/// - Typical values:
///   - Simple control: 1,000–10,000
///   - Atari: 1,000,000 (requires ~40GB RAM)
///   - Real-world robotics: 100,000–1,000,000
/// - Larger buffers reduce correlation between training samples (improves learning)
///
/// # Example
/// ```ignore
/// let agent = DQNAgent::builder(policy_net, target_net, device)
///     .epsilon(0.5)
///     .epsilon_min(0.05)
///     .history_size(5000)
///     .exploration_rate(0.1)
///     .build();
/// ```
#[derive(Debug)]
pub struct DQNAgentBuilder<E, const S: usize, const A: usize, B: AutodiffBackend, M: DQNModel<B, S>>
where
    E: Environment<S, A>,
    E::StateType: StateTensorConvertible<S> + State,
    E::ActionType: ActionTensorConvertible<A> + Action,
{
    policy_net: M,
    target_net: M,
    epsilon: f64,
    epsilon_min: f64,
    epsilon_decay: f64,
    device: B::Device,
    exploration_rate: f64,
    window_size: usize,
    state: PhantomData<E::StateType>,
    action: PhantomData<E::ActionType>,
    backend: PhantomData<B>,
}

impl<E, const S: usize, const A: usize, B: AutodiffBackend, M: DQNModel<B, S>>
    DQNAgentBuilder<E, S, A, B, M>
where
    E: Environment<S, A>,
    E::StateType: StateTensorConvertible<S> + State,
    E::ActionType: ActionTensorConvertible<A> + Action,
{
    /// Create a new builder with community-standard defaults.
    ///
    /// Community-standard defaults from the DQN literature:
    /// - `epsilon`: 1.0 (start with 100% exploration)
    /// - `epsilon_min`: 0.01 (explore at least 1% of the time)
    /// - `epsilon_decay`: 0.995 (per-step multiplicative decay)
    /// - `history_size`: 10000 (typical replay buffer capacity for Atari)
    /// - `exploration_rate`: 0.0 (no initial stat tracking)
    ///
    /// # Arguments
    /// - `policy_net` - The main Q-network for estimating action-values.
    /// - `target_net` - The target Q-network for stable temporal-difference updates.
    /// - `device` - Compute device (CPU/GPU) for tensor operations.
    fn new(policy_net: M, target_net: M, device: B::Device) -> Self {
        Self {
            policy_net,
            target_net,
            // Community-standard defaults from Nature DQN paper and subsequent works
            epsilon: 1.0,         // Full exploration at initialization
            epsilon_min: 0.01,    // Minimum exploration (1%)
            epsilon_decay: 0.995, // Multiplicative decay per step
            device,
            exploration_rate: 0.0, // Default stat tracking disabled
            window_size: 10000,    // Typical replay buffer size (Atari games)
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
        }
    }

    /// Set the initial exploration rate (epsilon).
    ///
    /// # Arguments
    /// * `epsilon` - Exploration probability in the range (0.0, 1.0]. Typical values: 0.1–1.0.
    ///
    /// # Example
    /// ```ignore
    /// builder.epsilon(0.05) // 5% exploration
    /// ```
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the minimum exploration rate (epsilon_min).
    ///
    /// The agent will never explore less than this rate.
    ///
    /// # Arguments
    /// * `epsilon_min` - Minimum exploration probability. Community standard: 0.01 (1%).
    ///
    /// # Example
    /// ```ignore
    /// builder.epsilon_min(0.05) // Never explore less than 5%
    /// ```
    pub fn epsilon_min(mut self, epsilon_min: f64) -> Self {
        self.epsilon_min = epsilon_min;
        self
    }

    /// Set the exploration decay factor (per-step multiplicative decay).
    ///
    /// # Arguments
    /// * `epsilon_decay` - Multiplicative factor ∈ (0.99, 1.0). After each step, epsilon *= decay.
    ///   Common values: 0.995 (gradual), 0.99 (slower), 0.9999 (very gradual).
    ///
    /// # Example
    /// ```ignore
    /// builder.epsilon_decay(0.999) // Very gradual exploration decay
    /// ```
    pub fn epsilon_decay(mut self, epsilon_decay: f64) -> Self {
        self.epsilon_decay = epsilon_decay;
        self
    }

    /// Set the replay buffer capacity (window_size).
    ///
    /// # Arguments
    /// * `window_size` - Maximum experiences stored before oldest are discarded (cyclic buffer).
    ///   Typical values: 5000–1000000 depending on environment/memory constraints.
    ///
    /// # Example
    /// ```ignore
    /// builder.window_size(50000) // Store up to 50k experiences
    /// ```
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set the initial exploration rate for statistics tracking.
    ///
    /// This controls how stats module tracks initial behavior, typically left at 0.0.
    ///
    /// # Arguments
    /// * `exploration_rate` - Initial rate for stat tracking. Default: 0.0.
    pub fn exploration_rate(mut self, exploration_rate: f64) -> Self {
        self.exploration_rate = exploration_rate;
        self
    }

    /// Build the final `DQNAgent` with the configured parameters.
    ///
    /// # Returns
    /// A fully initialized `DQNAgent` ready for interaction and training.
    ///
    /// # Example
    /// ```ignore
    /// let agent = DQNAgent::builder(policy_net, target_net, device)
    ///     .epsilon(0.1)
    ///     .history_size(5000)
    ///     .build();
    /// ```
    pub fn build(self) -> DQNAgent<E, S, A, B, M> {
        let stats: AgentStats<DqnMetrics> = AgentStats::new(self.window_size);
        DQNAgent {
            policy_net: Some(self.policy_net),
            target_net: Some(self.target_net),
            epsilon: self.epsilon,
            epsilon_min: self.epsilon_min,
            epsilon_decay: self.epsilon_decay,
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
            device: self.device,
            stats,
        }
    }
}

// Implement Agent trait for DQNAgent
impl<E, const S: usize, const A: usize, B: AutodiffBackend, M: DQNModel<B, S>> Agent<E, S, A, B>
    for DQNAgent<E, S, A, B, M>
where
    E: Environment<S, A>,
    E::StateType: StateTensorConvertible<S> + State,
    E::ActionType: ActionTensorConvertible<A> + Action,
{
    type Error = DQNAgentError;

    /// Choose an action using epsilon-greedy strategy
    fn act(&mut self, state: &E::StateType) -> Result<E::ActionType, Self::Error> {
        todo!()
        // // Check if policy network is initialized
        // let policy_net = self
        //     .policy_net
        //     .as_ref()
        //     .ok_or(DQNAgentError::ModelNotInitialized)?;

        // // Exploration vs exploitation
        // if random::<f64>() < self.epsilon {
        //     // Exploration: choose random action
        //     Ok(E::ActionType::random())
        // } else {
        //     // Exploitation: choose greedy action
        //     let state_tensor = state.to_tensor(&B::Device::default());
        //     let q_values = policy_net.forward(state_tensor.unsqueeze::<S>());

        //     // Find the action with maximum Q-value
        //     // For simplicity, assuming scalar output per action
        //     let max_index = q_values.argmax(-1).into_scalar() as usize;

        //     // Convert index to action
        //     Ok(E::ActionType::from_index(max_index))
        // }
    }

    fn learn(&mut self, batch: &TrainingBatch<B, S, A>) -> Result<f32, Self::Error> {
        // Implementation of learning step (DQN learning with target network)
        unimplemented!() // Placeholder - would implement actual training logic
    }

    fn update_exploration(&mut self, step: usize) -> Result<(), Self::Error> {
        // Decay epsilon
        self.epsilon = f64::max(self.epsilon_min, self.epsilon * self.epsilon_decay);
        Ok(())
    }

    fn exploration_rate(&self) -> f64 {
        self.epsilon
    }

    fn save(&self, path: &str) -> Result<(), Self::Error> {
        // Check if policy network exists to save
        let policy_net = self
            .policy_net
            .as_ref()
            .ok_or(DQNAgentError::ModelNotInitialized)?;

        // Serialize model weights
        // Implementation depends on burn's serialization capabilities
        unimplemented!()
    }

    fn load(&mut self, path: &str) -> Result<(), Self::Error> {
        // Load and deserialize model weights
        unimplemented!()
    }

    fn reset(&mut self) {
        todo!("Clear the AgentStats")
        // // Reset statistics, keep network weights
        // self.stats.episode_rewards.clear();
        // self.stats.episode_losses.clear();
    }
}

impl<E, const S: usize, const A: usize, B: AutodiffBackend, M> NeuralAgent<E, S, A, B>
    for DQNAgent<E, S, A, B, M>
where
    E: Environment<S, A>,
    M: DQNModel<B, S>,
{
    type Model = M;

    fn model(&self) -> Option<&Self::Model> {
        self.policy_net.as_ref()
    }

    fn model_mut(&mut self) -> Option<&mut Self::Model> {
        self.policy_net.as_mut()
    }
}

impl<E, const S: usize, const A: usize, B: AutodiffBackend, M, O> TrainableAgent<E, S, A, B, O>
    for DQNAgent<E, S, A, B, M>
where
    E: Environment<S, A>,
    M: DQNModel<B, S>,
    O: Optimizer<M, B>,
{
    fn train_step(
        &mut self,
        batch: &TrainingBatch<B, S, A>,
        optimizer: &mut O,
    ) -> Result<f32, Self::Error> {
        unimplemented!()
    }
}

impl<E, const S: usize, const A: usize, B: AutodiffBackend, M: DQNModel<B, S>>
    DQNAgent<E, S, A, B, M>
where
    E: Environment<S, A>,
    E::StateType: StateTensorConvertible<S> + State,
    E::ActionType: ActionTensorConvertible<A> + Action,
{
    // todo! remove here and suggest using the builder?
    pub fn new(
        policy_net: M,
        target_net: M,
        epsilon: f64,
        epsilon_min: f64,
        epsilon_decay: f64,
        device: B::Device,
        exploration_rate: f64,
        window_size: usize,
    ) -> Self {
        let stats: AgentStats<DqnMetrics> = AgentStats::new(window_size);
        Self {
            policy_net: Some(policy_net),
            target_net: Some(target_net),
            epsilon,
            epsilon_min,
            epsilon_decay,
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
            device,
            stats,
        }
    }

    /// Get a reference to the policy network
    pub fn policy_net(&self) -> Option<&M> {
        self.policy_net.as_ref()
    }

    /// Decay epsilon for exploration-exploitation tradeoff
    pub fn decay_epsilon(&mut self) {
        self.epsilon = f64::max(self.epsilon_min, self.epsilon * self.epsilon_decay);
    }

    /// Update the target network with polyak averaging
    pub fn update_target_network(&mut self, tau: f32) -> Result<(), DQNAgentError> {
        let policy_net = self
            .policy_net
            .as_ref()
            .ok_or(DQNAgentError::ModelNotInitialized)?;
        let target_net = self
            .target_net
            .as_ref()
            .ok_or(DQNAgentError::ModelNotInitialized)?;

        // Implement soft update: θ_target = τ*θ_policy + (1-τ)*θ_target

        unimplemented!()
    }

    /// Select action with epsilon-greedy strategy (alternative implementation)
    pub fn select_action_with_exploration(
        &self,
        state: &E::StateType,
        eps_threshold: f64,
    ) -> Result<E::ActionType, DQNAgentError> {
        todo!()
        // if random::<f64>() < eps_threshold {
        //     // Exploration
        //     Ok(E::ActionType::random())
        // } else {
        //     // Exploitation - use policy network
        //     let policy_net = self
        //         .policy_net
        //         .as_ref()
        //         .ok_or(DQNAgentError::ModelNotInitialized)?;

        //     let state_tensor = state.to_tensor(&B::Device::default());
        //     let q_values = policy_net.forward(state_tensor.unsqueeze::<E::StateType::R1>());
        //     let max_index = q_values.argmax(-1).into_scalar() as usize;

        //     Ok(E::ActionType::from_index(max_index))
        // }
    }

    /// Collect experience by interacting with the environment
    pub fn collect_experience<const CAP: usize>(
        &mut self,
        env: &mut E,
        memory: &mut ReplayBuffer<E, S, A, CAP>,
        num_steps: usize,
    ) -> Result<Vec<f32>, DQNAgentError> {
        todo!()
        // let mut rewards = Vec::new();

        // for _ in 0..num_steps {
        //     // Get current state
        //     let snapshot = env.reset();
        //     let state = snapshot.state;

        //     // Select action using epsilon-greedy
        //     let action = self.select_action_with_exploration(&state, self.epsilon)?;

        //     // Take step in environment
        //     let next_snapshot = env.step(action.clone())?;

        //     // Store experience in replay memory
        //     let experience = Experience {
        //         state,
        //         action,
        //         reward: next_snapshot.reward,
        //         next_state: next_snapshot.state.clone(),
        //         done: next_snapshot.done,
        //     };

        //     memory.push(experience);
        //     rewards.push(next_snapshot.reward.into());

        //     if next_snapshot.done {
        //         break;
        //     }
        // }

        // Ok(rewards)
    }

    /// Agent-Environment Interaction
    /// Complete learning step: collect experience and train
    /// The overarching process of the agent discovering the optimal policy (maximizing cumulative reward) through trial-and-error in the environment.
    /// Key Operations/Focus
    /// Action selection (e.g., epsilon-greedy), environment stepping, experience collection, and storing the experience in Replay Memory
    /// This method typically focuses on the agent's interaction with the environment, often corresponding to a single timestep or a few timesteps.
    /// Role: Handles the reinforcement learning aspect.
    /// - Takes the current state ($s$).
    /// - Decides an action ($a$) using the current policy (e.g., epsilon-greedy with the Q-netowrk).
    /// - Executes $a$ in the environment to get the next state ($s'$), reward ($r$), and whether the episode is done.
    /// - Stores the experience in the replay memory buffer.
    pub fn learn_step<const CAP: usize, O: Optimizer<M, B>>(
        &mut self,
        env: &mut E,
        memory: &mut ReplayBuffer<E, S, A, CAP>,
        optimizer: &mut O,
        config: &DQNTrainingConfig,
        step_count: usize,
    ) -> Result<DqnMetrics, DQNAgentError> {
        unimplemented!()
        // // 1. Collect new experiences
        // let collected_rewards = self.collect_experience(env, memory, config.steps_per_episode)?;

        // // 2. Check if we have enough samples to train
        // if memory.len() < config.batch_size {
        //     return Ok(LearningStats {
        //         loss: 0.0,
        //         avg_reward: collected_rewards.iter().sum::<f32>() / collected_rewards.len() as f32,
        //         epsilon: self.epsilon,
        //     });
        // }

        // // 3. Sample batch from replay memory
        // let mut rng = rand::rng();
        // let batch = memory.sample_batch(config.batch_size, &B::Device::default(), &mut rng)?;

        // // 4. Perform training step
        // let loss = self.learn(&batch)?;

        // // 5. Update target network periodically
        // if step_count % config.target_update_frequency == 0 {
        //     self.update_target_network(config.tau)?;
        // }

        // // 6. Decay epsilon
        // self.decay_epsilon();

        // Ok(LearningStats {
        //     loss,
        //     avg_reward: collected_rewards.iter().sum::<f32>() / collected_rewards.len() as f32,
        //     epsilon: self.epsilon,
        // })
    }

    fn stats(&self) -> AgentStats<DqnMetrics> {
        self.stats.clone() // todo! would it better to return a reference? Would this pass the ownership to the caller?
    }
}

// ---
impl<
        E,
        const S: usize,
        const A: usize,
        B: AutodiffBackend,
        M: DQNModel<B, S> + AutodiffModule<B>,
    > DQNAgent<E, S, A, B, M>
where
    E: Environment<S, A>,
{
    /// Neural Network/Q-Network
    /// The specific process of using stored experiences (data) to update teh neural network's weights to better approximate the optimal Q-function.
    /// This method focuses purely on the Deep Learning aspect, which is often performed much less frequently than the interaction step, and only after enough experiences have been collected.
    /// Role: Handles the supervised-like learning aspect (Q-function approximation):
    /// - Samples a random mini-batch of experiences from the replay memory.
    /// - Calculates the tredicted Q-values ($y_i$) using the Bellman Equation and the target network (for stability).
    /// Calculates the Predicted Q-Values for the same actions using the Main Q-Network.
    /// Computes the loss between the target and the predicted values.
    /// Performs backpropagation (gradient descent) to update the weights of the Main Q-Network.
    pub fn train<const CAP: usize>(
        &mut self,
        mut policy_net: M,
        memory: &ReplayBuffer<E, S, A, CAP>,
        optimizer: &mut (impl Optimizer<M, B> + Sized),
        config: &DQNTrainingConfig,
    ) -> M {
        unimplemented!()
    }

    // todo!
    // pub fn valid(mut self) -> DQNAgent<E, S, A, B::InnerBackend, M::InnerModule>
    // where
    //     <M as AutodiffModule<B>>::InnerModule: DQNModel<<B as AutodiffBackend>::InnerBackend>,
    // {
    //     DQNAgent::<E, B::InnerBackend, M::InnerModule>::new(
    //         self.policy_net.take().unwrap().valid(),
    //         self.target_net.take().unwrap().valid(),
    //         self.epsilon,
    //         self.device,
    //         self.epsilon,
    //         self.stats.history_size(),
    //     )
    // }
}
