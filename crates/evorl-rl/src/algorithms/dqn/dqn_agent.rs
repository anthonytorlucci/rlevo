use crate::agent::dqn::dqn_config::DQNTrainingConfig;
use crate::agent::dqn::dqn_model::DQNModel;
use crate::base::burnrl_action::{Action, ActionTensorConvertible};
use crate::base::burnrl_agent::{Agent, NeuralAgent, TrainableAgent};
use crate::base::burnrl_environment::Environment;
use crate::base::burnrl_memory::{Experience, ReplayMemory, TrainingBatch};
use crate::base::burnrl_state::{State, StateTensorConvertible};
// use crate::utils::{
//     convert_tensor_to_action, ref_to_action_tensor, ref_to_not_done_tensor, ref_to_reward_tensor,
//     ref_to_state_tensor, to_state_tensor, update_parameters,
// };
use crate::base::burnrl_metrics::{AgentStats, CoreMetrics, EpisodeStats};
use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::Optimizer;
use burn::tensor::backend::{AutodiffBackend, Backend};
use rand::random;
use std::marker::PhantomData;

// Define error type for DQNAgent
#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct LearningStats {
    pub loss: f32,
    pub avg_reward: f32,
    pub epsilon: f64,
}

/// Lightweight metrics for a single learning step (just an alias for clarity)
pub type StepMetrics = CoreMetrics;

#[derive(Debug, Clone)]
pub struct TrainingHistory {
    pub episodes: Vec<EpisodeStats>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            episodes: Vec::new(),
        }
    }

    pub fn add_episode(&mut self, stats: EpisodeStats) {
        self.episodes.push(stats);
    }

    pub fn best_reward(&self) -> Option<f32> {
        self.episodes
            .iter()
            .map(|e| e.total_reward)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Iterator over episodes
    pub fn iter(&self) -> impl Iterator<Item = &EpisodeStats> {
        self.episodes.iter()
    }
}

/// Deep Q-Network (DQN) agent
pub struct DQNAgent<E, const S: usize, const A: usize, B: AutodiffBackend, M: DQNModel<B>>
where
    E: Environment<S, A>,
    E::StateType: StateTensorConvertible<S> + State,
    E::ActionType: ActionTensorConvertible<A> + Action,
{
    policy_net: Option<M>,
    target_net: Option<M>,
    epsilon: f64,
    state: PhantomData<E::StateType>,
    action: PhantomData<E::ActionType>,
    backend: PhantomData<B>,
    device: B::Device,
    stats: AgentStats,
}

// Implement Agent trait for DQNAgent
impl<E, const S: usize, const A: usize, B: AutodiffBackend, M: DQNModel<B>> Agent<E, S, A, B>
    for DQNAgent<E, S, A, B, M>
where
    E: Environment<S, A>,
    E::StateType: StateTensorConvertible<S> + State,
    E::ActionType: ActionTensorConvertible<A> + Action,
{
    type Error = DQNAgentError;

    /// Choose an action using epsilon-greedy strategy
    fn act(&mut self, state: &E::StateType) -> Result<E::ActionType, Self::Error> {
        // Check if policy network is initialized
        let policy_net = self
            .policy_net
            .as_ref()
            .ok_or(DQNAgentError::ModelNotInitialized)?;

        // Exploration vs exploitation
        if random::<f64>() < self.epsilon {
            // Exploration: choose random action
            Ok(E::ActionType::random())
        } else {
            // Exploitation: choose greedy action
            let state_tensor = state.to_tensor(&B::Device::default());
            let q_values = policy_net.forward(state_tensor.unsqueeze::<E::StateType::R1>());

            // Find the action with maximum Q-value
            // For simplicity, assuming scalar output per action
            let max_index = q_values.argmax(-1).into_scalar() as usize;

            // Convert index to action
            Ok(E::ActionType::from_index(max_index))
        }
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
        // Reset statistics, keep network weights
        self.stats.episode_rewards.clear();
        self.stats.episode_losses.clear();
    }

    fn stats(&self) -> AgentStats {
        self.stats
    }

    // fn stats(&self) -> AgentStats {
    //     AgentStats {
    //         total_steps: self.stats.total_steps,
    //         total_episodes: self.stats.total_episodes,
    //         avg_reward: self.stats.episode_rewards.iter().sum::<f32>()
    //             / self.stats.episode_rewards.len().max(1) as f32,
    //         avg_loss: self.stats.episode_losses.iter().sum::<f32>()
    //             / self.stats.episode_losses.len().max(1) as f32,
    //         exploration_rate: self.epsilon,
    //     }
    // }
}

impl<E, const S: usize, const A: usize, B: AutodiffBackend, M> NeuralAgent<E, S, A, B>
    for DQNAgent<E, S, A, B, M>
where
    E: Environment<S, A>,
    M: DQNModel<B>,
{
    type Model = M;

    fn model(&self) -> &Self::Model {
        &self.policy_net
    }

    fn model_mut(&mut self) -> &mut Self::Model {
        &mut self.policy_net
    }
}

impl<E, const S: usize, const A: usize, B: AutodiffBackend, M, O> TrainableAgent<E, S, A, B, O>
    for DQNAgent<E, S, A, B, M>
where
    E: Environment<S, A>,
    M: DQNModel<B>,
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

impl<E, const S: usize, const A: usize, B: AutodiffBackend, M: DQNModel<B>> DQNAgent<E, S, A, B, M>
where
    E: Environment<S, A>,
    E::StateType: StateTensorConvertible<S> + State,
    E::ActionType: ActionTensorConvertible<A> + Action,
{
    pub fn new(policy_net: M, target_net: M, epsilon: f64, device: B::Device) -> Self {
        Self {
            policy_net: Some(policy_net),
            target_net: Some(target_net),
            epsilon,
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
            device,
            stats: PhantomData,
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
        if random::<f64>() < eps_threshold {
            // Exploration
            Ok(E::ActionType::random())
        } else {
            // Exploitation - use policy network
            let policy_net = self
                .policy_net
                .as_ref()
                .ok_or(DQNAgentError::ModelNotInitialized)?;

            let state_tensor = state.to_tensor(&B::Device::default());
            let q_values = policy_net.forward(state_tensor.unsqueeze::<E::StateType::R1>());
            let max_index = q_values.argmax(-1).into_scalar() as usize;

            Ok(E::ActionType::from_index(max_index))
        }
    }

    /// Collect experience by interacting with the environment
    pub fn collect_experience<const CAP: usize>(
        &mut self,
        env: &mut E,
        memory: &mut ReplayMemory<E, S, A, CAP>,
        num_steps: usize,
    ) -> Result<Vec<f32>, DQNAgentError> {
        let mut rewards = Vec::new();

        for _ in 0..num_steps {
            // Get current state
            let snapshot = env.reset()?;
            let state = snapshot.state;

            // Select action using epsilon-greedy
            let action = self.select_action_with_exploration(&state, self.epsilon)?;

            // Take step in environment
            let next_snapshot = env.step(action.clone())?;

            // Store experience in replay memory
            let experience = Experience {
                state,
                action,
                reward: next_snapshot.reward,
                next_state: next_snapshot.state.clone(),
                done: next_snapshot.done,
            };

            memory.push(experience);
            rewards.push(next_snapshot.reward.into());

            if next_snapshot.done {
                break;
            }
        }

        Ok(rewards)
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
        memory: &mut ReplayMemory<E, S, A, CAP>,
        optimizer: &mut O,
        config: &DQNTrainingConfig,
        step_count: usize,
    ) -> Result<LearningStats, DQNAgentError> {
        // 1. Collect new experiences
        let collected_rewards = self.collect_experience(env, memory, config.steps_per_episode)?;

        // 2. Check if we have enough samples to train
        if memory.len() < config.batch_size {
            return Ok(LearningStats {
                loss: 0.0,
                avg_reward: collected_rewards.iter().sum::<f32>() / collected_rewards.len() as f32,
                epsilon: self.epsilon,
            });
        }

        // 3. Sample batch from replay memory
        let mut rng = rand::rng();
        let batch = memory.sample_batch(config.batch_size, &B::Device::default(), &mut rng)?;

        // 4. Perform training step
        let loss = self.learn(&batch)?;

        // 5. Update target network periodically
        if step_count % config.target_update_frequency == 0 {
            self.update_target_network(config.tau)?;
        }

        // 6. Decay epsilon
        self.decay_epsilon();

        Ok(LearningStats {
            loss,
            avg_reward: collected_rewards.iter().sum::<f32>() / collected_rewards.len() as f32,
            epsilon: self.epsilon,
        })
    }
}

// ---
impl<E, const S: usize, const A: usize, B: AutodiffBackend, M: DQNModel<B> + AutodiffModule<B>>
    DQNAgent<E, S, A, B, M>
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
        memory: &ReplayMemory<E, S, A, CAP>,
        optimizer: &mut (impl Optimizer<M, B> + Sized),
        config: &DQNTrainingConfig,
    ) -> M {
        unimplemented!()
    }

    pub fn valid(mut self) -> DQNAgent<E, S, A, B::InnerBackend, M::InnerModule>
    where
        <M as AutodiffModule<B>>::InnerModule: DQNModel<<B as AutodiffBackend>::InnerBackend>,
    {
        DQNAgent::<E, B::InnerBackend, M::InnerModule>::new(
            self.policy_net.take().unwrap().valid(),
            self.target_net.take().unwrap().valid(),
            self.epsilon,
            self.device,
        )
    }
}
