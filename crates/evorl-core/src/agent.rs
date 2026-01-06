use crate::environment::Environment;
use crate::memory::TrainingBatch;
use crate::metrics::AgentStats;
use burn::module::AutodiffModule;
use burn::optim::Optimizer;
use burn::tensor::backend::{AutodiffBackend, Backend};
use std::fmt::Debug;

// Define a precise `Agent` trait tied to an environment, with methods for action selection and learning.

pub trait Agent<E, const S: usize, const A: usize, B: Backend>
where
    E: Environment<S, A>,
{
    type Error: std::error::Error + Debug + Send + Sync + 'static;

    /// Choose an action given the current state
    fn act(&mut self, state: &E::StateType) -> Result<E::ActionType, Self::Error>;

    /// Learn from a batch of experiences
    fn learn(&mut self, batch: &TrainingBatch<B, S, A>) -> Result<f32, Self::Error>;

    /// Update exploration parameters (e.g., decay epsilon)
    fn update_exploration(&mut self, step: usize) -> Result<(), Self::Error>;

    /// Get current exploration rate (for logging/monitoring)
    fn exploration_rate(&self) -> f64;

    /// Save the agent's state to a file
    fn save(&self, path: &str) -> Result<(), Self::Error>;

    /// Load the agent's state from a file
    fn load(&mut self, path: &str) -> Result<(), Self::Error>;

    /// Reset agent state between episodes
    fn reset(&mut self);

    /// Collect statistics about the agent's performance
    fn stats(&self) -> AgentStats;
}

/// Trait for agents that use neural network policies
pub trait NeuralAgent<E, const S: usize, const A: usize, B: AutodiffBackend>:
    Agent<E, S, A, B>
where
    E: Environment<S, A>,
{
    type Model: AutodiffModule<B> + Debug;

    /// Get a reference to the underlying model
    fn model(&self) -> Option<&Self::Model>;

    /// Get a mutable reference to the underlying model
    fn model_mut(&mut self) -> Option<&mut Self::Model>;
}

/// Trait for agents that support training with optimizers
pub trait TrainableAgent<E, const S: usize, const A: usize, B: AutodiffBackend, O>:
    NeuralAgent<E, S, A, B>
where
    O: Optimizer<Self::Model, B>,
    E: Environment<S, A>,
{
    /// Perform a training step with the given optimizer
    fn train_step(
        &mut self,
        batch: &TrainingBatch<B, S, A>,
        optimizer: &mut O,
    ) -> Result<f32, Self::Error>;
}
