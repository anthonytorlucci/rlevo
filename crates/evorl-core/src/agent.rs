use crate::dynamics::History;
use burn::module::AutodiffModule;
use burn::optim::Optimizer;
use burn::tensor::backend::{AutodiffBackend, Backend};
use std::fmt::Debug;

// todo! distinguish between RL agents and EA agents

// pub trait Agent<E, const S: usize, const A: usize, B: Backend>
// where
//     E: Environment<S, A>,
// {
//     type Error: std::error::Error + Debug + Send + Sync + 'static;

//     /// Choose an action given the current state
//     fn act(
//         &mut self,
//         state: &E::StateType,
//         device: &B::Device,
//     ) -> Result<E::ActionType, Self::Error>;

//     /// Learn from a batch of experiences
//     fn learn(&mut self, batch: &TrainingBatch<S, A, B>) -> Result<f32, Self::Error>;

//     /// Update exploration parameters (e.g., decay epsilon)
//     fn update_exploration(&mut self, step: usize) -> Result<(), Self::Error>;

//     /// Get current exploration rate (for logging/monitoring)
//     fn exploration_rate(&self) -> f64;

//     /// Save the agent's state to a file
//     fn save(&self, path: &str) -> Result<(), Self::Error>;

//     /// Load the agent's state from a file
//     fn load(&mut self, path: &str) -> Result<(), Self::Error>;

//     /// Reset agent state between episodes
//     fn reset(&mut self);
//
//     fn trajectory(
//         &self,
//     ) -> History<D, AD, Self::ObservationType, Self::ActionType, Self::RewardType>;
// }

// /// Trait for agents that use neural network policies
// pub trait NeuralAgent<E, const S: usize, const A: usize, B: AutodiffBackend, O>:
//     Agent<E, S, A, B>
// where
//     E: Environment<S, A>,
//     O: Optimizer<Self::Model, B>,
// {
//     type Model: AutodiffModule<B> + Debug;

//     /// Get a reference to the underlying model
//     fn model(&self) -> Option<&Self::Model>;

//     /// Get a mutable reference to the underlying model
//     fn model_mut(&mut self) -> Option<&mut Self::Model>;

//     /// Perform a training step with the given optimizer
//     fn train_step(
//         &mut self,
//         batch: &TrainingBatch<S, A, B>,
//         optimizer: &mut O,
//     ) -> Result<f32, Self::Error>;
// }
