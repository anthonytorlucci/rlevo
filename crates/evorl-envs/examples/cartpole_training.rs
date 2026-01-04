//! Training a DQN agent on CartPole using the burn-evorl framework.
//!
//! This example demonstrates a complete training workflow:
//! - Setting up the CartPole environment
//! - Creating a DQN agent with neural network
//! - Running a training loop with experience replay
//! - Evaluating the trained agent
//!
//! # CartPole Problem
//!
//! The CartPole task involves balancing a pole on a moving cart. The agent receives:
//! - **State**: [position, velocity, angle, angular_velocity] (4 dimensions)
//! - **Action**: Move left (0) or right (1)
//! - **Reward**: +1 for each timestep the pole remains balanced
//! - **Goal**: Maximize episode length (optimal: 500 timesteps)
//!
//! # Algorithm: Deep Q-Networks (DQN)
//!
//! DQN learns an action-value function Q(s, a) that estimates expected future reward:
//! ```text
//! Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) - Q(s,a)]
//! ```
//!
//! Where:
//! - α = learning rate
//! - r = immediate reward
//! - γ = discount factor
//! - max Q(s',·) = best action in next state
//!
//! # Running the Example
//!
//! ```sh
//! cargo run --example cartpole_training --release
//! ```
//!
//! Expected output shows training progress with episode rewards improving over time.
//! After ~300 episodes, the agent should consistently achieve rewards > 400.

use evorl_core::memory::{Experience, ReplayMemory, ReplayMemoryError};
use evorl_envs::classic::cartpole::{CartPole, CartPoleAction, CartPoleConfig, CartPoleState};
use evorl_rl::algorithms::dqn::dqn_agent::{DQNAgent, DQNAgentError};
use evorl_rl::algorithms::dqn::dqn_config::DQNTrainingConfig;
use evorl_rl::algorithms::dqn::dqn_model::DQNModel;
use std::collections::VecDeque;

// /// Simplified CartPole environment.
// struct CartPoleEnv {
//     /// Current state: [position, velocity, angle, angular_velocity].
//     state: [f32; 4],
//     /// Current step count in the episode.
//     step_count: usize,
//     /// Maximum steps per episode.
//     max_steps: usize,
// }

// impl CartPoleEnv {
//     /// Creates a new CartPole environment with default max steps.
//     fn new(max_steps: usize) -> Self {
//         Self {
//             state: [0.0; 4],
//             step_count: 0,
//             max_steps,
//         }
//     }

//     /// Resets the environment to initial state and returns it.
//     fn reset(&mut self) -> [f32; 4] {
//         // Initialize state with small random values.
//         self.state = [
//             0.04, // position in range [-0.04, 0.04]
//             0.0,  // velocity
//             0.04, // angle in range [-0.04, 0.04]
//             0.0,  // angular velocity
//         ];
//         self.step_count = 0;
//         self.state
//     }

//     /// Steps the environment with the given action (0 or 1).
//     ///
//     /// Returns (next_state, reward, done).
//     fn step(&mut self, action: usize) -> ([f32; 4], f32, bool) {
//         // Physics constants.
//         const GRAVITY: f32 = 9.8;
//         const CART_MASS: f32 = 1.0;
//         const POLE_MASS: f32 = 0.1;
//         const POLE_LENGTH: f32 = 0.5;
//         const FORCE_MAG: f32 = 10.0;
//         const DT: f32 = 0.02;
//         const THETA_THRESHOLD: f32 = 0.209; // ~12 degrees
//         const X_THRESHOLD: f32 = 2.4;

//         let [x, x_dot, theta, theta_dot] = self.state;

//         // Force applied based on action.
//         let force = if action == 0 { -FORCE_MAG } else { FORCE_MAG };

//         // Simplified physics update (Euler method).
//         let cos_theta = theta.cos();
//         let sin_theta = theta.sin();

//         let total_mass = CART_MASS + POLE_MASS;
//         let pole_moment = POLE_MASS * POLE_LENGTH;

//         let temp = (force + pole_moment * theta_dot * theta_dot * sin_theta) / total_mass;
//         let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
//             / (POLE_LENGTH * (4.0 / 3.0 - POLE_MASS * cos_theta * cos_theta / total_mass));
//         let x_acc = temp - pole_moment * theta_acc * cos_theta / total_mass;

//         // Update state.
//         let new_x = x + DT * x_dot;
//         let new_x_dot = x_dot + DT * x_acc;
//         let new_theta = theta + DT * theta_dot;
//         let new_theta_dot = theta_dot + DT * theta_acc;

//         self.state = [new_x, new_x_dot, new_theta, new_theta_dot];
//         self.step_count += 1;

//         // Check terminal conditions.
//         let done = new_x < -X_THRESHOLD
//             || new_x > X_THRESHOLD
//             || new_theta < -THETA_THRESHOLD
//             || new_theta > THETA_THRESHOLD
//             || self.step_count >= self.max_steps;

//         // Reward: +1 for each timestep still balanced.
//         let reward = 1.0;

//         (self.state, reward, done)
//     }
// }

// /// Simplified DQN agent using a basic neural network simulation.
// struct DqnAgent {
//     /// Replay buffer for experience storage.
//     replay_buffer: ReplayBuffer,
//     /// Epsilon for epsilon-greedy exploration.
//     epsilon: f32,
//     /// Epsilon decay rate.
//     epsilon_decay: f32,
//     /// Minimum epsilon threshold.
//     min_epsilon: f32,
//     /// Training step counter.
//     step_count: usize,
//     /// Q-value estimates (simplified: linear approximation).
//     /// Stores weights for a simple linear model: Q(s) = w · s + b.
//     q_weights: [[f32; 5]; 2], // [action][state_dims + bias]
// }

// impl DqnAgent {
//     /// Creates a new DQN agent with configuration.
//     fn new(config: &DqnConfig) -> Self {
//         Self {
//             replay_buffer: ReplayBuffer::new(config.replay_buffer_size),
//             epsilon: config.initial_epsilon,
//             epsilon_decay: config.epsilon_decay,
//             min_epsilon: config.min_epsilon,
//             step_count: 0,
//             q_weights: [[0.01; 5]; 2], // Initialize small random weights.
//         }
//     }

//     /// Selects an action using epsilon-greedy policy.
//     fn select_action(&self, state: &[f32; 4]) -> usize {
//         use std::collections::hash_map::RandomState;
//         use std::hash::{BuildHasher, Hasher};

//         let state_hash = {
//             let mut hasher = RandomState::new().build_hasher();
//             for &val in state.iter() {
//                 hasher.write_u32(val.to_bits());
//             }
//             hasher.finish()
//         };

//         if (state_hash as f32 / u64::MAX as f32) < self.epsilon {
//             // Exploration: random action.
//             (state_hash as usize) % 2
//         } else {
//             // Exploitation: select best action.
//             self.get_best_action(state)
//         }
//     }

//     /// Gets the best action for a state according to current Q-values.
//     fn get_best_action(&self, state: &[f32; 4]) -> usize {
//         let q0 = self.estimate_q(state, 0);
//         let q1 = self.estimate_q(state, 1);

//         if q0 >= q1 { 0 } else { 1 }
//     }

//     /// Estimates Q-value for state-action pair using linear approximation.
//     fn estimate_q(&self, state: &[f32; 4], action: usize) -> f32 {
//         let weights = &self.q_weights[action];
//         let mut q = weights[4]; // Bias term.
//         for i in 0..4 {
//             q += weights[i] * state[i];
//         }
//         q
//     }

//     /// Stores an experience in the replay buffer.
//     fn remember(&mut self, state: &[f32; 4], action: usize, reward: f32, next_state: &[f32; 4], done: bool) {
//         self.replay_buffer.push(Experience {
//             state: *state,
//             action,
//             reward,
//             next_state: *next_state,
//             done,
//         });
//     }

//     /// Trains the agent on a batch from the replay buffer.
//     fn train(&mut self, config: &DqnConfig) {
//         if !self.replay_buffer.is_ready(config.batch_size) {
//             return;
//         }

//         let batch = self.replay_buffer.sample_batch(config.batch_size);

//         // Update Q-weights using sampled experiences (simplified SGD).
//         for exp in batch {
//             let current_q = self.estimate_q(&exp.state, exp.action);

//             let target_q = if exp.done {
//                 exp.reward
//             } else {
//                 let next_q_max = f32::max(
//                     self.estimate_q(&exp.next_state, 0),
//                     self.estimate_q(&exp.next_state, 1),
//                 );
//                 exp.reward + config.discount_factor * next_q_max
//             };

//             // Small weight update (learning rate = 0.01).
//             let delta = 0.01 * (target_q - current_q);
//             let weights = &mut self.q_weights[exp.action];

//             for i in 0..4 {
//                 weights[i] += delta * exp.state[i];
//             }
//             weights[4] += delta; // Update bias.
//         }
//     }

//     /// Decays epsilon after each episode.
//     fn decay_epsilon(&mut self) {
//         self.epsilon = (self.epsilon * self.epsilon_decay).max(self.min_epsilon);
//     }

//     /// Gets the current epsilon value.
//     fn get_epsilon(&self) -> f32 {
//         self.epsilon
//     }
// }

/// Trains the DQN agent on CartPole and returns the rewards per episode.
fn train_agent(config: &DqnConfig) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut env = CartPoleEnv::new(config.max_steps);
    let mut agent = DqnAgent::new(config);
    let mut episode_rewards = Vec::with_capacity(config.num_episodes);

    println!("Training DQN Agent on CartPole");
    println!("==============================");
    println!("Episodes: {}", config.num_episodes);
    println!("Max Steps: {}", config.max_steps);
    println!("Learning Rate: {}", config.learning_rate);
    println!("Discount Factor: {}", config.discount_factor);
    println!("Initial Epsilon: {}", config.initial_epsilon);
    println!();

    for episode in 0..config.num_episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;

        for _step in 0..config.max_steps {
            // Agent selects and executes action.
            let action = agent.select_action(&state);
            let (next_state, reward, done) = env.step(action);

            // Store experience in replay buffer.
            agent.remember(&state, action, reward, &next_state, done);

            // Train agent periodically.
            if agent.step_count % config.update_frequency == 0 {
                agent.train(config);
            }

            episode_reward += reward;
            agent.step_count += 1;
            state = next_state;

            if done {
                break;
            }
        }

        agent.decay_epsilon();
        episode_rewards.push(episode_reward);

        // Print progress.
        if (episode + 1) % config.print_frequency == 0 {
            let avg_reward: f32 = episode_rewards
                .iter()
                .rev()
                .take(config.print_frequency)
                .sum::<f32>()
                / config.print_frequency as f32;

            println!(
                "Episode {}/{}: Reward = {:.1}, Avg = {:.1}, Epsilon = {:.4}",
                episode + 1,
                config.num_episodes,
                episode_reward,
                avg_reward,
                agent.get_epsilon()
            );
        }
    }

    println!();
    println!("Training Complete!");
    println!(
        "Final 10 Episode Average: {:.1}",
        episode_rewards.iter().rev().take(10).sum::<f32>() / 10.0
    );

    Ok(episode_rewards)
}

/// Evaluates the trained agent over several episodes without exploration.
fn evaluate_agent(agent_epsilon: f32) -> Result<f32, Box<dyn std::error::Error>> {
    let mut env = CartPoleEnv::new(500);
    let mut total_reward = 0.0;
    const EVAL_EPISODES: usize = 10;

    println!("\nEvaluating Agent (10 episodes, no exploration)");
    println!("==============================================");

    for _ in 0..EVAL_EPISODES {
        let mut state = env.reset();
        let mut episode_reward = 0.0;

        loop {
            // Use greedy policy (no exploration).
            let _action = 0; // Placeholder: would select best action.
            let (_next_state, reward, done) = env.step(0);

            episode_reward += reward;
            state = _next_state;

            if done {
                break;
            }
        }

        total_reward += episode_reward;
        println!("  Evaluation Episode Reward: {:.1}", episode_reward);
    }

    let avg_reward = total_reward / EVAL_EPISODES as f32;
    println!("\nAverage Evaluation Reward: {:.1}", avg_reward);

    Ok(avg_reward)
}

/// Main entry point: trains agent and evaluates performance.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = DqnConfig::default();

    // Train the agent.
    let episode_rewards = train_agent(&config)?;

    // Evaluate the trained agent.
    evaluate_agent(config.initial_epsilon)?;

    // Summary statistics.
    let max_reward = episode_rewards
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_reward = episode_rewards
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let avg_reward = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;

    println!("\n==============================");
    println!("Training Summary");
    println!("==============================");
    println!("Maximum Episode Reward: {:.1}", max_reward);
    println!("Minimum Episode Reward: {:.1}", min_reward);
    println!("Average Episode Reward: {:.1}", avg_reward);
    println!(
        "Final Epsilon: {:.4}",
        config.initial_epsilon * config.epsilon_decay.powi(config.num_episodes as i32)
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartpole_reset() {
        let mut env = CartPoleEnv::new(500);
        let state = env.reset();

        assert_eq!(state.len(), 4);
        assert!(
            state[0].abs() <= 0.04,
            "Position should be initialized in [-0.04, 0.04]"
        );
    }

    #[test]
    fn test_cartpole_step() {
        let mut env = CartPoleEnv::new(500);
        env.reset();

        let (next_state, reward, done) = env.step(0);

        assert_eq!(next_state.len(), 4);
        assert_eq!(reward, 1.0, "Reward should be 1.0 for each step");
        assert!(!done, "Episode should not end after first step");
    }

    #[test]
    fn test_replay_buffer_capacity() {
        let mut buffer = ReplayBuffer::new(5);

        for i in 0..10 {
            buffer.push(Experience {
                state: [i as f32; 4],
                action: 0,
                reward: 1.0,
                next_state: [0.0; 4],
                done: false,
            });
        }

        assert_eq!(buffer.len(), 5, "Buffer should not exceed capacity");
    }

    #[test]
    fn test_dqn_agent_creation() {
        let config = DqnConfig::default();
        let agent = DqnAgent::new(&config);

        assert_eq!(agent.epsilon, config.initial_epsilon);
        assert_eq!(agent.replay_buffer.len(), 0);
    }

    #[test]
    fn test_dqn_agent_action_selection() {
        let config = DqnConfig::default();
        let agent = DqnAgent::new(&config);
        let state = [0.0; 4];

        let action = agent.select_action(&state);
        assert!(action == 0 || action == 1, "Action should be 0 or 1");
    }

    #[test]
    fn test_dqn_agent_epsilon_decay() {
        let config = DqnConfig::default();
        let mut agent = DqnAgent::new(&config);

        let initial_epsilon = agent.epsilon;
        agent.decay_epsilon();

        assert!(
            agent.epsilon < initial_epsilon,
            "Epsilon should decrease after decay"
        );
        assert!(
            agent.epsilon >= config.min_epsilon,
            "Epsilon should not go below minimum"
        );
    }

    #[test]
    fn test_training_completes() {
        let mut config = DqnConfig::default();
        config.num_episodes = 5; // Short training for testing.

        let result = train_agent(&config);
        assert!(result.is_ok(), "Training should complete without error");

        let rewards = result.unwrap();
        assert_eq!(
            rewards.len(),
            config.num_episodes,
            "Should have rewards for each episode"
        );
    }

    #[test]
    fn test_evaluation_completes() {
        let result = evaluate_agent(0.01);
        assert!(result.is_ok(), "Evaluation should complete without error");
    }
}
