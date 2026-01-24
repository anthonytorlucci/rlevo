//! Training example for the 10-armed bandit problem.
//!
//! This example demonstrates three classical bandit algorithms solving the
//! multi-armed bandit problem, a fundamental challenge in reinforcement learning
//! that explores the exploration-exploitation trade-off.
//!
//! # The Problem
//!
//! You are faced repeatedly with a choice among k=10 different actions (arms).
//! After each choice, you receive a numerical reward from a stationary probability
//! distribution that depends on the action you selected. Your objective is to
//! maximize the expected total reward over time.
//!
//! Each arm's true reward is sampled from N(q*(a), 1) where q*(a) ~ N(0, 1).
//! The agent must learn which arm has the highest expected reward through
//! trial and error, balancing exploration (trying different arms to learn their
//! values) with exploitation (choosing the best known arm).
//!
//! # Algorithms
//!
//! ## 1. Epsilon-Greedy (Default)
//!
//! The simplest approach: with probability ε (epsilon), select a random action
//! (explore), otherwise select the action with the highest estimated value (exploit).
//!
//! - **Pros**: Simple, effective, well-understood
//! - **Cons**: Explores uniformly without considering uncertainty
//! - **Hyperparameter**: ε = 0.1 (10% exploration rate)
//! - **Expected Performance**: ~80% optimal action rate after 1000 steps
//!
//! ## 2. Upper Confidence Bound (UCB)
//!
//! Selects actions based on their potential to be optimal, using the formula:
//! ```text
//! UCB(a) = Q(a) + c × √(ln(t) / N(a))
//! ```
//!
//! The confidence term naturally decreases as an action is selected more often,
//! providing systematic exploration of uncertain actions.
//!
//! - **Pros**: Principled exploration, no randomness, better performance
//! - **Cons**: More complex, requires tuning of c parameter
//! - **Hyperparameter**: c = 2.0 (exploration parameter)
//! - **Expected Performance**: ~91% optimal action rate after 1000 steps
//!
//! ## 3. Thompson Sampling
//!
//! A Bayesian approach that maintains a probability distribution over each arm's
//! expected reward. At each step, it samples from these distributions and selects
//! the arm with the highest sample.
//!
//! Uses Beta(α, β) distributions where positive rewards increase α (successes)
//! and negative rewards increase β (failures).
//!
//! - **Pros**: Naturally balances exploration/exploitation, performs well empirically
//! - **Cons**: Requires probability distributions, more computational overhead
//! - **Hyperparameter**: None (uses uniform prior Beta(1, 1))
//! - **Expected Performance**: ~84% optimal action rate after 1000 steps
//!
//! # Usage
//!
//! Run with default epsilon-greedy:
//! ```bash
//! cargo run --example ten_armed_bandit_training
//! ```
//!
//! Run with UCB (best performance):
//! ```bash
//! cargo run --example ten_armed_bandit_training --features ucb
//! ```
//!
//! Run with Thompson Sampling:
//! ```bash
//! cargo run --example ten_armed_bandit_training --features thompson
//! ```
//!
//! # Output
//!
//! The example tracks and displays:
//! - Average reward per step (higher is better)
//! - Percentage of optimal actions taken (higher is better)
//! - Learned Q-values for each arm
//! - Progress updates every 100 steps
//!
//! # Reference
//!
//! Based on Chapter 2 of "Reinforcement Learning: An Introduction" (2nd Edition)
//! by Sutton and Barto (2018), pages 25-36.

use evorl_core::action::DiscreteAction;
// use evorl_core::environment::{Environment, Snapshot};
use evorl_envs::classic::ten_armed_bandit::{TenArmedBandit, TenArmedBanditAction};
use rand::rngs::StdRng;
use rand::SeedableRng;

use std::fmt::{Display, Formatter};

/// Number of arms in the bandit problem.
const NUM_ARMS: usize = 10;

/// Total number of training steps.
const TOTAL_STEPS: usize = 1000;

/// Hyperparameters for epsilon-greedy algorithm.
#[cfg(not(any(feature = "ucb", feature = "thompson")))]
mod config {
    /// Exploration probability (0.0 = pure exploitation, 1.0 = pure exploration).
    pub const EPSILON: f32 = 0.1;
}

/// Hyperparameters for UCB algorithm.
#[cfg(feature = "ucb")]
mod config {
    /// Exploration parameter controlling the confidence interval width.
    /// Higher values encourage more exploration.
    pub const C: f32 = 2.0;
}

/// Training statistics tracker.
#[derive(Debug, Clone)]
struct BanditMetrics {
    /// Total cumulative reward across all steps.
    total_reward: f32,
    /// Number of times the optimal action was selected.
    optimal_action_count: usize,
    /// Index of the true optimal arm (for evaluation purposes).
    optimal_arm: usize,
    /// Total number of steps taken.
    steps: usize,
}

impl BanditMetrics {
    /// Creates a new statistics tracker.
    fn new(optimal_arm: usize) -> Self {
        Self {
            total_reward: 0.0,
            optimal_action_count: 0,
            optimal_arm,
            steps: 0,
        }
    }

    /// Updates statistics with a new step.
    fn update(&mut self, reward: f32, action: usize) {
        self.total_reward += reward;
        if action == self.optimal_arm {
            self.optimal_action_count += 1;
        }
        self.steps += 1;
    }

    /// Returns the average reward per step.
    fn average_reward(&self) -> f32 {
        if self.steps == 0 {
            0.0
        } else {
            self.total_reward / self.steps as f32
        }
    }

    /// Returns the percentage of optimal actions taken.
    fn optimal_action_percentage(&self) -> f32 {
        if self.steps == 0 {
            0.0
        } else {
            (self.optimal_action_count as f32 / self.steps as f32) * 100.0
        }
    }
}

impl Display for BanditMetrics {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Steps: {}, Avg Reward: {:.3}, Optimal Action: {:.1}% ({}/{})",
            self.steps,
            self.average_reward(),
            self.optimal_action_percentage(),
            self.optimal_action_count,
            self.steps
        )
    }
}

// ============================================================================
// Epsilon-Greedy Agent
// ============================================================================

#[cfg(not(any(feature = "ucb", feature = "thompson")))]
mod agent {
    use super::*;

    /// Epsilon-greedy bandit agent.
    ///
    /// This agent maintains action-value estimates Q(a) for each arm and uses
    /// an ε-greedy policy: with probability ε it explores randomly, otherwise
    /// it exploits by choosing the arm with the highest estimated value.
    ///
    /// The action values are updated using incremental sample averaging:
    /// Q(a) ← Q(a) + α[R - Q(a)]
    /// where α = 1/n is the step size for arm a.
    pub struct EpsilonGreedyAgent {
        /// Action-value estimates Q(a) for each arm.
        q_values: [f32; NUM_ARMS],
        /// Number of times each arm has been selected.
        action_counts: [usize; NUM_ARMS],
        /// Random number generator for exploration.
        rng: StdRng,
        /// Exploration probability.
        epsilon: f32,
    }

    impl EpsilonGreedyAgent {
        /// Creates a new epsilon-greedy agent.
        pub fn new(epsilon: f32) -> Self {
            Self {
                q_values: [0.0; NUM_ARMS],
                action_counts: [0; NUM_ARMS],
                rng: StdRng::seed_from_u64(123),
                epsilon,
            }
        }

        /// Selects an action using ε-greedy policy.
        ///
        /// With probability ε, selects a random action (exploration).
        /// With probability 1-ε, selects the action with highest Q-value (exploitation).
        pub fn select_action(&mut self) -> usize {
            use rand::Rng;

            if self.rng.random::<f32>() < self.epsilon {
                // Explore: select random action
                self.rng.random_range(0..NUM_ARMS)
            } else {
                // Exploit: select action with highest Q-value
                self.q_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            }
        }

        /// Updates the action-value estimate based on received reward.
        ///
        /// Uses incremental update rule:
        /// Q(a) ← Q(a) + (1/n)[R - Q(a)]
        pub fn update(&mut self, action: usize, reward: f32) {
            self.action_counts[action] += 1;
            let n = self.action_counts[action] as f32;
            let step_size = 1.0 / n;

            // Incremental update: Q(a) ← Q(a) + α[R - Q(a)]
            self.q_values[action] += step_size * (reward - self.q_values[action]);
        }

        /// Returns the current Q-values for display.
        pub fn q_values(&self) -> &[f32; NUM_ARMS] {
            &self.q_values
        }
    }

    impl Display for EpsilonGreedyAgent {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "ε-Greedy(ε={:.2})", self.epsilon)
        }
    }
}

// ============================================================================
// Upper Confidence Bound (UCB) Agent
// ============================================================================

#[cfg(feature = "ucb")]
mod agent {
    use super::*;

    /// Upper Confidence Bound (UCB) bandit agent.
    ///
    /// UCB balances exploration and exploitation by selecting the action with
    /// the highest upper confidence bound:
    ///
    /// UCB(a) = Q(a) + c × √(ln(t) / N(a))
    ///
    /// where:
    /// - Q(a) is the estimated action value
    /// - c is the exploration parameter
    /// - t is the total number of steps
    /// - N(a) is the number of times action a was selected
    ///
    /// The confidence term decreases as an action is selected more often,
    /// naturally balancing exploration of uncertain actions with exploitation
    /// of high-value actions.
    pub struct UCBAgent {
        /// Action-value estimates Q(a) for each arm.
        q_values: [f32; NUM_ARMS],
        /// Number of times each arm has been selected.
        action_counts: [usize; NUM_ARMS],
        /// Total number of steps taken.
        total_steps: usize,
        /// Exploration parameter c.
        c: f32,
    }

    impl UCBAgent {
        /// Creates a new UCB agent.
        pub fn new(c: f32) -> Self {
            Self {
                q_values: [0.0; NUM_ARMS],
                action_counts: [0; NUM_ARMS],
                total_steps: 0,
                c,
            }
        }

        /// Selects an action using UCB policy.
        ///
        /// Computes UCB(a) = Q(a) + c × √(ln(t) / N(a)) for each action
        /// and selects the one with the highest value. Actions that have
        /// never been tried receive infinite confidence (are tried first).
        pub fn select_action(&mut self) -> usize {
            self.total_steps += 1;

            // Select action with highest UCB value
            self.action_counts
                .iter()
                .enumerate()
                .map(|(idx, &count)| {
                    if count == 0 {
                        // Unvisited actions have infinite UCB (try them first)
                        (idx, f32::INFINITY)
                    } else {
                        let q_value = self.q_values[idx];
                        let exploration_bonus =
                            self.c * ((self.total_steps as f32).ln() / count as f32).sqrt();
                        let ucb_value = q_value + exploration_bonus;
                        (idx, ucb_value)
                    }
                })
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        }

        /// Updates the action-value estimate based on received reward.
        pub fn update(&mut self, action: usize, reward: f32) {
            self.action_counts[action] += 1;
            let n = self.action_counts[action] as f32;
            let step_size = 1.0 / n;

            // Incremental update: Q(a) ← Q(a) + α[R - Q(a)]
            self.q_values[action] += step_size * (reward - self.q_values[action]);
        }

        /// Returns the current Q-values for display.
        pub fn q_values(&self) -> &[f32; NUM_ARMS] {
            &self.q_values
        }
    }

    impl Display for UCBAgent {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "UCB(c={:.2})", self.c)
        }
    }
}

// ============================================================================
// Thompson Sampling Agent
// ============================================================================

#[cfg(feature = "thompson")]
mod agent {
    use super::*;
    use rand_distr::{Beta, Distribution};

    /// Thompson Sampling bandit agent.
    ///
    /// This Bayesian approach maintains a Beta distribution Beta(α, β) for
    /// each arm, representing the posterior belief about that arm's reward
    /// probability. At each step:
    ///
    /// 1. Sample from each arm's Beta distribution
    /// 2. Select the arm with the highest sample
    /// 3. Update the selected arm's distribution based on the reward
    ///
    /// For continuous rewards in [-∞, +∞], we use a simple heuristic:
    /// - Rewards > 0 increment α (successes)
    /// - Rewards < 0 increment β (failures)
    /// - The magnitude influences the update size
    ///
    /// This naturally balances exploration (uncertainty → wide distributions)
    /// and exploitation (high success rate → distributions peaked near 1).
    pub struct ThompsonSamplingAgent {
        /// Alpha parameters (successes) for Beta distributions.
        alphas: [f32; NUM_ARMS],
        /// Beta parameters (failures) for Beta distributions.
        betas: [f32; NUM_ARMS],
        /// Random number generator for sampling.
        rng: StdRng,
    }

    impl ThompsonSamplingAgent {
        /// Creates a new Thompson Sampling agent.
        ///
        /// Initializes each arm's distribution as Beta(1, 1) (uniform prior).
        pub fn new() -> Self {
            Self {
                alphas: [1.0; NUM_ARMS],
                betas: [1.0; NUM_ARMS],
                rng: StdRng::seed_from_u64(456),
            }
        }

        /// Selects an action by Thompson sampling.
        ///
        /// Samples a value from each arm's Beta(α, β) distribution and
        /// selects the arm with the highest sampled value.
        pub fn select_action(&mut self) -> usize {
            let mut samples = [0.0; NUM_ARMS];

            for i in 0..NUM_ARMS {
                let beta_dist = Beta::new(self.alphas[i], self.betas[i]).unwrap();
                samples[i] = beta_dist.sample(&mut self.rng);
            }

            samples
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        }

        /// Updates the Beta distribution based on received reward.
        ///
        /// For continuous rewards, we use a heuristic mapping:
        /// - Positive rewards increase α (successes)
        /// - Negative rewards increase β (failures)
        /// - The magnitude determines the update strength
        pub fn update(&mut self, action: usize, reward: f32) {
            // Map continuous reward to Beta update
            // Scale the reward to create meaningful updates
            let scaled_reward = reward.abs().min(10.0) / 10.0;

            if reward > 0.0 {
                // Positive reward → increase alpha (success)
                self.alphas[action] += scaled_reward;
            } else {
                // Negative reward → increase beta (failure)
                self.betas[action] += scaled_reward;
            }
        }

        /// Returns the estimated Q-values (mean of Beta distributions) for display.
        pub fn q_values(&self) -> [f32; NUM_ARMS] {
            let mut q_values = [0.0; NUM_ARMS];
            for i in 0..NUM_ARMS {
                // Mean of Beta(α, β) = α / (α + β)
                q_values[i] = self.alphas[i] / (self.alphas[i] + self.betas[i]);
            }
            q_values
        }
    }

    impl Display for ThompsonSamplingAgent {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "Thompson Sampling")
        }
    }
}

// ============================================================================
// Main Training Loop
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("10-Armed Bandit Training Example");
    println!("=================================\n");

    // Initialize environment
    // let mut env = TenArmedBandit::new(false);
    // env.reset()?;

    // // Initialize agent based on feature flags
    // #[cfg(not(any(feature = "ucb", feature = "thompson")))]
    // let mut agent = agent::EpsilonGreedyAgent::new(config::EPSILON);

    // #[cfg(feature = "ucb")]
    // let mut agent = agent::UCBAgent::new(config::C);

    // #[cfg(feature = "thompson")]
    // let mut agent = agent::ThompsonSamplingAgent::new();

    // println!("Algorithm: {}", agent);
    // println!("Total steps: {}\n", TOTAL_STEPS);

    // // Determine the optimal arm (for evaluation only - agent doesn't see this)
    // let optimal_arm = determine_optimal_arm(&mut env);
    // println!("True optimal arm: {} (for evaluation)\n", optimal_arm);

    // // Initialize statistics tracker
    // let mut stats = BanditMetrics::new(optimal_arm);

    // // Training loop
    // println!("Training...\n");

    // for step in 0..TOTAL_STEPS {
    //     // Agent selects an action
    //     let action_idx = agent.select_action();
    //     let action = TenArmedBanditAction::from_index(action_idx);

    //     // Execute action in environment
    //     let snapshot = env.step(action)?;
    //     let reward = *snapshot.reward();

    //     // Update agent with received reward
    //     agent.update(action_idx, reward);

    //     // Update statistics
    //     stats.update(reward, action_idx);

    //     // Print progress every 100 steps
    //     if (step + 1) % 100 == 0 {
    //         println!("Step {:4}: {}", step + 1, stats);
    //     }
    // }

    // // Final results
    // println!("\n{}", "=".repeat(70));
    // println!("Training Complete!");
    // println!("{}", "=".repeat(70));
    // println!("\nFinal Statistics:");
    // println!("  {}", stats);
    // println!("\nLearned Q-values:");
    // print_q_values(&agent);

    Ok(())
}

// /// Determines the optimal arm by sampling each arm many times.
// ///
// /// This is only used for evaluation purposes - the agent does not have
// /// access to this information.
// fn determine_optimal_arm(env: &mut TenArmedBandit) -> usize {
//     const SAMPLES_PER_ARM: usize = 100;
//     let mut arm_rewards = [0.0; NUM_ARMS];

//     for arm_idx in 0..NUM_ARMS {
//         let mut total_reward = 0.0;

//         for _ in 0..SAMPLES_PER_ARM {
//             let action = TenArmedBanditAction::from_index(arm_idx);
//             let snapshot = env.step(action).unwrap();
//             total_reward += *snapshot.reward();
//         }

//         arm_rewards[arm_idx] = total_reward / SAMPLES_PER_ARM as f32;
//     }

//     arm_rewards
//         .iter()
//         .enumerate()
//         .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
//         .map(|(idx, _)| idx)
//         .unwrap()
// }

// /// Prints the learned Q-values in a formatted table.
// fn print_q_values(agent: &Agent) {
//     let q_values = agent.q_values();

//     println!("  Arm | Estimated Value");
//     println!("  {}", "-".repeat(23));

//     for (arm, &value) in q_values.iter().enumerate() {
//         println!("  {:3} | {:15.3}", arm, value);
//     }
// }

// Type alias for the agent based on features
#[cfg(not(any(feature = "ucb", feature = "thompson")))]
pub type Agent = agent::EpsilonGreedyAgent;

#[cfg(feature = "ucb")]
pub type Agent = agent::UCBAgent;

#[cfg(feature = "thompson")]
pub type Agent = agent::ThompsonSamplingAgent;
