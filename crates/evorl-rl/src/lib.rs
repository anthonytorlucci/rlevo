//! Deep reinforcement learning algorithms for `burn-evorl`.
//!
//! Provides implementations of deep RL algorithms (DQN and future additions
//! such as PPO) that operate over the `evorl-core` environment and agent
//! abstractions. All neural-network operations are performed through the
//! [Burn](https://github.com/tracel-ai/burn) tensor framework.
//!
//! # Structure
//!
//! - [`algorithms::dqn`] — Deep Q-Network: config, model trait, and agent.
//! - [`utils`] — Shared helpers (e.g., Bellman target computation).

pub mod algorithms {
    //! Reinforcement learning algorithm implementations.

    pub mod dqn {
        //! Deep Q-Network algorithm.

        pub mod dqn_agent;
        pub mod dqn_config;
        pub mod dqn_model;
        pub mod exploration;
        pub mod train;
    }
}

pub mod utils;
