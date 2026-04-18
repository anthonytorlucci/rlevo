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
//! - [`algorithms::c51`] — Categorical DQN (distributional): config, model
//!   trait, categorical projection, agent, and training loop.
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

    pub mod c51 {
        //! Categorical DQN (C51) distributional algorithm.

        pub mod c51_agent;
        pub mod c51_config;
        pub mod c51_model;
        pub mod loss;
        pub mod projection;
        pub mod train;
    }
}

pub mod utils;
