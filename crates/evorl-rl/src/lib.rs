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
//! - [`algorithms::qrdqn`] — Quantile Regression DQN (distributional):
//!   config, model trait, quantile Huber loss, agent, and training loop.
//! - [`algorithms::ppo`] — Proximal Policy Optimization (on-policy
//!   policy-gradient): config, policy/value traits, tanh-Gaussian + categorical
//!   built-in heads, rollout buffer with GAE, and training loop.
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

    pub mod qrdqn {
        //! Quantile Regression DQN (QR-DQN) distributional algorithm.

        pub mod qrdqn_agent;
        pub mod qrdqn_config;
        pub mod qrdqn_model;
        pub mod quantile_loss;
        pub mod train;
    }

    pub mod ppo {
        //! Proximal Policy Optimization (PPO) on-policy algorithm.

        pub mod losses;
        pub mod policies;
        pub mod ppo_agent;
        pub mod ppo_config;
        pub mod ppo_policy;
        pub mod ppo_value;
        pub mod rollout;
        pub mod train;
    }
}

pub mod utils;
