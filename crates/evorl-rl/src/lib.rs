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
//! - [`algorithms::ppg`] — Phasic Policy Gradient (on-policy with an auxiliary
//!   phase): extends PPO with a periodic auxiliary phase that retrains the
//!   value function plus an auxiliary value head on the policy network,
//!   distilling the pre-aux-phase policy via KL. v1 is discrete-only.
//! - [`algorithms::ddpg`] — Deep Deterministic Policy Gradient (off-policy,
//!   continuous actions): deterministic actor + Q-critic with Polyak-averaged
//!   target networks and Gaussian exploration noise.
//! - [`algorithms::td3`] — Twin Delayed DDPG (off-policy, continuous actions):
//!   DDPG with twin critics (min-target), Gaussian target-policy smoothing,
//!   and delayed actor/Polyak updates per Fujimoto et al. 2018.
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

    pub mod ddpg {
        //! Deep Deterministic Policy Gradient (DDPG): off-policy actor-critic
        //! for continuous action spaces.
        //!
        //! Pairs a [`ddpg_model::DeterministicPolicy`] actor with a
        //! [`ddpg_model::ContinuousQ`] critic, each with a Polyak-averaged
        //! target copy. Explores via Gaussian noise on the actor output
        //! ([`exploration::GaussianNoise`]) and learns off a uniform replay
        //! buffer. CleanRL's `ddpg_continuous_action.py` is the reference
        //! implementation.

        pub mod ddpg_agent;
        pub mod ddpg_config;
        pub mod ddpg_model;
        pub mod exploration;
        pub mod train;
    }

    pub mod td3 {
        //! Twin Delayed DDPG (TD3): off-policy actor-critic for continuous
        //! action spaces with twin critics and delayed policy updates.
        //!
        //! Extends [`super::ddpg`] with Fujimoto et al. 2018's three deltas:
        //! a `min`-of-twin-critics bootstrap target, Gaussian target-policy
        //! smoothing, and delayed actor/Polyak updates every
        //! `policy_frequency`-th critic step. Reuses
        //! [`super::ddpg::exploration::GaussianNoise`] at action-selection
        //! time and the [`super::ddpg::ddpg_model::DeterministicPolicy`] /
        //! [`super::ddpg::ddpg_model::ContinuousQ`] traits unchanged.
        //! CleanRL's `td3_continuous_action.py` is the reference
        //! implementation.

        pub mod target_smoothing;
        pub mod td3_agent;
        pub mod td3_config;
        pub mod td3_model;
        pub mod train;
    }

    pub mod ppg {
        //! Phasic Policy Gradient (PPG): PPO policy phase + periodic
        //! auxiliary phase with distillation.
        //!
        //! v1 ships a discrete-only [`policies::PpgCategoricalPolicyHead`]
        //! and reuses PPO's [`super::ppo::rollout::RolloutBuffer`],
        //! [`super::ppo::losses`], and [`super::ppo::ppo_value::PpoValue`].
        //! CartPole parity with PPO is the v1 target; Procgen-scale gains
        //! require vectorised envs + CNN encoders (deferred).

        pub mod aux_buffer;
        pub mod losses;
        pub mod policies;
        pub mod ppg_agent;
        pub mod ppg_config;
        pub mod ppg_policy;
        pub mod train;
    }
}

pub mod utils;
