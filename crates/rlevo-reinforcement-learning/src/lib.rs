//! Deep reinforcement learning algorithms for `rlevo`.
//!
//! Provides implementations of deep RL algorithms that operate over the
//! `rlevo-core` environment and agent abstractions. All neural-network
//! operations are performed through the
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
//! - [`algorithms::sac`] — Soft Actor-Critic (off-policy, continuous actions):
//!   stochastic squashed-Gaussian actor + twin critics with entropy-augmented
//!   Bellman backup and auto-tuned temperature α per Haarnoja et al. 2018.
//! - [`replay`] — The replay-strategy seam (ADR 0050): [`replay::Transition`],
//!   the [`replay::ReplayStrategy`] trait, and [`replay::UniformReplay`], the
//!   FIFO uniform buffer every off-policy agent above draws its batches from.
//! - [`target`] — The target-network update rule: [`target::TargetUpdate`]
//!   (a cadence plus a [`target::PolyakTau`] coefficient) expresses one
//!   mechanism, in which a hard weight copy is the degenerate case `$\tau = 1.0$`
//!   rather than a separate variant.
//! - [`utils`] — Shared helpers: Bellman target computation, Polyak averaging.

/// The upper bound on any replay-buffer capacity in this crate: `2^32`.
///
/// # Why a ceiling exists at all
///
/// The `SumTree` behind [`replay::PrioritizedReplay`] sizes its storage as
/// `capacity.next_power_of_two() * 2`, and the workspace root `Cargo.toml`
/// declares no `[profile]` section, so release builds inherit
/// `overflow-checks = false` and **both** of those operations wrap. Measured on
/// `aarch64` macOS with `rustc -O`:
///
/// | `capacity` | `leaf_base` | `nodes.len()` |
/// |---|---|---|
/// | `2^62 + 1` | `2^63` | `0` (the `* 2` wrapped) |
/// | `2^63` | `2^63` | `0` (the `* 2` wrapped) |
/// | `2^63 + 1` | `0` | `0` (`next_power_of_two` wrapped) |
/// | `usize::MAX` | `0` | `0` (`next_power_of_two` wrapped) |
///
/// In debug those same capacities panic on overflow; in release the constructor
/// *succeeds* and hands back a priority index with an empty node array, which
/// then indexes out of bounds — or, worse, is read as a coherent distribution.
/// A debug/release divergence in a constructor is not something a caller can
/// reason about, so the capacity is rejected before it can reach the
/// arithmetic.
///
/// # Why `2^32` specifically
///
/// The bound has to be *below* the wrap and *above* anything a researcher would
/// legitimately ask for, and `2^32` clears both by a wide margin:
///
/// - **It cannot wrap.** `2^32.next_power_of_two() * 2 == 2^33`, a factor of
///   `2^30` below the smallest capacity in the table above. Every intermediate
///   stays exact on a 64-bit `usize`.
/// - **It is far past the published state of the art.** 4.29e9 transitions is
///   three orders of magnitude above DQN's 1e6 (Mnih et al. 2015), R2D2's ~4e6
///   (Kapturowski et al. 2019), and Agent57's 1e7 (Badia et al. 2020). A
///   capacity above this is a misconfiguration — a unit slip, a
///   `usize::MAX` sentinel, a deserialized garbage field — not an experiment.
/// - **It is unreachable in memory anyway.** The `SumTree` alone would want
///   `2^33 * 8 B = 64 GiB` of `f64` nodes before a single transition is stored,
///   so the allocation fails long before the arithmetic would.
///
/// A tighter bound would be arbitrary; a looser one would have to argue about
/// the wrap. This one has to argue about neither.
pub const MAX_BUFFER_CAPACITY: usize = 1 << 32;

pub mod algorithms {
    //! Reinforcement learning algorithm implementations.

    pub(crate) mod shared;

    /// Shared test scaffolding for the truncation-vs-termination bootstrap
    /// guard in the six off-policy `train` loops.
    #[cfg(test)]
    pub(crate) mod bootstrap_mask;

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
        //! buffer. `CleanRL`'s `ddpg_continuous_action.py` is the reference
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
        //! `CleanRL`'s `td3_continuous_action.py` is the reference
        //! implementation.

        pub mod target_smoothing;
        pub mod td3_agent;
        pub mod td3_config;
        pub mod td3_model;
        pub mod train;
    }

    pub mod sac {
        //! Soft Actor-Critic (SAC): off-policy, stochastic-actor,
        //! maximum-entropy algorithm for continuous-action spaces.
        //!
        //! Pairs a squashed-Gaussian
        //! [`sac_policy::SquashedGaussianPolicyHead`] actor with two
        //! [`sac_model::ContinuousQ`] critics (each with a Polyak-averaged
        //! target), a scalar [`sac_alpha::LogAlpha`] learnable temperature,
        //! and a uniform replay buffer. The Bellman target includes the
        //! entropy term `$-\alpha \cdot \log \pi(a'|s')$`; the actor is trained via
        //! reparameterization, and α is auto-tuned toward the heuristic
        //! target entropy `-|A|` by default. `CleanRL`'s
        //! `sac_continuous_action.py` is the reference implementation.

        pub mod sac_agent;
        pub mod sac_alpha;
        pub mod sac_config;
        pub mod sac_model;
        pub mod sac_policy;
        pub mod train;
    }

    pub mod ppg {
        //! Phasic Policy Gradient (PPG): PPO policy phase + periodic
        //! auxiliary phase with distillation.
        //!
        //! v1 ships a discrete-only [`policies::PpgCategoricalPolicyHead`]
        //! and reuses PPO's [`super::ppo::rollout::RolloutBuffer`],
        //! [`super::ppo::losses`], and [`super::ppo::ppo_value::PpoValue`].
        //! `CartPole` parity with PPO is the v1 target; Procgen-scale gains
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

pub mod experience;
pub mod metrics;
pub mod replay;
pub mod target;
pub mod utils;
