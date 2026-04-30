//! Evolutionary Deep Reinforcement Learning built on the [Burn](https://github.com/tracel-ai/burn) framework.
//!
//! # Modules
//!
//! - [`core`] — foundational traits: `Environment`, `State`, `Action`, `Reward`, `TensorConvertible`
//! - [`envs`] — benchmark environments: classic control, gridworlds, `Box2D` physics, locomotion
//! - [`rl`] — deep RL algorithms: DQN, C51, QR-DQN, PPO, PPG, DDPG, TD3, SAC (and the replay buffer / experience / metrics modules they consume)
//! - [`evo`] — evolutionary algorithms: GA, ES, EP, DE, CGP with GPU kernels
//! - [`hybrid`] — combined evolutionary + RL strategies
//!
//! # Quick Start
//!
//! ```toml
//! [dependencies]
//! rlevo = "0.1"
//! ```
//!
//! ```rust
//! use rlevo::prelude::*;
//! ```
//!
//! For specific items use the sub-module paths directly:
//!
//! ```rust,no_run
//! use rlevo::core::environment::Environment;
//! use rlevo::envs::classic::cartpole::CartPole;
//! use rlevo::rl::algorithms::dqn::dqn_agent::DqnAgent;
//! ```

pub use rlevo_core as core;
pub use rlevo_environments as envs;
pub use rlevo_evolution as evo;
pub use rlevo_hybrid as hybrid;
pub use rlevo_reinforcement_learning as rl;

/// The most commonly used traits and types, importable with `use rlevo::prelude::*`.
///
/// # Contents
///
/// **Core base traits** (`rlevo::core::base`):
/// [`State`](core::base::State), [`Observation`](core::base::Observation),
/// [`Action`](core::base::Action), [`Reward`](core::base::Reward),
/// [`TensorConvertible`](core::base::TensorConvertible)
///
/// **Environment** (`rlevo::core::environment`):
/// [`Environment`](core::environment::Environment),
/// [`Snapshot`](core::environment::Snapshot),
/// [`SnapshotBase`](core::environment::SnapshotBase),
/// [`EpisodeStatus`](core::environment::EpisodeStatus),
/// [`EnvironmentError`](core::environment::EnvironmentError)
///
/// **Concrete reward** (`rlevo::core::reward`):
/// [`ScalarReward`](core::reward::ScalarReward)
///
/// **Action extensions** (`rlevo::core::action`):
/// [`DiscreteAction`](core::action::DiscreteAction),
/// [`MultiDiscreteAction`](core::action::MultiDiscreteAction),
/// [`ContinuousAction`](core::action::ContinuousAction),
/// [`BoundedAction`](core::action::BoundedAction)
///
/// **Error types**:
/// [`StateError`](core::state::StateError),
/// [`InvalidActionError`](core::action::InvalidActionError)
///
/// **Evolution** (`rlevo::evo`):
/// [`Strategy`](evo::strategy::Strategy),
/// [`FitnessFn`](evo::fitness::FitnessFn),
/// [`Population`](evo::population::Population)
pub mod prelude {
    // Base traits
    pub use rlevo_core::base::{Action, Observation, Reward, State, TensorConvertible};

    // Environment protocol
    pub use rlevo_core::environment::{
        Environment, EnvironmentError, EpisodeStatus, Snapshot, SnapshotBase,
    };

    // Reward
    pub use rlevo_core::reward::ScalarReward;

    // Action extensions
    pub use rlevo_core::action::{
        BoundedAction, ContinuousAction, DiscreteAction, InvalidActionError, MultiDiscreteAction,
    };

    // State and action errors
    pub use rlevo_core::state::StateError;

    // Evolution
    pub use rlevo_evolution::fitness::FitnessFn;
    pub use rlevo_evolution::population::Population;
    pub use rlevo_evolution::strategy::Strategy;
}
