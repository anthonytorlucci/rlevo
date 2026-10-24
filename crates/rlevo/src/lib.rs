//! Deep Reinforcement Learning with Evolutionary Optimization built on the [Burn](https://github.com/tracel-ai/burn) framework.
//!
//! # Modules
//!
//! - [`core`] — foundational traits and types: `Environment`, `Observation`, `Action`, `Snapshot`, `ScalarReward`, `TensorConvertible`
//! - [`envs`] — benchmark environments: classic control, gridworlds, `Box2D` physics, locomotion
//! - [`rl`] — deep RL algorithms: DQN, C51, QR-DQN, PPO, PPG, DDPG, TD3, SAC (and the replay and metrics modules they consume)
//! - [`evo`] — evolutionary algorithms: GA, ES, CMA-ES, DE, EP, GP, swarm metaheuristics, EDAs, memetic, neuroevolution, co-evolution
//! - [`hybrid`] — combined evolutionary + RL strategies
//! - [`harness`] — the evaluation harness: `Suite`, `Evaluator`, metrics and reporters
//!
//! # Quick Start
//!
//! ```toml
//! [dependencies]
//! rlevo = "0.5"
//! ```
//!
//! ```rust
//! use rlevo::prelude::*;
//! ```
//!
//! The prelude carries `rand`'s [`Rng`](prelude::Rng), the generator
//! [`Agent::act`](prelude::Agent::act) and every strategy's `init`, `ask` and
//! `tell` take,
//! and [`RngExt`](prelude::RngExt), whose methods draw from it. The rest of
//! `rand` is [`core::rand`], the release those signatures are written
//! against, so a consumer needs no `rand` entry of its own.
//!
//! For specific items use the sub-module paths directly:
//!
//! ```rust,no_run
//! use rlevo::core::environment::Environment;
//! use rlevo::envs::classic::cartpole::CartPole;
//! use rlevo::rl::algorithms::dqn::dqn_agent::DqnAgent;
//! ```
//!
//! # Backends
//!
//! The library crates name no Burn backend. A Cargo feature of this crate
//! selects one, and Burn itself is re-exported as [`burn`], so a consumer
//! needs no `burn` entry of its own:
//!
//! - `flex` (default) — CPU, through Burn's Flex backend.
//! - `wgpu` — GPU, through Burn's wgpu backend (Metal, Vulkan, DX12, WebGPU).
//!
//! Each also turns on Burn's `autodiff`, which the RL agents train through.
//! The backend is a runtime [`Device`](burn::tensor::Device), and the feature
//! decides which of its constructors exist: `Device::flex()` or
//! `Device::wgpu(..)`. Build a network on the plain device; an RL agent
//! switches autodiff on itself.
//!
//! A network you write derives Burn's `Module` (and its configuration,
//! `Config`). Those derives expand to `burn::` paths, so keep
//! `use rlevo::burn;` in the module that uses them. Without it they fail to
//! resolve `burn`, and adding `burn` to your own manifest instead risks a
//! second Burn release in the build.
//!
//! ```rust
//! use rlevo::burn;
//! use burn::nn::{Linear, LinearConfig};
//! use burn::prelude::*;
//!
//! #[derive(Module, Debug)]
//! struct Critic {
//!     head: Linear,
//! }
//!
//! let device = Device::flex();
//! let critic = Critic { head: LinearConfig::new(4, 1).init(&device) };
//! let value: Tensor<2> = critic.head.forward(Tensor::zeros([1, 4], &device));
//! assert_eq!(value.dims(), [1, 1]);
//! ```

pub use rlevo_core as core;
pub use rlevo_environments as envs;
pub use rlevo_evolution as evo;
/// The evaluation harness, [`rlevo_harness`]: register an environment in
/// a [`Suite`](harness::suite::Suite) and run an agent over it with
/// [`Evaluator::run_suite`](harness::evaluator::Evaluator::run_suite).
///
/// Always present. It is the harness at its default features, which link no
/// terminal or report dependency; this crate's `viz-tui` and `viz-report`
/// features turn those on.
pub use rlevo_harness as harness;
pub use rlevo_hybrid as hybrid;
pub use rlevo_reinforcement_learning as rl;

/// The Burn release this version of `rlevo` is built against, re-exported so
/// a consumer names Burn's types without a `burn` entry of its own.
///
/// This crate's `flex` and `wgpu` features decide which
/// [`Device`](burn::tensor::Device) constructors exist. A module that derives
/// Burn's `Module` or `Config` keeps `use rlevo::burn;` in scope, because the
/// derives expand to `burn::` paths (see the [crate docs](crate#backends)).
pub use burn;

/// The most commonly used traits and types, importable with `use rlevo::prelude::*`.
///
/// # Contents
///
/// **Core base traits** (`rlevo::core::base`):
/// [`Observation`](core::base::Observation),
/// [`Action`](core::base::Action),
/// [`HostRow`](core::base::HostRow),
/// [`TensorConvertible`](core::base::TensorConvertible)
///
/// **Agent** (`rlevo::core::agent`):
/// [`Agent`](core::agent::Agent),
/// [`Transition`](core::agent::Transition)
///
/// **Environment** (`rlevo::core::environment`):
/// [`Environment`](core::environment::Environment),
/// [`Snapshot`](core::environment::Snapshot),
/// [`EpisodeStatus`](core::environment::EpisodeStatus),
/// [`EnvironmentError`](core::environment::EnvironmentError)
///
/// **Concrete reward** (`rlevo::core::reward`):
/// [`ScalarReward`](core::reward::ScalarReward)
///
/// **Action extensions** (`rlevo::core::action`):
/// [`DiscreteAction`](core::action::DiscreteAction),
/// [`ContinuousAction`](core::action::ContinuousAction),
/// [`BoundedAction`](core::action::BoundedAction)
///
/// **Error types**:
/// [`StateError`](core::state::StateError),
/// [`InvalidActionError`](core::action::InvalidActionError)
///
/// **Evolution** (`rlevo::evo`):
/// [`Strategy`](evo::strategy::Strategy)
///
/// **Randomness** (`rlevo::core::rand`):
/// [`Rng`](core::rand::Rng),
/// [`RngExt`](core::rand::RngExt)
pub mod prelude {
    // Base traits
    pub use rlevo_core::base::{Action, HostRow, Observation, TensorConvertible};

    // Agent seam
    pub use rlevo_core::agent::{Agent, Transition};

    // Environment protocol
    pub use rlevo_core::environment::{Environment, EnvironmentError, EpisodeStatus, Snapshot};

    // Reward
    pub use rlevo_core::reward::ScalarReward;

    // Action extensions
    pub use rlevo_core::action::{
        BoundedAction, ContinuousAction, DiscreteAction, InvalidActionError,
    };

    // State and action errors
    pub use rlevo_core::state::StateError;

    // Evolution
    pub use rlevo_evolution::strategy::Strategy;

    // Randomness: the generator every `act`, `init`, `ask` and `tell` takes,
    // and the trait whose methods draw from it
    pub use rlevo_core::rand::{Rng, RngExt};
}
