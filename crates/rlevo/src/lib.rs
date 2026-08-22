//! Deep Reinforcement Learning with Evolutionary Optimization built on the [Burn](https://github.com/tracel-ai/burn) framework.
//!
//! # Modules
//!
//! - [`core`] — foundational traits: `Environment`, `State`, `Action`, `Reward`, `TensorConvertible`
//! - [`envs`] — benchmark environments: classic control, gridworlds, `Box2D` physics, locomotion
//! - [`rl`] — deep RL algorithms: DQN, C51, QR-DQN, PPO, PPG, DDPG, TD3, SAC (and the replay buffer / experience / metrics modules they consume)
//! - [`evo`] — evolutionary algorithms: GA, ES, EP, DE, CGP with GPU kernels
//! - [`hybrid`] — combined evolutionary + RL strategies
//! - [`benchmarks`] — the evaluation harness: `Evaluator`, `Suite`, reporters, and
//!   (features `viz-tui` / `viz-report`) the live TUI and static-HTML report
//!
//! # Quick Start
//!
//! ```toml
//! [dependencies]
//! rlevo = "0.3"
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

/// The evaluation harness (`rlevo-benchmarks`).
///
/// Reachable from the umbrella so `cargo add rlevo` is enough to run a suite —
/// a second, separately-versioned dependency on an internal crate is not part
/// of the advertised API (ADR 0080). [`benchmarks::fixtures`] holds preset suites
/// over the built-in environments; everything under [`envs`] can be fed to
/// [`benchmarks::evaluator::Evaluator`] directly.
///
/// The `viz-tui` and `viz-report` features on **this** crate forward into it,
/// and now actually reach an external consumer.
///
/// # Not named `bench`
///
/// `bench` is the built-in `#[bench]` attribute macro, so `rlevo::bench` is
/// ambiguous to rustdoc: `[`bench`]` resolves to neither and every intra-doc
/// link needs a `mod@` disambiguator — here **and in every downstream crate**
/// that links to it. Measured: two `ambiguous link` errors from `cargo doc -p
/// rlevo` under that name, zero under this one. The full crate name also
/// carries no mapping to learn, and `harness` was rejected because
/// [`evo::coevolution::harness`] and `EvolutionaryHarness` already use that
/// word for something else (ADR 0075).
pub use rlevo_benchmarks as benchmarks;
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
/// [`HostRow`](core::base::HostRow),
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
    pub use rlevo_core::base::{Action, HostRow, Observation, Reward, State, TensorConvertible};

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
