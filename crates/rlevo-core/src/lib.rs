//! Core types and traits for evolutionary deep reinforcement learning.
//!
//! `rlevo-core` defines the shared vocabulary used across the entire `rlevo`
//! workspace. Every other crate — `rlevo-reinforcement-learning`,
//! `rlevo-evolution`, `rlevo-environments`, `rlevo-harness` — depends on
//! these primitives. No concrete algorithms or environments live here.
//!
//! # Module map
//!
//! | Module | What it provides |
//! |---|---|
//! | [`base`] | [`Observation`], [`Action`], [`HostRow`] — the primitive trait vocabulary; `TensorConvertible` with the `burn` feature |
//! | [`action`] | [`DiscreteAction`], [`ContinuousAction`], [`BoundedAction`] — layered action-space extensions |
//! | [`state`] | [`StateError`] — why a value is not a legal state |
//! | [`environment`] | [`Environment`], [`Snapshot`], [`EpisodeStatus`], [`EnvironmentError`] — the agent/environment protocol |
//! | [`episode`] | [`EpisodeGuard`] — the state machine behind `Environment::step`'s post-terminal contract, held by every implementor |
//! | [`reward`] | [`ScalarReward`] — the standard single-value reward concrete type |
//! | [`evaluation`] | [`GenerationProbe`] — drive seam for self-paced work (evolutionary generation loops) |
//! | [`fitness`] | [`Landscape`], [`Metric`] — fitness evaluation, and the metric data type an agent or a generation loop reports |
//! | [`objective`] | [`ObjectiveSense`] — the maximise/minimise direction primitive reconciled at one chokepoint |
//! | [`config`] | [`Validate`], [`ConfigError`] — the shared config-validation convention checked at construction |
//! | [`bounds`] | [`Bounds`] — an inclusive range valid by construction (invariant `lo <= hi`: rejects `lo > hi` and `NaN`) |
//! | [`probability`] | [`Probability`] — a `[0, 1]` rate valid by construction (rejects `NaN`, `Inf`, out-of-range) |
//! | [`rate`] | [`NonNegativeRate`] — a finite non-negative magnitude valid by construction (BLX-`$\alpha$`, `$\sigma$`) |
//! | [`render`] | [`AsciiRenderable`], styled/palette/payload sub-modules — optional debug and report visualization layer |
//! | [`agent`] | [`Agent`], [`Transition`] — the seam a driver steps a policy or a learner through |
//! | [`util`] | Shared utility helpers |
//!
//! # Features
//!
//! | Feature | Default | What it adds |
//! |---|---|---|
//! | `burn` | off | `base::TensorConvertible`, implemented for every [`HostRow`] type, which uploads the row as a Burn tensor. Without it the crate does not depend on Burn |
//!
//! # Const-generic rank `R`
//!
//! [`Observation`] and [`Action`], and the traits built on them, are parameterized
//! by a const generic `R` (rank) that denotes the *number of tensor axes*
//! (equivalent to `NumPy`'s `ndim` or Burn's `Tensor<R>`). This encodes
//! shape compatibility at compile time: a rank-1 observation and a rank-2
//! observation cannot be mixed up without a compile error.
//!
//! ```text
//! Environment<OR, AR>
//!   ├── ObservationType: Observation<OR>  (shape: [usize; OR])
//!   ├── ActionType     : Action<AR>       (shape: [usize; AR])
//!   └── reset / step   → Snapshot<ObservationType>
//! ```
//!
//! # Episode loop sketch
//!
//! The basic agent/environment interaction loop follows this pattern:
//!
//! ```text
//! env.reset() → Snapshot { observation, reward, status: Running }
//!   loop:
//!     agent selects action from observation
//!     env.step(action) → Snapshot { observation, reward, status }
//!     break when status.is_done()
//! ```
//!
//! [`EpisodeStatus::Terminated`] and [`EpisodeStatus::Truncated`] are kept
//! distinct so RL algorithms can bootstrap value correctly at truncation
//! boundaries.
//!
//! [`Observation`]: crate::base::Observation
//! [`Action`]: crate::base::Action
//! [`HostRow`]: crate::base::HostRow
//! [`DiscreteAction`]: crate::action::DiscreteAction
//! [`ContinuousAction`]: crate::action::ContinuousAction
//! [`BoundedAction`]: crate::action::BoundedAction
//! [`StateError`]: crate::state::StateError
//! [`Environment`]: crate::environment::Environment
//! [`Snapshot`]: crate::environment::Snapshot
//! [`Snapshot`]: crate::environment::Snapshot
//! [`EpisodeStatus`]: crate::environment::EpisodeStatus
//! [`EpisodeStatus::Terminated`]: crate::environment::EpisodeStatus::Terminated
//! [`EpisodeStatus::Truncated`]: crate::environment::EpisodeStatus::Truncated
//! [`EnvironmentError`]: crate::environment::EnvironmentError
//! [`EpisodeGuard`]: crate::episode::EpisodeGuard
//! [`ScalarReward`]: crate::reward::ScalarReward
//! [`GenerationProbe`]: crate::evaluation::GenerationProbe
//! [`Agent`]: crate::agent::Agent
//! [`Transition`]: crate::agent::Transition
//! [`Landscape`]: crate::fitness::Landscape
//! [`Metric`]: crate::fitness::Metric
//! [`ObjectiveSense`]: crate::objective::ObjectiveSense
//! [`Validate`]: crate::config::Validate
//! [`ConfigError`]: crate::config::ConfigError
//! [`Bounds`]: crate::bounds::Bounds
//! [`Probability`]: crate::probability::Probability
//! [`NonNegativeRate`]: crate::rate::NonNegativeRate
//! [`AsciiRenderable`]: crate::render::AsciiRenderable

pub mod action;
pub mod agent;
pub mod base;
pub mod bounds;
pub mod config;
pub mod environment;
pub mod episode;
pub mod evaluation;
pub mod fitness;
pub mod objective;
pub mod probability;
pub mod rate;
pub mod render;
pub mod reward;
pub mod state;
pub mod util;

/// The `rand` release this crate's signatures are written against,
/// re-exported so a caller needs no `rand` entry of its own.
///
/// [`Agent::act`](crate::agent::Agent::act) and its hooks take a
/// `&mut dyn Rng`, as does every evolutionary strategy's `init`, `ask` and
/// `tell`. That `Rng` is this release's trait, which a generator from
/// another `rand` release does not implement. The methods that draw, such as
/// `random_range`, are on [`RngExt`](rand::RngExt).
pub use rand;
