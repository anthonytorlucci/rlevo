//! Core types and traits for evolutionary deep reinforcement learning.
//!
//! `rlevo-core` defines the shared vocabulary used across the entire `rlevo`
//! workspace. Every other crate — `rlevo-reinforcement-learning`,
//! `rlevo-evolution`, `rlevo-environments`, `rlevo-benchmarks` — depends on
//! these primitives. No concrete algorithms or environments live here.
//!
//! # Module map
//!
//! | Module | What it provides |
//! |---|---|
//! | [`base`] | [`Reward`], [`Observation`], [`State`], [`Action`], [`TensorConvertible`], [`UpdateFunction`] — the primitive trait vocabulary |
//! | [`action`] | [`DiscreteAction`], [`MultiDiscreteAction`], [`ContinuousAction`] — layered action-space extensions |
//! | [`state`] | [`MarkovState`], [`BeliefState`], [`HiddenState`], [`LatentState`], [`StateAggregation`], [`Observable`] — POMDP and latent-space extensions |
//! | [`environment`] | [`Environment`], [`Snapshot`], [`SnapshotBase`], [`EpisodeStatus`], [`EnvironmentError`] — the agent/environment protocol |
//! | [`reward`] | [`ScalarReward`] — the standard single-value reward concrete type |
//! | [`evaluation`] | [`BenchEnv`], [`BenchStep`], [`BenchError`] — object-safe environment interface for harnesses |
//! | [`fitness`] | [`BenchableAgent`], [`FitnessEvaluable`], [`Landscape`], [`Metric`], [`MetricsProvider`] — inference-only agent and fitness evaluation |
//! | [`objective`] | [`ObjectiveSense`] — the maximise/minimise direction primitive reconciled at one chokepoint |
//! | [`config`] | [`Validate`], [`ConfigError`] — the shared config-validation convention checked at construction |
//! | [`render`] | [`AsciiRenderable`], [`Renderer`](crate::render::Renderer), styled/palette/payload sub-modules — optional debug and TUI visualization layer |
//! | [`agent`] | Reserved; empty in v0.1.x while the unified agent trait hierarchy stabilizes |
//! | [`util`] | Shared utility helpers |
//!
//! # Const-generic `RANK`
//!
//! Several traits — [`Observation`], [`State`], [`Action`] — are parameterized
//! by a const generic `R` (rank) that denotes the *number of tensor axes*
//! (equivalent to NumPy's `ndim` or Burn's `Tensor<B, R>`). This encodes
//! shape compatibility at compile time: a rank-1 observation and a rank-2
//! observation cannot be mixed up without a compile error.
//!
//! ```text
//! Environment<R, SR, AR>
//!   ├── StateType     : State<SR>        (shape: [usize; SR])
//!   ├── ObservationType: Observation<R>  (shape: [usize; R])
//!   ├── ActionType    : Action<AR>       (shape: [usize; AR])
//!   └── SnapshotType  : Snapshot<R, ...>
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
//! [`Reward`]: crate::base::Reward
//! [`Observation`]: crate::base::Observation
//! [`State`]: crate::base::State
//! [`Action`]: crate::base::Action
//! [`TensorConvertible`]: crate::base::TensorConvertible
//! [`UpdateFunction`]: crate::base::UpdateFunction
//! [`DiscreteAction`]: crate::action::DiscreteAction
//! [`MultiDiscreteAction`]: crate::action::MultiDiscreteAction
//! [`ContinuousAction`]: crate::action::ContinuousAction
//! [`MarkovState`]: crate::state::MarkovState
//! [`BeliefState`]: crate::state::BeliefState
//! [`HiddenState`]: crate::state::HiddenState
//! [`LatentState`]: crate::state::LatentState
//! [`StateAggregation`]: crate::state::StateAggregation
//! [`Observable`]: crate::state::Observable
//! [`Environment`]: crate::environment::Environment
//! [`Snapshot`]: crate::environment::Snapshot
//! [`SnapshotBase`]: crate::environment::SnapshotBase
//! [`EpisodeStatus`]: crate::environment::EpisodeStatus
//! [`EpisodeStatus::Terminated`]: crate::environment::EpisodeStatus::Terminated
//! [`EpisodeStatus::Truncated`]: crate::environment::EpisodeStatus::Truncated
//! [`EnvironmentError`]: crate::environment::EnvironmentError
//! [`ScalarReward`]: crate::reward::ScalarReward
//! [`BenchEnv`]: crate::evaluation::BenchEnv
//! [`BenchStep`]: crate::evaluation::BenchStep
//! [`BenchError`]: crate::evaluation::BenchError
//! [`BenchableAgent`]: crate::fitness::BenchableAgent
//! [`FitnessEvaluable`]: crate::fitness::FitnessEvaluable
//! [`Landscape`]: crate::fitness::Landscape
//! [`Metric`]: crate::fitness::Metric
//! [`MetricsProvider`]: crate::fitness::MetricsProvider
//! [`ObjectiveSense`]: crate::objective::ObjectiveSense
//! [`Validate`]: crate::config::Validate
//! [`ConfigError`]: crate::config::ConfigError
//! [`AsciiRenderable`]: crate::render::AsciiRenderable
//! [`Renderer`]: crate::render::Renderer

/// Primitive trait vocabulary: [`Reward`], [`Observation`], [`State`],
/// [`Action`], [`TensorConvertible`], and [`UpdateFunction`].
///
/// All other modules in this crate depend on the types defined here.
///
/// [`Reward`]: crate::base::Reward
/// [`Observation`]: crate::base::Observation
/// [`State`]: crate::base::State
/// [`Action`]: crate::base::Action
/// [`TensorConvertible`]: crate::base::TensorConvertible
/// [`UpdateFunction`]: crate::base::UpdateFunction
pub mod base;

/// Layered action-space traits: [`DiscreteAction`], [`MultiDiscreteAction`],
/// and [`ContinuousAction`].
///
/// [`DiscreteAction`]: crate::action::DiscreteAction
/// [`MultiDiscreteAction`]: crate::action::MultiDiscreteAction
/// [`ContinuousAction`]: crate::action::ContinuousAction
pub mod action;

/// Reserved for a future unified agent trait hierarchy.
///
/// Empty in v0.3.x. Concrete RL and evolutionary agents currently live in
/// `rlevo-reinforcement-learning` and `rlevo-evolution` respectively.
pub mod agent;

/// Shared configuration-validation convention.
///
/// Provides [`Validate`], the trait every `*Config` implements to check its
/// invariants at construction, plus the structured [`ConfigError`] /
/// [`ConstraintKind`] it returns and ergonomic check helpers. Construction that
/// consumes a caller-supplied config returns `Result<_, ConfigError>` rather
/// than panicking (ADR 0026).
///
/// [`Validate`]: crate::config::Validate
/// [`ConfigError`]: crate::config::ConfigError
/// [`ConstraintKind`]: crate::config::ConstraintKind
pub mod config;

/// Agent/environment interaction protocol.
///
/// Defines [`Environment`], [`Snapshot`]/[`SnapshotBase`],
/// [`EpisodeStatus`], and [`EnvironmentError`].
///
/// [`Environment`]: crate::environment::Environment
/// [`Snapshot`]: crate::environment::Snapshot
/// [`SnapshotBase`]: crate::environment::SnapshotBase
/// [`EpisodeStatus`]: crate::environment::EpisodeStatus
/// [`EnvironmentError`]: crate::environment::EnvironmentError
pub mod environment;

/// Object-safe environment interface for benchmarking harnesses.
///
/// [`BenchEnv`] strips the const-generic dimensions from [`Environment`] so
/// harnesses do not need to be generic over them. Adapters live in
/// `rlevo-environments` behind the `bench` feature.
///
/// [`BenchEnv`]: crate::evaluation::BenchEnv
/// [`Environment`]: crate::environment::Environment
pub mod evaluation;

/// Inference-only agent and fitness-evaluation traits.
///
/// Provides [`BenchableAgent`], [`FitnessEvaluable`], [`Landscape`],
/// [`Metric`], and [`MetricsProvider`].
///
/// [`BenchableAgent`]: crate::fitness::BenchableAgent
/// [`FitnessEvaluable`]: crate::fitness::FitnessEvaluable
/// [`Landscape`]: crate::fitness::Landscape
/// [`Metric`]: crate::fitness::Metric
/// [`MetricsProvider`]: crate::fitness::MetricsProvider
pub mod fitness;

/// Objective direction primitive.
///
/// Provides [`ObjectiveSense`], the typed maximise/minimise direction that
/// reconciles the library's maximise-native engine with cost objectives at a
/// single chokepoint.
///
/// [`ObjectiveSense`]: crate::objective::ObjectiveSense
pub mod objective;

/// Optional rendering layer for debug output and TUI visualization.
///
/// Contains [`AsciiRenderable`], [`Renderer`] (via the [`ascii`] sub-module),
/// styled color primitives ([`styled`] / [`palette`]), and environment
/// payload types ([`payload`]).
///
/// [`AsciiRenderable`]: crate::render::AsciiRenderable
/// [`Renderer`]: crate::render::Renderer
/// [`ascii`]: crate::render::ascii
/// [`styled`]: crate::render::styled
/// [`palette`]: crate::render::palette
/// [`payload`]: crate::render::payload
pub mod render;

/// Concrete reward types.
///
/// Provides [`ScalarReward`], the standard single-`f32` reward used by most
/// environments.
///
/// [`ScalarReward`]: crate::reward::ScalarReward
pub mod reward;

/// Advanced state abstractions for POMDPs and latent representations.
///
/// Extends [`base::State`] with [`MarkovState`], [`BeliefState`],
/// [`HiddenState`], [`LatentState`], [`StateAggregation`], and [`Observable`]
/// (the modality-changing state→observation projection for `OR != SR`).
///
/// [`base::State`]: crate::base::State
/// [`MarkovState`]: crate::state::MarkovState
/// [`BeliefState`]: crate::state::BeliefState
/// [`HiddenState`]: crate::state::HiddenState
/// [`LatentState`]: crate::state::LatentState
/// [`StateAggregation`]: crate::state::StateAggregation
/// [`Observable`]: crate::state::Observable
pub mod state;

/// Shared utility helpers used across `rlevo-core` modules.
pub mod util;
