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
//! | [`base`] | [`Reward`], [`Observation`], [`State`], [`Action`], [`HostRow`], [`TensorConvertible`], [`UpdateFunction`] — the primitive trait vocabulary |
//! | [`action`] | [`DiscreteAction`], [`MultiDiscreteAction`], [`ContinuousAction`] — layered action-space extensions |
//! | [`state`] | [`MarkovState`], [`BeliefState`], [`HiddenState`], [`LatentState`], [`StateAggregation`], [`Observable`] — POMDP and latent-space extensions |
//! | [`environment`] | [`Environment`], [`Sensor`], [`Snapshot`], [`SnapshotBase`], [`EpisodeStatus`], [`EnvironmentError`] — the agent/environment protocol |
//! | [`reward`] | [`ScalarReward`] — the standard single-value reward concrete type |
//! | [`evaluation`] | [`BenchEnv`], [`BenchStep`], [`BenchError`] — object-safe environment interface for harnesses |
//! | [`fitness`] | [`BenchableAgent`], [`FitnessEvaluable`], [`Landscape`], [`Metric`], [`MetricsProvider`] — inference-only agent and fitness evaluation |
//! | [`objective`] | [`ObjectiveSense`] — the maximise/minimise direction primitive reconciled at one chokepoint |
//! | [`config`] | [`Validate`], [`ConfigError`] — the shared config-validation convention checked at construction |
//! | [`bounds`] | [`Bounds`] — an inclusive range valid by construction (invariant `lo <= hi`: rejects `lo > hi` and `NaN`) |
//! | [`probability`] | [`Probability`] — a `[0, 1]` rate valid by construction (rejects `NaN`, `Inf`, out-of-range) |
//! | [`rate`] | [`NonNegativeRate`] — a finite non-negative magnitude valid by construction (BLX-α, σ) |
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
//! [`HostRow`]: crate::base::HostRow
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
//! [`Sensor`]: crate::environment::Sensor
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
//! [`Bounds`]: crate::bounds::Bounds
//! [`Probability`]: crate::probability::Probability
//! [`NonNegativeRate`]: crate::rate::NonNegativeRate
//! [`AsciiRenderable`]: crate::render::AsciiRenderable
//! [`Renderer`]: crate::render::Renderer

/// Primitive trait vocabulary: [`Reward`], [`Observation`], [`State`],
/// [`Action`], [`HostRow`], [`TensorConvertible`], and [`UpdateFunction`].
///
/// All other modules in this crate depend on the types defined here.
///
/// [`Reward`]: crate::base::Reward
/// [`Observation`]: crate::base::Observation
/// [`State`]: crate::base::State
/// [`Action`]: crate::base::Action
/// [`HostRow`]: crate::base::HostRow
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

/// Validated closed-range primitive.
///
/// Provides [`Bounds`], an inclusive `[lo, hi]` range over `f32` that is valid
/// by construction (invariant `lo <= hi`: rejects `lo > hi` and `NaN`, permits a
/// one-sided infinite endpoint), plus its
/// [`BoundsError`]. Makes the `f32::clamp` panic and `x.max(lo).min(hi)`
/// silent-collapse hazards unrepresentable wherever a `Bounds` is held.
/// Complements the [`config`] convention rather than replacing it (ADR 0027).
///
/// [`Bounds`]: crate::bounds::Bounds
/// [`BoundsError`]: crate::bounds::BoundsError
pub mod bounds;

/// Agent/environment interaction protocol.
///
/// Defines [`Environment`], the env-side emission model [`Sensor`],
/// [`Snapshot`]/[`SnapshotBase`], [`EpisodeStatus`], and [`EnvironmentError`].
///
/// [`Environment`]: crate::environment::Environment
/// [`Sensor`]: crate::environment::Sensor
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

/// Validated unit-interval probability primitive.
///
/// Provides [`Probability`], a value in the closed interval `[0, 1]` valid by
/// construction (invariant `0.0 <= p <= 1.0`: rejects `NaN`, `Inf`, and
/// out-of-range values), plus its [`ProbabilityError`]. Makes the silent
/// all-false-mask degeneracy of a `NaN`/out-of-range Bernoulli rate
/// unrepresentable wherever a `Probability` is held. Complements the [`config`]
/// convention rather than replacing it (ADR 0031).
///
/// [`Probability`]: crate::probability::Probability
/// [`ProbabilityError`]: crate::probability::ProbabilityError
pub mod probability;

/// Validated non-negative rate primitive.
///
/// Provides [`NonNegativeRate`], a finite non-negative `f32` valid by
/// construction (invariant `is_finite() && r >= 0.0`), plus its
/// [`NonNegativeRateError`]. The unbounded companion to [`Probability`] for
/// magnitudes such as BLX-α's expansion factor or Gaussian mutation's σ, where
/// a `NaN`/`Inf` would otherwise poison the offspring tensor (ADR 0031).
///
/// [`NonNegativeRate`]: crate::rate::NonNegativeRate
/// [`NonNegativeRateError`]: crate::rate::NonNegativeRateError
/// [`Probability`]: crate::probability::Probability
pub mod rate;

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
