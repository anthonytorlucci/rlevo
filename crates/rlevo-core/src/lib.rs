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
//! | [`fitness`] | [`BenchableAgent`], [`Landscape`], [`Metric`], [`MetricsProvider`] — inference-only agent and fitness evaluation |
//! | [`objective`] | [`ObjectiveSense`] — the maximise/minimise direction primitive reconciled at one chokepoint |
//! | [`config`] | [`Validate`], [`ConfigError`] — the shared config-validation convention checked at construction |
//! | [`bounds`] | [`Bounds`] — an inclusive range valid by construction (invariant `lo <= hi`: rejects `lo > hi` and `NaN`) |
//! | [`probability`] | [`Probability`] — a `[0, 1]` rate valid by construction (rejects `NaN`, `Inf`, out-of-range) |
//! | [`rate`] | [`NonNegativeRate`] — a finite non-negative magnitude valid by construction (BLX-`$\alpha$`, `$\sigma$`) |
//! | [`render`] | [`AsciiRenderable`], styled/palette/payload sub-modules — optional debug and TUI visualization layer |
//! | [`agent`] | Reserved; empty while the unified agent trait hierarchy stabilizes |
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
//! [`BenchableAgent`]: crate::fitness::BenchableAgent
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

/// Primitive trait vocabulary: [`Observation`], [`Action`] and [`HostRow`],
/// plus `TensorConvertible` under the `burn` feature.
///
/// All other modules in this crate depend on the types defined here.
///
/// [`Observation`]: crate::base::Observation
/// [`Action`]: crate::base::Action
/// [`HostRow`]: crate::base::HostRow
pub mod base;

/// Layered action-space traits: [`DiscreteAction`], [`ContinuousAction`] and
/// [`BoundedAction`].
///
/// [`DiscreteAction`]: crate::action::DiscreteAction
/// [`ContinuousAction`]: crate::action::ContinuousAction
/// [`BoundedAction`]: crate::action::BoundedAction
pub mod action;

/// Reserved for a future unified agent trait hierarchy.
///
/// Empty while the unified agent trait hierarchy stabilizes. Concrete RL and
/// evolutionary agents currently live in `rlevo-reinforcement-learning` and
/// `rlevo-evolution` respectively.
pub mod agent;

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
/// Defines [`Environment`], [`Snapshot`], [`EpisodeStatus`],
/// and [`EnvironmentError`].
///
/// [`Environment`]: crate::environment::Environment
/// [`Snapshot`]: crate::environment::Snapshot
/// [`Snapshot`]: crate::environment::Snapshot
/// [`EpisodeStatus`]: crate::environment::EpisodeStatus
/// [`EnvironmentError`]: crate::environment::EnvironmentError
pub mod environment;

/// Episode-lifecycle guard.
///
/// Provides [`EpisodeGuard`], the state machine an [`Environment`] holds to
/// meet the post-terminal contract on `step`: a step taken after the episode
/// ended returns [`EnvironmentError::StepAfterEpisodeEnd`] and changes
/// nothing (ADR 0044).
///
/// [`EpisodeGuard`]: crate::episode::EpisodeGuard
/// [`Environment`]: crate::environment::Environment
/// [`EnvironmentError::StepAfterEpisodeEnd`]: crate::environment::EnvironmentError::StepAfterEpisodeEnd
pub mod episode;

/// Drive seam for work that paces itself.
///
/// [`GenerationProbe`] is what an evaluation harness drives when the work is
/// not an episodic agent/environment interaction — an evolutionary generation
/// loop, concretely. Episodic work is driven through [`Environment`] directly
/// (ADR 0076).
///
/// [`GenerationProbe`]: crate::evaluation::GenerationProbe
/// [`Environment`]: crate::environment::Environment
pub mod evaluation;

/// Inference-only agent and fitness-evaluation traits.
///
/// Provides [`BenchableAgent`], [`Landscape`], [`Metric`], and
/// [`MetricsProvider`].
///
/// [`BenchableAgent`]: crate::fitness::BenchableAgent
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
/// magnitudes such as BLX-`$\alpha$`'s expansion factor or Gaussian mutation's `$\sigma$`, where
/// a `NaN`/`Inf` would otherwise poison the offspring tensor (ADR 0031).
///
/// [`NonNegativeRate`]: crate::rate::NonNegativeRate
/// [`NonNegativeRateError`]: crate::rate::NonNegativeRateError
/// [`Probability`]: crate::probability::Probability
pub mod rate;

/// Optional rendering layer for debug output and TUI visualization.
///
/// Contains [`AsciiRenderable`] (via the [`ascii`] sub-module),
/// styled color primitives ([`styled`] / [`palette`]), and environment
/// payload types ([`payload`]).
///
/// [`AsciiRenderable`]: crate::render::AsciiRenderable
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

/// [`StateError`]: why a value is not a legal state of an environment.
///
/// [`StateError`]: crate::state::StateError
pub mod state;

/// Shared utility helpers used across `rlevo-core` modules.
pub mod util;
