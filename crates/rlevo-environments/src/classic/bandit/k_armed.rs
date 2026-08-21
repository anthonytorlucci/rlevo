//! K-armed bandit environment — Sutton & Barto, *Reinforcement Learning*, §2.
//!
//! Stateless k-armed bandit generic over the number of arms `K`. On each step
//! the agent selects an arm `$a \in \{0, \ldots, K-1\}$` and receives a reward sampled
//! from `N(q*(a), 1)` where the true means `q*(a)` are themselves drawn from
//! `N(0, 1)` at construction (and re-drawn from the same seed on
//! [`Environment::reset`]).
//!
//! The canonical 10-armed instance is exposed as the type alias
//! [`TenArmedBandit`](crate::classic::TenArmedBandit) for the classic
//! Sutton & Barto §2 testbed.
//!
//! # Example
//!
//! ```rust
//! use rlevo_core::environment::{ConstructableEnv, Environment, Snapshot};
//! use rlevo_environments::classic::{KArmedBandit, KArmedBanditAction};
//!
//! let mut env: KArmedBandit<10> =
//!     <KArmedBandit<10> as ConstructableEnv>::new(false);
//! let _ = <KArmedBandit<10> as Environment<1, 1, 1>>::reset(&mut env)
//!     .expect("reset succeeds");
//! let action = KArmedBanditAction::<10>::new(3).expect("arm index in range");
//! let snap = <KArmedBandit<10> as Environment<1, 1, 1>>::step(&mut env, action)
//!     .expect("valid action");
//! assert!(!snap.is_done());
//! ```

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{
    Action, HostRow, Observation, Reward, State, TensorConversionError, TensorConvertible,
};
use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::environment::{
    ConstructableEnv, Environment, EnvironmentError, Sensor, Snapshot, SnapshotBase,
};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use crate::episode::EpisodeGuard;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// K-armed bandit state.
///
/// Bandit problems are stateless (the optimal action is independent of
/// history), so this struct carries no fields. It exists to satisfy the
/// [`State`] trait contract expected by [`Environment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KArmedBanditState;

/// Observation for the k-armed bandit.
///
/// Empty for the same reason as [`KArmedBanditState`]. The [`Observation`]
/// impl reports `shape() = [1]` so tensor-based policies can still convert
/// through [`TensorConvertible`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct KArmedBanditObservation;

impl Observation<1> for KArmedBanditObservation {
    fn shape() -> [usize; 1] {
        [1]
    }
}

impl State<1> for KArmedBanditState {
    fn shape() -> [usize; 1] {
        [1]
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn numel(&self) -> usize {
        1
    }
}

impl Display for KArmedBanditState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "KArmedBanditState")
    }
}

impl HostRow<1> for KArmedBanditState {
    /// Row shape of the stateless bandit state: `[1]`.
    fn row_shape() -> [usize; 1] {
        [1]
    }

    /// Encodes the stateless bandit state as a single `0.0` element.
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.push(0.0_f32);
    }
}

impl<B: Backend> TensorConvertible<1, B> for KArmedBanditState {
    /// Accepts any rank-1 tensor of shape `[1]`; contents are ignored because
    /// the state carries no data.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape is not `[1]`.
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [1] {
            return Err(TensorConversionError {
                message: format!("expected shape [1], got {dims:?}"),
            });
        }
        Ok(Self)
    }
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// Action for the k-armed bandit family — the choice of which arm to pull.
///
/// Valid arm indices are `0..K`. Use [`KArmedBanditAction::new`] for fallible
/// construction from untrusted input, or
/// [`KArmedBanditAction::from_index`](DiscreteAction::from_index) when the
/// caller has already validated the index. Shared across [`KArmedBandit`],
/// [`super::non_stationary::NonStationaryBandit`], and
/// [`super::adversarial::AdversarialBandit`]; the contextual variant uses the
/// same action with a different observation type.
///
/// # Traits implemented
///
/// - [`Action<1>`]: validity check + shape reporting (`[K]`).
/// - [`DiscreteAction<1>`]: `ACTION_COUNT = K`, plus `from_index` /
///   `to_index` / `random` / `enumerate` (the last three via trait defaults).
/// - [`TensorConvertible<1, B>`]: one-hot encoding of length `K` for
///   neural-network integration.
/// - [`Display`]: renders as `"KArmedBanditAction<K>(arm=N)"`.
///
/// # Examples
///
/// ```rust
/// use rlevo_core::action::DiscreteAction;
/// use rlevo_environments::classic::KArmedBanditAction;
///
/// let a = KArmedBanditAction::<10>::new(5).expect("5 is in range");
/// assert_eq!(a.arm(), 5);
/// assert_eq!(a.to_index(), 5);
///
/// let all = KArmedBanditAction::<10>::enumerate();
/// assert_eq!(all.len(), KArmedBanditAction::<10>::ACTION_COUNT);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KArmedBanditAction<const K: usize> {
    /// The index of the selected arm (`0..K`).
    selected_arm: usize,
}

impl<const K: usize> KArmedBanditAction<K> {
    /// Fallible constructor: returns [`EnvironmentError::InvalidAction`] when
    /// `arm >= K`.
    ///
    /// Prefer this over [`DiscreteAction::from_index`] for any index that
    /// came from an external source (configuration, RPC, policy output
    /// without a saturating mask). `from_index` panics on out-of-range input
    /// by the [`DiscreteAction`] trait contract.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::InvalidAction`] if `arm >= K`.
    pub fn new(arm: usize) -> Result<Self, EnvironmentError> {
        if arm < K {
            Ok(Self { selected_arm: arm })
        } else {
            Err(EnvironmentError::InvalidAction(format!(
                "arm index {arm} out of range [0, {K})"
            )))
        }
    }

    /// The index of the arm this action selects.
    #[must_use]
    pub fn arm(&self) -> usize {
        self.selected_arm
    }

    /// Forges an action whose arm index is out of range (`selected_arm == K`).
    ///
    /// Test-only, and shared with the sibling bandit modules: every public
    /// constructor rejects an out-of-range index, so this is the only way to
    /// obtain the malformed action needed to prove that
    /// [`Environment::step`] consults the episode guard **before** it validates
    /// the action (ADR 0044 §5).
    #[cfg(test)]
    pub(super) fn out_of_range_for_tests() -> Self {
        Self { selected_arm: K }
    }
}

impl<const K: usize> Display for KArmedBanditAction<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "KArmedBanditAction<{K}>(arm={})", self.selected_arm)
    }
}

impl<const K: usize> Action<1> for KArmedBanditAction<K> {
    fn shape() -> [usize; 1] {
        [K]
    }

    fn is_valid(&self) -> bool {
        self.selected_arm < K
    }
}

impl<const K: usize> DiscreteAction<1> for KArmedBanditAction<K> {
    const ACTION_COUNT: usize = K;

    /// Constructs from a validated index. Panics on out-of-range input, per
    /// the [`DiscreteAction`] trait contract. Use [`KArmedBanditAction::new`]
    /// for a fallible alternative.
    fn from_index(index: usize) -> Self {
        assert!(
            index < K,
            "KArmedBanditAction index {index} out of range [0, {K})",
        );
        Self {
            selected_arm: index,
        }
    }

    fn to_index(&self) -> usize {
        self.selected_arm
    }
}

impl<const K: usize> HostRow<1> for KArmedBanditAction<K> {
    /// Row shape of the one-hot arm encoding: `[K]`.
    fn row_shape() -> [usize; 1] {
        [K]
    }

    /// One-hot encoding of the selected arm, length `K`.
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        let mut one_hot = [0.0_f32; K];
        one_hot[self.selected_arm] = 1.0;
        buf.extend_from_slice(&one_hot);
    }
}

impl<const K: usize, B: Backend> TensorConvertible<1, B> for KArmedBanditAction<K> {
    /// Reconstructs an action from a one-hot tensor by argmax.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape is not `[K]` or
    /// the argmax falls outside the valid arm range.
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [K] {
            return Err(TensorConversionError {
                message: format!("expected shape [{K}], got {dims:?}"),
            });
        }
        let data = tensor.into_data();
        let values: Vec<f32> = data.to_vec().map_err(|e| TensorConversionError {
            message: format!("failed to extract tensor data: {e:?}"),
        })?;
        let (argmax, _) = values.iter().enumerate().fold(
            (0_usize, f32::NEG_INFINITY),
            |(i_best, v_best), (i, &v)| {
                if v > v_best { (i, v) } else { (i_best, v_best) }
            },
        );
        KArmedBanditAction::<K>::new(argmax).map_err(|e| TensorConversionError {
            message: format!("invalid one-hot argmax: {e}"),
        })
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`KArmedBandit`].
///
/// Carries no per-arm data: `K` lives at the type level, so the same config
/// is reused regardless of the arm count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KArmedBanditConfig {
    /// Maximum number of steps before the episode terminates.
    pub max_steps: usize,
    /// RNG seed. [`Environment::reset`] re-draws arm means from this seed so
    /// `(config, action sequence)` fully determines the trajectory. Default:
    /// `42` (Sutton & Barto convention).
    pub seed: u64,
}

impl Default for KArmedBanditConfig {
    fn default() -> Self {
        Self {
            max_steps: 500,
            seed: 42,
        }
    }
}

impl Validate for KArmedBanditConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "KArmedBanditConfig";
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

/// Parses configs from `"max_steps=N"`, `"seed=S"`, `"max_steps=N,seed=S"`,
/// or a bare integer interpreted as `max_steps`.
impl FromStr for KArmedBanditConfig {
    type Err = String;

    /// Parses a string into a [`KArmedBanditConfig`].
    ///
    /// Supported formats:
    /// - `"500"` — a bare integer sets `max_steps`; other fields keep their defaults.
    /// - `"max_steps=500"` / `"seed=7"` — single key-value.
    /// - `"max_steps=500,seed=7"` — comma-separated key-value pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if the input matches none of the above, or if a
    /// numeric value fails to parse.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::str::FromStr;
    /// use rlevo_environments::classic::KArmedBanditConfig;
    ///
    /// let c: KArmedBanditConfig = "500".parse().unwrap();
    /// assert_eq!(c.max_steps, 500);
    /// let c: KArmedBanditConfig = "max_steps=1000,seed=7".parse().unwrap();
    /// assert_eq!(c.max_steps, 1000);
    /// assert_eq!(c.seed, 7);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();

        // Bare integer → max_steps.
        if let Ok(max_steps) = trimmed.parse::<usize>() {
            return Ok(Self {
                max_steps,
                ..Self::default()
            });
        }

        let mut cfg = Self::default();
        let mut saw_key = false;
        for pair in trimmed.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            let Some(eq_pos) = pair.find('=') else {
                return Err(format!(
                    "Invalid KArmedBanditConfig format. Expected either a number or 'key=value' pairs, got: {s}"
                ));
            };
            let key = pair[..eq_pos].trim();
            let value_str = pair[eq_pos + 1..].trim();
            match key {
                "max_steps" => {
                    cfg.max_steps = value_str
                        .parse::<usize>()
                        .map_err(|e| format!("Failed to parse max_steps value: {e}"))?;
                }
                "seed" => {
                    cfg.seed = value_str
                        .parse::<u64>()
                        .map_err(|e| format!("Failed to parse seed value: {e}"))?;
                }
                other => {
                    return Err(format!(
                        "Unknown KArmedBanditConfig key {other:?} (expected max_steps or seed)"
                    ));
                }
            }
            saw_key = true;
        }

        if saw_key {
            Ok(cfg)
        } else {
            Err(format!(
                "Invalid KArmedBanditConfig format. Expected either a number or 'key=value' pairs, got: {s}"
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// K-armed bandit environment, generic over the arm count `K`.
#[derive(Debug)]
pub struct KArmedBandit<const K: usize> {
    state: KArmedBanditState,
    steps: usize,
    /// Episode-lifecycle guard: rejects a [`Environment::step`] taken after the
    /// episode ended (ADR 0044). `EpisodeStatus` is the single source of truth
    /// for done-ness (`docs/rules.md` §10).
    guard: EpisodeGuard,
    config: KArmedBanditConfig,
    rng: StdRng,
    arm_means: [f32; K],
}

impl<const K: usize> Display for KArmedBandit<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KArmedBandit<{K}>(step={}/{}, status={:?})",
            self.steps,
            self.config.max_steps,
            self.guard.status()
        )
    }
}

impl<const K: usize> KArmedBandit<K> {
    /// Construct a bandit with a specific seed.
    ///
    /// Sets `config.seed = seed` and samples `arm_means` once from `N(0, 1)`.
    /// The arm means are fixed for the lifetime of the bandit — [`reset`] does
    /// not redraw them — so `rlevo-benchmarks` gets a reproducible problem per
    /// trial by constructing with a fixed seed. Keeps other config fields at
    /// their defaults.
    ///
    /// [`reset`]: Environment::reset
    ///
    /// # Panics
    ///
    /// Panics if the default configuration fails validation. This cannot happen
    /// for the shipped defaults; it would indicate a bug in
    /// `KArmedBanditConfig::default`, not misuse by the caller.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        let config = KArmedBanditConfig {
            seed,
            ..KArmedBanditConfig::default()
        };
        Self::with_config(config).expect("default-derived config must validate")
    }

    /// Construct with an explicit config.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]
    /// (`max_steps == 0`).
    pub fn with_config(config: KArmedBanditConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let arm_means = sample_arm_means::<K>(&mut rng);
        Ok(Self {
            state: KArmedBanditState,
            steps: 0,
            guard: EpisodeGuard::new(),
            config,
            rng,
            arm_means,
        })
    }

    /// Inherent reset — clears episode state only.
    ///
    /// The fixed arm means (the bandit problem) are sampled once at
    /// construction and **preserved** across resets, and the persistent RNG is
    /// **not** re-seeded, so successive episodes draw independent reward
    /// realisations from the same problem (per the host-RNG seeding convention,
    /// `docs/rules.md` §8). For a reproducible problem, construct a fresh bandit
    /// with the same seed.
    ///
    /// This is the bespoke entry point used by `rlevo-benchmarks`; it
    /// discards the snapshot return value. Prefer the
    /// [`Environment::reset`] trait method for new code — it returns a
    /// [`SnapshotBase`] for composition with wrappers.
    ///
    /// The episode guard is re-opened here, so both reset lanes (this one and
    /// [`Environment::reset`], which delegates to it) agree.
    pub fn reset(&mut self) {
        self.guard.reset();
        self.state = KArmedBanditState;
        self.steps = 0;
    }

    /// Pull `arm` and return a sampled reward from `N(q*(arm), 1)`.
    ///
    /// Advances the internal step counter and, like [`Environment::step`],
    /// records the resulting episode status on the guard: `Terminated` once
    /// `steps >= max_steps`, `Running` before that. Both entry points funnel
    /// through the same private `advance` helper, so the guard and the step
    /// counter can never disagree about where the episode is — pulling to the
    /// step budget closes the episode for the trait `step` too.
    ///
    /// Unlike [`Environment::step`], `pull` is **infallible and unguarded on
    /// entry**: it does not reject a post-terminal call. It is a bespoke
    /// non-trait entry point kept for `rlevo-benchmarks`, and making it
    /// fallible would be an unrelated API break. Callers that need the
    /// post-terminal contract of ADR 0044 must use [`Environment::step`].
    ///
    /// # Panics
    ///
    /// Panics if `arm >= K`. Use [`KArmedBanditAction::new`] when validating
    /// untrusted input.
    pub fn pull(&mut self, arm: usize) -> f32 {
        let action = KArmedBanditAction::<K>::new(arm).expect("arm index in range");
        let reward = self.sample_reward(action.arm());
        let _ = self.advance(action, ScalarReward(reward));
        reward
    }

    /// Ticks the step counter, builds the snapshot for this transition, and
    /// records **that snapshot's own status** on the guard.
    ///
    /// The single place the episode advances: shared by [`Environment::step`]
    /// and [`Self::pull`] so the guard cannot drift from the emitted snapshot
    /// (ADR 0044 §5).
    fn advance(
        &mut self,
        action: KArmedBanditAction<K>,
        reward: ScalarReward,
    ) -> SnapshotBase<1, KArmedBanditObservation, ScalarReward> {
        self.steps += 1;
        let obs = self.observe(&action, &self.state);
        let snap = if self.steps >= self.config.max_steps {
            SnapshotBase::terminated(obs, reward)
        } else {
            SnapshotBase::running(obs, reward)
        };
        self.guard.record(snap.status());
        snap
    }

    /// Read-only view of the true arm means.
    #[must_use]
    pub fn arm_means(&self) -> &[f32; K] {
        &self.arm_means
    }

    fn sample_reward(&mut self, arm: usize) -> f32 {
        let mean = self.arm_means[arm];
        Normal::new(mean, 1.0)
            .expect("N(mean, 1) is always valid")
            .sample(&mut self.rng)
    }
}

pub(super) fn sample_arm_means<const K: usize>(rng: &mut StdRng) -> [f32; K] {
    let normal = Normal::new(0.0_f32, 1.0).expect("N(0, 1) is always valid");
    let mut arm_means = [0.0_f32; K];
    for mean in &mut arm_means {
        *mean = normal.sample(rng);
    }
    arm_means
}

impl<const K: usize> ConstructableEnv for KArmedBandit<K> {
    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(KArmedBanditConfig::default()).expect("default config must validate")
    }
}

impl<const K: usize> Sensor<1, 1, 1> for KArmedBandit<K> {
    type Action = KArmedBanditAction<K>;
    type State = KArmedBanditState;
    type Observation = KArmedBanditObservation;

    /// Emits the (empty) stateless bandit observation; the action and resulting
    /// state carry no information the observation depends on.
    fn observe(&self, _action: &Self::Action, _next_state: &Self::State) -> Self::Observation {
        KArmedBanditObservation
    }

    /// Emits the initial (empty) stateless bandit observation.
    fn observe_reset(&self, _state: &Self::State) -> Self::Observation {
        KArmedBanditObservation
    }
}

impl<const K: usize> Environment<1, 1, 1> for KArmedBandit<K> {
    type StateType = KArmedBanditState;
    type ObservationType = KArmedBanditObservation;
    type ActionType = KArmedBanditAction<K>;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, KArmedBanditObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        KArmedBandit::reset(self);
        Ok(SnapshotBase::running(
            self.observe_reset(&self.state),
            ScalarReward::zero(),
        ))
    }

    /// # Errors
    ///
    /// Returns [`EnvironmentError::StepAfterEpisodeEnd`] when the episode has
    /// already ended — checked **first**, before the action is validated and
    /// before any RNG draw, so a rejected call leaves the environment (and its
    /// reward stream) untouched. Returns
    /// [`EnvironmentError::InvalidAction`] when the arm index is out of range.
    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        self.guard.check()?;
        if !action.is_valid() {
            return Err(EnvironmentError::InvalidAction(format!(
                "arm index {} out of range [0, {K})",
                action.arm(),
            )));
        }
        let reward = ScalarReward(self.sample_reward(action.arm()));
        Ok(self.advance(action, reward))
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl<const K: usize> crate::render::AsciiRenderable for KArmedBandit<K> {
    fn render_ascii(&self) -> String {
        let (best_arm, best_mean) = argmax(&self.arm_means);
        format!(
            "K-armed (K={K})  best_arm={best_arm} (q*={best_mean:.2})  step={}/{}",
            self.steps, self.config.max_steps
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        let line = self.render_ascii();
        crate::render::StyledFrame {
            lines: vec![style_bandit_line(&line)],
        }
    }
}

/// Argmax over arm means, returning `(index, value)`.
///
/// Defined locally so the impl doesn't need to import the `slice::iter`
/// dance at every callsite; shared across the four bandit variants by
/// living in `k_armed`.
pub(super) fn argmax(values: &[f32]) -> (usize, f32) {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or((0, 0.0), |(i, v)| (i, *v))
}

/// Style a bandit single-line render whose leading token is a label.
///
/// Used by every bandit variant. The label up to the first run of two
/// spaces is the agent marker; rest is unstyled.
pub(super) fn style_bandit_line(line: &str) -> crate::render::StyledLine {
    use crate::render::palette::{AGENT_FG, AGENT_MODIFIER};
    use crate::render::{SpanStyle, StyledLine, StyledSpan};

    let agent_style = SpanStyle::default()
        .fg(AGENT_FG)
        .with_modifier(AGENT_MODIFIER);

    if let Some(sep) = line.find("  ") {
        let label = &line[..sep];
        let rest = &line[sep..];
        StyledLine::from_spans(vec![
            StyledSpan::new(label, agent_style),
            StyledSpan::raw(rest.to_string()),
        ])
    } else {
        StyledLine::unstyled(line)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    // Exact comparison is intentional throughout this test module: the values
    // are literals or seeds read back without arithmetic, or two identically
    // seeded runs that must agree bit-for-bit. A tolerance would let a real
    // regression pass. Reviewed as a class, not site-by-site.
    #![allow(clippy::float_cmp)]

    use super::*;
    use crate::episode::assert_rejects_post_terminal_step;
    use rlevo_core::environment::EpisodeStatus;

    #[test]
    fn default_config_validates() {
        assert!(KArmedBanditConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_max_steps() {
        let bad = KArmedBanditConfig {
            max_steps: 0,
            seed: 0,
        };
        assert!(KArmedBandit::<3>::with_config(bad).is_err());
    }

    type TestBackend = burn::backend::Flex;
    const K: usize = 10;

    #[test]
    fn state_round_trips_through_tensor() {
        let device = Default::default();
        let state = KArmedBanditState;
        let tensor =
            <KArmedBanditState as TensorConvertible<1, TestBackend>>::to_tensor(&state, &device);
        let back = <KArmedBanditState as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
            .expect("round-trip should succeed for valid shape");
        assert_eq!(back, state);
    }

    #[test]
    fn state_from_tensor_rejects_wrong_shape() {
        use burn::tensor::{Tensor, TensorData as TD};
        let device = Default::default();
        let data = TD::new(vec![0.0_f32, 0.0_f32], [2]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);
        let err = <KArmedBanditState as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
            .expect_err("shape [2] should be rejected");
        assert!(err.message.contains("expected shape [1]"));
    }

    #[test]
    fn action_from_index_round_trips() {
        for i in 0..K {
            let action = KArmedBanditAction::<K>::from_index(i);
            assert_eq!(action.to_index(), i);
            assert!(action.is_valid());
        }
    }

    #[test]
    fn action_new_rejects_out_of_range() {
        let err = KArmedBanditAction::<K>::new(K).expect_err("expected InvalidAction");
        matches!(err, EnvironmentError::InvalidAction(_));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn action_from_index_panics_out_of_range() {
        let _ = KArmedBanditAction::<K>::from_index(K);
    }

    #[test]
    fn action_enumerate_covers_all_arms() {
        let all = KArmedBanditAction::<K>::enumerate();
        assert_eq!(all.len(), K);
        for (i, a) in all.iter().enumerate() {
            assert_eq!(a.to_index(), i);
        }
    }

    #[test]
    fn action_one_hot_round_trips_through_tensor() {
        let device = Default::default();
        for i in 0..K {
            let a = KArmedBanditAction::<K>::from_index(i);
            let t = <KArmedBanditAction<K> as TensorConvertible<1, TestBackend>>::to_tensor(
                &a, &device,
            );
            let back = <KArmedBanditAction<K> as TensorConvertible<1, TestBackend>>::from_tensor(t)
                .expect("valid one-hot");
            assert_eq!(back, a);
        }
    }

    #[test]
    fn action_from_tensor_rejects_wrong_shape() {
        use burn::tensor::{Tensor, TensorData as TD};
        let device = Default::default();
        let data = TD::new(vec![0.0_f32, 1.0_f32], [2]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);
        let err = <KArmedBanditAction<K> as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
            .expect_err("shape [2] should be rejected");
        assert!(err.message.contains("expected shape"));
    }

    #[test]
    fn environment_new_constructs() {
        let env = <KArmedBandit<K> as ConstructableEnv>::new(false);
        assert_eq!(env.steps, 0);
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Running,
            "a freshly constructed bandit must be steppable"
        );
    }

    #[test]
    fn environment_reset_yields_running_snapshot_with_zero_reward() {
        let mut env =
            KArmedBandit::<K>::with_config(KArmedBanditConfig::default()).expect("valid config");
        let snap = <KArmedBandit<K> as Environment<1, 1, 1>>::reset(&mut env).expect("reset");
        assert!(!snap.is_done());
        assert_eq!(f32::from(*snap.reward()), 0.0);
    }

    #[test]
    fn environment_step_terminates_at_max_steps() {
        let mut env = KArmedBandit::<K>::with_config(KArmedBanditConfig {
            max_steps: 3,
            seed: 1,
        })
        .expect("valid config");
        let action = KArmedBanditAction::<K>::from_index(0);
        let s1 = <KArmedBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(!s1.is_done());
        let s2 = <KArmedBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(!s2.is_done());
        let s3 = <KArmedBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(s3.is_terminated());
    }

    // ── post-terminal step guard ──────────────────────────────────────────
    //
    // This bandit used to carry a `done: bool` field that was written but
    // never read by `step()`: nothing consulted it, so a `step()` call made
    // after the episode had already terminated just kept incrementing
    // `self.steps`, sailed past the `steps >= max_steps` check, sampled a
    // *fresh* random reward, and emitted a *new* `Terminated` snapshot on
    // every extra call — an episode that looked like it kept generating
    // distinct terminal transitions after it was already over. Worse,
    // `KArmedBandit` also exposed a public inherent `is_done()` that acted
    // as a second, independent source of truth for done-ness, in violation
    // of `docs/rules.md` §10 ("`EpisodeStatus` is the single source of
    // truth for episode termination; never check done-ness by any other
    // means"); `is_done()` and the backing `done` field were exactly the
    // "other means" that rule forbids.
    //
    // Both are gone now (ADR 0044): `done: bool` was replaced by the
    // `guard: EpisodeGuard` field above, and the public inherent
    // `KArmedBandit::is_done()` was removed outright (a breaking public API
    // change, noted in the changelog) — callers read done-ness from the
    // snapshot via `Snapshot::is_done()`, per rules §10. `Environment::step`
    // now calls `self.guard.check()?` before any RNG draw, since this env
    // samples rewards on every step: a rejected post-terminal call must not
    // advance the RNG stream. The tests below exercise that contract via
    // `crate::episode::assert_rejects_post_terminal_step`.

    /// Step budget for the guard tests: small enough to burn through quickly.
    const GUARD_MAX_STEPS: usize = 3;

    fn guard_env() -> KArmedBandit<K> {
        KArmedBandit::<K>::with_config(KArmedBanditConfig {
            max_steps: GUARD_MAX_STEPS,
            seed: 7,
        })
        .expect("valid config")
    }

    /// Resets, then steps arm 0 until the step budget terminates the episode.
    fn drive_to_termination(
        env: &mut KArmedBandit<K>,
    ) -> SnapshotBase<1, KArmedBanditObservation, ScalarReward> {
        <KArmedBandit<K> as Environment<1, 1, 1>>::reset(env).expect("reset must succeed");
        let action = KArmedBanditAction::<K>::from_index(0);
        let mut snap = <KArmedBandit<K> as Environment<1, 1, 1>>::step(env, action)
            .expect("first step must succeed");
        while !snap.is_done() {
            snap = <KArmedBandit<K> as Environment<1, 1, 1>>::step(env, action)
                .expect("step must succeed while the episode is running");
        }
        snap
    }

    #[test]
    fn rejects_post_terminal_step() {
        let mut env = guard_env();
        assert_rejects_post_terminal_step(
            &mut env,
            drive_to_termination,
            KArmedBanditAction::<K>::from_index(0),
        );
    }

    #[test]
    fn post_terminal_step_is_rejected_before_action_validity() {
        // Being past the end of the episode is a call-sequence fact that does
        // not depend on the action being well-formed: replaying a malformed
        // action past a terminal must report StepAfterEpisodeEnd, not
        // InvalidAction.
        let mut env = guard_env();
        let terminal = drive_to_termination(&mut env);

        let malformed = KArmedBanditAction::<K>::out_of_range_for_tests();
        assert!(!malformed.is_valid(), "the replayed action is out of range");

        let err = <KArmedBandit<K> as Environment<1, 1, 1>>::step(&mut env, malformed)
            .expect_err("a step after termination must be rejected");
        match err {
            EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
                status,
                terminal.status(),
                "the call-sequence error wins over the action's own validity"
            ),
            other => panic!("expected StepAfterEpisodeEnd, got {other:?}"),
        }
    }

    #[test]
    fn post_terminal_step_does_not_advance_the_step_counter() {
        let mut env = guard_env();
        drive_to_termination(&mut env);
        let steps_at_end = env.steps;

        let _ = <KArmedBandit<K> as Environment<1, 1, 1>>::step(
            &mut env,
            KArmedBanditAction::<K>::from_index(0),
        )
        .expect_err("a step after termination must be rejected");

        assert_eq!(
            env.steps, steps_at_end,
            "a rejected step must not tick the step counter"
        );
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Terminated,
            "a rejected step must not reopen the episode"
        );
    }

    #[test]
    fn rejected_step_does_not_advance_the_rng_stream() {
        // ADR 0029: the env's RNG is persistent, observable state. A rejected
        // step must draw no randomness, so the next episode's rewards must be
        // exactly those of a same-seed env that never made the illegal call.
        let mut rejected = guard_env();
        let mut untouched = guard_env();
        drive_to_termination(&mut rejected);
        drive_to_termination(&mut untouched);

        let _ = <KArmedBandit<K> as Environment<1, 1, 1>>::step(
            &mut rejected,
            KArmedBanditAction::<K>::from_index(0),
        )
        .expect_err("a step after termination must be rejected");

        let after: Vec<f32> = {
            <KArmedBandit<K> as Environment<1, 1, 1>>::reset(&mut rejected).unwrap();
            (0..GUARD_MAX_STEPS).map(|_| rejected.pull(0)).collect()
        };
        let baseline: Vec<f32> = {
            <KArmedBandit<K> as Environment<1, 1, 1>>::reset(&mut untouched).unwrap();
            (0..GUARD_MAX_STEPS).map(|_| untouched.pull(0)).collect()
        };
        assert_eq!(
            after, baseline,
            "a rejected step must draw no randomness; the next episode must replay identically"
        );
    }

    #[test]
    fn reset_reopens_a_terminated_episode() {
        let mut env = guard_env();
        drive_to_termination(&mut env);
        let action = KArmedBanditAction::<K>::from_index(0);
        assert!(
            <KArmedBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).is_err(),
            "the episode has ended; a step must be rejected before reset()"
        );

        <KArmedBandit<K> as Environment<1, 1, 1>>::reset(&mut env).expect("reset must succeed");
        assert!(
            !<KArmedBandit<K> as Environment<1, 1, 1>>::step(&mut env, action)
                .expect("reset() must re-open the environment")
                .is_done(),
            "the first step of a fresh episode must not be done"
        );
    }

    #[test]
    fn inherent_reset_reopens_a_terminated_episode() {
        // `KArmedBandit::reset` is the bespoke benchmark lane; it must clear the
        // guard too, or the two reset paths disagree.
        let mut env = guard_env();
        drive_to_termination(&mut env);

        KArmedBandit::reset(&mut env);
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Running,
            "the inherent reset must re-open the guard as well"
        );
        assert!(
            <KArmedBandit<K> as Environment<1, 1, 1>>::step(
                &mut env,
                KArmedBanditAction::<K>::from_index(0)
            )
            .is_ok(),
            "the inherent reset must leave the environment steppable"
        );
    }

    #[test]
    fn pull_records_the_same_status_the_trait_step_would() {
        // `pull` is unguarded on entry but shares `advance`, so it closes the
        // episode at exactly the same step count the trait `step` does.
        let mut env = guard_env();
        for _ in 0..GUARD_MAX_STEPS - 1 {
            let _ = env.pull(0);
            assert_eq!(
                env.guard.status(),
                EpisodeStatus::Running,
                "the episode is still inside its step budget"
            );
        }
        let _ = env.pull(0);
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Terminated,
            "pulling to max_steps must terminate the episode for the trait step too"
        );
        assert!(
            <KArmedBandit<K> as Environment<1, 1, 1>>::step(
                &mut env,
                KArmedBanditAction::<K>::from_index(0)
            )
            .is_err(),
            "the guard and the step counter must not disagree across the two entry points"
        );
    }

    #[test]
    fn same_seed_produces_identical_trajectories() {
        let cfg = KArmedBanditConfig {
            max_steps: 64,
            seed: 7,
        };
        let mut a = KArmedBandit::<K>::with_config(cfg.clone()).expect("valid config");
        let mut b = KArmedBandit::<K>::with_config(cfg).expect("valid config");
        <KArmedBandit<K> as Environment<1, 1, 1>>::reset(&mut a).unwrap();
        <KArmedBandit<K> as Environment<1, 1, 1>>::reset(&mut b).unwrap();
        assert_eq!(a.arm_means(), b.arm_means());

        for step in 0..64 {
            let action = KArmedBanditAction::<K>::from_index(step % K);
            let snap_a = <KArmedBandit<K> as Environment<1, 1, 1>>::step(&mut a, action).unwrap();
            let snap_b = <KArmedBandit<K> as Environment<1, 1, 1>>::step(&mut b, action).unwrap();
            assert_eq!(f32::from(*snap_a.reward()), f32::from(*snap_b.reward()));
            assert_eq!(snap_a.status(), snap_b.status());
        }
    }

    #[test]
    fn reset_keeps_stable_arm_means() {
        // The bandit problem (arm means) is sampled once at construction and
        // preserved across resets — reset() clears only episode state.
        let cfg = KArmedBanditConfig {
            max_steps: 10,
            seed: 99,
        };
        let mut env = KArmedBandit::<K>::with_config(cfg).expect("valid config");
        let means_before = *env.arm_means();
        for _ in 0..5 {
            let _ = env.pull(0);
        }
        <KArmedBandit<K> as Environment<1, 1, 1>>::reset(&mut env).unwrap();
        let means_after = *env.arm_means();
        assert_eq!(means_before, means_after);
        assert_eq!(env.steps, 0);
    }

    #[test]
    fn successive_episodes_draw_independent_rewards() {
        // The persistent RNG advances across resets, so pulling the same arm
        // after a reset yields a different reward realisation — the same fixed
        // problem, independent noise per episode (not bit-identical replay).
        let cfg = KArmedBanditConfig {
            max_steps: 100,
            seed: 7,
        };
        let mut env = KArmedBandit::<K>::with_config(cfg).expect("valid config");
        let episode1: Vec<f32> = (0..16).map(|_| env.pull(0)).collect();
        <KArmedBandit<K> as Environment<1, 1, 1>>::reset(&mut env).unwrap();
        let episode2: Vec<f32> = (0..16).map(|_| env.pull(0)).collect();
        assert_ne!(
            episode1, episode2,
            "reward realisations must differ across episodes"
        );
    }

    #[test]
    fn alias_ten_armed_bandit_resolves_to_k_equals_10() {
        // Confirms the `pub type TenArmedBandit = KArmedBandit<10>` alias in
        // `super::mod` produces an equivalent environment.
        use crate::classic::{TenArmedBandit, TenArmedBanditAction};
        let mut env =
            TenArmedBandit::with_config(KArmedBanditConfig::default()).expect("valid config");
        <TenArmedBandit as Environment<1, 1, 1>>::reset(&mut env).unwrap();
        let action = TenArmedBanditAction::from_index(0);
        let snap = <TenArmedBandit as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(!snap.is_done());
        assert_eq!(env.arm_means().len(), 10);
    }

    #[test]
    fn k_other_than_10_constructs_and_steps() {
        // Smoke-test the genericity: a 4-armed bandit walks through reset/step.
        let mut env =
            KArmedBandit::<4>::with_config(KArmedBanditConfig::default()).expect("valid config");
        <KArmedBandit<4> as Environment<1, 1, 1>>::reset(&mut env).unwrap();
        assert_eq!(env.arm_means().len(), 4);
        let action = KArmedBanditAction::<4>::from_index(3);
        let _ = <KArmedBandit<4> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
    }

    #[test]
    fn fromstr_simple_number_sets_max_steps() {
        let c: KArmedBanditConfig = "500".parse().unwrap();
        assert_eq!(c.max_steps, 500);
        assert_eq!(c.seed, 42);
    }

    #[test]
    fn fromstr_with_whitespace() {
        let c: KArmedBanditConfig = "  750  ".parse().unwrap();
        assert_eq!(c.max_steps, 750);
    }

    #[test]
    fn fromstr_key_value_max_steps() {
        let c: KArmedBanditConfig = "max_steps=1000".parse().unwrap();
        assert_eq!(c.max_steps, 1000);
    }

    #[test]
    fn fromstr_key_value_seed() {
        let c: KArmedBanditConfig = "seed=17".parse().unwrap();
        assert_eq!(c.seed, 17);
        assert_eq!(c.max_steps, 500);
    }

    #[test]
    fn fromstr_two_keys() {
        let c: KArmedBanditConfig = "max_steps=50,seed=3".parse().unwrap();
        assert_eq!(c.max_steps, 50);
        assert_eq!(c.seed, 3);
    }

    #[test]
    fn fromstr_key_value_with_whitespace() {
        let c: KArmedBanditConfig = "max_steps = 2000".parse().unwrap();
        assert_eq!(c.max_steps, 2000);
    }

    #[test]
    fn fromstr_zero_steps() {
        let c: KArmedBanditConfig = "0".parse().unwrap();
        assert_eq!(c.max_steps, 0);
    }

    #[test]
    fn fromstr_large_number() {
        let c: KArmedBanditConfig = "999999999".parse().unwrap();
        assert_eq!(c.max_steps, 999_999_999);
    }

    #[test]
    fn fromstr_invalid_format_errors() {
        let err: String = "invalid".parse::<KArmedBanditConfig>().unwrap_err();
        assert!(err.contains("Invalid KArmedBanditConfig format"));
    }

    #[test]
    fn fromstr_non_numeric_errors() {
        let err = "not_a_number".parse::<KArmedBanditConfig>();
        assert!(err.is_err());
    }

    #[test]
    fn fromstr_invalid_kv_number_errors() {
        let err: String = "max_steps=invalid"
            .parse::<KArmedBanditConfig>()
            .unwrap_err();
        assert!(err.contains("Failed to parse max_steps value"));
    }

    #[test]
    fn fromstr_unknown_key_errors() {
        let err: String = "wrong_key=500".parse::<KArmedBanditConfig>().unwrap_err();
        assert!(err.contains("Unknown KArmedBanditConfig key"));
    }

    #[test]
    fn config_default_has_expected_values() {
        let c = KArmedBanditConfig::default();
        assert_eq!(c.max_steps, 500);
        assert_eq!(c.seed, 42);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let env: KArmedBandit<K> = KArmedBandit::with_seed(7);
        let plain = env.render_ascii();
        let styled = env.render_styled();
        assert_eq!(styled.lines.len(), 1);
        assert_eq!(styled.plain_text(), plain);
    }

    #[test]
    fn render_styled_uses_palette_consts() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER};

        let env: KArmedBandit<K> = KArmedBandit::with_seed(7);
        let styled = env.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text.starts_with("K-armed"))
            .expect("K-armed label span present");
        assert_eq!(label.style.fg, Some(AGENT_FG));
        assert!(label.style.modifier.contains(AGENT_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let env: KArmedBandit<K> = KArmedBandit::with_seed(7);
        for line in env.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
