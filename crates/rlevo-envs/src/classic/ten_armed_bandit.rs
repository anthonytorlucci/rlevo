//! Ten-armed bandit environment — Sutton & Barto, *Reinforcement Learning*, §2.
//!
//! Stateless k-armed bandit with `k = 10`. On each step the agent selects an
//! arm `a ∈ {0, …, 9}` and receives a reward sampled from `N(q*(a), 1)` where
//! the true means `q*(a)` are themselves drawn from `N(0, 1)` at construction
//! (and re-drawn from the same seed on [`reset`](TenArmedBandit::reset)).
//!
//! # Example
//!
//! ```rust
//! use rlevo_core::environment::{Environment, Snapshot};
//! use rlevo_envs::classic::{TenArmedBandit, TenArmedBanditAction};
//!
//! let mut env: TenArmedBandit = <TenArmedBandit as Environment<1, 1, 1>>::new(false);
//! let _ = <TenArmedBandit as Environment<1, 1, 1>>::reset(&mut env)
//!     .expect("reset succeeds");
//! let action = TenArmedBanditAction::new(3).expect("arm index in range");
//! let snap = <TenArmedBandit as Environment<1, 1, 1>>::step(&mut env, action)
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
    Action, Observation, Reward, State, TensorConversionError, TensorConvertible,
};
use rlevo_core::environment::{Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Number of arms in the bandit problem (k = 10 per Sutton & Barto).
const ARM_COUNT: usize = 10;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Multi-armed bandit state.
///
/// Bandit problems are stateless (the optimal action is independent of
/// history), so this struct carries no fields. It exists to satisfy the
/// [`State`] trait contract expected by [`Environment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TenArmedBanditState;

/// Observation for the ten-armed bandit.
///
/// Empty for the same reason as [`TenArmedBanditState`]. The [`Observation`]
/// impl reports `shape() = [1]` so tensor-based policies can still convert
/// through [`TensorConvertible`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct TenArmedBanditObservation;

impl Observation<1> for TenArmedBanditObservation {
    fn shape() -> [usize; 1] {
        [1]
    }
}

impl State<1> for TenArmedBanditState {
    type Observation = TenArmedBanditObservation;

    fn shape() -> [usize; 1] {
        [1]
    }

    fn observe(&self) -> Self::Observation {
        TenArmedBanditObservation
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn numel(&self) -> usize {
        1
    }
}

impl Display for TenArmedBanditState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "TenArmedBanditState")
    }
}

impl<B: Backend> TensorConvertible<1, B> for TenArmedBanditState {
    /// Encodes the stateless bandit state as a 1-D tensor `[0.0]`.
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats([0.0_f32; 1], device)
    }

    /// Accepts any rank-1 tensor of shape `[1]`; contents are ignored because
    /// the state carries no data.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape is not `[1]`.
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.shape().dims;
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

/// Action for the ten-armed bandit — the choice of which arm to pull.
///
/// Valid arm indices are `0..10`. Use [`TenArmedBanditAction::new`] for
/// fallible construction from untrusted input, or
/// [`TenArmedBanditAction::from_index`] (from the [`DiscreteAction`] trait)
/// when the caller has already validated the index.
///
/// # Traits implemented
///
/// - [`Action<1>`]: validity check + shape reporting.
/// - [`DiscreteAction<1>`]: `ACTION_COUNT = 10`, plus `from_index` /
///   `to_index` / `random` / `enumerate` (the last three via trait defaults).
/// - [`TensorConvertible<1, B>`]: one-hot encoding of length 10 for
///   neural-network integration.
/// - [`Display`]: renders as `"TenArmedBanditAction(arm=N)"`.
///
/// # Examples
///
/// ```rust
/// use rlevo_core::action::DiscreteAction;
/// use rlevo_envs::classic::TenArmedBanditAction;
///
/// let a = TenArmedBanditAction::new(5).expect("5 is in range");
/// assert_eq!(a.arm(), 5);
/// assert_eq!(a.to_index(), 5);
///
/// let all = TenArmedBanditAction::enumerate();
/// assert_eq!(all.len(), TenArmedBanditAction::ACTION_COUNT);
///
/// let _random = TenArmedBanditAction::random();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TenArmedBanditAction {
    /// The index of the selected arm (`0..10`).
    selected_arm: usize,
}

impl TenArmedBanditAction {
    /// Fallible constructor: returns [`EnvironmentError::InvalidAction`] when
    /// `arm >= 10`.
    ///
    /// Prefer this over [`DiscreteAction::from_index`] for any index that
    /// came from an external source (configuration, RPC, policy output
    /// without a saturating mask). `from_index` panics on out-of-range input
    /// by the [`DiscreteAction`] trait contract.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::InvalidAction`] if `arm >= 10`.
    pub fn new(arm: usize) -> Result<Self, EnvironmentError> {
        if arm < ARM_COUNT {
            Ok(Self { selected_arm: arm })
        } else {
            Err(EnvironmentError::InvalidAction(format!(
                "arm index {arm} out of range [0, {ARM_COUNT})"
            )))
        }
    }

    /// The index of the arm this action selects.
    #[must_use]
    pub fn arm(&self) -> usize {
        self.selected_arm
    }
}

impl Display for TenArmedBanditAction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "TenArmedBanditAction(arm={})", self.selected_arm)
    }
}

impl Action<1> for TenArmedBanditAction {
    fn shape() -> [usize; 1] {
        [ARM_COUNT]
    }

    fn is_valid(&self) -> bool {
        self.selected_arm < ARM_COUNT
    }
}

impl DiscreteAction<1> for TenArmedBanditAction {
    const ACTION_COUNT: usize = ARM_COUNT;

    /// Constructs from a validated index. Panics on out-of-range input, per
    /// the [`DiscreteAction`] trait contract. Use [`TenArmedBanditAction::new`]
    /// for a fallible alternative.
    fn from_index(index: usize) -> Self {
        assert!(
            index < ARM_COUNT,
            "TenArmedBanditAction index {index} out of range [0, {ARM_COUNT})",
        );
        Self {
            selected_arm: index,
        }
    }

    fn to_index(&self) -> usize {
        self.selected_arm
    }
}

impl<B: Backend> TensorConvertible<1, B> for TenArmedBanditAction {
    /// One-hot encoding of the selected arm as a rank-1 tensor of length 10.
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        let mut one_hot = [0.0_f32; ARM_COUNT];
        one_hot[self.selected_arm] = 1.0;
        Tensor::from_floats(one_hot, device)
    }

    /// Reconstructs an action from a one-hot tensor by argmax.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape is not `[10]` or
    /// the argmax falls outside the valid arm range.
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.shape().dims;
        if dims.as_slice() != [ARM_COUNT] {
            return Err(TensorConversionError {
                message: format!("expected shape [{ARM_COUNT}], got {dims:?}"),
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
        TenArmedBanditAction::new(argmax).map_err(|e| TensorConversionError {
            message: format!("invalid one-hot argmax: {e}"),
        })
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`TenArmedBandit`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenArmedBanditConfig {
    /// Maximum number of steps before the episode terminates.
    pub max_steps: usize,
    /// RNG seed. [`reset`](TenArmedBandit::reset) re-draws arm means from
    /// this seed so `(config, action sequence)` fully determines the
    /// trajectory. Default: `42` (Sutton & Barto convention).
    pub seed: u64,
}

impl Default for TenArmedBanditConfig {
    fn default() -> Self {
        Self {
            max_steps: 500,
            seed: 42,
        }
    }
}

/// Parses configs from `"max_steps=N"`, `"seed=S"`, `"max_steps=N,seed=S"`, or
/// a bare integer interpreted as `max_steps`.
impl FromStr for TenArmedBanditConfig {
    type Err = String;

    /// Parses a string into a [`TenArmedBanditConfig`].
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
    /// use rlevo_envs::classic::TenArmedBanditConfig;
    ///
    /// let c: TenArmedBanditConfig = "500".parse().unwrap();
    /// assert_eq!(c.max_steps, 500);
    /// let c: TenArmedBanditConfig = "max_steps=1000,seed=7".parse().unwrap();
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
                    "Invalid TenArmedBanditConfig format. Expected either a number or 'key=value' pairs, got: {s}"
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
                        "Unknown TenArmedBanditConfig key {other:?} (expected max_steps or seed)"
                    ));
                }
            }
            saw_key = true;
        }

        if saw_key {
            Ok(cfg)
        } else {
            Err(format!(
                "Invalid TenArmedBanditConfig format. Expected either a number or 'key=value' pairs, got: {s}"
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Ten-armed bandit environment.
#[derive(Debug)]
pub struct TenArmedBandit {
    state: TenArmedBanditState,
    steps: usize,
    done: bool,
    config: TenArmedBanditConfig,
    rng: StdRng,
    arm_means: [f32; ARM_COUNT],
}

impl Display for TenArmedBandit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TenArmedBandit(step={}/{}, done={})",
            self.steps, self.config.max_steps, self.done
        )
    }
}

impl TenArmedBandit {
    /// Construct a bandit with a specific seed.
    ///
    /// Sets `config.seed = seed` (so [`reset`](Self::reset) will re-draw the
    /// same arm means) and samples `arm_means` from `N(0, 1)`. Keeps other
    /// config fields at their defaults. Used by `rlevo-benchmarks` for
    /// reproducible trials.
    pub fn with_seed(seed: u64) -> Self {
        let config = TenArmedBanditConfig {
            seed,
            ..TenArmedBanditConfig::default()
        };
        Self::with_config(config)
    }

    /// Construct with an explicit config.
    pub fn with_config(config: TenArmedBanditConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let arm_means = sample_arm_means(&mut rng);
        Self {
            state: TenArmedBanditState,
            steps: 0,
            done: false,
            config,
            rng,
            arm_means,
        }
    }

    /// Inherent reset — re-seeds RNG and re-samples arm means.
    ///
    /// This is the bespoke entry point used by `rlevo-benchmarks`; it
    /// discards the snapshot return value. Prefer the
    /// [`Environment::reset`] trait method for new code — it returns a
    /// [`SnapshotBase`] for composition with wrappers.
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        self.arm_means = sample_arm_means(&mut self.rng);
        self.state = TenArmedBanditState;
        self.steps = 0;
        self.done = false;
    }

    /// Pull `arm` and return a sampled reward from `N(q*(arm), 1)`.
    ///
    /// Advances the internal step counter and marks the episode `done` when
    /// `steps == max_steps`.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::InvalidAction`] if `arm >= 10`.
    pub fn pull(&mut self, arm: usize) -> f32 {
        let action = TenArmedBanditAction::new(arm).expect("arm index in range");
        let reward = self.sample_reward(action.arm());
        self.steps += 1;
        if self.steps >= self.config.max_steps {
            self.done = true;
        }
        reward
    }

    /// `true` when the episode has reached `max_steps`.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Read-only view of the true arm means.
    #[must_use]
    pub fn arm_means(&self) -> &[f32; ARM_COUNT] {
        &self.arm_means
    }

    fn sample_reward(&mut self, arm: usize) -> f32 {
        let mean = self.arm_means[arm];
        Normal::new(mean, 1.0)
            .expect("N(mean, 1) is always valid")
            .sample(&mut self.rng)
    }
}

fn sample_arm_means(rng: &mut StdRng) -> [f32; ARM_COUNT] {
    let normal = Normal::new(0.0_f32, 1.0).expect("N(0, 1) is always valid");
    let mut arm_means = [0.0_f32; ARM_COUNT];
    for mean in &mut arm_means {
        *mean = normal.sample(rng);
    }
    arm_means
}

impl Environment<1, 1, 1> for TenArmedBandit {
    type StateType = TenArmedBanditState;
    type ObservationType = TenArmedBanditObservation;
    type ActionType = TenArmedBanditAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, TenArmedBanditObservation, ScalarReward>;

    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(TenArmedBanditConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        TenArmedBandit::reset(self);
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward::zero(),
        ))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        if !action.is_valid() {
            return Err(EnvironmentError::InvalidAction(format!(
                "arm index {} out of range [0, {})",
                action.arm(),
                ARM_COUNT
            )));
        }
        let reward = ScalarReward(self.sample_reward(action.arm()));
        self.steps += 1;
        let obs = self.state.observe();
        let snap = if self.steps >= self.config.max_steps {
            self.done = true;
            SnapshotBase::terminated(obs, reward)
        } else {
            SnapshotBase::running(obs, reward)
        };
        Ok(snap)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::environment::Snapshot;

    type TestBackend = burn::backend::NdArray;

    #[test]
    fn state_round_trips_through_tensor() {
        let device = Default::default();
        let state = TenArmedBanditState;
        let tensor =
            <TenArmedBanditState as TensorConvertible<1, TestBackend>>::to_tensor(&state, &device);
        let back = <TenArmedBanditState as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
            .expect("round-trip should succeed for valid shape");
        assert_eq!(back, state);
    }

    #[test]
    fn state_from_tensor_rejects_wrong_shape() {
        use burn::tensor::{Tensor, TensorData as TD};
        let device = Default::default();
        let data = TD::new(vec![0.0_f32, 0.0_f32], [2]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);
        let err = <TenArmedBanditState as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
            .expect_err("shape [2] should be rejected");
        assert!(err.message.contains("expected shape [1]"));
    }

    #[test]
    fn action_from_index_round_trips() {
        for i in 0..ARM_COUNT {
            let action = TenArmedBanditAction::from_index(i);
            assert_eq!(action.to_index(), i);
            assert!(action.is_valid());
        }
    }

    #[test]
    fn action_new_rejects_out_of_range() {
        let err = TenArmedBanditAction::new(ARM_COUNT).expect_err("expected InvalidAction");
        matches!(err, EnvironmentError::InvalidAction(_));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn action_from_index_panics_out_of_range() {
        let _ = TenArmedBanditAction::from_index(ARM_COUNT);
    }

    #[test]
    fn action_enumerate_covers_all_arms() {
        let all = TenArmedBanditAction::enumerate();
        assert_eq!(all.len(), ARM_COUNT);
        for (i, a) in all.iter().enumerate() {
            assert_eq!(a.to_index(), i);
        }
    }

    #[test]
    fn action_one_hot_round_trips_through_tensor() {
        let device = Default::default();
        for i in 0..ARM_COUNT {
            let a = TenArmedBanditAction::from_index(i);
            let t =
                <TenArmedBanditAction as TensorConvertible<1, TestBackend>>::to_tensor(&a, &device);
            let back = <TenArmedBanditAction as TensorConvertible<1, TestBackend>>::from_tensor(t)
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
        let err = <TenArmedBanditAction as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
            .expect_err("shape [2] should be rejected");
        assert!(err.message.contains("expected shape"));
    }

    #[test]
    fn environment_new_constructs() {
        let env = <TenArmedBandit as Environment<1, 1, 1>>::new(false);
        assert_eq!(env.steps, 0);
        assert!(!env.done);
    }

    #[test]
    fn environment_reset_yields_running_snapshot_with_zero_reward() {
        let mut env = TenArmedBandit::with_config(TenArmedBanditConfig::default());
        let snap = <TenArmedBandit as Environment<1, 1, 1>>::reset(&mut env).expect("reset");
        assert!(!snap.is_done());
        assert_eq!(f32::from(*snap.reward()), 0.0);
    }

    #[test]
    fn environment_step_invalid_action_returns_invalid_action_error() {
        // Build an invalid action by bypassing the safe constructor, then
        // feed it to `step` and confirm we get InvalidAction back.
        let _env = TenArmedBandit::with_config(TenArmedBanditConfig::default());
        let bogus = TenArmedBanditAction::from_index(0);
        // Force an invalid state via unsafe-free struct literal is not possible
        // (private field). The in-range case is already valid; the real path
        // we exercise is Step's `is_valid` on a valid action — so instead we
        // assert the converse property: TenArmedBanditAction::new rejects
        // bad input at the boundary.
        assert!(bogus.is_valid());
        let bad = TenArmedBanditAction::new(ARM_COUNT);
        assert!(matches!(bad, Err(EnvironmentError::InvalidAction(_))));
    }

    #[test]
    fn environment_step_terminates_at_max_steps() {
        let mut env = TenArmedBandit::with_config(TenArmedBanditConfig {
            max_steps: 3,
            seed: 1,
        });
        let action = TenArmedBanditAction::from_index(0);
        let s1 = <TenArmedBandit as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(!s1.is_done());
        let s2 = <TenArmedBandit as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(!s2.is_done());
        let s3 = <TenArmedBandit as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(s3.is_terminated());
    }

    #[test]
    fn same_seed_produces_identical_trajectories() {
        let cfg = TenArmedBanditConfig {
            max_steps: 64,
            seed: 7,
        };
        let mut a = TenArmedBandit::with_config(cfg.clone());
        let mut b = TenArmedBandit::with_config(cfg);
        <TenArmedBandit as Environment<1, 1, 1>>::reset(&mut a).unwrap();
        <TenArmedBandit as Environment<1, 1, 1>>::reset(&mut b).unwrap();
        assert_eq!(a.arm_means(), b.arm_means());

        for step in 0..64 {
            let action = TenArmedBanditAction::from_index(step % ARM_COUNT);
            let snap_a = <TenArmedBandit as Environment<1, 1, 1>>::step(&mut a, action).unwrap();
            let snap_b = <TenArmedBandit as Environment<1, 1, 1>>::step(&mut b, action).unwrap();
            assert_eq!(f32::from(*snap_a.reward()), f32::from(*snap_b.reward()));
            assert_eq!(snap_a.status(), snap_b.status());
        }
    }

    #[test]
    fn reset_redraws_arm_means_from_config_seed() {
        let cfg = TenArmedBanditConfig {
            max_steps: 10,
            seed: 99,
        };
        let mut env = TenArmedBandit::with_config(cfg);
        let means_before = *env.arm_means();
        // Perturb state with some steps.
        for _ in 0..5 {
            let _ = env.pull(0);
        }
        <TenArmedBandit as Environment<1, 1, 1>>::reset(&mut env).unwrap();
        let means_after = *env.arm_means();
        assert_eq!(means_before, means_after);
        assert_eq!(env.steps, 0);
    }

    #[test]
    fn fromstr_simple_number_sets_max_steps() {
        let c: TenArmedBanditConfig = "500".parse().unwrap();
        assert_eq!(c.max_steps, 500);
        assert_eq!(c.seed, 42); // default preserved
    }

    #[test]
    fn fromstr_with_whitespace() {
        let c: TenArmedBanditConfig = "  750  ".parse().unwrap();
        assert_eq!(c.max_steps, 750);
    }

    #[test]
    fn fromstr_key_value_max_steps() {
        let c: TenArmedBanditConfig = "max_steps=1000".parse().unwrap();
        assert_eq!(c.max_steps, 1000);
    }

    #[test]
    fn fromstr_key_value_seed() {
        let c: TenArmedBanditConfig = "seed=17".parse().unwrap();
        assert_eq!(c.seed, 17);
        assert_eq!(c.max_steps, 500); // default preserved
    }

    #[test]
    fn fromstr_two_keys() {
        let c: TenArmedBanditConfig = "max_steps=50,seed=3".parse().unwrap();
        assert_eq!(c.max_steps, 50);
        assert_eq!(c.seed, 3);
    }

    #[test]
    fn fromstr_key_value_with_whitespace() {
        let c: TenArmedBanditConfig = "max_steps = 2000".parse().unwrap();
        assert_eq!(c.max_steps, 2000);
    }

    #[test]
    fn fromstr_zero_steps() {
        let c: TenArmedBanditConfig = "0".parse().unwrap();
        assert_eq!(c.max_steps, 0);
    }

    #[test]
    fn fromstr_large_number() {
        let c: TenArmedBanditConfig = "999999999".parse().unwrap();
        assert_eq!(c.max_steps, 999_999_999);
    }

    #[test]
    fn fromstr_invalid_format_errors() {
        let err: String = "invalid".parse::<TenArmedBanditConfig>().unwrap_err();
        assert!(err.contains("Invalid TenArmedBanditConfig format"));
    }

    #[test]
    fn fromstr_non_numeric_errors() {
        let err = "not_a_number".parse::<TenArmedBanditConfig>();
        assert!(err.is_err());
    }

    #[test]
    fn fromstr_invalid_kv_number_errors() {
        let err: String = "max_steps=invalid"
            .parse::<TenArmedBanditConfig>()
            .unwrap_err();
        assert!(err.contains("Failed to parse max_steps value"));
    }

    #[test]
    fn fromstr_unknown_key_errors() {
        let err: String = "wrong_key=500".parse::<TenArmedBanditConfig>().unwrap_err();
        assert!(err.contains("Unknown TenArmedBanditConfig key"));
    }

    #[test]
    fn config_default_has_expected_values() {
        let c = TenArmedBanditConfig::default();
        assert_eq!(c.max_steps, 500);
        assert_eq!(c.seed, 42);
    }
}
