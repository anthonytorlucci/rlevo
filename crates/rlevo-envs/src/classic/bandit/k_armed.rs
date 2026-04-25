//! K-armed bandit environment — Sutton & Barto, *Reinforcement Learning*, §2.
//!
//! Stateless k-armed bandit generic over the number of arms `K`. On each step
//! the agent selects an arm `a ∈ {0, …, K-1}` and receives a reward sampled
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
//! use rlevo_core::environment::{Environment, Snapshot};
//! use rlevo_envs::classic::{KArmedBandit, KArmedBanditAction};
//!
//! let mut env: KArmedBandit<10> =
//!     <KArmedBandit<10> as Environment<1, 1, 1>>::new(false);
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
    Action, Observation, Reward, State, TensorConversionError, TensorConvertible,
};
use rlevo_core::environment::{Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

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
    type Observation = KArmedBanditObservation;

    fn shape() -> [usize; 1] {
        [1]
    }

    fn observe(&self) -> Self::Observation {
        KArmedBanditObservation
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

impl<B: Backend> TensorConvertible<1, B> for KArmedBanditState {
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
/// use rlevo_envs::classic::KArmedBanditAction;
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

impl<const K: usize, B: Backend> TensorConvertible<1, B> for KArmedBanditAction<K> {
    /// One-hot encoding of the selected arm as a rank-1 tensor of length `K`.
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        let mut one_hot = [0.0_f32; K];
        one_hot[self.selected_arm] = 1.0;
        Tensor::from_floats(one_hot, device)
    }

    /// Reconstructs an action from a one-hot tensor by argmax.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape is not `[K]` or
    /// the argmax falls outside the valid arm range.
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.shape().dims;
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
    /// use rlevo_envs::classic::KArmedBanditConfig;
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
    done: bool,
    config: KArmedBanditConfig,
    rng: StdRng,
    arm_means: [f32; K],
}

impl<const K: usize> Display for KArmedBandit<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KArmedBandit<{K}>(step={}/{}, done={})",
            self.steps, self.config.max_steps, self.done
        )
    }
}

impl<const K: usize> KArmedBandit<K> {
    /// Construct a bandit with a specific seed.
    ///
    /// Sets `config.seed = seed` (so [`Environment::reset`] will re-draw the
    /// same arm means) and samples `arm_means` from `N(0, 1)`. Keeps other
    /// config fields at their defaults. Used by `rlevo-benchmarks` for
    /// reproducible trials.
    pub fn with_seed(seed: u64) -> Self {
        let config = KArmedBanditConfig {
            seed,
            ..KArmedBanditConfig::default()
        };
        Self::with_config(config)
    }

    /// Construct with an explicit config.
    pub fn with_config(config: KArmedBanditConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let arm_means = sample_arm_means::<K>(&mut rng);
        Self {
            state: KArmedBanditState,
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
        self.arm_means = sample_arm_means::<K>(&mut self.rng);
        self.state = KArmedBanditState;
        self.steps = 0;
        self.done = false;
    }

    /// Pull `arm` and return a sampled reward from `N(q*(arm), 1)`.
    ///
    /// Advances the internal step counter and marks the episode `done` when
    /// `steps == max_steps`.
    ///
    /// # Panics
    ///
    /// Panics if `arm >= K`. Use [`KArmedBanditAction::new`] when validating
    /// untrusted input.
    pub fn pull(&mut self, arm: usize) -> f32 {
        let action = KArmedBanditAction::<K>::new(arm).expect("arm index in range");
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

impl<const K: usize> Environment<1, 1, 1> for KArmedBandit<K> {
    type StateType = KArmedBanditState;
    type ObservationType = KArmedBanditObservation;
    type ActionType = KArmedBanditAction<K>;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, KArmedBanditObservation, ScalarReward>;

    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(KArmedBanditConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        KArmedBandit::reset(self);
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward::zero(),
        ))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        if !action.is_valid() {
            return Err(EnvironmentError::InvalidAction(format!(
                "arm index {} out of range [0, {K})",
                action.arm(),
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
        let err =
            <KArmedBanditAction<K> as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
                .expect_err("shape [2] should be rejected");
        assert!(err.message.contains("expected shape"));
    }

    #[test]
    fn environment_new_constructs() {
        let env = <KArmedBandit<K> as Environment<1, 1, 1>>::new(false);
        assert_eq!(env.steps, 0);
        assert!(!env.done);
    }

    #[test]
    fn environment_reset_yields_running_snapshot_with_zero_reward() {
        let mut env = KArmedBandit::<K>::with_config(KArmedBanditConfig::default());
        let snap = <KArmedBandit<K> as Environment<1, 1, 1>>::reset(&mut env).expect("reset");
        assert!(!snap.is_done());
        assert_eq!(f32::from(*snap.reward()), 0.0);
    }

    #[test]
    fn environment_step_terminates_at_max_steps() {
        let mut env = KArmedBandit::<K>::with_config(KArmedBanditConfig {
            max_steps: 3,
            seed: 1,
        });
        let action = KArmedBanditAction::<K>::from_index(0);
        let s1 = <KArmedBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(!s1.is_done());
        let s2 = <KArmedBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(!s2.is_done());
        let s3 = <KArmedBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(s3.is_terminated());
    }

    #[test]
    fn same_seed_produces_identical_trajectories() {
        let cfg = KArmedBanditConfig {
            max_steps: 64,
            seed: 7,
        };
        let mut a = KArmedBandit::<K>::with_config(cfg.clone());
        let mut b = KArmedBandit::<K>::with_config(cfg);
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
    fn reset_redraws_arm_means_from_config_seed() {
        let cfg = KArmedBanditConfig {
            max_steps: 10,
            seed: 99,
        };
        let mut env = KArmedBandit::<K>::with_config(cfg);
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
    fn alias_ten_armed_bandit_resolves_to_k_equals_10() {
        // Confirms the `pub type TenArmedBandit = KArmedBandit<10>` alias in
        // `super::mod` produces an equivalent environment.
        use crate::classic::{TenArmedBandit, TenArmedBanditAction};
        let mut env = TenArmedBandit::with_config(KArmedBanditConfig::default());
        <TenArmedBandit as Environment<1, 1, 1>>::reset(&mut env).unwrap();
        let action = TenArmedBanditAction::from_index(0);
        let snap = <TenArmedBandit as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(!snap.is_done());
        assert_eq!(env.arm_means().len(), 10);
    }

    #[test]
    fn k_other_than_10_constructs_and_steps() {
        // Smoke-test the genericity: a 4-armed bandit walks through reset/step.
        let mut env = KArmedBandit::<4>::with_config(KArmedBanditConfig::default());
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
}
