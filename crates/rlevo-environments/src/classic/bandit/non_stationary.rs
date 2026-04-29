//! Non-stationary k-armed bandit — random-walk drift over arm means.
//!
//! Identical to [`super::k_armed::KArmedBandit`] except that after every
//! `step` each true arm mean is perturbed by independent Gaussian noise:
//!
//! ```text
//! q*(a) ← q*(a) + N(0, σ_walk²)   for all a ∈ {0, …, K-1}
//! ```
//!
//! This is the testbed from Sutton & Barto §2.5 used to demonstrate that
//! constant step-size update rules outperform sample-average estimators when
//! the underlying problem drifts. With `σ_walk = 0` the environment reduces
//! to the stationary [`KArmedBandit`](super::k_armed::KArmedBandit).
//!
//! # Example
//!
//! ```rust
//! use rlevo_core::environment::{Environment, Snapshot};
//! use rlevo_envs::classic::{
//!     KArmedBanditAction, NonStationaryBandit, NonStationaryBanditConfig,
//! };
//!
//! let cfg = NonStationaryBanditConfig {
//!     sigma_walk: 0.01,
//!     ..NonStationaryBanditConfig::default()
//! };
//! let mut env = NonStationaryBandit::<10>::with_config(cfg);
//! let _ = <NonStationaryBandit<10> as Environment<1, 1, 1>>::reset(&mut env)
//!     .expect("reset succeeds");
//! let action = KArmedBanditAction::<10>::new(3).expect("arm in range");
//! let _ = <NonStationaryBandit<10> as Environment<1, 1, 1>>::step(&mut env, action)
//!     .expect("step");
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use rlevo_core::base::{Action, Reward, State};
use rlevo_core::environment::{Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use super::k_armed::{
    KArmedBanditAction, KArmedBanditObservation, KArmedBanditState, sample_arm_means,
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`NonStationaryBandit`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonStationaryBanditConfig {
    /// Maximum number of steps before the episode terminates.
    pub max_steps: usize,
    /// RNG seed.
    pub seed: u64,
    /// Standard deviation of the per-step Gaussian random walk applied to
    /// every arm mean. Sutton & Barto §2.5 uses `0.01`. Setting this to `0.0`
    /// reduces the env to a stationary k-armed bandit.
    pub sigma_walk: f32,
}

impl Default for NonStationaryBanditConfig {
    fn default() -> Self {
        Self {
            max_steps: 500,
            seed: 42,
            sigma_walk: 0.01,
        }
    }
}

/// Parses `"N"` (sets `max_steps`) or comma-separated `key=value` pairs over
/// `max_steps`, `seed`, and `sigma_walk`.
impl FromStr for NonStationaryBanditConfig {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();

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
                    "Invalid NonStationaryBanditConfig format. Expected either a number or 'key=value' pairs, got: {s}"
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
                "sigma_walk" => {
                    cfg.sigma_walk = value_str
                        .parse::<f32>()
                        .map_err(|e| format!("Failed to parse sigma_walk value: {e}"))?;
                }
                other => {
                    return Err(format!(
                        "Unknown NonStationaryBanditConfig key {other:?} (expected max_steps, seed, or sigma_walk)"
                    ));
                }
            }
            saw_key = true;
        }

        if saw_key {
            Ok(cfg)
        } else {
            Err(format!(
                "Invalid NonStationaryBanditConfig format. Expected either a number or 'key=value' pairs, got: {s}"
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Non-stationary k-armed bandit with Gaussian random-walk drift over arm
/// means after each step.
#[derive(Debug)]
pub struct NonStationaryBandit<const K: usize> {
    state: KArmedBanditState,
    steps: usize,
    done: bool,
    config: NonStationaryBanditConfig,
    rng: StdRng,
    arm_means: [f32; K],
}

impl<const K: usize> Display for NonStationaryBandit<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NonStationaryBandit<{K}>(step={}/{}, sigma_walk={}, done={})",
            self.steps, self.config.max_steps, self.config.sigma_walk, self.done
        )
    }
}

impl<const K: usize> NonStationaryBandit<K> {
    /// Construct with a specific seed (other config fields default).
    pub fn with_seed(seed: u64) -> Self {
        let config = NonStationaryBanditConfig {
            seed,
            ..NonStationaryBanditConfig::default()
        };
        Self::with_config(config)
    }

    /// Construct with an explicit config.
    pub fn with_config(config: NonStationaryBanditConfig) -> Self {
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

    /// Read-only view of the current (drifting) arm means.
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

    fn drift_arm_means(&mut self) {
        if self.config.sigma_walk == 0.0 {
            return;
        }
        let walk = Normal::new(0.0_f32, self.config.sigma_walk)
            .expect("N(0, sigma_walk) is valid for finite sigma_walk >= 0");
        for mean in &mut self.arm_means {
            *mean += walk.sample(&mut self.rng);
        }
    }
}

impl<const K: usize> Environment<1, 1, 1> for NonStationaryBandit<K> {
    type StateType = KArmedBanditState;
    type ObservationType = KArmedBanditObservation;
    type ActionType = KArmedBanditAction<K>;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, KArmedBanditObservation, ScalarReward>;

    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(NonStationaryBanditConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        self.arm_means = sample_arm_means::<K>(&mut self.rng);
        self.state = KArmedBanditState;
        self.steps = 0;
        self.done = false;
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
        self.drift_arm_means();
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
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::environment::Snapshot;

    const K: usize = 10;

    #[test]
    fn environment_reset_yields_running_snapshot_with_zero_reward() {
        let mut env =
            NonStationaryBandit::<K>::with_config(NonStationaryBanditConfig::default());
        let snap =
            <NonStationaryBandit<K> as Environment<1, 1, 1>>::reset(&mut env).expect("reset");
        assert!(!snap.is_done());
        assert_eq!(f32::from(*snap.reward()), 0.0);
    }

    #[test]
    fn environment_step_terminates_at_max_steps() {
        let mut env = NonStationaryBandit::<K>::with_config(NonStationaryBanditConfig {
            max_steps: 3,
            seed: 1,
            sigma_walk: 0.01,
        });
        let action = KArmedBanditAction::<K>::from_index(0);
        let s1 =
            <NonStationaryBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(!s1.is_done());
        let _ = <NonStationaryBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        let s3 =
            <NonStationaryBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(s3.is_terminated());
    }

    #[test]
    fn arm_means_drift_after_each_step() {
        let mut env = NonStationaryBandit::<K>::with_config(NonStationaryBanditConfig {
            max_steps: 100,
            seed: 7,
            sigma_walk: 0.1, // large enough to make drift overwhelmingly likely
        });
        <NonStationaryBandit<K> as Environment<1, 1, 1>>::reset(&mut env).unwrap();
        let before = *env.arm_means();
        let action = KArmedBanditAction::<K>::from_index(0);
        <NonStationaryBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        let after = *env.arm_means();
        // With sigma_walk = 0.1 the probability that *every* arm's drift
        // sample is exactly zero is effectively nil; require at least one
        // mean to differ.
        assert!(
            before.iter().zip(after.iter()).any(|(b, a)| b != a),
            "expected arm means to drift; before={before:?}, after={after:?}"
        );
    }

    #[test]
    fn sigma_walk_zero_keeps_means_stationary() {
        let mut env = NonStationaryBandit::<K>::with_config(NonStationaryBanditConfig {
            max_steps: 100,
            seed: 7,
            sigma_walk: 0.0,
        });
        <NonStationaryBandit<K> as Environment<1, 1, 1>>::reset(&mut env).unwrap();
        let before = *env.arm_means();
        let action = KArmedBanditAction::<K>::from_index(0);
        for _ in 0..10 {
            <NonStationaryBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        }
        let after = *env.arm_means();
        assert_eq!(before, after);
    }

    #[test]
    fn same_seed_produces_identical_trajectories() {
        let cfg = NonStationaryBanditConfig {
            max_steps: 32,
            seed: 13,
            sigma_walk: 0.05,
        };
        let mut a = NonStationaryBandit::<K>::with_config(cfg.clone());
        let mut b = NonStationaryBandit::<K>::with_config(cfg);
        <NonStationaryBandit<K> as Environment<1, 1, 1>>::reset(&mut a).unwrap();
        <NonStationaryBandit<K> as Environment<1, 1, 1>>::reset(&mut b).unwrap();
        for step in 0..32 {
            let action = KArmedBanditAction::<K>::from_index(step % K);
            let snap_a =
                <NonStationaryBandit<K> as Environment<1, 1, 1>>::step(&mut a, action).unwrap();
            let snap_b =
                <NonStationaryBandit<K> as Environment<1, 1, 1>>::step(&mut b, action).unwrap();
            assert_eq!(f32::from(*snap_a.reward()), f32::from(*snap_b.reward()));
        }
        assert_eq!(a.arm_means(), b.arm_means());
    }

    #[test]
    fn fromstr_kv_with_sigma_walk() {
        let c: NonStationaryBanditConfig =
            "max_steps=200,seed=9,sigma_walk=0.05".parse().unwrap();
        assert_eq!(c.max_steps, 200);
        assert_eq!(c.seed, 9);
        assert!((c.sigma_walk - 0.05).abs() < 1e-6);
    }

    #[test]
    fn fromstr_unknown_key_errors() {
        let err: String = "wrong=1"
            .parse::<NonStationaryBanditConfig>()
            .unwrap_err();
        assert!(err.contains("Unknown NonStationaryBanditConfig key"));
    }
}
