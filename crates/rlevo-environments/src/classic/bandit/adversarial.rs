//! Adversarial k-armed bandit — oblivious adversary with a deterministic
//! periodic reward schedule.
//!
//! Unlike the stochastic variants in this module, the rewards here are not
//! sampled from per-arm distributions. Instead, an *oblivious adversary*
//! generates a deterministic per-arm reward sequence in `[0, amplitude]`
//! before the agent ever acts:
//!
//! ```text
//! r_t(a) = amplitude · 0.5 · (1 + cos(2π · (t + phase[a]) / period))
//! ```
//!
//! `phase[a]` is drawn once per [`Environment::reset`] from the seeded RNG so
//! a given `(seed, period, amplitude)` triple yields identical reward
//! sequences across processes — the standard reproducibility property
//! algorithms like EXP3 are benchmarked against.
//!
//! Because the schedule does not depend on agent behaviour, the adversary is
//! *oblivious* (the standard EXP3 setting). An adaptive "punish-the-leader"
//! adversary is intentionally out of scope; it would couple the env to agent
//! state and complicate reproducibility.
//!
//! # Example
//!
//! ```rust
//! use rlevo_core::environment::{Environment, Snapshot};
//! use rlevo_envs::classic::{
//!     AdversarialBandit, AdversarialBanditConfig, KArmedBanditAction,
//! };
//!
//! let cfg = AdversarialBanditConfig {
//!     period: 10,
//!     amplitude: 1.0,
//!     ..AdversarialBanditConfig::default()
//! };
//! let mut env = AdversarialBandit::<10>::with_config(cfg);
//! let _ = <AdversarialBandit<10> as Environment<1, 1, 1>>::reset(&mut env)
//!     .expect("reset succeeds");
//! let action = KArmedBanditAction::<10>::new(3).expect("arm in range");
//! let _ = <AdversarialBandit<10> as Environment<1, 1, 1>>::step(&mut env, action)
//!     .expect("step");
//! ```

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::base::{Action, Reward, State};
use rlevo_core::environment::{Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use super::k_armed::{KArmedBanditAction, KArmedBanditObservation, KArmedBanditState};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`AdversarialBandit`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialBanditConfig {
    /// Maximum number of steps before the episode terminates.
    pub max_steps: usize,
    /// RNG seed driving the per-arm phase draw.
    pub seed: u64,
    /// Period (in steps) of the cosine reward schedule. Each arm completes
    /// one full cycle every `period` steps. Default: `10`.
    pub period: usize,
    /// Peak reward amplitude. Rewards are bounded in `[0, amplitude]`.
    /// Default: `1.0` (the EXP3 standard `[0, 1]` range).
    pub amplitude: f32,
}

impl Default for AdversarialBanditConfig {
    fn default() -> Self {
        Self {
            max_steps: 500,
            seed: 42,
            period: 10,
            amplitude: 1.0,
        }
    }
}

/// Parses `"N"` (sets `max_steps`) or comma-separated `key=value` pairs over
/// `max_steps`, `seed`, `period`, and `amplitude`.
impl FromStr for AdversarialBanditConfig {
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
                    "Invalid AdversarialBanditConfig format. Expected either a number or 'key=value' pairs, got: {s}"
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
                "period" => {
                    cfg.period = value_str
                        .parse::<usize>()
                        .map_err(|e| format!("Failed to parse period value: {e}"))?;
                }
                "amplitude" => {
                    cfg.amplitude = value_str
                        .parse::<f32>()
                        .map_err(|e| format!("Failed to parse amplitude value: {e}"))?;
                }
                other => {
                    return Err(format!(
                        "Unknown AdversarialBanditConfig key {other:?} (expected max_steps, seed, period, or amplitude)"
                    ));
                }
            }
            saw_key = true;
        }

        if saw_key {
            Ok(cfg)
        } else {
            Err(format!(
                "Invalid AdversarialBanditConfig format. Expected either a number or 'key=value' pairs, got: {s}"
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Adversarial k-armed bandit with an oblivious periodic reward schedule.
#[derive(Debug)]
pub struct AdversarialBandit<const K: usize> {
    state: KArmedBanditState,
    steps: usize,
    done: bool,
    config: AdversarialBanditConfig,
    rng: StdRng,
    phases: [usize; K],
}

impl<const K: usize> Display for AdversarialBandit<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AdversarialBandit<{K}>(step={}/{}, period={}, done={})",
            self.steps, self.config.max_steps, self.config.period, self.done
        )
    }
}

impl<const K: usize> AdversarialBandit<K> {
    /// Construct with a specific seed (other config fields default).
    pub fn with_seed(seed: u64) -> Self {
        let config = AdversarialBanditConfig {
            seed,
            ..AdversarialBanditConfig::default()
        };
        Self::with_config(config)
    }

    /// Construct with an explicit config.
    pub fn with_config(config: AdversarialBanditConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let phases = sample_phases::<K>(&mut rng, config.period.max(1));
        Self {
            state: KArmedBanditState,
            steps: 0,
            done: false,
            config,
            rng,
            phases,
        }
    }

    /// Read-only view of the per-arm phase offsets used by the reward
    /// schedule.
    #[must_use]
    pub fn phases(&self) -> &[usize; K] {
        &self.phases
    }

    /// Compute the deterministic reward for `arm` at step index `t`.
    fn reward_at(&self, arm: usize, t: usize) -> f32 {
        let period = self.config.period.max(1) as f32;
        let theta = TAU * ((t + self.phases[arm]) as f32) / period;
        self.config.amplitude * 0.5 * (1.0 + theta.cos())
    }
}

fn sample_phases<const K: usize>(rng: &mut StdRng, period: usize) -> [usize; K] {
    let mut phases = [0_usize; K];
    for phase in &mut phases {
        *phase = rng.random_range(0..period);
    }
    phases
}

impl<const K: usize> Environment<1, 1, 1> for AdversarialBandit<K> {
    type StateType = KArmedBanditState;
    type ObservationType = KArmedBanditObservation;
    type ActionType = KArmedBanditAction<K>;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, KArmedBanditObservation, ScalarReward>;

    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(AdversarialBanditConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        self.phases = sample_phases::<K>(&mut self.rng, self.config.period.max(1));
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
        let reward = ScalarReward(self.reward_at(action.arm(), self.steps));
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
            AdversarialBandit::<K>::with_config(AdversarialBanditConfig::default());
        let snap =
            <AdversarialBandit<K> as Environment<1, 1, 1>>::reset(&mut env).expect("reset");
        assert!(!snap.is_done());
        assert_eq!(f32::from(*snap.reward()), 0.0);
    }

    #[test]
    fn rewards_are_bounded_by_amplitude() {
        let cfg = AdversarialBanditConfig {
            max_steps: 200,
            seed: 5,
            period: 7,
            amplitude: 1.0,
        };
        let mut env = AdversarialBandit::<K>::with_config(cfg);
        <AdversarialBandit<K> as Environment<1, 1, 1>>::reset(&mut env).unwrap();
        for step in 0..50 {
            let action = KArmedBanditAction::<K>::from_index(step % K);
            let snap =
                <AdversarialBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
            let r = f32::from(*snap.reward());
            assert!(
                (0.0..=1.0).contains(&r),
                "reward {r} outside [0, 1] at step {step}"
            );
        }
    }

    #[test]
    fn same_seed_produces_identical_reward_sequence() {
        let cfg = AdversarialBanditConfig {
            max_steps: 64,
            seed: 21,
            period: 8,
            amplitude: 1.0,
        };
        let mut a = AdversarialBandit::<K>::with_config(cfg.clone());
        let mut b = AdversarialBandit::<K>::with_config(cfg);
        <AdversarialBandit<K> as Environment<1, 1, 1>>::reset(&mut a).unwrap();
        <AdversarialBandit<K> as Environment<1, 1, 1>>::reset(&mut b).unwrap();
        assert_eq!(a.phases(), b.phases());
        for step in 0..32 {
            let action = KArmedBanditAction::<K>::from_index(step % K);
            let snap_a =
                <AdversarialBandit<K> as Environment<1, 1, 1>>::step(&mut a, action).unwrap();
            let snap_b =
                <AdversarialBandit<K> as Environment<1, 1, 1>>::step(&mut b, action).unwrap();
            assert_eq!(f32::from(*snap_a.reward()), f32::from(*snap_b.reward()));
        }
    }

    #[test]
    fn reward_schedule_is_periodic() {
        // Pulling the same arm `period` steps apart should yield identical
        // rewards (up to f32 round-off) because the schedule is exactly
        // periodic.
        let cfg = AdversarialBanditConfig {
            max_steps: 1000,
            seed: 1,
            period: 5,
            amplitude: 1.0,
        };
        let env = AdversarialBandit::<K>::with_config(cfg);
        let r0 = env.reward_at(2, 0);
        let r5 = env.reward_at(2, 5);
        let r10 = env.reward_at(2, 10);
        assert!((r0 - r5).abs() < 1e-5);
        assert!((r0 - r10).abs() < 1e-5);
    }

    #[test]
    fn environment_step_terminates_at_max_steps() {
        let mut env = AdversarialBandit::<K>::with_config(AdversarialBanditConfig {
            max_steps: 3,
            seed: 1,
            period: 4,
            amplitude: 1.0,
        });
        let action = KArmedBanditAction::<K>::from_index(0);
        let s1 =
            <AdversarialBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(!s1.is_done());
        let _ = <AdversarialBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        let s3 =
            <AdversarialBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(s3.is_terminated());
    }

    #[test]
    fn fromstr_kv_with_period_and_amplitude() {
        let c: AdversarialBanditConfig =
            "max_steps=200,seed=9,period=12,amplitude=0.5".parse().unwrap();
        assert_eq!(c.max_steps, 200);
        assert_eq!(c.seed, 9);
        assert_eq!(c.period, 12);
        assert!((c.amplitude - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fromstr_unknown_key_errors() {
        let err: String = "wrong=1"
            .parse::<AdversarialBanditConfig>()
            .unwrap_err();
        assert!(err.contains("Unknown AdversarialBanditConfig key"));
    }
}
