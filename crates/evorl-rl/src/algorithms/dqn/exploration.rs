//! ε-greedy exploration schedule for DQN.
//!
//! [`EpsilonGreedy`] tracks the current exploration rate, decays it
//! multiplicatively each step, and clamps it at a configurable floor. The
//! agent consults [`EpsilonGreedy::should_explore`] to decide between a random
//! action and an argmax over Q-values.

use rand::{Rng, RngExt};

use crate::algorithms::dqn::dqn_config::DqnTrainingConfig;

/// Multiplicative ε-decay schedule used by DQN's exploration policy.
#[derive(Clone, Debug)]
pub struct EpsilonGreedy {
    epsilon: f64,
    epsilon_min: f64,
    epsilon_decay: f64,
    step: usize,
}

impl EpsilonGreedy {
    /// Creates a new schedule with the given starting epsilon, floor, and
    /// multiplicative decay factor.
    #[must_use]
    pub fn new(epsilon_start: f64, epsilon_min: f64, epsilon_decay: f64) -> Self {
        Self {
            epsilon: epsilon_start,
            epsilon_min,
            epsilon_decay,
            step: 0,
        }
    }

    /// Builds a schedule from the epsilon fields of a [`DqnTrainingConfig`].
    #[must_use]
    pub fn from_config(config: &DqnTrainingConfig) -> Self {
        Self::new(config.epsilon_start, config.epsilon_end, config.epsilon_decay)
    }

    /// Current exploration rate.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.epsilon
    }

    /// Number of decay steps observed so far.
    #[must_use]
    pub fn step(&self) -> usize {
        self.step
    }

    /// Advances the schedule by one step, clamping at `epsilon_min`.
    pub fn decay(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
        self.step += 1;
    }

    /// Samples whether the agent should take a random action this step.
    pub fn should_explore<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        rng.random::<f64>() < self.epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn epsilon_greedy_decay_reaches_floor() {
        let mut schedule = EpsilonGreedy::new(1.0, 0.05, 0.9);
        for _ in 0..1_000 {
            schedule.decay();
        }
        assert!(
            (schedule.value() - 0.05).abs() < 1e-9,
            "epsilon should clamp exactly to the floor, got {}",
            schedule.value()
        );
    }

    #[test]
    fn epsilon_greedy_decay_is_monotonic() {
        let mut schedule = EpsilonGreedy::new(1.0, 0.0, 0.99);
        let mut prev = schedule.value();
        for _ in 0..50 {
            schedule.decay();
            let v = schedule.value();
            assert!(v <= prev);
            prev = v;
        }
    }

    #[test]
    fn should_explore_obeys_epsilon_bounds() {
        let mut rng = StdRng::seed_from_u64(7);
        let always = EpsilonGreedy::new(1.0, 1.0, 1.0);
        let never = EpsilonGreedy::new(0.0, 0.0, 1.0);
        for _ in 0..100 {
            assert!(always.should_explore(&mut rng));
            assert!(!never.should_explore(&mut rng));
        }
    }
}
