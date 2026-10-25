//! Your agent.
//!
//! `MyAgent` works as it stands, so the crate runs from its first `cargo
//! run`: Q-learning with a linear value per action, over the numbers an
//! observation writes as its host row. Replace its fields and the bodies of
//! its methods with your algorithm, and keep the five signatures in the
//! `Agent` impl: they are what the harness calls.
//!
//! It is here to show the hooks at work, not to do well. A linear value is a
//! poor fit for CartPole, and its row in the table moves a long way with the
//! seed, to either side of the random agent's.
//!
//! # What the harness calls, and when
//!
//! `Evaluator::run_suite` builds one agent per trial and drives it through
//! one loop. For every step of an episode it calls:
//!
//! 1. `act`, then steps the environment with the action;
//! 2. `on_transition`, whose `status` is the environment's, unchanged;
//! 3. when that status is done, `on_episode_end` with the same status.
//!
//! An episode the harness cuts at `EvaluatorConfig::max_steps` ends with
//! `on_episode_end(EpisodeStatus::Truncated)` after a last transition whose
//! status is still `Running`: the environment did not end it, so its last
//! observation has a future.
//!
//! `set_training(true)` comes before the `train_steps` of training and
//! `set_training(false)` before the evaluation episodes. `emit_metrics` is
//! read once, after the last evaluation episode.
//!
//! Every random draw comes from the `rng` the harness passes, so a seed
//! reproduces a run.

use std::marker::PhantomData;

use rlevo::core::fitness::Metric;
use rlevo::prelude::{Agent, DiscreteAction, EpisodeStatus, HostRow, Rng, RngExt, Transition};

/// Step size of each update.
const ALPHA: f32 = 0.1;
/// Discount on the value of the next observation.
const GAMMA: f32 = 0.99;
/// The exploration rate for the first training episode.
const EPSILON_START: f64 = 1.0;
/// The exploration rate once the schedule has run.
const EPSILON_END: f64 = 0.05;
/// Training episodes over which the exploration rate falls.
const EPSILON_EPISODES: f64 = 100.0;

/// The name the update count is reported under.
pub const UPDATES: &str = "my_agent/updates";

/// Q-learning with one linear value per action.
///
/// The value of action `a` is `w_a · [1, x]`, where `x` is the observation's
/// host row.
pub struct MyAgent<O, A> {
    /// One row per action: a bias, then a weight per number in the row.
    weights: Vec<Vec<f32>>,
    /// Whether the agent may explore and learn.
    training: bool,
    /// Training episodes finished, which the exploration rate falls over.
    episodes: u32,
    /// Updates made since the agent was built.
    updates: u64,
    types: PhantomData<fn(&O) -> A>,
}

impl<O: HostRow<1>, A: DiscreteAction<1>> MyAgent<O, A> {
    /// An agent with every weight zero, set to train.
    pub fn new() -> Self {
        let width = 1 + O::shape()[0];
        Self {
            weights: vec![vec![0.0; width]; A::ACTION_COUNT],
            training: true,
            episodes: 0,
            updates: 0,
            types: PhantomData,
        }
    }

    /// The value of every action at `obs`.
    fn values(&self, obs: &O) -> Vec<f32> {
        let x = features(obs);
        self.weights.iter().map(|w| dot(w, &x)).collect()
    }

    /// The exploration rate: falling while training, zero while measured.
    fn epsilon(&self) -> f64 {
        if !self.training {
            return 0.0;
        }
        let progress = (f64::from(self.episodes) / EPSILON_EPISODES).min(1.0);
        EPSILON_START + (EPSILON_END - EPSILON_START) * progress
    }
}

impl<O: HostRow<1>, A: DiscreteAction<1>> Default for MyAgent<O, A> {
    fn default() -> Self {
        Self::new()
    }
}

/// `[1, x]`: a bias, then the numbers `obs` writes as its host row.
fn features<O: HostRow<1>>(obs: &O) -> Vec<f32> {
    let mut x = vec![1.0];
    obs.write_host_row(&mut x);
    x
}

fn dot(w: &[f32], x: &[f32]) -> f32 {
    w.iter().zip(x).map(|(w, x)| w * x).sum()
}

/// The index of the largest value, the first among equals.
fn best(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

impl<O: HostRow<1>, A: DiscreteAction<1>> Agent<O, A> for MyAgent<O, A> {
    /// The best action, or while training, with probability ε, a random one.
    fn act(&mut self, obs: &O, rng: &mut dyn Rng) -> A {
        if rng.random_bool(self.epsilon()) {
            return A::from_index(rng.random_range(0..A::ACTION_COUNT));
        }
        A::from_index(best(&self.values(obs)))
    }

    /// One update toward `reward + γ · max_a Q(next_obs, a)`.
    ///
    /// The target bootstraps from `next_obs` unless the task ended there
    /// (`Terminated`). A step a time limit cut (`Truncated`) still has a
    /// future, and treating it as an end biases every value near the limit.
    fn on_transition(&mut self, t: Transition<&O, &A>, _rng: &mut dyn Rng) {
        if !self.training {
            return;
        }
        let future = if t.status.is_terminated() {
            0.0
        } else {
            let next = self.values(t.next_obs);
            GAMMA * next[best(&next)]
        };
        let x = features(t.obs);
        let w = &mut self.weights[t.action.to_index()];
        let error = t.reward + future - dot(w, &x);
        for (w, x) in w.iter_mut().zip(&x) {
            *w += ALPHA * error * x;
        }
        self.updates += 1;
    }

    /// Counts the training episodes the exploration rate falls over. An agent
    /// with state that lasts one episode, such as a recurrent state, resets
    /// it here.
    fn on_episode_end(&mut self, _status: EpisodeStatus) {
        if self.training {
            self.episodes += 1;
        }
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// What the agent reports about the trial so far. A counter is a total
    /// since the agent was built, and reading changes nothing.
    fn emit_metrics(&self) -> Vec<Metric> {
        vec![Metric::Counter {
            name: UPDATES.to_string(),
            count: self.updates,
        }]
    }
}

#[cfg(test)]
mod tests {
    use rlevo::core::rand::SeedableRng;
    use rlevo::core::rand::rngs::StdRng;
    use rlevo::envs::classic::{CartPoleAction, CartPoleObservation};

    use super::*;

    fn pole(angle: f32) -> CartPoleObservation {
        CartPoleObservation {
            cart_pos: 0.0,
            cart_vel: 0.0,
            pole_angle: angle,
            pole_ang_vel: 0.0,
        }
    }

    /// The value of pushing left from `obs` after one rewarded step, then
    /// one unrewarded step that ends with `status`.
    fn value_after(status: EpisodeStatus) -> f32 {
        let mut agent = MyAgent::<CartPoleObservation, CartPoleAction>::new();
        let mut rng = StdRng::seed_from_u64(0);
        let (obs, next_obs, action) = (pole(0.0), pole(0.1), CartPoleAction::Left);
        let step = |reward, status| Transition {
            obs: &obs,
            action: &action,
            reward,
            next_obs: &next_obs,
            status,
        };
        // Gives `next_obs` a value worth bootstrapping from.
        agent.on_transition(step(1.0, EpisodeStatus::Running), &mut rng);
        agent.on_transition(step(0.0, status), &mut rng);
        agent.values(&obs)[action.to_index()]
    }

    #[test]
    fn a_truncated_step_bootstraps_and_a_terminated_one_does_not() {
        let truncated = value_after(EpisodeStatus::Truncated);
        let terminated = value_after(EpisodeStatus::Terminated);
        assert!(
            truncated > terminated,
            "truncated {truncated}, terminated {terminated}"
        );
        assert_eq!(truncated, value_after(EpisodeStatus::Running));
    }

    #[test]
    fn an_agent_being_measured_learns_nothing() {
        let mut agent = MyAgent::<CartPoleObservation, CartPoleAction>::new();
        let mut rng = StdRng::seed_from_u64(0);
        agent.set_training(false);
        let (obs, action) = (pole(0.0), CartPoleAction::Right);
        let transition = Transition {
            obs: &obs,
            action: &action,
            reward: 1.0,
            next_obs: &obs,
            status: EpisodeStatus::Running,
        };
        agent.on_transition(transition, &mut rng);
        assert_eq!(agent.values(&obs), [0.0, 0.0]);
        assert!(matches!(
            agent.emit_metrics().as_slice(),
            [Metric::Counter { count: 0, .. }]
        ));
    }
}
