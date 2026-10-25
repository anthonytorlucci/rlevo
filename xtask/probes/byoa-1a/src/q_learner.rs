//! The submission: tabular Q-learning, written from scratch. Nothing in
//! `rlevo` implements it.
//!
//! The learner knows no environment. It is generic over a discrete action
//! and reads a state's index through the function it is built with, so this
//! file names nothing but the agent seam.

use std::marker::PhantomData;

use rlevo::core::fitness::Metric;
use rlevo::prelude::{Agent, DiscreteAction, EpisodeStatus, Rng, RngExt, Transition};

/// Learning rate α.
const ALPHA: f32 = 0.5;
/// Discount γ.
const GAMMA: f32 = 0.95;
/// ε for the first training episode.
const EPSILON_START: f64 = 1.0;
/// The floor ε falls to by the end of the schedule.
const EPSILON_END: f64 = 0.05;
/// Every value before its first update. On a map whose goal pays 1.0 and
/// every other tile 0, an untried action then looks best and the learner
/// sweeps the map. A zero table ties everywhere, the greedy choice is index
/// 0, and the learner spends its exploration pressing into one wall.
const OPTIMISTIC: f32 = 1.0;

/// The update count's name in a trial's report.
pub const UPDATES: &str = "agent.qlearn_updates";
/// The mean absolute TD error's name in a trial's report.
pub const MEAN_ABS_TD: &str = "agent.qlearn_mean_abs_td";

/// Tabular Q-learning over `A`'s actions and the states `state_of` numbers.
pub struct QLearner<O, A> {
    /// One row of `A::ACTION_COUNT` values per state.
    q: Vec<f32>,
    state_of: fn(&O) -> usize,
    /// Training episodes over which ε falls from its start to its floor.
    schedule: usize,
    episodes: usize,
    training: bool,
    updates: u64,
    abs_td_total: f64,
    action: PhantomData<A>,
}

impl<O, A: DiscreteAction<1>> QLearner<O, A> {
    /// A learner for `states` states, every value optimistic, set to train.
    pub fn new(states: usize, state_of: fn(&O) -> usize, schedule: usize) -> Self {
        assert!(schedule > 0, "the ε schedule needs at least one episode");
        Self {
            q: vec![OPTIMISTIC; states * A::ACTION_COUNT],
            state_of,
            schedule,
            episodes: 0,
            training: true,
            updates: 0,
            abs_td_total: 0.0,
            action: PhantomData,
        }
    }

    /// ε for the next action: linear from its start to its floor over the
    /// schedule while training, and 0 while the learner is measured.
    #[allow(clippy::cast_precision_loss)]
    fn epsilon(&self) -> f64 {
        if !self.training {
            return 0.0;
        }
        let progress = (self.episodes as f64 / self.schedule as f64).min(1.0);
        EPSILON_START + (EPSILON_END - EPSILON_START) * progress
    }

    fn row(&self, obs: &O) -> &[f32] {
        let start = (self.state_of)(obs) * A::ACTION_COUNT;
        &self.q[start..start + A::ACTION_COUNT]
    }

    /// The best action's index, the lowest among equals.
    fn greedy(&self, obs: &O) -> usize {
        let row = self.row(obs);
        let mut best = 0;
        for (index, value) in row.iter().enumerate() {
            if *value > row[best] {
                best = index;
            }
        }
        best
    }

    #[allow(clippy::cast_precision_loss)]
    fn mean_abs_td(&self) -> f64 {
        if self.updates == 0 {
            0.0
        } else {
            self.abs_td_total / self.updates as f64
        }
    }
}

impl<O, A: DiscreteAction<1>> Agent<O, A> for QLearner<O, A> {
    /// ε-greedy, every draw from the driver's `rng`.
    fn act(&mut self, obs: &O, rng: &mut dyn Rng) -> A {
        let index = if rng.random_bool(self.epsilon()) {
            rng.random_range(0..A::ACTION_COUNT)
        } else {
            self.greedy(obs)
        };
        A::from_index(index)
    }

    /// One Q-learning update while training. The target bootstraps from
    /// `next_obs` unless the task ended there: a step a time limit cut,
    /// `Truncated`, still has a future.
    fn on_transition(&mut self, t: Transition<&O, &A>, _rng: &mut dyn Rng) {
        if !self.training {
            return;
        }
        let future = if t.status.is_terminated() {
            0.0
        } else {
            GAMMA
                * self
                    .row(t.next_obs)
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
        };
        let cell = (self.state_of)(t.obs) * A::ACTION_COUNT + t.action.to_index();
        let td = t.reward + future - self.q[cell];
        self.q[cell] += ALPHA * td;
        self.updates += 1;
        self.abs_td_total += f64::from(td.abs());
    }

    /// Counts the training episodes the ε schedule runs over.
    fn on_episode_end(&mut self, _status: EpisodeStatus) {
        if self.training {
            self.episodes += 1;
        }
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn emit_metrics(&self) -> Vec<Metric> {
        vec![
            Metric::Counter {
                name: UPDATES.to_string(),
                count: self.updates,
            },
            Metric::Scalar {
                name: MEAN_ABS_TD.to_string(),
                value: self.mean_abs_td(),
            },
        ]
    }
}
