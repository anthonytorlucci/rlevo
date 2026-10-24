//! The agent seam: [`Agent`], the one trait a driver steps anything that
//! acts in an [`Environment`] through, and [`Transition`], what it is shown
//! after every step.
//!
//! A frozen policy implements [`Agent::act`] and nothing else. A learner also
//! implements the hooks it needs: [`Agent::on_transition`] sees every step,
//! [`Agent::on_episode_end`] the end of every episode, and
//! [`Agent::set_training`] which phase a run is in. There is no `learn`
//! method: when an agent learns from what the hooks show it, and how, is the
//! agent's own business.
//!
//! # The order of calls
//!
//! For every step of an episode a driver calls:
//!
//! 1. [`Agent::act`], and steps the environment with the action;
//! 2. [`Agent::on_transition`], whose `status` is the snapshot's, passed
//!    through unchanged;
//! 3. when that status is done, [`Agent::on_episode_end`] with the same
//!    status.
//!
//! An episode the driver cuts short at a step cap of its own ends with
//! `on_episode_end(EpisodeStatus::Truncated)` after a last transition whose
//! status is still `Running`. The environment did not end the episode, so its
//! last observation has a future, and a learner that bootstraps from it is
//! right to.
//!
//! `set_training(true)` precedes a phase in which the agent may explore and
//! learn, and `set_training(false)` a phase in which it is measured.
//!
//! [`Agent`]: crate::agent::Agent
//! [`Agent::act`]: crate::agent::Agent::act
//! [`Agent::on_transition`]: crate::agent::Agent::on_transition
//! [`Agent::on_episode_end`]: crate::agent::Agent::on_episode_end
//! [`Agent::set_training`]: crate::agent::Agent::set_training
//! [`Transition`]: crate::agent::Transition
//! [`Environment`]: crate::environment::Environment

use rand::Rng;

use crate::environment::EpisodeStatus;
use crate::fitness::Metric;

/// One step of an episode: the observation the agent acted on, its action,
/// and what the environment returned.
///
/// A driver shows an agent the borrowed form, `Transition<&Obs, &Act>`; an
/// agent that keeps transitions, as a replay buffer does, stores an owned
/// form such as `Transition<Obs, usize>`.
///
/// # `status`, not a `bool`
///
/// [`EpisodeStatus::Terminated`] means the task ended at `next_obs`, so there
/// is no value beyond it. [`EpisodeStatus::Truncated`] means the episode was
/// cut while the task went on, so `next_obs` still has a future and a value
/// target bootstraps from it. A single `done` flag would merge the two and
/// bias every bootstrapped estimate on a time-limited environment (Pardo et
/// al., "Time Limits in Reinforcement Learning", ICML 2018).
///
/// # Examples
///
/// ```
/// use rlevo_core::agent::Transition;
/// use rlevo_core::environment::EpisodeStatus;
///
/// let (obs, next_obs, action) = ([0.0_f32, 1.0], [1.0_f32, 1.0], 1_usize);
/// let shown = Transition {
///     obs: &obs,
///     action: &action,
///     reward: 0.5,
///     next_obs: &next_obs,
///     status: EpisodeStatus::Truncated,
/// };
/// // What a replay buffer keeps.
/// let stored: Transition<[f32; 2], usize> = Transition {
///     obs: *shown.obs,
///     action: *shown.action,
///     reward: shown.reward,
///     next_obs: *shown.next_obs,
///     status: shown.status,
/// };
/// assert!(!stored.status.is_terminated());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transition<O, A> {
    /// The observation the agent acted on.
    pub obs: O,
    /// The action it took.
    pub action: A,
    /// The reward the environment returned for the action.
    pub reward: f32,
    /// The observation the environment returned for the action.
    pub next_obs: O,
    /// How the episode stands after the action: running, or ended by the task
    /// (`Terminated`) or by a cut (`Truncated`).
    pub status: EpisodeStatus,
}

/// Anything that acts in an environment and that a driver steps: a frozen
/// policy, or a learner.
///
/// Only [`act`](Self::act) is required. The other four methods default to
/// doing nothing, so a frozen policy is one method long; the module docs give
/// the order a driver calls them in.
///
/// # Randomness
///
/// Every random draw an agent makes, in [`act`](Self::act) or in
/// [`on_transition`](Self::on_transition) (an exploration draw, a replay
/// minibatch, a rollout shuffle), comes from the `rng` the driver passes.
/// An agent that draws from a generator of its own breaks the driver's
/// promise that a seed reproduces a run.
///
/// # Examples
///
/// A frozen policy:
///
/// ```
/// use rlevo_core::agent::Agent;
/// use rlevo_core::rand::Rng;
///
/// struct Thermostat;
///
/// impl Agent<f32, bool> for Thermostat {
///     fn act(&mut self, temperature: &f32, _rng: &mut dyn Rng) -> bool {
///         *temperature < 20.0
///     }
/// }
/// ```
///
/// A learner that keeps a running mean of the reward each action earns:
///
/// ```
/// use rlevo_core::agent::{Agent, Transition};
/// use rlevo_core::rand::{Rng, RngExt};
///
/// #[derive(Default)]
/// struct Bandit {
///     sums: [f32; 2],
///     counts: [f32; 2],
///     training: bool,
/// }
///
/// impl Agent<(), usize> for Bandit {
///     fn act(&mut self, _obs: &(), rng: &mut dyn Rng) -> usize {
///         if self.training && rng.random_bool(0.1) {
///             return rng.random_range(0..2);
///         }
///         let mean = |a: usize| self.sums[a] / self.counts[a].max(1.0);
///         usize::from(mean(1) > mean(0))
///     }
///
///     fn on_transition(&mut self, t: Transition<&(), &usize>, _rng: &mut dyn Rng) {
///         if self.training {
///             self.sums[*t.action] += t.reward;
///             self.counts[*t.action] += 1.0;
///         }
///     }
///
///     fn set_training(&mut self, training: bool) {
///         self.training = training;
///     }
/// }
/// ```
pub trait Agent<Obs, Act> {
    /// Chooses an action for `obs`.
    ///
    /// Deterministic given `rng`'s state: see the trait's "Randomness".
    fn act(&mut self, obs: &Obs, rng: &mut dyn Rng) -> Act;

    /// Shows the agent the step it just took, once per environment step,
    /// after [`act`](Self::act) and before the next one.
    ///
    /// `transition.status` is the environment's, unchanged. A learner stores
    /// the step, updates on its own cadence, or both; `rng` is the driver's,
    /// as in `act`. The default does nothing.
    fn on_transition(&mut self, transition: Transition<&Obs, &Act>, rng: &mut dyn Rng) {
        let _ = (transition, rng);
    }

    /// Tells the agent its episode has ended, and how.
    ///
    /// Called after the episode's last [`on_transition`](Self::on_transition),
    /// with that transition's status, or with
    /// [`EpisodeStatus::Truncated`] when the driver cut the episode at its own
    /// step cap (see the module docs). An agent with per-episode state, such
    /// as a recurrent hidden state, resets it here. The default does nothing.
    fn on_episode_end(&mut self, status: EpisodeStatus) {
        let _ = status;
    }

    /// Sets whether the agent is training (it may explore and learn) or being
    /// evaluated (it acts on what it has learned, and learns nothing).
    ///
    /// The default does nothing, which is right for an agent that behaves the
    /// same in both.
    fn set_training(&mut self, training: bool) {
        let _ = training;
    }

    /// Method-specific metrics, pulled once per trial after its last episode
    /// and covering the whole trial, training included.
    ///
    /// A harness may record these under a namespace of its own; the
    /// [`Metric`] docs say how. The default reports none.
    fn emit_metrics(&self) -> Vec<Metric> {
        Vec::new()
    }
}
