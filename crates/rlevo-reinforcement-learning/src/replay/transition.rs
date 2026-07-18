//! The stored replay item: one `(s, a, r, s', terminated)` transition.
//!
//! Before ADR 0050 each of the six off-policy agents carried its own private
//! `Transition` struct — six definitions of one concept, differing only in the
//! action field (`usize` for the value-based three, `Vec<f32>` for the
//! continuous three). [`Transition<O, P>`] is that one concept, with the action
//! payload lifted into a type parameter and the two shipped payloads named by
//! [`DiscreteTransition`] and [`ContinuousTransition`].

/// A single `(s, a, r, s', terminated)` transition stored in a replay buffer.
///
/// Observations are kept in their original typed form `O` and converted to
/// tensors lazily at sample time, which avoids holding a large flat tensor
/// buffer in memory for the lifetime of the agent.
///
/// # The erased action payload `P`
///
/// `P` is the *stored action payload*, deliberately erased from the domain
/// action type `A: Action<AD>`. Storing `A` itself would impose
/// `Clone + 'static` on every action type in the workspace purely so a buffer
/// could hold it; the agents' staging paths need `usize` / `Vec<f32>` anyway
/// and would immediately erase a typed `A` again. The erasure is the design,
/// not an accident — see ADR 0050 §2 and its *Alternatives considered* entry
/// on `ExperienceTuple`.
///
/// Use the aliases rather than spelling the payload at each site:
/// [`DiscreteTransition<O>`] for index actions, [`ContinuousTransition<O>`] for
/// continuous vectors.
///
/// # Examples
///
/// ```
/// use rlevo_reinforcement_learning::replay::{ContinuousTransition, DiscreteTransition};
///
/// let discrete: DiscreteTransition<[f32; 2]> = DiscreteTransition {
///     obs: [0.0, 1.0],
///     action: 1,
///     reward: 0.5,
///     next_obs: [1.0, 1.0],
///     terminated: false,
/// };
/// assert_eq!(discrete.action, 1);
///
/// let continuous: ContinuousTransition<[f32; 2]> = ContinuousTransition {
///     obs: [0.0, 1.0],
///     action: vec![-0.25],
///     reward: 0.5,
///     next_obs: [1.0, 1.0],
///     terminated: true,
/// };
/// assert_eq!(continuous.action.len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Transition<O, P> {
    /// Observation at time `t`.
    pub obs: O,
    /// The action taken at time `t`, in its erased storage form.
    ///
    /// `usize` — an index into a discrete action space (see
    /// [`DiscreteAction::to_index`](rlevo_core::action::DiscreteAction::to_index))
    /// — or `Vec<f32>` — the raw component vector of a continuous action.
    pub action: P,
    /// Scalar reward received after taking the action.
    pub reward: f32,
    /// Observation at time `t + 1`.
    pub next_obs: O,
    /// `true` **only** for an environmental termination — the MDP reached an
    /// absorbing state, so the return beyond `next_obs` is zero by definition.
    ///
    /// Deliberately *not* `is_done()`: a truncation (time-limit cutoff) ends
    /// the episode without ending the MDP, and `next_obs` is then a genuine
    /// continuation state. Zeroing the bootstrap there biases every Q-value
    /// downward. Partial-episode bootstrapping: Pardo et al., "Time Limits in
    /// Reinforcement Learning", ICML 2018, Eq. 6.
    pub terminated: bool,
}

/// A transition whose action is an index into a discrete action space.
///
/// Stored by the value-based agents (DQN, C51, QR-DQN), which stage the index
/// into a rank-2 `Int` tensor for `gather`.
pub type DiscreteTransition<O> = Transition<O, usize>;

/// A transition whose action is a continuous component vector.
///
/// Stored by the continuous-control agents (DDPG, TD3, SAC), which stage the
/// vector into a rank-2 float tensor fed to the critic alongside the
/// observation.
pub type ContinuousTransition<O> = Transition<O, Vec<f32>>;
