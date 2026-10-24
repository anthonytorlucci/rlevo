//! The stored replay item: [`Transition`], the one step type of
//! `rlevo_core::agent`, with its action erased to the payload a buffer keeps.
//!
//! Before ADR 0050 each of the six off-policy agents carried its own private
//! `Transition` struct — six definitions of one concept, differing only in the
//! action field (`usize` for the value-based three, `Vec<f32>` for the
//! continuous three). [`Transition<O, P>`] is that one concept, with the action
//! payload lifted into a type parameter and the two shipped payloads named by
//! [`DiscreteTransition`] and [`ContinuousTransition`]. It is the core crate's
//! `Transition`, so an agent stores what a driver shows it: the episode's
//! `status`, not a `bool` that would merge termination with truncation.
//!
//! # The erased action payload
//!
//! The payload is deliberately erased from the domain action type. Storing
//! the action itself would impose `Clone + 'static` on every action type in
//! the workspace purely so a buffer could hold it; the agents' staging paths
//! need `usize` / `Vec<f32>` anyway and would immediately erase a typed
//! action again (ADR 0050).

pub use rlevo_core::agent::Transition;

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
