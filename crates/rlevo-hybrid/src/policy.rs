//! The policy contract driven across an environment rollout.
//!
//! [`StatefulPolicy`] is the seam: the evolved policy module carries
//! per-episode hidden state across a rollout, so **recurrent / memory policies
//! are first-class**. This is what the library's POMDP environments require —
//! the Santa Fe ant, whose one-bit percept is not a Markov state, cannot be
//! cleared by a memoryless reflex. A memoryless (Markov) policy, such as the
//! classic-control ones (`CartPole`, `MountainCar`, …), is the `Hidden = ()`
//! case.
//!
//! The trait is over the observation and action types alone: `act` owns the
//! **full** `&Obs -> Act` mapping (decode the observation, forward the net,
//! select, encode the action), so the rollout seam stays free of any
//! environment-specific decoding, and it names no backend or device.

/// A policy driven across an environment rollout, carrying per-episode state.
///
/// Implemented by the evolved policy module `M`: the reshaper unflattens a
/// genome into `M`, and the rollout drives `M` through [`reset`](Self::reset)
/// (once at episode start) + [`act`](Self::act) (per step, threading
/// `&mut Hidden`).
///
/// A reactive (Markov) policy sets [`Hidden`](Self::Hidden) to `()`, returns
/// `()` from `reset` and ignores `hidden` in `act`. A recurrent policy sets a
/// concrete `Hidden` (a GRU hidden tensor, an Elman activation, …).
///
/// Neither method takes a device: a policy that builds a tensor (an input
/// batch, a zero hidden state) builds it on the device its own weights live
/// on, e.g. `self.layer.weight.device()`.
pub trait StatefulPolicy<Obs, Act> {
    /// Per-episode hidden / recurrent state. `()` for a reactive policy.
    type Hidden;

    /// Fresh hidden state at episode start. `&self` lets the policy size the
    /// state from its own dimensions and place it on its own device.
    fn reset(&self) -> Self::Hidden;

    /// Advance one step: map `(hidden, observation) -> action`, mutating
    /// `hidden` in place. Owns the **full** mapping — decode the observation,
    /// forward the net, select (argmax/sample), encode the action.
    fn act(&self, hidden: &mut Self::Hidden, obs: &Obs) -> Act;
}
