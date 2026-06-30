//! Policy contracts driven across an environment rollout.
//!
//! [`StatefulPolicy`] is the general seam: the evolved policy module carries
//! per-episode hidden state across a rollout, so **recurrent / memory policies
//! are first-class**. This is what the library's POMDP environments require —
//! the Santa Fe ant ([`is_markov`](rlevo_core::state::MarkovState::is_markov)
//! `== false`) cannot be cleared by a memoryless reflex.
//!
//! [`ReactivePolicy`] is the `Hidden = ()` convenience for memoryless (Markov)
//! policies: implement one method and get [`StatefulPolicy`] for free via the
//! blanket impl below. Classic-control policies (CartPole, MountainCar, …) take
//! this path.
//!
//! Both traits are env-generic: `act` owns the **full** `&Observation ->
//! Action` mapping (decode the observation, forward the net, select, encode the
//! action), so the rollout seam stays free of any environment-specific decoding.

use burn::tensor::backend::Backend;

use rlevo_core::environment::Environment;

/// Device type for backend `B` (the `Device` associated type lives on the
/// `BackendTypes` supertrait in this Burn version).
type Dev<B> = <B as burn::tensor::backend::BackendTypes>::Device;

/// A policy driven across an environment rollout, carrying per-episode state.
///
/// Implemented by the evolved policy module `M`: the reshaper unflattens a
/// genome into `M`, and the rollout drives `M` through [`reset`](Self::reset)
/// (once at episode start) + [`act`](Self::act) (per step, threading
/// `&mut Hidden`).
///
/// A reactive (Markov) policy is the [`Hidden`](Self::Hidden) `= ()` case — see
/// [`ReactivePolicy`] for the one-method convenience that supplies this impl for
/// free. A recurrent policy implements this trait directly with a concrete
/// `Hidden` (a GRU hidden tensor, an Elman activation, …).
pub trait StatefulPolicy<B: Backend, E: Environment<1, 1, 1>> {
    /// Per-episode hidden / recurrent state. `()` for a reactive policy.
    type Hidden;

    /// Fresh hidden state at episode start. `&self` lets the policy size the
    /// state from its own dimensions; `device` places it on the backend device.
    fn reset(&self, device: &Dev<B>) -> Self::Hidden;

    /// Advance one step: map `(hidden, observation) -> action`, mutating
    /// `hidden` in place. Owns the **full** mapping — decode the observation,
    /// forward the net, select (argmax/sample), encode the action.
    fn act(
        &self,
        hidden: &mut Self::Hidden,
        obs: &E::ObservationType,
        device: &Dev<B>,
    ) -> E::ActionType;
}

/// A reactive (memoryless) policy: `observation -> action`.
///
/// The [`StatefulPolicy::Hidden`] `= ()` restriction, for Markov environments
/// (CartPole, MountainCar, …). Implement this single method and the blanket
/// impl below supplies [`StatefulPolicy`] for free.
///
/// A recurrent policy must **never** implement this trait — the blanket impl
/// would then overlap its direct [`StatefulPolicy`] impl.
pub trait ReactivePolicy<B: Backend, E: Environment<1, 1, 1>> {
    /// Map `observation -> action` with no carried state.
    fn act(&self, obs: &E::ObservationType, device: &Dev<B>) -> E::ActionType;
}

impl<B, E, P> StatefulPolicy<B, E> for P
where
    B: Backend,
    E: Environment<1, 1, 1>,
    P: ReactivePolicy<B, E>,
{
    type Hidden = ();

    fn reset(&self, _device: &Dev<B>) {}

    fn act(
        &self,
        _hidden: &mut (),
        obs: &E::ObservationType,
        device: &Dev<B>,
    ) -> E::ActionType {
        ReactivePolicy::act(self, obs, device)
    }
}
