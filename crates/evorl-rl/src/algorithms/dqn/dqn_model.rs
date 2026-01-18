use burn::module::{AutodiffModule, Module};
use burn::tensor::backend::AutodiffBackend;

/// A trait for DQN models that supports transition between
/// training (Autodiff) and target (Inner) states.
///
/// `DqnModel` requires the supertrait to ensure that any struct implementing `DqnModel` must also
/// support the transiftion to an `InnerModule`. This is the standard way in Burn to handle the
/// relationship between a traininable model and its non-differentiable counterpart.
pub trait DqnModel<B: AutodiffBackend>: AutodiffModule<B> {
    /// Update the target network (InnerModule) using weights from the active network (Self).
    ///
    /// Formula: target = (1 - tau) * target + tau * active
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule;
}
