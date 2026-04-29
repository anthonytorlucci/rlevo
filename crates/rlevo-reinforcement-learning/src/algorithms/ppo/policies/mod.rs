//! Built-in [`PpoPolicy`](crate::algorithms::ppo::ppo_policy::PpoPolicy)
//! implementations covering the two common action-space cases.
//!
//! - [`CategoricalPolicyHead`] — discrete action spaces; softmax over a
//!   learned logits head, sampling via Gumbel-max on CPU.
//! - [`TanhGaussianPolicyHead`] — continuous action spaces; state-independent
//!   `log_std` parameter, sampling via reparameterisation `z = μ + σ·ε` then
//!   `a = scale · tanh(z)`.
//!
//! Both heads are full MLPs with two hidden `tanh` layers (CleanRL default).
//! Users who need a different architecture can `impl PpoPolicy` directly on
//! their own [`burn::module::Module`] struct — the traits in
//! [`ppo_policy`](crate::algorithms::ppo::ppo_policy) /
//! [`ppo_value`](crate::algorithms::ppo::ppo_value) place no restrictions on
//! the network shape.

pub mod categorical;
pub mod gaussian;

pub use categorical::{CategoricalPolicyHead, CategoricalPolicyHeadConfig};
pub use gaussian::{TanhGaussianPolicyHead, TanhGaussianPolicyHeadConfig};
