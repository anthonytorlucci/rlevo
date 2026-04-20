//! Built-in PPG policy heads.
//!
//! v1 ships a discrete-only [`PpgCategoricalPolicyHead`]. A tanh-Gaussian
//! variant is a follow-up — continuous PPG is deferred until the discrete
//! variant is validated.

pub mod categorical;

pub use categorical::{PpgCategoricalPolicyHead, PpgCategoricalPolicyHeadConfig};
