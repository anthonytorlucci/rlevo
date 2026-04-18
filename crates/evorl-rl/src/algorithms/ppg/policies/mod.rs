//! Built-in PPG policy heads.
//!
//! v1 ships a discrete-only [`PpgCategoricalPolicyHead`]. A tanh-Gaussian
//! variant is a follow-up — matching the spec scope that defers continuous
//! PPG until a discrete shakedown passes.

pub mod categorical;

pub use categorical::{PpgCategoricalPolicyHead, PpgCategoricalPolicyHeadConfig};
