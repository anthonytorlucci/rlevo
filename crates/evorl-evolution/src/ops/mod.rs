//! Evolutionary operators.
//!
//! Operators are organized by role:
//!
//! - [`selection`] — parent selection (tournament, truncation).
//! - [`crossover`] — recombination (BLX-α, uniform).
//! - [`mutation`] — variation (Gaussian, Cauchy, uniform-reset).
//! - [`replacement`] — survivor selection (generational, elitist,
//!   (μ+λ), (μ,λ)).
//!
//! The [`kernels`] submodule houses custom CubeCL fused kernels (empty
//! until M4; see the `custom-kernels` crate feature).
//!
//! All operator functions take an explicit `&mut dyn RngCore` and a
//! `&B::Device` so the harness owns determinism. Operators never touch
//! thread-local or global RNG state.

pub mod crossover;
pub mod kernels;
pub mod mutation;
pub mod replacement;
pub mod selection;
