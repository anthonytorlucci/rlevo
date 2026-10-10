//! Evolutionary operators.
//!
//! Operators are organized by role:
//!
//! - [`selection`] — parent selection (tournament, truncation).
//! - [`crossover`] — recombination (BLX-`$\alpha$`, uniform).
//! - [`mutation`] — variation (Gaussian, Cauchy, uniform-reset).
//! - [`replacement`] — survivor selection (generational, elitist,
//!   (`$\mu$`+`$\lambda$`), (`$\mu$`,`$\lambda$`)).
//! - [`linalg`] — host-side dense linear algebra (Jacobi eigendecomposition,
//!   Cholesky) for the covariance-matrix strategies.
//!
//! All operator functions take an explicit `&mut dyn RngCore` and a
//! `&B::Device` so the harness owns determinism. Operators never touch
//! thread-local or global RNG state.

pub mod crossover;
pub mod linalg;
pub mod mutation;
pub mod replacement;
pub mod selection;
