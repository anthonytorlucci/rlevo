//! Concrete evolutionary algorithms.
//!
//! - [`ga`] — Genetic Algorithm (real- and binary-coded).
//! - [`es_classical`] — (1+1), (1+λ), (μ,λ), (μ+λ) Evolution Strategies.
//!
//! Future modules (EP, DE, CGP) land in later milestones.

pub mod de;
pub mod ep;
pub mod es_classical;
pub mod ga;
pub mod ga_binary;
pub mod gp_cgp;
