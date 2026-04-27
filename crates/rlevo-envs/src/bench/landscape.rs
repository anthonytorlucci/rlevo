//! [`Landscape`](rlevo_benchmarks::agent::Landscape) impls for the
//! numerical landscapes in [`crate::landscapes`].
//!
//! Feature-gated under `bench` so the trait dependency on
//! `rlevo-benchmarks` only appears when the harness is in use.

use rlevo_benchmarks::agent::Landscape;

use crate::landscapes::{ackley::Ackley, rastrigin::Rastrigin, sphere::Sphere};

impl Landscape for Sphere {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Sphere::evaluate(self, x)
    }
}

impl Landscape for Ackley {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Ackley::evaluate(self, x)
    }
}

impl Landscape for Rastrigin {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Rastrigin::evaluate(self, x)
    }
}
