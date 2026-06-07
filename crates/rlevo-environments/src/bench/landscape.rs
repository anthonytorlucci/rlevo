//! [`Landscape`] impls for the
//! numerical landscapes in [`crate::landscapes`].
//!
//! Feature-gated under `bench` so the trait impls compile only when the
//! harness adapter surface is in use.
//!
//! Each impl delegates to the landscape struct's own `evaluate` method, which
//! returns the raw scalar cost value (lower is better, following the
//! minimization convention used throughout `rlevo-evolution`).  These impls
//! make the landscapes usable directly with `rlevo-evolution`'s
//! `FromLandscape` adapter for evolutionary search benchmarking.

use rlevo_core::fitness::Landscape;

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
