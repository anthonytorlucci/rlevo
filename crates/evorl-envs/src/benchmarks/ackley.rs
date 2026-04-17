//! Ackley function — a classical multimodal benchmark for EAs.
//!
//! `f(x) = -a·exp(-b·√(1/n · Σ x_i²)) − exp(1/n · Σ cos(c·x_i)) + a + e`
//! with canonical constants `a = 20`, `b = 0.2`, `c = 2π`. Global
//! minimum at `x = 0` where `f(0) = 0`. Commonly evaluated over
//! `[-32.768, 32.768]^n`; a narrower `[-5.12, 5.12]^n` window is
//! convenient when comparing against Sphere / Rastrigin on the same
//! axis.

use std::f64::consts::{E, PI};

#[derive(Debug, Clone, Copy)]
pub struct Ackley {
    pub dim: usize,
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Ackley {
    /// Build an Ackley with Ackley's canonical constants (`a=20`, `b=0.2`, `c=2π`).
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self {
            dim,
            a: 20.0,
            b: 0.2,
            c: 2.0 * PI,
        }
    }

    /// Evaluate the Ackley function at `x`. Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        let n = x.len() as f64;
        let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        let sum_cos: f64 = x.iter().map(|xi| (self.c * xi).cos()).sum();
        -self.a * (-self.b * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + self.a + E
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-32.768, 32.768)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_origin() {
        let a = Ackley::new(5);
        assert_relative_eq!(a.evaluate(&[0.0; 5]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_elsewhere() {
        let a = Ackley::new(3);
        assert!(a.evaluate(&[1.0, 2.0, 3.0]) > 0.0);
    }
}
