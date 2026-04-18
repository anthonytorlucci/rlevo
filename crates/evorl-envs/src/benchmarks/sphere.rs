//! Sphere function — the simplest convex unimodal benchmark for EAs.
//!
//! `f(x) = Σ x_i²`, global minimum at `x = 0` where `f(0) = 0`.
//! Commonly evaluated over `[-5.12, 5.12]^n` for comparability with
//! Rastrigin / Ackley, though any symmetric domain works.

/// Sphere function evaluator with configurable dimensionality.
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    /// Number of input dimensions.
    pub dim: usize,
}

impl Sphere {
    /// Creates a `dim`-dimensional Sphere evaluator.
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Evaluate the Sphere function at `x`. Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        x.iter().map(|xi| xi * xi).sum()
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-5.12, 5.12)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_origin() {
        let s = Sphere::new(4);
        assert_relative_eq!(s.evaluate(&[0.0; 4]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_elsewhere() {
        let s = Sphere::new(3);
        assert_relative_eq!(s.evaluate(&[1.0, 2.0, 3.0]), 14.0, epsilon = 1e-12);
    }
}
