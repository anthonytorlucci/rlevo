//! Rastrigin function — a classical multimodal benchmark for EAs.
//!
//! `f(x) = A·n + Σ (x_i² − A·cos(2π x_i))`, with `A = 10`, global minimum
//! at `x = 0` where `f(0) = 0`. Commonly evaluated over `[-5.12, 5.12]^n`.

use std::f64::consts::PI;

/// Rastrigin function evaluator with configurable dimensionality.
#[derive(Debug, Clone, Copy)]
pub struct Rastrigin {
    /// Number of input dimensions.
    pub dim: usize,
    /// Amplitude constant (canonical: `10.0`).
    pub a: f64,
}

impl Rastrigin {
    /// Creates a `dim`-dimensional Rastrigin evaluator with `A = 10`.
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self { dim, a: 10.0 }
    }

    /// Evaluate the Rastrigin function at `x`. Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        let sum: f64 = x
            .iter()
            .map(|xi| xi * xi - self.a * (2.0 * PI * xi).cos())
            .sum();
        self.a * self.dim as f64 + sum
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
        let r = Rastrigin::new(5);
        assert_relative_eq!(r.evaluate(&[0.0; 5]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_elsewhere() {
        let r = Rastrigin::new(3);
        assert!(r.evaluate(&[1.0, 2.0, 3.0]) > 0.0);
    }
}
