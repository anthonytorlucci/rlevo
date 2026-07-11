//! Rastrigin function — a classical multimodal benchmark for EAs.
//!
//! `f(x) = A·n + Σ (x_i² − A·cos(2π x_i))`, with `A = 10`, global minimum
//! at `x = 0` where `f(0) = 0`. Commonly evaluated over `[-5.12, 5.12]^n`.

use std::f64::consts::PI;

use rlevo_core::config::{self, ConfigError};

/// Rastrigin function evaluator with configurable dimensionality.
#[derive(Debug, Clone, Copy)]
pub struct Rastrigin {
    /// Number of input dimensions.
    dim: usize,
    /// Amplitude constant (canonical: `10.0`).
    pub a: f64,
}

impl Rastrigin {
    /// Creates a `dim`-dimensional Rastrigin evaluator with `A = 10`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `dim == 0`: the `A·n` offset collapses to `0`
    /// and the per-coordinate sum is empty, so `f` is identically `0` — the
    /// Rastrigin's global optimum — for every input.
    pub fn new(dim: usize) -> Result<Self, ConfigError> {
        const C: &str = "Rastrigin";
        config::nonzero(C, "dim", dim)?;
        Ok(Self { dim, a: 10.0 })
    }

    /// Number of input dimensions.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Evaluate the Rastrigin function at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`.
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

    /// 2D projection of [`evaluate`](Self::evaluate) for visualisation.
    ///
    /// For 2-D landscapes this is the exact surface; for higher dimensions
    /// the remaining coordinates are held at zero so the rendered slice
    /// passes through the global optimum at the origin.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        let mut p = vec![0.0_f64; self.dim];
        if !p.is_empty() {
            p[0] = x;
        }
        if p.len() >= 2 {
            p[1] = y;
        }
        self.evaluate(&p)
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for Rastrigin {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Rastrigin",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Rastrigin",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dim_zero_is_rejected() {
        assert!(Rastrigin::new(0).is_err(), "dim = 0 must not construct");
    }

    #[test]
    fn dim_accessor_reports_configured_dim() {
        assert_eq!(Rastrigin::new(7).expect("dim >= 1").dim(), 7);
    }

    #[test]
    fn global_minimum_at_origin() {
        let r = Rastrigin::new(5).expect("dim >= 1");
        assert_relative_eq!(r.evaluate(&[0.0; 5]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_elsewhere() {
        let r = Rastrigin::new(3).expect("dim >= 1");
        assert!(r.evaluate(&[1.0, 2.0, 3.0]) > 0.0);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let r = Rastrigin::new(2).expect("dim >= 1");
        let plain_no_trailing: String = r.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(r.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let r = Rastrigin::new(2).expect("dim >= 1");
        let styled = r.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Rastrigin")
            .expect("Rastrigin label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let r = Rastrigin::new(2).expect("dim >= 1");
        for line in r.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
