//! Deb's function No.01 — `10^n` degenerate global minima of equal value.
//!
//! `f(x) = −(1/n)·Σ sin⁶(5π·x_i)`, global minimum `f* = −1` attained whenever
//! every `x_i ∈ {−0.9, −0.7, …, −0.1, 0.1, …, 0.9}` — ten optima per dimension,
//! `10^n` in total. The function is separable and differentiable; it tests
//! whether convergence metrics and population-diversity handling cope with
//! non-unique solutions.
//!
//! Evaluated over `[-1, 1]^n`. (The CEC 2013 `[0, 1]` "Equal Maxima" variant is a
//! different function with `5^n` optima.) Because the optima count explodes,
//! meaningful benchmarking is restricted to `n ≤ 2`. Requires `n ≥ 1`.

use std::f64::consts::PI;

use rlevo_core::config::{self, ConfigError};

/// Deb's function No.01 evaluator with configurable dimensionality.
#[derive(Debug, Clone, Copy)]
pub struct Deb1 {
    /// Number of input dimensions.
    dim: usize,
}

impl Deb1 {
    /// Creates a `dim`-dimensional Deb No.01 evaluator.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `dim == 0`: the `1/n` averaging factor divides
    /// the empty sum by zero, so evaluation yields `0.0 / 0.0` = `NaN` rather
    /// than a comparable fitness value.
    pub fn new(dim: usize) -> Result<Self, ConfigError> {
        const C: &str = "Deb1";
        config::nonzero(C, "dim", dim)?;
        Ok(Self { dim })
    }

    /// Number of input dimensions.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Evaluate Deb's function No.01 at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        let sum: f64 = x.iter().map(|xi| (5.0 * PI * xi).sin().powi(6)).sum();
        -sum / self.dim as f64
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-1.0, 1.0)
    }

    /// 2D projection of [`evaluate`](Self::evaluate) for visualisation.
    ///
    /// Coordinates beyond the first two are fixed at `0.1`, one of the
    /// per-dimension optima, so the rendered slice passes through a global minimum.
    fn evaluate_2d(self, x: f64, y: f64) -> f64 {
        let mut p = vec![0.1_f64; self.dim];
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

impl crate::render::AsciiRenderable for Deb1 {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(|x, y| self.evaluate_2d(x, y), self.bounds(), "Deb1")
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(|x, y| self.evaluate_2d(x, y), self.bounds(), "Deb1")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dim_zero_is_rejected() {
        assert!(Deb1::new(0).is_err(), "dim = 0 must not construct");
    }

    #[test]
    fn dim_accessor_reports_configured_dim() {
        assert_eq!(Deb1::new(7).expect("dim >= 1").dim(), 7);
    }

    #[test]
    fn global_minimum_at_known_location() {
        // x_i = 0.1 ⇒ sin(π/2)⁶ = 1 in every dimension ⇒ f = −1.
        let d = Deb1::new(3).expect("dim >= 1");
        assert_relative_eq!(d.evaluate(&[0.1; 3]), -1.0, epsilon = 1e-10);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        // f ∈ [−1, 0]; a non-optimal point exceeds the minimum −1.
        let d = Deb1::new(1).expect("dim >= 1");
        assert!(
            d.evaluate(&[0.15]) > -1.0,
            "non-optimal point must exceed the minimum −1"
        );
    }

    #[test]
    fn global_minimum_at_sample_optima() {
        let d = Deb1::new(3).expect("dim >= 1");
        assert_relative_eq!(d.evaluate(&[0.1, 0.1, 0.1]), -1.0, epsilon = 1e-10);
        assert_relative_eq!(d.evaluate(&[-0.9, 0.3, 0.7]), -1.0, epsilon = 1e-10);
    }

    #[test]
    fn local_minimum_shallower() {
        let d = Deb1::new(1).expect("dim >= 1");
        assert!(d.evaluate(&[0.15]) > -1.0);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let d = Deb1::new(2).expect("dim >= 1");
        let plain_no_trailing: String = d.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(d.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let d = Deb1::new(2).expect("dim >= 1");
        let styled = d.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Deb1")
            .expect("Deb1 label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let d = Deb1::new(2).expect("dim >= 1");
        for line in d.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
