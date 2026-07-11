//! Alpine function No.01 — a non-smooth n-D benchmark from absolute values.
//!
//! `f(x) = Σ |x_i·sin(x_i) + 0.1·x_i|`, global minimum `f* = 0` at `x = 0`.
//! Non-negative everywhere and **non-smooth**: each dimension contributes
//! roughly eight kinks across `[-10, 10]` (one at the origin plus the seven
//! roots of `sin(x_i) = −0.1`), so gradient methods stall at the creases away
//! from the optimum.
//!
//! Evaluated over `[-10, 10]^n`. Requires `n ≥ 1`.
//!
//! Note: the canonical formula is `|x_i·sin(x_i) + 0.1·x_i|`; some libraries
//! (e.g. NiaPy) drop the leading `x_i` factor — that is a different function.

use rlevo_core::config::{self, ConfigError};

/// Alpine function No.01 evaluator with configurable dimensionality.
#[derive(Debug, Clone, Copy)]
pub struct Alpine1 {
    /// Number of input dimensions.
    dim: usize,
}

impl Alpine1 {
    /// Creates a `dim`-dimensional Alpine No.01 evaluator.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `dim == 0`: `f` is a sum of per-coordinate
    /// absolute values, so an empty coordinate set sums to `0` — the Alpine
    /// No.01 global minimum `f* = 0` — making every evaluation look optimal.
    pub fn new(dim: usize) -> Result<Self, ConfigError> {
        const C: &str = "Alpine1";
        config::nonzero(C, "dim", dim)?;
        Ok(Self { dim })
    }

    /// Number of input dimensions.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Evaluate the Alpine No.01 function at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        // non-differentiable at zero crossings of x*sin(x) + 0.1*x
        x.iter().map(|xi| (xi * xi.sin() + 0.1 * xi).abs()).sum()
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-10.0, 10.0)
    }

    /// 2D projection of [`evaluate`](Self::evaluate) for visualisation.
    ///
    /// Coordinates beyond the first two are held at `0.0` so the rendered slice
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

impl crate::render::AsciiRenderable for Alpine1 {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Alpine1",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Alpine1",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn dim_zero_is_rejected() {
        assert!(Alpine1::new(0).is_err(), "dim = 0 must not construct");
    }

    #[test]
    fn dim_accessor_reports_configured_dim() {
        assert_eq!(Alpine1::new(7).expect("dim >= 1").dim(), 7);
    }

    #[test]
    fn global_minimum_at_known_location() {
        let a = Alpine1::new(4).expect("dim >= 1");
        assert_relative_eq!(a.evaluate(&[0.0; 4]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        let a = Alpine1::new(3).expect("dim >= 1");
        assert!(
            a.evaluate(&[5.0, -3.0, 7.0]) > 0.0,
            "value away from the origin must exceed the minimum 0"
        );
    }

    #[test]
    fn non_negative_everywhere() {
        let a = Alpine1::new(5).expect("dim >= 1");
        for x in [1.0, -3.7, 5.5, -9.1, 0.01_f64] {
            assert!(
                a.evaluate(&[x; 5]) >= 0.0,
                "Alpine1 is a sum of absolute values and must be non-negative"
            );
        }
    }

    #[test]
    fn known_value_at_pi() {
        // At x = π: |π·sin(π) + 0.1·π| = 0.1π.
        let a = Alpine1::new(1).expect("dim >= 1");
        let expected = (PI * PI.sin() + 0.1 * PI).abs();
        assert_relative_eq!(a.evaluate(&[PI]), expected, epsilon = 1e-10);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let a = Alpine1::new(2).expect("dim >= 1");
        let plain_no_trailing: String = a.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(a.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let a = Alpine1::new(2).expect("dim >= 1");
        let styled = a.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Alpine1")
            .expect("Alpine1 label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let a = Alpine1::new(2).expect("dim >= 1");
        for line in a.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
