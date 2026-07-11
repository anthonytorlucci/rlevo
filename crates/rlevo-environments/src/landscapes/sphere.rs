//! Sphere function — the simplest convex unimodal benchmark for EAs.
//!
//! `f(x) = Σ x_i²`, global minimum at `x = 0` where `f(0) = 0`.
//! Commonly evaluated over `[-5.12, 5.12]^n` for comparability with
//! Rastrigin / Ackley, though any symmetric domain works.

use rlevo_core::config::{self, ConfigError};

/// Sphere function evaluator with configurable dimensionality.
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    /// Number of input dimensions.
    dim: usize,
}

impl Sphere {
    /// Creates a `dim`-dimensional Sphere evaluator.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `dim == 0`: `f(x) = Σ x_i²` over an empty
    /// coordinate set is the empty sum `0`, which is exactly the Sphere's
    /// global optimum — a zero-dimensional instance would report itself solved
    /// on every evaluation.
    pub fn new(dim: usize) -> Result<Self, ConfigError> {
        const C: &str = "Sphere";
        config::nonzero(C, "dim", dim)?;
        Ok(Self { dim })
    }

    /// Number of input dimensions.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Evaluate the Sphere function at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`.
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

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for Sphere {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Sphere",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Sphere",
        )
    }
}

impl Sphere {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dim_zero_is_rejected() {
        assert!(Sphere::new(0).is_err(), "dim = 0 must not construct");
    }

    #[test]
    fn dim_accessor_reports_configured_dim() {
        assert_eq!(Sphere::new(7).expect("dim >= 1").dim(), 7);
    }

    #[test]
    fn global_minimum_at_origin() {
        let s = Sphere::new(4).expect("dim >= 1");
        assert_relative_eq!(s.evaluate(&[0.0; 4]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_elsewhere() {
        let s = Sphere::new(3).expect("dim >= 1");
        assert_relative_eq!(s.evaluate(&[1.0, 2.0, 3.0]), 14.0, epsilon = 1e-12);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let s = Sphere::new(2).expect("dim >= 1");
        let plain_no_trailing: String = s.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(s.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let s = Sphere::new(2).expect("dim >= 1");
        let styled = s.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Sphere")
            .expect("Sphere label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let s = Sphere::new(2).expect("dim >= 1");
        for line in s.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
