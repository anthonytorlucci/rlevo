//! Griewank function — a multimodal benchmark with a dense lattice of local minima.
//!
//! `f(x) = (1/4000)·Σ x_i² − Π cos(x_i / √i) + 1` (with `i` 1-based), global
//! minimum at `x = 0` where `f(0) = 0`. The quadratic bowl and the product of
//! cosines interact to produce many regularly-spaced local minima.
//!
//! Commonly evaluated over `[-600, 600]^n` (used here). Counterintuitively the
//! function becomes *easier* at higher `n` because the quadratic term dominates
//! the cosine perturbation; it is most discriminating at `n = 5–15`. A narrower
//! `[-10, 10]` window shows the local structure better than the full domain.
//!
//! Requires `n ≥ 1`.

use rlevo_core::config::{self, ConfigError};

/// Griewank function evaluator with configurable dimensionality.
#[derive(Debug, Clone, Copy)]
pub struct Griewank {
    /// Number of input dimensions.
    dim: usize,
}

impl Griewank {
    /// Creates a `dim`-dimensional Griewank evaluator.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `dim == 0`: the quadratic term is an empty sum
    /// (`0`) and the cosine term an empty product (`1`), so `f = 0 − 1 + 1 = 0`
    /// — precisely the Griewank's global minimum — for every input.
    pub fn new(dim: usize) -> Result<Self, ConfigError> {
        const C: &str = "Griewank";
        config::nonzero(C, "dim", dim)?;
        Ok(Self { dim })
    }

    /// Number of input dimensions.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Evaluate the Griewank function at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        let sum: f64 = x.iter().map(|xi| xi * xi).sum::<f64>() / 4000.0;
        let prod: f64 = x
            .iter()
            .enumerate()
            .map(|(idx, xi)| {
                let i = (idx + 1) as f64; // 1-based index per the canonical definition
                (xi / i.sqrt()).cos()
            })
            .product();
        sum - prod + 1.0
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-600.0, 600.0)
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

impl crate::render::AsciiRenderable for Griewank {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Griewank",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Griewank",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dim_zero_is_rejected() {
        assert!(Griewank::new(0).is_err(), "dim = 0 must not construct");
    }

    #[test]
    fn dim_accessor_reports_configured_dim() {
        assert_eq!(Griewank::new(7).expect("dim >= 1").dim(), 7);
    }

    #[test]
    fn global_minimum_at_known_location() {
        // sum_sq = 0, prod_cos = 1, so f(0) = 0 − 1 + 1 = 0 exactly.
        let g = Griewank::new(5).expect("dim >= 1");
        assert_relative_eq!(g.evaluate(&[0.0; 5]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        let g = Griewank::new(3).expect("dim >= 1");
        assert!(
            g.evaluate(&[100.0, -50.0, 25.0]) > 0.0,
            "Griewank must exceed its minimum away from the origin"
        );
    }

    #[test]
    fn known_value_at_ones() {
        // f([1,1]) = 2/4000 − cos(1)·cos(1/√2) + 1.
        let g = Griewank::new(2).expect("dim >= 1");
        let expected = 2.0 / 4000.0 - (1.0_f64).cos() * (1.0_f64 / 2.0_f64.sqrt()).cos() + 1.0;
        assert_relative_eq!(g.evaluate(&[1.0, 1.0]), expected, epsilon = 1e-12);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let g = Griewank::new(2).expect("dim >= 1");
        let plain_no_trailing: String = g.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(g.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let g = Griewank::new(2).expect("dim >= 1");
        let styled = g.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Griewank")
            .expect("Griewank label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let g = Griewank::new(2).expect("dim >= 1");
        for line in g.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
