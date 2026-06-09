//! Rosenbrock function — the classical "banana" valley benchmark for EAs.
//!
//! `f(x) = Σ_{i=1}^{n-1} [100·(x_{i+1} − x_i²)² + (x_i − 1)²]`, global minimum
//! at `x = (1, …, 1)` where `f(x*) = 0`. The minimum lies at the bottom of a
//! long, narrow, curved valley following `x_{i+1} = x_i²`; the surface is smooth
//! everywhere but the Hessian is nearly singular along the valley floor, so the
//! landscape looks deceptively flat in the render.
//!
//! Commonly evaluated over `[-30, 30]^n` (used here); de Jong's original used
//! `[-2.048, 2.048]` and the BBOB/COCO convention is `[-5, 10]`. For `n ≥ 4` the
//! function has exactly two local minima — the global at `(1, …, 1)` and a local
//! near `(-1, 1, …, 1)` — so restartless optimizers can stall at the wrong basin.
//!
//! Requires `n ≥ 2`; the sum is empty for `n = 1`.

/// Rosenbrock function evaluator with configurable dimensionality.
///
/// The `100` and `1` coefficients are fixed per the canonical definition, so the
/// struct carries no tunable constants.
#[derive(Debug, Clone, Copy)]
pub struct Rosenbrock {
    /// Number of input dimensions. Must be `≥ 2`.
    pub dim: usize,
}

impl Rosenbrock {
    /// Creates a `dim`-dimensional Rosenbrock evaluator.
    ///
    /// `dim` should be `≥ 2`; [`evaluate`](Self::evaluate) panics otherwise.
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Evaluate the Rosenbrock function at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`, or if `self.dim < 2` (the pairwise sum is
    /// undefined for a single coordinate).
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        assert!(self.dim >= 2, "Rosenbrock requires dim >= 2");
        x.windows(2)
            .map(|w| {
                let (xi, xn) = (w[0], w[1]);
                100.0 * (xn - xi * xi).powi(2) + (xi - 1.0).powi(2)
            })
            .sum()
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-30.0, 30.0)
    }

    /// 2D projection of [`evaluate`](Self::evaluate) for visualisation.
    ///
    /// Coordinates beyond the first two are fixed at `1.0` (the per-dimension
    /// optimum) so the rendered slice passes through the valley floor rather
    /// than a flat cross-section.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        let mut p = vec![1.0_f64; self.dim];
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

impl crate::render::AsciiRenderable for Rosenbrock {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Rosenbrock",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Rosenbrock",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_known_location() {
        let r = Rosenbrock::new(4);
        assert_relative_eq!(r.evaluate(&[1.0; 4]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        let r = Rosenbrock::new(3);
        assert!(
            r.evaluate(&[0.0, 0.0, 0.0]) > 0.0,
            "Rosenbrock must be positive away from (1,…,1)"
        );
    }

    #[test]
    fn minimum_at_ones() {
        // Every term vanishes: 100·(1−1)² + (1−1)² = 0.
        let r = Rosenbrock::new(5);
        assert_relative_eq!(r.evaluate(&[1.0; 5]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn known_value_at_origin() {
        // Each of the n−1 terms contributes 100·0 + 1 = 1, so f(0) = n−1.
        let r = Rosenbrock::new(4);
        assert_relative_eq!(r.evaluate(&[0.0; 4]), 3.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "requires dim >= 2")]
    fn panics_on_dim_one() {
        let r = Rosenbrock::new(1);
        let _ = r.evaluate(&[0.5]);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let r = Rosenbrock::new(2);
        let plain_no_trailing: String = r.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(r.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let r = Rosenbrock::new(2);
        let styled = r.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Rosenbrock")
            .expect("Rosenbrock label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let r = Rosenbrock::new(2);
        for line in r.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
