//! Needle-Eye function — extreme-precision failure-demonstration benchmark.
//!
//! ```text
//! f(x) = 1                                       if |x_i| < 1e-4 for all i
//!      = Σ (100 + |x_i|) · 1[|x_i| > 1e-4]        otherwise
//! ```
//! Global minimum `f* = 1` on a hypercube of side `2×10⁻⁴` centred at the
//! origin. Outside that needle the value is `≥ 100`. Piecewise constant and
//! **non-differentiable**; there is no gradient information to exploit.
//!
//! Evaluated over `[-10, 10]^n`. For `n ≥ 2` the optimal region is
//! astronomically small (volume fraction `≈ 10⁻¹⁰` at `n = 2`), so this is a
//! failure-demonstration function, not a solvable test. Requires `n ≥ 1`.

/// Needle-Eye function evaluator with configurable dimensionality.
#[derive(Debug, Clone, Copy)]
pub struct Needle {
    /// Number of input dimensions.
    pub dim: usize,
}

impl Needle {
    /// Half-width of the optimal needle: `|x_i| ≤ EYE` is "inside".
    pub const EYE: f64 = 1e-4;

    /// Creates a `dim`-dimensional Needle-Eye evaluator.
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Evaluate the Needle-Eye function at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        // non-differentiable; gradient-based agents cannot optimize this function.
        // `<= EYE` (matching the "inside the eye" boundary in the docs) so that a
        // point exactly on the boundary returns the minimum f = 1 rather than
        // falling through to an empty penalty sum (which would yield 0 < f*).
        if x.iter().all(|xi| xi.abs() <= Self::EYE) {
            1.0
        } else {
            x.iter()
                .filter(|xi| xi.abs() > Self::EYE)
                .map(|xi| 100.0 + xi.abs())
                .sum()
        }
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-10.0, 10.0)
    }

    /// 2D projection of [`evaluate`](Self::evaluate) for visualisation.
    ///
    /// Coordinates beyond the first two are held at `0.0` (inside the needle).
    /// The rendered surface is uniformly dense — the optimal patch is invisible
    /// at any reasonable grid resolution, which documents the function's difficulty.
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

impl crate::render::AsciiRenderable for Needle {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Needle",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Needle",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_known_location() {
        // Inside the needle, f = 1 (the minimum).
        let n = Needle::new(3);
        assert_relative_eq!(n.evaluate(&[0.0, 0.0, 0.0]), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        let n = Needle::new(1);
        assert!(
            n.evaluate(&[1.0]) > 1.0,
            "value outside the needle must exceed the minimum 1"
        );
    }

    #[test]
    fn inside_eye_gives_one() {
        let n = Needle::new(3);
        assert_relative_eq!(n.evaluate(&[0.0, 0.0, 0.0]), 1.0, epsilon = 1e-12);
        assert_relative_eq!(n.evaluate(&[5e-5, -5e-5, 0.0]), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn outside_eye_gives_penalty() {
        let n = Needle::new(1);
        // |1.0| > EYE, so f = 100 + 1.0 = 101.0.
        assert_relative_eq!(n.evaluate(&[1.0]), 101.0, epsilon = 1e-10);
    }

    #[test]
    fn exact_boundary_stays_at_minimum() {
        // A point exactly on the boundary |x_i| = EYE is inside the eye, so
        // f = 1 — never below the global minimum.
        let n = Needle::new(2);
        assert_relative_eq!(n.evaluate(&[Needle::EYE, -Needle::EYE]), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn just_outside_eye_gives_penalty() {
        let n = Needle::new(1);
        let x = 1.1e-4_f64; // just outside the eye
        assert!(n.evaluate(&[x]) > 1.0);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let n = Needle::new(2);
        let plain_no_trailing: String = n.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(n.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let n = Needle::new(2);
        let styled = n.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Needle")
            .expect("Needle label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let n = Needle::new(2);
        for line in n.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
