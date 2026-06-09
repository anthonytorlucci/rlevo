//! Generalized Penalized Function No.01 (Yao 1999, f14) — sinusoidal lattice
//! with quartic boundary penalties.
//!
//! ```text
//! f(x) = (π/n)·{ 10·sin²(π·y_1)
//!               + Σ_{i=1}^{n-1} (y_i−1)²·[1 + 10·sin²(π·y_{i+1})]
//!               + (y_n−1)² }
//!        + Σ_{i=1}^{n} u(x_i)
//! ```
//! where `y_i = 1 + (x_i + 1)/4` and the penalty
//! `u(x) = 100·(x−10)⁴` for `x > 10`, `100·(−x−10)⁴` for `x < −10`, else `0`.
//!
//! Global minimum at `x_i = −1` (which maps to `y_i = 1`, zeroing every
//! sinusoidal term) where `f(x*) = 0`. Evaluated over `[-50, 50]^n` with the
//! soft penalty walls activating at `±10`.
//!
//! This is **not** the Levy function: the modern Levy uses `y_i = 1 + (x_i−1)/4`
//! with the optimum at `x* = +1` and no external penalty. Differentiable except
//! for a kink in `f''` at `x_i = ±10`.
//!
//! Requires `n ≥ 1`.

use std::f64::consts::PI;

/// Quartic boundary penalty `u(x)` with `a = 10`, `k = 100`, `m = 4`.
fn penalty(x: f64) -> f64 {
    // non-differentiable in f'' at x = ±10 (penalty activation) — explicit
    // piecewise branches are intentional; no smooth approximation.
    if x > 10.0 {
        100.0 * (x - 10.0).powi(4)
    } else if x < -10.0 {
        100.0 * (-x - 10.0).powi(4)
    } else {
        0.0
    }
}

/// Generalized Penalized Function No.01 evaluator with configurable dimensionality.
#[derive(Debug, Clone, Copy)]
pub struct Penalized1 {
    /// Number of input dimensions.
    pub dim: usize,
}

impl Penalized1 {
    /// Creates a `dim`-dimensional Penalized No.01 evaluator.
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Evaluate the Penalized No.01 function at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        let y: Vec<f64> = x.iter().map(|xi| 1.0 + (xi + 1.0) / 4.0).collect();

        let mut body = 10.0 * (PI * y[0]).sin().powi(2);
        for w in y.windows(2) {
            body += (w[0] - 1.0).powi(2) * (1.0 + 10.0 * (PI * w[1]).sin().powi(2));
        }
        body += (y[self.dim - 1] - 1.0).powi(2);

        let penalty_sum: f64 = x.iter().map(|&xi| penalty(xi)).sum();
        (PI / self.dim as f64) * body + penalty_sum
    }

    /// Recommended search domain for each coordinate (the full domain; the soft
    /// penalty walls sit at `±10`).
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-50.0, 50.0)
    }

    /// 2D projection of [`evaluate`](Self::evaluate) for visualisation.
    ///
    /// Coordinates beyond the first two are fixed at `−1.0` (the per-dimension
    /// optimum) so the rendered slice passes through the global minimum.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        let mut p = vec![-1.0_f64; self.dim];
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

impl crate::render::AsciiRenderable for Penalized1 {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Penalized1",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Penalized1",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_known_location() {
        // x_i = −1 ⇒ y_i = 1 ⇒ every sinusoidal term and penalty vanishes.
        let p = Penalized1::new(4);
        assert_relative_eq!(p.evaluate(&[-1.0; 4]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        let p = Penalized1::new(3);
        assert!(
            p.evaluate(&[0.0, 0.0, 0.0]) > 0.0,
            "Penalized1 must exceed its minimum away from (−1,…,−1)"
        );
    }

    #[test]
    fn global_minimum_at_neg_one() {
        let p = Penalized1::new(2);
        assert_relative_eq!(p.evaluate(&[-1.0, -1.0]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn penalty_activates_outside_soft_bounds() {
        let p = Penalized1::new(1);
        assert!(
            p.evaluate(&[11.0]) > p.evaluate(&[9.0]),
            "penalty must raise f beyond +10"
        );
        assert!(
            p.evaluate(&[-11.0]) > p.evaluate(&[-9.0]),
            "penalty must raise f below −10"
        );
    }

    #[test]
    fn penalty_zero_inside_soft_bounds() {
        // u(5.0) and u(−5.0) lie inside [−10, 10] and contribute nothing.
        assert_relative_eq!(penalty(5.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(penalty(-5.0), 0.0, epsilon = 1e-12);
        // u(11.0) = 100·(1)⁴ = 100.
        assert_relative_eq!(penalty(11.0), 100.0, epsilon = 1e-12);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let p = Penalized1::new(2);
        let plain_no_trailing: String = p.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(p.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let p = Penalized1::new(2);
        let styled = p.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Penalized1")
            .expect("Penalized1 label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let p = Penalized1::new(2);
        for line in p.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
