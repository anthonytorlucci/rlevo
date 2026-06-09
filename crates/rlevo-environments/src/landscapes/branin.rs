//! Branin RCOS function No.01 — a 2-D benchmark with three equal global minima.
//!
//! ```text
//! f(x₁,x₂) = (x₂ − 5.1·x₁²/(4π²) + 5·x₁/π − 6)²
//!          + 10·(1 − 1/(8π))·cos(x₁) + 10
//! ```
//! Global minimum `f* ≈ 0.397887357729738` attained at three points that are
//! **not** related by symmetry: `(−π, 12.275)`, `(π, 2.275)`, `(3π, 2.475)`.
//! Differentiable; a canonical surrogate-modelling test.
//!
//! # Domain
//!
//! The true domain is asymmetric: `x₁ ∈ [-5, 10]`, `x₂ ∈ [0, 15]`.
//! [`bounds`](Branin::bounds) returns `(-5.0, 10.0)` (the `x₁` range) for the
//! renderer; the minimum `(π, 2.275)` falls inside that window. The evaluator
//! never clamps — the benchmark harness owns domain enforcement.

use std::f64::consts::PI;

/// Branin RCOS function No.01 (strictly 2-D).
#[derive(Debug, Clone, Copy)]
pub struct Branin;

impl Branin {
    /// Creates a Branin evaluator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Evaluate the Branin function at `(x1, x2)`.
    #[must_use]
    pub fn evaluate(&self, x1: f64, x2: f64) -> f64 {
        let a = x2 - 5.1 * x1 * x1 / (4.0 * PI * PI) + 5.0 * x1 / PI - 6.0;
        a * a + 10.0 * (1.0 - 1.0 / (8.0 * PI)) * x1.cos() + 10.0
    }

    /// Renderer-safe symmetric search domain (the `x₁` range). See the type-level
    /// docs for the full asymmetric domain.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-5.0, 10.0)
    }

    /// 2D projection used by the renderer — the exact surface for a 2-D function.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        self.evaluate(x, y)
    }
}

impl Default for Branin {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for Branin {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Branin",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Branin",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Certified global-minimum value (Branin & Hoo 1972).
    const F_OPT: f64 = 0.397_887_357_729_738;

    #[test]
    fn global_minimum_at_known_location() {
        assert_relative_eq!(Branin::new().evaluate(PI, 2.275), F_OPT, epsilon = 1e-4);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        assert!(
            Branin::new().evaluate(0.0, 0.0) > F_OPT,
            "value away from the three minima must exceed f*"
        );
    }

    #[test]
    fn three_global_minima_equal() {
        let b = Branin::new();
        assert_relative_eq!(b.evaluate(-PI, 12.275), F_OPT, epsilon = 1e-4);
        assert_relative_eq!(b.evaluate(PI, 2.275), F_OPT, epsilon = 1e-4);
        assert_relative_eq!(b.evaluate(3.0 * PI, 2.475), F_OPT, epsilon = 1e-4);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let b = Branin::new();
        let plain_no_trailing: String = b.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(b.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let b = Branin::new();
        let styled = b.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Branin")
            .expect("Branin label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let b = Branin::new();
        for line in b.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
