//! Bukin function No.06 — a non-smooth 2-D benchmark with a knife-edge ridge.
//!
//! `f(x₁, x₂) = 100·√|x₂ − 0.01·x₁²| + 0.01·|x₁ + 10|`, global minimum `f* = 0`
//! at `(−10, 1)`. The minimum lies on a parabolic ridge `x₂ = 0.01·x₁²` of
//! effectively zero width — the transverse gradient diverges to `+∞` as the
//! ridge is approached — which is what makes standard gradient and DE methods
//! fail.
//!
//! # Domain
//!
//! The true domain is asymmetric: `x₁ ∈ [-15, -5]`, `x₂ ∈ [-3, 3]`.
//! [`bounds`](Bukin6::bounds) returns `(-15.0, -5.0)` (the `x₁` range) for the
//! renderer. The evaluator never clamps.

// non-differentiable on the parabolic ridge x2 = 0.01*x1^2 and at x1 = -10

/// Bukin function No.06 (strictly 2-D).
#[derive(Debug, Clone, Copy)]
pub struct Bukin6;

impl Bukin6 {
    /// Creates a Bukin No.06 evaluator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Evaluate the Bukin No.06 function at `(x1, x2)`.
    #[must_use]
    pub fn evaluate(&self, x1: f64, x2: f64) -> f64 {
        // `(x2 − 0.01·x1²).abs()` is ≥ 0 by construction, so the sqrt is safe;
        // the derivative is undefined on the ridge where the argument is zero.
        100.0 * (x2 - 0.01 * x1 * x1).abs().sqrt() + 0.01 * (x1 + 10.0).abs()
    }

    /// Renderer-safe symmetric search domain (the `x₁` range). See the type-level
    /// docs for the full asymmetric domain.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-15.0, -5.0)
    }

    /// 2D projection used by the renderer — the exact surface for a 2-D function.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        self.evaluate(x, y)
    }
}

impl Default for Bukin6 {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for Bukin6 {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Bukin6",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Bukin6",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_known_location() {
        assert_relative_eq!(Bukin6::new().evaluate(-10.0, 1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        assert!(
            Bukin6::new().evaluate(-12.0, 0.0) > 0.0,
            "value off the ridge must exceed the minimum 0"
        );
    }

    #[test]
    fn global_minimum_zero() {
        assert_relative_eq!(Bukin6::new().evaluate(-10.0, 1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn on_parabolic_ridge_partial_zero() {
        // On the ridge x2 = 0.01·x1² the sqrt term vanishes; only 0.01·|x1+10| remains.
        let x1 = -8.0_f64;
        let x2 = 0.01 * x1 * x1; // 0.64
        let expected = 0.01 * (x1 + 10.0).abs(); // 0.02
        assert_relative_eq!(Bukin6::new().evaluate(x1, x2), expected, epsilon = 1e-10);
    }

    #[test]
    fn positive_off_ridge() {
        assert!(Bukin6::new().evaluate(-12.0, 0.0) > 0.0);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let b = Bukin6::new();
        let plain_no_trailing: String = b.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(b.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let b = Bukin6::new();
        let styled = b.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Bukin6")
            .expect("Bukin6 label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let b = Bukin6::new();
        for line in b.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
