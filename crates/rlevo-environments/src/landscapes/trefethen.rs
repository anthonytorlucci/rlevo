//! Trefethen's function — a 2-D benchmark with five incommensurate frequencies.
//!
//! ```text
//! f(x₁,x₂) = e^{sin(50·x₁)} + sin(60·e^{x₂}) + sin(70·sin(x₁))
//!          + sin(sin(80·x₂)) − sin(10·(x₁+x₂)) + (x₁² + x₂²)/4
//! ```
//! Global minimum `f* ≈ −3.3069` at `(−0.0244, 0.2106)`. The frequencies
//! `{50, 60, 70, 80, 10}` share no ratio, so there is no periodic lattice of
//! equivalent basins and no dominant spatial scale — dense grid search plus
//! local refinement is needed. The stabilising `(x₁²+x₂²)/4` term keeps the
//! minimum interior and well-defined.
//!
//! # Domain
//!
//! The canonical competition domain is `[-1, 1]²`; the EA-friendly extended
//! domain is `x₁ ∈ [-6.5, 6.5]`, `x₂ ∈ [-4.5, 4.5]`.
//! [`bounds`](Trefethen::bounds) returns `(-4.5, 4.5)` (the tighter `x₂` range)
//! for the renderer. From Lloyd Trefethen's 2002 SIAM 100-Digit Challenge.

/// Trefethen's function (strictly 2-D).
#[derive(Debug, Clone, Copy)]
pub struct Trefethen;

impl Trefethen {
    /// Creates a Trefethen evaluator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Evaluate Trefethen's function at `(x1, x2)`.
    ///
    /// The `sin(60·e^{x₂})` term is finite across the recommended domain, but for
    /// `x₂ ≳ 710` the inner `e^{x₂}` overflows to infinity and the result is
    /// `NaN`; callers operating far outside the domain must clamp `x₂` themselves.
    #[must_use]
    pub fn evaluate(&self, x1: f64, x2: f64) -> f64 {
        (50.0 * x1).sin().exp()
            + (60.0 * x2.exp()).sin()
            + (70.0 * x1.sin()).sin()
            + (80.0 * x2).sin().sin()
            - (10.0 * (x1 + x2)).sin()
            + (x1 * x1 + x2 * x2) / 4.0
    }

    /// Renderer-safe symmetric search domain (the tighter `x₂` range). See the
    /// type-level docs for the full asymmetric domain.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-4.5, 4.5)
    }

    /// 2D projection used by the renderer — the exact surface for a 2-D function.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        self.evaluate(x, y)
    }
}

impl Default for Trefethen {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for Trefethen {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Trefethen",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Trefethen",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_known_location() {
        let t = Trefethen::new();
        assert_relative_eq!(t.evaluate(-0.024_403, 0.210_612), -3.3069, epsilon = 1e-3);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        let t = Trefethen::new();
        assert!(
            t.evaluate(6.0, 4.0) > t.evaluate(-0.024_403, 0.210_612),
            "value far from the optimum must exceed f*"
        );
    }

    #[test]
    fn global_minimum_approx() {
        let t = Trefethen::new();
        assert_relative_eq!(t.evaluate(-0.024_403, 0.210_612), -3.3069, epsilon = 1e-3);
    }

    #[test]
    fn quadratic_term_dominates_far_from_origin() {
        // At (6, 4) the quadratic (36+16)/4 = 13 dominates, so f > 10.
        let t = Trefethen::new();
        assert!(t.evaluate(6.0, 4.0) > 10.0);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let t = Trefethen::new();
        let plain_no_trailing: String = t.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(t.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let t = Trefethen::new();
        let styled = t.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Trefethen")
            .expect("Trefethen label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let t = Trefethen::new();
        for line in t.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
