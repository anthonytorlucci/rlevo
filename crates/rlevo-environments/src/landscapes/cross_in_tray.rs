//! Cross-in-Tray function — a non-smooth 2-D benchmark with four equal minima.
//!
//! ```text
//! f(x₁,x₂) = −0.0001·(|sin(x₁)·sin(x₂)·exp(|100 − √(x₁²+x₂²)/π|)| + 1)^0.1
//! ```
//! Global minimum `f* ≈ −2.062611870822739` at the four sign combinations of
//! `(±1.349406608602084, ±1.349406608602084)`. The absolute value applied to the
//! oscillating product creates V-shaped kinks along the coordinate axes, giving
//! the function its cross pattern.
//!
//! Evaluated over `[-15, 15]²`. Over this window the inner exponential reaches
//! ≈ `e⁹³`, but the outer `0.0001·(·)^0.1` compresses it back — no `f64` overflow.

// non-differentiable at zero crossings of sin(x1)*sin(x2) (the cross ridges)

/// Cross-in-Tray function (strictly 2-D).
#[derive(Debug, Clone, Copy)]
pub struct CrossInTray;

impl CrossInTray {
    /// Creates a Cross-in-Tray evaluator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Evaluate the Cross-in-Tray function at `(x1, x2)`.
    #[must_use]
    pub fn evaluate(&self, x1: f64, x2: f64) -> f64 {
        use std::f64::consts::PI;
        let r = (x1 * x1 + x2 * x2).sqrt();
        let inner = (x1.sin() * x2.sin() * (100.0 - r / PI).abs().exp()).abs();
        -0.0001 * (inner + 1.0).powf(0.1)
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-15.0, 15.0)
    }

    /// 2D projection used by the renderer — the exact surface for a 2-D function.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        self.evaluate(x, y)
    }
}

impl Default for CrossInTray {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for CrossInTray {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "CrossInTray",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "CrossInTray",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Certified global-minimum value (Mishra 2006).
    const F_OPT: f64 = -2.062_611_870_822_739;
    /// Per-axis magnitude of each of the four minimizers.
    const X_OPT: f64 = 1.349_406_608_602_084;

    #[test]
    fn global_minimum_at_known_location() {
        assert_relative_eq!(
            CrossInTray::new().evaluate(X_OPT, X_OPT),
            F_OPT,
            epsilon = 1e-4
        );
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        // Every value is negative; the origin (f = −0.0001) exceeds the minimum.
        assert!(
            CrossInTray::new().evaluate(0.0, 0.0) > F_OPT,
            "value away from the four minima must exceed f*"
        );
    }

    #[test]
    fn four_global_minima_equal() {
        let c = CrossInTray::new();
        for (x1, x2) in [
            (X_OPT, X_OPT),
            (X_OPT, -X_OPT),
            (-X_OPT, X_OPT),
            (-X_OPT, -X_OPT),
        ] {
            assert_relative_eq!(c.evaluate(x1, x2), F_OPT, epsilon = 1e-4);
        }
    }

    #[test]
    fn negative_everywhere_in_domain() {
        assert!(CrossInTray::new().evaluate(0.0, 0.0) < 0.0);
        assert!(CrossInTray::new().evaluate(5.0, 7.0) < 0.0);
    }

    #[test]
    fn no_overflow_at_domain_corner() {
        // The inner exp reaches ≈ e^93 at the corner but stays finite after compression.
        let v = CrossInTray::new().evaluate(15.0, 15.0);
        assert!(
            v.is_finite(),
            "value must be finite at the domain corner, got {v}"
        );
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let c = CrossInTray::new();
        let plain_no_trailing: String = c.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(c.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let c = CrossInTray::new();
        let styled = c.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "CrossInTray")
            .expect("CrossInTray label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let c = CrossInTray::new();
        for line in c.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
