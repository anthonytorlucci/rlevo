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
//! # Domain
//!
//! [`bounds`](CrossInTray::bounds) returns `(-15.0, 15.0)`, which **is** the
//! canonical domain: Al-Roomi, *Unconstrained Single-Objective Benchmark Functions
//! Repository*, function #44 "Cross-in-Tray Function", gives `−15 ≤ xᵢ ≤ 15` with
//! `f* ≈ −2.0626` at the four symmetric points `xᵢ* ≈ ±1.3494`. All four lie inside
//! the box and none of it scores below `f*`, so both `bounds()` obligations hold
//! (ADR 0045: O1 reachability, O2 no spurious optimum) — pinned by unit tests below.
//!
//! Over this window the inner exponential reaches ≈ `e⁹³`, but the outer
//! `0.0001·(·)^0.1` compresses it back — no `f64` overflow.

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

    /// Recommended search box `(-15.0, 15.0)` applied per-coordinate — the canonical
    /// domain per Al-Roomi #44. Contains all four global minimizers `(±1.3494…, ±1.3494…)`.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-15.0, 15.0)
    }

    /// 2D projection used by the renderer — the exact surface for a 2-D function.
    fn evaluate_2d(self, x: f64, y: f64) -> f64 {
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

    /// Certified global-minimum value (Mishra 2006; Al-Roomi #44 gives −2.0626).
    const F_OPT: f64 = -2.062_611_870_822_739;
    /// Per-axis magnitude of each of the four minimizers (Al-Roomi #44: ±1.3494).
    const X_OPT: f64 = 1.349_406_608_602_084;
    /// All four certified global minimizers — the sign combinations of `±X_OPT`.
    const OPTIMA: [(f64, f64); 4] = [
        (X_OPT, X_OPT),
        (X_OPT, -X_OPT),
        (-X_OPT, X_OPT),
        (-X_OPT, -X_OPT),
    ];

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

    /// O1 (reachability) — the search box reaches every certified global optimum.
    ///
    /// `bounds()` is one `(lo, hi)` pair applied to *every* coordinate, so ALL FOUR
    /// degenerate minima must lie in `[lo, hi]` on BOTH axes.
    #[test]
    fn bounds_box_contains_optimum_on_both_axes() {
        let (lo, hi) = CrossInTray::new().bounds();
        for (x1, x2) in OPTIMA {
            assert!(
                lo <= x1 && x1 <= hi,
                "x1 of optimum ({x1}, {x2}) outside bounds [{lo}, {hi}]"
            );
            assert!(
                lo <= x2 && x2 <= hi,
                "x2 of optimum ({x1}, {x2}) outside bounds [{lo}, {hi}]"
            );
        }
    }

    /// O2 (no spurious optimum) — no point of the box scores below `f*`.
    ///
    /// A deterministic 401×401 sweep of `bounds()²`. `eps` is looser here (`1e-6`,
    /// i.e. ~5e-7 relative on the `≈2.06` scale) than for the polynomial landscapes:
    /// the grid steps by `0.075`, which lands a sample almost exactly on a minimizer
    /// (`±1.35` vs `±1.34941`), so the true margin at the tightest sample is small —
    /// and the `exp(≈99) → (·)^0.1` path is where any float noise would appear.
    #[test]
    fn no_point_in_bounds_beats_global_minimum() {
        const STEPS: u16 = 400;
        const EPS: f64 = 1e-6;

        let c = CrossInTray::new();
        let (lo, hi) = c.bounds();
        let span = hi - lo;
        let n = f64::from(STEPS);

        for i in 0..=STEPS {
            let x1 = lo + span * f64::from(i) / n;
            for j in 0..=STEPS {
                let x2 = lo + span * f64::from(j) / n;
                let v = c.evaluate(x1, x2);
                assert!(
                    v >= F_OPT - EPS,
                    "f({x1}, {x2}) = {v} beats f* = {F_OPT}: bounds admit a spurious optimum"
                );
            }
        }
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
