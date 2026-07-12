//! Goldstein-Price function — a 2-D benchmark with a six-order-of-magnitude range.
//!
//! ```text
//! f(x₁,x₂) = [1 + (x₁+x₂+1)²·(19 − 14x₁ + 3x₁² − 14x₂ + 6x₁x₂ + 3x₂²)]
//!          × [30 + (2x₁−3x₂)²·(18 − 32x₁ + 12x₁² + 48x₂ − 36x₁x₂ + 27x₂²)]
//! ```
//! Global minimum `f* = 3` (not `0`) at `(0, −1)`. Four local minima populate
//! the domain and the value ranges over `[3, ~1.3×10⁶]`, which makes the raw
//! form hostile to surrogate models. Differentiable everywhere.
//!
//! # Domain
//!
//! [`bounds`](GoldsteinPrice::bounds) returns `(-2.0, 2.0)`, which **is** the
//! canonical domain: Al-Roomi, *Unconstrained Single-Objective Benchmark Functions
//! Repository*, function #14 "Goldstein-Price's Function", gives `−2 ≤ xᵢ ≤ 2` with
//! `f* = 3` at `(0, −1)`. The optimum lies inside the box and no point of it scores
//! below `3`, so both `bounds()` obligations hold (ADR 0045: O1 reachability,
//! O2 no spurious optimum) — pinned by unit tests below.

/// Goldstein-Price function (strictly 2-D).
#[derive(Debug, Clone, Copy)]
pub struct GoldsteinPrice;

impl GoldsteinPrice {
    /// Creates a Goldstein-Price evaluator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Evaluate the Goldstein-Price function at `(x1, x2)`.
    #[must_use]
    pub fn evaluate(&self, x1: f64, x2: f64) -> f64 {
        let a = 1.0
            + (x1 + x2 + 1.0).powi(2)
                * (19.0 - 14.0 * x1 + 3.0 * x1 * x1 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2 * x2);
        let b = 30.0
            + (2.0 * x1 - 3.0 * x2).powi(2)
                * (18.0 - 32.0 * x1 + 12.0 * x1 * x1 + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2 * x2);
        a * b
    }

    /// Recommended search box `(-2.0, 2.0)` applied per-coordinate — the canonical
    /// domain per Al-Roomi #14. Contains the global minimizer `(0, −1)`.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-2.0, 2.0)
    }

    /// 2D projection used by the renderer — the exact surface for a 2-D function.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        self.evaluate(x, y)
    }
}

impl Default for GoldsteinPrice {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for GoldsteinPrice {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "GoldsteinPrice",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "GoldsteinPrice",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Certified global-minimum value (Al-Roomi #14).
    const F_OPT: f64 = 3.0;
    /// The single certified global minimizer `(0, −1)` (Al-Roomi #14).
    const OPTIMA: [(f64, f64); 1] = [(0.0, -1.0)];

    #[test]
    fn global_minimum_at_known_location() {
        assert_relative_eq!(
            GoldsteinPrice::new().evaluate(0.0, -1.0),
            3.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        assert!(
            GoldsteinPrice::new().evaluate(1.0, 1.0) > 3.0,
            "value must exceed the minimum 3 away from (0, −1)"
        );
    }

    #[test]
    fn global_minimum_at_zero_neg_one() {
        assert_relative_eq!(
            GoldsteinPrice::new().evaluate(0.0, -1.0),
            3.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn greater_than_minimum_elsewhere() {
        assert!(GoldsteinPrice::new().evaluate(1.0, 1.0) > 3.0);
    }

    /// O1 (reachability) — the search box reaches the certified global optimum.
    ///
    /// `bounds()` is one `(lo, hi)` pair applied to *every* coordinate, so `(0, −1)`
    /// must lie in `[lo, hi]` on BOTH axes.
    #[test]
    fn bounds_box_contains_optimum_on_both_axes() {
        let (lo, hi) = GoldsteinPrice::new().bounds();
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

    /// O2 (no spurious optimum) — no point of the box scores below `f* = 3`.
    ///
    /// A deterministic 401×401 sweep of `bounds()²`. The 400-step grid over `[-2, 2]`
    /// lands exactly on `(0, −1)`, so the tightest sample sits at `f*` itself; `eps`
    /// covers float error only. The surface spans `[3, ≈1.3e6]`, but near the minimum
    /// its scale is O(1), so an absolute `1e-9` is the right guard.
    #[test]
    fn no_point_in_bounds_beats_global_minimum() {
        const STEPS: u16 = 400;
        const EPS: f64 = 1e-9;

        let g = GoldsteinPrice::new();
        let (lo, hi) = g.bounds();
        let span = hi - lo;
        let n = f64::from(STEPS);

        for i in 0..=STEPS {
            let x1 = lo + span * f64::from(i) / n;
            for j in 0..=STEPS {
                let x2 = lo + span * f64::from(j) / n;
                let v = g.evaluate(x1, x2);
                assert!(
                    v >= F_OPT - EPS,
                    "f({x1}, {x2}) = {v} beats f* = {F_OPT}: bounds admit a spurious optimum"
                );
            }
        }
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let g = GoldsteinPrice::new();
        let plain_no_trailing: String = g.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(g.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let g = GoldsteinPrice::new();
        let styled = g.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "GoldsteinPrice")
            .expect("GoldsteinPrice label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let g = GoldsteinPrice::new();
        for line in g.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
