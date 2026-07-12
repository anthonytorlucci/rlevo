//! Easom's function — a 2-D "needle in a haystack" benchmark.
//!
//! `f(x₁, x₂) = −cos(x₁)·cos(x₂)·exp(−(x₁−π)² − (x₂−π)²)`, global minimum
//! `f* = −1` (analytically exact) at `(π, π)`. The surface is essentially flat
//! (≈ 0) everywhere except inside a small basin around `(π, π)`: the
//! half-maximum contour has radius `√(ln 2) ≈ 0.833`, so gradient methods that
//! start outside the basin find no signal and fail. Differentiable everywhere.
//!
//! # Domain
//!
//! [`bounds`](Easom::bounds) returns `(-10.0, 10.0)`, which **is** the canonical
//! domain: Al-Roomi, *Unconstrained Single-Objective Benchmark Functions
//! Repository*, function #22 "Easom's Function", gives `−10 ≤ xᵢ ≤ 10` with
//! `f* = −1` at `(π, π)`. The optimum sits inside this window, and since
//! `|cos(x₁)·cos(x₂)| ≤ 1` and `exp(−(x₁−π)² − (x₂−π)²) ≤ 1`, no point anywhere
//! scores below `−1` — so the box satisfies both `bounds()` obligations (ADR 0045:
//! O1 reachability, O2 no spurious optimum), pinned by the unit tests below.
//!
//! **Source divergence (not a defect).** Molga & Smutnicki and the SFU virtual
//! library state the domain as `[-100, 100]²`. We follow Al-Roomi, the repository's
//! authoritative table. `[-10, 10]²` is therefore correct as written — do not
//! "fix" it to match a different source.

use std::f64::consts::PI;

/// Easom's function (strictly 2-D).
#[derive(Debug, Clone, Copy)]
pub struct Easom;

impl Easom {
    /// Creates an Easom evaluator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Evaluate Easom's function at `(x1, x2)`.
    #[must_use]
    pub fn evaluate(&self, x1: f64, x2: f64) -> f64 {
        -x1.cos() * x2.cos() * (-(x1 - PI).powi(2) - (x2 - PI).powi(2)).exp()
    }

    /// Recommended search box `(-10.0, 10.0)` applied per-coordinate — the canonical
    /// domain per Al-Roomi #22. Contains the optimum `(π, π)`. See the type-level
    /// docs for the divergence from sources that state `[-100, 100]²`.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-10.0, 10.0)
    }

    /// 2D projection used by the renderer — the exact surface for a 2-D function.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        self.evaluate(x, y)
    }
}

impl Default for Easom {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for Easom {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(|x, y| self.evaluate_2d(x, y), self.bounds(), "Easom")
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Easom",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_known_location() {
        assert_relative_eq!(Easom::new().evaluate(PI, PI), -1.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        // Away from (π, π) the value is ≈ 0, which exceeds the minimum −1.
        assert!(
            Easom::new().evaluate(0.0, 0.0) > -1.0,
            "value away from (π, π) must exceed the minimum −1"
        );
    }

    #[test]
    fn global_minimum_at_pi_pi() {
        assert_relative_eq!(Easom::new().evaluate(PI, PI), -1.0, epsilon = 1e-12);
    }

    #[test]
    fn near_zero_far_from_optimum() {
        let v = Easom::new().evaluate(0.0, 0.0);
        assert!(v.abs() < 1e-6, "expected near-zero at origin, got {v}");
    }

    /// Certified global-minimum value, analytically exact (Al-Roomi #22).
    const F_OPT: f64 = -1.0;
    /// The single certified global minimizer `(π, π)` (Al-Roomi #22).
    const OPTIMA: [(f64, f64); 1] = [(PI, PI)];

    /// O1 (reachability) — the search box reaches the certified global optimum.
    ///
    /// `bounds()` is one `(lo, hi)` pair applied to *every* coordinate, so `(π, π)`
    /// must lie in `[lo, hi]` on BOTH axes.
    #[test]
    fn bounds_box_contains_optimum_on_both_axes() {
        let (lo, hi) = Easom::new().bounds();
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

    /// O2 (no spurious optimum) — no point of the box scores below `f* = −1`.
    ///
    /// A deterministic 401×401 sweep of `bounds()²`. The bound `f ≥ −1` is exact
    /// (a product of factors each of modulus ≤ 1), so `eps` guards float error only
    /// on a unit-scale surface.
    #[test]
    fn no_point_in_bounds_beats_global_minimum() {
        const STEPS: u16 = 400;
        const EPS: f64 = 1e-9;

        let e = Easom::new();
        let (lo, hi) = e.bounds();
        let span = hi - lo;
        let n = f64::from(STEPS);

        for i in 0..=STEPS {
            let x1 = lo + span * f64::from(i) / n;
            for j in 0..=STEPS {
                let x2 = lo + span * f64::from(j) / n;
                let v = e.evaluate(x1, x2);
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

        let e = Easom::new();
        let plain_no_trailing: String = e.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(e.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let e = Easom::new();
        let styled = e.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Easom")
            .expect("Easom label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let e = Easom::new();
        for line in e.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
