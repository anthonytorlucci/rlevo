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
//! [`bounds`](Bukin6::bounds) returns a single `(lo, hi)` pair applied
//! per-coordinate (the consuming renderer and search harnesses use one box for
//! every axis), so it returns the *square bounding box* `(-15.0, 3.0)` of that
//! asymmetric domain. This is the smallest square that still contains the full
//! domain, the parabolic ridge, and the optimum `(−10, 1)` — a per-axis `x₁`
//! range like `(-15, -5)` would exclude both the ridge (`x₂ = 0.01·x₁² ≈ 0..2.25`)
//! and the optimum. The evaluator never clamps.
//!
//! The hull also admits points outside the published rectangle (e.g. `x₁ = +2`).
//! That is harmless: `f` is a sum of two non-negative terms, so `f ≥ 0` on all
//! of `ℝ²` and `f* = 0` is the global infimum — no widening of the box can admit
//! a point better than `f*`. Both obligations are pinned by unit tests
//! (`bounds_box_contains_optimum_on_both_axes` /
//! `bounds_box_contains_full_asymmetric_domain` for reachability, and
//! `no_point_in_bounds_beats_global_minimum` for the absence of a spurious
//! optimum).
//!
//! # References
//!
//! Al-Roomi, A.R. (2015), *Unconstrained Single-Objective Benchmark Functions
//! Repository*, Dalhousie University — function #52: the source of the
//! asymmetric domain `x₁ ∈ [-15, -5]`, `x₂ ∈ [-3, 3]` and of the global minimum
//! `f* = 0` at `(−10, 1)`. Corroborated by Surjanovic, S. & Bingham, D.,
//! *Virtual Library of Simulation Experiments: Test Functions and Datasets*,
//! Simon Fraser University.

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

    /// Square bounding box `(-15.0, 3.0)` of the asymmetric domain, applied
    /// per-coordinate. Contains the full domain, the parabolic ridge, and the
    /// optimum `(−10, 1)`. See the type-level docs for the true asymmetric
    /// domain.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-15.0, 3.0)
    }

    /// 2D projection used by the renderer — the exact surface for a 2-D function.
    fn evaluate_2d(self, x: f64, y: f64) -> f64 {
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
    fn bounds_box_contains_optimum_on_both_axes() {
        // The single `(lo, hi)` pair is applied per-coordinate, so the optimum
        // (-10, 1) must lie inside [lo, hi] on BOTH axes for search harnesses
        // to be able to reach it.
        let (lo, hi) = Bukin6::new().bounds();
        let (opt_x1, opt_x2) = (-10.0_f64, 1.0_f64);
        assert!(lo <= opt_x1 && opt_x1 <= hi, "x1 optimum outside bounds");
        assert!(lo <= opt_x2 && opt_x2 <= hi, "x2 optimum outside bounds");
    }

    #[test]
    fn bounds_box_contains_full_asymmetric_domain() {
        // The square box must cover x1 ∈ [-15, -5] and x2 ∈ [-3, 3].
        let (lo, hi) = Bukin6::new().bounds();
        assert!(lo <= -15.0 && hi >= -5.0, "x1 domain not covered");
        assert!(lo <= -3.0 && hi >= 3.0, "x2 domain not covered");
    }

    #[test]
    fn no_point_in_bounds_beats_global_minimum() {
        // O2 — no spurious optimum. f = 100·√|x₂ − 0.01·x₁²| + 0.01·|x₁ + 10| is a
        // sum of two non-negative terms, so f ≥ 0 identically on ℝ² and f* = 0 at
        // (−10, 1). The square hull deliberately admits points outside the published
        // rectangle x₁ ∈ [-15, -5], x₂ ∈ [-3, 3] (e.g. x₁ = +2, which the sweep below
        // visits) — this test is precisely the check that those extra points are
        // harmless: none of them can beat f*.
        const STEPS: i32 = 300;
        let b = Bukin6::new();
        let (lo, hi) = b.bounds();
        let step = (hi - lo) / f64::from(STEPS);

        for i in 0..=STEPS {
            let x1 = lo + step * f64::from(i);
            for j in 0..=STEPS {
                let x2 = lo + step * f64::from(j);
                let v = b.evaluate(x1, x2);
                assert!(
                    v >= -1e-12,
                    "f({x1}, {x2}) = {v} beats the global minimum f* = 0 inside bounds()"
                );
            }
        }
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
