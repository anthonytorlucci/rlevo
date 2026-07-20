//! Branin RCOS function No.01 — a 2-D benchmark with three equal global minima.
//!
//! ```text
//! f(x₁,x₂) = (x₂ − 5.1·x₁²/(4π²) + 5·x₁/π − 6)²
//!          + 10·(1 − 1/(8π))·cos(x₁) + 10
//! ```
//! Global minimum `f* = 10/(8π) ≈ 0.397887357729738` attained at three points
//! that are **not** related by symmetry: `(−π, 12.275)`, `(π, 2.275)`,
//! `(3π, 2.475)`. Differentiable; a canonical surrogate-modelling test.
//!
//! # Domain
//!
//! The published domain is asymmetric: `x₁ ∈ [-5, 10]`, `x₂ ∈ [0, 15]`.
//! [`bounds`](Branin::bounds) returns a single `(lo, hi)` pair applied
//! per-coordinate (the consuming renderer and every search harness hold one
//! scalar box and apply it to every axis), so it returns the *square hull*
//! `(-5.0, 15.0)` of that asymmetric domain — the smallest square containing
//! both `[-5, 10]` and `[0, 15]`.
//!
//! The hull is required for **reachability**: the minimum `(−π, 12.275)` has
//! `x₂ = 12.275 > 10`, so a `(-5.0, 10.0)` box would place one of the three
//! certified minima outside the search space, where no optimiser could ever
//! find it.
//!
//! The hull also admits points outside the published rectangle (e.g.
//! `x₂ ∈ (10, 15]` paired with `x₁ ∈ (10, 15]`, or `x₂ < 0`). That is harmless:
//! `f* = 10/(8π)` is the global **infimum of Branin over all of ℝ²** (see the
//! derivation on [`bounds`](Branin::bounds)), so no widening of the box can
//! admit a point better than `f*`. Widening buys the third certified minimum
//! and nothing else. The evaluator never clamps — the benchmark harness owns
//! domain enforcement.
//!
//! # References
//!
//! Dixon, L.C.W. & Szegő, G.P. (1978), "The global optimization problem: an
//! introduction", *Towards Global Optimization 2*, pp. 1–15, North-Holland —
//! the source of the asymmetric domain and the three certified minima.

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

    /// Square hull `(-5.0, 15.0)` of the asymmetric domain, applied
    /// per-coordinate. Contains all three certified minima — `(−π, 12.275)`,
    /// `(π, 2.275)`, `(3π, 2.475)` — on **both** axes, and admits no point
    /// better than `f*`. See the type-level docs for the published asymmetric
    /// domain.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        // Why widening from the x1 range (-5, 10) to the hull (-5, 15) is safe:
        //
        //   f(x1,x2) = a² + 10·(1 − 1/(8π))·cos(x1) + 10,
        //   where a = x2 − 5.1·x1²/(4π²) + 5·x1/π − 6.
        //
        // a² ≥ 0 and cos(x1) ≥ −1, so for every (x1, x2) ∈ ℝ²:
        //
        //   f ≥ 0 + 10·(1 − 1/(8π))·(−1) + 10
        //     = 10 − 10·(1 − 1/(8π))
        //     = 10/(8π)
        //     = 0.397887357729738…  =  f*
        //
        // i.e. f* is the global *infimum of Branin over all of ℝ²*, not merely
        // over the published rectangle. Enlarging the search box therefore can
        // never expose a point with f < f* (obligation O2). It exposes exactly
        // the third certified minimum (−π, 12.275), whose x2 = 12.275 lies
        // above the x1 range's upper end of 10 (obligation O1).
        (-5.0, 15.0)
    }

    /// 2D projection used by the renderer — the exact surface for a 2-D function.
    fn evaluate_2d(self, x: f64, y: f64) -> f64 {
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
    fn bounds_box_contains_all_optima_on_both_axes() {
        // O1 — reachability. The single `(lo, hi)` pair is applied per-coordinate,
        // so EVERY certified minimum must lie inside [lo, hi] on BOTH axes for a
        // search harness to be able to reach it. The historical `(-5.0, 10.0)` box
        // failed here: (−π, 12.275) has x2 = 12.275 > 10.
        let (lo, hi) = Branin::new().bounds();
        for (opt_x1, opt_x2) in [(-PI, 12.275_f64), (PI, 2.275_f64), (3.0 * PI, 2.475_f64)] {
            assert!(
                lo <= opt_x1 && opt_x1 <= hi,
                "x1 optimum {opt_x1} outside bounds [{lo}, {hi}]"
            );
            assert!(
                lo <= opt_x2 && opt_x2 <= hi,
                "x2 optimum {opt_x2} outside bounds [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn bounds_box_contains_full_asymmetric_domain() {
        // O1 — the square hull must cover x1 ∈ [-5, 10] and x2 ∈ [0, 15].
        let (lo, hi) = Branin::new().bounds();
        assert!(lo <= -5.0 && hi >= 10.0, "x1 domain not covered");
        assert!(lo <= 0.0 && hi >= 15.0, "x2 domain not covered");
    }

    #[test]
    fn no_point_in_bounds_beats_global_minimum() {
        // O2 — no spurious optimum. Widening the box to the square hull must not
        // admit any point with f < f*. Guaranteed analytically (f* = 10/(8π) is the
        // infimum of Branin over all of ℝ² — see the comment on `bounds`); this is
        // the deterministic empirical check over a dense grid of the returned box.
        const CELLS: i32 = 400;

        let b = Branin::new();
        let (lo, hi) = b.bounds();
        let step = (hi - lo) / f64::from(CELLS);

        for i in 0..=CELLS {
            let x1 = f64::from(i).mul_add(step, lo);
            for j in 0..=CELLS {
                let x2 = f64::from(j).mul_add(step, lo);
                let f = b.evaluate(x1, x2);
                assert!(
                    f >= F_OPT - 1e-9,
                    "point ({x1}, {x2}) in bounds evaluates to {f} < f* = {F_OPT}"
                );
            }
        }
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
