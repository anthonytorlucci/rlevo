//! Six-Hump Camel-Back function — a 2-D polynomial benchmark with two equal minima.
//!
//! `f(x₁, x₂) = 4x₁² − 2.1x₁⁴ + x₁⁶/3 + x₁x₂ − 4x₂² + 4x₂⁴`. The surface has six
//! local minima in three point-symmetric pairs; exactly **two** are global, at
//! `(±0.08984…, ∓0.71266…)`, with `f* ≈ −1.031628…`. A pure polynomial,
//! differentiable everywhere, satisfying `f(−x₁, −x₂) = f(x₁, x₂)`.
//!
//! # Domain
//!
//! The canonical domain is `[-5, 5]²` (Al-Roomi, *Unconstrained Single-Objective
//! Benchmark Functions Repository*, function #23 "Six-Hump Camel-Back Function").
//! [`bounds`](SixHumpCamel::bounds) deliberately returns the **reduced range**
//! `(-2.0, 2.0)` — the narrow window that shows the six-hump structure, which is
//! what makes the surface worth rendering and searching.
//!
//! The reduction conforms to the `bounds()` contract (ADR 0045): `bounds()` is
//! the *recommended per-coordinate search box*, not the mathematical domain, and
//! it must (O1) contain every certified global optimum on every coordinate and
//! (O2) contain no point scoring below `f*`. Both hold here — the two global
//! minima `(±0.08984…, ∓0.71266…)` lie well inside `[-2, 2]²`, and `f* ≈ −1.031628`
//! is the global infimum of a coercive polynomial, so no point of the smaller box
//! can beat it. Both obligations are pinned by unit tests below.

/// Six-Hump Camel-Back function (strictly 2-D).
#[derive(Debug, Clone, Copy)]
pub struct SixHumpCamel;

impl SixHumpCamel {
    /// Creates a Six-Hump Camel-Back evaluator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Evaluate the Six-Hump Camel-Back function at `(x1, x2)`.
    #[must_use]
    pub fn evaluate(&self, x1: f64, x2: f64) -> f64 {
        // `x1.powi(6) / 3.0` avoids the rounding error of multiplying by 1.0/3.0.
        4.0 * x1 * x1 - 2.1 * x1.powi(4) + x1.powi(6) / 3.0 + x1 * x2 - 4.0 * x2 * x2
            + 4.0 * x2.powi(4)
    }

    /// Recommended search box `(-2.0, 2.0)` applied per-coordinate — a deliberate
    /// *reduced range* of the canonical `[-5, 5]²` domain, chosen because it frames
    /// the six-hump structure. Both global minima lie inside it and no point of the
    /// box scores below `f*`. See the type-level docs.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-2.0, 2.0)
    }

    /// 2D projection used by the renderer — the exact surface for a 2-D function.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        self.evaluate(x, y)
    }
}

impl Default for SixHumpCamel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for SixHumpCamel {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "SixHumpCamel",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "SixHumpCamel",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Certified global-minimum value (Dixon & Szego 1978; Al-Roomi #23 gives −1.03163).
    const F_OPT: f64 = -1.031_628_453_489_877;

    /// Both certified global minimizers `(±0.08984…, ∓0.71266…)` (Al-Roomi #23).
    const OPTIMA: [(f64, f64); 2] = [
        (0.089_842_013_683_013_31, -0.712_656_403_270_413_5),
        (-0.089_842_013_683_013_31, 0.712_656_403_270_413_5),
    ];

    #[test]
    fn global_minimum_at_known_location() {
        let c = SixHumpCamel::new();
        assert_relative_eq!(
            c.evaluate(0.089_842_013_683_013_31, -0.712_656_403_270_413_5),
            F_OPT,
            epsilon = 1e-12
        );
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        let c = SixHumpCamel::new();
        assert!(
            c.evaluate(1.5, 1.5) > F_OPT,
            "value away from the basins must exceed the global minimum"
        );
    }

    #[test]
    fn two_global_minima_equal() {
        let c = SixHumpCamel::new();
        let v1 = c.evaluate(0.089_842_013_683_013_31, -0.712_656_403_270_413_5);
        let v2 = c.evaluate(-0.089_842_013_683_013_31, 0.712_656_403_270_413_5);
        assert_relative_eq!(v1, F_OPT, epsilon = 1e-12);
        assert_relative_eq!(v2, F_OPT, epsilon = 1e-12);
    }

    #[test]
    fn point_symmetry() {
        let c = SixHumpCamel::new();
        assert_relative_eq!(
            c.evaluate(1.3, -0.7),
            c.evaluate(-1.3, 0.7),
            epsilon = 1e-12
        );
    }

    /// O1 (reachability) — the search box reaches every certified global optimum.
    ///
    /// `bounds()` is one `(lo, hi)` pair applied to *every* coordinate, so BOTH
    /// degenerate minima must lie in `[lo, hi]` on BOTH axes. `(-2, 2)` is a
    /// reduced range of the canonical `[-5, 5]²`, which is exactly the situation
    /// that silently excluded an optimum in issue #113.
    #[test]
    fn bounds_box_contains_optimum_on_both_axes() {
        let (lo, hi) = SixHumpCamel::new().bounds();
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

    /// O2 (no spurious optimum) — no point of the reduced box scores below `f*`.
    ///
    /// A deterministic 401×401 sweep of `bounds()²`. `eps` guards float error only:
    /// the surface has O(1) scale near the minimum, so `1e-9` is far below any real
    /// dip yet far above the ~1e-16 rounding of the polynomial.
    #[test]
    fn no_point_in_bounds_beats_global_minimum() {
        const STEPS: u16 = 400;
        const EPS: f64 = 1e-9;

        let c = SixHumpCamel::new();
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
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let c = SixHumpCamel::new();
        let plain_no_trailing: String = c.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(c.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let c = SixHumpCamel::new();
        let styled = c.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "SixHumpCamel")
            .expect("SixHumpCamel label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let c = SixHumpCamel::new();
        for line in c.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
