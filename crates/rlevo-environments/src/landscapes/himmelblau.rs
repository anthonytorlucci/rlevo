//! Himmelblau's function — a classic 2-D multimodal benchmark with four equal minima.
//!
//! `f(x₁, x₂) = (x₁ + x₂² − 7)² + (x₁² + x₂ − 11)²`, with four global minima of
//! value `0` arranged around the origin. A pure polynomial (no transcendental
//! terms), differentiable everywhere, with nine stationary points total (four
//! minima, four saddles, one local maximum). Widely used in niching competitions
//! (CEC 2013 F4) because the four equal-depth basins test species formation.
//!
//! # Domain
//!
//! [`bounds`](Himmelblau::bounds) returns `(-6.0, 6.0)`, which **is** the canonical
//! domain: Al-Roomi, *Unconstrained Single-Objective Benchmark Functions
//! Repository*, function #56 "Himmelblau's Function", gives `−6 ≤ xᵢ ≤ 6` with
//! `f* = 0` at `(3, 2)`, `(3.5844, −1.8481)`, `(−3.7793, −3.2832)` and
//! `(−2.8051, 3.1313)`. All four lie inside the box, and `f` is a sum of two squares
//! so nothing can score below `0` — both `bounds()` obligations hold (ADR 0045:
//! O1 reachability, O2 no spurious optimum), pinned by unit tests below.

/// Himmelblau's function (strictly 2-D).
#[derive(Debug, Clone, Copy)]
pub struct Himmelblau;

impl Himmelblau {
    /// Creates a Himmelblau evaluator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Evaluate Himmelblau's function at `(x1, x2)`.
    #[must_use]
    pub fn evaluate(&self, x1: f64, x2: f64) -> f64 {
        (x1 + x2 * x2 - 7.0).powi(2) + (x1 * x1 + x2 - 11.0).powi(2)
    }

    /// Recommended search box `(-6.0, 6.0)` applied per-coordinate — the canonical
    /// domain per Al-Roomi #56. Contains all four global minimizers.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-6.0, 6.0)
    }

    /// 2D projection used by the renderer — the exact surface for a 2-D function.
    fn evaluate_2d(self, x: f64, y: f64) -> f64 {
        self.evaluate(x, y)
    }
}

impl Default for Himmelblau {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for Himmelblau {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Himmelblau",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Himmelblau",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Certified global-minimum value (Al-Roomi #56).
    const F_OPT: f64 = 0.0;

    /// All four certified global minimizers, 12-digit (Al-Roomi 2015 via Maple;
    /// the source table rounds them to `(3,2)`, `(3.5844,−1.8481)`,
    /// `(−3.7793,−3.2832)`, `(−2.8051,3.1313)`).
    const OPTIMA: [(f64, f64); 4] = [
        (3.0, 2.0),
        (3.584_428_340_330, -1.848_126_526_964),
        (-3.779_310_253_378, -3.283_185_991_286),
        (-2.805_118_086_953, 3.131_312_518_250),
    ];

    #[test]
    fn global_minimum_at_known_location() {
        // The exact minimum (3, 2) evaluates to 0.
        assert_relative_eq!(Himmelblau::new().evaluate(3.0, 2.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        assert!(
            Himmelblau::new().evaluate(0.0, 0.0) > 0.0,
            "Himmelblau must exceed its minimum away from the four basins"
        );
    }

    #[test]
    fn all_four_global_minima_zero() {
        // 12-digit certified minimizers (Al-Roomi 2015 via Maple).
        let h = Himmelblau::new();
        assert_relative_eq!(h.evaluate(3.0, 2.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(
            h.evaluate(3.584_428_340_330, -1.848_126_526_964),
            0.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            h.evaluate(-3.779_310_253_378, -3.283_185_991_286),
            0.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            h.evaluate(-2.805_118_086_953, 3.131_312_518_250),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn positive_at_origin() {
        // f(0,0) = 49 + 121 = 170.
        assert_relative_eq!(Himmelblau::new().evaluate(0.0, 0.0), 170.0, epsilon = 1e-10);
    }

    /// O1 (reachability) — the search box reaches every certified global optimum.
    ///
    /// `bounds()` is one `(lo, hi)` pair applied to *every* coordinate, so ALL FOUR
    /// equal-depth minima must lie in `[lo, hi]` on BOTH axes.
    #[test]
    fn bounds_box_contains_optimum_on_both_axes() {
        let (lo, hi) = Himmelblau::new().bounds();
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

    /// O2 (no spurious optimum) — no point of the box scores below `f* = 0`.
    ///
    /// A deterministic 401×401 sweep of `bounds()²`. `f` is a sum of two squares, so
    /// `f ≥ 0` holds exactly and `eps` guards float error only (each square is
    /// non-negative in IEEE-754 too, so the margin is really zero-sided).
    #[test]
    fn no_point_in_bounds_beats_global_minimum() {
        const STEPS: u16 = 400;
        const EPS: f64 = 1e-9;

        let h = Himmelblau::new();
        let (lo, hi) = h.bounds();
        let span = hi - lo;
        let n = f64::from(STEPS);

        for i in 0..=STEPS {
            let x1 = lo + span * f64::from(i) / n;
            for j in 0..=STEPS {
                let x2 = lo + span * f64::from(j) / n;
                let v = h.evaluate(x1, x2);
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

        let h = Himmelblau::new();
        let plain_no_trailing: String = h.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(h.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let h = Himmelblau::new();
        let styled = h.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Himmelblau")
            .expect("Himmelblau label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let h = Himmelblau::new();
        for line in h.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
