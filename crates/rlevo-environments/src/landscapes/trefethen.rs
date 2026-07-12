//! Trefethen's function — a 2-D benchmark with five incommensurate frequencies.
//!
//! ```text
//! f(x₁,x₂) = e^{sin(50·x₁)} + sin(60·e^{x₂}) + sin(70·sin(x₁))
//!          + sin(sin(80·x₂)) − sin(10·(x₁+x₂)) + (x₁² + x₂²)/4
//! ```
//! Global minimum `f* ≈ −3.306_868_647_475_23` at `(−0.024403, 0.210612)`. The
//! frequencies `{50, 60, 70, 80, 10}` share no ratio, so there is no periodic
//! lattice of equivalent basins and no dominant spatial scale — dense grid
//! search plus local refinement is needed. The stabilising `(x₁²+x₂²)/4` term is
//! coercive, so it keeps the minimum interior and well-defined.
//!
//! # Domain
//!
//! Trefethen's original problem — Problem 4 of the SIAM 100-Digit Challenge
//! (*SIAM News*, Jan/Feb 2002) — is **unconstrained**: it states no box at all.
//! It is well-posed anyway because the `(x₁²+x₂²)/4` term is coercive, so a
//! finite global minimum exists without an artificial box. (The `[-1, 1]²`
//! framing often repeated in benchmark write-ups is a *visualization*
//! convention, not a published constraint — it is not cited here as one.)
//!
//! The **bounded benchmark box** used by the optimization literature —
//! `x₁ ∈ [-6.5, 6.5]`, `x₂ ∈ [-4.5, 4.5]` — was imposed by Mishra, S. (2006),
//! *"Some New Test Functions for Global Optimization and Performance of
//! Repulsive Particle Swarm Method"*, MPRA Paper 2718, and propagated via
//! Gavana (2013) → Al-Roomi (2015).
//!
//! That box is asymmetric. [`bounds`](Trefethen::bounds) returns a single
//! `(lo, hi)` pair applied per-coordinate (the consuming renderer and search
//! harnesses use one box for every axis), so it returns the *square hull*
//! `(-6.5, 6.5)` of that box — the smallest symmetric square containing the full
//! benchmark domain and the optimum `(−0.024403, 0.210612)` on both axes.
//! The evaluator never clamps.

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

    /// Square hull `(-6.5, 6.5)` of the asymmetric benchmark box
    /// `x₁ ∈ [-6.5, 6.5]`, `x₂ ∈ [-4.5, 4.5]`, applied per-coordinate. Contains
    /// the full domain and the optimum `(−0.024403, 0.210612)` on both axes. See
    /// the module-level docs for the provenance of that box.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        // Why widening x2 from ±4.5 to ±6.5 cannot admit a spurious optimum
        // (i.e. any point with f < f*):
        //
        // Every oscillatory term is bounded below:
        //     e^{sin(50·x₁)}   ≥ e^{-1}    ≈  0.3679
        //     sin(60·e^{x₂})   ≥ −1
        //     sin(70·sin(x₁))  ≥ −1
        //     sin(sin(80·x₂))  ≥ −sin(1)   ≈ −0.8415
        //     −sin(10·(x₁+x₂)) ≥ −1
        // Summing: the oscillatory part ≥ e^{-1} − 1 − 1 − sin(1) − 1 ≈ −3.4736.
        //
        // The stabiliser (x₁² + x₂²)/4 ≥ 0. So any point with
        // f < f* = −3.306_868_647_475_23 must satisfy
        //     (x₁² + x₂²)/4 < −3.30686… + 3.4736… ≈ 0.1667,
        // i.e. it lies within radius ≈ 0.817 of the origin — comfortably inside
        // both the old (±4.5) and the new (±6.5) box. Widening the box therefore
        // cannot expose any point better than f*; it only restores reachability
        // of the full x₁ domain, which ±4.5 was clipping.
        (-6.5, 6.5)
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

    /// Certified global minimum value of Trefethen's function.
    const F_OPT: f64 = -3.306_868_647_475_23;

    /// Certified global minimiser.
    const X_OPT: (f64, f64) = (-0.024_403, 0.210_612);

    #[test]
    fn global_minimum_at_known_location() {
        let t = Trefethen::new();
        assert_relative_eq!(t.evaluate(X_OPT.0, X_OPT.1), F_OPT, epsilon = 1e-3);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        let t = Trefethen::new();
        assert!(
            t.evaluate(6.0, 4.0) > t.evaluate(X_OPT.0, X_OPT.1),
            "value far from the optimum must exceed f*"
        );
    }

    #[test]
    fn bounds_box_contains_optimum_on_both_axes() {
        // The single `(lo, hi)` pair is applied per-coordinate, so the optimum
        // must lie inside [lo, hi] on BOTH axes for search harnesses to be able
        // to reach it (obligation O1 — reachability).
        let (lo, hi) = Trefethen::new().bounds();
        assert!(lo <= X_OPT.0 && X_OPT.0 <= hi, "x1 optimum outside bounds");
        assert!(lo <= X_OPT.1 && X_OPT.1 <= hi, "x2 optimum outside bounds");
    }

    #[test]
    fn bounds_box_contains_full_asymmetric_domain() {
        // The square hull must cover x1 ∈ [-6.5, 6.5] and x2 ∈ [-4.5, 4.5]
        // (obligation O1 — the ±4.5 box used to clip the x1 domain).
        let (lo, hi) = Trefethen::new().bounds();
        assert!(lo <= -6.5 && hi >= 6.5, "x1 domain not covered");
        assert!(lo <= -4.5 && hi >= 4.5, "x2 domain not covered");
    }

    #[test]
    fn no_point_in_bounds_beats_global_minimum() {
        // Obligation O2 — no spurious optimum: widening the box must not expose
        // any point with f < f*. Trefethen oscillates at frequencies 50–80, so a
        // 400×400 grid will NOT land near the true optimum — that is expected.
        // This test asserts only that the lower bound f* is never breached.
        let t = Trefethen::new();
        let (lo, hi) = t.bounds();
        const STEPS: usize = 400;
        #[expect(
            clippy::cast_precision_loss,
            reason = "STEPS is 400; grid indices are exactly representable in f64"
        )]
        let step = (hi - lo) / STEPS as f64;

        for i in 0..=STEPS {
            #[expect(
                clippy::cast_precision_loss,
                reason = "i <= 400; exactly representable in f64"
            )]
            let x1 = lo + step * i as f64;
            for j in 0..=STEPS {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "j <= 400; exactly representable in f64"
                )]
                let x2 = lo + step * j as f64;
                let f = t.evaluate(x1, x2);
                assert!(
                    f >= F_OPT - 1e-6,
                    "found point ({x1}, {x2}) with f = {f} below the global minimum {F_OPT}"
                );
            }
        }
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
