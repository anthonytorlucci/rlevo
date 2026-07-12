//! Generalized Modified Rosenbrock No.01 — the flat-ground bent knife-edge.
//!
//! ```text
//! f(x) = Σ_{i=1}^{n-1} [100·|x_{i+1} − x_i²| + (1 − x_i)²]
//! ```
//! Global minimum `f* = 0` at `x = (1, …, 1)`. Replacing the smooth square of
//! standard Rosenbrock with an absolute value turns the parabolic valley into a
//! V-shaped **knife edge**: the ridge `x_{i+1} = x_i²` is genuinely flat and
//! non-differentiable, so gradient methods stall on it and only the `(1 − x_i)²`
//! terms provide directional signal along the valley. Requires `n ≥ 2`.
//!
//! # Domain
//!
//! The canonical domain is `[-2000, 2000]^n` (Monismith 2010). [`bounds`](RosenbrockFlat::bounds)
//! returns `(-30.0, 30.0)` — a **cited reduced search range**, not a rendering
//! concession: the modified-Rosenbrock side constraints were themselves reduced
//! to `[-30, 30]` after Chen (1997), and Al-Roomi's repository records that
//! reduction alongside the canonical box.
//!
//! The reduction is the right call on both practical and contractual grounds:
//!
//! - `[-2000, 2000]^n` has a search volume of `4000^n` and, sampled at any
//!   sane resolution, renders as a featureless wall — every visible cell is
//!   dominated by the `100·|x_{i+1} − x_i²|` term, which reaches `4·10^8` at the
//!   corners while the entire structure of interest lives near the origin.
//! - The optimum `(1, …, 1)` sits *well inside* `[-30, 30]`, so the box is
//!   **reachable** — no certified minimiser is excluded on any coordinate.
//! - `f` is a sum of `100·|·|` and `(·)²` terms, hence `f ≥ 0` on all of `ℝⁿ`
//!   with `f = 0` attained at `(1, …, 1)`. No box — reduced or canonical —
//!   can contain a point better than `f* = 0`, so the reduction cannot
//!   introduce a **spurious optimum**.
//!
//! Both obligations are pinned by unit tests (`bounds_box_contains_optimum`,
//! `no_point_in_bounds_beats_global_minimum`). The evaluator never clamps — the
//! benchmark harness owns domain enforcement.
//!
//! # Global minimum — known source erratum
//!
//! Al-Roomi's page for this function family prints `f_min = 1` at `x* = 0`.
//! That is **falsified by its own formula**: at `x = (1, …, 1)` every
//! `100·|x_{i+1} − x_i²|` term and every `(1 − x_i)²` term vanishes, giving
//! `f = 0 < 1`. This crate deliberately implements the correct value,
//! `f* = 0` at `x = (1, …, 1)`. Do not "fix" it back to the cited value.
//!
//! # References
//!
//! Monismith, D.R. (2010), PhD dissertation, Oklahoma State University — source
//! of the canonical `[-2000, 2000]^n` domain, catalogued as function #170 in
//! Al-Roomi's benchmark repository.
//!
//! Chen, D. (1997), "Comparisons Among Stochastic Optimization Algorithms", MS
//! thesis, Oklahoma State University — source of the reduced `[-30, 30]` side
//! constraints that [`bounds`](RosenbrockFlat::bounds) returns, as recorded by
//! Al-Roomi.

use rlevo_core::config::{self, ConfigError};

/// Generalized Modified Rosenbrock No.01 evaluator with configurable dimensionality.
#[derive(Debug, Clone, Copy)]
pub struct RosenbrockFlat {
    /// Number of input dimensions. Always `≥ 2` — enforced by [`RosenbrockFlat::new`].
    dim: usize,
}

impl RosenbrockFlat {
    /// Creates a `dim`-dimensional Modified Rosenbrock No.01 evaluator.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `dim < 2`. Both the knife-edge term
    /// `100·|x_{i+1} − x_i²|` and the `(1 − x_i)²` pull term are defined only on
    /// adjacent coordinate pairs (`i = 1..n−1`); with a single coordinate the sum
    /// is empty, erasing the ridge the benchmark exists to test.
    pub fn new(dim: usize) -> Result<Self, ConfigError> {
        const C: &str = "RosenbrockFlat";
        config::at_least(C, "dim", dim, 2)?;
        Ok(Self { dim })
    }

    /// Number of input dimensions.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Evaluate the Modified Rosenbrock No.01 function at `x`.
    ///
    /// Every term is non-negative, so `f ≥ 0` everywhere on `ℝⁿ`; the global
    /// minimum is `f* = 0` at `x = (1, …, 1)`, where each term vanishes.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        // ERRATUM (do not "fix"): Al-Roomi lists f_min = 1 at x* = 0 for this
        // family. That contradicts this very formula — at x = (1,…,1) both the
        // 100·|x_{i+1} − x_i²| and (1 − x_i)² terms are zero, so f = 0 < 1. The
        // correct optimum is f* = 0 at (1,…,1); the cited value is a source error.
        x.windows(2)
            .map(|w| {
                let (xi, xn) = (w[0], w[1]);
                // non-differentiable on the manifold x_{i+1} = x_i^2 (the knife edge)
                100.0 * (xn - xi * xi).abs() + (1.0 - xi).powi(2)
            })
            .sum()
    }

    /// Cited reduced search range `(-30.0, 30.0)`, applied per-coordinate.
    ///
    /// This is **not** a render window: the modified-Rosenbrock side constraints
    /// were reduced from the canonical `[-2000, 2000]^n` (Monismith 2010) to
    /// `[-30, 30]` after Chen (1997). The reduced box contains the optimum
    /// `(1, …, 1)` on every coordinate, and — because `f` is a sum of `100·|·|`
    /// and `(·)²` terms, hence `f ≥ 0` identically with `f = 0` at the optimum —
    /// it contains no point better than `f* = 0`. See the module docs.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-30.0, 30.0)
    }

    /// 2D projection of [`evaluate`](Self::evaluate) for visualisation.
    ///
    /// Coordinates beyond the first two are fixed at `1.0` (the optimum) so the
    /// rendered slice passes through the valley.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        let mut p = vec![1.0_f64; self.dim];
        if !p.is_empty() {
            p[0] = x;
        }
        if p.len() >= 2 {
            p[1] = y;
        }
        self.evaluate(&p)
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for RosenbrockFlat {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "RosenbrockFlat",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "RosenbrockFlat",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_known_location() {
        let r = RosenbrockFlat::new(4).expect("dim >= 2");
        assert_relative_eq!(r.evaluate(&[1.0; 4]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        let r = RosenbrockFlat::new(3).expect("dim >= 2");
        assert!(
            r.evaluate(&[0.0; 3]) > 0.0,
            "value away from (1,…,1) must exceed the minimum 0"
        );
    }

    #[test]
    fn minimum_at_ones() {
        let r = RosenbrockFlat::new(5).expect("dim >= 2");
        assert_relative_eq!(r.evaluate(&[1.0; 5]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn flat_on_ridge() {
        // On the ridge x_{i+1} = x_i² (but not at x = 1) only the (1 − x_i)² term remains.
        let r = RosenbrockFlat::new(2).expect("dim >= 2");
        let x = [2.0_f64, 4.0]; // x2 = x1² = 4 ⇒ abs term is 0
        let expected = (1.0 - 2.0_f64).powi(2); // 1.0
        assert_relative_eq!(r.evaluate(&x), expected, epsilon = 1e-10);
    }

    #[test]
    fn new_rejects_dim_one() {
        assert!(
            RosenbrockFlat::new(1).is_err(),
            "dim = 1 leaves the pairwise sum empty"
        );
    }

    #[test]
    fn new_rejects_dim_zero() {
        assert!(
            RosenbrockFlat::new(0).is_err(),
            "dim = 0 has no coordinates"
        );
    }

    #[test]
    fn dim_accessor_reports_configured_dim() {
        assert_eq!(RosenbrockFlat::new(7).expect("dim >= 2").dim(), 7);
    }

    #[test]
    fn bounds_box_contains_optimum() {
        // O1 — reachability. The single `(lo, hi)` pair is applied per-coordinate,
        // so the optimum (1,…,1) must lie inside [lo, hi] on EVERY axis, or no
        // search harness could reach it. The cited reduced range [-30, 30] holds
        // it with room to spare.
        let r = RosenbrockFlat::new(6).expect("dim >= 2");
        let (lo, hi) = r.bounds();
        let optimum = vec![1.0_f64; r.dim()];
        assert!(
            optimum.iter().all(|&c| lo <= c && c <= hi),
            "optimum (1,…,1) outside bounds ({lo}, {hi})"
        );
        assert_relative_eq!(r.evaluate(&optimum), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn no_point_in_bounds_beats_global_minimum() {
        // O2 — no spurious optimum. f is a sum of 100·|·| and (·)² terms, so
        // f ≥ 0 identically on ℝⁿ and f* = 0 at (1,…,1). No box — the canonical
        // [-2000, 2000]^n or the cited reduced [-30, 30] — can therefore contain
        // a point better than f*. Pinned by a dense deterministic sweep of the
        // 2-D slice over bounds()².
        let r = RosenbrockFlat::new(2).expect("dim >= 2");
        let (lo, hi) = r.bounds();
        const STEPS: i32 = 300;
        let step = (hi - lo) / f64::from(STEPS);

        for i in 0..=STEPS {
            let x = lo + step * f64::from(i);
            for j in 0..=STEPS {
                let y = lo + step * f64::from(j);
                let v = r.evaluate(&[x, y]);
                assert!(
                    v >= -1e-12,
                    "f({x}, {y}) = {v} beats the global minimum f* = 0 inside bounds()"
                );
            }
        }
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let r = RosenbrockFlat::new(2).expect("dim >= 2");
        let plain_no_trailing: String = r.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(r.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let r = RosenbrockFlat::new(2).expect("dim >= 2");
        let styled = r.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "RosenbrockFlat")
            .expect("RosenbrockFlat label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let r = RosenbrockFlat::new(2).expect("dim >= 2");
        for line in r.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
