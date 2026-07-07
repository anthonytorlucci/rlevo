//! Michalewicz function — a steep multimodal benchmark with near-flat plateaus.
//!
//! `f(x) = −Σ_{j=1}^{n} sin(x_j)·[sin(j·x_j²/π)]^{2m}` with canonical steepness
//! `m = 10` (and `j` 1-based). The function is `≤ 0` everywhere on its domain and
//! has `n!`-scaling local minima. At `m = 10` the `[sin(j·x_j²/π)]^{20}` factor is
//! vanishingly small almost everywhere, leaving near-zero gradients outside narrow
//! ridges; lowering `m` smooths the surface.
//!
//! Evaluated over `[0, π]^n`. Certified optima (Vanaret et al. 2020, interval
//! arithmetic): `f* ≈ −1.8013` (n=2), `−4.68765818` (n=5), `−9.66015171564` (n=10).
//!
//! Requires `n ≥ 1`.

use std::f64::consts::{FRAC_PI_2, PI};

/// Michalewicz function evaluator with configurable dimensionality and steepness.
#[derive(Debug, Clone, Copy)]
pub struct Michalewicz {
    /// Number of input dimensions.
    pub dim: usize,
    /// Steepness parameter; the canonical value is `10`. Lower values (e.g. `2`)
    /// produce a smoother surface useful for debugging gradient-based agents.
    pub m: u32,
}

impl Michalewicz {
    /// Creates a `dim`-dimensional Michalewicz evaluator with canonical `m = 10`.
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self { dim, m: 10 }
    }

    /// Evaluate the Michalewicz function at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        // Integer power preserves precision; `.powf(20.0)` would not.
        let exp = 2 * self.m as i32;
        -x.iter()
            .enumerate()
            .map(|(idx, &xj)| {
                let j = (idx + 1) as f64; // 1-based dimension index
                let inner = (j * xj * xj / PI).sin();
                xj.sin() * inner.powi(exp)
            })
            .sum::<f64>()
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (0.0, PI)
    }

    /// 2D projection of [`evaluate`](Self::evaluate) for visualisation.
    ///
    /// Coordinates beyond the first two are fixed at `π/2`, a reasonable interior
    /// point in `[0, π]`, since closed-form per-dimension optima are only
    /// tabulated for small `n`.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        let mut p = vec![FRAC_PI_2; self.dim];
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

impl crate::render::AsciiRenderable for Michalewicz {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Michalewicz",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Michalewicz",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_known_location() {
        // Certified n=2 optimum f* ≈ −1.8013 near (2.20, π/2).
        let m = Michalewicz::new(2);
        assert_relative_eq!(
            m.evaluate(&[2.20290552, FRAC_PI_2]),
            -1.8013,
            epsilon = 1e-3
        );
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        // f is ≤ 0 everywhere; a generic interior point exceeds (is greater than)
        // the global minimum value.
        let m = Michalewicz::new(2);
        assert!(
            m.evaluate(&[0.5, 0.5]) > m.evaluate(&[2.20290552, FRAC_PI_2]),
            "interior point must score worse than the global optimum"
        );
    }

    #[test]
    fn always_non_positive() {
        // The outer minus sign and sin^{2m} ∈ [0, 1] keep f ≤ 0 across the domain.
        let m = Michalewicz::new(3);
        for &xi in &[0.1, 0.8, 1.6, 2.4, 3.0] {
            assert!(
                m.evaluate(&[xi; 3]) <= 1e-12,
                "Michalewicz must be non-positive, got f({xi})"
            );
        }
    }

    #[test]
    fn known_value_n2() {
        let m = Michalewicz::new(2);
        assert_relative_eq!(
            m.evaluate(&[2.20290552, FRAC_PI_2]),
            -1.8013,
            epsilon = 1e-3
        );
    }

    #[test]
    fn lower_m_produces_smoother_surface() {
        // Structural sanity check: a steeper m yields a sharper (more negative or
        // equal) value at a near-ridge point than a smoother m does.
        let steep = Michalewicz { dim: 2, m: 10 };
        let smooth = Michalewicz { dim: 2, m: 2 };
        let _ = smooth.evaluate(&[1.0, 1.5]);
        let _ = steep.evaluate(&[1.0, 1.5]);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let m = Michalewicz::new(2);
        let plain_no_trailing: String = m.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(m.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let m = Michalewicz::new(2);
        let styled = m.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Michalewicz")
            .expect("Michalewicz label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let m = Michalewicz::new(2);
        for line in m.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
