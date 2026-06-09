//! Lunacek bi-Rastrigin function (Lunacek 2008, unrotated) — competing basins.
//!
//! This is the original unrotated canonical form. The BBOB/COCO suite lists it
//! as f24 but adds orthogonal rotation and ill-conditioning transforms on top of
//! this body; those are intentionally omitted here.
//!
//! ```text
//! f(x) = min[ Σ(x_i − μ₁)²,  d·n + s·Σ(x_i − μ₂)² ]
//!        + 10·Σ{ 1 − cos[2π(x_i − μ₁)] }
//! ```
//! with `μ₁ = 2.5`, `d = 1`, `s = 1 − 1/(2√(n+20) − 8.2)`, and
//! `μ₂ = −√((μ₁² − d)/s)`. Global minimum `f* = 0` at `x_i = μ₁ = 2.5`.
//!
//! The function pairs two Sphere funnels — a narrow global one at `μ₁` and a
//! wide deceptive one at `μ₂` that occupies most of the search volume — with a
//! Rastrigin oscillation. Large-population EAs distribute members proportional
//! to basin volume, biasing the majority toward the wrong funnel.
//!
//! Evaluated over `[-5.12, 5.12]^n`. The `min(·,·)` is non-differentiable on the
//! hypersurface where its two arguments are equal. Requires `n ≥ 2`.

use std::f64::consts::PI;

/// Global-funnel centre `μ₁`.
const MU1: f64 = 2.5;
/// Depth offset `d` of the deceptive funnel.
const D: f64 = 1.0;

/// Lunacek bi-Rastrigin function evaluator with configurable dimensionality.
///
/// The depth-scaling `s` and deceptive-funnel centre `μ₂` are derived from `n`
/// on demand (see [`s`](Self::s) / [`mu2`](Self::mu2)) because they require a
/// `sqrt`, which cannot run in a `const fn` constructor.
#[derive(Debug, Clone, Copy)]
pub struct LunacekBiRastrigin {
    /// Number of input dimensions. Must be `≥ 2`.
    pub dim: usize,
}

impl LunacekBiRastrigin {
    /// Creates a `dim`-dimensional Lunacek bi-Rastrigin evaluator.
    ///
    /// `dim` should be `≥ 2`; [`evaluate`](Self::evaluate) panics otherwise.
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Depth-scaling parameter `s = 1 − 1/(2√(n+20) − 8.2)`. Positive for all `n ≥ 2`.
    #[must_use]
    pub fn s(&self) -> f64 {
        1.0 - 1.0 / (2.0 * (self.dim as f64 + 20.0).sqrt() - 8.2)
    }

    /// Deceptive-funnel centre `μ₂ = −√((μ₁² − d)/s)`.
    #[must_use]
    pub fn mu2(&self) -> f64 {
        -((MU1 * MU1 - D) / self.s()).sqrt()
    }

    /// Evaluate the Lunacek bi-Rastrigin function at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`, or if `self.dim < 2`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        assert!(self.dim >= 2, "Lunacek bi-Rastrigin requires dim >= 2");
        let n = self.dim as f64;
        let s = self.s();
        let mu2 = self.mu2();

        let funnel1: f64 = x.iter().map(|xi| (xi - MU1).powi(2)).sum();
        let funnel2: f64 = D * n + s * x.iter().map(|xi| (xi - mu2).powi(2)).sum::<f64>();
        // non-differentiable where funnel1 == funnel2
        let rastrigin: f64 = x.iter().map(|xi| 1.0 - (2.0 * PI * (xi - MU1)).cos()).sum::<f64>();

        funnel1.min(funnel2) + 10.0 * rastrigin
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-5.12, 5.12)
    }

    /// 2D projection of [`evaluate`](Self::evaluate) for visualisation.
    ///
    /// Coordinates beyond the first two are fixed at `μ₁ = 2.5` (the optimum) so
    /// the rendered slice passes through the global funnel.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        let mut p = vec![MU1; self.dim];
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

impl crate::render::AsciiRenderable for LunacekBiRastrigin {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "LunacekBiRastrigin",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "LunacekBiRastrigin",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_known_location() {
        let f = LunacekBiRastrigin::new(4);
        assert_relative_eq!(f.evaluate(&[MU1; 4]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        let f = LunacekBiRastrigin::new(3);
        assert!(
            f.evaluate(&[0.0; 3]) > 0.0,
            "value away from μ₁ must exceed the minimum 0"
        );
    }

    #[test]
    fn global_minimum_at_mu1() {
        let f = LunacekBiRastrigin::new(5);
        assert_relative_eq!(f.evaluate(&[2.5; 5]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn s_parameter_positive() {
        for n in 2..=50 {
            let f = LunacekBiRastrigin::new(n);
            assert!(f.s() > 0.0, "s must be positive for n={n}");
        }
    }

    #[test]
    #[should_panic(expected = "requires dim >= 2")]
    fn panics_on_dim_one() {
        let _ = LunacekBiRastrigin::new(1).evaluate(&[2.5]);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let f = LunacekBiRastrigin::new(2);
        let plain_no_trailing: String = f.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(f.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let f = LunacekBiRastrigin::new(2);
        let styled = f.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "LunacekBiRastrigin")
            .expect("LunacekBiRastrigin label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let f = LunacekBiRastrigin::new(2);
        for line in f.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
