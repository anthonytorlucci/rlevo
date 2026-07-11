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

use rlevo_core::config::{self, ConfigError};

/// Global-funnel centre `μ₁`.
const MU1: f64 = 2.5;
/// Depth offset `d` of the deceptive funnel.
const D: f64 = 1.0;

/// Lunacek bi-Rastrigin function evaluator with configurable dimensionality.
///
/// The depth-scaling `s` and deceptive-funnel centre `μ₂` are derived from `n`
/// on demand (see [`s`](Self::s) / [`mu2`](Self::mu2)) because they require a
/// `sqrt`, which cannot run in a `const fn` constructor. Both are total only
/// because [`new`](Self::new) rejects `dim < 2`: `s` is non-positive below `n = 2`,
/// which would send `μ₂` through the square root of a negative number.
#[derive(Debug, Clone, Copy)]
pub struct LunacekBiRastrigin {
    /// Number of input dimensions. Always `≥ 2` — enforced by [`LunacekBiRastrigin::new`].
    dim: usize,
}

impl LunacekBiRastrigin {
    /// Creates a `dim`-dimensional Lunacek bi-Rastrigin evaluator.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `dim < 2`. The depth-scaling parameter
    /// `s = 1 − 1/(2√(n+20) − 8.2)` is non-positive for `n < 2` (`s(0) ≈ −0.344`,
    /// `s(1) ≈ −0.036`), which makes `μ₂ = −√((μ₁² − d)/s)` the square root of a
    /// negative number — i.e. `NaN`, propagated silently through
    /// [`evaluate`](Self::evaluate) with no panic and no diagnostic. This bound is
    /// *derived* from the published parameterization (Lunacek, Whitley & Sutton,
    /// PPSN X, 2008); the paper does not state it explicitly.
    pub fn new(dim: usize) -> Result<Self, ConfigError> {
        const C: &str = "LunacekBiRastrigin";
        config::at_least(C, "dim", dim, 2)?;
        Ok(Self { dim })
    }

    /// Number of input dimensions.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Depth-scaling parameter `s = 1 − 1/(2√(n+20) − 8.2)`.
    ///
    /// Always strictly positive: `s` crosses zero between `n = 1` (`≈ −0.036`) and
    /// `n = 2` (`≈ +0.153`), and [`new`](Self::new) rejects `dim < 2`.
    #[must_use]
    pub fn s(&self) -> f64 {
        1.0 - 1.0 / (2.0 * (self.dim as f64 + 20.0).sqrt() - 8.2)
    }

    /// Deceptive-funnel centre `μ₂ = −√((μ₁² − d)/s)`.
    ///
    /// Always finite: [`s`](Self::s) is positive for every constructible `dim`, so
    /// the radicand `(μ₁² − d)/s` is positive and the `sqrt` is real.
    #[must_use]
    pub fn mu2(&self) -> f64 {
        -((MU1 * MU1 - D) / self.s()).sqrt()
    }

    /// Evaluate the Lunacek bi-Rastrigin function at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        let n = self.dim as f64;
        let s = self.s();
        let mu2 = self.mu2();

        let funnel1: f64 = x.iter().map(|xi| (xi - MU1).powi(2)).sum();
        let funnel2: f64 = D * n + s * x.iter().map(|xi| (xi - mu2).powi(2)).sum::<f64>();
        // non-differentiable where funnel1 == funnel2
        let rastrigin: f64 = x
            .iter()
            .map(|xi| 1.0 - (2.0 * PI * (xi - MU1)).cos())
            .sum::<f64>();

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
        let f = LunacekBiRastrigin::new(4).expect("dim >= 2");
        assert_relative_eq!(f.evaluate(&[MU1; 4]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        let f = LunacekBiRastrigin::new(3).expect("dim >= 2");
        assert!(
            f.evaluate(&[0.0; 3]) > 0.0,
            "value away from μ₁ must exceed the minimum 0"
        );
    }

    #[test]
    fn global_minimum_at_mu1() {
        let f = LunacekBiRastrigin::new(5).expect("dim >= 2");
        assert_relative_eq!(f.evaluate(&[2.5; 5]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn s_parameter_positive() {
        for n in 2..=50 {
            let f = LunacekBiRastrigin::new(n).expect("dim >= 2");
            assert!(f.s() > 0.0, "s must be positive for n={n}");
        }
    }

    /// The funnel parameters are the reason `dim >= 2` is a *construction*
    /// invariant: `s` goes non-positive below `n = 2`, and `mu2` then takes the
    /// square root of a negative number and yields NaN with no panic. Pin both at
    /// the boundary dimension, where `s` is smallest and closest to the zero
    /// crossing (`s(2) ≈ 0.153`).
    #[test]
    fn funnel_parameters_finite_at_minimum_dim() {
        let f = LunacekBiRastrigin::new(2).expect("dim >= 2");
        assert!(
            f.s() > 0.0,
            "s must be positive at the minimum dim, got {}",
            f.s()
        );
        assert!(
            f.mu2().is_finite(),
            "mu2 must be finite at the minimum dim, got {}",
            f.mu2()
        );
        assert!(f.mu2() < 0.0, "mu2 is the negative root by definition");
        // The whole surface stays finite too — no NaN leaks into evaluate.
        assert!(f.evaluate(&[0.0, 0.0]).is_finite());
    }

    #[test]
    fn new_rejects_dim_one() {
        // s(1) ≈ −0.036 ⇒ mu2 = −√(negative) = NaN, silently.
        assert!(LunacekBiRastrigin::new(1).is_err());
    }

    #[test]
    fn new_rejects_dim_zero() {
        // s(0) ≈ −0.344 ⇒ same NaN hole.
        assert!(LunacekBiRastrigin::new(0).is_err());
    }

    #[test]
    fn dim_accessor_reports_configured_dim() {
        assert_eq!(LunacekBiRastrigin::new(7).expect("dim >= 2").dim(), 7);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let f = LunacekBiRastrigin::new(2).expect("dim >= 2");
        let plain_no_trailing: String = f.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(f.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let f = LunacekBiRastrigin::new(2).expect("dim >= 2");
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

        let f = LunacekBiRastrigin::new(2).expect("dim >= 2");
        for line in f.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
