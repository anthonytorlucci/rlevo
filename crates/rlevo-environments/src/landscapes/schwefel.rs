//! Schwefel function (Problem 2.26) — a deceptive multimodal benchmark.
//!
//! `f(x) = −Σ x_i·sin(√|x_i|)`, global minimum at `x_i = 420.9687…` where
//! `f(x*) = −418.9829… · n`. The function is deceptive: many local minima
//! cluster near `x_i ≈ 0` while the true optimum sits at ~84% of the domain
//! radius from the centre, so an agent that converges near zero scores
//! `f ≈ 0` instead of `f ≈ −418.98·n`.
//!
//! Evaluated over `[-500, 500]^n`. The function is separable, so coordinate-wise
//! methods have a structural advantage. Differentiable except for a sign
//! discontinuity in the derivative at `x_i = 0`; forward evaluation is exact
//! everywhere.
//!
//! Requires `n ≥ 1`.

/// Per-dimension optimal coordinate, `x_i*`, at full `f64` precision.
//
// The certified literature value carries more decimal digits than f64 stores;
// we keep the published constant verbatim (spec principle: precise constants,
// not rounded approximations) and silence the precision lint accordingly.
#[allow(clippy::excessive_precision)]
const X_OPT: f64 = 420.968_746_359_982_025;
/// Per-dimension contribution to the optimum, `|f(x*)| / n`, at full precision.
///
/// Used by the test suite to assert the certified global-minimum value.
#[cfg(test)]
const F_PER_DIM: f64 = 418.982_887_274_338;

/// Schwefel function evaluator with configurable dimensionality.
#[derive(Debug, Clone, Copy)]
pub struct Schwefel {
    /// Number of input dimensions.
    pub dim: usize,
}

impl Schwefel {
    /// Creates a `dim`-dimensional Schwefel evaluator.
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Evaluate the Schwefel function at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        // `x_i.abs().sqrt()` is well-defined for all finite x_i.
        -x.iter().map(|xi| xi * xi.abs().sqrt().sin()).sum::<f64>()
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-500.0, 500.0)
    }

    /// 2D projection of [`evaluate`](Self::evaluate) for visualisation.
    ///
    /// Coordinates beyond the first two are fixed at the per-dimension optimum
    /// `x_i* = 420.9687…`, not `0.0`, so the rendered slice shows the global
    /// basin rather than a cross-section through the deceptive near-zero region.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        let mut p = vec![X_OPT; self.dim];
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

impl crate::render::AsciiRenderable for Schwefel {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Schwefel",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Schwefel",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_known_location() {
        // Numerically certified optimum; relative tolerance per the literature floor.
        let n = 4;
        let s = Schwefel::new(n);
        assert_relative_eq!(
            s.evaluate(&[X_OPT; 4]),
            -F_PER_DIM * n as f64,
            epsilon = 1e-4
        );
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        // f(0) = 0, which exceeds the true minimum −418.98·n for any n ≥ 1.
        let s = Schwefel::new(3);
        assert!(
            s.evaluate(&[0.0; 3]) > s.evaluate(&[X_OPT; 3]),
            "origin must score worse than the global optimum"
        );
    }

    #[test]
    fn origin_is_not_minimum() {
        let s = Schwefel::new(2);
        assert_relative_eq!(s.evaluate(&[0.0, 0.0]), 0.0, epsilon = 1e-12);
        assert!(s.evaluate(&[0.0, 0.0]) > s.evaluate(&[X_OPT, X_OPT]));
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let s = Schwefel::new(2);
        let plain_no_trailing: String = s.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(s.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let s = Schwefel::new(2);
        let styled = s.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Schwefel")
            .expect("Schwefel label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let s = Schwefel::new(2);
        for line in s.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
