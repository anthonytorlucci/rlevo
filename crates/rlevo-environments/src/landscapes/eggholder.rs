//! Eggholder function — a deceptive benchmark with a boundary-pinned optimum.
//!
//! ```text
//! f(x) = Σ_{i=1}^{n-1} [ −x_i·sin(√|x_i − x_{i+1} − 47|)
//!                        − (x_{i+1}+47)·sin(√|0.5·x_i + x_{i+1} + 47|) ]
//! ```
//! Global minimum `f* ≈ −959.6407` at `(512, 404.2318)` for `n = 2`. The optimum
//! is **pinned to the domain boundary** (`x₁* = 512` is not an interior critical
//! point), so interior gradient methods converge to nearby local minima and
//! never reach it. Highly multimodal; non-differentiable where the
//! absolute-value arguments cross zero.
//!
//! Evaluated over `[-512, 512]^n`. Requires `n ≥ 2`.

/// Eggholder function evaluator with configurable dimensionality.
#[derive(Debug, Clone, Copy)]
pub struct Eggholder {
    /// Number of input dimensions. Must be `≥ 2`.
    pub dim: usize,
}

impl Eggholder {
    /// Creates a `dim`-dimensional Eggholder evaluator.
    ///
    /// `dim` should be `≥ 2`; [`evaluate`](Self::evaluate) panics otherwise.
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Evaluate the Eggholder function at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`, or if `self.dim < 2`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        assert!(self.dim >= 2, "Eggholder requires dim >= 2");
        x.windows(2)
            .map(|w| {
                let (xi, xn) = (w[0], w[1]);
                // `abs()` guards the sqrt against negative arguments from rounding.
                let t1 = (xi - xn - 47.0).abs().sqrt();
                let t2 = (0.5 * xi + xn + 47.0).abs().sqrt();
                -xi * t1.sin() - (xn + 47.0) * t2.sin()
            })
            .sum()
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-512.0, 512.0)
    }

    /// 2D projection of [`evaluate`](Self::evaluate) for visualisation.
    ///
    /// The first two coordinates vary across the render window — and the `n = 2`
    /// optimum `(512, 404.2318)` lies in that plane — so coordinates beyond the
    /// first two are held at `0.0`.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        let mut p = vec![0.0_f64; self.dim];
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

impl crate::render::AsciiRenderable for Eggholder {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Eggholder",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Eggholder",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_known_location() {
        let e = Eggholder::new(2);
        assert_relative_eq!(e.evaluate(&[512.0, 404.2318]), -959.6407, epsilon = 1e-2);
    }

    #[test]
    fn positive_or_greater_elsewhere() {
        // The origin scores well above the boundary-pinned optimum.
        let e = Eggholder::new(2);
        assert!(
            e.evaluate(&[0.0, 0.0]) > e.evaluate(&[512.0, 404.2318]),
            "interior point must score worse than the global optimum"
        );
    }

    #[test]
    fn known_n2_global_minimum() {
        let e = Eggholder::new(2);
        assert_relative_eq!(e.evaluate(&[512.0, 404.2318]), -959.6407, epsilon = 1e-2);
    }

    #[test]
    fn no_nan_in_domain() {
        // At a zero crossing of the first absolute-value argument (x2 = x1 − 47),
        // sqrt(0) must not produce NaN.
        let e = Eggholder::new(2);
        let v = e.evaluate(&[47.0, 0.0]);
        assert!(!v.is_nan(), "NaN at zero crossing");
    }

    #[test]
    #[should_panic(expected = "requires dim >= 2")]
    fn panics_on_dim_one() {
        let _ = Eggholder::new(1).evaluate(&[0.0]);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let e = Eggholder::new(2);
        let plain_no_trailing: String = e.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(e.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let e = Eggholder::new(2);
        let styled = e.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Eggholder")
            .expect("Eggholder label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let e = Eggholder::new(2);
        for line in e.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
