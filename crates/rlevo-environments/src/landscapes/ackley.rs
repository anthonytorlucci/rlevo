//! Ackley function — a classical multimodal benchmark for EAs.
//!
//! `f(x) = -a·exp(-b·√(1/n · Σ x_i²)) − exp(1/n · Σ cos(c·x_i)) + a + e`
//! with canonical constants `a = 20`, `b = 0.2`, `c = 2π`. Global
//! minimum at `x = 0` where `f(0) = 0`. Commonly evaluated over
//! `[-32.768, 32.768]^n`; a narrower `[-5.12, 5.12]^n` window is
//! convenient when comparing against Sphere / Rastrigin on the same
//! axis.

use std::f64::consts::{E, PI};

/// Ackley function evaluator with configurable dimensionality and constants.
#[derive(Debug, Clone, Copy)]
pub struct Ackley {
    /// Number of input dimensions.
    pub dim: usize,
    /// Outer scaling constant (canonical: `20.0`).
    pub a: f64,
    /// Inner exponential decay constant (canonical: `0.2`).
    pub b: f64,
    /// Cosine frequency constant (canonical: `2π`).
    pub c: f64,
}

impl Ackley {
    /// Build an Ackley with Ackley's canonical constants (`a=20`, `b=0.2`, `c=2π`).
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self {
            dim,
            a: 20.0,
            b: 0.2,
            c: 2.0 * PI,
        }
    }

    /// Evaluate the Ackley function at `x`. Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");
        let n = x.len() as f64;
        let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        let sum_cos: f64 = x.iter().map(|xi| (self.c * xi).cos()).sum();
        -self.a * (-self.b * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + self.a + E
    }

    /// Recommended search domain for each coordinate.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (-32.768, 32.768)
    }

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

impl crate::render::AsciiRenderable for Ackley {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Ackley",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "Ackley",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn global_minimum_at_origin() {
        let a = Ackley::new(5);
        assert_relative_eq!(a.evaluate(&[0.0; 5]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn positive_elsewhere() {
        let a = Ackley::new(3);
        assert!(a.evaluate(&[1.0, 2.0, 3.0]) > 0.0);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let a = Ackley::new(2);
        let plain_no_trailing: String = a.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(a.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let a = Ackley::new(2);
        let styled = a.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Ackley")
            .expect("Ackley label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let a = Ackley::new(2);
        for line in a.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
