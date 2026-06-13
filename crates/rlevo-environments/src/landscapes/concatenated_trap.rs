//! Concatenated trap — a deceptive, additively decomposable binary landscape.
//!
//! The genome is `num_blocks · block_size` binary genes, partitioned into
//! contiguous, non-overlapping blocks of `k = block_size` bits each. Genes
//! are interpreted as bits via a `>= 0.5` threshold. Each block contributes a
//! cost that depends only on its *unitation* `u` (the number of 1-bits in the
//! block), and the total objective is the sum across blocks:
//!
//! ```text
//!   cost(u) = 0        if u == k        (all-ones block)
//!   cost(u) = u + 1    otherwise
//! ```
//!
//! For a trap of order `k = 5`, the per-block cost table is:
//!
//! | unitation `u` | 0 | 1 | 2 | 3 | 4 | 5 |
//! |---------------|---|---|---|---|---|---|
//! | `cost(u)`     | 1 | 2 | 3 | 4 | 5 | 0 |
//!
//! This is a *minimisation* problem (lower is better), consistent with the
//! convention used throughout `rlevo-evolution`. The **global optimum** is the
//! all-ones genome with total cost `0`. There is a strong **deceptive local
//! optimum** at the all-zeros genome with total cost `num_blocks`: from
//! all-zeros, flipping any single bit to `1` lifts that block from `cost(0) = 1`
//! to `cost(1) = 2`, so every Hamming-1 neighbour is strictly worse. The cost
//! gradient over a block therefore points *away* from the all-ones optimum for
//! every unitation below `k`.
//!
//! # Deception and EDAs
//!
//! Univariate estimation-of-distribution algorithms (UMDA, PBIL, cGA) model
//! each gene's marginal `P(x_i = 1)` independently. Because the trap rewards
//! the all-zeros block at every unitation except the very last, the marginal
//! statistics of selected individuals push each bit toward `0` — the univariate
//! model cannot represent the higher-order dependency that "all `k` bits must
//! flip together" to reach the optimum. Such algorithms reliably converge to
//! the all-zeros deceptive optimum. Models that capture inter-gene structure
//! (MIMIC's dependency chain, the Bayesian Optimization Algorithm) are required
//! to solve order-`k` traps, which is precisely why concatenated traps are the
//! canonical benchmark for multivariate EDAs.
//!
//! # References
//!
//! - Deb, K. & Goldberg, D. E. (1992). "Analyzing deception in trap functions."
//!   *Foundations of Genetic Algorithms 2*, 93–108.
//! - Pelikan, M., Goldberg, D. E. & Cantú-Paz, E. (1999). "BOA: The Bayesian
//!   Optimization Algorithm." *Proceedings of GECCO-99*, 525–532.

/// Concatenated trap evaluator: `num_blocks` order-`block_size` traps.
#[derive(Debug, Clone, Copy)]
pub struct ConcatenatedTrap {
    /// Number of contiguous, non-overlapping trap blocks.
    pub num_blocks: usize,
    /// Trap order `k` — the number of bits in each block.
    pub block_size: usize,
}

impl ConcatenatedTrap {
    /// Creates a concatenated trap of `num_blocks` blocks, each an order-`block_size` trap.
    #[must_use]
    pub const fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            num_blocks,
            block_size,
        }
    }

    /// Total genome dimension: `num_blocks · block_size`.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.num_blocks * self.block_size
    }

    /// Evaluate the concatenated trap at `x`.
    ///
    /// Genes are thresholded to bits via `>= 0.5`, partitioned into contiguous
    /// blocks of `block_size`, and each block's trap cost (`0` if the block is
    /// all-ones, else `u + 1` for unitation `u`) is summed. Lower is better;
    /// the all-ones genome scores `0`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim()`.
    #[must_use]
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim(), "input dimension mismatch");
        let total: usize = x
            .chunks_exact(self.block_size)
            .map(|block| {
                let u = block.iter().filter(|&&gene| gene >= 0.5).count();
                Self::block_cost(self.block_size, u)
            })
            .sum();
        // The total is bounded by num_blocks · (k + 1), well within the exact
        // integer range of f64 (2^53), so the cast is exact, not lossy.
        #[allow(clippy::cast_precision_loss)]
        let cost = total as f64;
        cost
    }

    /// Recommended search domain for each coordinate: the binary box `[0, 1]`.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    /// Per-block trap cost for a block of unitation `u` (count of 1-bits).
    ///
    /// Returns `0` when the block is all-ones (`u == block_size`), otherwise
    /// `u + 1`. This is the deceptive trap: cost rises with unitation right up
    /// until the final, rewarding all-ones configuration.
    const fn block_cost(block_size: usize, u: usize) -> usize {
        if u == block_size {
            0
        } else {
            u + 1
        }
    }

    /// 2D projection of [`evaluate`](Self::evaluate) for visualisation.
    ///
    /// `x` and `y` drive the fill fractions of the first two blocks; every
    /// remaining block is held at all-ones (zero cost), so the rendered slice
    /// passes through the global optimum at the `(1, 1)` corner. Each driven
    /// block uses a continuous relaxation of the trap cost:
    ///
    /// ```text
    ///   cost(p) = 1 + (k − 1)·p   for p < 1   (deceptive ramp away from 1)
    ///   cost(p) = 0               for p >= 1  (the rewarding all-ones state)
    /// ```
    ///
    /// `x`/`y` are clamped to `[0, 1]`. When `num_blocks == 1` only `x` drives
    /// the surface; `y` is ignored.
    fn evaluate_2d(&self, x: f64, y: f64) -> f64 {
        let k = self.block_size;
        // Continuous relaxation of the per-block trap cost over a fill fraction.
        #[allow(clippy::cast_precision_loss)]
        let relaxed = |p: f64| -> f64 {
            let p = p.clamp(0.0, 1.0);
            if p >= 1.0 {
                0.0
            } else {
                1.0 + (k.saturating_sub(1)) as f64 * p
            }
        };
        let mut cost = relaxed(x);
        if self.num_blocks >= 2 {
            cost += relaxed(y);
        }
        cost
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for ConcatenatedTrap {
    fn render_ascii(&self) -> String {
        super::render::render_landscape_ascii(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "ConcatenatedTrap",
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::render::render_landscape_styled(
            |x, y| self.evaluate_2d(x, y),
            self.bounds(),
            "ConcatenatedTrap",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dim_is_blocks_times_block_size() {
        let t = ConcatenatedTrap::new(4, 5);
        assert_eq!(t.dim(), 20, "4 blocks of 5 bits must yield dim 20");
    }

    #[test]
    fn global_minimum_all_ones() {
        let t = ConcatenatedTrap::new(4, 5);
        assert_relative_eq!(t.evaluate(&[1.0; 20]), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn deceptive_optimum_all_zeros_costs_num_blocks() {
        let t = ConcatenatedTrap::new(4, 5);
        // Each all-zeros block costs cost(0) = 1, so 4 blocks cost 4.
        assert_relative_eq!(t.evaluate(&[0.0; 20]), 4.0, epsilon = 1e-6);
    }

    #[test]
    fn single_flip_from_all_zeros_is_strictly_worse() {
        let t = ConcatenatedTrap::new(4, 5);
        let baseline = t.evaluate(&[0.0; 20]);
        for i in 0..t.dim() {
            let mut x = [0.0_f64; 20];
            x[i] = 1.0;
            let flipped = t.evaluate(&x);
            assert!(
                flipped > baseline,
                "flipping bit {i} alone must be strictly worse than all-zeros"
            );
            // The affected block moves from cost(0)=1 to cost(1)=2, lifting the
            // total from 4 to exactly 5.
            assert_relative_eq!(flipped, 5.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn per_block_cost_curve_matches_trap() {
        let t = ConcatenatedTrap::new(1, 5);
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 0.0];
        for (u, &want) in expected.iter().enumerate() {
            let mut x = [0.0_f64; 5];
            for bit in x.iter_mut().take(u) {
                *bit = 1.0;
            }
            assert_relative_eq!(t.evaluate(&x), want, epsilon = 1e-6);
        }
    }

    #[test]
    fn additively_decomposable_across_blocks() {
        let block = ConcatenatedTrap::new(1, 5);
        let two = ConcatenatedTrap::new(2, 5);
        // Two distinct blocks: one with u=2 leading ones, one with u=3.
        let a = [1.0, 1.0, 0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0, 0.0, 0.0];
        let mut joined = [0.0_f64; 10];
        joined[..5].copy_from_slice(&a);
        joined[5..].copy_from_slice(&b);
        assert_relative_eq!(
            two.evaluate(&joined),
            block.evaluate(&a) + block.evaluate(&b),
            epsilon = 1e-6,
        );
    }

    #[test]
    fn threshold_rounds_genes_to_bits() {
        let t = ConcatenatedTrap::new(1, 5);
        // 0.49 -> 0, 0.5 -> 1, 0.51 -> 1, 1.0 -> 1, 0.99 -> 1 ⇒ u = 4 ⇒ cost 5.
        assert_relative_eq!(
            t.evaluate(&[0.5, 0.49, 0.51, 1.0, 0.99]),
            5.0,
            epsilon = 1e-6,
        );
    }

    #[test]
    #[should_panic(expected = "input dimension mismatch")]
    fn evaluate_panics_on_dimension_mismatch() {
        let t = ConcatenatedTrap::new(4, 5);
        let _ = t.evaluate(&[0.0; 19]);
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let t = ConcatenatedTrap::new(4, 5);
        let plain_no_trailing: String = t.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(
            t.render_styled().plain_text(),
            plain_no_trailing,
            "styled glyphs must match the plain ASCII render",
        );
    }

    #[test]
    fn render_styled_uses_best_palette() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let t = ConcatenatedTrap::new(4, 5);
        let styled = t.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "ConcatenatedTrap")
            .expect("ConcatenatedTrap label span present");
        assert_eq!(
            label.style.fg,
            Some(BEST_FG),
            "label must carry the best-marker foreground colour",
        );
        assert!(
            label.style.modifier.contains(BEST_MODIFIER),
            "label must carry the best-marker modifier",
        );
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let t = ConcatenatedTrap::new(4, 5);
        for line in t.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
