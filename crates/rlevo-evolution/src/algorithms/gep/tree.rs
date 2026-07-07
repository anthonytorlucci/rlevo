//! The decoded GEP phenotype: a level-order expression tree.

use crate::function_set::{FunctionSet, Symbol};

use super::alphabet::{Alphabet, SymbolKind};

/// Magnitude cap applied to a diverged node value in [`ExpressionTree::eval`].
///
/// Sized so its square stays finite (`1e15² = 1e30 < f32::MAX ≈ 3.4e38`), even
/// summed over a large dataset, so a clamped prediction still produces a finite
/// — but very large — MSE. A diverged (`±Inf`) subtree therefore ranks worst
/// rather than collapsing to a deceptively small error.
const EVAL_CLAMP: f32 = 1e15;

/// Node-value sanitizer for the GEP evaluator: `NaN → 0.0`, `±Inf → ±`
/// [`EVAL_CLAMP`], finite values unchanged.
///
/// `f32::clamp` *propagates* `NaN`, so `NaN` must be handled before the clamp
/// (mirrors the EDA `bayesian_network` finiteness convention). Unlike the
/// shared [`finite_or_zero`](crate::function_set::finite_or_zero) rule used by
/// CGP, this preserves the sign and magnitude of an overflowed value so a
/// bloated subtree is penalized, not hidden (finding tree.rs §1.2).
#[must_use]
#[inline]
fn finite_or_clamp(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v.clamp(-EVAL_CLAMP, EVAL_CLAMP)
    }
}

/// A decoded expression tree stored in level-order (breadth-first) layout.
///
/// The coding prefix of a chromosome decodes to this structure (see
/// [`GepDecoder`](super::GepDecoder)). Nodes are held in BFS order, so every
/// node's children occupy a contiguous, strictly-higher index range. That lets
/// evaluation run as a single right-to-left sweep: when a parent is reached,
/// its children (higher indices) have already been computed.
#[derive(Clone, Debug)]
pub struct ExpressionTree {
    /// Symbols in level order; `nodes[0]` is the root.
    nodes: Vec<Symbol>,
    /// `arities[i]` is the number of children of node `i` (cached at decode).
    arities: Vec<usize>,
    /// `child_start[i]` is the index of node `i`'s first child; its children
    /// are `child_start[i] .. child_start[i] + arities[i]`.
    child_start: Vec<usize>,
}

impl ExpressionTree {
    /// Builds a tree from its level-order parts.
    ///
    /// Intended for [`GepDecoder`](super::GepDecoder); the three vectors must
    /// be parallel and internally consistent (BFS child assignment).
    #[must_use]
    pub(super) fn from_parts(
        nodes: Vec<Symbol>,
        arities: Vec<usize>,
        child_start: Vec<usize>,
    ) -> Self {
        debug_assert_eq!(nodes.len(), arities.len());
        debug_assert_eq!(nodes.len(), child_start.len());
        Self {
            nodes,
            arities,
            child_start,
        }
    }

    /// Number of coding nodes (the open-reading-frame length).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Symbols in level order (root first). Primarily for tests/inspection.
    #[must_use]
    pub fn nodes(&self) -> &[Symbol] {
        &self.nodes
    }

    /// Tree depth: edges on the longest root-to-leaf path (a single-node tree
    /// has depth 0). A bloat metric.
    #[must_use]
    pub fn depth(&self) -> usize {
        let n = self.nodes.len();
        if n == 0 {
            return 0;
        }
        // Right-to-left: a node's children have higher indices, so they are
        // resolved first.
        let mut node_depth = vec![0usize; n];
        for i in (0..n).rev() {
            let arity = self.arities[i];
            if arity == 0 {
                node_depth[i] = 0;
            } else {
                let start = self.child_start[i];
                let mut max_child = 0;
                for k in 0..arity {
                    max_child = max_child.max(node_depth[start + k]);
                }
                node_depth[i] = 1 + max_child;
            }
        }
        node_depth[0]
    }

    /// Evaluates the tree on one input row.
    ///
    /// Variable nodes resolve to `inputs[input_index]` (missing indices read as
    /// `0.0`); constant nodes resolve to their stored value; function nodes call
    /// [`FunctionSet::apply`](crate::function_set::FunctionSet::apply) with their
    /// children's already-computed results, then sanitize the result via
    /// [`finite_or_clamp`]: a `NaN` collapses to `0.0` (it has no meaningful
    /// sign or magnitude), while `±Inf` clamps to `±`[`EVAL_CLAMP`] (sign
    /// preserved). Clamping — rather than zeroing — a diverged (`±Inf`) subtree
    /// keeps its magnitude large, so it yields a large MSE and ranks *worst*
    /// instead of masquerading as a perfect `0.0` predictor near zero-valued
    /// targets (GEP finding tree.rs §1.2). This differs from the CGP evaluator's
    /// `finite_or_zero` rule, which zeros both.
    ///
    /// `alphabet` must be the same one the tree was decoded with.
    #[must_use]
    pub fn eval<F: FunctionSet>(&self, alphabet: &Alphabet<F>, inputs: &[f32]) -> f32 {
        let n = self.nodes.len();
        if n == 0 {
            return 0.0;
        }
        let mut results = vec![0.0f32; n];
        let max_arity = alphabet.max_arity().max(1);
        let mut arg_buf = vec![0.0f32; max_arity];

        for i in (0..n).rev() {
            let symbol = self.nodes[i];
            results[i] = match alphabet.classify(symbol) {
                SymbolKind::Variable { input_index } => {
                    inputs.get(input_index).copied().unwrap_or(0.0)
                }
                SymbolKind::Constant { value } => value,
                SymbolKind::Function { .. } => {
                    let arity = self.arities[i];
                    let start = self.child_start[i];
                    arg_buf[..arity].copy_from_slice(&results[start..start + arity]);
                    let v = alphabet.functions.apply(symbol, &arg_buf[..arity]);
                    finite_or_clamp(v)
                }
            };
        }
        results[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::gep::GepDecoder;
    use crate::algorithms::gep::decode::GenotypePhenotypeMap;
    use crate::function_set::ArithmeticFunctionSet;

    fn alphabet(n_vars: usize) -> Alphabet<ArithmeticFunctionSet> {
        Alphabet::new(ArithmeticFunctionSet, n_vars, vec![])
    }

    /// Genome `[+, x, 1]` (ids: add=0, var x=8, const-1=7) decodes to `x + 1`.
    #[test]
    fn evaluates_x_plus_one() {
        let a = alphabet(1);
        // head [0, 8, 7], tail [8] (terminals). ORF = first 3 (add needs 2
        // children: x and const-1).
        let genome = vec![Symbol::from_raw(0), Symbol::from_raw(8), Symbol::from_raw(7), Symbol::from_raw(8)];
        let tree = GepDecoder.decode(&a, &genome);
        assert_eq!(tree.node_count(), 3);
        approx::assert_relative_eq!(tree.eval(&a, &[2.0]), 3.0, epsilon = 1e-6);
        approx::assert_relative_eq!(tree.eval(&a, &[-5.0]), -4.0, epsilon = 1e-6);
    }

    /// `[*, x, x]` decodes to `x * x` with depth 1.
    #[test]
    fn evaluates_x_squared_with_depth_one() {
        let a = alphabet(1);
        let genome = vec![Symbol::from_raw(2), Symbol::from_raw(8), Symbol::from_raw(8), Symbol::from_raw(8)];
        let tree = GepDecoder.decode(&a, &genome);
        assert_eq!(tree.node_count(), 3);
        assert_eq!(tree.depth(), 1);
        approx::assert_relative_eq!(tree.eval(&a, &[3.0]), 9.0, epsilon = 1e-6);
    }

    /// A single terminal head decodes to a depth-0, one-node tree.
    #[test]
    fn single_terminal_is_leaf() {
        let a = alphabet(1);
        let genome = vec![Symbol::from_raw(8), Symbol::from_raw(8), Symbol::from_raw(8)];
        let tree = GepDecoder.decode(&a, &genome);
        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.depth(), 0);
        approx::assert_relative_eq!(tree.eval(&a, &[7.0]), 7.0, epsilon = 1e-6);
    }

    /// `finite_or_clamp`: `NaN → 0.0`, `±Inf → ±EVAL_CLAMP`, finite passes.
    #[test]
    fn test_finite_or_clamp_zeroes_nan_and_clamps_inf() {
        approx::assert_relative_eq!(finite_or_clamp(f32::NAN), 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(finite_or_clamp(f32::INFINITY), EVAL_CLAMP);
        approx::assert_relative_eq!(finite_or_clamp(f32::NEG_INFINITY), -EVAL_CLAMP);
        approx::assert_relative_eq!(finite_or_clamp(3.5), 3.5, epsilon = 1e-6);
    }

    /// An overflowing `x * x` clamps to `EVAL_CLAMP` (not `0.0`), so a diverged
    /// tree carries a large magnitude and is penalized (finding tree.rs §1.2).
    #[test]
    fn test_eval_clamps_overflow_instead_of_zeroing() {
        let a = alphabet(1);
        // `[*, x, x]` = x * x; x = 1e30 overflows f32 to +Inf.
        let genome = vec![
            Symbol::from_raw(2),
            Symbol::from_raw(8),
            Symbol::from_raw(8),
            Symbol::from_raw(8),
        ];
        let tree = GepDecoder.decode(&a, &genome);
        let pred = tree.eval(&a, &[1e30]);
        approx::assert_relative_eq!(pred, EVAL_CLAMP);
        // The squared error against a near-zero target is huge, not near-zero:
        // the old `finite_or_zero` rule would have made `pred == 0.0` here.
        assert!(
            (pred - 0.1).powi(2) > 1e20,
            "diverged prediction must yield a large error, got pred = {pred}"
        );
    }
}
