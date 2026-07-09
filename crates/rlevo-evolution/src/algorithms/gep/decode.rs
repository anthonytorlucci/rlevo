//! Genotype → phenotype decoding (the head/tail → expression-tree map).

use crate::function_set::{FunctionSet, Symbol};

use super::alphabet::Alphabet;
use super::tree::ExpressionTree;

/// Maps a linear GEP chromosome to its expression-tree phenotype.
///
/// Implementors decode `genome` (a head/tail symbol string) against `alphabet`
/// into an [`ExpressionTree`]. The decode is deterministic: the same genome and
/// alphabet always yield the same tree.
///
/// # Invariants
///
/// The `genome` is expected to satisfy the GEP head/tail layout: a head region
/// whose loci may hold functions or terminals, followed by a terminal-only tail
/// of length `t = h(n - 1) + 1`, where `h` is the head length and `n` the
/// alphabet's maximum arity (Ferreira 2001, eq. 3.4). Any chromosome produced by
/// the sampling and operator paths in this crate satisfies this by construction;
/// a hand-built genome that violates it is a precondition breach (see the
/// implementation's `# Panics`).
pub trait GenotypePhenotypeMap<F: FunctionSet> {
    /// Decodes a chromosome into its phenotype.
    ///
    /// # Arguments
    ///
    /// * `alphabet` — the symbol alphabet used to classify each locus (its
    ///   arity and kind); must be the same alphabet the `genome` was sampled
    ///   against.
    /// * `genome` — a linear head/tail chromosome (see the trait-level
    ///   `# Invariants`). An empty genome decodes to an empty tree.
    ///
    /// # Returns
    ///
    /// The [`ExpressionTree`] phenotype: the open-reading-frame coding region of
    /// `genome` in level order.
    ///
    /// # Panics
    ///
    /// Debug builds assert the head/tail precondition
    /// (`debug_assert!(needed == 0, ...)`); a genome that violates it panics here
    /// in debug. In release the assertion is absent — `decode` itself does not
    /// panic, but a malformed tree silently degrades to a finite-but-incorrect
    /// value on evaluation (see [`ExpressionTree::eval`]'s defensive clamp,
    /// issue #147).
    #[must_use]
    fn decode(&self, alphabet: &Alphabet<F>, genome: &[Symbol]) -> ExpressionTree;
}

/// The canonical Ferreira (2001) decoder: open-reading-frame scan, then
/// level-order (breadth-first) tree construction.
///
/// # Algorithm
///
/// 1. **ORF-length pass.** A single left-to-right scan tracks the number of
///    still-unfilled child slots, starting at 1 (the root). Each symbol fills
///    one slot and opens `arity` new ones; the coding region ends when no slots
///    remain. The tail-length constraint guarantees this terminates within the
///    chromosome.
/// 2. **BFS child assignment.** Because the coding prefix is already in level
///    order, a node's children are simply the next unread symbols. A running
///    read cursor assigns each node `arity` contiguous children — no explicit
///    queue is needed, since array order *is* BFS order.
///
/// # Examples
///
/// ```
/// use rlevo_evolution::algorithms::gep::{Alphabet, GenotypePhenotypeMap, GepDecoder};
/// use rlevo_evolution::function_set::ArithmeticFunctionSet;
/// use rlevo_evolution::rng::{seed_stream, SeedPurpose};
///
/// // A one-variable arithmetic alphabet (functions over a single `x`).
/// let alphabet = Alphabet::new(ArithmeticFunctionSet, 1, vec![]);
///
/// // Sample a valid head/tail genome: head loci draw any symbol, tail loci
/// // draw terminals only, with the tail sized per Ferreira (2001) eq. 3.4.
/// let mut rng = seed_stream(42, 0, SeedPurpose::Mutation);
/// let head_len = 3;
/// let tail_len = head_len * (alphabet.max_arity() - 1) + 1;
/// let mut genome = Vec::with_capacity(head_len + tail_len);
/// for _ in 0..head_len {
///     genome.push(alphabet.sample_head_symbol(&mut rng));
/// }
/// for _ in 0..tail_len {
///     genome.push(alphabet.sample_tail_symbol(&mut rng));
/// }
///
/// // Decode the linear chromosome into its expression-tree phenotype.
/// let tree = GepDecoder.decode(&alphabet, &genome);
/// assert!(tree.node_count() >= 1);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct GepDecoder;

impl<F: FunctionSet> GenotypePhenotypeMap<F> for GepDecoder {
    fn decode(&self, alphabet: &Alphabet<F>, genome: &[Symbol]) -> ExpressionTree {
        if genome.is_empty() {
            return ExpressionTree::from_parts(Vec::new(), Vec::new(), Vec::new());
        }

        // 1. ORF-length pass.
        let mut needed: usize = 1;
        let mut orf_len = 0;
        while needed > 0 && orf_len < genome.len() {
            let arity = alphabet.arity(genome[orf_len]);
            needed = needed - 1 + arity;
            orf_len += 1;
        }

        // A well-formed GEP gene (head ∈ F∪T, tail ∈ T strictly, tail length
        // t = h(n−1)+1 with n = max arity) always satisfies every open child
        // slot within the chromosome, so the scan must exit with `needed == 0`
        // (Ferreira 2001, Complex Systems 13(2), §3.2 eq. 3.4). A residual
        // `needed > 0` can only arise from a contract-violating genome (e.g.
        // one hand-built via `Symbol::from_raw` that bypasses the head/tail
        // rule): its child ranges would overrun `node_count()` and later panic
        // in `eval`. Flag that precondition breach in debug builds; `eval`
        // carries a matching release-time clamp so the failure degrades to a
        // finite value rather than an out-of-bounds slice.
        debug_assert!(
            needed == 0,
            "genome violates GEP head/tail invariant t = h(n-1)+1 (Ferreira \
             2001 eq. 3.4): {needed} child slot(s) left unfilled"
        );

        // 2. Level-order parts. Children are the next unread symbols; a single
        //    read cursor walks them in BFS order.
        let nodes: Vec<Symbol> = genome[..orf_len].to_vec();
        let mut arities = Vec::with_capacity(orf_len);
        let mut child_start = Vec::with_capacity(orf_len);
        let mut read = 1usize;
        for &symbol in &nodes {
            let arity = alphabet.arity(symbol);
            arities.push(arity);
            child_start.push(read);
            read += arity;
        }

        ExpressionTree::from_parts(nodes, arities, child_start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function_set::ArithmeticFunctionSet;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn alphabet(n_vars: usize) -> Alphabet<ArithmeticFunctionSet> {
        Alphabet::new(ArithmeticFunctionSet, n_vars, vec![])
    }

    /// ORF length for `[+, *, x, x, x, ...]`: root + needs 2 (the * and the
    /// last x), the * needs 2 more (x, x). 5 coding symbols.
    #[test]
    fn orf_length_for_nested_tree() {
        let a = alphabet(1);
        // ids: + = 0, * = 2, x = 8
        let genome = vec![
            Symbol::from_raw(0),
            Symbol::from_raw(2),
            Symbol::from_raw(8),
            Symbol::from_raw(8),
            Symbol::from_raw(8),
            Symbol::from_raw(8), // tail junk
            Symbol::from_raw(8),
        ];
        let tree = GepDecoder.decode(&a, &genome);
        // + (root) -> children {*, x}; * -> children {x, x}. 5 nodes.
        assert_eq!(tree.node_count(), 5);
    }

    /// Decode is deterministic: same genome twice -> identical node lists.
    #[test]
    fn decode_is_deterministic() {
        let a = alphabet(2);
        let genome = vec![
            Symbol::from_raw(0),
            Symbol::from_raw(1),
            Symbol::from_raw(8),
            Symbol::from_raw(9),
            Symbol::from_raw(8),
            Symbol::from_raw(9),
            Symbol::from_raw(8),
        ];
        let t1 = GepDecoder.decode(&a, &genome);
        let t2 = GepDecoder.decode(&a, &genome);
        assert_eq!(t1.nodes(), t2.nodes());
        assert_eq!(t1.node_count(), t2.node_count());
    }

    /// An all-terminal head yields a single-node ORF.
    #[test]
    fn all_terminal_head_is_one_node() {
        let a = alphabet(1);
        let genome = vec![
            Symbol::from_raw(8),
            Symbol::from_raw(8),
            Symbol::from_raw(8),
        ];
        let tree = GepDecoder.decode(&a, &genome);
        assert_eq!(tree.node_count(), 1);
    }

    // §7.1 -----------------------------------------------------------------

    /// An empty genome decodes to an empty (zero-node) tree without panic.
    #[test]
    fn empty_genome_decodes_to_zero_node_tree() {
        let a = alphabet(1);
        let tree = GepDecoder.decode(&a, &[]);
        assert_eq!(tree.node_count(), 0);
        // A zero-node tree evaluates to the inert 0.0.
        approx::assert_relative_eq!(tree.eval(&a, &[1.0]), 0.0, epsilon = 1e-6);
    }

    // §7.3 -----------------------------------------------------------------

    /// A unary (arity-1) function head decodes to a two-node ORF: `sin(x)`.
    #[test]
    fn arity_one_function_decodes_two_nodes() {
        let a = alphabet(1);
        // ids: sin = 4 (arity 1), var x = 8. head [4, 8], tail [8].
        let genome = vec![
            Symbol::from_raw(4),
            Symbol::from_raw(8),
            Symbol::from_raw(8),
        ];
        let tree = GepDecoder.decode(&a, &genome);
        // sin (root) -> child {x}. 2 nodes.
        assert_eq!(tree.node_count(), 2);
        approx::assert_relative_eq!(tree.eval(&a, &[0.0]), 0.0f32.sin(), epsilon = 1e-6);
    }

    /// A maximally nested (full-tail) binary chromosome decodes to a deep tree
    /// that consumes the whole coding region.
    #[test]
    fn full_tail_deep_tree() {
        let a = alphabet(1);
        // head all binary `+` (id 0), length h = 4; tail all `x` (id 8),
        // length t = h(n-1)+1 = 4*1+1 = 5. A left-full binary tree of 4
        // internal nodes has 5 leaves: 9 coding nodes total.
        let mut genome = vec![Symbol::from_raw(0); 4];
        genome.extend(std::iter::repeat_n(Symbol::from_raw(8), 5));
        let tree = GepDecoder.decode(&a, &genome);
        assert_eq!(tree.node_count(), 9);
        // 4 additions of x summed over the tree: value is 5*x for x-leaves? No
        // — a chain of `+` with x leaves sums the 5 leaves = 5x.
        approx::assert_relative_eq!(tree.eval(&a, &[2.0]), 10.0, epsilon = 1e-6);
    }

    /// An out-of-range head symbol is treated as an inert arity-0 terminal, so
    /// it terminates the ORF at a single node rather than opening child slots.
    #[test]
    fn out_of_range_symbol_is_terminal() {
        let a = alphabet(1);
        // id 999 is beyond len(); classify() reports arity 0.
        let genome = vec![
            Symbol::from_raw(999),
            Symbol::from_raw(8),
            Symbol::from_raw(8),
        ];
        let tree = GepDecoder.decode(&a, &genome);
        assert_eq!(tree.node_count(), 1);
        approx::assert_relative_eq!(tree.eval(&a, &[3.0]), 0.0, epsilon = 1e-6);
    }

    // §7.4 -----------------------------------------------------------------

    /// The key guard (issue #147 §1.1). Ferreira (2001, eq. 3.4) guarantees any
    /// well-formed head/tail gene decodes to a complete tree: the ORF scan never
    /// leaves an unfilled child slot, so every child range stays in bounds and
    /// `eval` never slices past `node_count()`. Generate many random but
    /// well-formed genomes (head ∈ F∪T, tail ∈ T strictly, tail length
    /// t = h(n−1)+1) and assert the guarantee holds structurally and that `eval`
    /// returns a finite value without panic.
    #[test]
    fn wellformed_genomes_always_decode_in_bounds() {
        let mut rng = StdRng::seed_from_u64(0x9E37_79B9_7F4A_7C15);
        let max_arity = 2; // max arity of ArithmeticFunctionSet.
        for n_vars in 1..=3usize {
            let alpha = alphabet(n_vars);
            for _ in 0..2_000 {
                let head_len = 1 + rng_usize(&mut rng, 12); // head length 1..=12
                let tail_len = head_len * (max_arity - 1) + 1; // Ferreira eq. 3.4.
                let mut genome = Vec::with_capacity(head_len + tail_len);
                for _ in 0..head_len {
                    genome.push(alpha.sample_head_symbol(&mut rng));
                }
                for _ in 0..tail_len {
                    genome.push(alpha.sample_tail_symbol(&mut rng));
                }

                let tree = GepDecoder.decode(&alpha, &genome);
                let node_count = tree.node_count();
                assert!(node_count >= 1, "coding region must be non-empty");

                // Every non-root coding node is exactly one child of exactly one
                // parent, so the total child count equals node_count - 1. This
                // is the public-API restatement of `child_start[i] + arity[i]
                // <= node_count` for all i (BFS layout): the scan filled every
                // slot (Ferreira eq. 3.4), i.e. the decoder's `needed == 0`.
                let total_children: usize = tree.nodes().iter().map(|&sym| alpha.arity(sym)).sum();
                assert_eq!(
                    total_children + 1,
                    node_count,
                    "well-formed genome left an unfilled child slot: {genome:?}"
                );

                // `eval` must not panic and must return a finite value for any
                // input row (the finite_or_clamp guard neutralizes overflow).
                let inputs: Vec<f32> = (0..n_vars).map(|_| rng_input(&mut rng)).collect();
                let value = tree.eval(&alpha, &inputs);
                assert!(value.is_finite(), "eval produced non-finite {value}");
            }
        }
    }

    /// Small uniform `usize` in `0..bound` from a seeded RNG (avoids pulling in
    /// the distribution imports the alphabet already re-exports).
    fn rng_usize(rng: &mut StdRng, bound: usize) -> usize {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let hi = bound as i32;
        #[allow(clippy::cast_sign_loss)]
        {
            rng.random_range(0..hi) as usize
        }
    }

    /// A bounded, occasionally-large input to exercise the overflow clamp.
    fn rng_input(rng: &mut StdRng) -> f32 {
        rng.random_range(-1.0e6f32..1.0e6f32)
    }
}
