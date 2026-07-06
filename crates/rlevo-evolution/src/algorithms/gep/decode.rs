//! Genotype → phenotype decoding (the head/tail → expression-tree map).

use crate::function_set::{FunctionSet, Symbol};

use super::alphabet::Alphabet;
use super::tree::ExpressionTree;

/// Maps a linear GEP chromosome to its expression-tree phenotype.
///
/// Implementors decode `genome` (a head/tail symbol string) against `alphabet`
/// into an [`ExpressionTree`]. The decode is deterministic: the same genome and
/// alphabet always yield the same tree.
pub trait GenotypePhenotypeMap<F: FunctionSet> {
    /// Decodes a chromosome into its phenotype.
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
        let genome = vec![Symbol::from_raw(8), Symbol::from_raw(8), Symbol::from_raw(8)];
        let tree = GepDecoder.decode(&a, &genome);
        assert_eq!(tree.node_count(), 1);
    }
}
