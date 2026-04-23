//! Zero-sized marker types for genome categories.
//!
//! [`GenomeKind`](evorl_core::evolution::GenomeKind) tags genome
//! representations at the type level so operators can specialize on the
//! element semantics (real-valued, binary, integer, or tree). Strategies
//! take a marker type as a const generic to pick the right operator set.
//!
//! The markers themselves carry no data — they exist purely to discriminate
//! trait impls.

use rlevo_core::evolution::GenomeKind;

/// Real-valued genome (each gene is an `f32`).
///
/// Populations are stored as `Tensor<B, 2>` of shape `(pop_size, dim)`.
/// All classical ES variants, real-coded GA, EP, and DE use this kind.
#[derive(Debug, Clone, Copy, Default)]
pub struct Real;

impl GenomeKind for Real {
    const DIM: usize = 0;
    type Element = f32;
}

/// Binary genome (each gene is a bit, stored as `i32` 0/1 on device).
///
/// Populations are stored as `Tensor<B, 2, Int>` of shape
/// `(pop_size, dim)`. Binary-coded GA uses this kind.
#[derive(Debug, Clone, Copy, Default)]
pub struct Binary;

impl GenomeKind for Binary {
    const DIM: usize = 0;
    type Element = i32;
}

/// Integer-valued genome (each gene is a non-negative integer index).
///
/// Populations are stored as `Tensor<B, 2, Int>` of shape
/// `(pop_size, dim)`. Permutation-coded GA and Cartesian GP use this kind.
#[derive(Debug, Clone, Copy, Default)]
pub struct Integer;

/// Tree-based genome (variable-length AST, stored host-side).
///
/// Reserved for classical Koza-style GP in a future release. Tree
/// genomes cannot be batched on a GPU and therefore have no tensor
/// representation in this crate.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tree;

/// Permutation genome (each row is a permutation of `0..n_nodes`).
///
/// Populations are stored as `Tensor<B, 2, Int>` of shape
/// `(pop_size, n_nodes)` where every row is a valid permutation. Used by
/// Ant Colony Optimization over combinatorial domains (TSP, QAP, …);
/// only a stubbed consumer ships in this release — a full implementation
/// is planned for a future release.
#[derive(Debug, Clone, Copy, Default)]
pub struct Permutation;

impl GenomeKind for Integer {
    const DIM: usize = 0;
    type Element = i32;
}

impl GenomeKind for Tree {
    const DIM: usize = 0;
    type Element = i32;
}

impl GenomeKind for Permutation {
    const DIM: usize = 0;
    type Element = i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_has_f32_element() {
        let _x: <Real as GenomeKind>::Element = 0.0_f32;
    }

    #[test]
    fn binary_has_i32_element() {
        let _x: <Binary as GenomeKind>::Element = 1_i32;
    }

    #[test]
    fn integer_has_i32_element() {
        let _x: <Integer as GenomeKind>::Element = 5_i32;
    }

    #[test]
    fn permutation_has_i32_element() {
        let _x: <Permutation as GenomeKind>::Element = 7_i32;
    }

    #[test]
    fn markers_are_debug() {
        let _ = format!(
            "{:?} {:?} {:?} {:?} {:?}",
            Real, Binary, Integer, Tree, Permutation
        );
    }
}
