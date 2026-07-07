//! Genome category trait and its zero-sized marker types.
//!
//! [`GenomeKind`] tags genome representations at the type level so operators
//! can specialize on the element semantics (real-valued, binary, integer,
//! or tree). Strategies take a marker type as a const generic to pick the
//! right operator set.
//!
//! The markers themselves carry no data — they exist purely to discriminate
//! trait impls.

use std::fmt::Debug;

use burn::tensor::{Int, Tensor, backend::Backend};

/// Shape-erased genome kind.
///
/// `GenomeKind` is a zero-sized marker that strategies parameterize on to
/// pick operators. Concrete kinds (`Real`, `Binary`, `Integer`, `Tree`,
/// `Permutation`) live below; new kinds can be added by implementing this
/// trait on a fresh marker type.
///
/// Genome width is a runtime property (`Population::genome_dim`), not a
/// type-level one: every shipping kind is either runtime-dimensioned
/// (`Real`/`Binary`/`Integer`) or variable-length (`Tree`/`Permutation`). A
/// structurally-fixed-width kind could add a compile-time length constant when
/// one is introduced — an associated const with a default is a non-breaking
/// addition.
pub trait GenomeKind: Debug + Copy + Send + Sync + 'static {
    /// Element type of the genome (typically `f32`, `i32`, or `bool`).
    type Element: Copy + Debug + Send + Sync + 'static;
}

/// Real-valued genome (each gene is an `f32`).
///
/// Populations are stored as `Tensor<B, 2>` of shape `(pop_size, dim)`.
/// All classical ES variants, real-coded GA, EP, and DE use this kind.
#[derive(Debug, Clone, Copy, Default)]
pub struct Real;

impl GenomeKind for Real {
    type Element = f32;
}

/// Binary genome (each gene is a bit, stored as `i32` 0/1 on device).
///
/// Populations are stored as `Tensor<B, 2, Int>` of shape
/// `(pop_size, dim)`. Binary-coded GA uses this kind.
#[derive(Debug, Clone, Copy, Default)]
pub struct Binary;

impl GenomeKind for Binary {
    type Element = i32;
}

/// Integer-valued genome (each gene is a non-negative integer index).
///
/// Populations are stored as `Tensor<B, 2, Int>` of shape
/// `(pop_size, dim)`. Cartesian GP (node indices), discrete parameter
/// search, and other problems where genes are bounded integer values use
/// this kind. For ordered-sequence problems (TSP, QAP) use [`Permutation`]
/// instead.
#[derive(Debug, Clone, Copy, Default)]
pub struct Integer;

/// Tree-based genome (variable-length AST, stored host-side).
///
/// Reserved for classical Koza-style GP in a future release. Tree
/// genomes cannot be batched on a GPU and therefore have no tensor
/// representation in this crate. The associated `Element` type is `i32`
/// as a placeholder (structural node IDs); the actual in-memory
/// representation will be defined when this kind is fully implemented.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tree;

/// Permutation genome (each row is a permutation of `0..n_nodes`).
///
/// Rectangular and device-resident: populations are stored as
/// `Tensor<B, 2, Int>` of shape `(pop_size, n_nodes)`, so this kind
/// implements [`TensorGenome`] and `Population<B, Permutation>` type-checks.
/// The permutation invariant (every row is a bijection of `0..n_nodes`) is
/// **not** enforced at construction — [`new_permutation`] validates only
/// shape, matching how `Binary`/`Integer` leave element values unchecked.
///
/// Used by Ant Colony Optimization over combinatorial domains (TSP, QAP, …);
/// only a stubbed consumer ships in this release — the operators are planned
/// for a future release.
///
/// [`new_permutation`]: crate::population::Population::new_permutation
#[derive(Debug, Clone, Copy, Default)]
pub struct Permutation;

impl GenomeKind for Integer {
    type Element = i32;
}

impl GenomeKind for Tree {
    type Element = i32;
}

impl GenomeKind for Permutation {
    type Element = i32;
}

/// Genome kinds with a rectangular, device-resident tensor representation.
///
/// This is the subset of [`GenomeKind`]s that a
/// [`Population`](crate::population::Population) can store on-device. The
/// associated [`Tensor`](TensorGenome::Tensor) type names *which* tensor flavour
/// backs the kind, tying the storage type to the marker at compile time: `Real`
/// maps to `Tensor<B, 2>`; `Binary`, `Integer`, and `Permutation` map to
/// `Tensor<B, 2, Int>`.
///
/// Because the storage type is chosen by the trait impl, `Population<B, K>` needs
/// only one field and no run-time tag — the wrong-tensor-for-this-kind state is
/// simply unrepresentable. Variable-length kinds such as [`Tree`] have no
/// rectangular form and deliberately do not implement this trait, so
/// `Population<B, Tree>` is not a valid type.
pub trait TensorGenome: GenomeKind {
    /// Device tensor type storing a whole population of this kind, shape
    /// `(pop_size, genome_dim)`.
    ///
    /// The `Send + Sync` bound keeps `Population<B, K>` thread-safe under the
    /// crate-wide [`Strategy: Send + Sync`](crate::strategy::Strategy) contract:
    /// without it, generic code over `K: TensorGenome` could not prove the
    /// stored tensor — and therefore the population — is shareable across
    /// threads, even though every concrete Burn tensor is.
    type Tensor<B: Backend>: Clone + Debug + Send + Sync;
}

impl TensorGenome for Real {
    type Tensor<B: Backend> = Tensor<B, 2>;
}

impl TensorGenome for Binary {
    type Tensor<B: Backend> = Tensor<B, 2, Int>;
}

impl TensorGenome for Integer {
    type Tensor<B: Backend> = Tensor<B, 2, Int>;
}

impl TensorGenome for Permutation {
    type Tensor<B: Backend> = Tensor<B, 2, Int>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_has_f32_element() {
        let _: <Real as GenomeKind>::Element = 0.0_f32;
    }

    #[test]
    fn binary_has_i32_element() {
        let _: <Binary as GenomeKind>::Element = 1_i32;
    }

    #[test]
    fn integer_has_i32_element() {
        let _: <Integer as GenomeKind>::Element = 5_i32;
    }

    #[test]
    fn permutation_has_i32_element() {
        let _: <Permutation as GenomeKind>::Element = 7_i32;
    }

    #[test]
    fn markers_are_debug() {
        let _ = format!("{Real:?} {Binary:?} {Integer:?} {Tree:?} {Permutation:?}");
    }

    #[test]
    fn permutation_impls_tensor_genome() {
        use burn::backend::Flex;

        // Compiles iff `Permutation: TensorGenome` with its tensor flavour
        // unifying with `Tensor<Flex, 2, Int>`.
        fn takes_perm_tensor(
            t: <Permutation as TensorGenome>::Tensor<Flex>,
        ) -> Tensor<Flex, 2, Int> {
            t
        }
        let _ = takes_perm_tensor;
    }

    #[test]
    fn tensor_genome_storage_is_send_sync() {
        use burn::backend::Flex;

        // Regression guard for the `Send + Sync` GAT bound: instantiating this
        // for each kind fails to compile if the bound is ever dropped.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<<Real as TensorGenome>::Tensor<Flex>>();
        assert_send_sync::<<Binary as TensorGenome>::Tensor<Flex>>();
        assert_send_sync::<<Integer as TensorGenome>::Tensor<Flex>>();
        assert_send_sync::<<Permutation as TensorGenome>::Tensor<Flex>>();
    }
}
