//! Coupled fitness evaluation for co-evolutionary algorithms.
//!
//! In a single-population strategy, fitness is an absolute function of one
//! genome. In co-evolution it is *relational*: an individual's score depends
//! on the individuals in the other population(s) — predator vs. prey, sorter
//! vs. test case, sub-solution vs. complementary sub-solution. The
//! [`CoupledFitness`] trait captures exactly that joint evaluation.

use burn::tensor::{Tensor, backend::Backend};

/// Joint fitness evaluation across two or more co-evolving populations.
///
/// `evaluate_coupled` receives every population at once and returns one
/// fitness vector per population, each ranked under the crate-wide canonical
/// **maximise convention** (higher is better — see
/// [`crate::strategy::Strategy`]).
///
/// # Invariants
///
/// - The returned `Vec` has the same length as `populations`, and
///   `result[i]` has shape `(pop_size_i,)` matching `populations[i]`'s row
///   count.
/// - Row order is preserved: `result[i][r]` is the fitness of the individual
///   at row `r` of `populations[i]`.
/// - The trait is **N-population-ready**: it accepts a slice so a future
///   `N > 2` extension needs no breaking change. v1 algorithms always pass
///   exactly two populations, and implementors should
///   `debug_assert_eq!(populations.len(), 2)` to catch misuse without a
///   compile-time const-generic barrier.
///
/// # Examples
///
/// A trivial separable implementor scoring each population by its row sums:
///
/// ```
/// use burn::backend::Flex;
/// use burn::tensor::{Tensor, TensorData, backend::Backend};
/// use rlevo_evolution::coevolution::CoupledFitness;
///
/// struct RowSum;
/// impl<B: Backend> CoupledFitness<B> for RowSum {
///     fn evaluate_coupled(&self, populations: &[Tensor<B, 2>]) -> Vec<Tensor<B, 1>> {
///         debug_assert_eq!(populations.len(), 2);
///         populations
///             .iter()
///             .map(|p| p.clone().sum_dim(1).squeeze_dim::<1>(1))
///             .collect()
///     }
/// }
///
/// let device = Default::default();
/// let a = Tensor::<Flex, 2>::from_data(TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]), &device);
/// let b = Tensor::<Flex, 2>::from_data(TensorData::new(vec![0.0_f32, 0.0, 1.0, 1.0], [2, 2]), &device);
/// let out = RowSum.evaluate_coupled(&[a, b]);
/// assert_eq!(out.len(), 2);
/// ```
pub trait CoupledFitness<B: Backend>: Send + Sync {
    /// Evaluate fitness for each population relative to the others.
    ///
    /// `populations[i]` is a `(pop_size_i, genome_dim_i)` tensor. Returns one
    /// fitness vector per input population, each of length `pop_size_i`, on
    /// the same device as its population.
    fn evaluate_coupled(&self, populations: &[Tensor<B, 2>]) -> Vec<Tensor<B, 1>>;

    /// Per-population archive sizes, used to populate co-evolution metrics.
    ///
    /// The default returns an empty `Vec` — a plain fitness function keeps no
    /// archive. The [`HallOfFameFitness`](crate::coevolution::HallOfFameFitness)
    /// wrapper overrides this to report each population's hall-of-fame size so
    /// [`CoEAMetrics`](crate::coevolution::CoEAMetrics) can carry
    /// `hof_size_{a,b}`. Adding this defaulted method is forward-compatible:
    /// existing implementors are unaffected.
    fn archive_sizes(&self) -> Vec<usize> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Flex, Wgpu};
    use burn::tensor::TensorData;

    /// Trivial implementor: each individual's fitness is its genome row sum.
    struct RowSum;

    impl<B: Backend> CoupledFitness<B> for RowSum {
        fn evaluate_coupled(&self, populations: &[Tensor<B, 2>]) -> Vec<Tensor<B, 1>> {
            debug_assert_eq!(populations.len(), 2);
            populations
                .iter()
                .map(|p| p.clone().sum_dim(1).squeeze_dim::<1>(1))
                .collect()
        }
    }

    /// `CoupledFitness<B>` compiles on the CPU (Flex / ndarray-family)
    /// **and** wgpu backends with a trivial test implementor. This is a
    /// compile-time monomorphization assertion — it never runs (no wgpu device
    /// is created), so it is cheap and CI-safe.
    const _: fn() = || {
        fn assert_coupled<B: Backend, F: CoupledFitness<B>>() {}
        assert_coupled::<Flex, RowSum>();
        assert_coupled::<Wgpu, RowSum>();
    };

    #[test]
    fn trivial_implementor_evaluates_and_preserves_row_order() {
        let device = Default::default();
        let a = Tensor::<Flex, 2>::from_data(TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]), &device);
        let b = Tensor::<Flex, 2>::from_data(TensorData::new(vec![0.0_f32, 0.0, 1.0, 1.0], [2, 2]), &device);
        let out = RowSum.evaluate_coupled(&[a, b]);
        assert_eq!(out.len(), 2);
        let va = out[0].clone().into_data().into_vec::<f32>().unwrap();
        let vb = out[1].clone().into_data().into_vec::<f32>().unwrap();
        assert_eq!(va, vec![3.0, 7.0]);
        assert_eq!(vb, vec![0.0, 2.0]);
    }

    #[test]
    fn default_archive_sizes_is_empty() {
        assert!(CoupledFitness::<Flex>::archive_sizes(&RowSum).is_empty());
    }
}
