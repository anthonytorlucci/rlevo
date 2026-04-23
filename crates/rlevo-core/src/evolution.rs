//! Minimal core traits for evolutionary optimization.
//!
//! This module contains the shape-erased vocabulary that the rest of the
//! workspace — in particular [`rlevo-evolution`][rlevo-evolution] — builds
//! on. Concrete strategies, genome containers, and operators live in
//! `rlevo-evolution`; this module only fixes the semantic contracts.
//!
//! # Traits
//!
//! - [`Fitness`] — scalar objective value with a well-defined "worst" element
//!   and a finiteness predicate. Blanket-implemented for `f32` and `f64`.
//! - [`GenomeKind`] — marker trait that tags genome categories at the type
//!   level (real, binary, integer, tree). Concrete kinds live in
//!   `rlevo-evolution::genome`.
//! - [`MultiFitness`] — minimal multi-objective fitness view. Pareto
//!   dominance and multi-objective selection machinery live in
//!   `rlevo-evolution` with the NSGA family.
//!
//! # Design notes
//!
//! The harness convention is "higher is better" (rewards). Strategies that
//! minimize internally are free to negate on input; that choice is not
//! encoded in the trait. See the `rlevo-evolution` crate for the
//! algorithm-by-algorithm convention.
//!
//! [rlevo-evolution]: https://docs.rs/rlevo-evolution

use std::fmt::Debug;

/// Scalar fitness contract.
///
/// Implementors must provide a [`worst`](Fitness::worst) sentinel used to
/// initialize best-so-far trackers, an [`is_finite`](Fitness::is_finite)
/// predicate so strategies can filter diverging evaluations, and an
/// [`as_f32`](Fitness::as_f32) conversion for metric reporting.
///
/// The trait is `Copy` to avoid lifetime plumbing inside tight generation
/// loops; fitness is always a scalar in the classical families.
///
/// # Examples
///
/// ```
/// use rlevo_core::evolution::Fitness;
///
/// let worst = <f32 as Fitness>::worst();
/// assert!(!worst.is_finite());
///
/// let good: f32 = 1.5;
/// assert!(good.is_finite());
/// assert_eq!(good.as_f32(), 1.5);
/// ```
pub trait Fitness: Copy + PartialOrd + Debug + Send + Sync + 'static {
    /// Returns the sentinel used to initialize "best-so-far" trackers.
    ///
    /// The convention is `Fitness::worst()` compares strictly less than any
    /// finite real fitness under `PartialOrd`, so comparisons of the form
    /// `candidate > best` initialize correctly.
    fn worst() -> Self;

    /// Whether this fitness value represents a finite real number.
    ///
    /// NaN and infinity both return `false`. Strategies use this to discard
    /// diverged candidates without poisoning selection.
    fn is_finite(&self) -> bool;

    /// Lossy conversion to `f32` for metric reporting and logging.
    fn as_f32(&self) -> f32;
}

impl Fitness for f32 {
    fn worst() -> Self {
        f32::NEG_INFINITY
    }

    fn is_finite(&self) -> bool {
        f32::is_finite(*self)
    }

    fn as_f32(&self) -> f32 {
        *self
    }
}

impl Fitness for f64 {
    fn worst() -> Self {
        f64::NEG_INFINITY
    }

    fn is_finite(&self) -> bool {
        f64::is_finite(*self)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn as_f32(&self) -> f32 {
        *self as f32
    }
}

/// Shape-erased genome kind.
///
/// `GenomeKind` is a zero-sized marker that strategies parameterize on to
/// pick operators. Concrete kinds (`Real`, `Binary`, `Integer`, `Tree`) live
/// in `evorl-evolution::genome`; keeping only the trait here lets
/// downstream crates add new kinds without modifying `evorl-core`.
///
/// The associated constant [`DIM`](GenomeKind::DIM) records the genome
/// dimensionality at the type level when it is compile-time known (for
/// variable-length representations like trees, impls set it to `0`).
pub trait GenomeKind: Debug + Copy + Send + Sync + 'static {
    /// Compile-time genome dimensionality, or `0` for variable-length kinds.
    const DIM: usize;

    /// Element type of the genome (typically `f32`, `i32`, or `bool`).
    type Element: Copy + Debug + Send + Sync + 'static;
}

/// Multi-objective fitness view.
///
/// This trait is deliberately minimal: it only exposes the slice of
/// objective values. Pareto dominance, non-dominated sorting, crowding
/// distance, and selection operators live in `evorl-evolution` where they
/// are actually used by the NSGA family. Keeping those algorithms out of
/// `evorl-core` means this crate has no opinion on how many objectives are
/// supported or how ties are broken.
///
/// # Examples
///
/// ```
/// use evorl_core::evolution::MultiFitness;
///
/// #[derive(Debug, Clone)]
/// struct Bi { values: [f32; 2] }
///
/// impl MultiFitness for Bi {
///     fn objectives(&self) -> &[f32] { &self.values }
/// }
///
/// let f = Bi { values: [1.0, -2.0] };
/// assert_eq!(f.objectives(), &[1.0, -2.0]);
/// ```
pub trait MultiFitness: Debug + Clone + Send + Sync {
    /// Returns the objective values associated with an individual.
    ///
    /// The slice length must remain stable across calls; strategies may
    /// assume a fixed number of objectives per run.
    fn objectives(&self) -> &[f32];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitness_f32_worst_is_less_than_any_finite() {
        let worst = <f32 as Fitness>::worst();
        assert!(worst < 0.0);
        assert!(worst < -1e30);
        assert!(worst < f32::MIN);
    }

    #[test]
    fn fitness_f64_worst_is_less_than_any_finite() {
        let worst = <f64 as Fitness>::worst();
        assert!(worst < 0.0);
        assert!(worst < -1e300);
        assert!(worst < f64::MIN);
    }

    #[test]
    fn fitness_is_finite_excludes_nan_and_inf() {
        let nan = f32::NAN;
        let pos_inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;
        let finite: f32 = 2.5;

        assert!(!<f32 as Fitness>::is_finite(&nan));
        assert!(!<f32 as Fitness>::is_finite(&pos_inf));
        assert!(!<f32 as Fitness>::is_finite(&neg_inf));
        assert!(<f32 as Fitness>::is_finite(&finite));
    }

    #[test]
    fn fitness_as_f32_f64_truncates() {
        let v: f64 = 1.5;
        assert_eq!(v.as_f32(), 1.5_f32);
    }

    #[derive(Debug, Clone, Copy)]
    struct TestKind;
    impl GenomeKind for TestKind {
        const DIM: usize = 7;
        type Element = f32;
    }

    #[test]
    fn genome_kind_records_dim_and_element() {
        assert_eq!(TestKind::DIM, 7);
        let _elem: <TestKind as GenomeKind>::Element = 0.0_f32;
    }

    #[derive(Debug, Clone)]
    struct ThreeObj {
        values: Vec<f32>,
    }

    impl MultiFitness for ThreeObj {
        fn objectives(&self) -> &[f32] {
            &self.values
        }
    }

    #[test]
    fn multi_fitness_exposes_objectives() {
        let f = ThreeObj {
            values: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(f.objectives(), &[1.0, 2.0, 3.0]);
        assert_eq!(f.objectives().len(), 3);
    }

    #[test]
    fn multi_fitness_clone_preserves_objectives() {
        let f = ThreeObj {
            values: vec![0.5, -0.5],
        };
        let g = f.clone();
        assert_eq!(f.objectives(), g.objectives());
    }
}
