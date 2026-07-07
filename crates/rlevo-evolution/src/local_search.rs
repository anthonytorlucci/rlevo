//! Host-side local-search refinement for memetic algorithms.
//!
//! A *memetic algorithm* (MA) interleaves a population-level evolutionary
//! strategy with a per-individual **local search** that polishes promising
//! genomes between generations. This module defines the local-search seam —
//! the [`LocalSearch`] trait — together with the four reference searchers
//! ([`HillClimbing`], [`NelderMead`], [`SimulatedAnnealing`],
//! [`RandomRestart`]) that a `MemeticWrapper` can compose with any existing
//! [`Strategy`](crate::strategy::Strategy).
//!
//! # Design seam
//!
//! Local search here is **host-side and gradient-free**. Searchers operate on
//! a flat `Vec<f32>` genome and a single-member
//! [`FitnessFn`], never touching device tensors or
//! computing gradients. This keeps refinement trivially reproducible and
//! independent of the Burn backend: it is pure host arithmetic plus calls to
//! the supplied fitness function.
//!
//! # Stochasticity convention
//!
//! Every searcher takes a `&mut dyn Rng` and **all** randomness flows through
//! it. Searchers never seed the process-wide backend RNG (`B::seed` +
//! `Tensor::random`) and never reach for `rand::rng()` / thread-local RNGs —
//! that would race the global Flex mutex under the parallel test runner and
//! destroy reproducibility (see [`crate::rng`] for the host-RNG convention).
//! A memetic wrapper derives a dedicated stream via
//! `seed_stream(base, generation, SeedPurpose::LocalSearch)` and threads it in
//! here.
//!
//! # Evaluation budget
//!
//! Each `refine` call is bounded by `Params::max_iters` **total**
//! [`FitnessFn::evaluate_one`] calls
//! (including the mandatory first re-evaluation of the input genome). The
//! shared `BudgetedEval` helper enforces this so the bound holds structurally
//! even on flat landscapes where no probe ever improves.
//!
//! [`refine_with_known_fitness`](LocalSearch::refine_with_known_fitness) skips
//! that seeding eval when the caller already knows the input's fitness, so the
//! same `max_iters` then buys one extra *probe* evaluation. The seeding eval
//! consumes no rng, so skipping it never shifts a searcher's random stream.

use burn::tensor::backend::Backend;
use rand::Rng;

use rlevo_core::bounds::Bounds;

use crate::fitness::FitnessFn;
// `sanitize_fitness` is the crate-wide fitness-hygiene primitive; it now lives in
// [`crate::fitness`]. Re-exported here so the searchers below (and existing
// `crate::fitness::sanitize_fitness` callers) keep resolving it unchanged.
pub(crate) use crate::fitness::sanitize_fitness;

pub mod hill_climbing;
pub mod nelder_mead;
pub mod random_restart;
pub mod simulated_annealing;

pub use hill_climbing::{HillClimbVariant, HillClimbing, HillClimbingParams};
pub use nelder_mead::{NelderMead, NelderMeadParams};
pub use random_restart::{RandomRestart, RandomRestartParams};
pub use simulated_annealing::{CoolingSchedule, SimulatedAnnealing, SimulatedAnnealingParams};

/// A gradient-free, host-side local search over real-valued genomes.
///
/// Implementors refine a single genome by repeatedly probing the supplied
/// [`FitnessFn`] and returning the best point found, subject to a strict
/// evaluation budget. They are the *meme* in a memetic algorithm: a
/// `MemeticWrapper` invokes `refine` on selected population members between an
/// inner strategy's `ask` and `tell`.
///
/// # Contract
///
/// For an input `genome` of length `D`, `refine` **must**:
///
/// 1. **Preserve dimensionality** — the returned `Vec<f32>` has length `D`.
/// 2. **Return a fresh, honest fitness** — the returned `f32` is the actual
///    value the supplied `fitness_fn` assigns to the returned genome (never a
///    stale or estimated value). For a deterministic `fitness_fn` this is
///    exact; for a stochastic one it is the value observed on the evaluation
///    that produced the returned genome.
/// 3. **Never worsen the input** (maximise, monotone non-worsening) — the
///    returned fitness is `>=` the fitness of the *input* `genome` under the
///    same `fitness_fn`. Implementors guarantee this structurally by
///    evaluating the input genome first and tracking a best-so-far pair that is
///    updated on *every* evaluation; the returned pair is always that tracked
///    best.
/// 4. **Terminate within budget** — make at most `Params::max_iters` total
///    `evaluate_one` calls, even on a perfectly flat landscape where no probe
///    ever improves.
/// 5. **Respect bounds** — every coordinate of the returned genome lies within
///    the `bounds` carried by `Params`.
///
/// Because the input genome is always evaluated once (contract item 3), a
/// `max_iters` of `0` cannot be honored honestly. Reference searchers treat
/// `max_iters == 0` as an invalid configuration and **panic**; implementors
/// should do the same rather than fabricate a fitness value. This holds on the
/// [`refine_with_known_fitness`](Self::refine_with_known_fitness) path too: the
/// reference searchers keep the `max_iters >= 1` panic even though that path
/// performs no seeding eval, so the two entry points share one budget contract.
///
/// All reference searchers route every evaluation — including the seeding eval
/// of the input — through a shared budget helper that
/// maps a `NaN` fitness to [`f32::NEG_INFINITY`], so a `NaN` probe can never seed or
/// displace a finite best-so-far and thus never propagates to the returned
/// fitness. The same rule applies to the `known_fitness` hint, which arrives
/// from a path that does *not* flow through the budget helper: every reference
/// override sanitizes the hint before seeding. Custom implementors that probe a
/// `fitness_fn` directly — or seed from a hint — should apply the same
/// sanitization rather than let a `NaN` reach their best-so-far tracker.
///
/// # Type parameters
///
/// - `B`: Burn backend. **Currently unused** by every reference searcher (they
///   are pure host code) and present only to reserve the seam for future
///   on-device searchers — e.g. a batched line search that materializes probe
///   tensors directly. Keeping `B` on the trait now avoids a breaking signature
///   change when such a searcher lands.
///
/// # Example
///
/// A one-line searcher that simply re-evaluates the input (the trivial,
/// always-valid refinement) illustrates the contract:
///
/// ```
/// use burn::backend::Flex;
/// use rand::{rngs::StdRng, Rng, SeedableRng};
/// use rlevo_evolution::fitness::FitnessFn;
/// use rlevo_evolution::local_search::LocalSearch;
///
/// struct Identity;
/// impl<B: burn::tensor::backend::Backend> LocalSearch<B> for Identity {
///     type Params = ();
///     fn refine(
///         &self,
///         _params: &(),
///         genome: Vec<f32>,
///         fitness_fn: &mut dyn FitnessFn<Vec<f32>>,
///         _rng: &mut dyn Rng,
///     ) -> (Vec<f32>, f32) {
///         let f = fitness_fn.evaluate_one(&genome); // fresh fitness
///         (genome, f)                               // same length, no worsening
///     }
/// }
///
/// struct Sphere;
/// impl FitnessFn<Vec<f32>> for Sphere {
///     fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
///         x.iter().map(|v| v * v).sum()
///     }
/// }
///
/// let searcher = Identity;
/// let mut fitness = Sphere;
/// let mut rng = StdRng::seed_from_u64(0);
/// let (refined, fit) = LocalSearch::<Flex>::refine(
///     &searcher,
///     &(),
///     vec![3.0, 4.0],
///     &mut fitness,
///     &mut rng,
/// );
/// assert_eq!(refined.len(), 2);
/// assert_eq!(fit, 25.0);
/// ```
pub trait LocalSearch<B: Backend>: Send + Sync {
    /// Static configuration for a refinement run (bounds, budget, step
    /// sizes, …). Cloned by a memetic wrapper once per generation.
    type Params: Clone + core::fmt::Debug + Send + Sync;

    /// Refines `genome` and returns `(refined_genome, refined_fitness)`.
    ///
    /// See the [trait-level contract](LocalSearch#contract) for the full set
    /// of invariants every implementation must uphold.
    fn refine(
        &self,
        params: &Self::Params,
        genome: Vec<f32>,
        fitness_fn: &mut dyn FitnessFn<Vec<f32>>,
        rng: &mut dyn Rng,
    ) -> (Vec<f32>, f32);

    /// Refines `genome`, seeding the best-so-far tracker with `known_fitness`
    /// instead of re-evaluating the input — saving exactly one
    /// [`FitnessFn::evaluate_one`] call per refinement.
    ///
    /// `known_fitness` **must** be the value `fitness_fn` assigns to `genome`
    /// (for a stochastic `fitness_fn`, a value it plausibly assigned). A `NaN`
    /// hint is sanitized to [`f32::NEG_INFINITY`] by the reference overrides, exactly
    /// as a `NaN` probe would be (see the [contract](LocalSearch#contract)). All
    /// other invariants — dimensionality, monotone non-worsening, budget, bounds
    /// — are identical to [`refine`](Self::refine); the only difference is that
    /// the seeding eval is elided, so a given `max_iters` buys one extra probe.
    ///
    /// Because the seeding eval consumes no rng, this method draws from the
    /// supplied `rng` exactly as [`refine`](Self::refine) would, leaving
    /// same-seed determinism intact.
    ///
    /// The default **ignores the hint** and delegates to
    /// [`refine`](Self::refine), preserving current behavior (and the seeding
    /// eval) for any implementor that does not override it.
    fn refine_with_known_fitness(
        &self,
        params: &Self::Params,
        genome: Vec<f32>,
        known_fitness: f32,
        fitness_fn: &mut dyn FitnessFn<Vec<f32>>,
        rng: &mut dyn Rng,
    ) -> (Vec<f32>, f32) {
        let _ = known_fitness;
        self.refine(params, genome, fitness_fn, rng)
    }
}

/// A fitness function wrapped with a hard evaluation budget.
///
/// `BudgetedEval` decrements its `remaining` counter on every successful
/// [`eval`](Self::eval) call and refuses further evaluations once the budget
/// is exhausted. Searchers route *all* their `evaluate_one` calls through it so
/// the [`LocalSearch`] budget invariant holds structurally rather than relying
/// on each searcher's loop bound being correct.
pub(crate) struct BudgetedEval<'a> {
    /// The underlying single-member fitness function.
    inner: &'a mut dyn FitnessFn<Vec<f32>>,
    /// Remaining permitted `evaluate_one` calls.
    remaining: usize,
}

impl<'a> BudgetedEval<'a> {
    /// Wraps `inner` with a budget of `max_evals` total evaluations.
    pub(crate) fn new(inner: &'a mut dyn FitnessFn<Vec<f32>>, max_evals: usize) -> Self {
        Self {
            inner,
            remaining: max_evals,
        }
    }

    /// Evaluates `genome`, consuming one unit of budget.
    ///
    /// Returns `Some(fitness)` while budget remains, or `None` once the budget
    /// is exhausted (in which case no underlying evaluation is performed).
    ///
    /// A `NaN` fitness is sanitized to [`f32::NEG_INFINITY`] via [`sanitize_fitness`]
    /// before it leaves this method. Because every searcher routes *all*
    /// evaluations — including the mandatory seeding eval of the input genome
    /// (contract item 3) — through `BudgetedEval`, this is the single chokepoint
    /// that keeps `NaN` out of the best-so-far trackers on the probe path; the
    /// `refine_with_known_fitness` overrides apply the same helper to the
    /// `known_fitness` hint. If *every* probe is `NaN` the searcher honestly
    /// returns `−inf` rather than `NaN`.
    pub(crate) fn eval(&mut self, genome: &Vec<f32>) -> Option<f32> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(sanitize_fitness(self.inner.evaluate_one(genome)))
    }

    /// Number of evaluations still permitted.
    pub(crate) fn remaining(&self) -> usize {
        self.remaining
    }
}

/// Clamps every coordinate of `genome` into the inclusive range `bounds`.
///
/// `bounds` is `(lo, hi)`; coordinates below `lo` are raised to `lo` and those
/// above `hi` are lowered to `hi`. Uses `x.max(lo).min(hi)`, so a degenerate
/// range where `lo > hi` collapses every coordinate to `hi` — the `min` is
/// applied last and wins. Callers are expected to supply valid bounds.
pub(crate) fn clamp_vec(genome: &mut [f32], bounds: Bounds) {
    let (lo, hi): (f32, f32) = bounds.into();
    for x in genome.iter_mut() {
        *x = x.max(lo).min(hi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts the genome length each `evaluate_one` sees; returns the sum of
    /// squares (sphere). Used to exercise the shared helpers in isolation.
    struct Sphere;
    impl FitnessFn<Vec<f32>> for Sphere {
        fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
            x.iter().map(|v| v * v).sum()
        }
    }

    #[test]
    fn budgeted_eval_decrements_and_exhausts() {
        let mut sphere = Sphere;
        let mut budget = BudgetedEval::new(&mut sphere, 2);
        assert_eq!(budget.remaining(), 2);
        assert_eq!(budget.eval(&vec![1.0, 0.0]), Some(1.0));
        assert_eq!(budget.remaining(), 1);
        assert_eq!(budget.eval(&vec![3.0, 4.0]), Some(25.0));
        assert_eq!(budget.remaining(), 0);
        // Budget exhausted: no further evaluation, returns None.
        assert_eq!(budget.eval(&vec![0.0, 0.0]), None);
    }

    /// Returns `NaN` for the origin, sphere otherwise — exercises the
    /// `BudgetedEval` NaN sanitization at the single evaluation chokepoint.
    struct NanAtOrigin;
    impl FitnessFn<Vec<f32>> for NanAtOrigin {
        fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
            let s: f32 = x.iter().map(|v| v * v).sum();
            if s == 0.0 { f32::NAN } else { s }
        }
    }

    #[test]
    fn budgeted_eval_sanitizes_nan_to_neg_infinity() {
        let mut f = NanAtOrigin;
        let mut budget = BudgetedEval::new(&mut f, 2);
        // A NaN probe is mapped to −inf (the worst value under the maximise
        // convention), so it can never seed or displace a finite best-so-far.
        assert_eq!(budget.eval(&vec![0.0, 0.0]), Some(f32::NEG_INFINITY));
        // A finite probe is passed through unchanged.
        assert_eq!(budget.eval(&vec![3.0, 4.0]), Some(25.0));
    }

    #[test]
    fn clamp_vec_respects_bounds() {
        let mut g = vec![-10.0_f32, 0.5, 10.0, -0.5];
        clamp_vec(&mut g, Bounds::new(-1.0, 1.0));
        assert_eq!(g, vec![-1.0, 0.5, 1.0, -0.5]);
    }
}
