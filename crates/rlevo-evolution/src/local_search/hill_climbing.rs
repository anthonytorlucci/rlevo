//! Coordinate-wise hill climbing.
//!
//! A simple, robust gradient-free local search: from the input genome, probe
//! neighbours one coordinate at a time and move whenever a probe strictly
//! improves the objective. Two acceptance strategies are offered via
//! [`HillClimbVariant`]:
//!
//! - [`FirstImprovement`](HillClimbVariant::FirstImprovement) — each iteration
//!   perturbs one randomly chosen coordinate by `±step_size` and accepts the
//!   move on the first strict improvement seen. Cheap per step; good when the
//!   budget is small.
//! - [`BestImprovement`](HillClimbVariant::BestImprovement) — each *sweep*
//!   probes `±step_size` along every coordinate (`2·dim` evaluations) and moves
//!   to the single best strict improver. More evaluations per move, but each
//!   move is greedier.
//!
//! Both variants shrink `step_size` by `step_decay` once a probe budget passes
//! without improvement, letting the search settle into a basin. The returned
//! pair is always the best `(genome, fitness)` observed across all evaluations,
//! which makes the [`LocalSearch`]
//! monotone-non-worsening and fresh-fitness invariants hold structurally.

use burn::tensor::backend::Backend;
use rand::{Rng, RngExt};

use crate::fitness::FitnessFn;
use crate::local_search::{clamp_vec, BudgetedEval, LocalSearch};

/// Acceptance strategy for [`HillClimbing`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HillClimbVariant {
    /// Probe one random coordinate per iteration; accept the first strict
    /// improvement.
    FirstImprovement,
    /// Probe `±step` along every coordinate per sweep; move to the best strict
    /// improver.
    BestImprovement,
}

/// Static configuration for a [`HillClimbing`] run.
#[derive(Debug, Clone)]
pub struct HillClimbingParams {
    /// Inclusive search-space bounds `(lo, hi)`; refined genomes are clamped
    /// here.
    pub bounds: (f32, f32),
    /// Hard cap on the **total** number of `evaluate_one` calls per `refine`,
    /// including the mandatory initial re-evaluation of the input genome.
    /// Default `100`.
    pub max_iters: usize,
    /// Per-coordinate perturbation magnitude. Default `0.1 * (hi - lo)` via
    /// [`HillClimbingParams::default_for`].
    pub step_size: f32,
    /// Multiplicative shrink applied to `step_size` after a failed probe
    /// budget. `0.5` halves the step; `1.0` keeps it fixed. Default `0.5`.
    pub step_decay: f32,
    /// Acceptance strategy. Default
    /// [`FirstImprovement`](HillClimbVariant::FirstImprovement).
    pub variant: HillClimbVariant,
}

impl HillClimbingParams {
    /// Default parameters derived from the search-space `bounds`.
    ///
    /// `step_size = 0.1 * (hi - lo)`, `step_decay = 0.5`, `max_iters = 100`,
    /// variant [`FirstImprovement`](HillClimbVariant::FirstImprovement).
    #[must_use]
    pub fn default_for(bounds: (f32, f32)) -> Self {
        let (lo, hi) = bounds;
        Self {
            bounds,
            max_iters: 100,
            step_size: 0.1 * (hi - lo),
            step_decay: 0.5,
            variant: HillClimbVariant::FirstImprovement,
        }
    }
}

/// Coordinate-wise hill-climbing local search.
///
/// A unit struct: all configuration lives in [`HillClimbingParams`], so one
/// instance can refine many genomes. See the [module docs](self) for the two
/// acceptance variants.
///
/// # Example
///
/// ```
/// use burn::backend::Flex;
/// use rand::{rngs::StdRng, SeedableRng};
/// use rlevo_evolution::fitness::FitnessFn;
/// use rlevo_evolution::local_search::{HillClimbing, HillClimbingParams, LocalSearch};
///
/// // Minimize the 2-D sphere; the optimum is the origin with fitness 0.
/// struct Sphere;
/// impl FitnessFn<Vec<f32>> for Sphere {
///     fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
///         x.iter().map(|v| v * v).sum()
///     }
/// }
///
/// let searcher = HillClimbing;
/// let params = HillClimbingParams::default_for((-5.12, 5.12));
/// let mut fitness = Sphere;
/// let mut rng = StdRng::seed_from_u64(7);
///
/// let start = vec![2.5_f32, -1.5];
/// let start_fit: f32 = start.iter().map(|v| v * v).sum();
/// let (refined, refined_fit) =
///     LocalSearch::<Flex>::refine(&searcher, &params, start, &mut fitness, &mut rng);
///
/// assert_eq!(refined.len(), 2);          // dimensionality preserved
/// assert!(refined_fit <= start_fit);     // monotone non-worsening
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct HillClimbing;

impl<B: Backend> LocalSearch<B> for HillClimbing {
    type Params = HillClimbingParams;

    /// # Panics
    ///
    /// Panics if `params.max_iters == 0`: a zero evaluation budget makes it
    /// impossible to return an honestly evaluated fitness, so it is treated
    /// as an invalid configuration (programming error), not runtime data.
    fn refine(
        &self,
        params: &HillClimbingParams,
        genome: Vec<f32>,
        fitness_fn: &mut dyn FitnessFn<Vec<f32>>,
        rng: &mut dyn Rng,
    ) -> (Vec<f32>, f32) {
        assert!(
            params.max_iters >= 1,
            "HillClimbingParams::max_iters must be >= 1 (the input genome is \
             always evaluated once to seed the best-so-far tracker)"
        );
        let mut budget = BudgetedEval::new(fitness_fn, params.max_iters);

        // First action: evaluate the input genome (1 eval) and seed the
        // best-so-far tracker. The assert above guarantees this succeeds.
        let Some(initial_fit) = budget.eval(&genome) else {
            unreachable!("budget of >= 1 cannot be exhausted before the first eval");
        };

        let mut current: Vec<f32> = genome;
        let mut current_fit = initial_fit;
        // The tracked best — always returned. Updated on EVERY evaluation, so
        // the monotone + fresh-fitness invariants hold structurally.
        let mut best: Vec<f32> = current.clone();
        let mut best_fit = current_fit;

        let mut step = params.step_size;
        let dim = current.len();
        if dim == 0 {
            return (best, best_fit);
        }

        match params.variant {
            HillClimbVariant::FirstImprovement => {
                // After `2 * dim` consecutive non-improving probes, shrink the
                // step. This budget gives every coordinate a chance in both
                // directions before concluding the current step is too coarse.
                let mut consecutive_failures: usize = 0;
                let failure_budget = 2 * dim;
                loop {
                    let coord = rng.random_range(0..dim);
                    let sign: f32 = if rng.random::<bool>() { 1.0 } else { -1.0 };
                    let mut candidate = current.clone();
                    candidate[coord] += sign * step;
                    clamp_vec(&mut candidate, params.bounds);

                    let Some(cand_fit) = budget.eval(&candidate) else {
                        break;
                    };
                    if cand_fit < best_fit {
                        best_fit = cand_fit;
                        best.clone_from(&candidate);
                    }
                    if cand_fit < current_fit {
                        current = candidate;
                        current_fit = cand_fit;
                        consecutive_failures = 0;
                    } else {
                        consecutive_failures += 1;
                        if consecutive_failures >= failure_budget {
                            step *= params.step_decay;
                            consecutive_failures = 0;
                            // Step underflowed to ~0: no neighbour will ever
                            // differ from `current`. Early-stop; the eval bound
                            // is the hard guarantee, this is just a courtesy.
                            if step <= f32::EPSILON {
                                break;
                            }
                        }
                    }
                }
            }
            HillClimbVariant::BestImprovement => {
                'sweeps: loop {
                    if budget.remaining() == 0 {
                        break;
                    }
                    let mut sweep_best_fit = current_fit;
                    let mut sweep_best: Option<Vec<f32>> = None;
                    for coord in 0..dim {
                        for &sign in &[1.0_f32, -1.0_f32] {
                            let mut candidate = current.clone();
                            candidate[coord] += sign * step;
                            clamp_vec(&mut candidate, params.bounds);
                            let Some(cand_fit) = budget.eval(&candidate) else {
                                // Budget ran out mid-sweep: commit whatever the
                                // partial sweep found and stop.
                                break 'sweeps;
                            };
                            if cand_fit < best_fit {
                                best_fit = cand_fit;
                                best.clone_from(&candidate);
                            }
                            if cand_fit < sweep_best_fit {
                                sweep_best_fit = cand_fit;
                                sweep_best = Some(candidate);
                            }
                        }
                    }
                    if let Some(next) = sweep_best {
                        current = next;
                        current_fit = sweep_best_fit;
                    } else {
                        // No coordinate improved this sweep: shrink the step.
                        step *= params.step_decay;
                        if step <= f32::EPSILON {
                            break;
                        }
                    }
                }
            }
        }

        (best, best_fit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type TestBackend = Flex;

    /// `f(x) = Σ x_i²` — convex, unimodal, optimum at the origin.
    struct Sphere;
    impl FitnessFn<Vec<f32>> for Sphere {
        fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
            x.iter().map(|v| v * v).sum()
        }
    }

    /// 2-D Rosenbrock — curved valley, optimum at `(1, 1)` with value 0.
    struct Rosenbrock;
    impl FitnessFn<Vec<f32>> for Rosenbrock {
        fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        }
    }

    /// Constant 1.0 — perfectly flat; no probe ever improves.
    struct Flat;
    impl FitnessFn<Vec<f32>> for Flat {
        fn evaluate_one(&mut self, _x: &Vec<f32>) -> f32 {
            1.0
        }
    }

    /// Wraps a fitness function and counts `evaluate_one` calls.
    struct Counting<'a> {
        inner: &'a mut dyn FitnessFn<Vec<f32>>,
        calls: usize,
    }
    impl<'a> Counting<'a> {
        fn new(inner: &'a mut dyn FitnessFn<Vec<f32>>) -> Self {
            Self { inner, calls: 0 }
        }
    }
    impl FitnessFn<Vec<f32>> for Counting<'_> {
        fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
            self.calls += 1;
            self.inner.evaluate_one(x)
        }
    }

    const BOUNDS: (f32, f32) = (-5.12, 5.12);

    fn random_start(rng: &mut StdRng, dim: usize, bounds: (f32, f32)) -> Vec<f32> {
        let (lo, hi) = bounds;
        (0..dim).map(|_| lo + (hi - lo) * rng.random::<f32>()).collect()
    }

    #[test]
    fn sphere_d2_converges_below_threshold() {
        let searcher = HillClimbing;
        let mut params = HillClimbingParams::default_for(BOUNDS);
        params.max_iters = 100;
        let mut fitness = Sphere;
        let mut rng = StdRng::seed_from_u64(1);
        let start = random_start(&mut rng, 2, BOUNDS);
        let (_g, fit) =
            LocalSearch::<TestBackend>::refine(&searcher, &params, start, &mut fitness, &mut rng);
        assert!(fit < 1e-3, "sphere D=2 should converge: best={fit}");
    }

    #[test]
    fn sphere_d10_strictly_improves() {
        let searcher = HillClimbing;
        let params = HillClimbingParams::default_for(BOUNDS);
        let mut fitness = Sphere;
        let mut rng = StdRng::seed_from_u64(2);
        let start = random_start(&mut rng, 10, BOUNDS);
        let start_fit: f32 = start.iter().map(|v| v * v).sum();
        let (_g, fit) =
            LocalSearch::<TestBackend>::refine(&searcher, &params, start, &mut fitness, &mut rng);
        assert!(fit < start_fit, "expected improvement: {fit} < {start_fit}");
    }

    #[test]
    fn output_len_equals_input_len() {
        let searcher = HillClimbing;
        let params = HillClimbingParams::default_for(BOUNDS);
        let mut fitness = Sphere;
        let mut rng = StdRng::seed_from_u64(3);
        for dim in [1_usize, 2, 5, 10] {
            let start = random_start(&mut rng, dim, BOUNDS);
            let (g, _f) = LocalSearch::<TestBackend>::refine(
                &searcher,
                &params,
                start,
                &mut fitness,
                &mut rng,
            );
            assert_eq!(g.len(), dim);
        }
    }

    #[test]
    fn returned_fitness_matches_fresh_eval() {
        let searcher = HillClimbing;
        let params = HillClimbingParams::default_for(BOUNDS);
        let mut fitness = Sphere;
        let mut rng = StdRng::seed_from_u64(4);
        let start = random_start(&mut rng, 4, BOUNDS);
        let (g, fit) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start,
            &mut fitness,
            &mut rng,
        );
        let fresh = fitness.evaluate_one(&g);
        approx::assert_relative_eq!(fit, fresh, epsilon = 1e-6);
    }

    #[test]
    fn rosenbrock_monotone_non_worsening() {
        let searcher = HillClimbing;
        let params = HillClimbingParams::default_for(BOUNDS);
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..6 {
            let start = random_start(&mut rng, 2, BOUNDS);
            let mut fitness = Rosenbrock;
            let start_fit = fitness.evaluate_one(&start);
            let (_g, fit) = LocalSearch::<TestBackend>::refine(
                &searcher,
                &params,
                start,
                &mut fitness,
                &mut rng,
            );
            assert!(fit <= start_fit, "monotone: {fit} <= {start_fit}");
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn flat_landscape_terminates_within_budget() {
        let searcher = HillClimbing;
        let mut params = HillClimbingParams::default_for(BOUNDS);
        params.max_iters = 37;
        let mut base = Flat;
        let mut counting = Counting::new(&mut base);
        let mut rng = StdRng::seed_from_u64(6);
        let start = vec![1.0_f32, 2.0, 3.0];
        let (g, fit) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start.clone(),
            &mut counting,
            &mut rng,
        );
        assert!(
            counting.calls <= params.max_iters,
            "evals {} must not exceed budget {}",
            counting.calls,
            params.max_iters
        );
        // On a flat landscape nothing improves: the returned genome is the
        // input and its fitness is the constant 1.0.
        assert_eq!(g, start);
        assert_eq!(fit, 1.0);
    }

    #[test]
    fn boundary_start_stays_within_bounds() {
        let searcher = HillClimbing;
        let mut params = HillClimbingParams::default_for(BOUNDS);
        // Big step relative to range, no decay, so probes push hard on bounds.
        params.step_size = 4.0;
        params.step_decay = 1.0;
        let mut fitness = Sphere;
        let mut rng = StdRng::seed_from_u64(8);
        // Start at the upper boundary in every coordinate.
        let start = vec![BOUNDS.1; 4];
        let (g, _f) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start,
            &mut fitness,
            &mut rng,
        );
        for &x in &g {
            assert!(
                x >= BOUNDS.0 && x <= BOUNDS.1,
                "coord {x} out of bounds {BOUNDS:?}"
            );
        }
    }

    /// Evaluations the given variant needs to drive 2-D sphere below `tol`
    /// from a fixed start/seed, or `None` if it never reaches `tol`.
    fn evals_to_tolerance(variant: HillClimbVariant, tol: f32) -> Option<usize> {
        let searcher = HillClimbing;
        let mut params = HillClimbingParams::default_for(BOUNDS);
        params.variant = variant;
        params.max_iters = 400;
        let mut base = Sphere;
        let mut counting = Counting::new(&mut base);
        let mut rng = StdRng::seed_from_u64(99);
        let start = vec![3.0_f32, -2.0];
        let (_g, fit) =
            LocalSearch::<TestBackend>::refine(&searcher, &params, start, &mut counting, &mut rng);
        if fit < tol {
            Some(counting.calls)
        } else {
            None
        }
    }

    #[test]
    fn best_improvement_competitive_with_first_improvement() {
        // On a smooth unimodal landscape, BestImprovement should reach the
        // same tolerance using no more evaluations than FirstImprovement.
        let tol = 1e-2_f32;
        let first = evals_to_tolerance(HillClimbVariant::FirstImprovement, tol)
            .expect("first-improvement should reach tolerance");
        let best = evals_to_tolerance(HillClimbVariant::BestImprovement, tol)
            .expect("best-improvement should reach tolerance");
        assert!(
            best <= first,
            "best-improvement evals {best} should be <= first-improvement evals {first}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn same_seed_is_bit_identical() {
        let searcher = HillClimbing;
        let params = HillClimbingParams::default_for(BOUNDS);
        let start = vec![2.0_f32, -3.0, 1.5];

        let mut fitness_a = Sphere;
        let mut rng_a = StdRng::seed_from_u64(123);
        let (g_a, f_a) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start.clone(),
            &mut fitness_a,
            &mut rng_a,
        );

        let mut fitness_b = Sphere;
        let mut rng_b = StdRng::seed_from_u64(123);
        let (g_b, f_b) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start,
            &mut fitness_b,
            &mut rng_b,
        );

        assert_eq!(g_a, g_b);
        assert_eq!(f_a, f_b);
    }
}
