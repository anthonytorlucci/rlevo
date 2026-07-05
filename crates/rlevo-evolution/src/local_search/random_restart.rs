//! Random-restart meta-search.
//!
//! Random restart wraps an inner
//! [`LocalSearch`] and runs it several times
//! from perturbed starting points, returning the best refinement found. Run `0`
//! starts from the unperturbed input (guaranteeing the
//! monotone-non-worsening invariant); runs `1..=restarts` start from the input
//! plus zero-mean Gaussian noise, clamped to bounds. This escapes the inner
//! searcher's local basins on multimodal landscapes.
//!
//! # Run-0-first ordering (load-bearing)
//!
//! Run `0` is executed **before any rng value is drawn for perturbation**, and
//! it refines the *unperturbed* input genome. This ordering is deliberate and
//! has two consequences that the contract and tests rely on:
//!
//! 1. Run `0` consumes the rng stream exactly as a bare
//!    `inner.refine(&params.inner, genome, fitness_fn, rng)` call would. Hence
//!    a `RandomRestart` with `restarts > 0` returns a result **bit-identical to
//!    or strictly better than** `restarts == 0` on the same seed: run `0` is
//!    shared between the two, and additional restarts only ever replace it on a
//!    strict improvement.
//! 2. Monotonicity versus the input is *structural*: run `0` already satisfies
//!    the [`LocalSearch`]
//!    monotone-non-worsening invariant (the inner searcher guarantees it), and
//!    the argmax over all runs can only be `>=` run `0`.
//!
//! # Evaluation budget
//!
//! `RandomRestart` owns **no cap of its own**. The total number of
//! `evaluate_one` calls is exactly the product
//! `(restarts + 1) * inner.max_iters`: one unperturbed run plus `restarts`
//! perturbed runs, each bounded by the inner searcher's own `max_iters`. The
//! inner searcher enforces its `max_iters >= 1` panic; `restarts == 0` is a
//! valid configuration equivalent to a plain inner run.

use core::fmt::Debug;

use burn::tensor::backend::Backend;
use rand::Rng;
use rand_distr::{Distribution as _, Normal};

use crate::fitness::FitnessFn;
use crate::local_search::{clamp_vec, LocalSearch};
use rlevo_core::bounds::Bounds;

/// Static configuration for a [`RandomRestart`] run.
///
/// The total evaluation budget is the product `(restarts + 1) * inner.max_iters`:
/// one unperturbed run plus `restarts` perturbed runs, each bounded by the inner
/// searcher's own `max_iters`. There is no second, outer cap.
///
/// # Type parameters
///
/// - `LP`: the inner searcher's `Params` type.
#[derive(Debug, Clone)]
pub struct RandomRestartParams<LP: Clone + Debug + Send + Sync> {
    /// Configuration handed to the inner searcher on every run.
    pub inner: LP,
    /// Number of *perturbed* restarts in addition to the unperturbed run `0`.
    /// Default `2` (so 3 runs total).
    ///
    /// Because `RandomRestart` adds no cap of its own, this directly scales the
    /// total evaluation budget: `(restarts + 1) * inner.max_iters` total
    /// `evaluate_one` calls. `0` is valid and reduces to a plain inner run.
    pub restarts: usize,
    /// Inclusive search-space bounds `(lo, hi)`; perturbed starts are clamped
    /// here.
    pub bounds: Bounds,
    /// Standard deviation of the Gaussian perturbation applied to the input
    /// genome for runs `1..=restarts`. Default `0.1 * (hi - lo)`.
    pub perturbation: f32,
}

impl<LP: Clone + Debug + Send + Sync> RandomRestartParams<LP> {
    /// Default parameters: `restarts = 2`, `perturbation = 0.1 * (hi - lo)`,
    /// wrapping the supplied inner `params`.
    #[must_use]
    pub fn default_for(inner: LP, bounds: Bounds) -> Self {
        let (lo, hi): (f32, f32) = bounds.into();
        debug_assert!(
            (hi - lo) > 0.0,
            "RandomRestartParams::default_for: zero-width bounds yields perturbation 0 (restarts cannot move)"
        );
        Self {
            inner,
            restarts: 2,
            bounds,
            perturbation: 0.1 * (hi - lo),
        }
    }
}

/// Random-restart wrapper around an inner [`LocalSearch`].
///
/// Runs the wrapped searcher `restarts + 1` times — once from the unperturbed
/// input and `restarts` times from Gaussian-perturbed, bounds-clamped starting
/// points — and returns the argmax over all runs (ties broken toward the
/// earliest run). The total evaluation budget is the product
/// `(restarts + 1) * inner.max_iters`; this wrapper adds no cap of its own.
///
/// # Type parameters
///
/// - `L`: the wrapped inner searcher.
///
/// # Example
///
/// ```
/// use burn::backend::Flex;
/// use rand::{rngs::StdRng, SeedableRng};
/// use rlevo_evolution::fitness::FitnessFn;
/// use rlevo_core::bounds::Bounds;
/// use rlevo_evolution::local_search::{
///     HillClimbing, HillClimbingParams, LocalSearch, RandomRestart, RandomRestartParams,
/// };
///
/// // Maximize the negated 2-D sphere; the optimum is the origin with fitness 0.
/// struct NegSphere;
/// impl FitnessFn<Vec<f32>> for NegSphere {
///     fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
///         -x.iter().map(|v| v * v).sum::<f32>()
///     }
/// }
///
/// // Wrap hill climbing in random restart: 3 perturbed restarts + run 0.
/// let searcher = RandomRestart::new(HillClimbing);
/// let inner = HillClimbingParams::default_for(Bounds::new(-5.12, 5.12));
/// let mut params = RandomRestartParams::default_for(inner, Bounds::new(-5.12, 5.12));
/// params.restarts = 3;
/// let mut fitness = NegSphere;
/// let mut rng = StdRng::seed_from_u64(7);
///
/// let start = vec![2.5_f32, -1.5];
/// let start_fit: f32 = -start.iter().map(|v| v * v).sum::<f32>();
/// let (refined, refined_fit) =
///     LocalSearch::<Flex>::refine(&searcher, &params, start, &mut fitness, &mut rng);
///
/// assert_eq!(refined.len(), 2); // dimensionality preserved
/// assert!(refined_fit >= start_fit); // monotone non-worsening
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RandomRestart<L> {
    /// The wrapped inner searcher, invoked once per run.
    inner: L,
}

impl<L> RandomRestart<L> {
    /// Wraps `inner` for multi-start refinement.
    #[must_use]
    pub fn new(inner: L) -> Self {
        Self { inner }
    }
}

impl<L> RandomRestart<L> {
    /// Shared body for [`refine`](LocalSearch::refine) and
    /// [`refine_with_known_fitness`](LocalSearch::refine_with_known_fitness).
    ///
    /// A `known` fitness describes the *unperturbed* input, so it is forwarded
    /// only to **run 0** (which refines that input); the `restarts` perturbed
    /// runs start from jittered points with no known fitness and always take the
    /// plain `inner.refine` path. Because the inner seeding eval draws no rng,
    /// forwarding the hint leaves run 0's rng consumption — and thus the
    /// load-bearing run-0-first ordering — bit-identical to the no-hint path.
    ///
    /// # Panics
    ///
    /// Panics if `params.restarts > 0` and `params.perturbation` is not
    /// strictly positive: a zero (or negative/non-finite) standard deviation
    /// cannot parameterize the Gaussian restart jitter, and silently degrading
    /// to unperturbed restarts would waste the entire restart budget on
    /// duplicate runs. `restarts == 0` is a valid configuration (a plain inner
    /// run) and never panics here. The inner searcher enforces its own
    /// `max_iters >= 1` invariant and will panic on a zero inner budget.
    fn refine_impl<B: Backend>(
        &self,
        params: &RandomRestartParams<L::Params>,
        genome: &[f32],
        known: Option<f32>,
        fitness_fn: &mut dyn FitnessFn<Vec<f32>>,
        rng: &mut dyn Rng,
    ) -> (Vec<f32>, f32)
    where
        L: LocalSearch<B>,
    {
        assert!(
            params.restarts == 0 || params.perturbation > 0.0,
            "RandomRestartParams::perturbation must be > 0 when restarts > 0 \
             (zero jitter would make every restart a duplicate of run 0)"
        );
        // Run 0 FIRST, from the UNPERTURBED input, before drawing ANY rng
        // values for perturbation. This ordering is load-bearing (see module
        // docs): it makes run 0 consume the rng stream exactly as a bare
        // `inner.refine` call would, so monotonicity is structural and the
        // `restarts > 0` result is bit-exactly `<=` the `restarts == 0` result
        // on the same seed. A known fitness describes this unperturbed input, so
        // it is forwarded here and nowhere else.
        let (mut best_genome, mut best_fit): (Vec<f32>, f32) = match known {
            Some(f) => self.inner.refine_with_known_fitness(
                &params.inner,
                genome.to_vec(),
                f,
                fitness_fn,
                rng,
            ),
            None => self
                .inner
                .refine(&params.inner, genome.to_vec(), fitness_fn, rng),
        };

        // Runs 1..=restarts: perturb the input with per-coordinate Gaussian
        // noise drawn through the passed rng, clamp to bounds, refine. Replace
        // the incumbent only on a STRICT improvement, so ties keep the earliest
        // run (run 0 wins ties).
        if params.restarts > 0 {
            let normal: Normal<f32> = Normal::new(0.0_f32, params.perturbation)
                .expect("perturbation std-dev is strictly positive (asserted above)");
            for _ in 0..params.restarts {
                let mut start: Vec<f32> = genome.to_vec();
                for coord in &mut start {
                    *coord += normal.sample(rng);
                }
                clamp_vec(&mut start, params.bounds);

                let (run_genome, run_fit): (Vec<f32>, f32) =
                    self.inner.refine(&params.inner, start, fitness_fn, rng);
                if run_fit > best_fit {
                    best_fit = run_fit;
                    best_genome = run_genome;
                }
            }
        }

        (best_genome, best_fit)
    }
}

impl<B: Backend, L: LocalSearch<B>> LocalSearch<B> for RandomRestart<L> {
    type Params = RandomRestartParams<L::Params>;

    /// # Panics
    ///
    /// Panics if `params.restarts > 0` and `params.perturbation` is not strictly
    /// positive; see `refine_impl`.
    fn refine(
        &self,
        params: &RandomRestartParams<L::Params>,
        genome: Vec<f32>,
        fitness_fn: &mut dyn FitnessFn<Vec<f32>>,
        rng: &mut dyn Rng,
    ) -> (Vec<f32>, f32) {
        self.refine_impl::<B>(params, &genome, None, fitness_fn, rng)
    }

    /// Forwards `known_fitness` to **run 0** (the unperturbed input) so its inner
    /// searcher skips its seeding eval; perturbed runs are unaffected. See
    /// `refine_impl`.
    ///
    /// # Panics
    ///
    /// Panics if `params.restarts > 0` and `params.perturbation` is not strictly
    /// positive; see `refine_impl`.
    fn refine_with_known_fitness(
        &self,
        params: &RandomRestartParams<L::Params>,
        genome: Vec<f32>,
        known_fitness: f32,
        fitness_fn: &mut dyn FitnessFn<Vec<f32>>,
        rng: &mut dyn Rng,
    ) -> (Vec<f32>, f32) {
        self.refine_impl::<B>(params, &genome, Some(known_fitness), fitness_fn, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_search::{HillClimbing, HillClimbingParams};
    use burn::backend::Flex;
    use rand::rngs::StdRng;
    use rand::{RngExt as _, SeedableRng};

    type TestBackend = Flex;

    const BOUNDS: Bounds = Bounds::new(-5.12, 5.12);

    /// Negated sphere `f(x) = -Σ x_i²` — concave bump; global maximum 0 at the
    /// origin.
    struct NegSphere;
    impl FitnessFn<Vec<f32>> for NegSphere {
        fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
            -x.iter().map(|v| v * v).sum::<f32>()
        }
    }

    /// Negated 2-D Rastrigin — highly multimodal; global maximum 0 at the
    /// origin.
    struct NegRastrigin;
    impl FitnessFn<Vec<f32>> for NegRastrigin {
        fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
            use core::f32::consts::PI;
            let a = 10.0_f32;
            // `a * D` constant folded per-coordinate to avoid a usize->f32 cast.
            -x.iter()
                .map(|&xi| a + xi * xi - a * (2.0 * PI * xi).cos())
                .sum::<f32>()
        }
    }

    /// Negated 2-D Rosenbrock — curved ridge; global maximum 0 at `(1, 1)`.
    struct NegRosenbrock;
    impl FitnessFn<Vec<f32>> for NegRosenbrock {
        fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            -(a * a + 100.0 * b * b)
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

    /// Builds a `RandomRestart<HillClimbing>` params set with the given restart
    /// count, sharing the supplied inner `HillClimbingParams`.
    fn rr_params(
        inner: HillClimbingParams,
        restarts: usize,
    ) -> RandomRestartParams<HillClimbingParams> {
        let mut params: RandomRestartParams<HillClimbingParams> =
            RandomRestartParams::default_for(inner, BOUNDS);
        params.restarts = restarts;
        params
    }

    #[test]
    fn budget_is_product_of_runs_and_inner_max_iters() {
        // On a flat landscape no probe ever improves, so every run burns its
        // full inner budget. Total evals must respect the product formula and
        // exceed a single inner run (proving the restarts actually ran).
        let searcher = RandomRestart::new(HillClimbing);
        let mut inner = HillClimbingParams::default_for(BOUNDS);
        inner.max_iters = 20;
        let restarts = 3_usize;
        let params = rr_params(inner.clone(), restarts);

        let mut base = Flat;
        let mut counting = Counting::new(&mut base);
        let mut rng = StdRng::seed_from_u64(1);
        let start = vec![1.0_f32, 2.0, 3.0];
        let _ = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start,
            &mut counting,
            &mut rng,
        );

        let upper = (restarts + 1) * inner.max_iters;
        assert!(
            counting.calls <= upper,
            "evals {} must not exceed product budget {}",
            counting.calls,
            upper
        );
        assert!(
            counting.calls > inner.max_iters,
            "evals {} must exceed a single inner run ({}) — restarts must run",
            counting.calls,
            inner.max_iters
        );
    }

    #[test]
    fn restarts_never_worse_than_zero_same_seed() {
        // On a multimodal landscape, restarts > 0 must never return a worse
        // result than restarts == 0 with the same seed (run 0 is shared).
        let searcher = RandomRestart::new(HillClimbing);
        let inner = HillClimbingParams::default_for(BOUNDS);
        let start = vec![3.7_f32, -2.9];

        let params_zero = rr_params(inner.clone(), 0);
        let mut fit_zero = NegRastrigin;
        let mut rng_zero = StdRng::seed_from_u64(42);
        let (_g0, f0) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params_zero,
            start.clone(),
            &mut fit_zero,
            &mut rng_zero,
        );

        let params_three = rr_params(inner, 3);
        let mut fit_three = NegRastrigin;
        let mut rng_three = StdRng::seed_from_u64(42);
        let (_g3, f3) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params_three,
            start,
            &mut fit_three,
            &mut rng_three,
        );

        assert!(
            f3 >= f0,
            "restarts=3 ({f3}) must not be worse than restarts=0 ({f0})"
        );
    }

    #[test]
    fn restarts_escape_local_basin() {
        // From a start trapped on a non-global Neg-Rastrigin peak, a single inner
        // run (restarts=0) settles onto that peak; restarts with healthy
        // perturbation escape to a strictly better fitness.
        let searcher = RandomRestart::new(HillClimbing);
        let mut inner = HillClimbingParams::default_for(BOUNDS);
        // Small step so run 0 stays trapped near the start peak.
        inner.step_size = 0.25;
        inner.max_iters = 120;
        // Start near a non-global Neg-Rastrigin local maximum (lattice point
        // (4, -3), a local minimum of the original Rastrigin).
        let start = vec![4.0_f32, -3.0];

        let params_zero = rr_params(inner.clone(), 0);
        let mut fit_zero = NegRastrigin;
        let mut rng_zero = StdRng::seed_from_u64(7);
        let (_g0, f0) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params_zero,
            start.clone(),
            &mut fit_zero,
            &mut rng_zero,
        );

        // Healthy perturbation lets restarts jump basins.
        let mut params_five = rr_params(inner, 5);
        params_five.perturbation = 2.5;
        let mut fit_five = NegRastrigin;
        let mut rng_five = StdRng::seed_from_u64(7);
        let (_g5, f5) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params_five,
            start,
            &mut fit_five,
            &mut rng_five,
        );

        assert!(
            f5 > f0,
            "restarts=5 ({f5}) should strictly beat restarts=0 ({f0})"
        );
    }

    #[test]
    fn rosenbrock_monotone_non_worsening() {
        let searcher = RandomRestart::new(HillClimbing);
        let inner = HillClimbingParams::default_for(BOUNDS);
        let params = rr_params(inner, 2);
        let mut rng = StdRng::seed_from_u64(11);
        let (lo, hi): (f32, f32) = BOUNDS.into();
        for _ in 0..5 {
            let start: Vec<f32> = (0..2)
                .map(|_| lo + (hi - lo) * rng.random::<f32>())
                .collect();
            let mut fitness = NegRosenbrock;
            let start_fit = fitness.evaluate_one(&start);
            let (_g, fit) = LocalSearch::<TestBackend>::refine(
                &searcher,
                &params,
                start,
                &mut fitness,
                &mut rng,
            );
            assert!(fit >= start_fit, "monotone: {fit} >= {start_fit}");
        }
    }

    #[test]
    fn output_len_equals_input_len() {
        let searcher = RandomRestart::new(HillClimbing);
        let inner = HillClimbingParams::default_for(BOUNDS);
        let params = rr_params(inner, 2);
        let mut fitness = NegSphere;
        let mut rng = StdRng::seed_from_u64(3);
        let (lo, hi): (f32, f32) = BOUNDS.into();
        for dim in [1_usize, 2, 5, 10] {
            let start: Vec<f32> = (0..dim)
                .map(|_| lo + (hi - lo) * rng.random::<f32>())
                .collect();
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
        let searcher = RandomRestart::new(HillClimbing);
        let inner = HillClimbingParams::default_for(BOUNDS);
        let params = rr_params(inner, 3);
        let mut fitness = NegRastrigin;
        let mut rng = StdRng::seed_from_u64(4);
        let start = vec![1.3_f32, -2.7];
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
    fn boundary_start_with_large_perturbation_stays_within_bounds() {
        let searcher = RandomRestart::new(HillClimbing);
        let mut inner = HillClimbingParams::default_for(BOUNDS);
        // Big inner step, no decay, so probes push hard on bounds too.
        inner.step_size = 4.0;
        inner.step_decay = 1.0;
        let mut params = rr_params(inner, 4);
        // Large perturbation relative to range: starts will spill past bounds
        // before clamping.
        params.perturbation = 10.0;
        let mut fitness = NegSphere;
        let mut rng = StdRng::seed_from_u64(5);
        // Start at the upper boundary in every coordinate.
        let start = vec![BOUNDS.hi(); 4];
        let (g, _f) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start,
            &mut fitness,
            &mut rng,
        );
        for &x in &g {
            assert!(
                x >= BOUNDS.lo() && x <= BOUNDS.hi(),
                "coord {x} out of bounds {BOUNDS:?}"
            );
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn same_seed_is_bit_identical() {
        let searcher = RandomRestart::new(HillClimbing);
        let inner = HillClimbingParams::default_for(BOUNDS);
        let params = rr_params(inner, 4);
        let start = vec![2.0_f32, -3.0, 1.5];

        let mut fitness_a = NegRastrigin;
        let mut rng_a = StdRng::seed_from_u64(123);
        let (g_a, f_a) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start.clone(),
            &mut fitness_a,
            &mut rng_a,
        );

        let mut fitness_b = NegRastrigin;
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

    #[test]
    fn known_fitness_saves_exactly_one_eval_total() {
        // The hint is forwarded only to run 0 (the unperturbed input); the
        // perturbed runs are untouched. With an inner budget large enough that
        // step-underflow (not the budget) terminates each run, total evals drop
        // by exactly one: run 0's seeding eval.
        let searcher = RandomRestart::new(HillClimbing);
        let mut inner = HillClimbingParams::default_for(BOUNDS);
        inner.max_iters = 10_000;
        let params = rr_params(inner, 3);
        let start = vec![1.0_f32, 2.0, 3.0];

        let refine_evals = {
            let mut base = Flat;
            let mut counting = Counting::new(&mut base);
            let mut rng = StdRng::seed_from_u64(51);
            let _ = LocalSearch::<TestBackend>::refine(
                &searcher,
                &params,
                start.clone(),
                &mut counting,
                &mut rng,
            );
            counting.calls
        };
        let hint_evals = {
            let mut base = Flat;
            let mut counting = Counting::new(&mut base);
            let mut rng = StdRng::seed_from_u64(51);
            let _ = LocalSearch::<TestBackend>::refine_with_known_fitness(
                &searcher,
                &params,
                start.clone(),
                1.0, // Flat fitness of the start
                &mut counting,
                &mut rng,
            );
            counting.calls
        };
        assert_eq!(
            hint_evals + 1,
            refine_evals,
            "hint must save exactly run 0's seeding eval ({hint_evals} vs {refine_evals})"
        );
    }

    #[test]
    fn nan_hint_does_not_propagate() {
        let searcher = RandomRestart::new(HillClimbing);
        let inner = HillClimbingParams::default_for(BOUNDS);
        let params = rr_params(inner, 3);
        let mut fitness = NegSphere;
        let mut rng = StdRng::seed_from_u64(52);
        let start = vec![2.0_f32, -1.0];
        let (g, fit) = LocalSearch::<TestBackend>::refine_with_known_fitness(
            &searcher,
            &params,
            start,
            f32::NAN,
            &mut fitness,
            &mut rng,
        );
        assert!(fit.is_finite(), "NaN hint must be sanitized, got {fit}");
        let fresh = fitness.evaluate_one(&g);
        approx::assert_relative_eq!(fit, fresh, epsilon = 1e-6);
    }
}
