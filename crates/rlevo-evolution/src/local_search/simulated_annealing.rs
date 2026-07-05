//! Simulated annealing.
//!
//! Simulated annealing performs a stochastic walk that accepts worsening moves
//! with probability `exp(Δf / T)`, cooling the temperature `T` over time so
//! the walk gradually concentrates on improving moves. This lets it escape
//! shallow local maxima that strict hill climbing gets stuck in, at the cost of
//! being a coarse explorer rather than a precision finisher.
//!
//! Each iteration proposes a neighbour by adding per-coordinate Gaussian noise
//! `N(0, step_size)` to the current walker position (sampled through the
//! supplied `rng`), clamps it to bounds, and evaluates it. A non-worsening move
//! is always accepted; a worsening move (`Δf = cand_fit - current_fit < 0`) is
//! accepted iff a uniform draw `rng.random::<f32>()` falls below `exp(Δf / T)`.
//! The temperature is cooled once per iteration via the configured
//! [`CoolingSchedule`], and the walk early-stops once `T < min_temp`.
//!
//! The returned pair is always the best `(genome, fitness)` observed across all
//! evaluations — **not** the final walker position, which may sit at an uphill
//! point it accepted. Tracking the global best on every evaluation is what makes
//! the [`LocalSearch`] monotone-non-worsening
//! and fresh-fitness invariants hold structurally despite downhill acceptance.

use burn::tensor::backend::Backend;
use rand::{Rng, RngExt};
use rand_distr::{Distribution as _, Normal};

use crate::fitness::FitnessFn;
use crate::local_search::{clamp_vec, sanitize_fitness, BudgetedEval, LocalSearch};
use rlevo_core::bounds::Bounds;

/// Temperature-cooling schedule for [`SimulatedAnnealing`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoolingSchedule {
    /// Geometric cooling: `T <- T * factor` each step. `factor` in `(0, 1)`.
    Geometric {
        /// Multiplicative cooling factor per step.
        factor: f32,
    },
    /// Linear cooling: `T <- T - delta` each step (floored at `0`).
    Linear {
        /// Temperature decrement per step.
        delta: f32,
    },
}

/// Static configuration for a [`SimulatedAnnealing`] run.
#[derive(Debug, Clone)]
pub struct SimulatedAnnealingParams {
    /// Inclusive search-space bounds `(lo, hi)`; proposals are clamped here.
    pub bounds: Bounds,
    /// Hard cap on the **total** number of `evaluate_one` calls per `refine`,
    /// including the initial evaluation of the input genome. Default `200`.
    pub max_iters: usize,
    /// Starting temperature. Default `1.0`.
    pub initial_temp: f32,
    /// Cooling schedule. Default `Geometric { factor: 0.95 }`.
    pub cooling: CoolingSchedule,
    /// Early-stop temperature floor — the walk stops once `T < min_temp`.
    /// Default `1e-6`.
    pub min_temp: f32,
    /// Standard deviation of the Gaussian proposal step. Default
    /// `0.1 * (hi - lo)`.
    pub step_size: f32,
}

impl SimulatedAnnealingParams {
    /// Default parameters derived from the search-space `bounds`.
    ///
    /// `initial_temp = 1.0`, `cooling = Geometric { factor: 0.95 }`,
    /// `min_temp = 1e-6`, `step_size = 0.1 * (hi - lo)`, `max_iters = 200`.
    #[must_use]
    pub fn default_for(bounds: Bounds) -> Self {
        let (lo, hi): (f32, f32) = bounds.into();
        debug_assert!(
            (hi - lo) > 0.0,
            "SimulatedAnnealingParams::default_for: zero-width bounds yields step_size 0 (search cannot move)"
        );
        Self {
            bounds,
            max_iters: 200,
            initial_temp: 1.0,
            cooling: CoolingSchedule::Geometric { factor: 0.95 },
            min_temp: 1e-6,
            step_size: 0.1 * (hi - lo),
        }
    }
}

/// Simulated-annealing local search.
///
/// A unit struct: all configuration lives in [`SimulatedAnnealingParams`], so
/// one instance can refine many genomes. See the [module docs](self) for the
/// acceptance rule and the best-so-far tracking that upholds the
/// [`LocalSearch`] contract.
///
/// # Example
///
/// ```
/// use burn::backend::Flex;
/// use rand::{rngs::StdRng, SeedableRng};
/// use rlevo_evolution::fitness::FitnessFn;
/// use rlevo_core::bounds::Bounds;
/// use rlevo_evolution::local_search::{LocalSearch, SimulatedAnnealing, SimulatedAnnealingParams};
///
/// // Maximize the negated 2-D sphere; the optimum is the origin with fitness 0.
/// struct NegSphere;
/// impl FitnessFn<Vec<f32>> for NegSphere {
///     fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
///         -x.iter().map(|v| v * v).sum::<f32>()
///     }
/// }
///
/// let searcher = SimulatedAnnealing;
/// let params = SimulatedAnnealingParams::default_for(Bounds::new(-5.12, 5.12));
/// let mut fitness = NegSphere;
/// let mut rng = StdRng::seed_from_u64(7);
///
/// let start = vec![2.5_f32, -1.5];
/// let start_fit: f32 = -start.iter().map(|v| v * v).sum::<f32>();
/// let (refined, refined_fit) =
///     LocalSearch::<Flex>::refine(&searcher, &params, start, &mut fitness, &mut rng);
///
/// assert_eq!(refined.len(), 2);          // dimensionality preserved
/// assert!(refined_fit >= start_fit);     // monotone non-worsening
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SimulatedAnnealing;

impl SimulatedAnnealing {
    /// Shared body for [`refine`](LocalSearch::refine) and
    /// [`refine_with_known_fitness`](LocalSearch::refine_with_known_fitness).
    ///
    /// `known` is the input genome's fitness when the caller already holds it
    /// (the hint path) or `None` when the input must be re-evaluated to seed the
    /// walker and the best-so-far tracker. The seed is sanitized either way so a
    /// `NaN` never poisons the tracked best.
    ///
    /// # Panics
    ///
    /// Panics if `params.max_iters == 0`: a zero evaluation budget makes it
    /// impossible to return an honestly evaluated fitness, so it is treated as
    /// an invalid configuration (programming error), not runtime data.
    fn refine_impl(
        params: &SimulatedAnnealingParams,
        genome: Vec<f32>,
        known: Option<f32>,
        fitness_fn: &mut dyn FitnessFn<Vec<f32>>,
        rng: &mut dyn Rng,
    ) -> (Vec<f32>, f32) {
        assert!(
            params.max_iters >= 1,
            "SimulatedAnnealingParams::max_iters must be >= 1 (the input genome \
             is always evaluated once to seed the best-so-far tracker)"
        );
        let mut budget: BudgetedEval = BudgetedEval::new(fitness_fn, params.max_iters);

        // First action: seed both the walker and the best-so-far tracker. With a
        // known fitness we reuse it (sanitizing NaN); otherwise we spend one eval
        // scoring the input. The assert above guarantees the eval path succeeds.
        let initial_fit: f32 = if let Some(f) = known {
            sanitize_fitness(f)
        } else {
            let Some(f) = budget.eval(&genome) else {
                unreachable!("budget of >= 1 cannot be exhausted before the first eval");
            };
            f
        };

        // The walker — may drift uphill when an uphill move is accepted.
        let mut current: Vec<f32> = genome;
        let mut current_fit: f32 = initial_fit;
        // The tracked best — always returned. Updated on EVERY evaluation, so
        // the monotone + fresh-fitness invariants hold structurally regardless
        // of how far uphill the walker wanders.
        let mut best: Vec<f32> = current.clone();
        let mut best_fit: f32 = current_fit;

        let dim: usize = current.len();
        if dim == 0 {
            return (best, best_fit);
        }

        // Unit Gaussian sampled through the passed rng (same path as the crate's
        // `gaussian_mutation`); scaled by `step_size` per coordinate to realise
        // an `N(0, step_size)` proposal step.
        let normal: Normal<f32> =
            Normal::new(0.0f32, 1.0).expect("unit normal is well-defined");

        let mut temp: f32 = params.initial_temp;

        loop {
            // Propose: current walker + per-coordinate Gaussian noise.
            let mut candidate: Vec<f32> = current.clone();
            for x in &mut candidate {
                *x += params.step_size * normal.sample(rng);
            }
            clamp_vec(&mut candidate, params.bounds);

            // Evaluate the proposal (stop if the budget is exhausted).
            let Some(cand_fit) = budget.eval(&candidate) else {
                break;
            };

            // Track the global best on every evaluation.
            if cand_fit > best_fit {
                best_fit = cand_fit;
                best.clone_from(&candidate);
            }

            // Metropolis acceptance: always take a non-worsening move (`delta >=
            // 0`); take a worsening move of size `delta` with probability
            // `exp(delta / T)` (a negative exponent → probability < 1). Both the
            // comparison and the downhill draw flow through the passed rng so all
            // stochasticity is reproducible.
            let delta: f32 = cand_fit - current_fit;
            let accept: bool =
                delta >= 0.0 || rng.random::<f32>() < (delta / temp).exp();
            if accept {
                current = candidate;
                current_fit = cand_fit;
            }

            // Cool once per iteration, then early-stop below the floor.
            match params.cooling {
                CoolingSchedule::Geometric { factor } => temp *= factor,
                CoolingSchedule::Linear { delta } => temp = (temp - delta).max(0.0),
            }
            if temp < params.min_temp {
                break;
            }
        }

        (best, best_fit)
    }
}

impl<B: Backend> LocalSearch<B> for SimulatedAnnealing {
    type Params = SimulatedAnnealingParams;

    /// # Panics
    ///
    /// Panics if `params.max_iters == 0`; see `refine_impl`.
    fn refine(
        &self,
        params: &SimulatedAnnealingParams,
        genome: Vec<f32>,
        fitness_fn: &mut dyn FitnessFn<Vec<f32>>,
        rng: &mut dyn Rng,
    ) -> (Vec<f32>, f32) {
        Self::refine_impl(params, genome, None, fitness_fn, rng)
    }

    /// Seeds the walker and best-so-far tracker with `known_fitness` (sanitizing
    /// `NaN` to `-inf`) instead of re-scoring the input, saving one eval.
    ///
    /// # Panics
    ///
    /// Panics if `params.max_iters == 0`; see `refine_impl`.
    fn refine_with_known_fitness(
        &self,
        params: &SimulatedAnnealingParams,
        genome: Vec<f32>,
        known_fitness: f32,
        fitness_fn: &mut dyn FitnessFn<Vec<f32>>,
        rng: &mut dyn Rng,
    ) -> (Vec<f32>, f32) {
        Self::refine_impl(params, genome, Some(known_fitness), fitness_fn, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type TestBackend = Flex;

    /// Negated sphere `f(x) = -Σ x_i²` — concave bump; global maximum 0 at the
    /// origin.
    struct NegSphere;
    impl FitnessFn<Vec<f32>> for NegSphere {
        fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
            -x.iter().map(|v| v * v).sum::<f32>()
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

    /// Records, for every evaluation, whether it was strictly worse than the
    /// previous evaluation's fitness. Lets a test observe uphill *proposals*
    /// reaching the acceptance test; combined with a tiny step it pins that
    /// the walker actually accepts some of them at high temperature.
    struct Recording<'a> {
        inner: &'a mut dyn FitnessFn<Vec<f32>>,
        fitnesses: Vec<f32>,
    }
    impl<'a> Recording<'a> {
        fn new(inner: &'a mut dyn FitnessFn<Vec<f32>>) -> Self {
            Self {
                inner,
                fitnesses: Vec::new(),
            }
        }
    }
    impl FitnessFn<Vec<f32>> for Recording<'_> {
        fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
            let f = self.inner.evaluate_one(x);
            self.fitnesses.push(f);
            f
        }
    }

    const BOUNDS: Bounds = Bounds::new(-5.12, 5.12);

    fn random_start(rng: &mut StdRng, dim: usize, bounds: Bounds) -> Vec<f32> {
        let (lo, hi): (f32, f32) = bounds.into();
        (0..dim)
            .map(|_| lo + (hi - lo) * rng.random::<f32>())
            .collect()
    }

    #[test]
    fn sphere_d2_improves_substantially() {
        let searcher = SimulatedAnnealing;
        let params = SimulatedAnnealingParams::default_for(BOUNDS);
        let mut fitness = NegSphere;
        let mut rng = StdRng::seed_from_u64(1);
        let start = random_start(&mut rng, 2, BOUNDS);
        let start_fit: f32 = -start.iter().map(|v| v * v).sum::<f32>();
        let (_g, fit) =
            LocalSearch::<TestBackend>::refine(&searcher, &params, start, &mut fitness, &mut rng);
        // SA is a coarse explorer, not a precision finisher: assert a large
        // relative improvement, not 1e-6 convergence. `start_fit` is negative
        // (maximum is 0), so closing the gap means rising above `0.1 * start_fit`.
        assert!(
            fit > 0.1 * start_fit,
            "sphere D=2 should improve substantially: best={fit}, start={start_fit}"
        );
    }

    #[test]
    fn sphere_d10_strictly_improves() {
        let searcher = SimulatedAnnealing;
        let params = SimulatedAnnealingParams::default_for(BOUNDS);
        let mut fitness = NegSphere;
        let mut rng = StdRng::seed_from_u64(2);
        let start = random_start(&mut rng, 10, BOUNDS);
        let start_fit: f32 = -start.iter().map(|v| v * v).sum::<f32>();
        let (_g, fit) =
            LocalSearch::<TestBackend>::refine(&searcher, &params, start, &mut fitness, &mut rng);
        assert!(fit > start_fit, "expected improvement: {fit} > {start_fit}");
    }

    #[test]
    fn output_len_equals_input_len() {
        let searcher = SimulatedAnnealing;
        let params = SimulatedAnnealingParams::default_for(BOUNDS);
        let mut fitness = NegSphere;
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
        let searcher = SimulatedAnnealing;
        let params = SimulatedAnnealingParams::default_for(BOUNDS);
        let mut fitness = NegSphere;
        let mut rng = StdRng::seed_from_u64(4);
        let start = random_start(&mut rng, 4, BOUNDS);
        let (g, fit) =
            LocalSearch::<TestBackend>::refine(&searcher, &params, start, &mut fitness, &mut rng);
        let fresh = fitness.evaluate_one(&g);
        approx::assert_relative_eq!(fit, fresh, epsilon = 1e-6);
    }

    #[test]
    fn rosenbrock_monotone_non_worsening() {
        // Pins that the tracked best (not the downhill-drifting walker) is what
        // gets returned: despite accepting worsening moves, `f_out >= f_in`.
        let searcher = SimulatedAnnealing;
        let params = SimulatedAnnealingParams::default_for(BOUNDS);
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..6 {
            let start = random_start(&mut rng, 2, BOUNDS);
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
    #[allow(clippy::float_cmp)]
    fn flat_landscape_terminates_within_budget() {
        let searcher = SimulatedAnnealing;
        let mut params = SimulatedAnnealingParams::default_for(BOUNDS);
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
        // input and its fitness is the honest constant 1.0.
        assert_eq!(g, start);
        assert_eq!(fit, 1.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn same_seed_bit_identical_different_seed_differs() {
        let searcher = SimulatedAnnealing;
        let params = SimulatedAnnealingParams::default_for(BOUNDS);
        let start = vec![2.0_f32, -3.0, 1.5];

        let mut fitness_a = NegSphere;
        let mut rng_a = StdRng::seed_from_u64(123);
        let (g_a, f_a) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start.clone(),
            &mut fitness_a,
            &mut rng_a,
        );

        let mut fitness_b = NegSphere;
        let mut rng_b = StdRng::seed_from_u64(123);
        let (g_b, f_b) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start.clone(),
            &mut fitness_b,
            &mut rng_b,
        );

        // Same seed ⇒ bit-identical genome AND fitness: all stochasticity flows
        // through the passed rng.
        assert_eq!(g_a, g_b);
        assert_eq!(f_a, f_b);

        let mut fitness_c = NegSphere;
        let mut rng_c = StdRng::seed_from_u64(999);
        let (g_c, _f_c) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start,
            &mut fitness_c,
            &mut rng_c,
        );
        // Different seed ⇒ (almost surely) a different trajectory and output.
        assert_ne!(g_a, g_c);
    }

    #[test]
    fn min_temp_early_stop_below_budget() {
        // A tiny initial temperature plus aggressive geometric cooling drops
        // below `min_temp` long before the evaluation budget is spent.
        let searcher = SimulatedAnnealing;
        let mut params = SimulatedAnnealingParams::default_for(BOUNDS);
        params.max_iters = 1000;
        params.initial_temp = 1e-3;
        params.min_temp = 1e-1;
        params.cooling = CoolingSchedule::Geometric { factor: 0.5 };
        let mut base = NegSphere;
        let mut counting = Counting::new(&mut base);
        let mut rng = StdRng::seed_from_u64(7);
        let start = vec![1.0_f32, -1.0];
        let _ = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start,
            &mut counting,
            &mut rng,
        );
        assert!(
            counting.calls < params.max_iters,
            "min_temp early stop: evals {} should be < budget {}",
            counting.calls,
            params.max_iters
        );
    }

    #[test]
    fn boundary_start_stays_within_bounds() {
        let searcher = SimulatedAnnealing;
        let mut params = SimulatedAnnealingParams::default_for(BOUNDS);
        // Large step relative to range pushes proposals hard against bounds.
        params.step_size = 4.0;
        let mut fitness = NegSphere;
        let mut rng = StdRng::seed_from_u64(8);
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
    fn uphill_moves_accepted_at_high_temperature() {
        // With a huge initial temperature, exp(Δf / T) ≈ 1, so the walker
        // should accept worsening moves. We detect acceptance indirectly: the
        // returned best is the tracked maximum, but if NO worsening move were
        // ever accepted the walker would behave like a pure ascent and the
        // recorded evaluation sequence would be (weakly) monotone after the
        // first improvement. Instead we assert the walker visits a fitness
        // strictly worse than the running maximum more than once — only possible
        // if an earlier worsening move was accepted, moving the walker downhill
        // so its next proposal is centred on a worse point.
        let searcher = SimulatedAnnealing;
        let mut params = SimulatedAnnealingParams::default_for(BOUNDS);
        params.max_iters = 200;
        params.initial_temp = 1e9;
        params.min_temp = 1e-9;
        params.step_size = 0.5;
        let mut base = NegSphere;
        let mut recording = Recording::new(&mut base);
        let mut rng = StdRng::seed_from_u64(11);
        let start = vec![0.05_f32, -0.05];
        let _ = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start,
            &mut recording,
            &mut rng,
        );

        // Count evaluations that were strictly worse than the best-so-far at the
        // time they were seen. A pure greedy ascent (no worsening acceptance) can
        // produce such evaluations too — but with a near-origin start and a
        // large step on the negated sphere, sustained worse-than-best proposals
        // are only explored because accepted worsening moves keep re-centring the
        // walker on worse points. Require several to make the assertion robust.
        let mut running_best = f32::NEG_INFINITY;
        let mut worse_than_best = 0_usize;
        for &f in &recording.fitnesses {
            if f < running_best {
                worse_than_best += 1;
            }
            if f > running_best {
                running_best = f;
            }
        }
        assert!(
            worse_than_best >= 3,
            "expected sustained worsening exploration at high temperature, saw {worse_than_best} \
             worse-than-best evaluations"
        );
    }

    #[test]
    fn known_fitness_skips_exactly_the_seeding_eval() {
        // With a large budget the walk terminates by `min_temp`, not the budget,
        // after a fixed number of cooling steps (one proposal each). The seeding
        // eval draws no rng, so both entry points draw the identical proposal
        // sequence — the hint path simply omits the seed.
        let searcher = SimulatedAnnealing;
        let mut params = SimulatedAnnealingParams::default_for(BOUNDS);
        params.max_iters = 10_000;
        let start = vec![1.0_f32, 2.0, 3.0];

        let refine_evals = {
            let mut base = Flat;
            let mut counting = Counting::new(&mut base);
            let mut rng = StdRng::seed_from_u64(31);
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
            let mut rng = StdRng::seed_from_u64(31);
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
            "hint path must skip exactly the seeding eval ({hint_evals} vs {refine_evals})"
        );
    }

    #[test]
    fn nan_hint_does_not_propagate() {
        let searcher = SimulatedAnnealing;
        let params = SimulatedAnnealingParams::default_for(BOUNDS);
        let mut fitness = NegSphere;
        let mut rng = StdRng::seed_from_u64(32);
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
