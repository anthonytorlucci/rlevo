//! Nelder–Mead downhill simplex.
//!
//! The Nelder–Mead method maintains a simplex of `n + 1` vertices in an
//! `n`-dimensional space and refines it via reflection, expansion,
//! contraction, and shrink steps until the spread of vertex fitnesses falls
//! below a tolerance or the evaluation budget is exhausted.
//!
//! # Algorithm
//!
//! Each iteration sorts the simplex by fitness and, using the centroid of all
//! but the worst vertex, attempts (in order) a **reflection** of the worst
//! vertex through that centroid, an **expansion** further along the reflection
//! ray when reflection produced a new best, a **contraction** toward the
//! centroid when reflection is no better than the second-worst vertex, and
//! finally a **shrink** of every vertex toward the current best when even
//! contraction fails. Every trial point is clamped into [`NelderMeadParams`]
//! `bounds` *before* evaluation, so the fitness always corresponds to a
//! feasible point.
//!
//! # Invariants
//!
//! The input genome is the first simplex vertex and is therefore the first
//! point evaluated; a best-so-far `(genome, fitness)` pair is updated on every
//! evaluation and is what `refine` returns. This makes the
//! [`LocalSearch`]
//! monotone-non-worsening and fresh-fitness invariants hold structurally, and
//! keeps even degenerate budgets (fewer evaluations than `n + 1`) safe — they
//! simply return the best vertex evaluated before the budget ran out.
//!
//! Nelder–Mead is fully deterministic: the `rng` argument is unused.

use burn::tensor::backend::Backend;
use rand::Rng;

use crate::fitness::FitnessFn;
use crate::local_search::{clamp_vec, BudgetedEval, LocalSearch};

/// Static configuration for a [`NelderMead`] run.
#[derive(Debug, Clone)]
pub struct NelderMeadParams {
    /// Inclusive search-space bounds `(lo, hi)`; refined genomes are clamped
    /// here and simplex vertices are flipped inward at the boundary.
    pub bounds: (f32, f32),
    /// Hard cap on the **total** number of `evaluate_one` calls per `refine`,
    /// counting the up-to-`n + 1` simplex-initialization evaluations.
    /// Default `200`.
    pub max_iters: usize,
    /// Reflection coefficient (α). Standard value `1.0`.
    pub alpha: f32,
    /// Expansion coefficient (γ). Standard value `2.0`.
    pub gamma: f32,
    /// Contraction coefficient (ρ). Standard value `0.5`.
    pub rho: f32,
    /// Shrink coefficient (σ). Standard value `0.5`.
    pub sigma: f32,
    /// Axis nudge used to build the initial simplex from the input vertex.
    /// Vertex `j` (for `j` in `1..=n`) perturbs coordinate `j - 1` of the input
    /// by `+initial_step`, flipped to `-initial_step` when the forward nudge
    /// would leave `bounds`. Default `0.05 * (hi - lo)`.
    pub initial_step: f32,
    /// Early-stop tolerance on the spread of vertex fitnesses (best vs worst).
    /// The main loop terminates once `f_worst - f_best < tolerance`.
    /// Default `1e-8`.
    pub tolerance: f32,
}

impl NelderMeadParams {
    /// Default parameters derived from the search-space `bounds`.
    ///
    /// `alpha = 1.0`, `gamma = 2.0`, `rho = 0.5`, `sigma = 0.5`,
    /// `initial_step = 0.05 * (hi - lo)`, `tolerance = 1e-8`,
    /// `max_iters = 200`.
    #[must_use]
    pub fn default_for(bounds: (f32, f32)) -> Self {
        let (lo, hi) = bounds;
        Self {
            bounds,
            max_iters: 200,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
            initial_step: 0.05 * (hi - lo),
            tolerance: 1e-8,
        }
    }
}

/// Nelder–Mead downhill-simplex local search.
///
/// A unit struct: all configuration lives in [`NelderMeadParams`], so one
/// instance can refine many genomes. The method is gradient-free, host-side,
/// and fully deterministic — see the [module docs](self) for the per-iteration
/// reflection/expansion/contraction/shrink schedule.
///
/// # Example
///
/// ```
/// use burn::backend::Flex;
/// use rand::{rngs::StdRng, SeedableRng};
/// use rlevo_evolution::fitness::FitnessFn;
/// use rlevo_evolution::local_search::{LocalSearch, NelderMead, NelderMeadParams};
///
/// // Minimize the 2-D sphere; the optimum is the origin with fitness 0.
/// struct Sphere;
/// impl FitnessFn<Vec<f32>> for Sphere {
///     fn evaluate_one(&mut self, x: &Vec<f32>) -> f32 {
///         x.iter().map(|v| v * v).sum()
///     }
/// }
///
/// let searcher = NelderMead;
/// let params = NelderMeadParams::default_for((-5.12, 5.12));
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
pub struct NelderMead;

/// One simplex vertex: a clamped, feasible point and its (fresh) fitness.
struct Vertex {
    /// Coordinates, always within `bounds`.
    point: Vec<f32>,
    /// Fitness of `point` under the budgeted fitness function.
    fitness: f32,
}

impl<B: Backend> LocalSearch<B> for NelderMead {
    type Params = NelderMeadParams;

    /// # Panics
    ///
    /// Panics if `params.max_iters == 0`: a zero evaluation budget makes it
    /// impossible to return an honestly evaluated fitness, so it is treated
    /// as an invalid configuration (programming error), not runtime data.
    #[allow(clippy::too_many_lines)]
    fn refine(
        &self,
        params: &NelderMeadParams,
        genome: Vec<f32>,
        fitness_fn: &mut dyn FitnessFn<Vec<f32>>,
        _rng: &mut dyn Rng,
    ) -> (Vec<f32>, f32) {
        assert!(
            params.max_iters >= 1,
            "NelderMeadParams::max_iters must be >= 1 (the input genome is \
             always evaluated once to seed the best-so-far tracker)"
        );
        let mut budget: BudgetedEval<'_> = BudgetedEval::new(fitness_fn, params.max_iters);

        let dim: usize = genome.len();

        // The tracked best — always returned. Updated on EVERY evaluation via
        // `evaluate`, so the monotone + fresh-fitness invariants hold
        // structurally. The input genome is the first point evaluated.
        let (lo, hi): (f32, f32) = params.bounds;

        // First action: evaluate the input genome (1 eval), clamped to bounds.
        let mut vertex0: Vec<f32> = genome;
        clamp_vec(&mut vertex0, params.bounds);
        let mut best: Vec<f32> = vertex0.clone();
        // The assert above guarantees the first eval succeeds.
        let Some(initial_fit) = budget.eval(&vertex0) else {
            unreachable!("budget of >= 1 cannot be exhausted before the first eval");
        };
        let mut best_fit: f32 = initial_fit;

        // A zero-dimensional genome has no neighbours to explore: the single
        // (empty) point is already the answer.
        if dim == 0 {
            return (best, best_fit);
        }

        // Build the simplex: vertex 0 is the (clamped) input; vertices 1..=dim
        // each nudge one coordinate by `initial_step`, flipped inward at a
        // bound. Each costs one eval, truncated by the budget — a budget too
        // small to finish initialization simply returns the best vertex
        // evaluated so far.
        let mut simplex: Vec<Vertex> = Vec::with_capacity(dim + 1);
        simplex.push(Vertex {
            point: vertex0.clone(),
            fitness: initial_fit,
        });

        for j in 0..dim {
            let mut point: Vec<f32> = vertex0.clone();
            let forward: f32 = point[j] + params.initial_step;
            if forward > hi || forward < lo {
                point[j] -= params.initial_step;
            } else {
                point[j] = forward;
            }
            clamp_vec(&mut point, params.bounds);

            let Some(fitness) = budget.eval(&point) else {
                // Budget exhausted mid-initialization: return the best vertex
                // found so far (never worse than the input).
                update_best(&mut best, &mut best_fit, &simplex);
                return (best, best_fit);
            };
            if fitness < best_fit {
                best_fit = fitness;
                best.clone_from(&point);
            }
            simplex.push(Vertex { point, fitness });
        }

        // Main Nelder–Mead loop. Every `eval_clamped` call updates the
        // best-so-far tracker and consumes budget; a `None` return means the
        // budget is exhausted and we stop immediately.
        let n: usize = simplex.len(); // == dim + 1
        loop {
            // Sort ascending by fitness: index 0 is best, last is worst.
            simplex.sort_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });

            let f_best: f32 = simplex[0].fitness;
            let f_worst: f32 = simplex[n - 1].fitness;
            let f_second_worst: f32 = simplex[n - 2].fitness;

            // f-spread convergence test.
            if f_worst - f_best < params.tolerance {
                break;
            }
            if budget.remaining() == 0 {
                break;
            }

            // Centroid of all vertices except the worst.
            let centroid: Vec<f32> = centroid_excluding_worst(&simplex, dim);
            let worst_point: &[f32] = &simplex[n - 1].point;

            // Reflection: x_r = centroid + alpha * (centroid - worst).
            let reflected: Vec<f32> =
                affine(&centroid, &centroid, worst_point, params.alpha, params.bounds);
            let Some(f_reflected) =
                eval_clamped(&mut budget, &reflected, &mut best, &mut best_fit)
            else {
                break;
            };

            if f_reflected < f_best {
                // Reflection improved on the best: try to expand further.
                let expanded: Vec<f32> = affine(
                    &centroid,
                    &reflected,
                    &centroid,
                    params.gamma,
                    params.bounds,
                );
                let Some(f_expanded) =
                    eval_clamped(&mut budget, &expanded, &mut best, &mut best_fit)
                else {
                    break;
                };
                if f_expanded < f_reflected {
                    replace_worst(&mut simplex, expanded, f_expanded);
                } else {
                    replace_worst(&mut simplex, reflected, f_reflected);
                }
            } else if f_reflected < f_second_worst {
                // Reflection is a middling improvement: accept it.
                replace_worst(&mut simplex, reflected, f_reflected);
            } else {
                // Reflection is no better than the second-worst: contract.
                // Outside contraction if reflection still beats the worst,
                // otherwise inside contraction toward the centroid.
                let (target, target_fit): (&[f32], f32) = if f_reflected < f_worst {
                    (&reflected, f_reflected)
                } else {
                    (worst_point, f_worst)
                };
                let contracted: Vec<f32> =
                    affine(&centroid, target, &centroid, params.rho, params.bounds);
                let Some(f_contracted) =
                    eval_clamped(&mut budget, &contracted, &mut best, &mut best_fit)
                else {
                    break;
                };

                if f_contracted < target_fit {
                    replace_worst(&mut simplex, contracted, f_contracted);
                } else {
                    // Contraction failed: shrink every non-best vertex toward
                    // the best. Costs up to n - 1 evals, truncated by budget.
                    let best_point: Vec<f32> = simplex[0].point.clone();
                    for v in simplex.iter_mut().skip(1) {
                        let mut shrunk: Vec<f32> = Vec::with_capacity(dim);
                        for (b, c) in best_point.iter().zip(v.point.iter()) {
                            shrunk.push(b + params.sigma * (c - b));
                        }
                        clamp_vec(&mut shrunk, params.bounds);
                        let Some(f_shrunk) =
                            eval_clamped(&mut budget, &shrunk, &mut best, &mut best_fit)
                        else {
                            // Budget exhausted mid-shrink: the partially shrunk
                            // simplex is fine; we return the tracked best next
                            // loop iteration once `remaining() == 0` is hit.
                            // Commit what we computed for this vertex and stop.
                            return (best, best_fit);
                        };
                        v.point = shrunk;
                        v.fitness = f_shrunk;
                    }
                }
            }
        }

        (best, best_fit)
    }
}

/// Evaluates `point` through the budget, updating the best-so-far tracker.
///
/// `point` is assumed already clamped. Returns `None` (without touching the
/// tracker) once the budget is exhausted.
fn eval_clamped(
    budget: &mut BudgetedEval<'_>,
    point: &Vec<f32>,
    best: &mut Vec<f32>,
    best_fit: &mut f32,
) -> Option<f32> {
    let fitness: f32 = budget.eval(point)?;
    if fitness < *best_fit {
        *best_fit = fitness;
        best.clone_from(point);
    }
    Some(fitness)
}

/// Folds the simplex into the best-so-far tracker (used on early init exit).
fn update_best(best: &mut Vec<f32>, best_fit: &mut f32, simplex: &[Vertex]) {
    for v in simplex {
        if v.fitness < *best_fit {
            *best_fit = v.fitness;
            best.clone_from(&v.point);
        }
    }
}

/// Centroid of every simplex vertex except the worst (the last, after sorting).
fn centroid_excluding_worst(simplex: &[Vertex], dim: usize) -> Vec<f32> {
    let count: usize = simplex.len() - 1;
    #[allow(clippy::cast_precision_loss)]
    let inv: f32 = 1.0 / count as f32;
    let mut centroid: Vec<f32> = vec![0.0; dim];
    for v in &simplex[..count] {
        for (c, &p) in centroid.iter_mut().zip(v.point.iter()) {
            *c += p;
        }
    }
    for c in &mut centroid {
        *c *= inv;
    }
    centroid
}

/// Computes `base + coeff * (a - b)`, then clamps into `bounds`.
///
/// This is the shared shape of the reflection, expansion, and contraction
/// updates; the caller picks `base`, `a`, `b`, and `coeff` accordingly.
fn affine(base: &[f32], a: &[f32], b: &[f32], coeff: f32, bounds: (f32, f32)) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(base.len());
    for k in 0..base.len() {
        out.push(base[k] + coeff * (a[k] - b[k]));
    }
    clamp_vec(&mut out, bounds);
    out
}

/// Overwrites the worst (last, after sorting) vertex with a new point/fitness.
fn replace_worst(simplex: &mut [Vertex], point: Vec<f32>, fitness: f32) {
    let last: usize = simplex.len() - 1;
    simplex[last] = Vertex { point, fitness };
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

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
        (0..dim)
            .map(|_| lo + (hi - lo) * rng.random::<f32>())
            .collect()
    }

    #[test]
    fn sphere_d2_converges_below_threshold() {
        let searcher = NelderMead;
        let params = NelderMeadParams::default_for(BOUNDS);
        let mut fitness = Sphere;
        let mut rng = StdRng::seed_from_u64(1);
        let start = random_start(&mut rng, 2, BOUNDS);
        let (_g, fit) =
            LocalSearch::<TestBackend>::refine(&searcher, &params, start, &mut fitness, &mut rng);
        assert!(fit < 1e-6, "sphere D=2 should converge: best={fit}");
    }

    #[test]
    fn sphere_d10_strictly_improves() {
        let searcher = NelderMead;
        let params = NelderMeadParams::default_for(BOUNDS);
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
        let searcher = NelderMead;
        let params = NelderMeadParams::default_for(BOUNDS);
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
        let searcher = NelderMead;
        let params = NelderMeadParams::default_for(BOUNDS);
        let mut fitness = Sphere;
        let mut rng = StdRng::seed_from_u64(4);
        let start = random_start(&mut rng, 4, BOUNDS);
        let (g, fit) =
            LocalSearch::<TestBackend>::refine(&searcher, &params, start, &mut fitness, &mut rng);
        let fresh = fitness.evaluate_one(&g);
        approx::assert_relative_eq!(fit, fresh, epsilon = 1e-6);
    }

    #[test]
    fn rosenbrock_monotone_non_worsening() {
        let searcher = NelderMead;
        let params = NelderMeadParams::default_for(BOUNDS);
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
    fn eval_count_never_exceeds_budget() {
        let searcher = NelderMead;
        let mut params = NelderMeadParams::default_for(BOUNDS);
        params.max_iters = 37;
        let mut base = Flat;
        let mut counting = Counting::new(&mut base);
        let mut rng = StdRng::seed_from_u64(6);
        let start = vec![1.0_f32, 2.0, 3.0, 4.0];
        let (g, _f) = LocalSearch::<TestBackend>::refine(
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
        assert_eq!(g.len(), start.len());
    }

    #[test]
    fn degenerate_budget_no_worse_than_input() {
        // D = 5 needs n + 1 = 6 evals just to initialize the simplex; a budget
        // of 2 dies mid-init and must still return a safe result.
        let searcher = NelderMead;
        let mut params = NelderMeadParams::default_for(BOUNDS);
        params.max_iters = 2;
        let mut fitness = Sphere;
        let mut rng = StdRng::seed_from_u64(7);
        let start = random_start(&mut rng, 5, BOUNDS);
        let start_fit: f32 = start.iter().map(|v| v * v).sum();
        let (g, fit) =
            LocalSearch::<TestBackend>::refine(&searcher, &params, start, &mut fitness, &mut rng);
        assert_eq!(g.len(), 5, "dimensionality preserved");
        assert!(fit <= start_fit, "no worse than input: {fit} <= {start_fit}");
    }

    #[test]
    fn tolerance_early_stops_before_budget() {
        // With a generous budget on the smooth sphere, the f-spread tolerance
        // test should fire well before `max_iters` evaluations are spent.
        let searcher = NelderMead;
        let mut params = NelderMeadParams::default_for(BOUNDS);
        params.max_iters = 1000;
        let mut base = Sphere;
        let mut counting = Counting::new(&mut base);
        let mut rng = StdRng::seed_from_u64(8);
        let start = vec![1.0_f32, -0.5];
        let (_g, _f) = LocalSearch::<TestBackend>::refine(
            &searcher,
            &params,
            start,
            &mut counting,
            &mut rng,
        );
        assert!(
            counting.calls < params.max_iters,
            "tolerance should early-stop: evals {} < budget {}",
            counting.calls,
            params.max_iters
        );
    }

    #[test]
    fn boundary_start_stays_within_bounds() {
        let searcher = NelderMead;
        let params = NelderMeadParams::default_for(BOUNDS);
        let mut fitness = Sphere;
        let mut rng = StdRng::seed_from_u64(9);
        // Start at the upper boundary in every coordinate, so the forward axis
        // nudge would leave bounds and must flip inward.
        let start = vec![BOUNDS.1; 4];
        let (g, _f) =
            LocalSearch::<TestBackend>::refine(&searcher, &params, start, &mut fitness, &mut rng);
        for &x in &g {
            assert!(
                x >= BOUNDS.0 && x <= BOUNDS.1,
                "coord {x} out of bounds {BOUNDS:?}"
            );
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn same_seed_is_bit_identical() {
        let searcher = NelderMead;
        let params = NelderMeadParams::default_for(BOUNDS);
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
