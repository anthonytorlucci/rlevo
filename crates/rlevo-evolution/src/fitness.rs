//! Fitness evaluation traits and adapters.
//!
//! Two traits model the two evaluation shapes strategies expect:
//!
//! - [`FitnessFn`] — evaluates a single member. Callers hand the fitness
//!   function a host-side genome row (typically `Vec<f32>`) and receive a
//!   scalar. Useful for simple benchmarks and for unit-testing operators.
//! - [`BatchFitnessFn`] — evaluates an entire population in one call and
//!   returns a device-resident `Tensor<B, 1>` of shape `(pop_size,)`. This
//!   is the hot path — strategies call it once per generation.
//!
//! Two adapters bridge host-side scalar fitness code into [`BatchFitnessFn`]:
//!
//! - [`FromFitnessEvaluable`] — wraps
//!   `rlevo-core::fitness::FitnessEvaluable<Individual = Vec<f64>, Landscape = L>`.
//!   Use this when an evaluator and a landscape type are already defined
//!   separately (e.g. `RastriginEvaluator` + `RastriginLandscape`).
//! - [`FromLandscape`] — wraps `rlevo-core::fitness::Landscape` directly.
//!   Use this when the landscape is self-evaluating (Sphere, Ackley, Rastrigin)
//!   and a separate evaluator shim would add no value.
//!
//! Both adapters pull each population row to host as `f32`, widen to `f64`,
//! evaluate on the CPU, and re-upload the results as a `Tensor<B, 1>`.
//! Purpose-built batched-on-device landscapes should implement
//! [`BatchFitnessFn`] directly to avoid that round-trip.

use burn::tensor::{Tensor, TensorData, backend::Backend};

use rlevo_core::fitness::{FitnessEvaluable, Landscape};
use rlevo_core::objective::ObjectiveSense;

/// Single-member fitness evaluation.
///
/// Implementors may hold mutable state (e.g. a counter for number of
/// evaluations) and are therefore `&mut self`.
pub trait FitnessFn<G>: Send {
    /// Evaluates one genome and returns its scalar fitness.
    fn evaluate_one(&mut self, member: &G) -> f32;
}

/// Batched fitness evaluation over a population genome container `G`.
///
/// The returned tensor has shape `(pop_size,)` on the supplied device.
/// Implementors must preserve row order — `fitness[i]` refers to the
/// individual at row `i` of `population`.
pub trait BatchFitnessFn<B: Backend, G>: Send {
    /// Evaluates every member of `population` and returns a fitness tensor in
    /// the objective's **natural** value space (no hand-negation).
    ///
    /// The returned `Tensor<B, 1>` has shape `(pop_size,)` and is placed on
    /// `device`. Row order is preserved: `fitness[i]` corresponds to the
    /// individual at row `i` of `population`. Cost objectives return their
    /// natural cost; the harness reconciles direction via [`sense`](Self::sense).
    ///
    /// The returned tensor **may contain `NaN` or `±∞`** — implementors are not
    /// required to sanitize. The
    /// [`EvolutionaryHarness`](crate::strategy::EvolutionaryHarness) canonicalizes
    /// and then sanitizes (ADR 0034) before any [`Strategy::tell`](crate::strategy::Strategy::tell),
    /// so a non-finite fitness cannot poison selection or best-so-far tracking on
    /// harness-driven runs.
    fn evaluate_batch(
        &mut self,
        population: &G,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1>;

    /// The optimisation direction of this objective.
    ///
    /// This is the **single source of truth** the
    /// [`EvolutionaryHarness`](crate::strategy::EvolutionaryHarness) reads to
    /// reconcile a cost objective into the engine's canonical (maximise) space.
    /// It is **required, with no default**, so a reward/accuracy objective
    /// cannot silently inherit the wrong direction by omission — declare it
    /// explicitly ([`ObjectiveSense::Maximize`] for a reward,
    /// [`ObjectiveSense::Minimize`] for a cost). The bundled landscape adapters
    /// ([`FromLandscape`], [`FromFitnessEvaluable`]) forward the landscape's
    /// declared sense.
    fn sense(&self) -> ObjectiveSense;
}

/// Adapter from `FitnessEvaluable` to [`BatchFitnessFn<B, Tensor<B, 2>>`].
///
/// Each row of the population is pulled to host, converted to `Vec<f64>`,
/// and passed to the underlying evaluator with the configured landscape.
/// Fitness is computed on the host and then re-uploaded as a single
/// `Tensor<B, 1>`.
///
/// # Precision
///
/// Populations are read as `f32` and widened to `f64` for the evaluator
/// call; the returned `f64` fitness is narrowed back to `f32` before it
/// is uploaded as a `Tensor<B, 1>`. Fitness values that exceed `f32`
/// range (or rely on sub-ulp precision) will lose information at the
/// narrowing step. Purpose-built batched-on-device landscapes should
/// implement [`BatchFitnessFn`] directly to avoid the round-trip.
///
/// # Type Parameters
///
/// - `FE`: Concrete [`FitnessEvaluable`] implementation.
/// - `L`: Landscape type; must match `FE::Landscape`.
///
/// # Panics
///
/// `evaluate_batch` panics if the supplied population tensor is not rank
/// 2, or if its data cannot be read as `f32` (e.g. an integer backend).
#[derive(Debug)]
pub struct FromFitnessEvaluable<FE, L> {
    evaluator: FE,
    landscape: L,
    sense: ObjectiveSense,
}

impl<FE, L> FromFitnessEvaluable<FE, L> {
    /// Builds the adapter from an evaluator and a landscape, defaulting the
    /// objective sense to [`ObjectiveSense::Minimize`] (the cost convention a
    /// [`FitnessEvaluable`] follows).
    ///
    /// Use [`with_sense`](Self::with_sense) to declare a maximisation objective
    /// (reward, accuracy) explicitly.
    pub fn new(evaluator: FE, landscape: L) -> Self {
        Self::with_sense(evaluator, landscape, ObjectiveSense::Minimize)
    }

    /// Builds the adapter with an explicit [`ObjectiveSense`].
    pub fn with_sense(evaluator: FE, landscape: L, sense: ObjectiveSense) -> Self {
        Self {
            evaluator,
            landscape,
            sense,
        }
    }

    /// Returns a reference to the wrapped landscape.
    pub fn landscape(&self) -> &L {
        &self.landscape
    }
}

impl<FE, L, B> BatchFitnessFn<B, Tensor<B, 2>> for FromFitnessEvaluable<FE, L>
where
    B: Backend,
    FE: FitnessEvaluable<Individual = Vec<f64>, Landscape = L> + Send,
    L: Send + Sync,
{
    fn evaluate_batch(
        &mut self,
        population: &Tensor<B, 2>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        let dims = population.dims();
        assert_eq!(dims.len(), 2, "population tensor must be rank 2");
        let pop_size = dims[0];
        let genome_dim = dims[1];

        let flat = population
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("tensor data must be readable as f32");
        debug_assert_eq!(flat.len(), pop_size * genome_dim);

        let mut fitness = Vec::with_capacity(pop_size);
        let mut individual = Vec::with_capacity(genome_dim);
        for row in 0..pop_size {
            individual.clear();
            let start = row * genome_dim;
            individual.extend(
                flat[start..start + genome_dim]
                    .iter()
                    .map(|&v| f64::from(v)),
            );
            let f = self.evaluator.evaluate(&individual, &self.landscape);
            #[allow(clippy::cast_possible_truncation)]
            fitness.push(f as f32);
        }

        let data = TensorData::new(fitness, [pop_size]);
        Tensor::<B, 1>::from_data(data, device)
    }

    fn sense(&self) -> ObjectiveSense {
        self.sense
    }
}

/// Adapter from [`Landscape`] to [`BatchFitnessFn<B, Tensor<B, 2>>`].
///
/// Use this when the landscape carries its own `evaluate(&[f64]) -> f64`
/// (Sphere, Ackley, Rastrigin) so the example does not need a separate
/// `FitnessEvaluable` shim. Each row is pulled to host as `f32`, widened
/// to `f64`, evaluated, and re-uploaded as a `Tensor<B, 1>` — same
/// precision caveats as [`FromFitnessEvaluable`] apply.
///
/// # Panics
///
/// `evaluate_batch` panics if the supplied population tensor is not rank
/// 2, or if its data cannot be read as `f32` (e.g. an integer backend).
#[derive(Debug)]
pub struct FromLandscape<L> {
    landscape: L,
    sense: ObjectiveSense,
}

impl<L: Landscape> FromLandscape<L> {
    /// Builds the adapter from a self-evaluating landscape, taking the
    /// objective sense from the landscape's [`Landscape::sense`] (which
    /// defaults to [`ObjectiveSense::Minimize`]).
    pub fn new(landscape: L) -> Self {
        let sense = landscape.sense();
        Self { landscape, sense }
    }

    /// Builds the adapter with an explicit [`ObjectiveSense`], overriding the
    /// landscape's declared sense. Examples and showcases spell out
    /// [`ObjectiveSense::Minimize`] here so intent is visible at the call site.
    pub fn with_sense(landscape: L, sense: ObjectiveSense) -> Self {
        Self { landscape, sense }
    }

    /// Returns a reference to the wrapped landscape.
    pub fn landscape(&self) -> &L {
        &self.landscape
    }
}

impl<L, B> BatchFitnessFn<B, Tensor<B, 2>> for FromLandscape<L>
where
    B: Backend,
    L: Landscape,
{
    fn evaluate_batch(
        &mut self,
        population: &Tensor<B, 2>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        let dims = population.dims();
        assert_eq!(dims.len(), 2, "population tensor must be rank 2");
        let pop_size = dims[0];
        let genome_dim = dims[1];

        let flat = population
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("tensor data must be readable as f32");
        debug_assert_eq!(flat.len(), pop_size * genome_dim);

        let mut fitness = Vec::with_capacity(pop_size);
        let mut individual = Vec::with_capacity(genome_dim);
        for row in 0..pop_size {
            individual.clear();
            let start = row * genome_dim;
            individual.extend(
                flat[start..start + genome_dim]
                    .iter()
                    .map(|&v| f64::from(v)),
            );
            let f = self.landscape.evaluate(&individual);
            #[allow(clippy::cast_possible_truncation)]
            fitness.push(f as f32);
        }

        let data = TensorData::new(fitness, [pop_size]);
        Tensor::<B, 1>::from_data(data, device)
    }

    fn sense(&self) -> ObjectiveSense {
        self.sense
    }
}

/// Sanitizes one **canonical (maximise-space)** fitness value: `NaN →`
/// [`f32::NEG_INFINITY`], `+∞ →` [`f32::MAX`], everything else (including `−∞`)
/// passes through.
///
/// This is the crate-wide fitness-hygiene primitive and the single rule of the
/// canonical convention (ADR 0023 / ADR 0034):
///
/// - `NaN → −∞`: `−∞` is the worst value under the maximise convention, so a
///   sanitized `NaN` can never seed or displace a finite best-so-far. Rust's
///   `f32::NAN` is a *positive* NaN, so `total_cmp` would otherwise rank it as
///   the **maximum** (`rules.md` §3) — the exact inversion this prevents.
/// - `+∞ → f32::MAX`: a genuinely optimal individual (a landscape hitting its
///   optimum, an unbounded reward) is mapped to a **finite** value, so it can be
///   compared, stored, and summed **at all**. Ordering is unaffected — the map
///   is monotone, so `total_cmp`, sorts, and argmax still rank it at or above
///   every other member (it ties with a legitimately-`f32::MAX` individual).
///
///   **This bounds one *value*; it does not bound a *reduction* over values.**
///   Because `f32::MAX` is finite, a sanitized `+∞` *joins* a sum instead of
///   being excluded from it, and `f32::MAX + f32::MAX == f32::INFINITY` — two
///   saturated members suffice to drive an `f32` mean to `+∞` (issues #132,
///   #1062), and squaring one suffices for an `f32` variance. What bounds a
///   reduction is the **accumulator width**, not this clamp; no finite sentinel
///   could substitute. Reduce fitness with [`sanitized_mean`] or
///   [`sanitized_sum`], which accumulate in `f64` and narrow at most once, after
///   the reduction (ADR 0069 §Decision 1, extending ADR 0034). A reduction
///   performed by a Burn device op cannot widen its accumulator and instead
///   bounds its terms, as [`shaping::z_score`](crate::shaping::z_score) does
///   (ADR 0069 §Decision 4).
/// - `−∞` passes through: it is the worst-value sentinel *and* the
///   uninitialized `best_fitness_ever` seed, and it must stay non-finite so the
///   mean-over-finite statistic in
///   [`StrategyMetrics::from_host_fitness`](crate::strategy::StrategyMetrics::from_host_fitness)
///   can see and count it as a broken member.
///
/// Applied by [`BudgetedEval::eval`](crate::local_search::BudgetedEval) (every
/// probe, including the seeding eval), the searchers'
/// [`refine_with_known_fitness`](crate::local_search::LocalSearch::refine_with_known_fitness)
/// overrides, the EDA `tell` chokepoint
/// ([`crate::algorithms::eda::EdaStrategy`]), and every NaN-safe fitness sort
/// across the crate (selection, replacement, the ES/NEAT/ACO rankers). For the
/// finite benchmark landscapes the searchers ship against no branch is taken, so
/// it costs only one `is_nan` / `is_infinite` check.
pub(crate) fn sanitize_fitness(f: f32) -> f32 {
    if f.is_nan() {
        f32::NEG_INFINITY
    } else if f.is_infinite() && f.is_sign_positive() {
        // `f == f32::INFINITY` would trip the float-equality lint (rules §5/§8).
        f32::MAX
    } else {
        f
    }
}

/// Mean of `values` under [`sanitize_fitness`], accumulated in `f64` and
/// narrowed to `f32` **once**, after the division (ADR 0069 §Decision 1).
///
/// # Why the accumulator is `f64`
///
/// [`sanitize_fitness`] maps `+∞` to [`f32::MAX`], which is **finite** — so a
/// sanitized `+∞` is not excluded from a sum, it *joins* it. And
/// `f32::MAX + f32::MAX == f32::INFINITY`: **two** saturated members are enough
/// to blow an `f32` accumulator to `+∞`, producing exactly the infinite mean the
/// clamp is often assumed to prevent — a pattern that recurred at more than one
/// call site before it was generalized here rather than patched locally again.
///
/// The clamp and the accumulator width answer two different questions. The clamp
/// bounds one *value*, so it can be compared, stored, and summed **at all**; only
/// the accumulator width bounds a *reduction*. No finite sentinel could do the
/// latter — a sentinel `S` protects a sum over `N` terms only while `N·S <
/// f32::MAX`, which already fails at `N = 2` for `f32::MAX`. `f64` does deliver
/// it: `f64::MAX / f64::from(f32::MAX) ≈ 5.3e269`, so an `f64` accumulator
/// absorbs any population of `f32::MAX` terms that can physically exist. It also
/// removes the ~1 ULP-per-addition drift an `f32` sum accrues over a large
/// population. (ADR 0069, extending ADR 0034.)
///
/// # Contract
///
/// - Every value is sanitized **before** it is accumulated, so no raw `NaN`
///   reaches the sum. `+∞` cannot survive sanitization either, so `−∞ + (+∞)` is
///   unreachable: the result is never `NaN`.
/// - The mean is over **every** value supplied. A single sanitized-`−∞` (broken)
///   member therefore drives the whole result to `−∞`. A caller wanting the
///   mean-over-finite-members statistic — the one
///   [`StrategyMetrics::from_host_fitness`](crate::strategy::StrategyMetrics::from_host_fitness)
///   reports (ADR 0034) — filters the iterator before calling.
/// - **An empty input yields [`f32::NEG_INFINITY`]**, and this function never
///   panics. Rationale: `−∞` is the canonical worst-value sentinel of the
///   maximise-native convention (ADR 0023), it is already what
///   `from_host_fitness` reports for an all-broken population, and the IEEE
///   answer (`0/0 → NaN`) is the one value the crate's hygiene rule exists to
///   eliminate. Making the primitive total rather than panicking also keeps it
///   usable from `pub` functions that take runtime-supplied fitness slices,
///   where a panic would violate `rules.md` §4.
/// - The single narrowing cannot overflow to `±∞` for a non-empty input: every
///   sanitized term is `≤ f32::MAX`, so their mean is too.
pub(crate) fn sanitized_mean(values: impl IntoIterator<Item = f32>) -> f32 {
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for v in values {
        sum += f64::from(sanitize_fitness(v));
        count += 1;
    }
    if count == 0 {
        return f32::NEG_INFINITY;
    }
    // `usize as f64` is exact for any realistic population size.
    #[allow(clippy::cast_precision_loss)]
    let n = count as f64;
    // The single narrowing of the whole reduction; bounded by `f32::MAX` above.
    #[allow(clippy::cast_possible_truncation)]
    let mean = (sum / n) as f32;
    mean
}

/// Sum of `values` under [`sanitize_fitness`], accumulated in — and **returned
/// as** — `f64` (ADR 0069 §Decision 1).
///
/// See [`sanitized_mean`] for why the accumulator is `f64`: `sanitize_fitness`
/// maps `+∞` to the *finite* [`f32::MAX`], which therefore joins the sum, and
/// `f32::MAX + f32::MAX == f32::INFINITY`.
///
/// # Why the return type is `f64`
///
/// Unlike a mean, a sum of `N` sanitized terms legitimately exceeds `f32` range,
/// so there is no width to narrow back to. Returning `f32` would re-introduce at
/// the boundary the very overflow the `f64` accumulator removed. The caller
/// narrows deliberately — or, as in
/// [`allocate_offspring`](crate::neuroevolution::species::allocate_offspring),
/// where the sum is only ever a divisor, does not narrow at all.
///
/// There is deliberately **no `-> f32` sum variant** (ADR 0069 §Decision 2); a
/// caller that genuinely needs one is asking for a mean.
///
/// # Contract
///
/// - Every value is sanitized before it is accumulated, so the result is never
///   `NaN`: raw `NaN` becomes `−∞`, and `+∞` (which would make `−∞ + (+∞)`
///   reachable) cannot survive sanitization.
/// - A sanitized-`−∞` term makes the whole sum `−∞`.
/// - **An empty input yields `0.0`**, the additive identity, and this function
///   never panics. (`sanitized_mean` differs deliberately — an empty *mean* has
///   no identity to fall back on.)
pub(crate) fn sanitized_sum(values: impl IntoIterator<Item = f32>) -> f64 {
    values
        .into_iter()
        .map(|v| f64::from(sanitize_fitness(v)))
        .sum()
}

/// Tensor-level [`sanitize_fitness`] for the driver chokepoints — a single
/// device op over a `(pop_size,)` **canonical-space** fitness vector.
///
/// Applies the same rule (`NaN → −∞`, `+∞ → f32::MAX`, `−∞` pass-through) to a
/// whole fitness tensor without a device→host→device round-trip, so the
/// [`EvolutionaryHarness`](crate::strategy::EvolutionaryHarness) and the
/// coevolution coupled-fitness path can sanitize on the hot path (ADR 0034).
///
/// Order matters: the `NaN → −∞` `mask_fill` runs first (so no `NaN` reaches the
/// clamp, which would propagate it), then `clamp_max(f32::MAX)` caps `+∞` while
/// leaving `−∞` and every finite value untouched. Mirrors the `is_nan` +
/// `mask_fill` + clamp idiom already used by `EdaStrategy::tell`'s gene backstop.
#[must_use]
pub(crate) fn sanitize_fitness_tensor<B: Backend>(fitness: Tensor<B, 1>) -> Tensor<B, 1> {
    let nan_mask = fitness.clone().is_nan();
    fitness
        .mask_fill(nan_mask, f32::NEG_INFINITY)
        .clamp_max(f32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    type TestBackend = Flex;

    #[derive(Debug, Clone, Copy)]
    struct Sphere;

    struct SphereFit;
    impl FitnessEvaluable for SphereFit {
        type Individual = Vec<f64>;
        type Landscape = Sphere;
        fn evaluate(&self, x: &Self::Individual, _: &Self::Landscape) -> f64 {
            x.iter().map(|v| v * v).sum()
        }
    }

    #[test]
    fn from_fitness_evaluable_preserves_row_order() {
        let device = Default::default();
        let data = TensorData::new(
            vec![1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
            [3, 3],
        );
        let pop = Tensor::<TestBackend, 2>::from_data(data, &device);

        let mut adapter = FromFitnessEvaluable::new(SphereFit, Sphere);
        let fitness = adapter.evaluate_batch(&pop, &device);

        let values = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness host-read of a tensor this test just built");
        assert_eq!(values.len(), 3);
        approx::assert_relative_eq!(values[0], 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(values[1], 4.0, epsilon = 1e-6);
        approx::assert_relative_eq!(values[2], 9.0, epsilon = 1e-6);
    }

    #[test]
    fn from_landscape_preserves_row_order() {
        let device = Default::default();
        let data = TensorData::new(
            vec![1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
            [3, 3],
        );
        let pop = Tensor::<TestBackend, 2>::from_data(data, &device);

        let mut adapter = FromLandscape::new(SphereLandscape);
        let fitness = adapter.evaluate_batch(&pop, &device);

        let values = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness host-read of a tensor this test just built");
        assert_eq!(values.len(), 3);
        approx::assert_relative_eq!(values[0], 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(values[1], 4.0, epsilon = 1e-6);
        approx::assert_relative_eq!(values[2], 9.0, epsilon = 1e-6);
    }

    struct SphereLandscape;
    impl Landscape for SphereLandscape {
        fn evaluate(&self, x: &[f64]) -> f64 {
            x.iter().map(|v| v * v).sum()
        }
    }

    /// The scalar hygiene rule (ADR 0034): `NaN → −∞`, `+∞ → f32::MAX`, `−∞` and
    /// finite values pass through unchanged.
    #[test]
    fn sanitize_fitness_scalar_applies_canonical_rule() {
        // `−∞` sentinel: assert via `is_infinite`/sign, not float `==` (rules §5/§8).
        let nan_out: f32 = sanitize_fitness(f32::NAN);
        assert!(
            nan_out.is_infinite() && nan_out.is_sign_negative(),
            "NaN → −∞"
        );
        approx::assert_relative_eq!(sanitize_fitness(f32::INFINITY), f32::MAX);
        let neg_out: f32 = sanitize_fitness(f32::NEG_INFINITY);
        assert!(
            neg_out.is_infinite() && neg_out.is_sign_negative(),
            "−∞ passes through"
        );
        approx::assert_relative_eq!(sanitize_fitness(2.5), 2.5, epsilon = 1e-6);
        approx::assert_relative_eq!(sanitize_fitness(-7.0), -7.0, epsilon = 1e-6);
    }

    /// The tensor sibling applies the identical rule element-wise, and — crucially
    /// — leaves `−∞` non-finite (it is not clamped to `−f32::MAX`) so downstream
    /// mean-over-finite logic can still detect and count it.
    #[test]
    fn sanitize_fitness_tensor_matches_scalar_rule() {
        let device = Default::default();
        let data = TensorData::new(
            vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3.0_f32, -4.0],
            [5],
        );
        let t = Tensor::<TestBackend, 1>::from_data(data, &device);
        let out = sanitize_fitness_tensor(t)
            .into_data()
            .into_vec::<f32>()
            .expect("fitness host-read of a tensor this test just built");

        assert!(
            out[0].is_infinite() && out[0].is_sign_negative(),
            "NaN → −∞"
        );
        approx::assert_relative_eq!(out[1], f32::MAX); // +∞ → f32::MAX
        assert!(
            out[2].is_infinite() && out[2].is_sign_negative(),
            "−∞ passes through, stays non-finite"
        );
        approx::assert_relative_eq!(out[3], 3.0, epsilon = 1e-6);
        approx::assert_relative_eq!(out[4], -4.0, epsilon = 1e-6);
    }

    /// The case that motivated ADR 0069: **two** sanitized `+∞` members.
    ///
    /// `sanitize_fitness(+∞) == f32::MAX`, which is finite and therefore joins
    /// the sum — and `f32::MAX + f32::MAX == f32::INFINITY` in `f32`. With the
    /// `f64` accumulator the mean of two `f32::MAX` values is `f32::MAX`, not
    /// `+∞`. The `f32`-accumulator control below is what the assertion is
    /// distinguishing itself from.
    #[test]
    fn sanitized_mean_two_saturated_members_do_not_overflow() {
        // Control: the same reduction in `f32` is exactly the defect.
        let f32_control: f32 = [f32::INFINITY, f32::INFINITY]
            .into_iter()
            .map(sanitize_fitness)
            .sum::<f32>()
            / 2.0;
        assert!(
            f32_control.is_infinite() && f32_control.is_sign_positive(),
            "control: an `f32` accumulator saturates two clamped members to `+∞`"
        );

        let mean = sanitized_mean([f32::INFINITY, f32::INFINITY]);
        assert!(
            mean.is_finite(),
            "an `f64` accumulator keeps the mean of two clamped members finite"
        );
        approx::assert_relative_eq!(mean, f32::MAX);

        // The sum is the same reduction without the division: it exceeds `f32`
        // range, which is why `sanitized_sum` returns `f64`.
        let sum = sanitized_sum([f32::INFINITY, f32::INFINITY]);
        assert!(
            sum.is_finite(),
            "the `f64` sum of two clamped members is finite"
        );
        approx::assert_relative_eq!(sum, 2.0 * f64::from(f32::MAX));
        #[allow(clippy::cast_possible_truncation)]
        let narrowed = sum as f32;
        assert!(
            narrowed.is_infinite(),
            "and it does not fit in `f32` — the reason there is no `-> f32` sum"
        );
    }

    /// An all-`−∞` (all-broken) input reduces to `−∞`, not `NaN`: `−∞` passes
    /// through sanitization and `−∞ / n == −∞`.
    #[test]
    fn sanitized_mean_all_negative_infinity_stays_negative_infinity() {
        let mean = sanitized_mean([f32::NEG_INFINITY; 4]);
        assert!(
            mean.is_infinite() && mean.is_sign_negative(),
            "an all-broken input means to `−∞`, not `NaN`"
        );
        let sum = sanitized_sum([f32::NEG_INFINITY; 4]);
        assert!(
            sum.is_infinite() && sum.is_sign_negative(),
            "and sums to `−∞`"
        );
    }

    /// Mixed `−∞` / `f32::MAX`: the sanitization rule makes `−∞ + (+∞)` — the one
    /// arithmetic combination that yields `NaN` — unreachable, because `+∞` never
    /// survives `sanitize_fitness`. A single broken member therefore drives the
    /// whole reduction to `−∞`, deterministically.
    #[test]
    fn sanitized_mean_mixed_broken_and_saturated_is_negative_infinity_not_nan() {
        let mean = sanitized_mean([f32::NEG_INFINITY, f32::INFINITY, 1.0]);
        assert!(!mean.is_nan(), "no `NaN` can arise: `+∞` is clamped first");
        assert!(
            mean.is_infinite() && mean.is_sign_negative(),
            "one broken member drives the unfiltered mean to `−∞`"
        );

        let sum = sanitized_sum([f32::NEG_INFINITY, f32::INFINITY, 1.0]);
        assert!(!sum.is_nan(), "no `NaN` can arise in the sum either");
        assert!(
            sum.is_infinite() && sum.is_sign_negative(),
            "the sum is `−∞`"
        );

        // Filtering the broken member out — how `from_host_fitness` spells
        // mean-over-finite-members — recovers the finite statistic.
        let finite_mean = sanitized_mean(
            [f32::NEG_INFINITY, f32::INFINITY, 1.0]
                .into_iter()
                .filter(|&f| sanitize_fitness(f).is_finite()),
        );
        #[allow(clippy::cast_possible_truncation)]
        let expected = f64::midpoint(f64::from(f32::MAX), 1.0) as f32;
        approx::assert_relative_eq!(finite_mean, expected);
    }

    /// A raw `NaN` is sanitized to `−∞` *before* it reaches the accumulator, so
    /// it can never propagate through the reduction as a `NaN`.
    #[test]
    fn sanitized_mean_raw_nan_is_sanitized_before_accumulation() {
        let mean = sanitized_mean([1.0, f32::NAN, 3.0]);
        assert!(!mean.is_nan(), "a raw `NaN` never reaches the sum");
        assert!(
            mean.is_infinite() && mean.is_sign_negative(),
            "`NaN → −∞` (worst) dominates the unfiltered mean"
        );

        let sum = sanitized_sum([1.0, f32::NAN, 3.0]);
        assert!(!sum.is_nan(), "a raw `NaN` never reaches the sum");
        assert!(
            sum.is_infinite() && sum.is_sign_negative(),
            "the sum is `−∞`"
        );
    }

    /// Empty input: `sanitized_mean` is **total** and returns the canonical
    /// worst-value sentinel `−∞` (ADR 0023) rather than panicking or yielding
    /// `NaN`; `sanitized_sum` returns the additive identity `0.0`.
    #[test]
    fn sanitized_reductions_are_total_on_empty_input() {
        let mean = sanitized_mean(std::iter::empty::<f32>());
        assert!(
            mean.is_infinite() && mean.is_sign_negative(),
            "an empty mean is the worst-value sentinel, not `NaN`"
        );
        approx::assert_relative_eq!(sanitized_sum(std::iter::empty::<f32>()), 0.0);
    }

    /// A large input count: `100_000` members, half of them saturated to
    /// `f32::MAX`. The `f32` control overflows to `+∞` after the *second*
    /// saturated addition; the `f64` accumulator carries `50_000 × f32::MAX ≈
    /// 1.7e43` without trouble, and the narrowing of the mean lands back inside
    /// `f32` range because a mean of values bounded by `f32::MAX` is bounded by
    /// `f32::MAX`.
    #[test]
    fn sanitized_mean_large_population_of_saturated_members() {
        const N: usize = 100_000;
        let values: Vec<f32> = (0..N)
            .map(|i| if i % 2 == 0 { f32::INFINITY } else { 0.0 })
            .collect();

        let f32_control: f32 = values.iter().copied().map(sanitize_fitness).sum();
        assert!(
            f32_control.is_infinite(),
            "control: the `f32` accumulator is already `+∞`"
        );

        let sum = sanitized_sum(values.iter().copied());
        assert!(sum.is_finite(), "the `f64` accumulator carries ~1.7e43");
        #[allow(clippy::cast_precision_loss)]
        let expected_sum = (N / 2) as f64 * f64::from(f32::MAX);
        approx::assert_relative_eq!(sum, expected_sum, max_relative = 1e-12);

        let mean = sanitized_mean(values.iter().copied());
        assert!(mean.is_finite(), "and the narrowed mean is finite");
        approx::assert_relative_eq!(mean, f32::MAX / 2.0, max_relative = 1e-6);
    }

    /// The ordinary case, pinned so the `f64` detour cannot silently change a
    /// finite answer: mean and sum of a plain finite population.
    #[test]
    fn sanitized_reductions_match_the_plain_answer_on_finite_input() {
        approx::assert_relative_eq!(
            sanitized_mean([1.0_f32, 2.0, 3.0, 4.0]),
            2.5,
            epsilon = 1e-6
        );
        approx::assert_relative_eq!(
            sanitized_sum([1.0_f32, 2.0, 3.0, 4.0]),
            10.0,
            epsilon = 1e-12
        );
        // Single member: the mean is that member.
        approx::assert_relative_eq!(sanitized_mean([-7.25_f32]), -7.25, epsilon = 1e-6);
    }

    /// Regression test for the load-bearing `BatchFitnessFn` invariant
    /// documented in the fitness chapter of the user-book: `evaluate_batch`
    /// returns a `Tensor<B, 1>` of shape `(pop_size,)` with row order
    /// preserved.
    ///
    /// The population is deliberately **non-square** (`pop_size != genome_dim`)
    /// so a row/column transposition — reading the genome axis as the
    /// population axis — cannot hide behind a square shape, and the output
    /// length is asserted against `pop_size` (the rows), not `genome_dim`.
    /// Row `i` is `[i + 1, 0]`, so Sphere fitness is `(i + 1)^2`: a permuted
    /// mapping yields a different, detectable vector.
    #[test]
    fn from_fitness_evaluable_output_is_pop_size_shaped_and_row_aligned() {
        let device = Default::default();
        // 4 individuals, 2 genes each.
        let data = TensorData::new(vec![1.0_f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], [4, 2]);
        let pop = Tensor::<TestBackend, 2>::from_data(data, &device);

        let mut adapter = FromFitnessEvaluable::new(SphereFit, Sphere);
        let fitness = adapter.evaluate_batch(&pop, &device);

        // Shape `(pop_size,)`: rank 1, exactly `pop_size` (4) elements —
        // not `genome_dim` (2).
        assert_eq!(fitness.dims(), [4]);

        let values = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness host-read of a tensor this test just built");
        for (i, &v) in values.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let expected = ((i + 1) * (i + 1)) as f32;
            approx::assert_relative_eq!(v, expected, epsilon = 1e-6);
        }
    }

    /// Same invariant for [`FromLandscape`] — the two adapters carry
    /// independent copies of the row-walking loop, so each pins the shape and
    /// row alignment separately.
    #[test]
    fn from_landscape_output_is_pop_size_shaped_and_row_aligned() {
        let device = Default::default();
        // 4 individuals, 2 genes each — deliberately non-square.
        let data = TensorData::new(vec![1.0_f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], [4, 2]);
        let pop = Tensor::<TestBackend, 2>::from_data(data, &device);

        let mut adapter = FromLandscape::new(SphereLandscape);
        let fitness = adapter.evaluate_batch(&pop, &device);

        assert_eq!(fitness.dims(), [4]);

        let values = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness host-read of a tensor this test just built");
        for (i, &v) in values.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let expected = ((i + 1) * (i + 1)) as f32;
            approx::assert_relative_eq!(v, expected, epsilon = 1e-6);
        }
    }
}
