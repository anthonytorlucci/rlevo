//! Central [`Strategy`] trait and [`EvolutionaryLoop`], which drives one.
//!
//! # The ask / tell contract
//!
//! A [`Strategy`] exposes three methods. One starts a run, and the other
//! two, called in turn, are one generation:
//!
//! 1. [`init`](Strategy::init) — build the initial state (sampling the
//!    population, initializing `$\sigma$`, generation counter, etc).
//! 2. [`ask`](Strategy::ask) — propose the next population as a genome
//!    container.
//! 3. [`tell`](Strategy::tell) — consume that population together with
//!    its fitness and produce the next state plus a metrics snapshot.
//!
//! All three methods take the RNG explicitly so the loop owns all
//! stochasticity; strategies carry *no* internal PRNG state.
//!
//! # Fitness convention
//!
//! The engine is **maximise-native**: the fitness tensor passed to
//! [`tell`](Strategy::tell) is a **canonical** value where *higher is
//! better*, and strategies maximise it directly. The
//! [`StrategyMetrics::best_fitness`] field is the largest value observed in
//! a generation; [`StrategyMetrics::best_fitness_ever`] is a rolling
//! maximum. Strategies are **sense-unaware** — they never see an
//! [`ObjectiveSense`]. Cost
//! objectives (e.g. the benchmark landscapes) are negated into canonical
//! space at exactly one chokepoint, [`EvolutionaryLoop`], which also
//! maps metrics back to the objective's declared sense for reporting.
//!
//! # The loop
//!
//! A strategy holds nothing between calls, so something has to: the state
//! `ask` and `tell` pass back and forth, the RNG, the fitness function and
//! the count of generations run. [`EvolutionaryLoop`] is that something. It
//! pairs a strategy with any [`BatchFitnessFn`] and turns the contract above
//! into three calls:
//!
//! | Call | What it does |
//! |---|---|
//! | [`reset`](EvolutionaryLoop::reset) | seeds the RNG and calls [`init`](Strategy::init) |
//! | [`step`](EvolutionaryLoop::step) | one generation: [`ask`](Strategy::ask), score, [`tell`](Strategy::tell) |
//! | [`is_done`](EvolutionaryLoop::is_done) | whether the generation budget is spent |
//!
//! ```text
//! evo_loop.reset();
//! while !evo_loop.is_done() {
//!     evo_loop.step();
//! }
//! ```
//!
//! It also implements
//! [`GenerationProbe`](rlevo_core::evaluation::GenerationProbe), so the
//! evaluation harness can drive it as a generation loop — not as an
//! environment, which it is not (ADR 0076). [`crate::probe`] sets the two
//! ways of driving a loop side by side.

use std::fmt::Debug;
use std::marker::PhantomData;

use burn::tensor::{Tensor, backend::Backend};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::objective::ObjectiveSense;

use crate::fitness::BatchFitnessFn;
use crate::observer::{PopulationSnapshot, SharedPopulationObserver};

/// Central evolutionary-strategy abstraction.
///
/// The trait is intentionally pure — [`ask`](Self::ask) and
/// [`tell`](Self::tell) return a new `State` rather than mutating
/// through `&mut self`. That keeps strategies free of interior
/// mutability (so many instances can run in parallel without locks) and
/// makes [`Clone`]-based checkpointing straightforward.
///
/// # Example
///
/// The example below uses [`GeneticAlgorithm`] as a concrete strategy and
/// drives one ask/tell cycle by hand. Concrete strategies expose their state
/// fields directly; generic code over `S: Strategy<B>` must access state
/// only through [`Strategy::best`] and the tuple returns of `ask`/`tell`.
///
/// ```no_run
/// use burn::backend::Flex;
/// use burn::tensor::TensorData;
/// use rlevo_evolution::Strategy;
/// use rlevo_evolution::algorithms::ga::{GaConfig, GeneticAlgorithm};
/// use rand::{rngs::StdRng, SeedableRng};
///
/// let device = Default::default();
/// let strategy = GeneticAlgorithm::<Flex>::new();
/// let params = GaConfig::default_for(64, 10);
/// let mut rng = StdRng::seed_from_u64(0);
/// let state = strategy.init(&params, &mut rng, &device);
/// // state.population is a GaState field; dims() is (pop_size, genome_dim).
/// assert_eq!(state.population.dims(), [64, 10]);
/// ```
///
/// [`GeneticAlgorithm`]: crate::algorithms::ga::GeneticAlgorithm
///
/// # Type Parameters
///
/// - `B`: Burn backend.
///
/// # What the loop that drives it does
///
/// [`EvolutionaryLoop`] calls [`init`](Self::init) once per
/// [`reset`](EvolutionaryLoop::reset), then [`ask`](Self::ask) and
/// [`tell`](Self::tell) once per generation, all with the same `rng`. Every
/// draw a strategy makes comes from that `rng` and none from the backend's
/// own generator (`docs/rules.md`, Host-RNG seeding convention). `tell`
/// receives the population and the state that the matching `ask` returned,
/// with a fitness tensor of shape `(pop_size,)` that is already canonical
/// (higher is better) and already sanitised.
///
/// # Associated Types
///
/// - `Params`: Static configuration for a run (population size, `$\sigma$`, F,
///   CR, …). Adaptive algorithms mutate their adaptive quantities inside
///   `State`, not `Params`. [`EvolutionaryLoop::new`] accepts a strategy only
///   if its `Params` implements
///   [`Validate`], and checks them there.
/// - `State`: Generation-to-generation state (current population, `$\sigma$`,
///   best-so-far, RNG-free sub-statistics). Must be clonable so the
///   loop can snapshot before a risky step if needed.
/// - `Genome`: Genome container produced by `ask` and consumed by
///   `tell`. Typically a `Tensor<B, 2>` for real-valued strategies or a
///   `Tensor<B, 2, Int>` for binary/integer kinds.
pub trait Strategy<B: Backend>: Send + Sync {
    /// Static parameters for a run.
    type Params: Clone + Debug + Send + Sync;

    /// Generation-to-generation state.
    type State: Clone + Debug + Send;

    /// Genome container produced by [`ask`](Self::ask).
    type Genome: Clone + Send;

    /// Build the initial state.
    ///
    /// Samples the initial population, primes adaptive quantities, and
    /// sets the generation counter to zero.
    fn init(
        &self,
        params: &Self::Params,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::State;

    /// Propose the next population.
    ///
    /// Takes the current `state` and returns the genome to evaluate
    /// together with an updated state. The returned state typically
    /// carries pre-computed bookkeeping (e.g. the parent indices a
    /// tournament-based GA sampled) so [`tell`](Self::tell) can reuse
    /// them without re-sampling.
    fn ask(
        &self,
        params: &Self::Params,
        state: &Self::State,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Self::Genome, Self::State);

    /// Consume fitness values and produce the next state.
    ///
    /// `fitness` has shape `(pop_size,)` on the same device as the
    /// population. Strategies pull it to host only if they need to —
    /// e.g. for tournament index lookups.
    ///
    /// # Invariants
    ///
    /// When driven by [`EvolutionaryLoop`], the `fitness` tensor is
    /// **canonical (maximise) and sanitized** (ADR 0034): every element is finite
    /// or `f32::NEG_INFINITY` — no `NaN`, no `$+\infty$`.
    ///
    /// An implementation must not rely on that alone. The guarantee is the
    /// loop's, and `tell` can be called without the loop: by a unit test, or
    /// by a driver that calls `ask` and `tell` itself. So a `tell` that pulls
    /// `fitness` to host sanitizes it **once at the pull**: `NaN` becomes
    /// `f32::NEG_INFINITY` and `$+\infty$` becomes `f32::MAX`. Every ordering,
    /// argmax and champion-write site downstream *in that function* may then
    /// assume it (`rules.md` section 3, the correctness floor; ADR 0034
    /// Decision 3). On the loop's path that sanitize changes nothing. On a
    /// bypass path it is what keeps a `NaN` from being crowned. The rule is
    /// one function, [`sanitize_fitness`](crate::fitness::sanitize_fitness),
    /// and a strategy written outside this crate calls the same one:
    /// `fitness.iter().map(|&f| sanitize_fitness(f))` on the `Vec<f32>` the
    /// pull produced.
    ///
    /// Every shipping strategy in [`algorithms::metaheuristic`] does this
    /// internally, so a direct `ask`/`tell` caller of that family need not
    /// pre-sanitize. Coverage stops at the tensor argument, though: a `NaN` written
    /// straight into a state's fitness cache via the `pub` `*State::try_new`
    /// constructors (or `PsoState`'s `pub` fields) bypasses `tell` entirely, is
    /// never re-sanitized, and still freezes that slot for the run.
    ///
    /// [`algorithms::metaheuristic`]: crate::algorithms::metaheuristic
    fn tell(
        &self,
        params: &Self::Params,
        population: Self::Genome,
        fitness: Tensor<B, 1>,
        state: Self::State,
        rng: &mut dyn Rng,
    ) -> (Self::State, StrategyMetrics);

    /// Best-so-far accessor.
    ///
    /// Returns `None` before the first [`tell`](Self::tell) call.
    /// The tuple is `(genome, fitness)` where `fitness` is the **canonical**
    /// (maximise-convention) scalar — the largest value seen across all
    /// completed generations. The loop maps it back to the objective's
    /// declared sense before surfacing it to callers.
    fn best(&self, state: &Self::State) -> Option<(Self::Genome, f32)>;
}

/// Per-generation summary reported by [`Strategy::tell`].
///
/// All statistics refer to the generation that just finished evaluating.
/// These values are in **canonical (maximise) space**: *higher is better*.
/// Strategies are sense-unaware, so the metrics they emit are always
/// canonical. [`EvolutionaryLoop::latest_metrics`] maps them back to the
/// objective's declared sense before surfacing them to callers, so a
/// `Minimize` landscape reads as its natural cost (Sphere → 0).
///
/// When printed in a benchmark showcase (e.g. `ackley_showcase`), the
/// two most informative fields are:
///
/// - **`best_fitness_ever`** — the best (canonical: largest) fitness seen
///   across *all* generations so far. This is a rolling maximum that tells
///   you how close the best individual ever found came to the optimum.
/// - **`mean_fitness`** — the arithmetic mean of the current generation's
///   per-individual fitness vector. This tells you the average quality of
///   the population in that generation.
///
/// A large gap between `best_fitness_ever` and `mean_fitness` in the final
/// generation usually indicates premature convergence: a few elite
/// individuals found a good basin while the rest of the population is still
/// scattered. A small gap suggests the whole population has settled near the
/// same optimum.
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Number of generations completed, this one included: 1 after the
    /// first `tell`.
    generation: usize,
    /// Number of individuals evaluated in this generation.
    population_size: usize,
    /// Best (canonical: largest) fitness observed in this generation.
    best_fitness: f32,
    /// Mean fitness across this generation's population.
    ///
    /// This is the arithmetic mean of the per-individual fitness vector
    /// for the generation that just finished. In a showcase table printed
    /// after a run, this value reflects the *final* generation's average
    /// quality. See the struct-level docs for how to interpret the gap
    /// between this field and [`Self::best_fitness_ever`].
    mean_fitness: f32,
    /// Worst (canonical: smallest) fitness observed in this generation.
    worst_fitness: f32,
    /// Best fitness seen across *all* generations to date.
    ///
    /// This is a rolling maximum (`previous_best.max(current_generation_best)`)
    /// in canonical space. When mapped back to the objective's sense and
    /// printed in a benchmark showcase, it represents the best solution
    /// quality found during the entire run. For landscapes whose global
    /// optimum is known (e.g. Ackley → 0), the loop-reported value tells
    /// you how close the algorithm got to the theoretical optimum.
    best_fitness_ever: f32,
    /// Number of individuals whose sanitized fitness was non-finite (`$-\infty$`) in
    /// this generation — i.e. members that evaluated to `NaN` (or a genuine
    /// worst-sentinel `$-\infty$`) and were therefore **excluded from
    /// [`mean_fitness`](Self::mean_fitness)** (ADR 0034). Zero on a healthy run;
    /// a non-zero value flags a population carrying broken individuals without
    /// letting them blank the mean to `$-\infty$`.
    broken_count: usize,
}

impl StrategyMetrics {
    /// Computes population statistics from a host-side fitness slice.
    ///
    /// Each value is passed through the crate's fitness-hygiene primitive
    /// `sanitize_fitness` before folding, so
    /// `$\text{NaN} \to -\infty$` and `$+\infty \to \texttt{f32::MAX}$` (the
    /// maximise convention, ADR 0023/0034)
    /// *consistently* across every statistic — `best`/`worst` can no longer
    /// silently drop a `NaN` (comparisons against `NaN` are false) while the sum
    /// propagates it.
    ///
    /// `mean_fitness` is computed **over the finite members only**: a sanitized
    /// `$-\infty$` member (a `NaN` evaluation, or a genuine worst-sentinel) is excluded
    /// from the average and counted in [`broken_count`](Self::broken_count)
    /// instead (ADR 0034). This keeps a single broken individual from blanking
    /// the whole mean to `$-\infty$` while still surfacing that the population is
    /// unhealthy. `$+\infty \to \texttt{f32::MAX}$` members are finite and *are* included. If
    /// *every* member is broken, `$\text{mean\_fitness} = -\infty$` (degenerate but
    /// well-defined).
    ///
    /// The mean is **accumulated in `f64`** and narrowed to `f32` once, after the
    /// division — the crate's `sanitized_mean` primitive (ADR 0069 §Decision 1).
    /// The clamp alone does not make the mean safe: `f32::MAX` is finite and
    /// therefore admitted into the sum, but *two* clamped members saturate an
    /// `f32` accumulator (`f32::MAX + f32::MAX == f32::INFINITY`), which would
    /// report `$\text{mean\_fitness} = +\infty$` for a population of optimal individuals. `f64`
    /// carries the widened sum (`$\approx 6.8 \times 10^{38}$` per pair, far
    /// below `f64::MAX`), so it
    /// is the accumulator width — not the clamp — that delivers ADR 0034's
    /// "cannot blow a mean up" guarantee. Averaging in `f64` also removes the ~1
    /// ULP-per-addition drift an `f32` sum accrues over a large population.
    ///
    /// # Panics
    ///
    /// Panics if `fitnesses` is empty. Callers hold a non-empty population by
    /// construction — `pop_size` is validated non-zero at the loop
    /// constructor (ADR 0026).
    #[must_use]
    pub fn from_host_fitness(generation: usize, fitnesses: &[f32], best_fitness_ever: f32) -> Self {
        assert!(!fitnesses.is_empty(), "fitness slice must be non-empty");
        let population_size = fitnesses.len();
        // Canonical (maximise) space: best is the largest value, worst the smallest, best-ever a
        // rolling maximum. Each value is sanitized up front so all statistics agree on the
        // crate-wide convention. The mean is taken over finite members only; non-finite
        // (`$-\infty$`) members are counted as broken rather than dragging the mean to `$-\infty$`.
        //
        // Extrema and the broken tally are order statistics, not reductions, so
        // they stay `f32` (ADR 0069 §Decision 1 excludes ordering explicitly).
        let mut best = f32::NEG_INFINITY;
        let mut worst = f32::INFINITY;
        let mut broken_count = 0_usize;
        for &f in fitnesses {
            let f = crate::fitness::sanitize_fitness(f);
            if f > best {
                best = f;
            }
            if f < worst {
                worst = f;
            }
            if !f.is_finite() {
                broken_count += 1;
            }
        }
        // The mean *is* a reduction, and it runs through `sanitized_mean`: `f64` accumulator, one
        // narrowing after the division (ADR 0069 §Decision 2). The primitive averages every value
        // it is given, so the mean-over-finite-members semantics is expressed by the `filter` — a
        // sanitized `$-\infty$` is excluded here and counted as broken above, whereas a sanitized
        // `$+\infty$` is `f32::MAX`, is finite, and *is* averaged in.
        //
        // The filter re-sanitizes (`sanitize_fitness` is idempotent and is two
        // branches) rather than materialising the sanitized values, keeping this
        // once-per-generation statistic allocation-free.
        //
        // An all-broken population leaves the filter empty, and `sanitized_mean` documents that
        // case as `$-\infty$` — degenerate but well-defined, and exactly the value this function
        // reported before it adopted the primitive.
        let mean = crate::fitness::sanitized_mean(
            fitnesses
                .iter()
                .copied()
                .filter(|&f| crate::fitness::sanitize_fitness(f).is_finite()),
        );
        Self {
            generation,
            population_size,
            best_fitness: best,
            mean_fitness: mean,
            worst_fitness: worst,
            best_fitness_ever: best_fitness_ever.max(best),
            broken_count,
        }
    }

    /// Number of generations completed, this one included: 1 after the
    /// first `tell`.
    #[must_use]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Number of individuals evaluated in this generation.
    #[must_use]
    pub fn population_size(&self) -> usize {
        self.population_size
    }

    /// Best (canonical: largest) fitness observed in this generation.
    #[must_use]
    pub fn best_fitness(&self) -> f32 {
        self.best_fitness
    }

    /// Mean fitness across this generation's population.
    ///
    /// Averaged over the **finite** members only; broken (`$-\infty$`) members are
    /// excluded and reported by [`broken_count`](Self::broken_count) (ADR 0034).
    #[must_use]
    pub fn mean_fitness(&self) -> f32 {
        self.mean_fitness
    }

    /// Number of non-finite (broken) individuals excluded from
    /// [`mean_fitness`](Self::mean_fitness) this generation (ADR 0034).
    ///
    /// Zero on a healthy run; non-zero flags a population carrying `NaN`/`$-\infty$`
    /// members.
    #[must_use]
    pub fn broken_count(&self) -> usize {
        self.broken_count
    }

    /// Worst (canonical: smallest) fitness observed in this generation.
    #[must_use]
    pub fn worst_fitness(&self) -> f32 {
        self.worst_fitness
    }

    /// Best (canonical: largest) fitness seen across *all* generations to date.
    #[must_use]
    pub fn best_fitness_ever(&self) -> f32 {
        self.best_fitness_ever
    }
}

/// Builds a per-generation [`PopulationSnapshot`] from a host-side fitness
/// vector, or `None` when the vector is empty.
///
/// `fitnesses` is in **natural (user-sense)** space: the best individual is the
/// smallest value for a [`Minimize`](ObjectiveSense::Minimize) objective and the
/// largest for [`Maximize`](ObjectiveSense::Maximize). Returning `None` on an
/// empty vector guards against emitting an out-of-range `best_index` (the fold
/// would otherwise default to `0`, indexing into a zero-length slice).
fn build_population_snapshot(
    generation: u32,
    fitnesses: Vec<f32>,
    sense: ObjectiveSense,
) -> Option<PopulationSnapshot> {
    if fitnesses.is_empty() {
        return None;
    }
    let best_index = fitnesses
        .iter()
        .enumerate()
        .reduce(|best, cur| {
            let better = match sense {
                ObjectiveSense::Minimize => cur.1 < best.1,
                ObjectiveSense::Maximize => cur.1 > best.1,
            };
            if better { cur } else { best }
        })
        .map_or(0, |(i, _)| u32::try_from(i).unwrap_or(0));
    Some(PopulationSnapshot {
        generation,
        fitnesses,
        diversity: None,
        best_index,
        best_genome_digest: None,
        parents_of_best: Vec::new(),
    })
}

/// Runs a [`Strategy`] against a [`BatchFitnessFn`], one generation per call.
///
/// A [`Strategy`] is stateless. It proposes a population
/// ([`ask`](Strategy::ask)) and learns from a scored one
/// ([`tell`](Strategy::tell)), and it holds nothing in between. The loop holds
/// what a run needs and the strategy does not: the fitness function, the
/// current state, a seeded RNG, the device, the count of generations run and
/// the generation budget.
///
/// # Driving it
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_core::fitness::FitnessEvaluable;
/// use rlevo_evolution::algorithms::ga::{GaConfig, GeneticAlgorithm};
/// use rlevo_evolution::fitness::FromFitnessEvaluable;
/// use rlevo_evolution::strategy::EvolutionaryLoop;
///
/// struct Sphere;
/// struct SphereFit;
/// impl FitnessEvaluable for SphereFit {
///     type Individual = Vec<f64>;
///     type Landscape = Sphere;
///     fn evaluate(&self, x: &Self::Individual, _: &Self::Landscape) -> f64 {
///         x.iter().map(|v| v * v).sum()
///     }
/// }
///
/// let device = Default::default();
/// let mut evo_loop = EvolutionaryLoop::<Flex, _, _>::new(
///     GeneticAlgorithm::<Flex>::new(),
///     GaConfig::default_for(32, 5),
///     FromFitnessEvaluable::new(SphereFit, Sphere),
///     0, device, 100,
/// ).expect("valid params");
///
/// evo_loop.reset();
/// while !evo_loop.is_done() {
///     evo_loop.step();
/// }
/// let metrics = evo_loop.latest_metrics().expect("a generation has run");
/// println!("best cost found: {}", metrics.best_fitness_ever());
/// ```
///
/// - [`reset`](Self::reset) starts a run. A `step` before it panics.
/// - [`step`](Self::step) runs one generation and returns nothing.
/// - [`is_done`](Self::is_done) says whether the generation budget is spent.
/// - [`latest_metrics`](Self::latest_metrics) reads the summary of the last
///   generation, and [`best`](Self::best) the best genome found so far.
///   `latest_metrics` is `None` until a generation has run, which with a
///   budget of zero is never.
///
/// To stop on a condition of your own, read
/// [`latest_metrics`](Self::latest_metrics) after a `step` and leave the loop.
/// The loop itself stops on nothing but the budget.
///
/// # Why it is shaped this way
///
/// ADR 0085 records these decisions; the reasons in short:
///
/// **`step` returns nothing, and `is_done` is a separate call.** Advancing
/// the run and checking whether it is over are two calls on purpose. A `step`
/// that returned a flag would be written `while !evo_loop.step() {}`, and the
/// reader would have to know which way the flag points. With the check
/// separate, the driving loop reads as what it does, and it checks *before*
/// it runs: a budget of zero runs no generation. `is_done` is also the
/// question an episode loop asks of a
/// [`Snapshot`](rlevo_core::environment::Snapshot), the result of one
/// environment step, so an evolutionary run and an episode are stopped by the
/// same word.
///
/// **`step` does not return the metrics either.** Most callers only run to
/// the budget and read the result once at the end. One who wants each
/// generation's summary reads [`latest_metrics`](Self::latest_metrics), at
/// any time and not only at the moment of the step.
///
/// **`step` does not check the budget.** The budget is where
/// [`is_done`](Self::is_done) turns `true`, and it is the caller's loop that
/// stops there. A `step` that checked would have to do nothing, and then a
/// call that looks like a generation is not one and a caller's mistake hides
/// behind a run that looks complete; or panic, and then a run cannot be
/// carried a little past its budget when that is what the caller means. The
/// cost is that a caller who counts iterations and never asks `is_done` can
/// overrun without an error.
///
/// **`reset` is a call of its own, separate from [`new`](Self::new).** `new`
/// checks the configuration and stores the parts. `reset` builds the first
/// state, and it is also how a run is replayed: the same call starts the
/// first run and every later one, each from the same seed, and it is what
/// [`GenerationProbe::begin`](rlevo_core::evaluation::GenerationProbe::begin)
/// forwards to. The seed is fixed at `new`, so a different seed means a new
/// loop, not another `reset`.
///
/// # What one step does
///
/// 1. [`Strategy::ask`] proposes a population.
/// 2. The fitness function scores it, in the objective's own sense: a cost
///    as a cost, a score as a score.
/// 3. The loop reads the fitness function's [`ObjectiveSense`] and negates a
///    `Minimize` objective, so that every strategy maximises.
/// 4. The loop sanitises the result (ADR 0034): `NaN` becomes `$-\infty$`,
///    the worst value, and `$+\infty$` becomes `f32::MAX`.
/// 5. [`Strategy::tell`] consumes the population and that fitness, and
///    returns the next state.
/// 6. The loop maps the generation's metrics back to the objective's sense,
///    stores them, emits a `tracing` event and calls the observer, if one is
///    attached.
///
/// Steps 3 and 4 are the reason to use the loop instead of calling `ask` and
/// `tell` by hand. For a [`Strategy`] they happen here and nowhere else, so a
/// strategy never sees an [`ObjectiveSense`] or a `NaN`, and a caller never
/// negates a cost.
///
/// # Through the evaluation harness
///
/// The loop also implements
/// [`GenerationProbe`](rlevo_core::evaluation::GenerationProbe). That is how
/// the evaluation harness (`rlevo-harness`, which `rlevo` re-exports as
/// `rlevo::harness`) runs it: many seeded trials, a report at the end. You
/// give `Evaluator::run_trials` a closure that builds one loop per trial from
/// the seed the closure is handed, wrapped in a `GenerationTrial`, and you
/// call neither `reset` nor `step`; the harness makes every call.
/// [`crate::probe`] shows the two ways side by side, how the second is built
/// from the first, and what to set so that the harness does not cut a run
/// short.
///
/// # Determinism and parallel execution
///
/// A run is a function of its seed. Every random number a strategy draws
/// comes from the loop's own host RNG (see [`crate::rng`]), and none from the
/// backend's process-global generator, so nothing outside the loop can reach
/// its trajectory.
///
/// - **On `Flex`**, two runs from one seed are bit-identical, and they stay
///   bit-identical when the evaluation harness runs many loops at once on a
///   thread pool. `tests/determinism.rs` pins the first, and
///   `parallel_trials_match_serial_trials` in the umbrella's
///   `tests/generation_trial_seam.rs` pins the second. No thread pinning is
///   needed for reproducible evolution trials.
/// - **On `Wgpu`** the same host draws are used, so a run follows the `Flex`
///   trajectory to within floating-point differences in the GPU's
///   arithmetic. Bit-identical repeats are not promised there: the order of
///   a reduction belongs to the device.
///
/// The guarantee covers the strategy and the loop, not what you plug into
/// them. A fitness function that draws from the backend generator
/// (`Tensor::random`, or a Burn initialiser run per genome) brings back the
/// dependence on thread schedule that the host-RNG convention removes.
pub struct EvolutionaryLoop<B, S, F>
where
    B: Backend,
    S: Strategy<B>,
    F: BatchFitnessFn<B, S::Genome>,
{
    strategy: S,
    params: S::Params,
    fitness_fn: F,
    state: Option<S::State>,
    rng: StdRng,
    seed: u64,
    device: B::Device,
    generation: usize,
    max_generations: usize,
    latest_metrics: Option<StrategyMetrics>,
    observer: Option<SharedPopulationObserver>,
    _backend: PhantomData<B>,
}

impl<B, S, F> Debug for EvolutionaryLoop<B, S, F>
where
    B: Backend,
    S: Strategy<B>,
    F: BatchFitnessFn<B, S::Genome>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvolutionaryLoop")
            .field("seed", &self.seed)
            .field("generation", &self.generation)
            .field("max_generations", &self.max_generations)
            .field("latest_metrics", &self.latest_metrics)
            .finish_non_exhaustive()
    }
}

impl<B, S, F> EvolutionaryLoop<B, S, F>
where
    B: Backend,
    S: Strategy<B>,
    F: BatchFitnessFn<B, S::Genome>,
{
    /// Build a new loop from its parts.
    ///
    /// `seed` is this loop's seed: it seeds the run's RNG and is fixed for the
    /// life of the loop; [`reset`](Self::reset) re-seeds from it. Under the
    /// evaluation harness it is a seed the harness derived for one trial and
    /// handed to the closure that builds the loop, not the suite-wide
    /// `EvaluatorConfig::base_seed` those are derived from. `device` is where the
    /// population tensors are built. `max_generations` is the generation
    /// budget, the point at which [`is_done`](Self::is_done) turns `true`.
    ///
    /// The caller-supplied `params` are validated up front — this is the
    /// loop consumption chokepoint (ADR 0026), so an invalid configuration
    /// is rejected here rather than surfacing as a panic deep inside a
    /// strategy's tensor code.
    ///
    /// The loop is lazily initialized — the first [`reset`](Self::reset)
    /// call materializes the initial state on the supplied device.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] when `params` fails [`Validate::validate`],
    /// naming the offending field and violated invariant.
    pub fn new(
        strategy: S,
        params: S::Params,
        fitness_fn: F,
        seed: u64,
        device: B::Device,
        max_generations: usize,
    ) -> Result<Self, ConfigError>
    where
        S::Params: Validate,
    {
        params.validate()?;
        Ok(Self {
            strategy,
            params,
            fitness_fn,
            state: None,
            rng: StdRng::seed_from_u64(seed),
            seed,
            device,
            generation: 0,
            max_generations,
            latest_metrics: None,
            observer: None,
            _backend: PhantomData,
        })
    }

    /// Attach a per-generation [`PopulationObserver`].
    ///
    /// The observer is called once per [`step`](Self::step) call, after the
    /// canonical `tracing::info!("evolution generation", …)` event. It
    /// receives a [`PopulationSnapshot`]
    /// carrying the full per-individual fitness vector for the completed
    /// generation. The intended consumer is a benchmark-tier recording sink
    /// that persists population-level data alongside the scalar metric stream.
    ///
    /// Attaching an observer adds one device→host transfer of the fitness
    /// tensor per generation; runs without an observer pay nothing.
    ///
    /// [`PopulationObserver`]: crate::observer::PopulationObserver
    #[must_use]
    pub fn with_observer(mut self, observer: SharedPopulationObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Snapshot of the most recent generation's metrics, if any.
    #[must_use]
    pub fn latest_metrics(&self) -> Option<&StrategyMetrics> {
        self.latest_metrics.as_ref()
    }

    /// Generation counter — number of completed `tell` calls.
    #[must_use]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// The configured generation budget.
    ///
    /// The loop owns this number. A driver should ask
    /// [`is_done`](Self::is_done) rather than keep a second copy of the budget
    /// and count against it.
    #[must_use]
    pub const fn max_generations(&self) -> usize {
        self.max_generations
    }

    /// Whether the generation budget is spent.
    ///
    /// `true` once [`generation`](Self::generation) has reached
    /// [`max_generations`](Self::max_generations). This is the condition a
    /// driving loop checks before each [`step`](Self::step):
    ///
    /// ```text
    /// while !evo_loop.is_done() {
    ///     evo_loop.step();
    /// }
    /// ```
    ///
    /// It reports the budget and nothing else. It does not say whether the
    /// search has converged; for that, read
    /// [`latest_metrics`](Self::latest_metrics). It does not stop
    /// [`step`](Self::step) either, which runs a generation whenever it is
    /// called.
    ///
    /// [`GenerationProbe::advance`](rlevo_core::evaluation::GenerationProbe::advance)
    /// is this check followed by one `step`, which is why the evaluation
    /// harness never runs a generation past the budget.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.generation >= self.max_generations
    }

    /// Borrow the current strategy state if it exists.
    #[must_use]
    pub fn state(&self) -> Option<&S::State> {
        self.state.as_ref()
    }

    /// The best genome found so far in the run, and its fitness: not the
    /// best of the last generation. `None` until a generation has run.
    ///
    /// Forwards to [`Strategy::best`]. For the tensor-genome strategies the
    /// genome is a one-row tensor, `(1, genome_dim)`; read it to host with
    /// `into_data().into_vec::<f32>()`.
    ///
    /// The strategy tracks the best genome in **canonical (maximise)** space;
    /// the returned fitness is mapped back to the objective's declared sense so
    /// a `Minimize` landscape reads as its natural cost.
    pub fn best(&self) -> Option<(S::Genome, f32)> {
        let sense = self.fitness_fn.sense();
        self.state
            .as_ref()
            .and_then(|s| self.strategy.best(s))
            .map(|(genome, canonical)| (genome, sense.from_canonical(canonical)))
    }

    /// Start a run: seed the RNG, zero the generation count and build the
    /// strategy's first state with [`Strategy::init`].
    ///
    /// Call it once before the first [`step`](Self::step). Calling it again
    /// starts over: the RNG is re-seeded from the loop's seed, so two
    /// `reset`-to-budget runs of the same loop are byte-identical replays.
    ///
    /// Infallible: the configuration was validated in [`new`](Self::new), and
    /// nothing else here can fail.
    /// [`GenerationProbe::begin`](rlevo_core::evaluation::GenerationProbe::begin)
    /// forwards to it.
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.seed);
        self.generation = 0;
        self.latest_metrics = None;
        self.state = Some(
            self.strategy
                .init(&self.params, &mut self.rng, &self.device),
        );
    }

    /// Run one generation: ask, evaluate, tell.
    ///
    /// Returns nothing. Ask [`is_done`](Self::is_done) whether the budget is
    /// spent, and read the generation's summary from
    /// [`latest_metrics`](Self::latest_metrics). The type's documentation
    /// lists what a generation consists of, and why this method reports no
    /// status of its own.
    ///
    /// `step` does not consult the budget. Called when `is_done` is already
    /// `true`, it runs one more generation: a check that did nothing would
    /// hide a caller's mistake behind a run that looks finished, and one that
    /// panicked would make a deliberate continuation an error (ADR 0085). A
    /// caller who counts iterations and never asks `is_done` therefore
    /// overruns without an error.
    ///
    /// # Panics
    ///
    /// Panics if [`reset`](Self::reset) has not been called first. Also panics
    /// if an observer is attached and the natural-fitness tensor cannot be read
    /// back to host as `f32` (a device→host transfer failure).
    pub fn step(&mut self) {
        let state = self
            .state
            .take()
            .expect("EvolutionaryLoop::reset must be called before step");
        let (population, state) =
            self.strategy
                .ask(&self.params, &state, &mut self.rng, &self.device);
        // The fitness function reports NATURAL values; the loop is the sole
        // canonicaliser. `sense` is the single source of truth (read off the
        // fitness fn, so the ctor and the adapter can never disagree).
        let sense = self.fitness_fn.sense();
        let fitness_natural = self.fitness_fn.evaluate_batch(&population, &self.device);
        // Mirror the NATURAL fitness tensor to host only if someone's actually
        // listening — the device→host transfer is the expensive part. The
        // observer records natural (user-sense) per-individual fitness.
        let snapshot_fitness: Option<Vec<f32>> = self.observer.as_ref().map(|_| {
            fitness_natural
                .clone()
                .into_data()
                .into_vec::<f32>()
                .expect("fitness tensor must be readable as f32")
        });
        // Canonicalise into the engine's maximise-native space: a `Minimize`
        // objective is negated (one device op), a `Maximize` one passes through.
        let fitness_canon = match sense {
            ObjectiveSense::Maximize => fitness_natural,
            ObjectiveSense::Minimize => fitness_natural.neg(),
        };
        // Fitness-hygiene chokepoint (ADR 0034). Sanitize in CANONICAL (maximise) space —
        // `$\text{NaN} \to -\infty$` (worst), `$+\infty \to \text{f32::MAX}$` — so no
        // `Strategy::tell` impl can be poisoned by a non-finite fitness and every downstream best/
        // leader/metric is finite-or-`$-\infty$`. This runs *after* the `sense` negation on
        // purpose: "NaN = worst" is defined in maximise space, so sanitizing the natural tensor
        // before `neg()` would flip a `NaN` cost to `$+\infty$` (canonical *best*) under
        // `Minimize`.
        let fitness_canon = crate::fitness::sanitize_fitness_tensor(fitness_canon);
        let (new_state, metrics_canon) = self.strategy.tell(
            &self.params,
            population,
            fitness_canon,
            state,
            &mut self.rng,
        );
        self.state = Some(new_state);
        self.generation += 1;
        // Map the canonical metrics back into the objective's declared sense so
        // every surfaced value (tracing, `latest_metrics`, records) reads in
        // user space — a `Minimize` landscape's `best_fitness` is its natural
        // cost (Sphere → 0).
        let metrics = StrategyMetrics {
            generation: metrics_canon.generation,
            population_size: metrics_canon.population_size,
            best_fitness: sense.from_canonical(metrics_canon.best_fitness),
            mean_fitness: sense.from_canonical(metrics_canon.mean_fitness),
            worst_fitness: sense.from_canonical(metrics_canon.worst_fitness),
            best_fitness_ever: sense.from_canonical(metrics_canon.best_fitness_ever),
            // A count, not a value — sense-invariant, carried through verbatim.
            broken_count: metrics_canon.broken_count,
        };
        // Structured per-generation event. Picked up by the
        // canonical-metric registry in
        // `rlevo-harness::tui::log_layer::CANONICAL_METRICS` so the
        // live TUI's fitness sparkline lights up without coupling this
        // crate to the dashboard. Field names match the registry
        // verbatim; renaming any of them requires a paired update on
        // the benchmarks side.
        tracing::info!(
            generation = metrics.generation,
            population_size = metrics.population_size,
            best_fitness = f64::from(metrics.best_fitness),
            mean_fitness = f64::from(metrics.mean_fitness),
            worst_fitness = f64::from(metrics.worst_fitness),
            best_fitness_ever = f64::from(metrics.best_fitness_ever),
            broken_count = metrics.broken_count,
            "evolution generation",
        );
        if let (Some(observer), Some(fitnesses)) = (self.observer.as_ref(), snapshot_fitness) {
            let generation = u32::try_from(metrics.generation).unwrap_or(u32::MAX);
            match build_population_snapshot(generation, fitnesses, sense) {
                Some(snapshot) => {
                    // Isolate the observer: a panicking third-party sink drops
                    // this snapshot but must not abort an otherwise-healthy
                    // optimization run. `SharedPopulationObserver` is backed by a
                    // `parking_lot::Mutex` (no poisoning), so the guard drops
                    // during unwind and the next generation re-locks cleanly.
                    let dispatched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        observer.lock().on_population(snapshot);
                    }));
                    if dispatched.is_err() {
                        tracing::warn!(
                            generation,
                            "population observer panicked; dropping snapshot and continuing",
                        );
                    }
                }
                None => {
                    // An empty fitness vector means the device→host transfer at
                    // `snapshot_fitness` yielded nothing (a masked conversion
                    // failure). Surface it rather than emitting an out-of-range
                    // `best_index` into a zero-length vector.
                    tracing::warn!(
                        generation,
                        "empty population fitness vector; skipping observer snapshot \
                         (device→host transfer likely failed)",
                    );
                }
            }
        }
        self.latest_metrics = Some(metrics);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::TensorData;
    type TestBackend = Flex;

    /// Trivial strategy for unit-testing the loop plumbing: it
    /// ignores `ask`/`tell` semantics and always reports the same best
    /// fitness. Nothing here exercises real evolutionary dynamics.
    #[derive(Debug, Clone, Copy)]
    struct Constant;

    #[derive(Debug, Clone)]
    struct Params {
        pop_size: usize,
        dim: usize,
    }

    impl Validate for Params {
        fn validate(&self) -> Result<(), ConfigError> {
            rlevo_core::config::nonzero("Params", "pop_size", self.pop_size)?;
            rlevo_core::config::nonzero("Params", "dim", self.dim)?;
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    struct State {
        generation: usize,
        best: f32,
    }

    impl Strategy<TestBackend> for Constant {
        type Params = Params;
        type State = State;
        type Genome = Tensor<TestBackend, 2>;

        fn init(
            &self,
            params: &Params,
            _: &mut dyn Rng,
            device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
        ) -> State {
            let _ = device;
            let _ = params;
            State {
                generation: 0,
                best: f32::NEG_INFINITY,
            }
        }

        fn ask(
            &self,
            params: &Params,
            state: &State,
            _: &mut dyn Rng,
            device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
        ) -> (Tensor<TestBackend, 2>, State) {
            let data = TensorData::new(
                vec![0.0f32; params.pop_size * params.dim],
                [params.pop_size, params.dim],
            );
            let pop = Tensor::<TestBackend, 2>::from_data(data, device);
            (pop, state.clone())
        }

        fn tell(
            &self,
            _: &Params,
            _: Tensor<TestBackend, 2>,
            fitness: Tensor<TestBackend, 1>,
            mut state: State,
            _: &mut dyn Rng,
        ) -> (State, StrategyMetrics) {
            let values = fitness
                .into_data()
                .into_vec::<f32>()
                .expect("fitness host-read of a tensor this test just built");
            state.generation += 1;
            let metrics = StrategyMetrics::from_host_fitness(state.generation, &values, state.best);
            state.best = metrics.best_fitness_ever();
            (state, metrics)
        }

        fn best(&self, _state: &State) -> Option<(Tensor<TestBackend, 2>, f32)> {
            None
        }
    }

    /// Constant fitness = 42 regardless of input.
    struct FortyTwo;
    impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2>> for FortyTwo {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2>,
            device: &<B as burn::tensor::backend::BackendTypes>::Device,
        ) -> Tensor<B, 1> {
            let n = population.dims()[0];
            let data = TensorData::new(vec![42.0f32; n], [n]);
            Tensor::<B, 1>::from_data(data, device)
        }

        fn sense(&self) -> ObjectiveSense {
            // Treated as a cost so the loop reports natural 42 while `tell` sees the canonical
            // `$-42$`.
            ObjectiveSense::Minimize
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn loop_runs_one_generation() {
        let device = Default::default();
        let strategy = Constant;
        let params = Params {
            pop_size: 4,
            dim: 3,
        };
        let mut evo_loop =
            EvolutionaryLoop::<TestBackend, _, _>::new(strategy, params, FortyTwo, 1, device, 5)
                .expect("valid params");
        evo_loop.reset();
        evo_loop.step();
        assert!(!evo_loop.is_done());
        assert_eq!(evo_loop.generation(), 1);
        // What `tell` saw is canonical: the cost 42 arrives negated.
        assert_eq!(evo_loop.state().expect("state after step").best, -42.0);
        let m = evo_loop.latest_metrics().unwrap();
        assert_eq!(m.generation, 1);
        assert_eq!(m.population_size, 4);
        approx::assert_relative_eq!(m.best_fitness, 42.0, epsilon = 1e-6);
    }

    #[test]
    fn loop_reports_done_after_budget() {
        let device = Default::default();
        let mut evo_loop = EvolutionaryLoop::<TestBackend, _, _>::new(
            Constant,
            Params {
                pop_size: 2,
                dim: 2,
            },
            FortyTwo,
            1,
            device,
            2,
        )
        .expect("valid params");
        evo_loop.reset();
        assert!(!evo_loop.is_done(), "a fresh run with a budget is not done");
        evo_loop.step();
        assert!(!evo_loop.is_done());
        evo_loop.step();
        assert!(evo_loop.is_done());
    }

    /// `is_done` is asked before a step, so for a budget of zero it must be
    /// `true` straight after `reset`: a driving loop then runs no generation.
    #[test]
    fn zero_budget_is_done_before_any_step() {
        let device = Default::default();
        let mut evo_loop = EvolutionaryLoop::<TestBackend, _, _>::new(
            Constant,
            Params {
                pop_size: 2,
                dim: 2,
            },
            FortyTwo,
            1,
            device,
            0,
        )
        .expect("valid params");
        evo_loop.reset();
        assert!(evo_loop.is_done());
        assert!(evo_loop.latest_metrics().is_none());
    }

    /// `step` does not consult the budget: past it, a call still runs a
    /// generation, and `is_done` stays `true`. Stopping at the budget is the
    /// driving loop's job, which is why it asks `is_done` first.
    #[test]
    fn step_past_the_budget_runs_another_generation() {
        let device = Default::default();
        let mut evo_loop = EvolutionaryLoop::<TestBackend, _, _>::new(
            Constant,
            Params {
                pop_size: 2,
                dim: 2,
            },
            FortyTwo,
            1,
            device,
            1,
        )
        .expect("valid params");
        evo_loop.reset();
        evo_loop.step();
        assert!(evo_loop.is_done());
        evo_loop.step();
        assert_eq!(evo_loop.generation(), 2, "the second call ran a generation");
        assert!(evo_loop.is_done());
    }

    #[test]
    fn from_host_fitness_computes_stats() {
        let m = StrategyMetrics::from_host_fitness(5, &[3.0, 1.0, 5.0, 2.0], 4.0);
        // Read through the public accessors (fields are private).
        assert_eq!(m.generation(), 5);
        assert_eq!(m.population_size(), 4);
        // Canonical maximise: best is the largest, worst the smallest.
        approx::assert_relative_eq!(m.best_fitness(), 5.0, epsilon = 1e-6);
        approx::assert_relative_eq!(m.worst_fitness(), 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(m.mean_fitness(), 2.75, epsilon = 1e-6);
        // best_fitness_ever = max(prior=4.0, current=5.0)
        approx::assert_relative_eq!(m.best_fitness_ever(), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn from_host_fitness_sanitizes_nan() {
        // A NaN is sanitized to `$-\infty$` (worst under maximise): it never becomes best, and it
        // drags `worst` to `$-\infty$`. Under ADR 0034 it is *excluded* from the mean (counted as
        // broken) rather than blanking the mean to `$-\infty$`: the mean is over the finite members
        // {1, 3, 2} = 2.0, with broken_count == 1.
        let m = StrategyMetrics::from_host_fitness(0, &[1.0, f32::NAN, 3.0, 2.0], 0.0);
        approx::assert_relative_eq!(m.best_fitness(), 3.0, epsilon = 1e-6);
        assert!(m.worst_fitness().is_infinite() && m.worst_fitness().is_sign_negative());
        approx::assert_relative_eq!(m.mean_fitness(), 2.0, epsilon = 1e-6);
        assert_eq!(m.broken_count(), 1);
        approx::assert_relative_eq!(m.best_fitness_ever(), 3.0, epsilon = 1e-6);
    }

    #[test]
    fn from_host_fitness_pos_inf_ranks_top_but_mean_stays_finite() {
        // `$+\infty$` → f32::MAX (ADR 0034): it stays best/finite and is *included* in the mean (no
        // `$-\infty$`/broken). The clamp is only half of why the mean survives — the other half is
        // the `f64` accumulator, which `from_host_fitness_two_pos_inf_members_keep_mean_finite`
        // pins for the multi-member case this single-member test cannot reach.
        let m = StrategyMetrics::from_host_fitness(0, &[1.0, f32::INFINITY, 3.0], 0.0);
        approx::assert_relative_eq!(m.best_fitness(), f32::MAX);
        assert_eq!(m.broken_count(), 0);
        assert!(m.mean_fitness().is_finite());
    }

    #[test]
    fn from_host_fitness_two_pos_inf_members_keep_mean_finite() {
        // Regression: ADR 0034 maps `$+\infty \to \text{f32::MAX}$` and claims the clamped value
        // "cannot blow a mean up". `f32::MAX` passes `is_finite()`, so it is admitted into the
        // accumulator — and *two* such members saturate an `f32` sum:
        // `f32::MAX + f32::MAX == f32::INFINITY`. The accumulator, not the clamp, is what has to
        // carry the guarantee, so the sum runs in `f64`
        // (`$\text{f32::MAX} + \text{f32::MAX} \approx 6.8 \times 10^{38}$` is nowhere near
        // `f64::MAX`) and is narrowed back to `f32` once, after the division. The mean here is
        // `$(1 + \text{MAX} + \text{MAX} + 3)/4 \approx 1.7 \times 10^{38}$`, comfortably
        // representable in `f32`.
        let m = StrategyMetrics::from_host_fitness(
            0,
            &[1.0, f32::INFINITY, f32::INFINITY, 3.0],
            f32::NEG_INFINITY,
        );
        assert_eq!(
            m.broken_count(),
            0,
            "clamped +∞ members are finite, so none is broken"
        );
        assert!(
            m.mean_fitness().is_finite(),
            "two f32::MAX members must not saturate the mean; got {}",
            m.mean_fitness()
        );
        approx::assert_relative_eq!(m.best_fitness(), f32::MAX);
    }

    #[test]
    fn from_host_fitness_all_broken_yields_neg_inf_mean() {
        // Degenerate but well-defined: every member broken → mean = `$-\infty$`.
        let m = StrategyMetrics::from_host_fitness(0, &[f32::NAN, f32::NAN], 0.0);
        assert_eq!(m.broken_count(), 2);
        assert!(m.mean_fitness().is_infinite() && m.mean_fitness().is_sign_negative());
    }

    /// A misbehaving objective: row 0 → `NaN`, row 1 → `$+\infty$`, the rest finite.
    /// `Maximize` so natural == canonical (no `neg()` obscuring the sanitize).
    struct NonFiniteFitness;
    impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2>> for NonFiniteFitness {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2>,
            device: &<B as burn::tensor::backend::BackendTypes>::Device,
        ) -> Tensor<B, 1> {
            let n = population.dims()[0];
            #[allow(clippy::cast_precision_loss)] // tiny test population indices
            let mut vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
            vals[0] = f32::NAN;
            if n > 1 {
                vals[1] = f32::INFINITY;
            }
            Tensor::<B, 1>::from_data(TensorData::new(vals, [n]), device)
        }
        fn sense(&self) -> ObjectiveSense {
            ObjectiveSense::Maximize
        }
    }

    /// A strategy that **trusts the loop guarantee** (ADR 0034): its `tell`
    /// stores the fitness tensor it receives verbatim, *without* re-sanitizing.
    /// This is the whole point of the chokepoint — a `tell` impl should be able
    /// to build best/leaders directly from `fitness`. `received` lets the test
    /// assert what actually crossed the seam.
    #[derive(Debug, Clone, Copy)]
    struct TrustingStrategy;

    #[derive(Debug, Clone)]
    struct TrustingState {
        generation: usize,
        best: f32,
        received: Vec<f32>,
    }

    impl Strategy<TestBackend> for TrustingStrategy {
        type Params = Params;
        type State = TrustingState;
        type Genome = Tensor<TestBackend, 2>;

        fn init(
            &self,
            _: &Params,
            _: &mut dyn Rng,
            _: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
        ) -> TrustingState {
            TrustingState {
                generation: 0,
                best: f32::NEG_INFINITY,
                received: Vec::new(),
            }
        }

        fn ask(
            &self,
            params: &Params,
            state: &TrustingState,
            _: &mut dyn Rng,
            device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
        ) -> (Tensor<TestBackend, 2>, TrustingState) {
            let data = TensorData::new(
                vec![0.0f32; params.pop_size * params.dim],
                [params.pop_size, params.dim],
            );
            (
                Tensor::<TestBackend, 2>::from_data(data, device),
                state.clone(),
            )
        }

        fn tell(
            &self,
            _: &Params,
            _: Tensor<TestBackend, 2>,
            fitness: Tensor<TestBackend, 1>,
            mut state: TrustingState,
            _: &mut dyn Rng,
        ) -> (TrustingState, StrategyMetrics) {
            // Deliberately NOT sanitized here — the loop must have done it.
            let values: Vec<f32> = fitness
                .into_data()
                .into_vec::<f32>()
                .expect("fitness host-read of a tensor this test just built");
            state.received = values.clone();
            state.generation += 1;
            let metrics: StrategyMetrics =
                StrategyMetrics::from_host_fitness(state.generation, &values, state.best);
            state.best = metrics.best_fitness_ever();
            (state, metrics)
        }

        fn best(&self, _: &TrustingState) -> Option<(Tensor<TestBackend, 2>, f32)> {
            None
        }
    }

    /// End-to-end proof that the `EvolutionaryLoop::step` chokepoint (ADR
    /// 0034) sanitizes a non-finite fitness **before** it reaches a real
    /// `Strategy::tell`. This is the widest-blast-radius line in the design
    /// (it covers every `Strategy` impl), so it gets a direct test rather than
    /// relying on `StrategyMetrics::from_host_fitness`'s own sanitize: the
    /// `TrustingStrategy` captures the raw tensor it was handed, so deleting or
    /// mis-ordering the loop sanitize (relative to `neg()`) fails here.
    #[test]
    fn loop_sanitizes_non_finite_fitness_before_tell() {
        let device = Default::default();
        let mut evo_loop = EvolutionaryLoop::<TestBackend, _, _>::new(
            TrustingStrategy,
            Params {
                pop_size: 4,
                dim: 2,
            },
            NonFiniteFitness,
            7,
            device,
            1,
        )
        .expect("valid params");
        evo_loop.reset();
        evo_loop.step();

        let received = &evo_loop.state().expect("state after step").received;
        assert_eq!(received.len(), 4);
        // The chokepoint guarantee: what `tell` saw is finite-or-`$-\infty$`.
        assert!(
            received.iter().all(|f| !f.is_nan()),
            "the loop must strip NaN before tell; got {received:?}"
        );
        assert!(
            received
                .iter()
                .all(|f| !(f.is_infinite() && f.is_sign_positive())),
            "the loop must clamp +∞ before tell; got {received:?}"
        );
        // Row 0 was NaN → `$-\infty$` (worst); row 1 was `$+\infty$` → f32::MAX (finite best).
        assert!(
            received[0].is_infinite() && received[0].is_sign_negative(),
            "NaN row → −∞"
        );
        approx::assert_relative_eq!(received[1], f32::MAX);

        // Metrics stay honest: best is finite (the f32::MAX row), one broken member.
        let m = evo_loop.latest_metrics().expect("metrics after step");
        assert!(
            m.best_fitness().is_finite(),
            "best must be finite, got {}",
            m.best_fitness()
        );
        assert_eq!(m.broken_count(), 1, "the NaN row is the one broken member");
        assert!(
            m.mean_fitness().is_finite(),
            "mean over finite members stays finite"
        );
    }

    #[test]
    fn build_population_snapshot_empty_returns_none() {
        assert!(build_population_snapshot(0, Vec::new(), ObjectiveSense::Minimize).is_none());
    }

    #[test]
    fn build_population_snapshot_picks_best_for_sense() {
        // Values: [0.3, 0.1, 0.9]. Minimize → best is the smallest (index 1);
        // Maximize → best is the largest (index 2).
        let min = build_population_snapshot(7, vec![0.3, 0.1, 0.9], ObjectiveSense::Minimize)
            .expect("non-empty");
        assert_eq!(min.best_index, 1);
        assert_eq!(min.generation, 7);
        let max = build_population_snapshot(7, vec![0.3, 0.1, 0.9], ObjectiveSense::Maximize)
            .expect("non-empty");
        assert_eq!(max.best_index, 2);
    }

    /// Per-individual fitness = `1.0 / (i + 1)` so the best (smallest)
    /// is always at index `pop_size - 1` — a deterministic shape the
    /// observer test can pin against.
    struct RankedFitness;
    impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2>> for RankedFitness {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2>,
            device: &<B as burn::tensor::backend::BackendTypes>::Device,
        ) -> Tensor<B, 1> {
            let n = population.dims()[0];
            #[allow(clippy::cast_precision_loss)]
            let values: Vec<f32> = (0..n).map(|i| 1.0 / (i as f32 + 1.0)).collect();
            let data = TensorData::new(values, [n]);
            Tensor::<B, 1>::from_data(data, device)
        }

        fn sense(&self) -> ObjectiveSense {
            // Cost: the best (smallest) is the last index, which the observer
            // test pins via the sense-aware `best_index`.
            ObjectiveSense::Minimize
        }
    }

    #[derive(Debug, Default)]
    struct CountingObserver {
        snapshots: Vec<PopulationSnapshot>,
    }

    impl crate::observer::PopulationObserver for CountingObserver {
        fn on_population(&mut self, snapshot: PopulationSnapshot) {
            self.snapshots.push(snapshot);
        }
    }

    #[test]
    fn loop_fires_observer_per_generation() {
        use std::sync::Arc;

        use parking_lot::Mutex;
        let device = Default::default();
        let observer = Arc::new(Mutex::new(CountingObserver::default()));
        let mut evo_loop = EvolutionaryLoop::<TestBackend, _, _>::new(
            Constant,
            Params {
                pop_size: 5,
                dim: 2,
            },
            RankedFitness,
            1,
            device,
            3,
        )
        .expect("valid params")
        .with_observer(observer.clone() as SharedPopulationObserver);
        evo_loop.reset();
        for _ in 0..3 {
            evo_loop.step();
        }
        let guard = observer.lock();
        assert_eq!(guard.snapshots.len(), 3);
        // pop_size = 5, ranked fitness = [1/1, 1/2, 1/3, 1/4, 1/5]; best
        // (smallest) is the last element.
        assert_eq!(guard.snapshots[0].fitnesses.len(), 5);
        assert_eq!(guard.snapshots[0].best_index, 4);
        assert_eq!(guard.snapshots[2].generation, 3);
        // M8.1 leaves these fields empty / None — see observer.rs docs.
        assert!(guard.snapshots[0].diversity.is_none());
        assert!(guard.snapshots[0].best_genome_digest.is_none());
        assert!(
            guard.snapshots[0].parents_of_best.is_empty(),
            "guard.snapshots[0].parents_of_best must be empty, got {:?}",
            guard.snapshots[0].parents_of_best
        );
    }

    /// Observer whose callback always panics — used to prove the loop
    /// isolates a misbehaving sink instead of aborting the run.
    #[derive(Debug, Default)]
    struct PanicObserver;

    impl crate::observer::PopulationObserver for PanicObserver {
        fn on_population(&mut self, _snapshot: PopulationSnapshot) {
            panic!("observer intentionally panics");
        }
    }

    #[test]
    fn loop_survives_panicking_observer() {
        use std::sync::Arc;

        use parking_lot::Mutex;
        let device = Default::default();
        let observer = Arc::new(Mutex::new(PanicObserver));
        let mut evo_loop = EvolutionaryLoop::<TestBackend, _, _>::new(
            Constant,
            Params {
                pop_size: 4,
                dim: 2,
            },
            RankedFitness,
            1,
            device,
            2,
        )
        .expect("valid params")
        .with_observer(observer.clone() as SharedPopulationObserver);
        evo_loop.reset();
        // Each step's observer dispatch panics; the loop must swallow it and
        // keep advancing generations to completion.
        evo_loop.step();
        assert!(!evo_loop.is_done());
        evo_loop.step();
        assert!(evo_loop.is_done());
        assert_eq!(evo_loop.generation(), 2);
    }

    #[test]
    fn loop_without_observer_skips_host_transfer() {
        // Smoke: no observer attached → step() still works, no panic,
        // no transfer cost. Observability is verified above; here we
        // just want the no-observer path to remain functional.
        let device = Default::default();
        let mut evo_loop = EvolutionaryLoop::<TestBackend, _, _>::new(
            Constant,
            Params {
                pop_size: 3,
                dim: 1,
            },
            RankedFitness,
            1,
            device,
            1,
        )
        .expect("valid params");
        evo_loop.reset();
        evo_loop.step();
        assert!(evo_loop.is_done());
    }

    // ------------------------------------------------------------------
    // ADR 0069 §Decision 5 — behavioural finiteness property.
    // ------------------------------------------------------------------

    use proptest::collection::vec as prop_vec;
    // Explicit list, not the glob prelude: the prelude re-exports a `Rng` that
    // would collide with the `rlevo_core`/`rand` `Rng` pulled in by `use super::*`.
    use proptest::prelude::{ProptestConfig, any, prop_assert, prop_assert_eq, proptest};

    /// `$\texttt{f32::MAX} \cdot 2^{-e}$`, built in `f64` so the shift is exact and `e` may reach
    /// past `f32`'s exponent range without an intermediate `$2^e$` overflowing.
    fn scaled_max(e: u32, negative: bool) -> f32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let m = (f64::from(f32::MAX) * 0.5_f64.powi(e as i32)) as f32;
        if negative { -m } else { m }
    }

    proptest! {
        // Pure host arithmetic over a slice of at most 65 values — no backend, no
        // tensors. ADR 0036 §5's "cheap structural" tier.
        #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

        /// **ADR 0069 §Decision 5, property 2.** `from_host_fitness` reports a
        /// **finite** `mean_fitness` whenever every member is finite — including
        /// an all-`f32::MAX` population.
        ///
        /// This is the property that fails with an `f32` accumulator: `f32::MAX`
        /// is finite, so ADR 0034's sanitized `$+\infty$` *joins* the sum rather than
        /// being excluded from it, and `f32::MAX + f32::MAX == f32::INFINITY`: an `f32`
        /// accumulator reports `$\text{mean\_fitness} = +\infty$` for a population of two optimal
        /// individuals. Only §Decision 1's `f64` accumulator width makes the
        /// reduction safe; the clamp bounds a value, not a fold.
        ///
        /// Magnitudes are generated as `$\texttt{f32::MAX} \cdot 2^{-e}$` so the top of the `f32`
        /// range — where the defect lives — is sampled directly rather than being
        /// reached by luck. The all-`f32::MAX` population §Decision 5 names is
        /// asserted **deterministically on every case** (at length `$n + 1 \geq 2$`,
        /// since a single member cannot overflow anything).
        ///
        /// The mean is also bracketed by `[worst, best]`: finiteness alone is
        /// satisfied by any constant, and the bracket is the cheapest assertion
        /// that a *mean* is what was computed. The slack is a relative ULP
        /// allowance for the single narrowing at the end of the reduction, taken
        /// against the `f64` span so `$\text{best} - \text{worst}$` cannot itself overflow.
        #[test]
        fn prop_mean_fitness_is_finite_for_a_finite_population(
            members in prop_vec((0u32..=140, any::<bool>()), 1..=64),
            generation in 0usize..1000,
        ) {
            let values: Vec<f32> = members
                .iter()
                .map(|&(e, negative)| scaled_max(e, negative))
                .collect();
            prop_assert!(
                values.iter().all(|v| v.is_finite()),
                "generator must only produce finite members"
            );

            let m = StrategyMetrics::from_host_fitness(generation, &values, f32::NEG_INFINITY);
            prop_assert!(
                m.mean_fitness().is_finite(),
                "mean of a finite population must be finite, got {} for {:?}",
                m.mean_fitness(),
                values
            );
            prop_assert_eq!(m.broken_count(), 0, "no member is broken");

            let (lo, hi) = (f64::from(m.worst_fitness()), f64::from(m.best_fitness()));
            let slack = (hi - lo).abs() * 1e-6;
            let mean = f64::from(m.mean_fitness());
            prop_assert!(
                mean >= lo - slack && mean <= hi + slack,
                "mean {mean} escaped [{lo}, {hi}]"
            );

            // The named case: every member is the sanitized `$+\infty$` sentinel. Its mean is
            // `f32::MAX`; an `f32` accumulator saturates at `n = 2`.
            let saturated = vec![f32::MAX; values.len() + 1];
            let m = StrategyMetrics::from_host_fitness(generation, &saturated, f32::NEG_INFINITY);
            prop_assert!(
                m.mean_fitness().is_finite(),
                "an all-`f32::MAX` population of {} must keep a finite mean, got {}",
                saturated.len(),
                m.mean_fitness()
            );
            // Exact, not approximate: the mean of `n` identical values is that
            // value bit-for-bit under `f64` accumulation, and asserting it
            // exactly is what rules out a "finite but wrong" mean.
            prop_assert!(
                m.mean_fitness().to_bits() == f32::MAX.to_bits(),
                "and it is exactly `f32::MAX`, got {}",
                m.mean_fitness()
            );
        }
    }
}
