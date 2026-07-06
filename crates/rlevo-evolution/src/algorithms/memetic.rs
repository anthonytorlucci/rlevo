//! Memetic-algorithm strategy adapter.
//!
//! A *memetic algorithm* (MA) interleaves a population-level evolutionary
//! [`Strategy`] with a per-individual [`LocalSearch`] that polishes promising
//! genomes between an inner strategy's `ask` and `tell`. [`MemeticWrapper`]
//! makes that interleaving a *zero-cost-to-adopt* upgrade: wrap any existing
//! `Strategy<B, Genome = Tensor<B, 2>>` together with any
//! [`LocalSearch<B>`](crate::local_search::LocalSearch) and the wrapper itself
//! implements `Strategy<B>`, so it drops straight into
//! [`EvolutionaryHarness`](crate::strategy::EvolutionaryHarness).
//!
//! # Lamarckian / Baldwinian / Partial refinement
//!
//! After the inner strategy proposes a population, the wrapper refines a subset
//! of individuals (the [`CoveragePolicy`]) and then decides — per the
//! [`WritebackPolicy`] — whether each refined genome is written *back* into the
//! population handed to the inner `tell`:
//!
//! - **Lamarckian** — refined genome *and* refined fitness flow to `tell`. The
//!   inherited traits (the genome) change.
//! - **Baldwinian** — the *original* genome flows to `tell` but carries the
//!   *refined* fitness. The phenotype's learned advantage shows up as fitness
//!   pressure without altering the inherited genome.
//! - **`Partial(p)`** — each refined individual is written back Lamarckian-style
//!   with probability `p` (drawn per refined individual), Baldwinian otherwise.
//!
//! Regardless of policy, the refined fitness *always* replaces the original
//! fitness for covered rows — Baldwinian differs only in that the genome is left
//! untouched.
//!
//! # Intentional break of the `Strategy` purity convention
//!
//! [`Strategy`] documents itself as *pure*: `ask`/`tell` take `&self` and carry
//! no interior mutability so many instances can run in parallel without locks.
//! **This wrapper deliberately breaks that convention.** It holds a
//! [`parking_lot::Mutex`] around its fitness function because local search needs
//! `&mut F` (an [`FitnessFn`] is `&mut self`) while `tell` only has `&self`. The
//! lock is wrapper-private and uncontended in the single-harness driving model
//! (one `tell` in flight at a time), so it costs an uncontended lock/unlock per
//! generation and never blocks. A reader writing code generic over
//! `S: Strategy<B>` should be aware that `MemeticWrapper` is *not* a pure,
//! lock-free strategy like the others in this crate.
//!
//! # RNG discipline
//!
//! All refinement randomness flows through [`seed_stream`]; the wrapper never
//! touches the process-wide backend RNG. See the
//! [`tell`](MemeticWrapper#impl-Strategy<B>-for-MemeticWrapper<B,+S,+L,+F>)
//! flow for the two-stream scheme that makes `Partial(1.0)` bit-identical to
//! `Lamarckian` and `Partial(0.0)` to `Baldwinian`.

use std::fmt::Debug;
use std::marker::PhantomData;

use burn::tensor::{Tensor, TensorData, backend::Backend};
use parking_lot::Mutex;
use rand::{Rng, RngExt};

use crate::fitness::{BatchFitnessFn, FitnessFn};
use crate::local_search::LocalSearch;
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};
use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::objective::ObjectiveSense;
use rlevo_core::probability::Probability;

/// Controls how a refined genome's gains are written back into the population.
///
/// See the [module docs](self) for the semantics of each policy. The default,
/// [`Partial(0.5)`](WritebackPolicy::Partial), is a deliberate middle ground:
/// half of refined individuals inherit their refined genome, the other half keep
/// their original genome but pay the refined fitness — a blend that avoids both
/// Lamarckian premature convergence and the slow Baldwin effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WritebackPolicy {
    /// Refined genome *and* refined fitness flow to the inner `tell`.
    Lamarckian,
    /// Original genome flows to the inner `tell`, carrying the refined fitness.
    Baldwinian,
    /// Per refined individual: write the refined genome back (Lamarckian) with
    /// probability `p`, otherwise keep the original genome (Baldwinian). The
    /// refined fitness is used either way.
    ///
    /// `p` is a [`Probability`], valid by construction (`[0, 1]`, NaN/Inf
    /// rejected), so the writeback draw `rng < p` can never silently degenerate.
    ///
    /// Because `Partial` writeback draws from a dedicated mask RNG stream that
    /// is independent of the refinement stream, `Partial(Probability::new(1.0))`
    /// is **bit-identical** to [`Lamarckian`](WritebackPolicy::Lamarckian) and
    /// `Partial(Probability::new(0.0))` is bit-identical to
    /// [`Baldwinian`](WritebackPolicy::Baldwinian) on the same seed.
    Partial(Probability),
}

impl Default for WritebackPolicy {
    fn default() -> Self {
        Self::Partial(Probability::new(0.5))
    }
}

/// Determines which population members are refined each generation.
///
/// # Cost and tuning
///
/// Coverage is the dominant cost knob: each refined row spends up to
/// `Params::max_iters` fitness evaluations, so [`Full`](Self::Full) costs
/// `pop_size`× a [`TopK { k: 1 }`](Self::TopK) generation. When the budget that
/// matters is
/// *evaluations to reach a target* (not wall-clock or final-gen fitness), wide
/// coverage with a heavy searcher can lose to bare evolution: it spends its
/// eval budget polishing individuals that selection would have discarded
/// anyway. **Tune against evals-to-target**, not against a fixed generation
/// count — a fixed-gens comparison hides the refinement evals and flatters wide
/// coverage. The default, [`TopK { k: 1 }`](Self::TopK), refines only the
/// single best individual and is the cheapest sane starting point.
///
/// One caveat cuts the other way: on a *separable* landscape with basin-width
/// search steps, axis-aligned hill climbing is nearly a direct solver, so wide
/// coverage with an untuned high-`max_iters` searcher can dominate. That is a
/// landscape artifact, not a config to copy — re-tune per problem.
///
/// The wrapper avoids the seeding-eval waste that an unaware caller would pay:
/// it hands each searcher the fitness the harness already computed via
/// [`LocalSearch::refine_with_known_fitness`], so a refined row spends its evals
/// on probes rather than re-scoring its own input (ADR 0016 reversal criteria).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoveragePolicy {
    /// Refine every individual.
    Full,
    /// Refine only the `k` fittest (largest-fitness, canonical maximise)
    /// individuals, ties broken by lower index. `k` is clamped to the
    /// population size.
    TopK {
        /// Number of fittest individuals to refine.
        k: usize,
    },
}

impl Default for CoveragePolicy {
    fn default() -> Self {
        Self::TopK { k: 1 }
    }
}

/// Static parameters for a [`MemeticWrapper`] run.
///
/// Composes the inner strategy's parameters with the local searcher's, plus the
/// two memetic policies.
#[derive(Clone, Debug)]
pub struct MemeticParams<SP, LP> {
    /// Parameters forwarded verbatim to the inner [`Strategy`].
    pub inner: SP,
    /// Parameters forwarded verbatim to the [`LocalSearch`] on every `refine`.
    pub local: LP,
    /// How refined gains are written back. See [`WritebackPolicy`].
    pub writeback: WritebackPolicy,
    /// Which individuals are refined. See [`CoveragePolicy`].
    pub coverage: CoveragePolicy,
}

/// Validation delegates to the wrapped inner strategy's config — the memetic
/// wrapper is the harness chokepoint for that inner config. Only `SP: Validate`
/// is required (the local-searcher params `LP` carry only simple step-size
/// knobs and are left unconstrained), so `MemeticParams` stays usable with any
/// local searcher while still rejecting an invalid inner configuration.
impl<SP: Validate, LP> Validate for MemeticParams<SP, LP> {
    fn validate(&self) -> Result<(), ConfigError> {
        self.inner.validate()
    }
}

/// Generation-to-generation state for a [`MemeticWrapper`].
///
/// Wraps the inner strategy's state and carries the memetic generation counter
/// (used to derive deterministic per-generation refinement seeds).
///
/// Fields are private for consistency with the rest of the crate's state
/// types; the wrapped `inner` is an opaque `St` with no invariant this wrapper
/// can check, so construction is the infallible [`new`](MemeticState::new)
/// rather than a `try_new`.
#[derive(Clone, Debug)]
pub struct MemeticState<St> {
    /// The wrapped inner [`Strategy`] state.
    inner: St,
    /// Number of completed `tell` calls — the generation index threaded into
    /// [`seed_stream`] so each generation refines from an independent stream.
    generation: u64,
}

impl<St> MemeticState<St> {
    /// Wraps an inner strategy state with a memetic generation counter.
    #[must_use]
    pub fn new(inner: St, generation: u64) -> Self {
        Self { inner, generation }
    }

    /// Borrows the wrapped inner strategy state.
    #[must_use]
    pub fn inner(&self) -> &St {
        &self.inner
    }

    /// Mutably borrows the wrapped inner strategy state.
    pub fn inner_mut(&mut self) -> &mut St {
        &mut self.inner
    }

    /// Number of completed `tell` calls.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Wraps an inner [`Strategy`] with per-individual [`LocalSearch`] refinement.
///
/// `MemeticWrapper` is itself a `Strategy<B, Genome = Tensor<B, 2>>`, so it
/// composes with any real-valued strategy and drops into
/// [`EvolutionaryHarness`](crate::strategy::EvolutionaryHarness) unchanged.
///
/// # The two fitness instances
///
/// The harness owns *its own* fitness instance (it calls `evaluate_batch` once
/// per generation to score the asked population); this wrapper owns a *separate*
/// instance behind a [`Mutex`], used only to score local-search probes.
///
/// **If `F` is stateful (counters, caches, RNG), the two instances must share
/// that state via interior mutability (e.g. `Arc<AtomicUsize>`) — otherwise
/// they silently diverge.** A naive `#[derive(Clone)]`-then-pass approach gives
/// each instance an independent counter, and an evaluation-budget accounting
/// across both will under-count. The headline Rastrigin benchmark shares a
/// single `Arc<AtomicUsize>` eval counter across both instances for exactly this
/// reason.
///
/// # Example
///
/// Wrap Differential Evolution with hill-climbing refinement and drive a couple
/// of generations by hand:
///
/// ```
/// use burn::backend::Flex;
/// use burn::tensor::{Tensor, TensorData, backend::Backend};
/// use rand::{rngs::StdRng, SeedableRng};
/// use rlevo_evolution::Strategy;
/// use rlevo_evolution::algorithms::de::{DeConfig, DifferentialEvolution};
/// use rlevo_evolution::algorithms::memetic::{
///     CoveragePolicy, MemeticParams, MemeticWrapper, WritebackPolicy,
/// };
/// use rlevo_evolution::fitness::BatchFitnessFn;
/// use rlevo_evolution::local_search::{HillClimbing, HillClimbingParams};
/// use rlevo_core::bounds::Bounds;
///
/// // Sphere objective: sum of squares per row (a cost → Minimize).
/// use rlevo_core::objective::ObjectiveSense;
/// struct Sphere;
/// impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2>> for Sphere {
///     fn evaluate_batch(
///         &mut self,
///         pop: &Tensor<B, 2>,
///         device: &B::Device,
///     ) -> Tensor<B, 1> {
///         let squared = pop.clone() * pop.clone();
///         squared.sum_dim(1).squeeze_dim::<1>(1)
///     }
///     fn sense(&self) -> ObjectiveSense { ObjectiveSense::Minimize }
/// }
///
/// let device = Default::default();
/// let bounds = Bounds::new(-5.12, 5.12);
/// let strategy = MemeticWrapper::<Flex, _, _, _>::new(
///     DifferentialEvolution::<Flex>::new(),
///     HillClimbing,
///     Sphere,
/// );
/// let params = MemeticParams {
///     inner: DeConfig::default_for(16, 4),
///     local: HillClimbingParams::default_for(bounds),
///     writeback: WritebackPolicy::Lamarckian,
///     coverage: CoveragePolicy::TopK { k: 2 },
/// };
///
/// let mut rng = StdRng::seed_from_u64(0);
/// let mut state = strategy.init(&params, &mut rng, &device);
/// let mut scorer = Sphere;
/// for _ in 0..3 {
///     let (pop, asked) = strategy.ask(&params, &state, &mut rng, &device);
///     // The harness would do this; here we score it ourselves.
///     let fitness = scorer.evaluate_batch(&pop, &device);
///     let (next, _metrics) = strategy.tell(&params, pop, fitness, asked, &mut rng);
///     state = next;
/// }
/// assert!(strategy.best(&state).is_some());
/// ```
pub struct MemeticWrapper<B, S, L, F>
where
    B: Backend,
    S: Strategy<B, Genome = Tensor<B, 2>>,
    L: LocalSearch<B>,
    F: BatchFitnessFn<B, Tensor<B, 2>>,
{
    inner: S,
    local: L,
    fitness: Mutex<F>,
    _backend: PhantomData<fn() -> B>,
}

impl<B, S, L, F> MemeticWrapper<B, S, L, F>
where
    B: Backend,
    S: Strategy<B, Genome = Tensor<B, 2>>,
    L: LocalSearch<B>,
    F: BatchFitnessFn<B, Tensor<B, 2>>,
{
    /// Builds a memetic wrapper from an inner strategy, a local searcher, and a
    /// fitness function used **only** for local-search probes.
    ///
    /// The harness owns a separate fitness instance; **if `F` is stateful
    /// (counters, caches, RNG), the two instances must share that state via
    /// interior mutability (e.g. `Arc<AtomicUsize>`) — otherwise they silently
    /// diverge.** See the [type-level docs](MemeticWrapper#the-two-fitness-instances).
    pub fn new(inner: S, local: L, fitness: F) -> Self {
        Self {
            inner,
            local,
            fitness: Mutex::new(fitness),
            _backend: PhantomData,
        }
    }
}

// `F` is not `Debug`; mirror the `EvolutionaryHarness` precedent and use
// `finish_non_exhaustive()` so the `missing_debug_implementations` lint is
// satisfied without bounding `F: Debug`.
impl<B, S, L, F> Debug for MemeticWrapper<B, S, L, F>
where
    B: Backend,
    S: Strategy<B, Genome = Tensor<B, 2>>,
    L: LocalSearch<B>,
    F: BatchFitnessFn<B, Tensor<B, 2>>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `S`/`L`/`F` are not bounded `Debug`; mirror `EvolutionaryHarness` and
        // emit a non-exhaustive shell so the lint is satisfied without forcing
        // those bounds onto the public type.
        f.debug_struct("MemeticWrapper").finish_non_exhaustive()
    }
}

/// Adapts a population-level [`BatchFitnessFn`] into a single-row
/// [`FitnessFn`] for local search, in **canonical (maximise)** space.
///
/// Each [`evaluate_one`](FitnessFn::evaluate_one) builds a `[1, D]` tensor from
/// the host-side row, calls `evaluate_batch`, pulls the single scalar back, and
/// maps it into canonical space via the wrapped fn's
/// [`sense`](BatchFitnessFn::sense). Local searchers maximise, and the seed
/// fitness the memetic wrapper hands them (the harness-canonicalised value) is
/// also canonical, so both ends agree. This is deliberately the slow path —
/// local search re-uploads one row at a time — and is private plumbing
/// internal to the wrapper.
struct RowFitness<'a, B: Backend, F> {
    inner: &'a mut F,
    device: &'a B::Device,
    sense: ObjectiveSense,
}

impl<B, F> FitnessFn<Vec<f32>> for RowFitness<'_, B, F>
where
    B: Backend,
    F: BatchFitnessFn<B, Tensor<B, 2>>,
{
    fn evaluate_one(&mut self, member: &Vec<f32>) -> f32 {
        let dim: usize = member.len();
        let data: TensorData = TensorData::new(member.clone(), [1, dim]);
        let row: Tensor<B, 2> = Tensor::<B, 2>::from_data(data, self.device);
        let fitness: Tensor<B, 1> = self.inner.evaluate_batch(&row, self.device);
        let values: Vec<f32> = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let natural = values.first().copied().unwrap_or(f32::NEG_INFINITY);
        self.sense.to_canonical(natural)
    }
}

impl<B, S, L, F> Strategy<B> for MemeticWrapper<B, S, L, F>
where
    B: Backend,
    S: Strategy<B, Genome = Tensor<B, 2>>,
    L: LocalSearch<B>,
    F: BatchFitnessFn<B, Tensor<B, 2>>,
{
    type Params = MemeticParams<S::Params, L::Params>;
    type State = MemeticState<S::State>;
    type Genome = Tensor<B, 2>;

    /// Delegates to the inner strategy's `init` and seeds the memetic
    /// generation counter to zero.
    fn init(
        &self,
        params: &Self::Params,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::State {
        let inner: S::State = self.inner.init(&params.inner, rng, device);
        MemeticState {
            inner,
            generation: 0,
        }
    }

    /// Pure delegation to the inner strategy's `ask`. The generation counter is
    /// unchanged here — it increments only in [`tell`](Self::tell).
    fn ask(
        &self,
        params: &Self::Params,
        state: &Self::State,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Self::Genome, Self::State) {
        let (population, inner): (Tensor<B, 2>, S::State) =
            self.inner.ask(&params.inner, &state.inner, rng, device);
        (
            population,
            MemeticState {
                inner,
                generation: state.generation,
            },
        )
    }

    /// Refines a covered subset of the population, writes back the refined gains
    /// per the [`WritebackPolicy`], then delegates to the inner `tell`.
    ///
    /// # Flow
    ///
    /// 1. Host-pull the fitness vector and one flat read-only host copy of the
    ///    population; read `[pop_size, dim]` and the device.
    /// 2. Compute coverage indices ([`Full`](CoveragePolicy::Full) = all;
    ///    [`TopK`](CoveragePolicy::TopK) = the `k` largest fitnesses, ties by
    ///    lower index), then process them in ascending index order so RNG
    ///    consumption is a pure function of the `(fitness, index)` ranking.
    /// 3. Draw **exactly one** `rng.next_u64()` unconditionally (so the harness
    ///    RNG stream position is policy-invariant) and derive two independent
    ///    sub-streams: `ls_rng` for refinement
    ///    ([`SeedPurpose::LocalSearch`]) and `mask_rng` for the writeback
    ///    Bernoulli ([`SeedPurpose::Replacement`]). The split is load-bearing:
    ///    mask draws never perturb refinement draws, which makes `Partial(1.0)`
    ///    bit-identical to `Lamarckian` and `Partial(0.0)` to `Baldwinian`.
    /// 4. Lock the fitness once, refine each covered row, always set
    ///    `refined_fit[i]` to the refined fitness, and decide writeback
    ///    (Lamarckian → always; Baldwinian → never; `Partial(p)` → one
    ///    `mask_rng` Bernoulli per refined index).
    /// 5. Write back only Lamarckian rows via `slice_assign` onto the *original*
    ///    population tensor. When there are zero writeback rows, the exact tensor
    ///    returned by `ask` is handed to the inner `tell` — no host round-trip.
    /// 6. Rebuild the fitness tensor and delegate to the inner `tell`, returning
    ///    its metrics verbatim alongside `generation + 1`.
    ///
    /// Refinement runs on **every** `tell`, including the first. For a wrapped
    /// DE this means gen-0 refinement happens before DE's empty-fitness sentinel
    /// stash; under Baldwinian writeback the inner population still carries the
    /// *unrefined* genomes but the *refined* fitness, which raises DE's greedy
    /// replacement bar — the intended Baldwin effect.
    ///
    /// Refined fitness is **never** clamped against the old fitness: the
    /// [`LocalSearch`] contract already guarantees monotone non-worsening, and
    /// clamping would manufacture a stale fitness on Lamarckian rows.
    fn tell(
        &self,
        params: &Self::Params,
        population: Self::Genome,
        fitness: Tensor<B, 1>,
        state: Self::State,
        rng: &mut dyn Rng,
    ) -> (Self::State, StrategyMetrics) {
        let generation: u64 = state.generation;

        // (1) Host-pull fitness and one flat read-only host copy of the population.
        let mut refined_fit: Vec<f32> = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let dims: [usize; 2] = population.dims();
        let pop_size: usize = dims[0];
        let dim: usize = dims[1];
        let device: B::Device = population.device();
        let flat: Vec<f32> = population
            .clone()
            .into_data()
            .into_vec::<f32>()
            .unwrap_or_default();

        // (2) Coverage indices, processed in ascending index order.
        let mut indices: Vec<usize> = coverage_indices(&params.coverage, &refined_fit, pop_size);
        indices.sort_unstable();

        // (3) Exactly one host RNG draw — unconditionally — so the harness
        // stream position is policy- and coverage-invariant.
        let base: u64 = rng.next_u64();
        let mut ls_rng = seed_stream(base, generation, SeedPurpose::LocalSearch);
        let mut mask_rng = seed_stream(base, generation, SeedPurpose::Replacement);

        // `WritebackPolicy::Partial` now carries a `Probability` (valid by
        // construction), so the old debug-only range assert is unnecessary —
        // see ADR 0031.

        // (4) Refine each covered row; collect Lamarckian writebacks.
        let mut writeback_rows: Vec<(usize, Vec<f32>)> = Vec::with_capacity(indices.len());
        {
            let mut guard = self.fitness.lock();
            // Read the sense before the mutable borrow for `inner`; local
            // search runs in canonical space, so `RowFitness` canonicalises.
            let sense = guard.sense();
            let mut row_fitness: RowFitness<'_, B, F> = RowFitness {
                inner: &mut *guard,
                device: &device,
                sense,
            };
            for &i in &indices {
                let start: usize = i * dim;
                let row: Vec<f32> = flat[start..start + dim].to_vec();
                // The harness already scored this row this generation; hand that
                // fitness to the searcher so it skips the seeding eval instead of
                // re-scoring its own input. `refined_fit[i]` still holds the
                // original harness value — it is overwritten only below, and each
                // covered index `i` is distinct.
                let known_fit: f32 = refined_fit[i];
                let (refined, f_refined): (Vec<f32>, f32) = self.local.refine_with_known_fitness(
                    &params.local,
                    row,
                    known_fit,
                    &mut row_fitness,
                    &mut ls_rng,
                );
                debug_assert_eq!(refined.len(), dim, "local search must preserve genome length");
                // Baldwinian keeps the original genome but pays refined fitness;
                // Lamarckian writes the genome too. Either way the fitness is
                // the refined value.
                refined_fit[i] = f_refined;
                let writeback: bool = match params.writeback {
                    WritebackPolicy::Lamarckian => true,
                    WritebackPolicy::Baldwinian => false,
                    // One Bernoulli draw per refined index, from the dedicated
                    // mask stream, so Lamarckian/Baldwinian runs share an
                    // identical `ls_rng` schedule (they draw nothing here).
                    WritebackPolicy::Partial(p) => mask_rng.random::<f32>() < p.get(),
                };
                if writeback {
                    writeback_rows.push((i, refined));
                }
            }
        } // guard dropped before delegating to the inner tell.

        // (5) Writeback. Start from the ORIGINAL population tensor (moved,
        // untouched). With zero writeback rows the inner `tell` receives the
        // exact tensor `ask` returned — no host round-trip, no rebuild.
        let mut new_pop: Tensor<B, 2> = population;
        for (i, row) in writeback_rows {
            let data: TensorData = TensorData::new(row, [1, dim]);
            let row_tensor: Tensor<B, 2> = Tensor::<B, 2>::from_data(data, &device);
            new_pop = new_pop.slice_assign([i..i + 1, 0..dim], row_tensor);
        }

        // (6) Rebuild fitness and delegate.
        let new_fit: Tensor<B, 1> =
            Tensor::<B, 1>::from_data(TensorData::new(refined_fit, [pop_size]), &device);
        let (inner, metrics): (S::State, StrategyMetrics) =
            self.inner
                .tell(&params.inner, new_pop, new_fit, state.inner, rng);
        (
            MemeticState {
                inner,
                generation: generation + 1,
            },
            metrics,
        )
    }

    /// Delegates to the inner strategy's `best`.
    fn best(&self, state: &Self::State) -> Option<(Self::Genome, f32)> {
        self.inner.best(&state.inner)
    }
}

/// Computes the refinement coverage indices for a generation (unsorted).
///
/// `Full` yields `0..pop_size`; `TopK { k }` yields the indices of the `k`
/// largest fitness values (canonical maximise: higher is fitter), ties broken
/// by lower index (stable sort over `(fitness, index)`), with `k` clamped to
/// `pop_size`. The caller is responsible for sorting the result into ascending
/// index order.
fn coverage_indices(policy: &CoveragePolicy, fitness: &[f32], pop_size: usize) -> Vec<usize> {
    match *policy {
        CoveragePolicy::Full => (0..pop_size).collect(),
        CoveragePolicy::TopK { k } => {
            let k: usize = k.min(pop_size);
            let mut ranked: Vec<usize> = (0..pop_size).collect();
            // Sanitize NaN → −inf (worst) so a NaN-fitness member can never be
            // covered as a top-k member. Stable sort by (fitness desc, index):
            // `sort_by` is stable so equal fitnesses keep ascending-index order,
            // making ties break by lower index.
            let sane: Vec<f32> = fitness
                .iter()
                .map(|&f| crate::fitness::sanitize_fitness(f))
                .collect();
            ranked.sort_by(|&a, &b| sane[b].total_cmp(&sane[a]));
            ranked.truncate(k);
            ranked
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::de::{DeConfig, DifferentialEvolution};
    use crate::algorithms::ga::{GaConfig, GeneticAlgorithm};
    use crate::local_search::{
        HillClimbing, HillClimbingParams, SimulatedAnnealing, SimulatedAnnealingParams,
    };
    use crate::strategy::EvolutionaryHarness;
    use rlevo_core::bounds::Bounds;
    use burn::backend::Flex;
    use burn::tensor::backend::BackendTypes;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type TestBackend = Flex;

    #[test]
    fn memetic_state_new_round_trips() {
        let mut state = MemeticState::new(7_u32, 3);
        assert_eq!(*state.inner(), 7);
        assert_eq!(state.generation(), 3);
        *state.inner_mut() = 11;
        assert_eq!(*state.inner(), 11);
    }

    const BOUNDS: Bounds = Bounds::new(-5.12, 5.12);

    // ---------------------------------------------------------------------
    // Probes.
    // ---------------------------------------------------------------------

    /// A strategy probe (mirrors the `strategy.rs` Constant-probe pattern):
    /// `ask` returns a fixed population built from `state`; `tell` records the
    /// exact population tensor + fitness it received so a test can assert on
    /// them. No real evolutionary dynamics.
    #[derive(Debug, Clone, Copy)]
    struct RecordingStrategy;

    #[derive(Debug, Clone)]
    struct RecParams {
        /// The fixed population every `ask` returns (row-major, `[pop, dim]`).
        rows: Vec<f32>,
        pop: usize,
        dim: usize,
    }

    #[derive(Debug, Clone)]
    struct RecState {
        /// Population tensor handed to the most recent `tell`, as flat host f32.
        received_pop: Option<Vec<f32>>,
        /// Fitness handed to the most recent `tell`, as host f32.
        received_fit: Option<Vec<f32>>,
        best: f32,
        generation: usize,
    }

    impl Strategy<TestBackend> for RecordingStrategy {
        type Params = RecParams;
        type State = RecState;
        type Genome = Tensor<TestBackend, 2>;

        fn init(
            &self,
            _params: &RecParams,
            _rng: &mut dyn Rng,
            _device: &<TestBackend as BackendTypes>::Device,
        ) -> RecState {
            RecState {
                received_pop: None,
                received_fit: None,
                best: f32::NEG_INFINITY,
                generation: 0,
            }
        }

        fn ask(
            &self,
            params: &RecParams,
            state: &RecState,
            _rng: &mut dyn Rng,
            device: &<TestBackend as BackendTypes>::Device,
        ) -> (Tensor<TestBackend, 2>, RecState) {
            let data = TensorData::new(params.rows.clone(), [params.pop, params.dim]);
            let pop = Tensor::<TestBackend, 2>::from_data(data, device);
            (pop, state.clone())
        }

        fn tell(
            &self,
            _params: &RecParams,
            population: Tensor<TestBackend, 2>,
            fitness: Tensor<TestBackend, 1>,
            mut state: RecState,
            _rng: &mut dyn Rng,
        ) -> (RecState, StrategyMetrics) {
            let pop_host = population.into_data().into_vec::<f32>().unwrap();
            let fit_host = fitness.into_data().into_vec::<f32>().unwrap();
            state.received_pop = Some(pop_host);
            state.received_fit = Some(fit_host.clone());
            state.generation += 1;
            let metrics =
                StrategyMetrics::from_host_fitness(state.generation, &fit_host, state.best);
            state.best = metrics.best_fitness_ever();
            (state, metrics)
        }

        fn best(&self, _state: &RecState) -> Option<(Tensor<TestBackend, 2>, f32)> {
            None
        }
    }

    /// Negated-sphere fitness (a maximise objective with optimum 0 at the
    /// origin) counting the total number of evaluated ROWS. Each
    /// `RowFitness::evaluate_one` is one `[1, D]` batch, so refinement evals are
    /// counted too.
    #[derive(Debug, Default)]
    struct CountingBatchFitness {
        rows: usize,
    }

    impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2>> for CountingBatchFitness {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2>,
            device: &<B as BackendTypes>::Device,
        ) -> Tensor<B, 1> {
            let dims = population.dims();
            self.rows += dims[0];
            let flat = population.clone().into_data().into_vec::<f32>().unwrap();
            let (pop, dim) = (dims[0], dims[1]);
            let mut out = Vec::with_capacity(pop);
            for r in 0..pop {
                let start = r * dim;
                let f: f32 = -flat[start..start + dim].iter().map(|v| v * v).sum::<f32>();
                out.push(f);
            }
            Tensor::<B, 1>::from_data(TensorData::new(out, [pop]), device)
        }

        fn sense(&self) -> ObjectiveSense {
            ObjectiveSense::Maximize
        }
    }

    /// Negated sphere on a flat host genome — the canonical fitness of a row,
    /// for re-deriving expected values. Higher (closer to 0) is better.
    fn neg_sphere(row: &[f32]) -> f32 {
        -row.iter().map(|v| v * v).sum::<f32>()
    }

    fn rec_params(rows: Vec<f32>, pop: usize, dim: usize) -> RecParams {
        RecParams { rows, pop, dim }
    }

    /// A deterministic, spread-out population whose fitnesses are all distinct.
    fn fixed_population(pop: usize, dim: usize) -> Vec<f32> {
        let mut rows = Vec::with_capacity(pop * dim);
        for r in 0..pop {
            for c in 0..dim {
                #[allow(clippy::cast_precision_loss)]
                let v = 0.5 + (r as f32) * 0.37 + (c as f32) * 0.11;
                rows.push(v);
            }
        }
        rows
    }

    // ---------------------------------------------------------------------
    // 0. Documented defaults (falsifiable equality checks).
    // ---------------------------------------------------------------------

    #[test]
    fn writeback_policy_default_is_partial_half() {
        assert_eq!(WritebackPolicy::default(), WritebackPolicy::Partial(Probability::new(0.5)));
    }

    #[test]
    fn coverage_policy_default_is_top_k_one() {
        assert_eq!(CoveragePolicy::default(), CoveragePolicy::TopK { k: 1 });
    }

    #[test]
    fn coverage_indices_never_covers_nan_fitness() {
        // NaN sanitises to −inf (worst), so a NaN-fitness member must never be
        // selected as a top-k covered member ahead of a finite one.
        let fitness = [3.0f32, f32::NAN, 5.0, 1.0];
        let top3 = coverage_indices(&CoveragePolicy::TopK { k: 3 }, &fitness, 4);
        // Best-first among finite fitnesses: 5.0 (idx 2), 3.0 (idx 0), 1.0 (idx 3);
        // the NaN member (idx 1) is excluded.
        assert_eq!(top3, vec![2, 0, 3]);
        assert!(!top3.contains(&1));
        // Covering all four ranks the NaN member last.
        let all = coverage_indices(&CoveragePolicy::TopK { k: 4 }, &fitness, 4);
        assert_eq!(all, vec![2, 0, 3, 1]);
    }

    // ---------------------------------------------------------------------
    // 1. Baldwinian bit-identity.
    // ---------------------------------------------------------------------

    #[test]
    #[allow(clippy::float_cmp)]
    fn baldwinian_population_bit_identical_to_ask() {
        let device = <TestBackend as BackendTypes>::Device::default();
        let (pop, dim) = (5usize, 3usize);
        let rows = fixed_population(pop, dim);

        let strategy = MemeticWrapper::<TestBackend, _, _, _>::new(
            RecordingStrategy,
            HillClimbing,
            CountingBatchFitness::default(),
        );
        let params = MemeticParams {
            inner: rec_params(rows.clone(), pop, dim),
            local: HillClimbingParams::default_for(BOUNDS),
            writeback: WritebackPolicy::Baldwinian,
            coverage: CoveragePolicy::TopK { k: 2 },
        };

        let mut rng = StdRng::seed_from_u64(7);
        let state = strategy.init(&params, &mut rng, &device);
        let (ask_pop, asked) = strategy.ask(&params, &state, &mut rng, &device);
        let ask_bytes = ask_pop.clone().into_data().into_vec::<f32>().unwrap();
        // Original fitness for the asked population.
        let mut orig_fit = CountingBatchFitness::default();
        let orig =
            <CountingBatchFitness as BatchFitnessFn<TestBackend, _>>::evaluate_batch(
                &mut orig_fit,
                &ask_pop,
                &device,
            )
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        let fit = Tensor::<TestBackend, 1>::from_data(TensorData::new(orig.clone(), [pop]), &device);

        let (next, _m) = strategy.tell(&params, ask_pop, fit, asked, &mut rng);

        // Population handed to RecordingStrategy::tell is byte-identical to ask.
        let recv_pop = next.inner.received_pop.clone().unwrap();
        assert_eq!(recv_pop, ask_bytes, "Baldwinian must not alter the genome");

        // Covered rows (TopK{2} = the two fittest, highest-canonical rows) have
        // refined fitness >= original (canonical maximise); all others
        // unchanged. Covered = indices 0,1 here (canonical −sphere fitness
        // decreases with row index for this population, so the lowest indices
        // are the fittest).
        let recv_fit = next.inner.received_fit.clone().unwrap();
        for i in 0..pop {
            if i < 2 {
                assert!(
                    recv_fit[i] >= orig[i],
                    "covered row {i}: refined {} must be >= original {}",
                    recv_fit[i],
                    orig[i]
                );
                // The refined fitness cannot exceed the global maximum of the
                // negated-sphere objective (0 at the origin).
                assert!(recv_fit[i] <= 1e-6);
            } else {
                assert_eq!(recv_fit[i], orig[i], "uncovered row {i} must be unchanged");
            }
        }
    }

    // ---------------------------------------------------------------------
    // 2. Lamarckian row equality.
    // ---------------------------------------------------------------------

    #[test]
    #[allow(clippy::float_cmp)]
    fn lamarckian_covered_rows_change_uncovered_identical() {
        let device = <TestBackend as BackendTypes>::Device::default();
        let (pop, dim) = (5usize, 3usize);
        let rows = fixed_population(pop, dim);

        let strategy = MemeticWrapper::<TestBackend, _, _, _>::new(
            RecordingStrategy,
            HillClimbing,
            CountingBatchFitness::default(),
        );
        let params = MemeticParams {
            inner: rec_params(rows.clone(), pop, dim),
            local: HillClimbingParams::default_for(BOUNDS),
            writeback: WritebackPolicy::Lamarckian,
            coverage: CoveragePolicy::TopK { k: 2 },
        };

        let mut rng = StdRng::seed_from_u64(11);
        let state = strategy.init(&params, &mut rng, &device);
        let (ask_pop, asked) = strategy.ask(&params, &state, &mut rng, &device);
        let ask_bytes = ask_pop.clone().into_data().into_vec::<f32>().unwrap();
        let mut fitfn = CountingBatchFitness::default();
        let orig =
            <CountingBatchFitness as BatchFitnessFn<TestBackend, _>>::evaluate_batch(
                &mut fitfn, &ask_pop, &device,
            )
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        let fit = Tensor::<TestBackend, 1>::from_data(TensorData::new(orig, [pop]), &device);

        let (next, _m) = strategy.tell(&params, ask_pop, fit, asked, &mut rng);
        let recv_pop = next.inner.received_pop.clone().unwrap();
        let recv_fit = next.inner.received_fit.clone().unwrap();

        // Indexing several parallel host buffers by row; an iterator over one of
        // them would not read more clearly than the explicit row index.
        #[allow(clippy::needless_range_loop)]
        for i in 0..pop {
            let start = i * dim;
            let recv_row = &recv_pop[start..start + dim];
            let ask_row = &ask_bytes[start..start + dim];
            if i < 2 {
                // Covered rows changed (HillClimbing improves the negated
                // sphere from a non-optimal start).
                assert_ne!(recv_row, ask_row, "covered row {i} should have changed");
                // received fitness[i] equals a fresh canonical eval of received
                // row i (the negated sphere).
                approx::assert_relative_eq!(recv_fit[i], neg_sphere(recv_row), epsilon = 1e-5);
            } else {
                assert_eq!(recv_row, ask_row, "uncovered row {i} must be bit-identical");
            }
        }
    }

    // ---------------------------------------------------------------------
    // 3. Partial boundaries (stochastic searcher pins the two-stream split).
    // ---------------------------------------------------------------------

    /// Drives `gens` wrapper tell cycles and returns the full trajectory of
    /// (received population, received fitness) the `RecordingStrategy` saw.
    fn sa_trajectory(
        writeback: WritebackPolicy,
        seed: u64,
        gens: usize,
    ) -> Vec<(Vec<f32>, Vec<f32>)> {
        let device = <TestBackend as BackendTypes>::Device::default();
        let (pop, dim) = (4usize, 3usize);
        let rows = fixed_population(pop, dim);
        let strategy = MemeticWrapper::<TestBackend, _, _, _>::new(
            RecordingStrategy,
            SimulatedAnnealing,
            CountingBatchFitness::default(),
        );
        let params = MemeticParams {
            inner: rec_params(rows, pop, dim),
            local: SimulatedAnnealingParams::default_for(BOUNDS),
            writeback,
            coverage: CoveragePolicy::Full,
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let mut state = strategy.init(&params, &mut rng, &device);
        let mut trajectory = Vec::with_capacity(gens);
        for _ in 0..gens {
            let (ask_pop, asked) = strategy.ask(&params, &state, &mut rng, &device);
            let mut fitfn = CountingBatchFitness::default();
            let orig =
                <CountingBatchFitness as BatchFitnessFn<TestBackend, _>>::evaluate_batch(
                    &mut fitfn, &ask_pop, &device,
                );
            let (next, _m) = strategy.tell(&params, ask_pop, orig, asked, &mut rng);
            trajectory.push((
                next.inner.received_pop.clone().unwrap(),
                next.inner.received_fit.clone().unwrap(),
            ));
            state = next;
        }
        trajectory
    }

    #[test]
    fn partial_one_equals_lamarckian_partial_zero_equals_baldwinian() {
        let lam = sa_trajectory(WritebackPolicy::Lamarckian, 33, 3);
        let p1 = sa_trajectory(WritebackPolicy::Partial(Probability::new(1.0)), 33, 3);
        assert_eq!(lam, p1, "Partial(1.0) must be bit-identical to Lamarckian");

        let bald = sa_trajectory(WritebackPolicy::Baldwinian, 33, 3);
        let p0 = sa_trajectory(WritebackPolicy::Partial(Probability::new(0.0)), 33, 3);
        assert_eq!(bald, p0, "Partial(0.0) must be bit-identical to Baldwinian");
    }

    // ---------------------------------------------------------------------
    // 4. Partial mask seeded-replay.
    // ---------------------------------------------------------------------

    #[test]
    fn partial_is_seed_reproducible_and_seed_sensitive() {
        let a = sa_trajectory(WritebackPolicy::Partial(Probability::new(0.5)), 55, 3);
        let b = sa_trajectory(WritebackPolicy::Partial(Probability::new(0.5)), 55, 3);
        assert_eq!(a, b, "same seed must replay identically");

        // A different seed (almost surely) yields a different writeback pattern.
        let c = sa_trajectory(WritebackPolicy::Partial(Probability::new(0.5)), 56, 3);
        assert_ne!(a, c, "different seed should diverge");
    }

    // ---------------------------------------------------------------------
    // 5. TopK count.
    // ---------------------------------------------------------------------

    #[test]
    #[allow(clippy::float_cmp)]
    fn topk_refines_exactly_k_rows_and_k_ge_pop_equals_full() {
        let device = <TestBackend as BackendTypes>::Device::default();
        let (pop, dim) = (6usize, 2usize);
        let rows = fixed_population(pop, dim);

        let run = |coverage: CoveragePolicy| -> Vec<bool> {
            let strategy = MemeticWrapper::<TestBackend, _, _, _>::new(
                RecordingStrategy,
                HillClimbing,
                CountingBatchFitness::default(),
            );
            let params = MemeticParams {
                inner: rec_params(rows.clone(), pop, dim),
                local: HillClimbingParams::default_for(BOUNDS),
                writeback: WritebackPolicy::Lamarckian,
                coverage,
            };
            let mut rng = StdRng::seed_from_u64(3);
            let state = strategy.init(&params, &mut rng, &device);
            let (ask_pop, asked) = strategy.ask(&params, &state, &mut rng, &device);
            let ask_bytes = ask_pop.clone().into_data().into_vec::<f32>().unwrap();
            let mut fitfn = CountingBatchFitness::default();
            let orig =
                <CountingBatchFitness as BatchFitnessFn<TestBackend, _>>::evaluate_batch(
                    &mut fitfn, &ask_pop, &device,
                );
            let (next, _m) = strategy.tell(&params, ask_pop, orig, asked, &mut rng);
            let recv = next.inner.received_pop.clone().unwrap();
            // A row "changed" iff its bytes differ from ask.
            (0..pop)
                .map(|i| {
                    let s = i * dim;
                    recv[s..s + dim] != ask_bytes[s..s + dim]
                })
                .collect()
        };

        let changed_k3 = run(CoveragePolicy::TopK { k: 3 });
        assert_eq!(
            changed_k3.iter().filter(|&&c| c).count(),
            3,
            "TopK{{3}} must refine exactly 3 rows"
        );

        let changed_full = run(CoveragePolicy::Full);
        let changed_big_k = run(CoveragePolicy::TopK { k: pop + 4 });
        assert_eq!(changed_full, changed_big_k, "TopK{{k>=pop}} must equal Full");
    }

    // ---------------------------------------------------------------------
    // 6. DE round-trip.
    // ---------------------------------------------------------------------

    /// Inline negated-sphere `BatchFitnessFn` (maximise, optimum 0 at origin)
    /// evaluated on-device.
    #[derive(Debug, Default)]
    struct SphereBatch;
    impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2>> for SphereBatch {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2>,
            device: &<B as BackendTypes>::Device,
        ) -> Tensor<B, 1> {
            let dims = population.dims();
            let flat = population.clone().into_data().into_vec::<f32>().unwrap();
            let (pop, dim) = (dims[0], dims[1]);
            let mut out: Vec<f32> = Vec::with_capacity(pop);
            for r in 0..pop {
                let start = r * dim;
                out.push(-flat[start..start + dim].iter().map(|v| v * v).sum::<f32>());
            }
            Tensor::<B, 1>::from_data(TensorData::new(out, [pop]), device)
        }

        fn sense(&self) -> ObjectiveSense {
            ObjectiveSense::Maximize
        }
    }

    #[test]
    fn de_roundtrip_improves_over_generations() {
        let device = <TestBackend as BackendTypes>::Device::default();
        let dim = 4usize;
        let strategy = MemeticWrapper::<TestBackend, _, _, _>::new(
            DifferentialEvolution::<TestBackend>::new(),
            HillClimbing,
            SphereBatch,
        );
        let params = MemeticParams {
            inner: DeConfig::default_for(20, dim),
            local: HillClimbingParams::default_for(BOUNDS),
            writeback: WritebackPolicy::Lamarckian,
            coverage: CoveragePolicy::TopK { k: 3 },
        };
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, SphereBatch, 17, device, 20,
        ).expect("valid params");
        harness.reset();
        let _ = harness.step(());
        let first: f32 = harness.latest_metrics().unwrap().best_fitness_ever();
        loop {
            if harness.step(()).done {
                break;
            }
        }
        let last: f32 = harness.latest_metrics().unwrap().best_fitness_ever();
        assert!(last.is_finite(), "best must stay finite");
        // Maximise objective: best_fitness_ever climbs toward the optimum 0.
        assert!(last >= first, "best_fitness_ever must improve: {last} >= {first}");
    }

    // ---------------------------------------------------------------------
    // 7. GA round-trip smoke.
    // ---------------------------------------------------------------------

    #[test]
    fn ga_roundtrip_smoke() {
        let device = <TestBackend as BackendTypes>::Device::default();
        let dim = 4usize;
        let strategy = MemeticWrapper::<TestBackend, _, _, _>::new(
            GeneticAlgorithm::<TestBackend>::new(),
            HillClimbing,
            SphereBatch,
        );
        let params = MemeticParams {
            inner: GaConfig::default_for(16, dim),
            local: HillClimbingParams::default_for(BOUNDS),
            writeback: WritebackPolicy::default(),
            coverage: CoveragePolicy::default(),
        };
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, SphereBatch, 5, device, 5,
        ).expect("valid params");
        harness.reset();
        for _ in 0..5 {
            let _ = harness.step(());
        }
        assert!(harness.latest_metrics().unwrap().best_fitness_ever().is_finite());
    }

    // ---------------------------------------------------------------------
    // 8. One-draw invariance: the harness RNG advances identically regardless
    //    of policy/coverage.
    // ---------------------------------------------------------------------

    #[test]
    fn one_draw_invariant_across_policies() {
        let device = <TestBackend as BackendTypes>::Device::default();
        let (pop, dim) = (5usize, 3usize);
        let rows = fixed_population(pop, dim);

        // For RecordingStrategy, `tell` never draws from the rng, so the only
        // wrapper-side consumption is the single `next_u64()`. After one tell
        // the rng's next value must be equal across every policy/coverage.
        let next_after = |writeback: WritebackPolicy, coverage: CoveragePolicy| -> u64 {
            let strategy = MemeticWrapper::<TestBackend, _, _, _>::new(
                RecordingStrategy,
                HillClimbing,
                CountingBatchFitness::default(),
            );
            let params = MemeticParams {
                inner: rec_params(rows.clone(), pop, dim),
                local: HillClimbingParams::default_for(BOUNDS),
                writeback,
                coverage,
            };
            let mut rng = StdRng::seed_from_u64(101);
            let state = strategy.init(&params, &mut rng, &device);
            let (ask_pop, asked) = strategy.ask(&params, &state, &mut rng, &device);
            let mut fitfn = CountingBatchFitness::default();
            let orig =
                <CountingBatchFitness as BatchFitnessFn<TestBackend, _>>::evaluate_batch(
                    &mut fitfn, &ask_pop, &device,
                );
            let (_next, _m) = strategy.tell(&params, ask_pop, orig, asked, &mut rng);
            rng.next_u64()
        };

        let baseline = next_after(WritebackPolicy::Lamarckian, CoveragePolicy::TopK { k: 1 });
        assert_eq!(
            baseline,
            next_after(WritebackPolicy::Baldwinian, CoveragePolicy::TopK { k: 1 }),
        );
        assert_eq!(
            baseline,
            next_after(WritebackPolicy::Partial(Probability::new(0.5)), CoveragePolicy::Full),
        );
        assert_eq!(
            baseline,
            next_after(WritebackPolicy::Lamarckian, CoveragePolicy::Full),
        );
    }
}
