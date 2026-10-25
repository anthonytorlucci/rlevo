//! NEAT — topology-evolving neuroevolution.
//!
//! `rlevo` implements NEAT (`NeuroEvolution` of Augmenting Topologies; Stanley &
//! Miikkulainen, 2002). [`NeatStrategy`] grows network topology *and* weights
//! open-endedly from a minimal seed. It is a [`Strategy`] whose genome is a
//! population of host-side graphs, `Vec<TopologyGenome>`, with no tensor
//! representation. A [`BatchFitnessFn`] over that genome scores it: one network
//! at a time through
//! [`InterpretedPhenotype`](crate::neuroevolution::phenotype::InterpretedPhenotype),
//! or the whole population at once through [`BatchGraphFitness`].
//!
//! # A run
//!
//! An [`EvolutionaryLoop`] drives it, and the evaluation harness runs many
//! through the loop's
//! [`GenerationProbe`](rlevo_core::evaluation::GenerationProbe), as it runs any
//! other strategy:
//!
//! ```ignore
//! let params = NeatParams::default_for(150, 2, 1);
//! let mut evo = EvolutionaryLoop::new(NeatStrategy::new(), params, xor_fitness, seed, device, 300)?;
//! evo.reset();
//! while !evo.is_done() {
//!     evo.step();
//! }
//! let (champion, fitness) = evo.best().expect("a generation has run");
//! let network = InterpretedPhenotype::new(&champion[0]);
//! ```
//!
//! # Orientation and negative fitness
//!
//! NEAT is **maximization** (higher fitness is better), matching the crate-wide
//! maximise convention (canonical space; see
//! [`ObjectiveSense`]). The loop negates a `Minimize` objective before
//! [`tell`](Strategy::tell), so a task never hand-negates here. Species are
//! allotted offspring in proportion to their mean fitness, which needs values
//! that are not negative: when a generation's smallest finite fitness is
//! negative, every fitness is shifted by it before the shares are taken (see
//! [`speciate`](crate::neuroevolution::species::speciate)). A generation with no
//! negative fitness is shared as it stands.
//!
//! # Determinism and host RNG
//!
//! Every stochastic decision derives from a `seed_stream(rng.next_u64(),
//! generation, SeedPurpose::…)` host-side `StdRng` substream — the crate-wide
//! host-RNG convention. Combined with the per-run [`InnovationRegistry`]'s
//! caches, the same
//! seed + same mutation sequence yields identical innovation *and* node ids.
//! Never `B::seed_from_u64` + `Tensor::random` (process-wide backend RNG mutex
//! races parallel tests).
//!
//! # Gradient isolation
//!
//! The phenotype is a forward-only bare-tensor evaluator; no autodiff graph is
//! ever built.
//!
//! [`EvolutionaryLoop`]: crate::strategy::EvolutionaryLoop

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use burn::tensor::{Device, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::{Rng, RngExt};
use rand_distr::{Distribution as _, Normal};

use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::objective::ObjectiveSense;

use crate::fitness::BatchFitnessFn;
use crate::neuroevolution::innovation::InnovationRegistry;
use crate::neuroevolution::phenotype::DensePaddedEvaluator;
use crate::neuroevolution::species::{self, Species, SpeciesId};
use crate::neuroevolution::topology::{
    ActivationFn, ConnectionGene, NodeGene, NodeId, NodeKind, TopologyGenome,
};
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for a NEAT run.
///
/// Build the canonical defaults with [`NeatParams::default_for`]; the
/// compatibility threshold and the structural-mutation rates (`p_add_node`,
/// `p_add_connection`) are the knobs most worth tuning per task.
/// [`EvolutionaryLoop::new`](crate::strategy::EvolutionaryLoop::new) checks
/// them through the [`Validate`] impl.
#[derive(Debug, Clone)]
pub struct NeatParams {
    /// Number of individuals per generation.
    pub pop_size: usize,
    /// Number of input (sensor) nodes.
    pub num_inputs: usize,
    /// Number of output nodes.
    pub num_outputs: usize,
    /// Excess-gene coefficient `c1` in the compatibility distance.
    pub c1: f32,
    /// Disjoint-gene coefficient `c2` in the compatibility distance.
    pub c2: f32,
    /// Weight-difference coefficient `c3` in the compatibility distance.
    pub c3: f32,
    /// Compatibility-distance threshold below which a genome joins a species.
    pub compat_threshold: f32,
    /// Generations of no best-fitness improvement before a species is culled.
    pub stagnation_limit: u64,
    /// Fraction of each species (by fitness) eligible to reproduce.
    pub survival_threshold: f32,
    /// Species with strictly **more** members than this copy their champion
    /// unchanged each generation (canonical NEAT: species larger than ~5).
    pub elitism_min_species_size: usize,
    /// Per-genome probability that weight mutation occurs at all.
    pub p_mutate_weight: f32,
    /// Standard deviation of the Gaussian weight/bias perturbation.
    pub weight_perturb_std: f32,
    /// Per-gene probability a weight is replaced (vs. perturbed) when mutating.
    pub p_weight_replace: f32,
    /// Per-genome probability of an add-connection mutation.
    pub p_add_connection: f32,
    /// Per-genome probability of an add-node mutation.
    pub p_add_node: f32,
    /// Per-genome probability of an enable/disable toggle mutation.
    pub p_toggle_enable: f32,
    /// Probability a child gene is disabled when disabled in either parent.
    pub p_disable_inherited: f32,
    /// Probability a crossover draws its second parent from another species.
    pub interspecies_mating_rate: f32,
    /// Fraction of offspring produced by mutation only (no crossover).
    pub mutate_only_fraction: f32,
    /// Standard deviation used to initialize (and replace) weights and biases.
    pub weight_init_std: f32,
}

impl NeatParams {
    /// The canonical NEAT defaults (Stanley & Miikkulainen, 2002) for the given
    /// population and I/O sizes.
    #[must_use]
    pub fn default_for(pop_size: usize, num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            pop_size,
            num_inputs,
            num_outputs,
            c1: 1.0,
            c2: 1.0,
            c3: 0.4,
            compat_threshold: 3.0,
            stagnation_limit: 15,
            survival_threshold: 0.2,
            elitism_min_species_size: 5,
            p_mutate_weight: 0.8,
            weight_perturb_std: 0.5,
            p_weight_replace: 0.1,
            p_add_connection: 0.05,
            p_add_node: 0.03,
            p_toggle_enable: 0.01,
            p_disable_inherited: 0.75,
            interspecies_mating_rate: 0.001,
            mutate_only_fraction: 0.25,
            weight_init_std: 1.0,
        }
    }
}

impl Validate for NeatParams {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "NeatParams";
        config::at_least(C, "pop_size", self.pop_size, 1)?;
        config::at_least(C, "num_inputs", self.num_inputs, 1)?;
        config::at_least(C, "num_outputs", self.num_outputs, 1)?;
        let non_negative = [
            ("c1", self.c1),
            ("c2", self.c2),
            ("c3", self.c3),
            ("compat_threshold", self.compat_threshold),
            ("weight_perturb_std", self.weight_perturb_std),
            ("weight_init_std", self.weight_init_std),
        ];
        for (field, value) in non_negative {
            config::in_range(C, field, 0.0, f64::INFINITY, f64::from(value))?;
        }
        let probabilities = [
            ("survival_threshold", self.survival_threshold),
            ("p_mutate_weight", self.p_mutate_weight),
            ("p_weight_replace", self.p_weight_replace),
            ("p_add_connection", self.p_add_connection),
            ("p_add_node", self.p_add_node),
            ("p_toggle_enable", self.p_toggle_enable),
            ("p_disable_inherited", self.p_disable_inherited),
            ("interspecies_mating_rate", self.interspecies_mating_rate),
            ("mutate_only_fraction", self.mutate_only_fraction),
        ];
        for (field, value) in probabilities {
            config::in_range(C, field, 0.0, 1.0, f64::from(value))?;
        }
        Ok(())
    }
}

/// Generation-to-generation NEAT state.
///
/// `#[derive(Clone)]` is cheap: all fields are host-side data, and the
/// `registry` field clones the shared `Arc` handle (the per-run registry itself
/// is not duplicated). A clone therefore shares the registry with the state it
/// was cloned from: it is not an independent snapshot of the run, because
/// [`ask`](Strategy::ask) allocates innovation and node ids in that registry.
/// A run started by [`init`](Strategy::init) gets a registry of its own.
#[derive(Clone, Debug)]
pub struct NeatState {
    /// Current resident population.
    pub population: Vec<TopologyGenome>,
    /// Resident fitness (maximization); empty until the first `tell`.
    pub fitness: Vec<f32>,
    /// Current species partition over `population`.
    pub species: Vec<Species>,
    /// Per-run innovation registry, shared across the run.
    pub registry: Arc<InnovationRegistry>,
    /// Completed-generation counter (number of `tell` calls).
    pub generation: u64,
    /// Next species id to allocate.
    pub next_species_id: SpeciesId,
    /// Best genome seen across all generations, if any.
    pub best: Option<TopologyGenome>,
    /// Best fitness seen across all generations (`$-\infty$` before the first `tell`).
    pub best_fitness: f32,
}

/// NEAT as a [`Strategy`] over `Vec<TopologyGenome>`; the module docs show a
/// run.
#[derive(Debug, Clone, Copy, Default)]
pub struct NeatStrategy {}

impl NeatStrategy {
    /// Build a new (stateless) NEAT strategy.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl Strategy for NeatStrategy {
    type Params = NeatParams;
    type State = NeatState;
    type Genome = Vec<TopologyGenome>;

    /// Build the initial state: a per-run registry plus `pop_size` minimal
    /// genomes (aligned ids, per-individual random weights from one
    /// [`SeedPurpose::Init`] substream).
    ///
    /// `device` is part of the signature for symmetry with the tensor
    /// strategies; NEAT genomes are host-side, so it is unused here.
    ///
    /// # Panics
    ///
    /// Panics if `weight_init_std` is non-finite (`$+\infty$` or `NaN`), which
    /// [`NeatParams`]' [`Validate`] impl rejects before a loop runs.
    fn init(&self, params: &NeatParams, rng: &mut dyn Rng, device: &Device) -> NeatState {
        let _ = device;
        let registry = Arc::new(InnovationRegistry::new(
            params.num_inputs + params.num_outputs,
            params.num_inputs * params.num_outputs,
        ));
        let mut init_rng = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let population: Vec<TopologyGenome> = (0..params.pop_size)
            .map(|_| {
                TopologyGenome::minimal(
                    params.num_inputs,
                    params.num_outputs,
                    &registry,
                    &mut init_rng,
                    params.weight_init_std,
                )
            })
            .collect();

        NeatState {
            population,
            fitness: Vec::new(),
            species: Vec::new(),
            registry,
            generation: 0,
            next_species_id: SpeciesId::new(0),
            best: None,
            best_fitness: f32::NEG_INFINITY,
        }
    }

    /// Propose the next population.
    ///
    /// Before the first [`tell`](Self::tell) (resident fitness still empty), the
    /// unchanged resident population is returned for evaluation. Afterwards the
    /// already-speciated residents drive reproduction: stagnant species are
    /// removed (top-K protected), offspring are apportioned by size-adjusted
    /// fitness (largest-remainder, summing exactly to `pop_size`), and each
    /// species contributes its champion unchanged (if large enough) plus
    /// offspring from intra-species crossover (rare interspecies mating) or
    /// mutation-only, all followed by the four mutation operators. Four `StdRng`
    /// substreams (selection, crossover, mutation, misc) keep operator RNG
    /// independent.
    ///
    /// `device` is unused: NEAT genomes are host-side.
    fn ask(
        &self,
        params: &NeatParams,
        state: &NeatState,
        rng: &mut dyn Rng,
        _device: &Device,
    ) -> (Vec<TopologyGenome>, NeatState) {
        // First call: the seed population has not been evaluated yet.
        if state.fitness.is_empty() {
            return (state.population.clone(), state.clone());
        }

        let generation = state.generation;
        let mut next = state.clone();
        species::remove_stagnant(&mut next.species, generation, params.stagnation_limit);
        let counts = species::allocate_offspring(&next.species, params.pop_size);

        let mut rngs = ReproRngs {
            selection: seed_stream(rng.next_u64(), generation, SeedPurpose::Selection),
            crossover: seed_stream(rng.next_u64(), generation, SeedPurpose::Crossover),
            mutation: seed_stream(rng.next_u64(), generation, SeedPurpose::Mutation),
            misc: seed_stream(rng.next_u64(), generation, SeedPurpose::Other),
        };

        let mut offspring: Vec<TopologyGenome> = Vec::with_capacity(params.pop_size);
        {
            let ctx = ReproContext {
                params,
                population: &state.population,
                fitness: &state.fitness,
                species: &next.species,
                registry: &state.registry,
            };
            for (si, &count) in counts.iter().enumerate() {
                produce_offspring(&ctx, &mut rngs, si, count, &mut offspring);
            }
        }

        // Defensive: apportionment + reproduction already total `pop_size`; this
        // only guards against a degenerate empty-species edge.
        while offspring.len() < params.pop_size {
            let idx = rngs.selection.random_range(0..state.population.len());
            let mut child = state.population[idx].clone();
            apply_mutations(&mut child, params, &state.registry, &mut rngs.mutation);
            offspring.push(child);
        }
        offspring.truncate(params.pop_size);

        (offspring, next)
    }

    /// Install the evaluated population and its (maximization) fitness, speciate,
    /// update per-species best/stagnation and the global best-so-far, and bump
    /// the generation.
    ///
    /// Speciation lives here — not in `ask` — because this is the only point
    /// where the new population, its fitness, and the prior species' cloned
    /// representatives all coexist consistently (so member indices stay valid and
    /// each species' next representative is cloned from a live member).
    ///
    /// # Fitness hygiene
    ///
    /// Under [`EvolutionaryLoop`](crate::strategy::EvolutionaryLoop) the
    /// fitness arrives canonical and sanitised. `tell` sanitises it again where
    /// it reaches the host, as [`Strategy::tell`] asks, before it is stored or
    /// handed to [`species::speciate`], so that for a caller who drives the
    /// strategy by hand a non-finite fitness can never poison speciation,
    /// offspring apportionment, or best-so-far tracking (ADR 0034).
    ///
    /// # Panics
    ///
    /// Panics if the `fitness` tensor cannot be read back to the host as `f32`,
    /// or if `population.len()` differs from its length.
    fn tell(
        &self,
        params: &NeatParams,
        population: Vec<TopologyGenome>,
        fitness: Tensor<1>,
        mut state: NeatState,
        rng: &mut dyn Rng,
    ) -> (NeatState, StrategyMetrics) {
        let fitness = fitness
            .into_data()
            .try_into_vec::<f32>()
            .expect("fitness tensor must be readable as f32");
        assert_eq!(
            population.len(),
            fitness.len(),
            "population and fitness must have equal length"
        );
        let generation = state.generation;
        let mut rep_rng = seed_stream(rng.next_u64(), generation, SeedPurpose::Representative);

        state.population = population;
        // Sanitise once at the pull, before the store and `speciate`.
        state.fitness = fitness
            .into_iter()
            .map(crate::fitness::sanitize_fitness)
            .collect();
        species::speciate(
            &state.population,
            &state.fitness,
            &mut state.species,
            params.c1,
            params.c2,
            params.c3,
            params.compat_threshold,
            &mut state.next_species_id,
            generation,
            &mut rep_rng,
        );

        // Sanitize NaN → `$-\infty$` (worst) so a NaN fitness can never become best.
        if let Some((idx, best)) = state
            .fitness
            .iter()
            .enumerate()
            .map(|(i, &f)| (i, crate::fitness::sanitize_fitness(f)))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            && best > state.best_fitness
        {
            state.best_fitness = best;
            state.best = Some(state.population[idx].clone());
        }

        state.generation += 1;
        let metrics = StrategyMetrics::from_host_fitness(
            usize::try_from(state.generation).unwrap_or(usize::MAX),
            &state.fitness,
            state.best_fitness,
        );
        (state, metrics)
    }

    /// Best genome seen across all generations, as a one-genome population,
    /// and its fitness.
    ///
    /// Returns `None` before the first [`tell`](Self::tell).
    fn best(&self, state: &NeatState) -> Option<(Vec<TopologyGenome>, f32)> {
        state
            .best
            .clone()
            .map(|genome| (vec![genome], state.best_fitness))
    }
}

/// A [`BatchFitnessFn`] that scores the whole population in **one** device-resident
/// pass via a [`DensePaddedEvaluator`], instead of building each genome's
/// [`InterpretedPhenotype`](crate::neuroevolution::phenotype::InterpretedPhenotype)
/// in turn.
///
/// It is a drop-in alternative to an interpreted fitness, so the strategy stays
/// evaluation-agnostic and `NeatStrategy`'s determinism is untouched.
///
/// The `reducer` maps one genome's `$\text{batch} \times \text{action\_dim}$` output slab (row-major,
/// as produced by [`DensePaddedEvaluator::evaluate_population`]) to a single
/// fitness in the declared `sense` — the batched analogue of the per-genome
/// scoring an interpreted fitness does on `phenotype.forward(...)`.
pub struct BatchGraphFitness {
    evaluator: DensePaddedEvaluator,
    obs: Tensor<2>,
    sense: ObjectiveSense,
    #[allow(clippy::type_complexity)]
    reducer: Box<dyn Fn(&[f32]) -> f32 + Send + Sync>,
}

impl BatchGraphFitness {
    /// Build a batched fitness from an evaluator, a shared `[batch, obs_dim]`
    /// observation tensor, the direction of the `reducer`'s scores, and a
    /// per-genome `reducer` over the
    /// `$\text{batch} \times \text{action\_dim}$` output slab (row-major).
    pub fn new(
        evaluator: DensePaddedEvaluator,
        obs: Tensor<2>,
        sense: ObjectiveSense,
        reducer: impl Fn(&[f32]) -> f32 + Send + Sync + 'static,
    ) -> Self {
        Self {
            evaluator,
            obs,
            sense,
            reducer: Box::new(reducer),
        }
    }
}

impl std::fmt::Debug for BatchGraphFitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchGraphFitness")
            .field("obs", &self.obs)
            .finish_non_exhaustive()
    }
}

impl BatchFitnessFn<Vec<TopologyGenome>> for BatchGraphFitness {
    fn evaluate_batch(&mut self, population: &Vec<TopologyGenome>, device: &Device) -> Tensor<1> {
        let scores: Vec<f32> = if population.is_empty() {
            Vec::new()
        } else {
            let out = self
                .evaluator
                .evaluate_population(population, self.obs.clone(), device);
            let [pop, batch, action] = out.dims();
            let flat = out
                .into_data()
                .try_into_vec::<f32>()
                .expect("output tensor must be readable as f32");
            let slab = batch * action;
            (0..pop)
                .map(|p| (self.reducer)(&flat[p * slab..(p + 1) * slab]))
                .collect()
        };
        let n = scores.len();
        Tensor::<1>::from_data(TensorData::new(scores, [n]), device)
    }

    fn sense(&self) -> ObjectiveSense {
        self.sense
    }
}

// ---------------------------------------------------------------------------
// Reproduction (private)
// ---------------------------------------------------------------------------

/// Read-only reproduction context shared across all species in one `ask`.
struct ReproContext<'a> {
    params: &'a NeatParams,
    population: &'a [TopologyGenome],
    fitness: &'a [f32],
    species: &'a [Species],
    registry: &'a InnovationRegistry,
}

/// The four independent RNG substreams used during reproduction.
struct ReproRngs {
    selection: StdRng,
    crossover: StdRng,
    mutation: StdRng,
    misc: StdRng,
}

/// Append `count` offspring for `species_idx` to `out`.
fn produce_offspring(
    ctx: &ReproContext<'_>,
    rngs: &mut ReproRngs,
    species_idx: usize,
    count: usize,
    out: &mut Vec<TopologyGenome>,
) {
    if count == 0 {
        return;
    }
    let sp = &ctx.species[species_idx];
    let mut members = sp.members.clone();
    // Sanitize NaN → `$-\infty$` (worst) so it can never rank as best; descending.
    let sane: Vec<f32> = ctx
        .fitness
        .iter()
        .map(|&f| crate::fitness::sanitize_fitness(f))
        .collect();
    members.sort_by(|&a, &b| sane[b].total_cmp(&sane[a]));

    let mut produced = 0usize;
    // Champion elitism: copy the best member unchanged (H5).
    if members.len() > ctx.params.elitism_min_species_size {
        out.push(ctx.population[members[0]].clone());
        produced += 1;
    }

    let n_eligible = eligible_count(members.len(), ctx.params.survival_threshold);
    let eligible = &members[..n_eligible];
    while produced < count {
        let mut child = make_child(ctx, rngs, eligible, species_idx);
        apply_mutations(&mut child, ctx.params, ctx.registry, &mut rngs.mutation);
        out.push(child);
        produced += 1;
    }
}

/// Number of top-fitness members eligible to reproduce (at least one).
fn eligible_count(size: usize, survival_threshold: f32) -> usize {
    // Casts: species sizes are small positive integers.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let raw = (size as f32 * survival_threshold).ceil() as usize;
    raw.max(1).min(size)
}

/// Produce one child: a mutation-only clone or an innovation-aligned crossover.
fn make_child(
    ctx: &ReproContext<'_>,
    rngs: &mut ReproRngs,
    eligible: &[usize],
    species_idx: usize,
) -> TopologyGenome {
    if eligible.len() == 1 || rngs.misc.random::<f32>() < ctx.params.mutate_only_fraction {
        let parent = eligible[rngs.selection.random_range(0..eligible.len())];
        return ctx.population[parent].clone();
    }
    let pa = eligible[rngs.selection.random_range(0..eligible.len())];
    let pb = if ctx.species.len() > 1
        && rngs.misc.random::<f32>() < ctx.params.interspecies_mating_rate
    {
        let other = pick_other_species(ctx.species.len(), species_idx, &mut rngs.selection);
        let other_members = &ctx.species[other].members;
        other_members[rngs.selection.random_range(0..other_members.len())]
    } else {
        eligible[rngs.selection.random_range(0..eligible.len())]
    };
    crossover(
        &ctx.population[pa],
        ctx.fitness[pa],
        &ctx.population[pb],
        ctx.fitness[pb],
        ctx.params,
        &mut rngs.crossover,
    )
}

/// Pick a species index different from `exclude` (caller guarantees `len > 1`).
///
/// Draws from the `len - 1` other indices and skips over `exclude`, so it
/// consumes exactly one RNG value and always terminates (no rejection loop).
fn pick_other_species(len: usize, exclude: usize, rng: &mut StdRng) -> usize {
    let candidate = rng.random_range(0..len - 1);
    if candidate >= exclude {
        candidate + 1
    } else {
        candidate
    }
}

/// Apply the four NEAT mutation operators in sequence, each gated by its rate.
fn apply_mutations(
    genome: &mut TopologyGenome,
    params: &NeatParams,
    registry: &InnovationRegistry,
    rng: &mut StdRng,
) {
    if rng.random::<f32>() < params.p_mutate_weight {
        mutate_weights(genome, params, rng);
    }
    if rng.random::<f32>() < params.p_add_connection {
        mutate_add_connection(genome, params, registry, rng);
    }
    if rng.random::<f32>() < params.p_add_node {
        mutate_add_node(genome, registry, rng);
    }
    if rng.random::<f32>() < params.p_toggle_enable {
        mutate_toggle_enable(genome, rng);
    }
}

// ---------------------------------------------------------------------------
// Mutation operators (private)
// ---------------------------------------------------------------------------

/// Bounded attempts to find a valid (non-duplicate, non-cyclic) node pair for an
/// add-connection mutation before giving up for this generation.
const ADD_CONNECTION_ATTEMPTS: usize = 20;

/// Perturb (or occasionally replace) every connection weight and every
/// non-input node bias. Bias is, functionally, a weight, and mutating it is
/// required for XOR.
///
/// # Panics
///
/// Panics if `weight_perturb_std` or `weight_init_std` is non-finite
/// (`$+\infty$` or `NaN`).
fn mutate_weights(genome: &mut TopologyGenome, params: &NeatParams, rng: &mut StdRng) {
    let perturb = Normal::new(0.0_f32, params.weight_perturb_std).unwrap_or_else(|err| {
        panic!(
            "weight_perturb_std must be finite, got {}: {err}",
            params.weight_perturb_std
        )
    });
    let replace = Normal::new(0.0_f32, params.weight_init_std).unwrap_or_else(|err| {
        panic!(
            "weight_init_std must be finite, got {}: {err}",
            params.weight_init_std
        )
    });

    for conn in &mut genome.connections {
        if rng.random::<f32>() < params.p_weight_replace {
            conn.weight = replace.sample(rng);
        } else {
            conn.weight += perturb.sample(rng);
        }
    }
    for node in &mut genome.nodes {
        if matches!(node.kind, NodeKind::Input) {
            continue;
        }
        if rng.random::<f32>() < params.p_weight_replace {
            node.bias = replace.sample(rng);
        } else {
            node.bias += perturb.sample(rng);
        }
    }
}

/// Add a connection between two currently-unconnected nodes, rejecting any edge
/// that would create a cycle (feedforward invariant, H2). No-op if no valid pair
/// is found within a bounded number of attempts.
///
/// # Panics
///
/// Panics if `weight_init_std` is non-finite (`$+\infty$` or `NaN`).
fn mutate_add_connection(
    genome: &mut TopologyGenome,
    params: &NeatParams,
    registry: &InnovationRegistry,
    rng: &mut StdRng,
) {
    let init = Normal::new(0.0_f32, params.weight_init_std).unwrap_or_else(|err| {
        panic!(
            "weight_init_std must be finite, got {}: {err}",
            params.weight_init_std
        )
    });

    // Sources may be any non-output node; targets any non-input node.
    let sources: Vec<NodeId> = genome
        .nodes
        .iter()
        .filter(|n| !matches!(n.kind, NodeKind::Output))
        .map(|n| n.id)
        .collect();
    let targets: Vec<NodeId> = genome
        .nodes
        .iter()
        .filter(|n| !matches!(n.kind, NodeKind::Input))
        .map(|n| n.id)
        .collect();
    if sources.is_empty() || targets.is_empty() {
        return;
    }

    for _ in 0..ADD_CONNECTION_ATTEMPTS {
        let source = sources[rng.random_range(0..sources.len())];
        let target = targets[rng.random_range(0..targets.len())];
        if source == target || genome.is_connected(source, target) {
            continue;
        }
        if genome.would_create_cycle(source, target) {
            continue;
        }
        let innovation = registry.register_connection(source, target);
        genome.insert_connection_sorted(ConnectionGene {
            innovation,
            source,
            target,
            weight: init.sample(rng),
            enabled: true,
        });
        return;
    }
}

/// Split a random enabled connection with a new hidden node: disable the old
/// edge, add `source -> new` (weight `1.0`) and `new -> target` (the old
/// weight). The unit incoming weight makes the split function-preserving at the
/// instant of mutation, so a new node never disrupts behaviour before its
/// weights are tuned.
fn mutate_add_node(genome: &mut TopologyGenome, registry: &InnovationRegistry, rng: &mut StdRng) {
    let enabled: Vec<usize> = genome
        .connections
        .iter()
        .enumerate()
        .filter(|(_, c)| c.enabled)
        .map(|(i, _)| i)
        .collect();
    if enabled.is_empty() {
        return;
    }
    let idx = enabled[rng.random_range(0..enabled.len())];
    let split_innovation = genome.connections[idx].innovation;
    let split = registry.register_node_split(split_innovation);

    // Guard the toggle-re-enable edge case: if this split's node already
    // materialized in this genome, re-adding it would duplicate the node.
    if genome.node(split.new_node).is_some() {
        return;
    }

    let (source, target, old_weight) = {
        let conn = &mut genome.connections[idx];
        conn.enabled = false;
        (conn.source, conn.target, conn.weight)
    };

    genome.nodes.push(NodeGene {
        id: split.new_node,
        kind: NodeKind::Hidden,
        activation: ActivationFn::Sigmoid,
        bias: 0.0,
    });
    genome.insert_connection_sorted(ConnectionGene {
        innovation: split.in_innov,
        source,
        target: split.new_node,
        weight: 1.0,
        enabled: true,
    });
    genome.insert_connection_sorted(ConnectionGene {
        innovation: split.out_innov,
        source: split.new_node,
        target,
        weight: old_weight,
        enabled: true,
    });
}

/// Flip the enabled bit of a random connection. Always cycle-safe: re-enabling
/// an existing edge keeps the enabled subgraph a subset of the all-edges DAG.
fn mutate_toggle_enable(genome: &mut TopologyGenome, rng: &mut StdRng) {
    if genome.connections.is_empty() {
        return;
    }
    let idx = rng.random_range(0..genome.connections.len());
    genome.connections[idx].enabled = !genome.connections[idx].enabled;
}

// ---------------------------------------------------------------------------
// Crossover (private)
// ---------------------------------------------------------------------------

/// Innovation-aligned crossover.
///
/// Matching genes are inherited from a random parent (disabled with
/// `p_disable_inherited` if disabled in either); disjoint/excess genes from the
/// fitter parent (from both when fitness is equal). Candidate edges are then
/// filtered through a cycle check so a child that combines `A→B` and `B→A` from
/// divergent parents stays feedforward (such drops are rare). Child nodes are the
/// referenced endpoints (preferring the fitter parent) plus all input/output
/// nodes.
fn crossover(
    p1: &TopologyGenome,
    f1: f32,
    p2: &TopologyGenome,
    f2: f32,
    params: &NeatParams,
    rng: &mut StdRng,
) -> TopologyGenome {
    // Relative tolerance so "equal fitness" holds across magnitudes (XOR's 0..4 as well as a
    // deferred consumer's large `$-\text{cost}$` scores).
    let equal = (f1 - f2).abs() <= f32::EPSILON * f1.abs().max(f2.abs()).max(1.0);
    let p1_fitter = f1 > f2;

    let c1 = &p1.connections;
    let c2 = &p2.connections;
    let mut candidates: Vec<ConnectionGene> = Vec::with_capacity(c1.len().max(c2.len()));
    let (mut i, mut j) = (0usize, 0usize);
    while i < c1.len() && j < c2.len() {
        match c1[i].innovation.cmp(&c2[j].innovation) {
            std::cmp::Ordering::Equal => {
                let mut gene = if rng.random::<bool>() {
                    c1[i].clone()
                } else {
                    c2[j].clone()
                };
                let disabled_in_either = !c1[i].enabled || !c2[j].enabled;
                let disable =
                    disabled_in_either && rng.random::<f32>() < params.p_disable_inherited;
                gene.enabled = !disable;
                candidates.push(gene);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                if equal || p1_fitter {
                    candidates.push(c1[i].clone());
                }
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                if equal || !p1_fitter {
                    candidates.push(c2[j].clone());
                }
                j += 1;
            }
        }
    }
    while i < c1.len() {
        if equal || p1_fitter {
            candidates.push(c1[i].clone());
        }
        i += 1;
    }
    while j < c2.len() {
        if equal || !p1_fitter {
            candidates.push(c2[j].clone());
        }
        j += 1;
    }

    // Cycle-safe assembly: keep candidates (innovation order) that do not close a
    // cycle over the all-edges graph built so far.
    let mut probe = TopologyGenome {
        nodes: Vec::new(),
        connections: Vec::new(),
    };
    for gene in candidates {
        if probe.would_create_cycle(gene.source, gene.target) {
            continue;
        }
        probe.connections.push(gene);
    }
    let child_conns = probe.connections;

    // Node genes: prefer the fitter parent; always include all input/output nodes.
    let (primary, secondary) = if p1_fitter || equal {
        (p1, p2)
    } else {
        (p2, p1)
    };
    let mut node_map: HashMap<NodeId, NodeGene> = HashMap::new();
    for node in &primary.nodes {
        if matches!(
            node.kind,
            NodeKind::Input | NodeKind::Output | NodeKind::Bias
        ) {
            node_map.insert(node.id, node.clone());
        }
    }
    let mut referenced: HashSet<NodeId> = HashSet::new();
    for conn in &child_conns {
        referenced.insert(conn.source);
        referenced.insert(conn.target);
    }
    for id in referenced {
        if node_map.contains_key(&id) {
            continue;
        }
        if let Some(node) = primary.node(id).or_else(|| secondary.node(id)) {
            node_map.insert(id, node.clone());
        }
    }
    let mut nodes: Vec<NodeGene> = node_map.into_values().collect();
    nodes.sort_by_key(|n| n.id);

    TopologyGenome {
        nodes,
        connections: child_conns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuroevolution::topology::InnovationId;
    use rand::SeedableRng;

    fn node(id: u64, kind: NodeKind) -> NodeGene {
        NodeGene {
            id: NodeId::new(id),
            kind,
            activation: ActivationFn::Sigmoid,
            bias: 0.0,
        }
    }

    fn conn(innovation: u64, source: u64, target: u64) -> ConnectionGene {
        ConnectionGene {
            innovation: InnovationId::new(innovation),
            source: NodeId::new(source),
            target: NodeId::new(target),
            weight: 0.5,
            enabled: true,
        }
    }

    /// Replaying a fixed mutation script twice (fresh registry + same seed)
    /// yields identical innovation AND node id sequences.
    #[test]
    fn test_innovation_numbering_is_deterministic() {
        fn replay() -> (Vec<u64>, Vec<NodeId>) {
            let registry = InnovationRegistry::new(3, 2);
            let mut rng = StdRng::seed_from_u64(99);
            let params = NeatParams::default_for(10, 2, 1);
            let mut g = TopologyGenome::minimal(2, 1, &registry, &mut rng, 1.0);
            mutate_add_node(&mut g, &registry, &mut rng);
            mutate_add_connection(&mut g, &params, &registry, &mut rng);
            mutate_add_node(&mut g, &registry, &mut rng);
            mutate_weights(&mut g, &params, &mut rng);
            mutate_add_connection(&mut g, &params, &registry, &mut rng);
            let innovs: Vec<u64> = g.connections.iter().map(|c| c.innovation.get()).collect();
            let mut nodes: Vec<NodeId> = g.nodes.iter().map(|n| n.id).collect();
            nodes.sort_unstable();
            (innovs, nodes)
        }
        let (i1, n1) = replay();
        let (i2, n2) = replay();
        assert_eq!(i1, i2, "innovation id sequence must be reproducible");
        assert_eq!(n1, n2, "node id sequence must be reproducible");
    }

    /// Crossover classifies matching / disjoint / excess and inherits
    /// disjoint+excess from the fitter parent only.
    #[test]
    fn test_crossover_inherits_disjoint_excess_from_fitter() {
        // Nodes 0,1 inputs; 2 output; 3 hidden. p1 fitter.
        let nodes_p1 = vec![
            node(0, NodeKind::Input),
            node(1, NodeKind::Input),
            node(2, NodeKind::Output),
            node(3, NodeKind::Hidden),
        ];
        // p1 conns: 0->2 (i0), 1->2 (i1), 0->3 (i2), 3->2 (i3).
        let p1 = TopologyGenome::new(
            nodes_p1,
            vec![conn(0, 0, 2), conn(1, 1, 2), conn(2, 0, 3), conn(3, 3, 2)],
        );
        // p2 (less fit) conns: 0->2 (i0), 1->2 (i1), 1->4 (i4) with hidden 4.
        let nodes_p2 = vec![
            node(0, NodeKind::Input),
            node(1, NodeKind::Input),
            node(2, NodeKind::Output),
            node(4, NodeKind::Hidden),
        ];
        let p2 = TopologyGenome::new(nodes_p2, vec![conn(0, 0, 2), conn(1, 1, 2), conn(4, 1, 4)]);

        let params = NeatParams::default_for(10, 2, 1);
        let mut rng = StdRng::seed_from_u64(3);
        let child = crossover(&p1, 2.0, &p2, 1.0, &params, &mut rng);

        let innovs: Vec<u64> = child
            .connections
            .iter()
            .map(|c| c.innovation.get())
            .collect();
        assert_eq!(
            innovs,
            vec![0, 1, 2, 3],
            "matching (0,1) + fitter disjoint/excess (2,3); less-fit excess (4) dropped"
        );
        assert!(
            child.is_innovation_sorted(),
            "child stays innovation-sorted"
        );
        assert!(
            child.node(NodeId::new(4)).is_none(),
            "node only reachable via dropped gene is excluded"
        );
        assert!(
            child.node(NodeId::new(3)).is_some(),
            "hidden node from inherited gene is kept"
        );
    }

    /// Equal fitness inherits disjoint/excess from both parents.
    #[test]
    fn test_crossover_equal_fitness_takes_from_both() {
        let nodes_p1 = vec![
            node(0, NodeKind::Input),
            node(1, NodeKind::Input),
            node(2, NodeKind::Output),
        ];
        let p1 = TopologyGenome::new(nodes_p1.clone(), vec![conn(0, 0, 2), conn(2, 1, 2)]);
        let p2 = TopologyGenome::new(nodes_p1, vec![conn(0, 0, 2), conn(3, 1, 2)]);
        let params = NeatParams::default_for(10, 2, 1);
        let mut rng = StdRng::seed_from_u64(5);
        let child = crossover(&p1, 1.0, &p2, 1.0, &params, &mut rng);
        let innovs: Vec<u64> = child
            .connections
            .iter()
            .map(|c| c.innovation.get())
            .collect();
        assert_eq!(
            innovs,
            vec![0, 2, 3],
            "equal fitness keeps disjoint/excess from both"
        );
    }

    /// Add-connection never closes a cycle: in a fully-connected feedforward
    /// triangle whose only remaining pair is a back-edge, nothing is added.
    #[test]
    fn test_add_connection_rejects_cycles() {
        // Feedforward: 0(in) -> 2(h) -> 3(h) -> 1(out), fully forward-connected.
        let nodes = vec![
            node(0, NodeKind::Input),
            node(2, NodeKind::Hidden),
            node(3, NodeKind::Hidden),
            node(1, NodeKind::Output),
        ];
        let conns = vec![
            conn(0, 0, 2),
            conn(1, 0, 3),
            conn(2, 0, 1),
            conn(3, 2, 3),
            conn(4, 2, 1),
            conn(5, 3, 1),
        ];
        let mut g = TopologyGenome::new(nodes, conns);
        let before = g.connections.len();
        let registry = InnovationRegistry::new(4, 6);
        let params = NeatParams::default_for(10, 1, 1);
        let mut rng = StdRng::seed_from_u64(7);
        // The only non-duplicate, non-self forward-source/target pair left is the
        // back-edge 3->2, which would cycle; it must never be added.
        for _ in 0..50 {
            mutate_add_connection(&mut g, &params, &registry, &mut rng);
        }
        assert_eq!(g.connections.len(), before, "no cyclic edge is ever added");
        assert!(
            !g.is_connected(NodeId::new(3), NodeId::new(2)),
            "the back-edge 3->2 is rejected"
        );
    }

    /// Add-node splits an enabled connection: disables it, inserts one hidden
    /// node, and adds the two replacement edges (`source → new` weight `1.0`,
    /// `new → target` old weight), staying innovation-sorted.
    #[test]
    fn test_add_node_splits_connection() {
        let registry = InnovationRegistry::new(3, 2);
        let mut rng = StdRng::seed_from_u64(1);
        let mut g = TopologyGenome::minimal(2, 1, &registry, &mut rng, 1.0);
        let nodes_before = g.nodes.len();
        let conns_before = g.connections.len();

        mutate_add_node(&mut g, &registry, &mut rng);

        assert_eq!(g.nodes.len(), nodes_before + 1, "one hidden node inserted");
        assert_eq!(
            g.connections.len(),
            conns_before + 2,
            "split adds two connections"
        );
        assert_eq!(
            g.connections.iter().filter(|c| !c.enabled).count(),
            1,
            "the split connection is disabled"
        );
        let hidden: Vec<&NodeGene> = g
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Hidden))
            .collect();
        assert_eq!(hidden.len(), 1);
        assert_eq!(
            hidden[0].id.get(),
            3,
            "new hidden node id follows the seed nodes"
        );
        assert!(
            g.is_innovation_sorted(),
            "connections stay innovation-sorted"
        );
        assert!(
            g.connections
                .iter()
                .any(|c| c.target == NodeId::new(3) && c.enabled && (c.weight - 1.0).abs() < 1e-6),
            "the source -> new edge has the function-preserving weight 1.0"
        );
    }

    /// Toggle flips exactly one connection's enabled bit.
    #[test]
    fn test_toggle_enable_flips_one_connection() {
        let registry = InnovationRegistry::new(3, 2);
        let mut rng = StdRng::seed_from_u64(2);
        let mut g = TopologyGenome::minimal(2, 1, &registry, &mut rng, 1.0);
        let before: Vec<bool> = g.connections.iter().map(|c| c.enabled).collect();
        mutate_toggle_enable(&mut g, &mut rng);
        let after: Vec<bool> = g.connections.iter().map(|c| c.enabled).collect();
        let flipped = before.iter().zip(&after).filter(|(a, b)| a != b).count();
        assert_eq!(flipped, 1, "exactly one connection's enabled bit flips");
    }

    /// The add-node guard: if a connection's split node already exists in the
    /// genome (e.g. the split edge was re-enabled by a toggle), splitting it
    /// again must not duplicate the node or its edges.
    #[test]
    fn test_add_node_guard_prevents_duplicate_node() {
        let registry = InnovationRegistry::new(3, 2);
        let split = registry.register_node_split(InnovationId::new(0)); // allocates node 3
        // Genome already holds node 3; only innovation 0 (its split edge) is
        // enabled, so add-node is forced to re-select it.
        let nodes = vec![
            node(0, NodeKind::Input),
            node(1, NodeKind::Input),
            node(2, NodeKind::Output),
            node(3, NodeKind::Hidden),
        ];
        let conns = vec![
            ConnectionGene {
                innovation: InnovationId::new(0),
                source: NodeId::new(0),
                target: NodeId::new(2),
                weight: 0.5,
                enabled: true,
            },
            ConnectionGene {
                innovation: split.in_innov,
                source: NodeId::new(0),
                target: NodeId::new(3),
                weight: 1.0,
                enabled: false,
            },
            ConnectionGene {
                innovation: split.out_innov,
                source: NodeId::new(3),
                target: NodeId::new(2),
                weight: 0.5,
                enabled: false,
            },
        ];
        let mut g = TopologyGenome::new(nodes, conns);
        let nodes_before = g.nodes.len();
        let conns_before = g.connections.len();
        let mut rng = StdRng::seed_from_u64(4);

        mutate_add_node(&mut g, &registry, &mut rng);

        assert_eq!(
            g.nodes.len(),
            nodes_before,
            "guard prevents a duplicate split node"
        );
        assert_eq!(
            g.connections.len(),
            conns_before,
            "no duplicate split edges added"
        );
    }

    /// The canonical defaults validate, and each kind of bad value is refused
    /// by name: a zero count, a probability above one, a negative or
    /// non-finite coefficient.
    #[test]
    fn test_params_validate_accepts_defaults_and_rejects_bad_values() {
        assert!(NeatParams::default_for(150, 2, 1).validate().is_ok());

        let refused = |edit: fn(&mut NeatParams)| {
            let mut params = NeatParams::default_for(150, 2, 1);
            edit(&mut params);
            params.validate().unwrap_err().field
        };
        assert_eq!(refused(|p| p.pop_size = 0), "pop_size");
        assert_eq!(refused(|p| p.num_inputs = 0), "num_inputs");
        assert_eq!(refused(|p| p.num_outputs = 0), "num_outputs");
        assert_eq!(refused(|p| p.p_add_node = 1.5), "p_add_node");
        assert_eq!(
            refused(|p| p.survival_threshold = -0.1),
            "survival_threshold"
        );
        assert_eq!(
            refused(|p| p.mutate_only_fraction = f32::NAN),
            "mutate_only_fraction"
        );
        assert_eq!(refused(|p| p.c3 = -1.0), "c3");
        assert_eq!(
            refused(|p| p.compat_threshold = f32::INFINITY),
            "compat_threshold"
        );
        assert_eq!(refused(|p| p.weight_init_std = f32::NAN), "weight_init_std");
    }

    /// `ask`/`tell` run a few generations, preserving `pop_size` and tracking the
    /// best. A trivial maximization fitness rewards more enabled connections.
    #[test]
    fn test_ask_tell_smoke_preserves_pop_size_and_tracks_best() {
        struct ConnCountFitness;
        impl BatchFitnessFn<Vec<TopologyGenome>> for ConnCountFitness {
            fn evaluate_batch(
                &mut self,
                population: &Vec<TopologyGenome>,
                device: &Device,
            ) -> Tensor<1> {
                // Cast: enabled-connection counts are tiny in this smoke test.
                #[allow(clippy::cast_precision_loss)]
                let counts: Vec<f32> = population
                    .iter()
                    .map(|g| g.connections.iter().filter(|c| c.enabled).count() as f32)
                    .collect();
                let n = counts.len();
                Tensor::<1>::from_data(TensorData::new(counts, [n]), device)
            }

            fn sense(&self) -> ObjectiveSense {
                ObjectiveSense::Maximize
            }
        }

        let device = Device::flex();
        let mut params = NeatParams::default_for(24, 2, 1);
        params.p_add_node = 0.3;
        params.p_add_connection = 0.5;

        let strat = NeatStrategy::new();
        let mut rng = StdRng::seed_from_u64(11);
        let mut state = strat.init(&params, &mut rng, &device);
        let mut fitness_fn = ConnCountFitness;

        for _ in 0..8 {
            let (population, next) = strat.ask(&params, &state, &mut rng, &device);
            assert_eq!(
                population.len(),
                params.pop_size,
                "ask returns pop_size genomes"
            );
            let fitness = fitness_fn.evaluate_batch(&population, &device);
            state = strat.tell(&params, population, fitness, next, &mut rng).0;
            assert!(
                !state.species.is_empty(),
                "speciation always yields >= 1 species"
            );
        }
        assert_eq!(state.generation, 8);
        let (_, best) = strat.best(&state).expect("best exists after tell");
        assert!(best >= 2.0, "best rewards enabled connections; got {best}");
    }

    /// Regression (ADR 0034): `tell` is NEAT's driver chokepoint and
    /// must sanitize non-finite fitness before storing/speciating. A `NaN` member
    /// must never become the champion nor poison per-species bookkeeping, and a
    /// `$+\infty$` member must rank top but as a **finite** value (`f32::MAX`).
    #[test]
    fn test_tell_sanitizes_nan_and_inf_fitness() {
        let device = Device::flex();
        let params = NeatParams::default_for(4, 2, 1);
        let strat = NeatStrategy::new();
        let mut rng = StdRng::seed_from_u64(7);
        let state = strat.init(&params, &mut rng, &device);

        let (population, next) = strat.ask(&params, &state, &mut rng, &device);
        let n = population.len();
        // Member 0 is NaN (must NOT champion), member 1 is `$+\infty$` (top but finite), the rest
        // are a plain finite value.
        let mut fitness: Vec<f32> = vec![1.0_f32; n];
        fitness[0] = f32::NAN;
        fitness[1] = f32::INFINITY;
        let fitness = Tensor::<1>::from_data(TensorData::new(fitness, [n]), &device);
        let (state, metrics) = strat.tell(&params, population, fitness, next, &mut rng);
        assert_eq!(
            metrics.broken_count(),
            1,
            "the NaN member is counted as broken"
        );
        assert_eq!(
            metrics.generation(),
            1,
            "the first tell completes generation 1"
        );

        // Stored fitness is sanitized: none is NaN; NaN → `$-\infty$`, `$+\infty$` → f32::MAX.
        assert!(
            state.fitness.iter().all(|f| !f.is_nan()),
            "no stored fitness is NaN after the tell chokepoint"
        );
        assert!(
            state.fitness[0].is_infinite() && state.fitness[0].is_sign_negative(),
            "the NaN member is sanitized to the maximise-space worst sentinel (−∞)"
        );
        approx::assert_relative_eq!(state.fitness[1], f32::MAX);

        // The champion is the sanitized `$+\infty$` = f32::MAX: ranks top but is finite, and is
        // never the NaN member.
        let (_, best) = strat.best(&state).expect("best exists after tell");
        assert!(
            best.is_finite(),
            "champion fitness is finite (never NaN/±∞); got {best}"
        );
        approx::assert_relative_eq!(best, f32::MAX);

        // No species' best or size-adjusted mean is NaN-poisoned.
        for s in &state.species {
            assert!(!s.best_fitness.is_nan(), "species best_fitness is not NaN");
            assert!(
                !s.adjusted_fitness_sum.is_nan(),
                "a NaN member never poisons adjusted_fitness_sum"
            );
        }
    }
}
