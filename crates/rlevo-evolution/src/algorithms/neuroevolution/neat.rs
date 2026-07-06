//! NEAT — topology-evolving neuroevolution as a custom harness.
//!
//! `rlevo` implements NEAT (`NeuroEvolution` of Augmenting Topologies; Stanley &
//! Miikkulainen, 2002). [`NeatStrategy`] grows network topology *and* weights
//! open-endedly from a minimal seed. Like its siblings
//! [`WeightOnly`](super::weight_only::WeightOnly) and
//! [`ArchNasStrategy`](super::arch_nas::ArchNasStrategy), it is a **custom
//! harness** with inherent `init`/`ask`/`tell`/`best` and does **not** implement
//! [`Strategy`](crate::strategy::Strategy). Its genome is host-side graph data
//! ([`TopologyGenome`]) with no tensor representation, so the `Strategy<B>`
//! tensor-genome contract does not fit; the parallel [`GraphFitnessFn`] seam
//! plays the role [`BatchFitnessFn`](crate::fitness::BatchFitnessFn) plays for
//! tensor strategies.
//!
//! # Orientation — maximization
//!
//! NEAT is **maximization** (higher fitness is better), matching the crate-wide
//! maximise convention (canonical space; see
//! [`ObjectiveSense`](rlevo_core::objective::ObjectiveSense)). Fitness sharing
//! additionally assumes **non-negative** raw fitness — a NEAT-specific
//! precondition orthogonal to objective sense. Cost objectives are reconciled
//! into canonical (maximise) space by the harness/adapter chokepoint, so a
//! task never hand-negates here.
//!
//! # Generational loop
//!
//! The caller drives the loop manually, exactly as the `arch_nas` integration
//! test does:
//!
//! ```ignore
//! let strat = NeatStrategy::<B>::new();
//! let mut state = strat.init(&params, &mut rng, &device);
//! let builder = InterpretedBuilder;
//! for _ in 0..generations {
//!     let (population, next) = strat.ask(&params, &state, &mut rng);
//!     let fitness = graph_fitness.evaluate(&population, &builder, &device);
//!     state = strat.tell(&params, population, fitness, next, &mut rng);
//!     if let Some((_, best)) = strat.best(&state) { if best >= target { break; } }
//! }
//! ```
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
//! Generic over `B: Backend`, never `AutodiffBackend`. The phenotype is a
//! forward-only bare-tensor evaluator; no autodiff graph is ever built.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use rand::rngs::StdRng;
use rand::{Rng, RngExt};
use rand_distr::{Distribution as _, Normal};

use crate::neuroevolution::innovation::InnovationRegistry;
use crate::neuroevolution::phenotype::{BatchPhenotypeEvaluator, PhenotypeBuilder};
use crate::neuroevolution::species::{self, Species, SpeciesId};
use crate::neuroevolution::topology::{
    ActivationFn, ConnectionGene, NodeGene, NodeId, NodeKind, TopologyGenome,
};
use crate::rng::{SeedPurpose, seed_stream};

/// Static configuration for a NEAT run.
///
/// Build the canonical defaults with [`NeatParams::default_for`]; the
/// compatibility threshold and the structural-mutation rates (`p_add_node`,
/// `p_add_connection`) are the knobs most worth tuning per task.
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

/// Generation-to-generation NEAT state.
///
/// `#[derive(Clone)]` is cheap and correct: all fields are host-side data, and
/// the `registry` field clones the shared `Arc` handle (the per-run registry
/// itself is not duplicated).
#[derive(Clone, Debug)]
pub struct NeatState {
    /// Current resident population.
    pub population: Vec<TopologyGenome>,
    /// Resident fitness (maximization); empty until the first `tell`.
    pub fitness: Vec<f32>,
    /// Current species partition over `population`.
    pub species: Vec<Species>,
    /// Per-run innovation registry, shared across the harness.
    pub registry: Arc<InnovationRegistry>,
    /// Completed-generation counter (number of `tell` calls).
    pub generation: u64,
    /// Next species id to allocate.
    pub next_species_id: SpeciesId,
    /// Best genome seen across all generations, if any.
    pub best: Option<TopologyGenome>,
    /// Best fitness seen across all generations (`−∞` before the first `tell`).
    pub best_fitness: f32,
}

/// Custom NEAT harness. Does **not** implement
/// [`Strategy`](crate::strategy::Strategy) — see the module docs.
#[derive(Debug, Clone, Copy, Default)]
pub struct NeatStrategy<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> NeatStrategy<B> {
    /// Build a new (stateless) NEAT harness.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    /// Build the initial state: a per-run registry plus `pop_size` minimal
    /// genomes (aligned ids, per-individual random weights from one
    /// [`SeedPurpose::Init`] substream).
    ///
    /// `device` is part of the harness signature for symmetry with the tensor
    /// strategies; NEAT genomes are host-side, so it is unused here.
    ///
    /// # Panics
    ///
    /// Panics if `weight_init_std` is negative (degenerate normal).
    #[must_use]
    pub fn init(
        &self,
        params: &NeatParams,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> NeatState {
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
    #[must_use]
    pub fn ask(
        &self,
        params: &NeatParams,
        state: &NeatState,
        rng: &mut dyn Rng,
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
    /// # Panics
    ///
    /// Panics if `population.len()` differs from `fitness.len()`.
    #[must_use]
    pub fn tell(
        &self,
        params: &NeatParams,
        population: Vec<TopologyGenome>,
        fitness: Vec<f32>,
        mut state: NeatState,
        rng: &mut dyn Rng,
    ) -> NeatState {
        assert_eq!(
            population.len(),
            fitness.len(),
            "population and fitness must have equal length"
        );
        let generation = state.generation;
        let mut rep_rng = seed_stream(rng.next_u64(), generation, SeedPurpose::Representative);

        state.population = population;
        state.fitness = fitness;
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

        // Sanitize NaN → −inf (worst) so a NaN fitness can never become best.
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
        state
    }

    /// Best genome and its fitness seen across all generations.
    ///
    /// Returns `None` before the first [`tell`](Self::tell).
    #[must_use]
    pub fn best<'s>(&self, state: &'s NeatState) -> Option<(&'s TopologyGenome, f32)> {
        state.best.as_ref().map(|g| (g, state.best_fitness))
    }
}

/// Evaluation seam for graph genomes — the [`BatchFitnessFn`] analogue for NEAT.
///
/// Maximization-oriented (higher is better). The caller drives the generational
/// loop manually, calling `evaluate` between `ask` and `tell`.
///
/// # Example
///
/// ```ignore
/// struct SumOutputs<B: Backend> { inputs: Tensor<B, 2> }
/// impl<B: Backend> GraphFitnessFn<B> for SumOutputs<B> {
///     fn evaluate(&self, pop: &[TopologyGenome], builder: &dyn PhenotypeBuilder<B>,
///                 device: &B::Device) -> Vec<f32> {
///         pop.iter().map(|g| {
///             let net = builder.build(g, device);
///             let out = net.forward(self.inputs.clone());
///             out.into_data().into_vec::<f32>().unwrap().iter().sum() // higher = better
///         }).collect()
///     }
/// }
/// ```
///
/// [`BatchFitnessFn`]: crate::fitness::BatchFitnessFn
pub trait GraphFitnessFn<B: Backend>: Send + Sync {
    /// Score every genome in `population`, returning one **maximization**
    /// fitness per genome (higher is better, matching the crate-wide canonical
    /// convention). Fitness sharing assumes the values are non-negative. The
    /// returned `Vec` has one entry per input genome, in order. Build each
    /// genome's network with `builder` on `device`.
    fn evaluate(
        &self,
        population: &[TopologyGenome],
        builder: &dyn PhenotypeBuilder<B>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Vec<f32>;
}

/// A [`GraphFitnessFn`] that scores the whole population in **one** device-resident
/// pass via a [`BatchPhenotypeEvaluator`], instead of the per-genome interpreted
/// loop.
///
/// It is a drop-in alternative to an interpreted `GraphFitnessFn` in the same
/// manual `init → ask → evaluate → tell` loop: the `builder` argument is ignored
/// (the batched evaluator owns evaluation), so the harness stays
/// evaluation-agnostic and `NeatStrategy`'s determinism is untouched.
///
/// The `reducer` maps one genome's `batch × action_dim` output slab (row-major,
/// as produced by [`BatchPhenotypeEvaluator::evaluate_population`]) to a single
/// **maximization** fitness — the batched analogue of the per-genome scoring an
/// interpreted `GraphFitnessFn` does on `phenotype.forward(...)`.
pub struct BatchGraphFitness<B: Backend, E> {
    evaluator: E,
    obs: Tensor<B, 2>,
    #[allow(clippy::type_complexity)]
    reducer: Box<dyn Fn(&[f32]) -> f32 + Send + Sync>,
}

impl<B: Backend, E> BatchGraphFitness<B, E> {
    /// Build a batched fitness from an evaluator, a shared `[batch, obs_dim]`
    /// observation tensor, and a per-genome `reducer` over the
    /// `batch × action_dim` output slab (row-major).
    pub fn new(
        evaluator: E,
        obs: Tensor<B, 2>,
        reducer: impl Fn(&[f32]) -> f32 + Send + Sync + 'static,
    ) -> Self {
        Self {
            evaluator,
            obs,
            reducer: Box::new(reducer),
        }
    }
}

impl<B: Backend, E> std::fmt::Debug for BatchGraphFitness<B, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchGraphFitness")
            .field("obs", &self.obs)
            .finish_non_exhaustive()
    }
}

impl<B: Backend, E: BatchPhenotypeEvaluator<B>> GraphFitnessFn<B> for BatchGraphFitness<B, E> {
    fn evaluate(
        &self,
        population: &[TopologyGenome],
        _builder: &dyn PhenotypeBuilder<B>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Vec<f32> {
        if population.is_empty() {
            return Vec::new();
        }
        let out = self
            .evaluator
            .evaluate_population(population, self.obs.clone(), device);
        let [pop, batch, action] = out.dims();
        let flat = out.into_data().into_vec::<f32>().unwrap();
        let slab = batch * action;
        (0..pop)
            .map(|p| (self.reducer)(&flat[p * slab..(p + 1) * slab]))
            .collect()
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
    // Sanitize NaN → −inf (worst) so it can never rank as best; descending.
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
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
/// Panics if `weight_perturb_std` or `weight_init_std` is negative.
fn mutate_weights(genome: &mut TopologyGenome, params: &NeatParams, rng: &mut StdRng) {
    let perturb = Normal::new(0.0_f32, params.weight_perturb_std)
        .expect("weight_perturb_std must be finite and non-negative");
    let replace = Normal::new(0.0_f32, params.weight_init_std)
        .expect("weight_init_std must be finite and non-negative");

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
/// Panics if `weight_init_std` is negative.
fn mutate_add_connection(
    genome: &mut TopologyGenome,
    params: &NeatParams,
    registry: &InnovationRegistry,
    rng: &mut StdRng,
) {
    let init = Normal::new(0.0_f32, params.weight_init_std)
        .expect("weight_init_std must be finite and non-negative");

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
    // Relative tolerance so "equal fitness" holds across magnitudes (XOR's
    // 0..4 as well as a deferred consumer's large `−cost` scores).
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
    let (primary, secondary) = if p1_fitter || equal { (p1, p2) } else { (p2, p1) };
    let mut node_map: HashMap<NodeId, NodeGene> = HashMap::new();
    for node in &primary.nodes {
        if matches!(node.kind, NodeKind::Input | NodeKind::Output | NodeKind::Bias) {
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

        let innovs: Vec<u64> = child.connections.iter().map(|c| c.innovation.get()).collect();
        assert_eq!(
            innovs,
            vec![0, 1, 2, 3],
            "matching (0,1) + fitter disjoint/excess (2,3); less-fit excess (4) dropped"
        );
        assert!(child.is_innovation_sorted(), "child stays innovation-sorted");
        assert!(child.node(NodeId::new(4)).is_none(), "node only reachable via dropped gene is excluded");
        assert!(child.node(NodeId::new(3)).is_some(), "hidden node from inherited gene is kept");
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
        let innovs: Vec<u64> = child.connections.iter().map(|c| c.innovation.get()).collect();
        assert_eq!(innovs, vec![0, 2, 3], "equal fitness keeps disjoint/excess from both");
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
        assert!(!g.is_connected(NodeId::new(3), NodeId::new(2)), "the back-edge 3->2 is rejected");
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
        assert_eq!(g.connections.len(), conns_before + 2, "split adds two connections");
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
        assert_eq!(hidden[0].id.get(), 3, "new hidden node id follows the seed nodes");
        assert!(g.is_innovation_sorted(), "connections stay innovation-sorted");
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
            ConnectionGene { innovation: InnovationId::new(0), source: NodeId::new(0), target: NodeId::new(2), weight: 0.5, enabled: true },
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

        assert_eq!(g.nodes.len(), nodes_before, "guard prevents a duplicate split node");
        assert_eq!(g.connections.len(), conns_before, "no duplicate split edges added");
    }

    /// `ask`/`tell` run a few generations, preserving `pop_size` and tracking the
    /// best. A trivial maximization fitness rewards more enabled connections.
    #[test]
    fn test_ask_tell_smoke_preserves_pop_size_and_tracks_best() {
        use burn::backend::Flex;
        type TestBackend = Flex;

        struct ConnCountFitness;
        impl GraphFitnessFn<TestBackend> for ConnCountFitness {
            fn evaluate(
                &self,
                population: &[TopologyGenome],
                _builder: &dyn PhenotypeBuilder<TestBackend>,
                _device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
            ) -> Vec<f32> {
                // Cast: enabled-connection counts are tiny in this smoke test.
                #[allow(clippy::cast_precision_loss)]
                population
                    .iter()
                    .map(|g| g.connections.iter().filter(|c| c.enabled).count() as f32)
                    .collect()
            }
        }

        let device = Default::default();
        let mut params = NeatParams::default_for(24, 2, 1);
        params.p_add_node = 0.3;
        params.p_add_connection = 0.5;

        let strat = NeatStrategy::<TestBackend>::new();
        let mut rng = StdRng::seed_from_u64(11);
        let mut state = strat.init(&params, &mut rng, &device);
        let builder = crate::neuroevolution::phenotype::InterpretedBuilder;
        let fitness_fn = ConnCountFitness;

        for _ in 0..8 {
            let (population, next) = strat.ask(&params, &state, &mut rng);
            assert_eq!(population.len(), params.pop_size, "ask returns pop_size genomes");
            let fitness = fitness_fn.evaluate(&population, &builder, &device);
            state = strat.tell(&params, population, fitness, next, &mut rng);
            assert!(!state.species.is_empty(), "speciation always yields >= 1 species");
        }
        assert_eq!(state.generation, 8);
        let (_, best) = strat.best(&state).expect("best exists after tell");
        assert!(best >= 2.0, "best rewards enabled connections; got {best}");
    }
}
