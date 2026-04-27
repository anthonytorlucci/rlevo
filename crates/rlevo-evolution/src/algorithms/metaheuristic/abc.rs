//! Artificial Bee Colony.
//!
//! Canonical ABC fused into a single `Strategy::ask` / `tell` round per
//! generation. Each generation produces `2 · pop_size` candidate
//! solutions:
//!
//! 1. **Employed phase** (`pop_size` candidates). For every bee `i`, pick
//!    a neighbour `k ≠ i`, pick a random dimension `j`, and perturb:
//!    `v_ij = x_ij + φ·(x_ij − x_kj)` with `φ ∈ U[−1, 1]`.
//! 2. **Onlooker phase** (`pop_size` candidates). Draw a target `t` via
//!    tournament selection (fitness-biased), then perturb exactly as in
//!    the employed phase.
//!
//! `tell` scores the `2N` candidates, greedy-accepts the best
//! improvement per target bee, and increments the target's `trial`
//! counter when no candidate improved it. Scout bees — those with
//! `trial > limit` — are replaced by fresh uniform samples on device.
//!
//! # References
//!
//! - Karaboga (2005), *An idea based on honey bee swarm for numerical
//!   optimization* (Erciyes Univ. Tech. Report TR06).

use std::marker::PhantomData;

use burn::tensor::{Distribution, Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;

use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for [`ArtificialBeeColony`].
#[derive(Debug, Clone)]
pub struct AbcConfig {
    /// Colony size. The algorithm draws `2 · pop_size` candidates per
    /// generation (employed + onlooker).
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds.
    pub bounds: (f32, f32),
    /// Scout trigger. A bee with `trial > limit` is reinitialized.
    /// Karaboga's canonical default is `pop_size · genome_dim / 2`.
    pub limit: usize,
    /// Tournament size for onlooker selection. Canonical ABC uses
    /// roulette (fitness-proportionate); tournament is a GPU-friendly
    /// equivalent that reuses [`crate::ops::selection::tournament_select`].
    pub tournament_size: usize,
}

impl AbcConfig {
    /// Default configuration for a given population size and genome dimensionality.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: (-5.12, 5.12),
            limit: (pop_size * genome_dim) / 2,
            tournament_size: 3,
        }
    }
}

/// Generation state for [`ArtificialBeeColony`].
#[derive(Debug, Clone)]
pub struct AbcState<B: Backend> {
    /// Current colony, shape `(pop_size, D)`.
    pub colony: Tensor<B, 2>,
    /// Host-side fitness cache.
    pub fitness: Vec<f32>,
    /// Per-bee trial counter.
    pub trial: Vec<usize>,
    /// Target-bee mapping recorded by `ask` so `tell` knows which bee
    /// each candidate belongs to. Length `2 · pop_size` after the
    /// first productive `ask`.
    pub target_of_candidate: Vec<usize>,
    /// Best-so-far genome.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Artificial Bee Colony strategy.
///
/// # Panics
///
/// [`Strategy::init`] panics if `params.pop_size < 2`, since the
/// employed-phase neighbour `k ≠ i` cannot be drawn from a colony of
/// one.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use rlevo_evolution::algorithms::metaheuristic::abc::{AbcConfig, ArtificialBeeColony};
///
/// let strategy = ArtificialBeeColony::<NdArray>::new();
/// let params = AbcConfig::default_for(30, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ArtificialBeeColony<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> ArtificialBeeColony<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    fn build_candidates(
        targets: &[usize],
        neighbors: &[usize],
        dims: &[usize],
        phi: &[f32],
        colony: &Tensor<B, 2>,
        pop_size: usize,
        genome_dim: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        // Base = copy of targets' rows (we only modify one dim each).
        #[allow(clippy::cast_possible_wrap)]
        let target_idx: Vec<i64> = targets.iter().map(|&i| i as i64).collect();
        let _ = pop_size; // number of candidates is inferred below
        let n_cand = targets.len();
        let target_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(target_idx, [n_cand]), device);
        let base = colony.clone().select(0, target_tensor);

        // Compute the perturbation for the single selected dim per row.
        #[allow(clippy::cast_possible_wrap)]
        let neighbor_idx: Vec<i64> = neighbors.iter().map(|&i| i as i64).collect();
        let neighbor_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(neighbor_idx, [n_cand]), device);
        let neighbor_rows = colony.clone().select(0, neighbor_tensor);

        // Build a (n_cand, D) mask with `1` at (row, dims[row]).
        let mut mask = vec![0i64; n_cand * genome_dim];
        for (row, &j) in dims.iter().enumerate() {
            mask[row * genome_dim + j] = 1;
        }
        let mask_bool =
            Tensor::<B, 2, Int>::from_data(TensorData::new(mask, [n_cand, genome_dim]), device)
                .equal_elem(1);

        // φ is per-row; broadcast to (n_cand, D).
        let phi_row = Tensor::<B, 1>::from_data(TensorData::new(phi.to_vec(), [n_cand]), device)
            .unsqueeze_dim::<2>(1)
            .expand([n_cand, genome_dim]);
        let delta = phi_row.mul(base.clone() - neighbor_rows);
        let perturbed = base.clone() + delta;
        base.mask_where(mask_bool, perturbed)
    }
}

impl<B: Backend> Strategy<B> for ArtificialBeeColony<B>
where
    B::Device: Clone,
{
    type Params = AbcConfig;
    type State = AbcState<B>;
    type Genome = Tensor<B, 2>;

    fn init(&self, params: &AbcConfig, rng: &mut dyn Rng, device: &B::Device) -> AbcState<B> {
        assert!(params.pop_size >= 2, "ABC requires pop_size >= 2");
        let (lo, hi) = params.bounds;
        B::seed(device, rng.next_u64());
        let colony = Tensor::<B, 2>::random(
            [params.pop_size, params.genome_dim],
            Distribution::Uniform(f64::from(lo), f64::from(hi)),
            device,
        );
        AbcState {
            colony,
            fitness: Vec::new(),
            trial: vec![0; params.pop_size],
            target_of_candidate: Vec::new(),
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &AbcConfig,
        state: &AbcState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, AbcState<B>) {
        if state.fitness.is_empty() {
            return (state.colony.clone(), state.clone());
        }

        let pop = params.pop_size;
        let genome_dim = params.genome_dim;
        let n_cand = 2 * pop;

        let mut stream = seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Other);

        let mut targets = Vec::with_capacity(n_cand);
        let mut neighbors = Vec::with_capacity(n_cand);
        let mut dims = Vec::with_capacity(n_cand);
        let mut phis = Vec::with_capacity(n_cand);

        // Employed phase — every bee is a target exactly once.
        for i in 0..pop {
            targets.push(i);
        }
        // Onlooker phase — tournament selection, fitness-biased.
        for _ in 0..pop {
            let mut best = stream.random_range(0..pop);
            for _ in 1..params.tournament_size {
                let c = stream.random_range(0..pop);
                if state.fitness[c] < state.fitness[best] {
                    best = c;
                }
            }
            targets.push(best);
        }
        // Neighbour + dim + φ for every candidate.
        for &t in &targets {
            let mut k = stream.random_range(0..pop);
            if k == t {
                k = (k + 1) % pop;
            }
            neighbors.push(k);
            dims.push(stream.random_range(0..genome_dim));
            let phi = 2.0 * stream.random::<f32>() - 1.0;
            phis.push(phi);
        }

        let candidates = Self::build_candidates(
            &targets,
            &neighbors,
            &dims,
            &phis,
            &state.colony,
            pop,
            genome_dim,
            device,
        );
        let (lo, hi) = params.bounds;
        let candidates = candidates.clamp(lo, hi);

        let mut next = state.clone();
        next.target_of_candidate = targets;
        (candidates, next)
    }

    fn tell(
        &self,
        params: &AbcConfig,
        candidates: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: AbcState<B>,
        rng: &mut dyn Rng,
    ) -> (AbcState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let device = candidates.device();
        let pop = params.pop_size;
        let genome_dim = params.genome_dim;

        // First tell: population is the initial colony being scored.
        if state.fitness.is_empty() {
            state.fitness = fitness_host.clone();
            let best_idx = argmin(&fitness_host);
            state.best_fitness = fitness_host[best_idx];
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            state.best_genome = Some(candidates.clone().select(0, idx));
            state.colony = candidates;
            state.generation += 1;
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever;
            return (state, m);
        }

        // For every target, find the best improving candidate (if any).
        // `best_per_target[t] = (cand_idx, cand_fit)` when improvement.
        let mut best_per_target: Vec<Option<(usize, f32)>> = vec![None; pop];
        for (cand_idx, &t) in state.target_of_candidate.iter().enumerate() {
            let cand_fit = fitness_host[cand_idx];
            if cand_fit <= state.fitness[t] {
                match best_per_target[t] {
                    None => best_per_target[t] = Some((cand_idx, cand_fit)),
                    Some((_, prev)) if cand_fit < prev => {
                        best_per_target[t] = Some((cand_idx, cand_fit));
                    }
                    _ => {}
                }
            }
        }

        // Apply replacements via gather: we build an index tensor
        // `row_source[i]` that is either `i` (keep current) pointing
        // into `state.colony`, or `pop + cand_idx` pointing into a
        // stacked tensor `[state.colony; candidates]`.
        let stacked = Tensor::cat(vec![state.colony.clone(), candidates.clone()], 0);
        let mut rs: Vec<i64> = (0..pop).map(|i| i as i64).collect();
        let mut new_fitness = state.fitness.clone();
        for t in 0..pop {
            match best_per_target[t] {
                Some((cand_idx, cand_fit)) => {
                    #[allow(clippy::cast_possible_wrap)]
                    {
                        rs[t] = (pop + cand_idx) as i64;
                    }
                    new_fitness[t] = cand_fit;
                    state.trial[t] = 0;
                }
                None => {
                    state.trial[t] += 1;
                }
            }
        }
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(rs, [pop]), &device);
        state.colony = stacked.select(0, idx);
        state.fitness = new_fitness;

        // Scout phase: reinit any bee whose trial exceeded the limit.
        let mut scouts: Vec<usize> = Vec::new();
        for (i, trial) in state.trial.iter_mut().enumerate() {
            if *trial > params.limit {
                scouts.push(i);
                *trial = 0;
            }
        }
        if !scouts.is_empty() {
            let (lo, hi) = params.bounds;
            B::seed(&device, rng.next_u64());
            let fresh = Tensor::<B, 2>::random(
                [scouts.len(), genome_dim],
                Distribution::Uniform(f64::from(lo), f64::from(hi)),
                &device,
            );
            // Overwrite those rows via gather-trick.
            let mut rs2: Vec<i64> = (0..pop).map(|i| i as i64).collect();
            for (k, &scout) in scouts.iter().enumerate() {
                #[allow(clippy::cast_possible_wrap)]
                {
                    rs2[scout] = (pop + k) as i64;
                }
                // Scout fitness is unknown until next generation —
                // carry INF so any candidate improves it.
                state.fitness[scout] = f32::INFINITY;
            }
            let stacked2 = Tensor::cat(vec![state.colony.clone(), fresh], 0);
            let idx2 = Tensor::<B, 1, Int>::from_data(TensorData::new(rs2, [pop]), &device);
            state.colony = stacked2.select(0, idx2);
        }

        // Update best-so-far from the refreshed colony's fitness cache
        // (excluding INF-tagged scouts, which next `ask` evaluates).
        let best_idx = argmin(&state.fitness);
        if state.fitness[best_idx].is_finite() && state.fitness[best_idx] < state.best_fitness {
            state.best_fitness = state.fitness[best_idx];
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            state.best_genome = Some(state.colony.clone().select(0, idx));
        }

        state.generation += 1;
        let m =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = m.best_fitness_ever;
        (state, m)
    }

    fn best(&self, state: &AbcState<B>) -> Option<(Tensor<B, 2>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

fn argmin(xs: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best = f32::INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v < best {
            best = v;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::FromFitnessEvaluable;
    use crate::strategy::EvolutionaryHarness;
    use burn::backend::NdArray;
    use rlevo_benchmarks::agent::FitnessEvaluable;

    type TestBackend = NdArray;

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
    fn abc_converges_on_sphere_d10() {
        let device = Default::default();
        let strategy = ArtificialBeeColony::<TestBackend>::new();
        let params = AbcConfig::default_for(30, 10);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 13, device, 400,
        );
        harness.reset();
        while !harness.step(()).done {}
        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        assert!(best < 1e-4, "ABC D10 best={best}");
    }
}
