//! Differential Evolution.
//!
//! Classical DE over `Tensor<B, 2>` populations with all common
//! mutation/crossover variants enumerated in [`DeVariant`].
//!
//! # Variants
//!
//! | Variant | Mutation formula |
//! |---|---|
//! | [`DeVariant::Rand1Bin`], [`DeVariant::Rand1Exp`] | `v = x_{r1} + F · (x_{r2} − x_{r3})` |
//! | [`DeVariant::Best1Bin`] | `v = x_{best} + F · (x_{r2} − x_{r3})` |
//! | [`DeVariant::CurrentToBest1Bin`] | `v = x_i + F · (x_{best} − x_i) + F · (x_{r1} − x_{r2})` |
//! | [`DeVariant::Rand2Bin`] | `v = x_{r1} + F · (x_{r2} − x_{r3}) + F · (x_{r4} − x_{r5})` |
//!
//! The suffix `Bin`/`Exp` selects between binomial and exponential
//! crossover. All index draws reject repeated and self-referential
//! indices.
//!
//! # Hot path
//!
//! A fused `CubeCL` kernel for trial-vector construction is tracked as
//! follow-up work (see [`crate::ops::kernels`]). Until then this module
//! uses host-sampled indices and composes the update from primitive
//! tensor ops.
//!
//! # Reference
//!
//! - Storn & Price (1997), *Differential Evolution — A Simple and
//!   Efficient Heuristic for Global Optimization over Continuous
//!   Spaces*.

use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::{Rng, RngExt};

use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Mutation + crossover variant for differential evolution.
///
/// # Convergence caveats
///
/// Not every variant converges to machine precision on every landscape
/// within the same budget. On unimodal landscapes like Sphere,
/// [`Best1Bin`](DeVariant::Best1Bin) and
/// [`CurrentToBest1Bin`](DeVariant::CurrentToBest1Bin) tend to
/// **converge prematurely**: the population collapses around the
/// current best before the differential search has fully explored, and
/// the per-generation variance `F · (x_{r2} − x_{r3})` shrinks to zero.
/// Classical DE literature documents this as the core trade-off of
/// best-biased variants. The crate's integration tests therefore only
/// require strong *reduction* from the random baseline for those
/// variants, not optimality — see
/// `algorithms::de::tests::all_variants_converge_on_sphere_d10` for the
/// per-variant tolerance choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeVariant {
    /// `x_{r1} + F · (x_{r2} − x_{r3})`, binomial crossover. Balanced
    /// exploration / exploitation; reaches machine precision on Sphere
    /// within a few hundred generations.
    Rand1Bin,
    /// `x_{best} + F · (x_{r2} − x_{r3})`, binomial crossover.
    ///
    /// Strong exploitation — the mutation base is always the current
    /// best, so the population concentrates quickly. Prone to
    /// **premature convergence** on landscapes where the current best
    /// is far from the global optimum; on Sphere-D10 with 500 gens this
    /// variant stalls around `best_fitness ≈ 1` while `Rand1Bin` reaches
    /// `< 1e-20`.
    Best1Bin,
    /// `x_i + F · (x_{best} − x_i) + F · (x_{r1} − x_{r2})`, binomial.
    ///
    /// Hybrid of the current individual and the best-so-far. Still
    /// **prone to premature convergence** because the
    /// `F · (x_{best} − x_i)` term dominates once the population is
    /// near the best. Useful on multimodal landscapes where pure-best
    /// variants get stuck in local basins, less useful on Sphere.
    CurrentToBest1Bin,
    /// `x_{r1} + F · (x_{r2} − x_{r3}) + F · (x_{r4} − x_{r5})`,
    /// binomial. Higher variance than `Rand1Bin` thanks to two
    /// difference vectors; converges on Sphere but more slowly.
    Rand2Bin,
    /// `x_{r1} + F · (x_{r2} − x_{r3})`, exponential crossover.
    /// Identical mutation to `Rand1Bin`, different crossover mask shape.
    /// Performance comparable to `Rand1Bin` in practice.
    Rand1Exp,
}

impl DeVariant {
    /// Number of distinct random indices the variant needs (in
    /// addition to the current individual `i`).
    const fn random_indices(self) -> usize {
        match self {
            DeVariant::Rand1Bin | DeVariant::Rand1Exp => 3,
            DeVariant::Best1Bin | DeVariant::CurrentToBest1Bin => 2,
            DeVariant::Rand2Bin => 5,
        }
    }

    /// Whether this variant uses exponential crossover.
    const fn is_exponential(self) -> bool {
        matches!(self, DeVariant::Rand1Exp)
    }
}

/// Static configuration for a [`DifferentialEvolution`] run.
#[derive(Debug, Clone)]
pub struct DeConfig {
    /// Population size (≥ 5 for `Rand2Bin`, ≥ 4 otherwise).
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds (initialization and clamping).
    pub bounds: (f32, f32),
    /// Differential weight (F). Typical range [0.4, 0.9].
    pub f: f32,
    /// Crossover probability (CR). Typical range [0.1, 0.9].
    pub cr: f32,
    /// Variant.
    pub variant: DeVariant,
}

impl DeConfig {
    /// Default configuration (`Rand1Bin`, F = 0.5, CR = 0.9) for a given
    /// dimensionality.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: (-5.12, 5.12),
            f: 0.5,
            cr: 0.9,
            variant: DeVariant::Rand1Bin,
        }
    }
}

/// Generation state for [`DifferentialEvolution`].
#[derive(Debug, Clone)]
pub struct DeState<B: Backend> {
    /// Current population, shape `(pop_size, D)`.
    pub population: Tensor<B, 2>,
    /// Current fitness (host-side cache).
    pub fitness: Vec<f32>,
    /// Index of the current best within `population`.
    pub best_index: usize,
    /// Best-so-far genome, shape `(1, D)`.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Classical DE/rand/1/bin (and friends).
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use rlevo_evolution::algorithms::de::{DeConfig, DeVariant, DifferentialEvolution};
///
/// let strategy = DifferentialEvolution::<NdArray>::new();
/// let mut params = DeConfig::default_for(30, 10);
/// params.variant = DeVariant::Rand1Bin;
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct DifferentialEvolution<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> DifferentialEvolution<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    fn sample_initial_population(
        params: &DeConfig,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let (lo, hi) = params.bounds;
        B::seed(device, rng.next_u64());
        Tensor::<B, 2>::random(
            [params.pop_size, params.genome_dim],
            burn::tensor::Distribution::Uniform(f64::from(lo), f64::from(hi)),
            device,
        )
    }

    /// Samples `k` indices from `0..pop_size`, all distinct and all
    /// different from `self_idx`.
    ///
    /// # Panics
    ///
    /// Panics if `pop_size <= k`, since the rejection loop cannot make
    /// progress without enough candidates outside `self_idx`.
    fn sample_distinct_excluding(
        self_idx: usize,
        pop_size: usize,
        k: usize,
        rng: &mut dyn Rng,
    ) -> Vec<usize> {
        assert!(
            pop_size > k,
            "DE: pop_size must exceed the number of distinct indices required"
        );
        let mut chosen = Vec::with_capacity(k);
        while chosen.len() < k {
            let candidate = rng.random_range(0..pop_size);
            if candidate != self_idx && !chosen.contains(&candidate) {
                chosen.push(candidate);
            }
        }
        chosen
    }
}

impl<B: Backend> Strategy<B> for DifferentialEvolution<B>
where
    B::Device: Clone,
{
    type Params = DeConfig;
    type State = DeState<B>;
    type Genome = Tensor<B, 2>;

    fn init(&self, params: &DeConfig, rng: &mut dyn Rng, device: &B::Device) -> DeState<B> {
        let population = Self::sample_initial_population(params, rng, device);
        DeState {
            population,
            fitness: Vec::new(),
            best_index: 0,
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
        }
    }

    #[allow(clippy::too_many_lines, clippy::many_single_char_names)]
    fn ask(
        &self,
        params: &DeConfig,
        state: &DeState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, DeState<B>) {
        // First call: evaluate the initial population.
        if state.fitness.is_empty() {
            return (state.population.clone(), state.clone());
        }

        let DeConfig {
            pop_size,
            genome_dim,
            f,
            cr,
            variant,
            ..
        } = *params;

        let mut trial_rng =
            seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Trial);

        // ------------------------------------------------------------------
        // 1. Build the mutant vector v_i for every i, host-side gathers.
        //    We assemble three index tensors (a, b, c [and d, e for rand2])
        //    and do the arithmetic on-device in one sweep.
        // ------------------------------------------------------------------
        let k = variant.random_indices();
        let mut rand_indices: Vec<Vec<usize>> =
            (0..k).map(|_| Vec::with_capacity(pop_size)).collect();
        for i in 0..pop_size {
            let chosen = Self::sample_distinct_excluding(i, pop_size, k, &mut trial_rng);
            for (j, idx) in chosen.into_iter().enumerate() {
                rand_indices[j].push(idx);
            }
        }

        let gather = |idxs: &[usize]| -> Tensor<B, 2> {
            #[allow(clippy::cast_possible_wrap)]
            let v: Vec<i64> = idxs.iter().map(|&i| i as i64).collect();
            let t = Tensor::<B, 1, Int>::from_data(TensorData::new(v, [pop_size]), device);
            state.population.clone().select(0, t)
        };

        let v = match variant {
            DeVariant::Rand1Bin | DeVariant::Rand1Exp => {
                let a = gather(&rand_indices[0]);
                let b = gather(&rand_indices[1]);
                let c = gather(&rand_indices[2]);
                a + (b - c).mul_scalar(f)
            }
            DeVariant::Best1Bin => {
                #[allow(clippy::single_range_in_vec_init)]
                let best = state
                    .population
                    .clone()
                    .slice([state.best_index..state.best_index + 1])
                    .expand([pop_size, genome_dim]);
                let b = gather(&rand_indices[0]);
                let c = gather(&rand_indices[1]);
                best + (b - c).mul_scalar(f)
            }
            DeVariant::CurrentToBest1Bin => {
                #[allow(clippy::single_range_in_vec_init)]
                let best = state
                    .population
                    .clone()
                    .slice([state.best_index..state.best_index + 1])
                    .expand([pop_size, genome_dim]);
                let current = state.population.clone();
                let a = gather(&rand_indices[0]);
                let b = gather(&rand_indices[1]);
                current.clone() + (best - current).mul_scalar(f) + (a - b).mul_scalar(f)
            }
            DeVariant::Rand2Bin => {
                let a = gather(&rand_indices[0]);
                let b = gather(&rand_indices[1]);
                let c = gather(&rand_indices[2]);
                let d = gather(&rand_indices[3]);
                let e = gather(&rand_indices[4]);
                a + (b - c).mul_scalar(f) + (d - e).mul_scalar(f)
            }
        };

        // ------------------------------------------------------------------
        // 2. Crossover: binomial or exponential. Always preserve at
        //    least one mutant gene per row (j_rand).
        // ------------------------------------------------------------------
        let mut cross_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Crossover,
        );
        let mut cross_mask = vec![false; pop_size * genome_dim];
        if variant.is_exponential() {
            for row in 0..pop_size {
                let start = cross_rng.random_range(0..genome_dim);
                let mut len = 1;
                while len < genome_dim && cross_rng.random::<f32>() < cr {
                    len += 1;
                }
                for k in 0..len {
                    let j = (start + k) % genome_dim;
                    cross_mask[row * genome_dim + j] = true;
                }
            }
        } else {
            for row in 0..pop_size {
                let j_rand = cross_rng.random_range(0..genome_dim);
                for j in 0..genome_dim {
                    if j == j_rand || cross_rng.random::<f32>() < cr {
                        cross_mask[row * genome_dim + j] = true;
                    }
                }
            }
        }
        #[allow(clippy::cast_possible_wrap)]
        let mask_int: Vec<i64> = cross_mask.iter().map(|&b| i64::from(b)).collect();
        let mask_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(mask_int, [pop_size, genome_dim]),
            device,
        );
        let mask_bool = mask_tensor.equal_elem(1);

        // Where cross_mask == 1, take from v; otherwise from state.population.
        let trial = state.population.clone().mask_where(mask_bool, v);
        let (lo, hi) = params.bounds;
        let trial = trial.clamp(lo, hi);

        (trial, state.clone())
    }

    fn tell(
        &self,
        _params: &DeConfig,
        trial: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: DeState<B>,
        _rng: &mut dyn Rng,
    ) -> (DeState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();

        // First `tell`: stash fitness for the initial population.
        if state.fitness.is_empty() {
            state.fitness.clone_from(&fitness_host);
            state.best_index = argmin(&fitness_host);
            state.generation += 1;
            update_best(&mut state, &trial, &fitness_host);
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever;
            state.population = trial;
            return (state, m);
        }

        // Greedy per-slot replacement: trial replaces current iff
        // trial is at least as good.
        let device = trial.device();
        let pop_size = state.fitness.len();
        let mut replace_mask = vec![0i64; pop_size];
        let mut new_fit = state.fitness.clone();
        for i in 0..pop_size {
            if fitness_host[i] <= state.fitness[i] {
                replace_mask[i] = 1;
                new_fit[i] = fitness_host[i];
            }
        }

        let mask_int =
            Tensor::<B, 1, Int>::from_data(TensorData::new(replace_mask, [pop_size]), &device);
        let mask_bool_row = mask_int.equal_elem(1);
        let genome_dim = state.population.shape().dims[1];
        let mask_bool = mask_bool_row
            .unsqueeze_dim::<2>(1)
            .expand([pop_size, genome_dim]);
        let next_pop = state
            .population
            .clone()
            .mask_where(mask_bool, trial.clone());

        state.population = next_pop;
        state.fitness.clone_from(&new_fit);
        state.best_index = argmin(&new_fit);
        state.generation += 1;
        update_best(&mut state, &trial, &fitness_host);
        let m = StrategyMetrics::from_host_fitness(state.generation, &new_fit, state.best_fitness);
        state.best_fitness = m.best_fitness_ever;
        (state, m)
    }

    fn best(&self, state: &DeState<B>) -> Option<(Tensor<B, 2>, f32)> {
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

fn update_best<B: Backend>(state: &mut DeState<B>, pop: &Tensor<B, 2>, fitness: &[f32]) {
    if fitness.is_empty() {
        return;
    }
    let best_idx = argmin(fitness);
    let best_f = fitness[best_idx];
    if best_f < state.best_fitness {
        let device = pop.device();
        #[allow(clippy::cast_possible_wrap)]
        let idx =
            Tensor::<B, 1, Int>::from_data(TensorData::new(vec![best_idx as i64], [1]), &device);
        state.best_genome = Some(pop.clone().select(0, idx));
        state.best_fitness = best_f;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::FromFitnessEvaluable;
    use crate::strategy::EvolutionaryHarness;
    use burn::backend::NdArray;
    use rlevo_core::fitness::FitnessEvaluable;
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

    fn run_de(variant: DeVariant, dim: usize, gens: usize) -> f32 {
        let device = Default::default();
        let mut params = DeConfig::default_for(30, dim);
        params.variant = variant;
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            DifferentialEvolution::<TestBackend>::new(),
            params,
            fitness_fn,
            11,
            device,
            gens,
        );
        harness.reset();
        loop {
            if harness.step(()).done {
                break;
            }
        }
        harness.latest_metrics().unwrap().best_fitness_ever
    }

    /// All five DE variants converge on Sphere-D10 within budget.
    ///
    /// The Burn ndarray backend seeds its RNG through a process-wide
    /// mutex, so separate `#[test]` functions that call `Tensor::random`
    /// race on seeding and produce non-deterministic trajectories. This
    /// single test runs the variants sequentially inside one function
    /// so their seed state is not contended.
    ///
    /// Per-variant tolerance reflects classical characterizations:
    /// `rand1`/`rand2` converge to optimum, `best1` / current-to-best
    /// suffer from premature convergence on unimodal landscapes.
    #[test]
    fn all_variants_converge_on_sphere_d10() {
        let rand1bin = run_de(DeVariant::Rand1Bin, 10, 500);
        assert!(rand1bin < 1e-6, "DE/rand/1/bin best={rand1bin}");

        let rand2bin = run_de(DeVariant::Rand2Bin, 10, 800);
        assert!(rand2bin < 1e-6, "DE/rand/2/bin best={rand2bin}");

        let rand1exp = run_de(DeVariant::Rand1Exp, 10, 500);
        assert!(rand1exp < 1e-6, "DE/rand/1/exp best={rand1exp}");

        let best1bin = run_de(DeVariant::Best1Bin, 10, 500);
        assert!(best1bin < 1.0, "DE/best/1/bin best={best1bin}");

        let c2b = run_de(DeVariant::CurrentToBest1Bin, 10, 500);
        assert!(c2b < 2.0, "DE/current-to-best/1/bin best={c2b}");
    }
}
