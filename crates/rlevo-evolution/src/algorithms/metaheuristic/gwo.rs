//! Grey Wolf Optimizer.
//!
//! Each generation ranks the pack by fitness, promotes the top three
//! wolves to `α`, `β`, `δ`, and updates every wolf toward a weighted
//! average of the three. A linearly decreasing coefficient `a ∈ [2, 0]`
//! drives the exploration-to-exploitation transition.
//!
//! # Update rule
//!
//! For each wolf `i` and each leader `k ∈ {α, β, δ}`:
//!
//! - `A_k = 2·a·r1 − a`, `C_k = 2·r2` with `r1, r2 ∈ U[0, 1]`,
//! - `D_k = |C_k · X_k − X_i|`,
//! - `X_k' = X_k − A_k · D_k`,
//! - `X_i ← (X_α' + X_β' + X_δ') / 3`.
//!
//! # Candor
//!
//! Legacy comparator. Camacho Villalón, Dorigo & Stützle (2020)
//! demonstrate that GWO is algorithmically equivalent to a weighted
//! PSO-like update; it has no novel search mechanism over Kennedy &
//! Eberhart (1995). Ship it for API completeness — users who expect a
//! GWO implementation will find one — but prefer CMA-ES or LSHADE for
//! serious work when those are available.
//!
//! # References
//!
//! - Mirjalili, Mirjalili & Lewis (2014), *Grey Wolf Optimizer*.
//! - Camacho Villalón, Dorigo & Stützle (2020), *Grey Wolf, Firefly and
//!   Bat Algorithms: Three Widespread Algorithms that Do Not Contain
//!   Any Novelty*.

use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, Validate};

use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for [`GreyWolfOptimizer`].
#[derive(Debug, Clone)]
pub struct GwoConfig {
    /// Pack size. The algorithm requires `pop_size ≥ 3`.
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds.
    pub bounds: Bounds,
    /// Budget used to schedule `a = 2·(1 − t/max_generations)`.
    /// The strategy itself is memoryless with respect to the harness's
    /// stopping criterion, so this value simply paces the exploration
    /// coefficient and need not equal the harness budget.
    pub max_generations: usize,
}

impl GwoConfig {
    /// Default configuration for a given population size and genome dimensionality.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: Bounds::new(-5.12, 5.12),
            max_generations: 500,
        }
    }
}

impl Validate for GwoConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "GwoConfig";
        config::at_least(C, "pop_size", self.pop_size, 3)?;
        config::nonzero(C, "genome_dim", self.genome_dim)?;
        config::at_least(C, "max_generations", self.max_generations, 1)?;
        Ok(())
    }
}

/// Generation state for [`GreyWolfOptimizer`].
#[derive(Debug, Clone)]
pub struct GwoState<B: Backend> {
    /// Current pack positions, shape `(pop_size, D)`.
    pub pack: Tensor<B, 2>,
    /// Host-side fitness cache.
    pub fitness: Vec<f32>,
    /// Best-so-far genome, shape `(1, D)` — corresponds to α.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Grey Wolf Optimizer strategy.
///
/// # Panics
///
/// [`Strategy::init`] panics if `params.pop_size < 3`, since the update
/// rule needs three distinct leaders (`α`, `β`, `δ`).
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_evolution::algorithms::metaheuristic::gwo::{GreyWolfOptimizer, GwoConfig};
///
/// let strategy = GreyWolfOptimizer::<Flex>::new();
/// let params = GwoConfig::default_for(32, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct GreyWolfOptimizer<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> GreyWolfOptimizer<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    fn sample_initial(params: &GwoConfig, rng: &mut dyn Rng, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Tensor<B, 2> {
        let (lo, hi): (f32, f32) = params.bounds.into();
        // Host-sample from a deterministic `seed_stream` rather than the
        // process-wide Flex RNG (`B::seed` + `Tensor::random`), whose draws
        // interleave with sibling tests under the parallel runner and are
        // not reproducible across thread schedules.
        let pop = params.pop_size;
        let genome_dim = params.genome_dim;
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let mut rows = Vec::with_capacity(pop * genome_dim);
        for _ in 0..pop * genome_dim {
            rows.push(lo + (hi - lo) * stream.random::<f32>());
        }
        Tensor::<B, 2>::from_data(TensorData::new(rows, [pop, genome_dim]), device)
    }
}

impl<B: Backend> Strategy<B> for GreyWolfOptimizer<B>
where
    B::Device: Clone,
{
    type Params = GwoConfig;
    type State = GwoState<B>;
    type Genome = Tensor<B, 2>;

    /// Samples the initial pack uniformly within [`GwoConfig::bounds`] using
    /// the host-RNG convention and sets the generation counter to zero.
    ///
    /// The `pop_size >= 3` invariant is enforced by [`Validate::validate`] at
    /// the harness chokepoint.
    fn init(&self, params: &GwoConfig, rng: &mut dyn Rng, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> GwoState<B> {
        debug_assert!(params.validate().is_ok(), "invalid GwoConfig reached init: {params:?}");
        let pack = Self::sample_initial(params, rng, device);
        GwoState {
            pack,
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
            generation: 0,
        }
    }

    /// Proposes the next pack positions.
    ///
    /// On the first call (before any [`tell`](Strategy::tell)), returns the
    /// initial pack unchanged so it can be evaluated before the first rank
    /// step. On subsequent calls, promotes the three highest-fitness wolves to
    /// `α`, `β`, `δ` and computes the weighted-average update, then clamps
    /// the result to [`GwoConfig::bounds`].
    fn ask(
        &self,
        params: &GwoConfig,
        state: &GwoState<B>,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Tensor<B, 2>, GwoState<B>) {
        // First call: evaluate initial pack so `tell` can rank it.
        if state.fitness.is_empty() {
            return (state.pack.clone(), state.clone());
        }

        let pop_size = params.pop_size;
        let genome_dim = params.genome_dim;

        // Rank the pack host-side — O(N log N) is noise compared to the
        // device ops below, and sorting on the host keeps us free of
        // backend-specific `argsort` quirks.
        let top3 = argtop3_max(&state.fitness);

        #[allow(clippy::cast_possible_wrap)]
        let idx = Tensor::<B, 1, Int>::from_data(
            TensorData::new(vec![top3[0] as i64, top3[1] as i64, top3[2] as i64], [3]),
            device,
        );
        let leaders = state.pack.clone().select(0, idx); // (3, D)

        // Linearly decrease a from 2 to 0.
        #[allow(clippy::cast_precision_loss)]
        let t = state.generation as f32;
        #[allow(clippy::cast_precision_loss)]
        let max_t = params.max_generations.max(1) as f32;
        let a = 2.0 * (1.0 - (t / max_t).min(1.0));

        let mut update = Tensor::<B, 2>::zeros([pop_size, genome_dim], device);
        // Host-sample r1, r2 ∈ U[0,1) per leader from deterministic
        // `seed_stream`s (distinct purposes) so the draws stay reproducible
        // across thread schedules, rather than racing the global Flex RNG.
        #[allow(clippy::cast_sign_loss)]
        for k in 0..3 {
            let gen_k = state.generation as u64 * 3 + k as u64;
            let r1 = {
                let mut s = seed_stream(rng.next_u64(), gen_k, SeedPurpose::Other);
                let mut rows = Vec::with_capacity(pop_size * genome_dim);
                for _ in 0..pop_size * genome_dim {
                    rows.push(s.random::<f32>());
                }
                Tensor::<B, 2>::from_data(TensorData::new(rows, [pop_size, genome_dim]), device)
            };
            let r2 = {
                let mut s = seed_stream(rng.next_u64(), gen_k, SeedPurpose::Mutation);
                let mut rows = Vec::with_capacity(pop_size * genome_dim);
                for _ in 0..pop_size * genome_dim {
                    rows.push(s.random::<f32>());
                }
                Tensor::<B, 2>::from_data(TensorData::new(rows, [pop_size, genome_dim]), device)
            };
            let a_mat = r1.mul_scalar(2.0 * a).sub_scalar(a);
            let c_mat = r2.mul_scalar(2.0);

            #[allow(clippy::single_range_in_vec_init)]
            let leader_row = leaders.clone().slice([k..k + 1]);
            let leader_exp = leader_row.expand([pop_size, genome_dim]);
            let d_k = (c_mat.mul(leader_exp.clone()) - state.pack.clone()).abs();
            let x_k_prime = leader_exp - a_mat.mul(d_k);
            update = update + x_k_prime;
        }
        let new_pack = update.div_scalar(3.0);
        let (lo, hi): (f32, f32) = params.bounds.into();
        let new_pack = new_pack.clamp(lo, hi);

        let mut next = state.clone();
        next.pack.clone_from(&new_pack);
        (new_pack, next)
    }

    /// Records evaluated fitness, updates best-so-far, and increments the
    /// generation counter.
    ///
    /// Returns the updated [`GwoState`] and a [`StrategyMetrics`] snapshot
    /// for the completed generation.
    fn tell(
        &self,
        _params: &GwoConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: GwoState<B>,
        _rng: &mut dyn Rng,
    ) -> (GwoState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        state.fitness.clone_from(&fitness_host);
        state.pack.clone_from(&population);
        let best_idx = argmax(&fitness_host);
        if fitness_host[best_idx] > state.best_fitness {
            state.best_fitness = fitness_host[best_idx];
            let device = population.device();
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            state.best_genome = Some(population.select(0, idx));
        }
        state.generation += 1;
        let m =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = m.best_fitness_ever;
        (state, m)
    }

    /// Returns the α (best-so-far) genome and its fitness, or `None` before
    /// the first [`tell`](Strategy::tell) call.
    fn best(&self, state: &GwoState<B>) -> Option<(Tensor<B, 2>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Indices of the three largest values in `xs`. Panics if `xs.len() < 3`.
fn argtop3_max(xs: &[f32]) -> [usize; 3] {
    assert!(xs.len() >= 3, "argtop3_max requires at least 3 elements");
    let mut idx = [0usize, 1, 2];
    let mut vals = [xs[0], xs[1], xs[2]];
    // Sort the initial three descending.
    if vals[0] < vals[1] {
        vals.swap(0, 1);
        idx.swap(0, 1);
    }
    if vals[1] < vals[2] {
        vals.swap(1, 2);
        idx.swap(1, 2);
    }
    if vals[0] < vals[1] {
        vals.swap(0, 1);
        idx.swap(0, 1);
    }
    for (i, &v) in xs.iter().enumerate().skip(3) {
        if v > vals[2] {
            vals[2] = v;
            idx[2] = i;
            if vals[1] < vals[2] {
                vals.swap(1, 2);
                idx.swap(1, 2);
            }
            if vals[0] < vals[1] {
                vals.swap(0, 1);
                idx.swap(0, 1);
            }
        }
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::FromFitnessEvaluable;
    use crate::strategy::EvolutionaryHarness;
    use burn::backend::Flex;
    use rlevo_core::fitness::FitnessEvaluable;

    type TestBackend = Flex;

    #[test]
    fn default_config_validates() {
        assert!(GwoConfig::default_for(30, 10).validate().is_ok());
    }

    #[test]
    fn rejects_pop_size_below_three() {
        let mut cfg = GwoConfig::default_for(30, 10);
        cfg.pop_size = 2;
        assert_eq!(cfg.validate().unwrap_err().field, "pop_size");
    }

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
    fn argtop3_max_finds_three_largest() {
        let xs = [5.0, 2.0, 8.0, 1.0, 3.0, 9.0, 0.5];
        let top = argtop3_max(&xs);
        // Values at indices: 5 (9.0), 2 (8.0), 0 (5.0).
        assert_eq!(top, [5, 2, 0]);
    }

    #[test]
    fn gwo_converges_on_sphere_d10() {
        // GWO is a "legacy comparator" per the module-level candor note;
        // it converges strongly on Sphere but does not reach machine
        // precision as quickly as DE/Rand1/bin. Budget 600 gens keeps
        // the test within the acceptance bar (rank-1 acceptable within
        // 2× of the classical baselines).
        let device = Default::default();
        let strategy = GreyWolfOptimizer::<TestBackend>::new();
        let params = GwoConfig::default_for(32, 10);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 11, device, 600,
        ).expect("valid params");
        harness.reset();
        while !harness.step(()).done {}
        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        assert!(best < 1e-3, "GWO D10 best={best}");
    }
}
