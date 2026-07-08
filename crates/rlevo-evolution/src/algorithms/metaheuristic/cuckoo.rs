//! Cuckoo Search via Lévy flights.
//!
//! Each generation every nest proposes a new egg by taking a
//! Lévy-stable step from its current position:
//!
//! - `u ∼ N(0, σ_u²)`, `v ∼ N(0, 1)`,
//! - `step = u / |v|^(1/β)`,
//! - `x'_i = x_i + α · step`,
//!
//! where `σ_u = (Γ(1+β)·sin(π·β/2) / (Γ((1+β)/2)·β·2^((β−1)/2)))^(1/β)`
//! (Mantegna's algorithm, β ≈ 1.5).
//!
//! `tell` greedy-accepts each new egg against its own slot, then
//! abandons the `p_a · N` worst nests and reinitializes them from the
//! search bounds. Abandoned slots carry sentinel `+∞` fitness so the
//! next generation's Lévy proposal always lands.
//!
//! # Numerical parity caveat
//!
//! The fractional power `|v|^(1/β)` is FMA-reorder-sensitive — wgpu
//! reductions can drift ~`1e-3` relative from flex on the same seed.
//! The backend-parity test relaxes tolerance for CS accordingly.
//!
//! # References
//!
//! - Yang & Deb (2009), *Cuckoo Search via Lévy Flights*.
//! - Mantegna (1994), *Fast, accurate algorithm for numerical simulation
//!   of Lévy stable stochastic processes*.

use std::f32::consts::PI;
use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;
use rand_distr::{Distribution as RandDistDist, Normal};

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, Validate};

use super::len_matches_pop;
use crate::ops::selection::argmax_host;
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for [`CuckooSearch`].
#[derive(Debug, Clone)]
pub struct CuckooConfig {
    /// Nest count.
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds.
    pub bounds: Bounds,
    /// Step size scale (`α` in the paper). Canonical `α = 0.01`
    /// multiplied by the search-space width; strategy users should
    /// tune relative to their domain.
    pub alpha: f32,
    /// Lévy index (`β`). Must be in `(0, 2)`; canonical 1.5.
    pub beta: f32,
    /// Nest abandonment probability (`p_a`). Canonical 0.25.
    pub p_a: f32,
}

impl CuckooConfig {
    /// Default configuration for a given population size and genome dimensionality.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: Bounds::new(-5.12, 5.12),
            alpha: 0.05,
            beta: 1.5,
            p_a: 0.25,
        }
    }
}

impl Validate for CuckooConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "CuckooConfig";
        config::at_least(C, "pop_size", self.pop_size, 1)?;
        config::nonzero(C, "genome_dim", self.genome_dim)?;
        config::positive(C, "alpha", f64::from(self.alpha))?;
        // β ∈ (0, 2), open on both ends.
        config::positive(C, "beta", f64::from(self.beta))?;
        config::ordered(C, "beta", f64::from(self.beta), 2.0)?;
        config::in_range(C, "p_a", 0.0, 1.0, f64::from(self.p_a))?;
        Ok(())
    }
}

/// Generation state for [`CuckooSearch`].
#[derive(Debug, Clone)]
pub struct CuckooState<B: Backend> {
    /// Current nests, shape `(pop_size, D)`.
    nests: Tensor<B, 2>,
    /// Host-side fitness cache; `+∞` for abandoned slots.
    fitness: Vec<f32>,
    /// Best-so-far genome.
    best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    best_fitness: f32,
    /// Generation counter.
    generation: usize,
}

impl<B: Backend> CuckooState<B> {
    /// Assembles a nest state, checking the fitness cache matches `pop`.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `nests` has zero rows or if `fitness` is
    /// non-empty with a length other than `pop_size`.
    pub fn try_new(
        nests: Tensor<B, 2>,
        fitness: Vec<f32>,
        best_genome: Option<Tensor<B, 2>>,
        best_fitness: f32,
        generation: usize,
    ) -> Result<Self, ConfigError> {
        let pop = nests.dims()[0];
        config::nonzero("CuckooState", "pop_size", pop)?;
        len_matches_pop("CuckooState", "fitness", pop, fitness.len())?;
        Ok(Self {
            nests,
            fitness,
            best_genome,
            best_fitness,
            generation,
        })
    }

    /// Current nests, shape `(pop_size, D)`.
    #[must_use]
    pub fn nests(&self) -> &Tensor<B, 2> {
        &self.nests
    }

    /// Host-side fitness cache (empty at bootstrap, else `pop_size` long).
    #[must_use]
    pub fn fitness(&self) -> &[f32] {
        &self.fitness
    }

    /// Best-so-far genome, or `None` before the first `tell`.
    #[must_use]
    pub fn best_genome(&self) -> Option<&Tensor<B, 2>> {
        self.best_genome.as_ref()
    }

    /// Best-so-far (canonical, maximise) fitness.
    #[must_use]
    pub fn best_fitness(&self) -> f32 {
        self.best_fitness
    }

    /// Generation counter.
    #[must_use]
    pub fn generation(&self) -> usize {
        self.generation
    }
}

/// Cuckoo Search strategy.
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_evolution::algorithms::metaheuristic::cuckoo::{CuckooConfig, CuckooSearch};
///
/// let strategy = CuckooSearch::<Flex>::new();
/// let params = CuckooConfig::default_for(30, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CuckooSearch<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> CuckooSearch<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    /// Mantegna's `σ_u` for the `u ∼ N(0, σ_u²)` draw.
    fn mantegna_sigma_u(beta: f32) -> f32 {
        // Γ(1 + β) · sin(π·β/2)  /  ( Γ((1+β)/2) · β · 2^((β-1)/2) ) ) ^ (1/β)
        let num = gamma(1.0 + beta) * ((PI * beta) / 2.0).sin();
        let den = gamma(f32::midpoint(1.0, beta)) * beta * 2f32.powf((beta - 1.0) / 2.0);
        (num / den).powf(1.0 / beta)
    }
}

/// Lanczos approximation for `Γ(z)` on positive reals.
///
/// Used host-side by [`CuckooSearch::mantegna_sigma_u`] to evaluate the
/// `σ_u` constant for Mantegna's Lévy-stable sampler. Accurate to `~1e-3`
/// for `z ∈ [0.5, 5]`, which covers the valid range of the Lévy index
/// `β ∈ (0, 2)`.
#[allow(clippy::many_single_char_names)]
fn gamma(z: f32) -> f32 {
    // 5-term Lanczos coefficients (g = 7). Enough for `z ∈ [0.5, 5]`
    // which covers the Lévy-flight parameter range.
    let g = 7.0_f32;
    let p: [f32; 9] = [
        0.999_999_999_999_809_93,
        676.520_4,
        -1_259.139_2,
        771.323_4,
        -176.615_04,
        12.507_343,
        -0.138_571_1,
        9.984_369e-6,
        1.505_632_7e-7,
    ];
    if z < 0.5 {
        return PI / ((PI * z).sin() * gamma(1.0 - z));
    }
    let z = z - 1.0;
    let mut x = p[0];
    for (i, &coef) in p.iter().enumerate().skip(1) {
        #[allow(clippy::cast_precision_loss)]
        let i_f32 = i as f32;
        x += coef / (z + i_f32);
    }
    let t = z + g + 0.5;
    (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

/// One Mantegna Lévy step component `u / |w|^(1/β)`.
///
/// Guards the measure-zero pathological draw: a Normal draw `w == 0` (or
/// any `w` whose `|w|^(1/β)` rounds to `0` or a non-finite value) makes the
/// denominator degenerate. Un-guarded, `0/0` is `NaN` and `x/0` is `±inf` —
/// both survive the downstream bounds clamp and would poison a nest slot
/// forever. A non-finite or zero denominator folds the step to `0.0`
/// (a no-op) so the next draw can move the nest.
///
/// This is the pure host-side core the `ask` Lévy loop is built on; keeping
/// it out of the tensor pipeline makes the guard directly unit-testable with
/// injected pathological `(u, w)` inputs.
fn levy_step(u: f32, w: f32, beta: f32) -> f32 {
    let denom: f32 = w.abs().powf(1.0 / beta);
    if denom.is_finite() && denom > 0.0 {
        u / denom
    } else {
        0.0
    }
}

impl<B: Backend> Strategy<B> for CuckooSearch<B>
where
    B::Device: Clone,
{
    type Params = CuckooConfig;
    type State = CuckooState<B>;
    type Genome = Tensor<B, 2>;

    /// Build the initial nest population by host-sampling `pop_size`
    /// positions uniformly in `[bounds.lo, bounds.hi]`.
    ///
    /// The `fitness` field is left empty so the first [`ask`] → [`tell`]
    /// pair bootstraps the fitness cache before any greedy acceptance or
    /// abandonment logic runs.  Positions are drawn from a deterministic
    /// [`seed_stream`]; the process-wide Flex RNG is never touched.
    ///
    /// [`ask`]: Strategy::ask
    /// [`tell`]: Strategy::tell
    fn init(
        &self,
        params: &CuckooConfig,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> CuckooState<B> {
        debug_assert!(
            params.validate().is_ok(),
            "invalid CuckooConfig reached init: {params:?}"
        );
        let (lo, hi): (f32, f32) = params.bounds.into();
        // Host-sample the initial nests from a deterministic `seed_stream`
        // rather than the process-wide Flex RNG (`B::seed` + `Tensor::random`),
        // whose draws interleave with sibling tests under the parallel runner
        // and are not reproducible across thread schedules.
        let pop = params.pop_size;
        let genome_dim = params.genome_dim;
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let mut nest_rows = Vec::with_capacity(pop * genome_dim);
        for _ in 0..pop * genome_dim {
            nest_rows.push(lo + (hi - lo) * stream.random::<f32>());
        }
        let nests =
            Tensor::<B, 2>::from_data(TensorData::new(nest_rows, [pop, genome_dim]), device);
        CuckooState {
            nests,
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
            generation: 0,
        }
    }

    /// Propose new egg positions via Mantegna's Lévy-stable step.
    ///
    /// On the first call (`state.fitness` is empty) returns the initial
    /// nests unchanged so the caller can evaluate generation zero.
    ///
    /// On subsequent calls, samples `u ∼ N(0, σ_u²)` and `v ∼ N(0, 1)`
    /// host-side from a deterministic [`seed_stream`], then forms
    /// `step = u / |v|^(1/β)` and proposes
    /// `x'_i = x_i + α · step`, clipped to `params.bounds`.
    fn ask(
        &self,
        params: &CuckooConfig,
        state: &CuckooState<B>,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Tensor<B, 2>, CuckooState<B>) {
        if state.fitness.is_empty() {
            return (state.nests.clone(), state.clone());
        }

        let pop = params.pop_size;
        let d = params.genome_dim;
        let sigma_u = Self::mantegna_sigma_u(params.beta);

        let mut stream = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Mutation,
        );
        let normal_u = Normal::new(0.0_f32, sigma_u).expect("σ_u > 0");
        let normal_v = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let mut step = vec![0f32; pop * d];
        for v in &mut step {
            let u: f32 = normal_u.sample(&mut stream);
            let w: f32 = normal_v.sample(&mut stream);
            // `levy_step` guards the degenerate `w == 0` denominator (±∞/NaN
            // survive the bounds clamp and would poison the slot forever).
            *v = levy_step(u, w, params.beta);
        }
        let step_tensor = Tensor::<B, 2>::from_data(TensorData::new(step, [pop, d]), device);

        let (lo, hi): (f32, f32) = params.bounds.into();
        let new_nests = (state.nests.clone() + step_tensor.mul_scalar(params.alpha)).clamp(lo, hi);

        let mut next = state.clone();
        next.nests.clone_from(&new_nests);
        (new_nests, next)
    }

    /// Ingest egg fitness values, apply greedy per-slot acceptance, abandon
    /// the worst nests, and advance the generation counter.
    ///
    /// On the first call (generation zero bootstrap) all eggs are
    /// unconditionally accepted and nest abandonment is skipped.
    ///
    /// On subsequent calls:
    ///
    /// 1. **Greedy accept** — egg `i` replaces nest `i` iff
    ///    `fitness[i] ≤ state.fitness[i]`.
    /// 2. **Abandonment** — the `⌊p_a · pop_size⌋` worst nests are
    ///    re-initialized from `bounds` via [`seed_stream`]; abandoned
    ///    slots carry sentinel `+∞` fitness so the next generation's Lévy
    ///    proposal always lands on them.
    fn tell(
        &self,
        params: &CuckooConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: CuckooState<B>,
        rng: &mut dyn Rng,
    ) -> (CuckooState<B>, StrategyMetrics) {
        let fitness_host = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness tensor must be readable as f32");
        let device = population.device();
        let pop = params.pop_size;
        let d = params.genome_dim;

        if state.fitness.is_empty() {
            state.fitness.clone_from(&fitness_host);
            let best_idx = argmax_host(&fitness_host);
            state.best_fitness = fitness_host[best_idx];
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            state.best_genome = Some(population.clone().select(0, idx));
            state.nests = population;
            state.generation += 1;
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever();
            return (state, m);
        }

        // Greedy accept per slot.
        #[allow(clippy::cast_possible_wrap)]
        let mut rs: Vec<i64> = (0..pop).map(|i| i as i64).collect();
        let mut new_fitness = state.fitness.clone();
        for i in 0..pop {
            if fitness_host[i] >= state.fitness[i] {
                #[allow(clippy::cast_possible_wrap)]
                {
                    rs[i] = (pop + i) as i64;
                }
                new_fitness[i] = fitness_host[i];
            }
        }
        let stacked = Tensor::cat(vec![state.nests.clone(), population.clone()], 0);
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(rs, [pop]), &device);
        state.nests = stacked.select(0, idx);
        state.fitness = new_fitness;

        // Abandon worst `p_a · pop` nests — reinit with uniform sample;
        // mark fitness −∞ (worst under maximise) so next ask's Lévy
        // proposal always lands.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let n_abandon = (params.p_a * pop as f32) as usize;
        if n_abandon > 0 {
            let mut rank: Vec<usize> = (0..pop).collect();
            // Ascending: lowest fitness (worst under maximise) first. Sanitize
            // NaN → −inf so a NaN-fitness nest is treated as worst (abandoned).
            let sane: Vec<f32> = state
                .fitness
                .iter()
                .map(|&f| crate::fitness::sanitize_fitness(f))
                .collect();
            rank.sort_by(|&a, &b| sane[a].total_cmp(&sane[b]));
            let worst: Vec<usize> = rank.into_iter().take(n_abandon).collect();
            let (lo, hi): (f32, f32) = params.bounds.into();
            // Host-sample abandoned-nest replacements from a deterministic
            // `seed_stream` so the refill is reproducible across thread
            // schedules rather than racing the global Flex RNG.
            let mut abandon_stream = seed_stream(
                rng.next_u64(),
                state.generation as u64,
                SeedPurpose::Replacement,
            );
            let mut fresh_rows = Vec::with_capacity(n_abandon * d);
            for _ in 0..n_abandon * d {
                fresh_rows.push(lo + (hi - lo) * abandon_stream.random::<f32>());
            }
            let fresh =
                Tensor::<B, 2>::from_data(TensorData::new(fresh_rows, [n_abandon, d]), &device);
            #[allow(clippy::cast_possible_wrap)]
            let mut rs2: Vec<i64> = (0..pop).map(|i| i as i64).collect();
            for (k, &slot) in worst.iter().enumerate() {
                #[allow(clippy::cast_possible_wrap)]
                {
                    rs2[slot] = (pop + k) as i64;
                }
                state.fitness[slot] = f32::NEG_INFINITY;
            }
            let stacked2 = Tensor::cat(vec![state.nests.clone(), fresh], 0);
            let idx2 = Tensor::<B, 1, Int>::from_data(TensorData::new(rs2, [pop]), &device);
            state.nests = stacked2.select(0, idx2);
        }

        // Best-so-far from finite-fitness slots.
        let best_idx = argmax_host(&state.fitness);
        if state.fitness[best_idx].is_finite() && state.fitness[best_idx] > state.best_fitness {
            state.best_fitness = state.fitness[best_idx];
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            state.best_genome = Some(state.nests.clone().select(0, idx));
        }

        state.generation += 1;
        let m =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = m.best_fitness_ever();
        (state, m)
    }

    /// Returns the best-so-far `(genome, fitness)` pair, or `None` before
    /// the first [`tell`](Strategy::tell) call.
    fn best(&self, state: &CuckooState<B>) -> Option<(Tensor<B, 2>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
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
    fn try_new_checks_fitness_length() {
        let device = Default::default();
        let nests = Tensor::<TestBackend, 2>::zeros([3, 2], &device);
        assert!(CuckooState::try_new(nests.clone(), vec![1.0; 3], None, 1.0, 1).is_ok());
        assert!(CuckooState::try_new(nests.clone(), vec![], None, f32::MIN, 0).is_ok());
        assert!(CuckooState::try_new(nests, vec![1.0; 2], None, 1.0, 1).is_err());
        let empty = Tensor::<TestBackend, 2>::zeros([0, 2], &device);
        assert!(CuckooState::try_new(empty, vec![], None, 1.0, 0).is_err());
    }

    #[test]
    fn default_config_validates() {
        assert!(CuckooConfig::default_for(25, 10).validate().is_ok());
    }

    #[test]
    fn rejects_beta_at_upper_bound() {
        let mut cfg = CuckooConfig::default_for(25, 10);
        cfg.beta = 2.0;
        assert_eq!(cfg.validate().unwrap_err().field, "beta");
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
    fn gamma_matches_known_values() {
        // Γ(1) = 1, Γ(2) = 1, Γ(5) = 24, Γ(0.5) = √π.
        approx::assert_relative_eq!(gamma(1.0), 1.0, epsilon = 1e-4);
        approx::assert_relative_eq!(gamma(2.0), 1.0, epsilon = 1e-4);
        approx::assert_relative_eq!(gamma(5.0), 24.0, epsilon = 1e-3);
        approx::assert_relative_eq!(gamma(0.5), PI.sqrt(), epsilon = 1e-3);
    }

    #[test]
    fn mantegna_sigma_u_is_finite() {
        let s = CuckooSearch::<TestBackend>::mantegna_sigma_u(1.5);
        assert!(s.is_finite() && s > 0.0);
    }

    #[test]
    fn cuckoo_reduces_on_sphere_d10() {
        // Pure-Lévy CS has no gradient-biased update — it's a biased
        // random walk with abandonment. The Lévy flights are the
        // interesting part; otherwise CS is a thin wrapper around
        // random walk + abandonment, so convergence to machine
        // precision is not expected within reasonable budgets on
        // Sphere-D10. Threshold 20.0 in 800 generations is still a ~4×
        // reduction from the uniform-random baseline (≈ 87) — it
        // verifies the Lévy machinery composes correctly.
        let device = Default::default();
        let strategy = CuckooSearch::<TestBackend>::new();
        let mut params = CuckooConfig::default_for(30, 10);
        params.alpha = 0.2;
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 19, device, 800,
        )
        .expect("valid params");
        harness.reset();
        while !harness.step(()).done {}
        let best = harness.latest_metrics().unwrap().best_fitness_ever();
        assert!(best < 20.0, "Cuckoo D10 best={best}");
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact by design: 0.0 fold + byte-identical pass-through
    fn levy_step_folds_pathological_denominator_to_zero() {
        // Deterministic reproducer for #156 (Cuckoo): the Lévy step
        // component `u / |w|^(1/β)`. A zero Normal draw `w` makes the
        // denominator zero; un-guarded, `0/0` is `NaN` and `x/0` is `±inf`.
        // Both survive the bounds clamp and permanently poison a nest slot,
        // so `levy_step` folds any non-finite/zero-denominator case to `0.0`.
        //
        // Each pathological assertion below FAILS against the pre-fix loop
        // body (which computed `u / denom` unconditionally), shown by the
        // `unguarded` reference expressions being non-finite.
        let beta: f32 = 1.5;

        // w == 0, u == 0 → un-guarded `0/0 = NaN`.
        let unguarded_nan: f32 = 0.0_f32 / 0.0_f32.abs().powf(1.0 / beta);
        assert!(unguarded_nan.is_nan());
        assert_eq!(levy_step(0.0, 0.0, beta), 0.0);

        // w == 0, u != 0 → un-guarded `x/0 = ±inf`.
        let unguarded_inf: f32 = 1.0_f32 / 0.0_f32.abs().powf(1.0 / beta);
        assert!(!unguarded_inf.is_finite());
        assert_eq!(levy_step(1.0, 0.0, beta), 0.0);

        // A NaN Normal draw `w` makes the denominator non-finite → folded to 0.
        assert_eq!(levy_step(1.0, f32::NAN, beta), 0.0);

        // Normal case: finite and byte-identical to the un-guarded value
        // (the guard is a pass-through whenever the denominator is sound).
        let expected: f32 = 0.5_f32 / 1.2_f32.abs().powf(1.0 / beta);
        let got: f32 = levy_step(0.5, 1.2, beta);
        assert!(got.is_finite());
        approx::assert_relative_eq!(got, expected, epsilon = 1e-6);
        // Byte-identical: same operations, no reorder.
        assert_eq!(got, expected);
    }
}
