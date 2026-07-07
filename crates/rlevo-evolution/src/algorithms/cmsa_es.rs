//! Covariance Matrix Self-Adaptation Evolution Strategy (CMSA-ES).
//!
//! CMSA-ES (Beyer & Sendhoff, 2008) is the path-free cousin of CMA-ES. It
//! drops the evolution paths and Cumulative Step-size Adaptation entirely and
//! instead:
//!
//! - self-adapts the step size **per individual** with the classical
//!   log-normal rule `σᵢ = σ̄ · exp(τ · N(0, 1))`, then recombines the selected
//!   `σᵢ` into the next `σ̄` (the same σ-self-adaptation mechanism as
//!   [`crate::algorithms::es_classical`], so the two ES σ-adaptation families
//!   share one mutation rule);
//! - blends the covariance toward the rank-μ maximum-likelihood estimate of the
//!   selected mutation steps with time constant
//!   `τ_c = 1 + D(D+1)/(2μ)`:
//!   `C ← (1 − 1/τ_c) C + (1/τ_c) · (1/μ) Σ s_{(i)} s_{(i)}ᵀ`.
//!
//! Sampling needs only a Cholesky factor of `C` (no eigendecomposition,
//! no `C^{-1/2}`), so each generation is cheaper than CMA-ES.
//!
//! # On the τ constant
//!
//! The canonical CMSA-ES learning rate is `τ = 1/√(2D)` (Beyer & Sendhoff,
//! 2008), used here. Note this differs from
//! [`EsConfig`](crate::algorithms::es_classical::EsConfig)'s `1/√(2√D)`: the two
//! strategies share the log-normal σ-self-adaptation *mechanism* (ADR 0021 §5),
//! but CMSA-ES keeps its own algorithm-faithful constant.
//!
//! # Relationship to the EDA / `ProbabilityModel` family
//!
//! Like [`CmaEs`](crate::algorithms::cma_es), this is a self-contained
//! [`Strategy`]; per ADR 0021 it does not instantiate
//! [`ProbabilityModel`](crate::ProbabilityModel). The rank-μ covariance blend
//! is closer to an EMNA-style maximum-likelihood update than CMA-ES's
//! path-driven adaptation, but the per-individual σ self-adaptation keeps it on
//! the ES side of the boundary (research note `eda-vs-cma-es-boundary`).
//!
//! # References
//!
//! - Beyer, H.-G. & Sendhoff, B. (2008), *Covariance Matrix Adaptation
//!   Revisited — The CMSA Evolution Strategy*, PPSN X, LNCS 5199.

use std::marker::PhantomData;

use burn::tensor::{Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;
use rand_distr::{Distribution as _, Normal};

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};

use crate::ops::linalg::{cholesky, matvec};
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Cholesky factor of `cov`, recovering from a non-positive-definite covariance
/// by adding **trace-proportional** diagonal jitter and retrying with
/// geometrically growing magnitude.
///
/// The jitter base scales with the mean eigenvalue (`trace/D`), so it stays
/// meaningful regardless of the covariance magnitude — a fixed absolute jitter
/// is too small once `C` has grown to `O(1)` entries. Only if every retry fails
/// (a genuinely degenerate `C`) does it fall back to the identity factor, which
/// keeps the sampling distribution valid at the cost of one generation's learned
/// shape.
fn cholesky_with_jitter(cov: &[f32], d: usize) -> Vec<f32> {
    if let Some(l) = cholesky(cov, d) {
        return l;
    }
    let trace: f32 = (0..d).map(|i| cov[i * d + i]).sum();
    #[allow(clippy::cast_precision_loss)]
    let mean_diag: f32 = (trace / d as f32).max(f32::MIN_POSITIVE);
    let mut jitter: f32 = mean_diag * 1e-8;
    for _ in 0..6 {
        let mut jittered: Vec<f32> = cov.to_vec();
        for i in 0..d {
            jittered[i * d + i] += jitter;
        }
        if let Some(l) = cholesky(&jittered, d) {
            return l;
        }
        jitter *= 10.0;
    }
    // Degenerate covariance: fall back to the identity factor.
    let mut id: Vec<f32> = vec![0.0; d * d];
    for i in 0..d {
        id[i * d + i] = 1.0;
    }
    id
}

/// Static configuration for a CMSA-ES run.
#[derive(Debug, Clone)]
pub struct CmsaEsConfig {
    /// Offspring population size `λ`.
    pub pop_size: usize,
    /// Genome dimensionality `D`.
    pub genome_dim: usize,
    /// Search-space bounds; used only to sample the initial mean `m⁰`.
    pub bounds: Bounds,
    /// Initial global step size `σ̄`.
    pub initial_sigma: f32,
    /// Number of selected parents `μ = ⌊λ/2⌋`.
    pub mu: usize,
    /// Log-normal σ-self-adaptation learning rate `τ = 1/√(2D)`.
    pub tau: f32,
    /// Covariance time constant `τ_c = 1 + D(D+1)/(2μ)`.
    pub tau_c: f32,
}

impl CmsaEsConfig {
    /// Default configuration for dimensionality `D`, with the Hansen-style
    /// population `λ = 4 + ⌊3 ln D⌋` and `μ = ⌊λ/2⌋`.
    ///
    /// Sets `bounds = (-5.12, 5.12)` and `initial_sigma = 1.0`.
    #[must_use]
    pub fn default_for(genome_dim: usize) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let d = genome_dim as f32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let lambda = 4 + (3.0 * d.ln()).floor() as usize;
        Self::with_pop_size(lambda, genome_dim)
    }

    /// Configuration with an explicit population size `λ`.
    ///
    /// The `pop_size ≥ 2` invariant is enforced by [`Validate::validate`] at the
    /// harness chokepoint, not by this infallible producer.
    #[must_use]
    pub fn with_pop_size(pop_size: usize, genome_dim: usize) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let d = genome_dim as f32;
        let mu: usize = pop_size / 2;
        #[allow(clippy::cast_precision_loss)]
        let mu_f = mu as f32;
        let tau: f32 = 1.0 / (2.0 * d).sqrt();
        let tau_c: f32 = 1.0 + d * (d + 1.0) / (2.0 * mu_f);
        Self {
            pop_size,
            genome_dim,
            bounds: Bounds::new(-5.12, 5.12),
            initial_sigma: 1.0,
            mu,
            tau,
            tau_c,
        }
    }
}

impl Validate for CmsaEsConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "CmsaEsConfig";
        config::at_least(C, "pop_size", self.pop_size, 2)?;
        config::nonzero(C, "genome_dim", self.genome_dim)?;
        config::positive(C, "initial_sigma", f64::from(self.initial_sigma))?;
        config::at_least(C, "mu", self.mu, 1)?;
        if self.mu > self.pop_size {
            return Err(ConfigError {
                config: C,
                field: "mu",
                kind: ConstraintKind::Custom("mu must not exceed pop_size"),
            });
        }
        config::positive(C, "tau", f64::from(self.tau))?;
        config::in_range(C, "tau_c", 1.0, f64::INFINITY, f64::from(self.tau_c))?;
        Ok(())
    }
}

/// Generation state for [`CmsaEs`].
#[derive(Debug, Clone)]
pub struct CmsaEsState<B: Backend> {
    /// Distribution mean `m`, length `D`.
    pub mean: Vec<f32>,
    /// Covariance matrix `C`, row-major `D × D`.
    pub cov: Vec<f32>,
    /// Global step size `σ̄`.
    pub sigma: f32,
    /// Per-offspring step sizes `σᵢ`, carried `ask → tell` (length `λ`, empty
    /// before the first `ask`). Mirrors the σ-scratchpad pattern in
    /// [`EsState`](crate::algorithms::es_classical::EsState).
    pub offspring_sigmas: Vec<f32>,
    /// Completed-generation counter.
    pub generation: usize,
    /// Best-so-far genome, shape `(1, D)`.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness (canonical maximise convention).
    pub best_fitness: f32,
}

/// Covariance Matrix Self-Adaptation Evolution Strategy.
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_evolution::algorithms::cmsa_es::{CmsaEsConfig, CmsaEs};
///
/// let strategy = CmsaEs::<Flex>::new();
/// let params = CmsaEsConfig::default_for(10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CmsaEs<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> CmsaEs<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Strategy<B> for CmsaEs<B>
where
    B::Device: Clone,
{
    type Params = CmsaEsConfig;
    type State = CmsaEsState<B>;
    type Genome = Tensor<B, 2>;

    /// Initializes `m⁰` uniformly in `params.bounds` (host-RNG convention),
    /// `C = I`, and `σ̄ = initial_sigma`.
    fn init(
        &self,
        params: &CmsaEsConfig,
        rng: &mut dyn Rng,
        _device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> CmsaEsState<B> {
        debug_assert!(params.validate().is_ok(), "invalid CmsaEsConfig reached init: {params:?}");
        let d = params.genome_dim;
        let (lo, hi): (f32, f32) = params.bounds.into();
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let mean: Vec<f32> = (0..d).map(|_| lo + (hi - lo) * stream.random::<f32>()).collect();
        let mut cov: Vec<f32> = vec![0.0; d * d];
        for i in 0..d {
            cov[i * d + i] = 1.0;
        }
        CmsaEsState {
            mean,
            cov,
            sigma: params.initial_sigma,
            offspring_sigmas: Vec::new(),
            generation: 0,
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
        }
    }

    /// Samples `λ` offspring with per-individual log-normal step sizes.
    ///
    /// For each offspring: `σᵢ = σ̄ · exp(τ · N(0,1))`, `sᵢ = A zᵢ`
    /// (`A` the Cholesky factor of `C`, `zᵢ ~ N(0, I)`), `xᵢ = m + σᵢ · sᵢ`.
    /// All draws come from one deterministic [`SeedPurpose::CmaSampling`]
    /// stream; the `σᵢ` are stashed in state for [`tell`](Self::tell).
    fn ask(
        &self,
        params: &CmsaEsConfig,
        state: &CmsaEsState<B>,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Tensor<B, 2>, CmsaEsState<B>) {
        let d = params.genome_dim;
        let lambda = params.pop_size;

        // Cholesky factor A of C, with trace-relative jitter recovery on a PD
        // failure (see `cholesky_with_jitter`).
        let factor: Vec<f32> = cholesky_with_jitter(&state.cov, d);

        let mut stream = seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::CmaSampling);
        let normal = Normal::new(0.0f32, 1.0).expect("unit normal is well-defined");

        let mut rows: Vec<f32> = Vec::with_capacity(lambda * d);
        let mut sigmas: Vec<f32> = Vec::with_capacity(lambda);
        for _ in 0..lambda {
            let sigma_i: f32 = state.sigma * (params.tau * normal.sample(&mut stream)).exp();
            let z: Vec<f32> = (0..d).map(|_| normal.sample(&mut stream)).collect();
            let s: Vec<f32> = matvec(&factor, &z, d);
            for (mean_i, s_i) in state.mean.iter().zip(s.iter()) {
                rows.push(mean_i + sigma_i * s_i);
            }
            sigmas.push(sigma_i);
        }

        let population = Tensor::<B, 2>::from_data(TensorData::new(rows, [lambda, d]), device);
        let mut next: CmsaEsState<B> = state.clone();
        next.offspring_sigmas = sigmas;
        (population, next)
    }

    /// Recombines the `μ` best offspring into the next mean, step size, and
    /// rank-μ covariance blend.
    fn tell(
        &self,
        params: &CmsaEsConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: CmsaEsState<B>,
        _rng: &mut dyn Rng,
    ) -> (CmsaEsState<B>, StrategyMetrics) {
        let d = params.genome_dim;
        let lambda = params.pop_size;
        let mu = params.mu;

        let fitness_host: Vec<f32> = fitness.into_data().into_vec::<f32>().expect("fitness tensor must be readable as f32");
        let pop_host: Vec<f32> = population
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("population tensor must be readable as f32");

        // Rank descending (canonical maximise); take the μ best (highest).
        let mut ranked: Vec<usize> = (0..lambda).collect();
        // Sanitize NaN → −inf (worst) so it can never rank as best, then order
        // by `total_cmp` (deterministic; sanitized NaN sorts last).
        let sane: Vec<f32> = fitness_host
            .iter()
            .map(|&f| crate::fitness::sanitize_fitness(f))
            .collect();
        ranked.sort_by(|&a, &b| sane[b].total_cmp(&sane[a]));

        let m_old: Vec<f32> = state.mean.clone();
        #[allow(clippy::cast_precision_loss)]
        let inv_mu: f32 = 1.0 / mu as f32;

        // New mean (equal-weight recombination), new σ̄ (mean of selected σᵢ),
        // and the mutation steps s_{(i)} = (x_{(i)} − m) / σᵢ for rank-μ.
        let mut mean_new: Vec<f32> = vec![0.0; d];
        let mut sigma_sum: f32 = 0.0;
        let mut s_sel: Vec<Vec<f32>> = Vec::with_capacity(mu);
        for &idx in ranked.iter().take(mu) {
            let sigma_i = state.offspring_sigmas.get(idx).copied().unwrap_or(state.sigma);
            sigma_sum += sigma_i;
            let mut si: Vec<f32> = vec![0.0; d];
            for i in 0..d {
                let xi = pop_host[idx * d + i];
                mean_new[i] += inv_mu * xi;
                si[i] = (xi - m_old[i]) / sigma_i;
            }
            s_sel.push(si);
        }
        let sigma_new: f32 = sigma_sum * inv_mu;

        // Rank-μ ML covariance blend:
        // C ← (1 − 1/τ_c) C + (1/τ_c) (1/μ) Σ s_{(i)} s_{(i)}ᵀ.
        let blend: f32 = 1.0 / params.tau_c;
        let c_old: Vec<f32> = state.cov.clone();
        let mut cov_new: Vec<f32> = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut rankmu: f32 = 0.0;
                for si in &s_sel {
                    rankmu += si[i] * si[j];
                }
                rankmu *= inv_mu;
                cov_new[i * d + j] = (1.0 - blend) * c_old[i * d + j] + blend * rankmu;
            }
        }

        update_best(&mut state, &population, &fitness_host);

        state.generation += 1;
        let metrics = StrategyMetrics::from_host_fitness(
            state.generation,
            &fitness_host,
            state.best_fitness,
        );
        state.best_fitness = metrics.best_fitness_ever();

        state.mean = mean_new;
        state.cov = cov_new;
        state.sigma = sigma_new;
        state.offspring_sigmas = Vec::new();

        (state, metrics)
    }

    /// Returns the best-so-far genome and its fitness, or `None` before the
    /// first [`tell`](Self::tell) call.
    fn best(&self, state: &CmsaEsState<B>) -> Option<(Tensor<B, 2>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

/// Updates `state.best_genome` / `state.best_fitness` if this generation
/// improved on the best-so-far.
fn update_best<B: Backend>(state: &mut CmsaEsState<B>, pop: &Tensor<B, 2>, fitness: &[f32]) {
    if fitness.is_empty() {
        return;
    }
    let mut best_idx: usize = 0;
    let mut best: f32 = f32::NEG_INFINITY;
    for (i, &f) in fitness.iter().enumerate() {
        if f > best {
            best = f;
            best_idx = i;
        }
    }
    if best > state.best_fitness {
        let device = pop.device();
        #[allow(clippy::cast_possible_wrap)]
        let idx = Tensor::<B, 1, burn::tensor::Int>::from_data(
            TensorData::new(vec![best_idx as i64], [1]),
            &device,
        );
        state.best_genome = Some(pop.clone().select(0, idx));
        state.best_fitness = best;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        assert!(CmsaEsConfig::default_for(10).validate().is_ok());
    }

    #[test]
    fn rejects_pop_size_below_two() {
        let mut cfg = CmsaEsConfig::default_for(10);
        cfg.pop_size = 1;
        assert_eq!(cfg.validate().unwrap_err().field, "pop_size");
    }

    #[test]
    fn default_for_d10_constants() {
        let cfg = CmsaEsConfig::default_for(10);
        assert_eq!(cfg.pop_size, 10);
        assert_eq!(cfg.mu, 5);
        // τ = 1/√20 ≈ 0.2236.
        approx::assert_relative_eq!(cfg.tau, 1.0 / 20.0_f32.sqrt(), epsilon = 1e-6);
        // τ_c = 1 + 10·11/(2·5) = 1 + 11 = 12.
        approx::assert_relative_eq!(cfg.tau_c, 12.0, epsilon = 1e-5);
    }

    #[test]
    fn tau_differs_from_es_classical() {
        // CMSA-ES uses the canonical 1/√(2D); es_classical uses 1/√(2√D).
        let cfg = CmsaEsConfig::default_for(10);
        #[allow(clippy::cast_precision_loss)]
        let d = 10.0_f32;
        let es_classical_tau = 1.0 / (2.0 * d.sqrt()).sqrt();
        assert!(
            (cfg.tau - es_classical_tau).abs() > 0.1,
            "canonical CMSA τ must differ from es_classical τ"
        );
    }
}
