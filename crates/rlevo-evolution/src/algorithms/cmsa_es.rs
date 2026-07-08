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

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};

use crate::ops::linalg::{cholesky, matvec, symmetrize};
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
///
/// All adaptive quantities live here (not in [`CmsaEsConfig`]) so instances stay
/// lock-free across parallel runs. Linear-algebra state — the mean and
/// covariance — is held host-side as `Vec<f32>`; only the offspring population
/// crosses to the device.
#[derive(Debug, Clone)]
pub struct CmsaEsState<B: Backend> {
    /// Distribution mean `m`, length `D`.
    mean: Vec<f32>,
    /// Covariance matrix `C`, row-major `D × D`.
    cov: Vec<f32>,
    /// Global step size `σ̄`.
    sigma: f32,
    /// Per-offspring step sizes `σᵢ`, carried `ask → tell` (length `λ`, empty
    /// before the first `ask`). Mirrors the σ-scratchpad pattern in
    /// [`EsState`](crate::algorithms::es_classical::EsState).
    offspring_sigmas: Vec<f32>,
    /// Completed-generation counter.
    generation: usize,
    /// Best-so-far genome, shape `(1, D)`.
    best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness (canonical maximise convention).
    best_fitness: f32,
}

impl<B: Backend> CmsaEsState<B> {
    /// Assembles a CMSA-ES state, checking the distribution parameters are
    /// dimensionally consistent and normalizing `cov` to exact symmetry.
    ///
    /// The supplied `cov` is symmetrized in place via
    /// [`crate::ops::linalg::symmetrize`] before construction. The in-loop
    /// covariance blend in [`tell`](CmsaEs::tell) already preserves bit-exact
    /// symmetry — IEEE-754 multiplication is commutative and the two triangle
    /// entries `C[i,j]` / `C[j,i]` accumulate the identical rank-μ terms in the
    /// identical order — so caller-supplied construction is the *only* asymmetry
    /// entry point. Normalizing it here mirrors `pycma` practice and the
    /// ADR 0034 sanitize-at-chokepoint convention.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `mean` is empty, if `cov` is not `D × D`
    /// row-major (`D = mean.len()`), or if `sigma` is not strictly positive and
    /// finite. No length constraint is imposed on `offspring_sigmas`: it may be
    /// empty (the pre-`ask` state) or any length — [`tell`](CmsaEs::tell) falls
    /// back to `sigma` for any missing entry.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        mean: Vec<f32>,
        mut cov: Vec<f32>,
        sigma: f32,
        offspring_sigmas: Vec<f32>,
        generation: usize,
        best_genome: Option<Tensor<B, 2>>,
        best_fitness: f32,
    ) -> Result<Self, ConfigError> {
        let d = mean.len();
        config::nonzero("CmsaEsState", "mean", d)?;
        if cov.len() != d * d {
            return Err(ConfigError {
                config: "CmsaEsState",
                field: "cov",
                kind: ConstraintKind::Custom("covariance must be a row-major D × D matrix"),
            });
        }
        config::positive("CmsaEsState", "sigma", f64::from(sigma))?;
        symmetrize(&mut cov, d);
        Ok(Self {
            mean,
            cov,
            sigma,
            offspring_sigmas,
            generation,
            best_genome,
            best_fitness,
        })
    }

    /// Distribution mean `m`, length `D`.
    #[must_use]
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    /// Covariance matrix `C`, row-major `D × D`.
    #[must_use]
    pub fn cov(&self) -> &[f32] {
        &self.cov
    }

    /// Global step size `σ̄`.
    #[must_use]
    pub fn sigma(&self) -> f32 {
        self.sigma
    }

    /// Per-offspring step sizes `σᵢ`, carried `ask → tell` (empty before the
    /// first `ask`).
    #[must_use]
    pub fn offspring_sigmas(&self) -> &[f32] {
        &self.offspring_sigmas
    }

    /// Completed-generation counter.
    #[must_use]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Best-so-far genome (shape `(1, D)`), or `None` before the first `tell`.
    #[must_use]
    pub fn best_genome(&self) -> Option<&Tensor<B, 2>> {
        self.best_genome.as_ref()
    }

    /// Best-so-far (canonical, maximise) fitness.
    #[must_use]
    pub fn best_fitness(&self) -> f32 {
        self.best_fitness
    }
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
        debug_assert!(
            params.validate().is_ok(),
            "invalid CmsaEsConfig reached init: {params:?}"
        );
        let d = params.genome_dim;
        let (lo, hi): (f32, f32) = params.bounds.into();
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let mean: Vec<f32> = (0..d)
            .map(|_| lo + (hi - lo) * stream.random::<f32>())
            .collect();
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

        let mut stream = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::CmaSampling,
        );
        let mut rows: Vec<f32> = Vec::with_capacity(lambda * d);
        let mut sigmas: Vec<f32> = Vec::with_capacity(lambda);
        for _ in 0..lambda {
            // Floor σᵢ at the smallest positive f32. `exp` of a large negative
            // draw underflows to exactly `0.0` in f32; `tell` would then compute
            // sᵢ = (xᵢ − m)/σᵢ = 0/0 = NaN, which permanently poisons the
            // covariance blend. This floor matches the CSA σ floor in
            // cma_es.rs. A floored σᵢ yields a *benign zero step* — the
            // offspring collapses to ≈m and contributes ~0 to the rank-μ blend —
            // not a corrected tiny step.
            let sigma_i: f32 = (state.sigma
                * (params.tau * crate::sampling::standard_normal(&mut stream)).exp())
            .max(f32::MIN_POSITIVE);
            let z: Vec<f32> = (0..d)
                .map(|_| crate::sampling::standard_normal(&mut stream))
                .collect();
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

        let fitness_host: Vec<f32> = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness tensor must be readable as f32");
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
            let sigma_i = state
                .offspring_sigmas
                .get(idx)
                .copied()
                .unwrap_or(state.sigma);
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
        // Defensive float-drift hygiene (pycma-style): the Beyer & Sendhoff
        // (2008, PPSN X) rank-μ blend is symmetric by construction, so this
        // re-symmetrization is a no-op today; it guards the solver's symmetry
        // assumption against a future edit that reorders the accumulation.
        symmetrize(&mut cov_new, d);

        update_best(&mut state, &population, &fitness_host);

        state.generation += 1;
        let metrics =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
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
    use burn::backend::Flex;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Reconstruct `L · Lᵀ` for a row-major `n × n` lower-triangular factor.
    fn recon_llt(l: &[f32], n: usize) -> Vec<f32> {
        let mut out: Vec<f32> = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc: f32 = 0.0;
                for k in 0..n {
                    acc += l[i * n + k] * l[j * n + k];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    #[test]
    fn cholesky_with_jitter_recovers_from_non_pd_covariance() {
        // Issue #147 §7.2 jitter-recovery coverage. Fixture: a diagonal (hence
        // symmetric) NON-positive-definite covariance with eigenvalues
        // {−1e-5, 1} — for a diagonal matrix the eigenvalues *are* the diagonal
        // entries. The −1e-5 pivot makes the un-jittered `cholesky` return
        // `None`, forcing the trace-proportional jitter path.
        //
        // mean_diag = (−1e-5 + 1)/2 ≈ 0.5, so jitter starts at ≈5e-9 and grows
        // ×10 per retry. A `+jitter·I` shift moves every eigenvalue by exactly
        // +jitter, so the smallest eigenvalue (−1e-5 + jitter) first turns
        // positive at jitter = 5e-5 — retry index 4 of 6, two retries to spare.
        // Empirically the JITTER-RECOVERY branch fires here (factor
        // ≈ [6.32e-3, 0, 0, 1.000025]); the identity fallback does NOT.
        let cov: Vec<f32> = vec![-1e-5, 0.0, 0.0, 1.0];
        let factor: Vec<f32> = cholesky_with_jitter(&cov, 2);

        assert!(
            factor.iter().all(|x| x.is_finite()),
            "factor has non-finite entries: {factor:?}"
        );
        // Lower-triangular: the strict-upper entry is exactly zero.
        approx::assert_relative_eq!(factor[1], 0.0, epsilon = 1e-9);
        // Genuine Cholesky factor: strictly positive pivots.
        assert!(
            factor[0] > 0.0 && factor[3] > 0.0,
            "non-positive pivots: {factor:?}"
        );

        // L·Lᵀ ≈ the jittered covariance. The (0,0) entry is the recovered
        // ≈4e-5, NOT 1.0 — the proof the identity fallback did NOT fire (that
        // branch would return L = I, giving L·Lᵀ (0,0) = 1.0).
        let recon: Vec<f32> = recon_llt(&factor, 2);
        assert!(
            recon[0] < 0.5,
            "identity fallback fired instead of jitter recovery: recon = {recon:?}"
        );
        // The (1,1) entry stays ≈1 (jitter is only O(1e-5)).
        approx::assert_relative_eq!(recon[3], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn cholesky_with_jitter_falls_back_to_identity_when_degenerate() {
        // Issue #147 §7.2 fallback-branch coverage. Fixture: a symmetric,
        // strongly indefinite covariance [[1, 2], [2, 1]] with eigenvalues
        // {3, −1} (the same indefinite matrix `linalg::cholesky_rejects_non_
        // positive_definite` uses). The jitter shifts every eigenvalue by
        // +jitter, but jitter tops out at mean_diag·1e-8·10⁵ = 1·1e-3 after the
        // 6 retries — far too small to lift the −1 eigenvalue positive, so
        // every retry's (1,1) pivot stays negative. Empirically all 6 retries
        // fail and the function returns the IDENTITY factor.
        let cov: Vec<f32> = vec![1.0, 2.0, 2.0, 1.0];
        let factor: Vec<f32> = cholesky_with_jitter(&cov, 2);

        assert!(
            factor.iter().all(|x| x.is_finite()),
            "factor has non-finite entries: {factor:?}"
        );
        // Exactly the identity factor: diagonal ones, zero off-diagonal.
        approx::assert_relative_eq!(factor[0], 1.0, epsilon = 1e-9);
        approx::assert_relative_eq!(factor[1], 0.0, epsilon = 1e-9);
        approx::assert_relative_eq!(factor[2], 0.0, epsilon = 1e-9);
        approx::assert_relative_eq!(factor[3], 1.0, epsilon = 1e-9);
    }

    #[test]
    fn ask_tell_round_trip_survives_non_pd_covariance() {
        // Issue #153 ask/tell round-trip guard on the ill-conditioned-covariance
        // hazard. `try_new` symmetrizes `cov`, so a caller cannot inject
        // asymmetry — but it CAN inject a SYMMETRIC non-PD covariance, which is
        // the reachable path into `cholesky_with_jitter`. We reuse the
        // recovery-branch fixture (diagonal, eigenvalues {−1e-5, 1}); `ask` must
        // route it through the jitter recovery, sample valid offspring, and the
        // `ask → tell` round-trip must leave cov/mean/σ̄ finite (no NaN/inf
        // leaking from the ill-conditioned factor). Mirrors the structure of
        // `sigma_i_underflow_does_not_poison_covariance`.
        let strategy = CmsaEs::<Flex>::new();
        let params = CmsaEsConfig::with_pop_size(8, 2);
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(1);

        // Symmetric but non-PD: eigenvalue −1e-5 survives `symmetrize` (the
        // matrix is already symmetric) and reaches `ask`'s Cholesky.
        let state: CmsaEsState<Flex> = CmsaEsState::try_new(
            vec![0.0, 0.0],
            vec![-1e-5, 0.0, 0.0, 1.0],
            1.0,
            Vec::new(),
            0,
            None,
            f32::NEG_INFINITY,
        )
        .expect("valid state");

        let (population, asked) = strategy.ask(&params, &state, &mut rng, &device);
        // Any finite fitness — ranking is irrelevant to the non-finite hazard.
        let fitness = Tensor::<Flex, 1>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8]),
            &device,
        );
        let (told, _metrics) = strategy.tell(&params, population, fitness, asked, &mut rng);

        assert!(
            told.cov().iter().all(|c| c.is_finite()),
            "covariance has non-finite entries: {:?}",
            told.cov()
        );
        assert!(
            told.mean().iter().all(|m| m.is_finite()),
            "mean has non-finite entries: {:?}",
            told.mean()
        );
        assert!(
            told.sigma().is_finite() && told.sigma() > 0.0,
            "sigma is not finite and positive: {}",
            told.sigma()
        );
    }

    #[test]
    fn sigma_i_underflow_does_not_poison_covariance() {
        // Regression for the σᵢ underflow. With a minuscule σ̄, a negative
        // log-normal draw makes the raw σᵢ = σ̄·exp(τ·N) underflow to exactly
        // 0.0. Without the `.max(f32::MIN_POSITIVE)` floor in `ask`, `tell` then
        // forms sᵢ = (xᵢ − m)/σᵢ = 0/0 = NaN and poisons the rank-μ covariance
        // blend — reverting the floor turns this test red (NaN in `cov()`,
        // confirmed manually). The floor clamps those raw zeros up to
        // `f32::MIN_POSITIVE`, a benign zero step (the offspring collapses to
        // ≈m and contributes ~0 to the blend), so cov/mean/σ̄ all stay finite.
        //
        // We seed σ̄ at the smallest positive **subnormal** f32 rather than
        // `f32::MIN_POSITIVE` (the smallest *normal*): from the smallest normal,
        // an exact-0.0 underflow needs N < −33 (a ~33σ event that never fires),
        // whereas from the smallest subnormal any N < ≈−1.4 flushes to exactly
        // 0.0 — the realistic hazard. Because σ̄ is subnormal, *every* raw σᵢ
        // sits below `f32::MIN_POSITIVE`, so with the floor active every entry
        // reads back as exactly `f32::MIN_POSITIVE`; the precondition asserts the
        // floor engaged on at least one offspring (it is the observable proxy for
        // "an underflow would have occurred").
        let strategy = CmsaEs::<Flex>::new();
        let params = CmsaEsConfig::with_pop_size(8, 2);
        let device = Default::default();
        // Seed 1's `SeedPurpose::CmaSampling` stream draws at least one N < ≈−1.4
        // across the 8 offspring, the draw whose raw σᵢ underflows to 0.0.
        let mut rng = StdRng::seed_from_u64(1);

        let state: CmsaEsState<Flex> = CmsaEsState::try_new(
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            f32::from_bits(1),
            Vec::new(),
            0,
            None,
            f32::NEG_INFINITY,
        )
        .expect("valid state");

        let (population, asked) = strategy.ask(&params, &state, &mut rng, &device);
        assert!(
            asked.offspring_sigmas().contains(&f32::MIN_POSITIVE),
            "test precondition: the σᵢ floor must engage on at least one offspring"
        );
        // Any finite fitness — the ranking is irrelevant to the NaN hazard.
        let fitness = Tensor::<Flex, 1>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8]),
            &device,
        );
        let (told, _metrics) = strategy.tell(&params, population, fitness, asked, &mut rng);

        assert!(
            told.cov().iter().all(|c| c.is_finite()),
            "covariance has non-finite entries: {:?}",
            told.cov()
        );
        assert!(
            told.mean().iter().all(|m| m.is_finite()),
            "mean has non-finite entries: {:?}",
            told.mean()
        );
        assert!(
            told.sigma().is_finite() && told.sigma() > 0.0,
            "sigma is not finite and positive: {}",
            told.sigma()
        );
    }

    #[test]
    fn try_new_rejects_empty_mean() {
        let err = CmsaEsState::<Flex>::try_new(
            Vec::new(),
            Vec::new(),
            0.5,
            Vec::new(),
            0,
            None,
            f32::MIN,
        )
        .unwrap_err();
        assert_eq!(err.field, "mean");
    }

    #[test]
    fn try_new_rejects_wrong_cov_length() {
        // D = 2 wants a 4-entry cov; supply 3.
        let err = CmsaEsState::<Flex>::try_new(
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            0.5,
            Vec::new(),
            0,
            None,
            f32::MIN,
        )
        .unwrap_err();
        assert_eq!(err.field, "cov");
    }

    #[test]
    fn try_new_rejects_non_positive_sigma() {
        let err = CmsaEsState::<Flex>::try_new(
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            0.0,
            Vec::new(),
            0,
            None,
            f32::MIN,
        )
        .unwrap_err();
        assert_eq!(err.field, "sigma");
    }

    #[test]
    fn try_new_symmetrizes_covariance() {
        // Asymmetric off-diagonals 0.4 / 0.2 average to 0.3 on both sides.
        let state = CmsaEsState::<Flex>::try_new(
            vec![0.0, 0.0],
            vec![1.0, 0.4, 0.2, 1.0],
            0.5,
            Vec::new(),
            0,
            None,
            f32::MIN,
        )
        .expect("valid state");
        approx::assert_relative_eq!(state.cov()[1], 0.3, epsilon = 1e-6);
        approx::assert_relative_eq!(state.cov()[2], 0.3, epsilon = 1e-6);
    }

    #[test]
    fn accessors_round_trip_constructor_values() {
        let genome = Tensor::<Flex, 2>::from_data(
            TensorData::new(vec![1.0f32, 2.0], [1, 2]),
            &Default::default(),
        );
        let state = CmsaEsState::<Flex>::try_new(
            vec![1.0, -2.0],
            vec![2.0, 0.0, 0.0, 3.0],
            0.75,
            vec![0.1, 0.2, 0.3],
            7,
            Some(genome),
            42.0,
        )
        .expect("valid state");
        assert_eq!(state.mean(), &[1.0, -2.0]);
        assert_eq!(state.cov(), &[2.0, 0.0, 0.0, 3.0]);
        approx::assert_relative_eq!(state.sigma(), 0.75, epsilon = 1e-6);
        assert_eq!(state.offspring_sigmas(), &[0.1, 0.2, 0.3]);
        assert_eq!(state.generation(), 7);
        assert!(state.best_genome().is_some());
        approx::assert_relative_eq!(state.best_fitness(), 42.0, epsilon = 1e-6);
    }

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

    /// Issue #147 §7.2 determinism: two runs from the same seed produce
    /// bit-identical trajectories. CMSA-ES host-samples off a `StdRng` threaded
    /// through `init`/`ask` (which key their `seed_stream`s on `rng.next_u64()`
    /// and the generation counter), so an identical seed and identical call
    /// sequence must reproduce the mean, covariance, and σ̄ exactly.
    #[test]
    fn same_seed_yields_identical_trajectories() {
        fn run() -> (Vec<f32>, Vec<f32>, f32) {
            let strategy = CmsaEs::<Flex>::new();
            let params = CmsaEsConfig::with_pop_size(8, 3);
            let device = Default::default();
            let mut rng = StdRng::seed_from_u64(0xD37E_2711);
            let mut state = strategy.init(&params, &mut rng, &device);
            for _ in 0..4 {
                let (population, asked) = strategy.ask(&params, &state, &mut rng, &device);
                let fitness = Tensor::<Flex, 1>::from_data(
                    TensorData::new(vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [8]),
                    &device,
                );
                let (told, _metrics) = strategy.tell(&params, population, fitness, asked, &mut rng);
                state = told;
            }
            (state.mean().to_vec(), state.cov().to_vec(), state.sigma())
        }

        let (mean_a, cov_a, sigma_a): (Vec<f32>, Vec<f32>, f32) = run();
        let (mean_b, cov_b, sigma_b): (Vec<f32>, Vec<f32>, f32) = run();
        assert_eq!(
            mean_a, mean_b,
            "mean trajectory diverged under a fixed seed"
        );
        assert_eq!(
            cov_a, cov_b,
            "covariance trajectory diverged under a fixed seed"
        );
        assert_eq!(
            sigma_a.to_bits(),
            sigma_b.to_bits(),
            "σ̄ trajectory diverged under a fixed seed"
        );
    }

    /// Issue #147 §7.2/§7.3 regression: the rank-μ covariance blend must keep `C`
    /// bit-exactly symmetric across several generations. Guards the explicit
    /// `symmetrize` at the blend chokepoint (defensive float-drift hygiene per
    /// Beyer & Sendhoff 2008) against a future edit that reorders the outer-
    /// product accumulation and breaks the commutativity assumption.
    #[test]
    fn covariance_stays_symmetric_across_generations() {
        let strategy = CmsaEs::<Flex>::new();
        let params = CmsaEsConfig::with_pop_size(8, 3);
        let d: usize = params.genome_dim;
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(0x5A11_9E77);

        let mut state = strategy.init(&params, &mut rng, &device);
        for generation in 0..5 {
            let (population, asked) = strategy.ask(&params, &state, &mut rng, &device);
            let fitness = Tensor::<Flex, 1>::from_data(
                TensorData::new(vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [8]),
                &device,
            );
            let (told, _metrics) = strategy.tell(&params, population, fitness, asked, &mut rng);

            let cov: &[f32] = told.cov();
            for i in 0..d {
                for j in 0..d {
                    assert_eq!(
                        cov[i * d + j].to_bits(),
                        cov[j * d + i].to_bits(),
                        "asymmetry at ({i}, {j}) in generation {generation}"
                    );
                }
            }
            state = told;
        }
    }
}
