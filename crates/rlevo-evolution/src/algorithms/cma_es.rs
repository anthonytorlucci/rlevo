//! Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
//!
//! CMA-ES (Hansen & Ostermeier, 2001; Hansen, 2016) samples each generation
//! from a multivariate normal `N(m, σ²C)` and adapts the mean `m`, the global
//! step size `σ`, and the covariance matrix `C` from the ranked offspring. Two
//! evolution paths drive the adaptation:
//!
//! - the **conjugate path** `p_σ` feeds Cumulative Step-size Adaptation (CSA),
//!   which lengthens or shrinks `σ` depending on whether consecutive steps are
//!   correlated or anti-correlated;
//! - the **anisotropic path** `p_c` feeds the rank-1 update of `C`.
//!
//! A rank-μ update mixes in the empirical covariance of the selected steps. The
//! conjugate path requires `C^{-1/2}`, obtained from a symmetric
//! eigendecomposition of `C` (see [`crate::ops::linalg::jacobi_eigen`]).
//!
//! # Relationship to the EDA / `ProbabilityModel` family
//!
//! A full-covariance multivariate-Gaussian EDA (EMNA) is CMA-ES *minus* the
//! evolution paths and step-size decoupling: it re-estimates `m`/`C` by maximum
//! likelihood each generation. CMA-ES keeps the path-based momentum and CSA, so
//! it does **not** fit the [`ProbabilityModel`](crate::ProbabilityModel)
//! `fit → sample` seam — the CSA and path updates live in
//! [`Strategy::tell`], not in a model fit. Per ADR 0021 this strategy is a
//! self-contained [`Strategy`]; `ProbabilityModel<B>` is available but
//! deliberately unused (research note `eda-vs-cma-es-boundary`). For the
//! path-free sibling that self-adapts σ per individual, see
//! [`crate::algorithms::cmsa_es`].
//!
//! # References
//!
//! - Hansen, N. (2016), *The CMA Evolution Strategy: A Tutorial*,
//!   arXiv:1604.00772 (default parameters: Table 1).
//! - Hansen, N. & Ostermeier, A. (2001), *Completely Derandomized
//!   Self-Adaptation in Evolution Strategies*, Evolutionary Computation 9(2).

use std::marker::PhantomData;

use burn::tensor::{Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;
use rand_distr::{Distribution as _, Normal};

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate, Violations};

use crate::ops::linalg::{jacobi_eigen, matvec};
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Absolute backstop floor for eigenvalues (guards against an all-zero `C`).
const EIGENVALUE_FLOOR: f32 = 1e-20;

/// Relative eigenvalue floor: eigenvalues below `λ_max · CONDITION_FLOOR` are
/// clamped before taking `√Λ` / `1/√Λ`, capping the covariance condition number
/// near `1e14` (pycma's condition-number treatment). Without this, a single
/// eigenvalue drifting toward zero would make a `C^{-1/2}` column explode and
/// drive `σ` to `+∞` through the CSA update.
const CONDITION_FLOOR: f32 = 1e-14;

/// Per-eigenvalue floor for the current covariance: the larger of the absolute
/// backstop and `λ_max · CONDITION_FLOOR`.
fn eigenvalue_floor(eigvals: &[f32]) -> f32 {
    let lmax: f32 = eigvals.iter().copied().fold(0.0_f32, f32::max);
    (lmax * CONDITION_FLOOR).max(EIGENVALUE_FLOOR)
}

/// Static configuration for a CMA-ES run.
///
/// Construct with [`CmaEsConfig::default_for`] (derives `λ` from the dimension
/// per Hansen 2016) or [`CmaEsConfig::with_pop_size`] (explicit `λ`, e.g. a
/// larger population for multimodal landscapes). The recombination weights and
/// learning rates are all derived from `(λ, D)` and cached as fields so
/// [`Strategy::tell`] reads them without recomputing.
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Offspring population size `λ`.
    pub pop_size: usize,
    /// Genome dimensionality `D`.
    pub genome_dim: usize,
    /// Search-space bounds; used only to sample the initial mean `m⁰`.
    /// Offspring are **not** clamped (CMA-ES samples in unbounded ℝᴰ).
    pub bounds: Bounds,
    /// Initial global step size `σ`.
    pub initial_sigma: f32,
    /// Number of selected parents `μ = ⌊λ/2⌋`.
    pub mu: usize,
    /// Recombination weights `wᵢ` (length `μ`, positive, summing to 1).
    pub weights: Vec<f32>,
    /// Variance-effective selection mass `μ_eff = 1 / Σ wᵢ²`.
    pub mu_eff: f32,
    /// CSA learning rate `c_σ`.
    pub c_sigma: f32,
    /// CSA damping `d_σ`.
    pub d_sigma: f32,
    /// Anisotropic-path learning rate `c_c`.
    pub c_c: f32,
    /// Rank-1 covariance learning rate `c_1`.
    pub c_1: f32,
    /// Rank-μ covariance learning rate `c_μ`.
    pub c_mu: f32,
    /// Expected length of `N(0, I)`, `χ_n ≈ √D (1 − 1/4D + 1/21D²)`.
    pub chi_n: f32,
}

impl CmaEsConfig {
    /// Default configuration for dimensionality `D`, with the Hansen-2016
    /// population `λ = 4 + ⌊3 ln D⌋`.
    ///
    /// Sets `bounds = (-5.12, 5.12)` (the standard Sphere/Rastrigin domain) and
    /// `initial_sigma = 1.0`.
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
    /// Larger `λ` improves basin-finding on multimodal landscapes (Hansen 2016,
    /// §A); all derived weights and learning rates follow from `(λ, D)`.
    ///
    /// The `pop_size ≥ 2` invariant is enforced by [`Validate::validate`] at the
    /// harness chokepoint, not by this infallible producer.
    #[must_use]
    pub fn with_pop_size(pop_size: usize, genome_dim: usize) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let d = genome_dim as f32;
        let mu: usize = pop_size / 2;

        // Positive recombination weights w'ᵢ = ln(μ + ½) − ln(i), normalized.
        let raw: Vec<f32> = (1..=mu)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let fi = i as f32;
                #[allow(clippy::cast_precision_loss)]
                let mu_f = mu as f32;
                (mu_f + 0.5).ln() - fi.ln()
            })
            .collect();
        let sum: f32 = raw.iter().sum();
        let weights: Vec<f32> = raw.iter().map(|w| w / sum).collect();
        let sum_sq: f32 = weights.iter().map(|w| w * w).sum();
        let mu_eff: f32 = 1.0 / sum_sq;

        let c_sigma: f32 = (mu_eff + 2.0) / (d + mu_eff + 5.0);
        let d_sigma: f32 =
            1.0 + 2.0 * (((mu_eff - 1.0) / (d + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let c_c: f32 = (4.0 + mu_eff / d) / (d + 4.0 + 2.0 * mu_eff / d);
        let c_1: f32 = 2.0 / ((d + 1.3) * (d + 1.3) + mu_eff);
        let c_mu: f32 = (1.0 - c_1).min(
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((d + 2.0) * (d + 2.0) + mu_eff),
        );
        let chi_n: f32 = d.sqrt() * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d));

        Self {
            pop_size,
            genome_dim,
            bounds: Bounds::new(-5.12, 5.12),
            initial_sigma: 1.0,
            mu,
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            chi_n,
        }
    }
}

impl Validate for CmaEsConfig {
    /// Fail-fast: reports the first violation, derived from [`validate_all`] so
    /// the two never disagree.
    ///
    /// [`validate_all`]: CmaEsConfig::validate_all
    fn validate(&self) -> Result<(), ConfigError> {
        self.validate_all().map_err(|mut errs| errs.remove(0))
    }

    /// Accumulate-all: reports every violated invariant in one pass.
    ///
    /// Unlike most configs, `CmaEsConfig` exposes its **derived** fields —
    /// recombination `weights`, `mu_eff`, and the five learning rates — as
    /// public struct fields (so callers can construct one by hand). The
    /// [`default_for`] / [`with_pop_size`] producers keep them mutually
    /// consistent, but a hand-built literal can desync several at once; listing
    /// all violations then beats fixing them one recompile at a time.
    ///
    /// [`default_for`]: CmaEsConfig::default_for
    /// [`with_pop_size`]: CmaEsConfig::with_pop_size
    fn validate_all(&self) -> Result<(), Vec<ConfigError>> {
        const C: &str = "CmaEsConfig";
        let mut v = Violations::new();

        // Primary inputs.
        v.check(config::at_least(C, "pop_size", self.pop_size, 2));
        v.check(config::nonzero(C, "genome_dim", self.genome_dim));
        v.check(config::positive(C, "initial_sigma", f64::from(self.initial_sigma)));
        v.check(config::at_least(C, "mu", self.mu, 1));
        if self.mu > self.pop_size {
            v.check(Err(ConfigError {
                config: C,
                field: "mu",
                kind: ConstraintKind::Custom("mu must not exceed pop_size"),
            }));
        }

        // Derived recombination weights: length μ, strictly positive, sum ≈ 1.
        if self.weights.len() != self.mu {
            v.check(Err(ConfigError {
                config: C,
                field: "weights",
                kind: ConstraintKind::Custom("weights length must equal mu"),
            }));
        }
        if !self.weights.iter().all(|w| *w > 0.0) {
            v.check(Err(ConfigError {
                config: C,
                field: "weights",
                kind: ConstraintKind::Custom("recombination weights must all be positive"),
            }));
        }
        let weight_sum = f64::from(self.weights.iter().sum::<f32>());
        v.check(config::in_range(C, "weights", 1.0 - 1e-3, 1.0 + 1e-3, weight_sum));

        // Derived scalars. mu_eff = 1/Σwᵢ² ≥ 1; d_sigma and chi_n are positive
        // denominators/scales — a non-positive value diverges the step-size
        // control or the covariance update.
        v.check(config::in_range(C, "mu_eff", 1.0, f64::INFINITY, f64::from(self.mu_eff)));
        v.check(config::positive(C, "d_sigma", f64::from(self.d_sigma)));
        v.check(config::positive(C, "chi_n", f64::from(self.chi_n)));

        // Covariance/step-size learning rates each live in [0, 1], and the pair
        // (c_1, c_mu) must not sum past 1: the rank-update retention factor is
        // `1 − c_1 − c_mu`, so c_1 + c_mu > 1 turns it negative and the
        // covariance matrix loses positive-definiteness.
        v.check(config::in_range(C, "c_sigma", 0.0, 1.0, f64::from(self.c_sigma)));
        v.check(config::in_range(C, "c_c", 0.0, 1.0, f64::from(self.c_c)));
        v.check(config::in_range(C, "c_1", 0.0, 1.0, f64::from(self.c_1)));
        v.check(config::in_range(C, "c_mu", 0.0, 1.0, f64::from(self.c_mu)));
        v.check(config::in_range(
            C,
            "c_1_plus_c_mu",
            0.0,
            1.0,
            f64::from(self.c_1) + f64::from(self.c_mu),
        ));

        v.into_result()
    }
}

/// Generation state for [`CmaEs`].
///
/// All adaptive quantities live here (not in [`CmaEsConfig`]) so instances stay
/// lock-free across parallel runs. Linear-algebra state — the mean, covariance,
/// and evolution paths — is held host-side as `Vec<f32>`; only the offspring
/// population crosses to the device.
#[derive(Debug, Clone)]
pub struct CmaEsState<B: Backend> {
    /// Distribution mean `m`, length `D`.
    pub mean: Vec<f32>,
    /// Covariance matrix `C`, row-major `D × D`.
    pub cov: Vec<f32>,
    /// Conjugate evolution path `p_σ`, length `D`.
    pub p_sigma: Vec<f32>,
    /// Anisotropic evolution path `p_c`, length `D`.
    pub p_c: Vec<f32>,
    /// Global step size `σ`.
    pub sigma: f32,
    /// Completed-generation counter.
    pub generation: usize,
    /// Best-so-far genome, shape `(1, D)`.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness (canonical maximise convention).
    pub best_fitness: f32,
}

/// Covariance Matrix Adaptation Evolution Strategy.
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_evolution::algorithms::cma_es::{CmaEsConfig, CmaEs};
///
/// let strategy = CmaEs::<Flex>::new();
/// let params = CmaEsConfig::default_for(10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CmaEs<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> CmaEs<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Strategy<B> for CmaEs<B>
where
    B::Device: Clone,
{
    type Params = CmaEsConfig;
    type State = CmaEsState<B>;
    type Genome = Tensor<B, 2>;

    /// Initializes `m⁰` uniformly in `params.bounds` (host-RNG convention),
    /// `C = I`, `σ = initial_sigma`, and both evolution paths to zero.
    fn init(
        &self,
        params: &CmaEsConfig,
        rng: &mut dyn Rng,
        _device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> CmaEsState<B> {
        debug_assert!(params.validate().is_ok(), "invalid CmaEsConfig reached init: {params:?}");
        let d = params.genome_dim;
        let (lo, hi): (f32, f32) = params.bounds.into();
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let mean: Vec<f32> = (0..d).map(|_| lo + (hi - lo) * stream.random::<f32>()).collect();
        let mut cov: Vec<f32> = vec![0.0; d * d];
        for i in 0..d {
            cov[i * d + i] = 1.0;
        }
        CmaEsState {
            mean,
            cov,
            p_sigma: vec![0.0; d],
            p_c: vec![0.0; d],
            sigma: params.initial_sigma,
            generation: 0,
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
        }
    }

    /// Samples `λ` offspring from `N(m, σ²C)`.
    ///
    /// The covariance is eigendecomposed into `C = B diag(Λ) Bᵀ`; each
    /// offspring is `xᵢ = m + σ · B diag(√Λ) zᵢ` for `zᵢ ~ N(0, I)`, drawn
    /// host-side from a deterministic [`SeedPurpose::CmaSampling`] stream. The
    /// state is returned unchanged (the mean/covariance update happens in
    /// [`tell`](Self::tell), which recomputes the steps from the population).
    fn ask(
        &self,
        params: &CmaEsConfig,
        state: &CmaEsState<B>,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Tensor<B, 2>, CmaEsState<B>) {
        let d = params.genome_dim;
        let lambda = params.pop_size;

        // Sampling transform B·diag(√Λ) from the eigendecomposition of C.
        let (eigvals, eigvecs) = jacobi_eigen(&state.cov, d);
        let floor: f32 = eigenvalue_floor(&eigvals);
        let mut bd: Vec<f32> = vec![0.0; d * d];
        for i in 0..d {
            for k in 0..d {
                bd[i * d + k] = eigvecs[i * d + k] * eigvals[k].max(floor).sqrt();
            }
        }

        let mut stream = seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::CmaSampling);
        let normal = Normal::new(0.0f32, 1.0).expect("unit normal is well-defined");
        let mut rows: Vec<f32> = Vec::with_capacity(lambda * d);
        for _ in 0..lambda {
            let z: Vec<f32> = (0..d).map(|_| normal.sample(&mut stream)).collect();
            let bdz: Vec<f32> = matvec(&bd, &z, d);
            for (mean_i, bdz_i) in state.mean.iter().zip(bdz.iter()) {
                rows.push(mean_i + state.sigma * bdz_i);
            }
        }
        let population = Tensor::<B, 2>::from_data(TensorData::new(rows, [lambda, d]), device);
        (population, state.clone())
    }

    /// Ranks the offspring, recombines the mean, and runs CSA + the rank-1 /
    /// rank-μ covariance updates.
    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    fn tell(
        &self,
        params: &CmaEsConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: CmaEsState<B>,
        _rng: &mut dyn Rng,
    ) -> (CmaEsState<B>, StrategyMetrics) {
        let d = params.genome_dim;
        let lambda = params.pop_size;
        let mu = params.mu;

        let fitness_host: Vec<f32> = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let pop_host: Vec<f32> = population
            .clone()
            .into_data()
            .into_vec::<f32>()
            .unwrap_or_default();

        // Rank offspring descending (canonical maximise): ranked[0] is the
        // best (highest fitness). The recombination weights `params.weights`
        // are assigned to rank positions unchanged — only the ordering of
        // which individuals occupy those ranks inverts relative to a
        // minimisation engine. Against a `Minimize` landscape the harness
        // feeds the engine `−cost`, so this descending canonical order
        // matches the `pycma` ascending-cost order point-for-point.
        let mut ranked: Vec<usize> = (0..lambda).collect();
        // Sanitize NaN → −inf (worst) so it can never rank as best, then order
        // by `total_cmp` (deterministic; sanitized NaN sorts last).
        let sane: Vec<f32> = fitness_host
            .iter()
            .map(|&f| crate::fitness::sanitize_fitness(f))
            .collect();
        ranked.sort_by(|&a, &b| sane[b].total_cmp(&sane[a]));

        let m_old: Vec<f32> = state.mean.clone();
        let sigma_old: f32 = state.sigma;

        // Selection steps yᵢ = (x_{(i)} − m) / σ for the μ best, plus the
        // recombination y_w = Σ wᵢ y_{(i)}.
        let mut y_sel: Vec<Vec<f32>> = Vec::with_capacity(mu);
        let mut y_w: Vec<f32> = vec![0.0; d];
        for (&idx, &w) in ranked.iter().take(mu).zip(params.weights.iter()) {
            let mut yi: Vec<f32> = vec![0.0; d];
            for i in 0..d {
                yi[i] = (pop_host[idx * d + i] - m_old[i]) / sigma_old;
                y_w[i] += w * yi[i];
            }
            y_sel.push(yi);
        }

        // New mean: m ← m + σ · y_w (cₘ = 1).
        let mut mean_new: Vec<f32> = vec![0.0; d];
        for i in 0..d {
            mean_new[i] = m_old[i] + sigma_old * y_w[i];
        }

        // C^{-1/2} = B diag(1/√Λ) Bᵀ from the eigendecomposition of the old C.
        let (eigvals, eigvecs) = jacobi_eigen(&state.cov, d);
        let floor: f32 = eigenvalue_floor(&eigvals);
        let inv_sqrt: Vec<f32> = eigvals
            .iter()
            .map(|&l| 1.0 / l.max(floor).sqrt())
            .collect();
        let mut c_inv_sqrt: Vec<f32> = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut acc: f32 = 0.0;
                for k in 0..d {
                    acc += eigvecs[i * d + k] * inv_sqrt[k] * eigvecs[j * d + k];
                }
                c_inv_sqrt[i * d + j] = acc;
            }
        }

        // Conjugate path: p_σ ← (1−c_σ) p_σ + √(c_σ(2−c_σ)μ_eff) · C^{-1/2} y_w.
        let cs_factor: f32 = (params.c_sigma * (2.0 - params.c_sigma) * params.mu_eff).sqrt();
        let c_inv_yw: Vec<f32> = matvec(&c_inv_sqrt, &y_w, d);
        let mut p_sigma: Vec<f32> = vec![0.0; d];
        for i in 0..d {
            p_sigma[i] = (1.0 - params.c_sigma) * state.p_sigma[i] + cs_factor * c_inv_yw[i];
        }
        let p_sigma_norm: f32 = p_sigma.iter().map(|v| v * v).sum::<f32>().sqrt();

        // CSA step-size update: σ ← σ · exp((c_σ/d_σ)(‖p_σ‖/χ_n − 1)). Floor at
        // the smallest positive f32 so a collapsing σ can never reach exactly
        // zero (which would make next generation's yᵢ = (xᵢ − m)/σ a 0/0 NaN).
        let sigma_new: f32 = (sigma_old
            * ((params.c_sigma / params.d_sigma) * (p_sigma_norm / params.chi_n - 1.0)).exp())
        .max(f32::MIN_POSITIVE);

        // Heaviside stall guard hσ on the anisotropic path.
        let gen_count: f32 = state.generation as f32 + 1.0;
        let denom: f32 = (1.0 - (1.0 - params.c_sigma).powf(2.0 * gen_count)).sqrt();
        let h_sigma: f32 = if p_sigma_norm / denom < (1.4 + 2.0 / (params.genome_dim as f32 + 1.0)) * params.chi_n
        {
            1.0
        } else {
            0.0
        };

        // Anisotropic path: p_c ← (1−c_c) p_c + hσ √(c_c(2−c_c)μ_eff) y_w.
        let pc_factor: f32 = (params.c_c * (2.0 - params.c_c) * params.mu_eff).sqrt();
        let mut p_c: Vec<f32> = vec![0.0; d];
        for i in 0..d {
            p_c[i] = (1.0 - params.c_c) * state.p_c[i] + h_sigma * pc_factor * y_w[i];
        }

        // Covariance update: rank-1 (p_c) + rank-μ (selected steps).
        // δ(hσ) keeps E[C] unbiased when the rank-1 term is stalled.
        let delta_h: f32 = (1.0 - h_sigma) * params.c_c * (2.0 - params.c_c);
        let c_old: Vec<f32> = state.cov.clone();
        let mut cov_new: Vec<f32> = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                let decay: f32 = 1.0 - params.c_1 - params.c_mu;
                let rank1: f32 = params.c_1 * (p_c[i] * p_c[j] + delta_h * c_old[i * d + j]);
                let mut rankmu: f32 = 0.0;
                for (rank, yi) in y_sel.iter().enumerate() {
                    rankmu += params.weights[rank] * yi[i] * yi[j];
                }
                rankmu *= params.c_mu;
                cov_new[i * d + j] = decay * c_old[i * d + j] + rank1 + rankmu;
            }
        }

        // Track the best individual this generation.
        update_best(&mut state, &population, &fitness_host);

        state.generation += 1;
        let metrics = StrategyMetrics::from_host_fitness(
            state.generation,
            &fitness_host,
            state.best_fitness,
        );
        state.best_fitness = metrics.best_fitness_ever;

        state.mean = mean_new;
        state.cov = cov_new;
        state.p_sigma = p_sigma;
        state.p_c = p_c;
        state.sigma = sigma_new;

        (state, metrics)
    }

    /// Returns the best-so-far genome and its fitness, or `None` before the
    /// first [`tell`](Self::tell) call.
    fn best(&self, state: &CmaEsState<B>) -> Option<(Tensor<B, 2>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

/// Updates `state.best_genome` / `state.best_fitness` if this generation
/// improved on the best-so-far.
fn update_best<B: Backend>(state: &mut CmaEsState<B>, pop: &Tensor<B, 2>, fitness: &[f32]) {
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
        assert!(CmaEsConfig::default_for(10).validate().is_ok());
    }

    #[test]
    fn rejects_pop_size_below_two() {
        let mut cfg = CmaEsConfig::default_for(10);
        cfg.pop_size = 1;
        assert_eq!(cfg.validate().unwrap_err().field, "pop_size");
    }

    #[test]
    fn default_config_validates_all() {
        assert!(CmaEsConfig::default_for(10).validate_all().is_ok());
    }

    #[test]
    fn rejects_desynced_weights() {
        // A hand-built literal that dropped a weight: length no longer equals μ
        // and the remaining weights no longer sum to 1.
        let mut cfg = CmaEsConfig::default_for(10);
        cfg.weights.pop();
        let err = cfg.validate().unwrap_err();
        assert_eq!(err.field, "weights");
    }

    #[test]
    fn rejects_diverging_covariance_rates() {
        let mut cfg = CmaEsConfig::default_for(10);
        // c_1 + c_mu > 1 makes the rank-update retention factor negative.
        cfg.c_1 = 0.7;
        cfg.c_mu = 0.7;
        let err = cfg.validate().unwrap_err();
        assert_eq!(err.field, "c_1_plus_c_mu");
    }

    #[test]
    fn validate_all_reports_every_violation() {
        // Desync three independent derived fields at once; fail-fast would hide
        // all but the first, validate_all surfaces them together.
        let mut cfg = CmaEsConfig::default_for(10);
        cfg.weights.pop(); // weights length + sum
        cfg.d_sigma = -1.0; // non-positive damping
        cfg.c_1 = 0.7;
        cfg.c_mu = 0.7; // c_1 + c_mu > 1
        let errs = cfg.validate_all().unwrap_err();
        let fields: Vec<&str> = errs.iter().map(|e| e.field).collect();
        assert!(fields.contains(&"weights"));
        assert!(fields.contains(&"d_sigma"));
        assert!(fields.contains(&"c_1_plus_c_mu"));
        assert!(errs.len() >= 3, "expected all violations, got {fields:?}");
        // validate() stays consistent — it is the first of these.
        assert_eq!(cfg.validate().unwrap_err(), errs[0]);
    }

    #[test]
    fn default_for_d10_constants() {
        // Hansen 2016 Table 1 reference values for D = 10.
        let cfg = CmaEsConfig::default_for(10);
        // λ = 4 + ⌊3 ln 10⌋ = 4 + ⌊6.907⌋ = 10; μ = 5.
        assert_eq!(cfg.pop_size, 10);
        assert_eq!(cfg.mu, 5);
        assert_eq!(cfg.weights.len(), 5);
        // Weights are positive, descending, and normalized.
        let sum: f32 = cfg.weights.iter().sum();
        approx::assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
        for pair in cfg.weights.windows(2) {
            assert!(pair[0] >= pair[1], "weights must be descending");
        }
        // μ_eff lies in (1, μ].
        assert!(cfg.mu_eff > 1.0 && cfg.mu_eff <= 5.0, "mu_eff = {}", cfg.mu_eff);
        // Learning rates are in their valid ranges.
        assert!(cfg.c_sigma > 0.0 && cfg.c_sigma < 1.0);
        assert!(cfg.d_sigma >= 1.0);
        assert!(cfg.c_c > 0.0 && cfg.c_c < 1.0);
        assert!(cfg.c_1 > 0.0 && cfg.c_1 < 1.0);
        assert!(cfg.c_mu > 0.0);
        assert!(cfg.c_1 + cfg.c_mu <= 1.0, "c_1 + c_mu must not exceed 1");
        // χ_n = √10·(1 − 1/40 + 1/2100) ≈ 3.0847 (just below √10 ≈ 3.162).
        approx::assert_relative_eq!(cfg.chi_n, 3.084_7_f32, epsilon = 1e-3);
    }

    #[test]
    fn with_pop_size_scales_mu() {
        let cfg = CmaEsConfig::with_pop_size(50, 10);
        assert_eq!(cfg.pop_size, 50);
        assert_eq!(cfg.mu, 25);
        let sum: f32 = cfg.weights.iter().sum();
        approx::assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
    }
}
