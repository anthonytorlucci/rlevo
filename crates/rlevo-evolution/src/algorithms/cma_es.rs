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

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate, Violations};

use crate::ops::linalg::{SymEigen, jacobi_eigen, matvec};
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
        let c_mu: f32 =
            (1.0 - c_1).min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((d + 2.0) * (d + 2.0) + mu_eff));
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
        v.check(config::positive(
            C,
            "initial_sigma",
            f64::from(self.initial_sigma),
        ));
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
        v.check(config::in_range(
            C,
            "weights",
            1.0 - 1e-3,
            1.0 + 1e-3,
            weight_sum,
        ));

        // Derived scalars. mu_eff = 1/Σwᵢ² ≥ 1; d_sigma and chi_n are positive
        // denominators/scales — a non-positive value diverges the step-size
        // control or the covariance update.
        v.check(config::in_range(
            C,
            "mu_eff",
            1.0,
            f64::INFINITY,
            f64::from(self.mu_eff),
        ));
        v.check(config::positive(C, "d_sigma", f64::from(self.d_sigma)));
        v.check(config::positive(C, "chi_n", f64::from(self.chi_n)));

        // Covariance/step-size learning rates each live in [0, 1], and the pair
        // (c_1, c_mu) must not sum past 1: the rank-update retention factor is
        // `1 − c_1 − c_mu`, so c_1 + c_mu > 1 turns it negative and the
        // covariance matrix loses positive-definiteness.
        v.check(config::in_range(
            C,
            "c_sigma",
            0.0,
            1.0,
            f64::from(self.c_sigma),
        ));
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
    mean: Vec<f32>,
    /// Covariance matrix `C`, row-major `D × D`.
    cov: Vec<f32>,
    /// Conjugate evolution path `p_σ`, length `D`.
    p_sigma: Vec<f32>,
    /// Anisotropic evolution path `p_c`, length `D`.
    p_c: Vec<f32>,
    /// Global step size `σ`.
    sigma: f32,
    /// Completed-generation counter.
    generation: usize,
    /// Best-so-far genome, shape `(1, D)`.
    best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness (canonical maximise convention).
    best_fitness: f32,
    /// Cached symmetric eigendecomposition of the **current** `cov`.
    ///
    /// The eigendecomposition is the most expensive host op per generation and
    /// is needed twice on an unchanged `C`: [`Strategy::ask`] builds the
    /// sampling transform `B·diag(√Λ)` from it, and the following
    /// [`Strategy::tell`] builds the conditioning matrix `C^{-1/2}` from the
    /// same decomposition. This field memoizes the raw
    /// [`SymEigen`](crate::ops::linalg::SymEigen) so `tell` reuses `ask`'s work.
    ///
    /// # Invariant
    ///
    /// This is a **pure memo** of the decomposition of the `cov` field as it
    /// stands *right now* — never an independent source of truth. Two rules keep
    /// it coherent:
    ///
    /// - **Any code path that writes `cov` must first clear or take this memo**
    ///   (set it to `None`, or `take()` it), so a stale decomposition of a
    ///   superseded `C` can never be read back.
    /// - **`ask` produces, never trusts.** It unconditionally recomputes the
    ///   decomposition of the current `cov` and *overwrites* this field with the
    ///   fresh result; it never reads the prior memo. `tell` is the sole
    ///   consumer — it `take()`s the memo (falling back to a fresh
    ///   `jacobi_eigen` if a state skipped `ask`).
    ///
    /// Because `jacobi_eigen` is deterministic, reusing the memo is bit-identical
    /// to recomputing it, so the cache is transparent to same-seed determinism.
    eig: Option<SymEigen>,
}

impl<B: Backend> CmaEsState<B> {
    /// Assembles a CMA-ES state, checking the distribution parameters are
    /// dimensionally consistent.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `mean` is empty, if `cov` is not `D × D`
    /// row-major (`D = mean.len()`), if `p_sigma` or `p_c` differs from `D`,
    /// or if `sigma` is not strictly positive and finite.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        mean: Vec<f32>,
        mut cov: Vec<f32>,
        p_sigma: Vec<f32>,
        p_c: Vec<f32>,
        sigma: f32,
        generation: usize,
        best_genome: Option<Tensor<B, 2>>,
        best_fitness: f32,
    ) -> Result<Self, ConfigError> {
        let d = mean.len();
        config::nonzero("CmaEsState", "mean", d)?;
        if cov.len() != d * d {
            return Err(ConfigError {
                config: "CmaEsState",
                field: "cov",
                kind: ConstraintKind::Custom("covariance must be a row-major D × D matrix"),
            });
        }
        if p_sigma.len() != d {
            return Err(ConfigError {
                config: "CmaEsState",
                field: "p_sigma",
                kind: ConstraintKind::Custom("evolution path length must equal D"),
            });
        }
        if p_c.len() != d {
            return Err(ConfigError {
                config: "CmaEsState",
                field: "p_c",
                kind: ConstraintKind::Custom("evolution path length must equal D"),
            });
        }
        config::positive("CmaEsState", "sigma", f64::from(sigma))?;
        // Normalize a caller-supplied `cov` to exact symmetry: the
        // eigendecomposition the strategy runs on `C` assumes symmetry. The
        // in-loop rank-1 / rank-μ updates preserve symmetry only up to
        // floating-point rounding (a few ULPs): the rank-μ accumulation forms
        // `(w · yi[i]) · yi[j]` for the (i,j) entry but `(w · yi[j]) · yi[i]`
        // for its transpose, which are equal under commutativity but *not*
        // associativity, so the two triangle entries can diverge slightly (see
        // the `cma_es_drive_preserves_invariants` property test's rationale).
        // This `try_new` symmetrization still averages caller-supplied triangles
        // (pycma-style) — better than a tolerance-based rejection, mirroring the
        // sanitize-at-the-chokepoint convention of ADR 0034 rather than pushing
        // the problem back onto the caller.
        crate::ops::linalg::symmetrize(&mut cov, d);
        Ok(Self {
            mean,
            cov,
            p_sigma,
            p_c,
            sigma,
            generation,
            best_genome,
            best_fitness,
            // Internal cache state, not caller-suppliable: a freshly
            // constructed state has no decomposition memoized yet; the first
            // `ask` produces one.
            eig: None,
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

    /// Conjugate evolution path `p_σ`, length `D`.
    #[must_use]
    pub fn p_sigma(&self) -> &[f32] {
        &self.p_sigma
    }

    /// Anisotropic evolution path `p_c`, length `D`.
    #[must_use]
    pub fn p_c(&self) -> &[f32] {
        &self.p_c
    }

    /// Global step size `σ`.
    #[must_use]
    pub fn sigma(&self) -> f32 {
        self.sigma
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
        debug_assert!(
            params.validate().is_ok(),
            "invalid CmaEsConfig reached init: {params:?}"
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
        CmaEsState {
            mean,
            cov,
            p_sigma: vec![0.0; d],
            p_c: vec![0.0; d],
            sigma: params.initial_sigma,
            generation: 0,
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
            // No decomposition memoized yet; the first `ask` produces one.
            eig: None,
        }
    }

    /// Samples `λ` offspring from `N(m, σ²C)`.
    ///
    /// The covariance is eigendecomposed into `C = B diag(Λ) Bᵀ`; each
    /// offspring is `xᵢ = m + σ · B diag(√Λ) zᵢ` for `zᵢ ~ N(0, I)`, drawn
    /// host-side from a deterministic [`SeedPurpose::CmaSampling`] stream. The
    /// distribution parameters are returned unchanged (the mean/covariance
    /// update happens in [`tell`](Self::tell), which recomputes the steps from
    /// the population).
    ///
    /// The one thing `ask` *does* mutate on the returned state is the
    /// eigendecomposition memo ([`CmaEsState::eig`]): it stores the fresh
    /// decomposition of the current `C` so the paired `tell` reuses it to build
    /// `C^{-1/2}` instead of decomposing the same unchanged matrix a second
    /// time. `ask` produces the memo and never trusts a prior one.
    fn ask(
        &self,
        params: &CmaEsConfig,
        state: &CmaEsState<B>,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Tensor<B, 2>, CmaEsState<B>) {
        let d = params.genome_dim;
        let lambda = params.pop_size;

        // Sampling transform B·diag(√Λ) from the eigendecomposition of C. The
        // raw decomposition is kept whole (not destructured) so it can be
        // memoized on the returned state for `tell` to reuse. The eigenvalue
        // floor is applied *here* per-use — `ask` needs `√Λ`, `tell` needs
        // `1/√Λ`, so only the raw values are cached and each site floors them.
        let eig: SymEigen = jacobi_eigen(&state.cov, d);
        let floor: f32 = eigenvalue_floor(&eig.values);
        let mut bd: Vec<f32> = vec![0.0; d * d];
        for i in 0..d {
            for k in 0..d {
                bd[i * d + k] = eig.vectors[i * d + k] * eig.values[k].max(floor).sqrt();
            }
        }

        let mut stream = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::CmaSampling,
        );
        let mut rows: Vec<f32> = Vec::with_capacity(lambda * d);
        for _ in 0..lambda {
            let z: Vec<f32> = (0..d)
                .map(|_| crate::sampling::standard_normal(&mut stream))
                .collect();
            let bdz: Vec<f32> = matvec(&bd, &z, d);
            for (mean_i, bdz_i) in state.mean.iter().zip(bdz.iter()) {
                rows.push(mean_i + state.sigma * bdz_i);
            }
        }
        let population = Tensor::<B, 2>::from_data(TensorData::new(rows, [lambda, d]), device);
        // Clone first, then overwrite the memo on the clone: the decomposition
        // just built is exactly the decomposition of this state's (unchanged)
        // `cov`, so it is a valid memo for the paired `tell` to consume.
        let mut next = state.clone();
        next.eig = Some(eig);
        (population, next)
    }

    /// Ranks the offspring, recombines the mean, and runs CSA + the rank-1 /
    /// rank-μ covariance updates.
    ///
    /// # Lost generations
    ///
    /// The rank-μ update needs `μ` *usable* selection steps. Ranking already
    /// sanitizes (`NaN → −∞`) and sorts with `total_cmp`, so a non-finite
    /// fitness can never rank among the best — but if **fewer than `μ`**
    /// sanitized values are finite, non-usable individuals would still fill out
    /// the selected `μ` and feed meaningless steps `yᵢ = (xᵢ − m)/σ` into the
    /// mean and covariance updates. When that happens `tell` takes a deliberate
    /// **lost generation**: the entire adaptive update (mean, `C`, `p_σ`, `p_c`,
    /// `σ`, and the eigendecomposition memo) is skipped and the search
    /// distribution is left exactly unchanged. A legitimate `−∞` counts as
    /// non-usable here — it marks a member evaluation that broke, so it cannot
    /// contribute a meaningful recombination step.
    ///
    /// A lost generation still **advances the generation counter and updates
    /// best-so-far tracking**. Advancing the counter matters for determinism:
    /// the per-generation sampling stream is keyed on
    /// `seed_stream(_, generation, _)`, so bumping it ensures the next `ask`
    /// draws a *fresh* offspring batch rather than replaying the identical draw
    /// that just failed. The retained eigendecomposition memo stays coherent
    /// because `cov` is untouched.
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

        // Best-tracking (`update_best`, below) reads this raw fitness directly
        // and relies on the harness-side sanitize chokepoint (ADR 0034) to have
        // already mapped `+∞ → f32::MAX` before `tell`; that `+∞` hygiene is
        // pre-existing and out of scope here. The adaptive update below reads
        // only the locally-sanitized `sane` copy.
        let fitness_host: Vec<f32> = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness tensor must be readable as f32");
        let pop_host: Vec<f32> = population
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("population tensor must be readable as f32");

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

        // Lost-generation guard: the rank-μ update needs μ *usable* (finite)
        // steps. If fewer than μ sanitized values are finite, the selected μ
        // would include non-usable members (`−∞`, a sanitized `NaN`, or a
        // broken `−∞` evaluation) whose steps corrupt the mean/covariance
        // update. Freeze the whole search distribution — mean, `C`, `p_σ`,
        // `p_c`, `σ`, and the eig memo all stay untouched (the retained memo
        // remains coherent because `cov` is unchanged) — but still advance the
        // generation counter (so the next `ask` draws a fresh stream, not a
        // replay) and best-so-far tracking. See the `# Lost generations` doc
        // section above.
        let n_finite: usize = sane.iter().filter(|f| f.is_finite()).count();
        if n_finite < mu {
            update_best(&mut state, &population, &fitness_host);
            state.generation += 1;
            let metrics = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = metrics.best_fitness_ever();
            return (state, metrics);
        }

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
        // Reuse the memo `ask` stored for this exact (unchanged) `C`; `take()`
        // it so the stale decomposition cannot outlive the `cov` overwrite at
        // the end of this method. The fallback keeps `tell` correct for a state
        // that reached here without a paired `ask`. The floor is applied here
        // as `1/√Λ` (vs `ask`'s `√Λ`), so only the raw eigenvalues are cached.
        let SymEigen {
            values: eigvals,
            vectors: eigvecs,
        } = state
            .eig
            .take()
            .unwrap_or_else(|| jacobi_eigen(&state.cov, d));
        let floor: f32 = eigenvalue_floor(&eigvals);
        let inv_sqrt: Vec<f32> = eigvals.iter().map(|&l| 1.0 / l.max(floor).sqrt()).collect();
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
        let h_sigma: f32 = if p_sigma_norm / denom
            < (1.4 + 2.0 / (params.genome_dim as f32 + 1.0)) * params.chi_n
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
        let metrics =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = metrics.best_fitness_ever();

        state.mean = mean_new;
        // Overwrites `cov`; the eig memo was already `take()`n above, so there
        // is no stale-decomposition hazard — `state.eig` is `None` on return
        // and the next `ask` will produce a fresh memo for this new `C`.
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
    use burn::backend::Flex;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn try_new_checks_dimensions() {
        // D = 2: cov is 2×2 = 4 entries, both paths length 2, σ > 0.
        assert!(
            CmaEsState::<Flex>::try_new(
                vec![0.0, 0.0],
                vec![1.0, 0.0, 0.0, 1.0],
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                0.5,
                0,
                None,
                f32::MIN,
            )
            .is_ok()
        );
        // cov length 3 ≠ D·D.
        assert!(
            CmaEsState::<Flex>::try_new(
                vec![0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                0.5,
                0,
                None,
                f32::MIN,
            )
            .is_err()
        );
        // Non-positive σ.
        assert!(
            CmaEsState::<Flex>::try_new(
                vec![0.0, 0.0],
                vec![1.0, 0.0, 0.0, 1.0],
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                0.0,
                0,
                None,
                f32::MIN,
            )
            .is_err()
        );
    }

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
        assert!(
            cfg.mu_eff > 1.0 && cfg.mu_eff <= 5.0,
            "mu_eff = {}",
            cfg.mu_eff
        );
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

    /// Lost generation: with fewer than μ finite fitness values, `tell` must
    /// freeze the entire search distribution (mean, `C`, `σ`, both paths) yet
    /// still advance the generation counter and best-so-far tracking.
    #[test]
    fn tell_freezes_distribution_on_too_few_finite() {
        let strategy = CmaEs::<Flex>::new();
        let params = CmaEsConfig::with_pop_size(6, 2); // μ = 3.
        assert_eq!(params.mu, 3);
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(0xF10E);

        let state = strategy.init(&params, &mut rng, &device);
        let (population, asked) = strategy.ask(&params, &state, &mut rng, &device);

        // Snapshot the pre-`tell` distribution (all bit-exact).
        let mean0: Vec<f32> = asked.mean().to_vec();
        let cov0: Vec<f32> = asked.cov().to_vec();
        let p_sigma0: Vec<f32> = asked.p_sigma().to_vec();
        let p_c0: Vec<f32> = asked.p_c().to_vec();
        let sigma0: f32 = asked.sigma();
        let gen0: usize = asked.generation();

        // Only one finite value; μ = 3 → lost generation.
        let fitness = Tensor::<Flex, 1>::from_data(
            TensorData::new(
                vec![1.0f32, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN],
                [6],
            ),
            &device,
        );
        let (told, _metrics) = strategy.tell(&params, population, fitness, asked, &mut rng);

        // Distribution frozen, bit-for-bit.
        assert_eq!(told.mean(), mean0.as_slice());
        assert_eq!(told.cov(), cov0.as_slice());
        assert_eq!(told.p_sigma(), p_sigma0.as_slice());
        assert_eq!(told.p_c(), p_c0.as_slice());
        assert_eq!(told.sigma().to_bits(), sigma0.to_bits());
        // Counter advanced; best tracked from the single finite value.
        assert_eq!(told.generation(), gen0 + 1);
        assert_eq!(told.best_fitness().to_bits(), 1.0f32.to_bits());
    }

    /// Cache coherence: `tell` reusing the eigendecomposition memo `ask` stored
    /// produces a state bit-identical to `tell` on an equivalent state whose
    /// memo is absent (rebuilt via `try_new`, which recomputes the
    /// decomposition). `jacobi_eigen` is deterministic, so the two must agree.
    #[test]
    fn tell_cache_reuse_matches_recompute() {
        let strategy = CmaEs::<Flex>::new();
        let params = CmaEsConfig::with_pop_size(6, 2);
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(0x00CA_C4E5);

        let state = strategy.init(&params, &mut rng, &device);
        let (population, asked) = strategy.ask(&params, &state, &mut rng, &device);

        // Rebuild an equivalent state from the asked-state accessors. `try_new`
        // never populates the memo, so its `tell` recomputes the decomposition.
        let rebuilt = CmaEsState::<Flex>::try_new(
            asked.mean().to_vec(),
            asked.cov().to_vec(),
            asked.p_sigma().to_vec(),
            asked.p_c().to_vec(),
            asked.sigma(),
            asked.generation(),
            asked.best_genome().cloned(),
            asked.best_fitness(),
        )
        .expect("valid state");

        // Identical fitness (≥ μ finite → full adaptive update runs).
        let fitness_vals: Vec<f32> = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let f_cached =
            Tensor::<Flex, 1>::from_data(TensorData::new(fitness_vals.clone(), [6]), &device);
        let f_recomp = Tensor::<Flex, 1>::from_data(TensorData::new(fitness_vals, [6]), &device);

        // `tell` ignores its `_rng`; fresh RNGs are only for the signature.
        let mut rng_a = StdRng::seed_from_u64(1);
        let mut rng_b = StdRng::seed_from_u64(2);
        let (told_cached, _) =
            strategy.tell(&params, population.clone(), f_cached, asked, &mut rng_a);
        let (told_recomp, _) = strategy.tell(&params, population, f_recomp, rebuilt, &mut rng_b);

        assert_eq!(told_cached.mean(), told_recomp.mean());
        assert_eq!(told_cached.cov(), told_recomp.cov());
        assert_eq!(told_cached.p_sigma(), told_recomp.p_sigma());
        assert_eq!(told_cached.p_c(), told_recomp.p_c());
        assert_eq!(told_cached.sigma().to_bits(), told_recomp.sigma().to_bits());
    }

    /// `try_new` normalizes a caller-supplied asymmetric covariance to exact
    /// symmetry by averaging the triangles (pycma-style construction boundary).
    #[test]
    fn try_new_symmetrizes_covariance() {
        // Off-diagonals (0,1) = 0.4 and (1,0) = 0.2 → both become 0.3.
        let state = CmaEsState::<Flex>::try_new(
            vec![0.0, 0.0],
            vec![1.0, 0.4, 0.2, 1.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            0.5,
            0,
            None,
            f32::NEG_INFINITY,
        )
        .expect("valid state");
        let cov: &[f32] = state.cov();
        approx::assert_relative_eq!(cov[1], 0.3, epsilon = 1e-6);
        approx::assert_relative_eq!(cov[2], 0.3, epsilon = 1e-6);
    }

    /// Memo-hygiene: a full adaptive `tell` overwrites `cov`, so it must leave
    /// the eigendecomposition memo empty. Locks the "any `cov` write clears the
    /// memo" invariant against a future refactor that adds a second
    /// cov-mutation path but forgets to `take()`/clear `eig`.
    #[test]
    fn tell_clears_eig_memo_after_cov_update() {
        let strategy = CmaEs::<Flex>::new();
        let params = CmaEsConfig::with_pop_size(6, 2);
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(0x00EE_6011);

        let state = strategy.init(&params, &mut rng, &device);
        let (population, asked) = strategy.ask(&params, &state, &mut rng, &device);
        // `ask` produced a memo for this state.
        assert!(asked.eig.is_some(), "ask must populate the eig memo");

        // ≥ μ finite → full adaptive update runs and overwrites `cov`.
        let fitness = Tensor::<Flex, 1>::from_data(
            TensorData::new(vec![6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0], [6]),
            &device,
        );
        let (told, _metrics) = strategy.tell(&params, population, fitness, asked, &mut rng);

        assert!(
            told.eig.is_none(),
            "a cov-mutating tell must leave the eig memo empty"
        );
    }

    /// Two-generation sequential drive: init → ask → tell → ask → tell. The
    /// second `ask` must produce a *fresh* memo (of the first `tell`'s new `C`),
    /// and the second `tell` must consume it and leave the search distribution
    /// finite.
    #[test]
    fn two_generation_sequence_refreshes_memo() {
        let strategy = CmaEs::<Flex>::new();
        let params = CmaEsConfig::with_pop_size(6, 2);
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(0x00A2_9E11);

        let fitness = |dev: &_| {
            Tensor::<Flex, 1>::from_data(
                TensorData::new(vec![6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0], [6]),
                dev,
            )
        };

        // Generation 1.
        let s0 = strategy.init(&params, &mut rng, &device);
        let (pop0, asked0) = strategy.ask(&params, &s0, &mut rng, &device);
        assert!(asked0.eig.is_some(), "first ask must populate the memo");
        let (told0, _m0) = strategy.tell(&params, pop0, fitness(&device), asked0, &mut rng);
        assert!(told0.eig.is_none(), "first tell must clear the memo");

        // Generation 2: a fresh memo of the updated `C`.
        let (pop1, asked1) = strategy.ask(&params, &told0, &mut rng, &device);
        assert!(
            asked1.eig.is_some(),
            "second ask must build a fresh memo off the updated cov"
        );
        let (told1, _m1) = strategy.tell(&params, pop1, fitness(&device), asked1, &mut rng);
        assert!(told1.eig.is_none(), "second tell must clear the memo");

        // The distribution stayed finite across both generations.
        assert!(told1.mean().iter().all(|v| v.is_finite()), "mean finite");
        assert!(told1.cov().iter().all(|v| v.is_finite()), "cov finite");
        assert!(told1.sigma().is_finite(), "sigma finite");
    }

    /// Issue #147 §7.2: a full adaptive `tell` must leave `C` symmetric and
    /// positive-definite. The rank-1 / rank-μ update preserves symmetry only up
    /// to a few ULPs (the transposed triangle entries accumulate the same
    /// factors in a different order, and float multiplication is not
    /// associative); this single-seed test happens to round identically for its
    /// specific seed/dims, so it can assert exact equality — the general
    /// relative-tolerance guarantee is covered by the
    /// `cma_es_drive_preserves_invariants` property. PD is checked via a
    /// symmetric eigendecomposition (all eigenvalues strictly positive), which
    /// is exactly the property `ask`'s `√Λ` sampling and `tell`'s `C^{-1/2}`
    /// conditioning rely on.
    #[test]
    fn tell_keeps_covariance_symmetric_and_positive_definite() {
        let strategy = CmaEs::<Flex>::new();
        let params = CmaEsConfig::with_pop_size(6, 3);
        let d: usize = params.genome_dim;
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(0x5EED_C0DE);

        let state = strategy.init(&params, &mut rng, &device);
        let (population, asked) = strategy.ask(&params, &state, &mut rng, &device);
        let fitness = Tensor::<Flex, 1>::from_data(
            TensorData::new(vec![6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0], [6]),
            &device,
        );
        let (told, _metrics) = strategy.tell(&params, population, fitness, asked, &mut rng);

        let cov: &[f32] = told.cov();
        // Symmetric (bit-exact by construction).
        for i in 0..d {
            for j in 0..d {
                assert_eq!(
                    cov[i * d + j].to_bits(),
                    cov[j * d + i].to_bits(),
                    "asymmetry at ({i}, {j})"
                );
            }
        }
        // Positive-definite: every eigenvalue strictly positive.
        let eig: SymEigen = jacobi_eigen(cov, d);
        assert!(
            eig.values.iter().all(|&l| l > 0.0),
            "covariance not positive-definite: eigenvalues {:?}",
            eig.values
        );
        // Diagonal (the variances) is strictly positive too.
        for i in 0..d {
            assert!(cov[i * d + i] > 0.0, "non-positive variance at {i}");
        }
    }

    /// Issue #147 §7.2 best-tracking: `best()` is `None` before any `tell`, and
    /// `Some((genome, fitness))` after — reporting the highest-fitness offspring
    /// (canonical maximise) with the correct `(1, D)` genome shape.
    #[test]
    fn best_is_none_before_tell_and_some_after() {
        let strategy = CmaEs::<Flex>::new();
        let params = CmaEsConfig::with_pop_size(6, 2);
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(0xB357_7E57);

        let state = strategy.init(&params, &mut rng, &device);
        assert!(
            strategy.best(&state).is_none(),
            "best must be None before the first tell"
        );

        let (population, asked) = strategy.ask(&params, &state, &mut rng, &device);
        let fitness = Tensor::<Flex, 1>::from_data(
            TensorData::new(vec![6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0], [6]),
            &device,
        );
        let (told, _metrics) = strategy.tell(&params, population, fitness, asked, &mut rng);

        let best = strategy.best(&told).expect("best is Some after a tell");
        let (genome, fit): (Tensor<Flex, 2>, f32) = best;
        approx::assert_relative_eq!(fit, 6.0, epsilon = 1e-6);
        assert_eq!(genome.dims(), [1, 2]);
    }

    /// Issue #147 §7.2 eigenvalue-floor clamp: a degenerate (exactly zero)
    /// eigenvalue is floored to the relative floor `λ_max · CONDITION_FLOOR`,
    /// strictly above zero, so `√Λ` and `1/√Λ` both stay finite. Without the
    /// floor the `1/√Λ` used in `tell`'s `C^{-1/2}` would diverge to `+∞`.
    #[test]
    fn eigenvalue_floor_clamps_degenerate_eigenvalue() {
        // λ_max = 1, one exactly-zero eigenvalue.
        let eigvals: Vec<f32> = vec![1.0, 0.0];
        let floor: f32 = eigenvalue_floor(&eigvals);
        // Relative floor dominates the absolute backstop: 1·1e-14 > 1e-20.
        assert_eq!(floor.to_bits(), CONDITION_FLOOR.to_bits());
        assert!(floor > EIGENVALUE_FLOOR);

        // The zero eigenvalue is lifted strictly above zero.
        let clamped: f32 = eigvals[1].max(floor);
        assert!(clamped > 0.0, "floored eigenvalue must be positive");
        assert!(clamped.sqrt().is_finite(), "√Λ must be finite");
        assert!((1.0 / clamped.sqrt()).is_finite(), "1/√Λ must be finite");

        // Contrast: the un-floored zero eigenvalue would diverge under 1/√Λ.
        assert!(
            !(1.0f32 / eigvals[1].sqrt()).is_finite(),
            "un-floored 1/√0 must diverge — proves the floor is load-bearing"
        );
    }

    /// Issue #147 §7.2: `update_best` on an empty population is a no-op — it
    /// short-circuits before touching the population tensor, leaving best-so-far
    /// tracking untouched (no panic, no spurious best).
    #[test]
    fn update_best_empty_population_is_noop() {
        let strategy = CmaEs::<Flex>::new();
        let params = CmaEsConfig::with_pop_size(6, 2);
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(0x0E11_0E11);

        let mut state = strategy.init(&params, &mut rng, &device);
        // Any population tensor; the empty fitness slice short-circuits before it
        // is read.
        let pop = Tensor::<Flex, 2>::from_data(TensorData::new(vec![0.0f32, 0.0], [1, 2]), &device);
        update_best(&mut state, &pop, &[]);

        assert!(
            state.best_genome().is_none(),
            "empty population must not set a best genome"
        );
        assert_eq!(
            state.best_fitness().to_bits(),
            f32::NEG_INFINITY.to_bits(),
            "empty population must not move best fitness off its sentinel"
        );
    }

    proptest! {
        // Backend-heavy property: each case instantiates `Flex` and runs several
        // full generations, so the case count and shrink budget are capped to
        // keep CI cost bounded (task §239 §7.3).
        #![proptest_config(ProptestConfig {
            cases: 16,
            max_shrink_iters: 256,
            ..ProptestConfig::default()
        })]

        /// Issue #239 §7.3: across a bounded `(λ, D, seed)` space, a full
        /// `init → ask → tell` drive over several generations preserves the
        /// CMA-ES structural invariants — offspring shape `[λ, D]`, bit-exact
        /// covariance symmetry, positive-definiteness (every eigenvalue and
        /// diagonal variance strictly positive), a finite search distribution,
        /// and the `best()` lifecycle (`None` before the first `tell`, then a
        /// `Some((genome, fit))` with a `[1, D]` genome).
        ///
        /// RNG boundary (ADR 0029): proptest samples *only* host config; the
        /// algorithm draws from a seeded `StdRng`, so proptest's PRNG never
        /// touches Burn and every assertion is thread-count-invariant.
        #[test]
        fn cma_es_drive_preserves_invariants(
            lambda in 2usize..=64,
            d in 1usize..=20,
            seed in any::<u64>(),
        ) {
            let strategy = CmaEs::<Flex>::new();
            let params = CmaEsConfig::with_pop_size(lambda, d);
            // Restrict the sampled `(λ, D)` box to the valid-config subset: in
            // the small-`D` / large-`λ` corner the derived `c_1 + c_mu` rounds
            // fractionally past 1.0, which `validate()` rejects. We only drive
            // valid configs here; the `Err` path is covered by dedicated tests.
            prop_assume!(params.validate().is_ok());
            let device = Default::default();
            let mut rng = StdRng::seed_from_u64(seed);

            // Synthetic strictly-descending fitness of length λ (canonical
            // maximise: row 0 is the fittest offspring).
            // Precision loss is irrelevant — these are small ordinal ranks used
            // only for ordering, never compared for exact magnitude.
            #[allow(clippy::cast_precision_loss)]
            let fitness_vals: Vec<f32> = (0..lambda).map(|i| (lambda - i) as f32).collect();

            let mut state = strategy.init(&params, &mut rng, &device);
            // best() lifecycle: `None` before the first `tell`.
            prop_assert!(
                strategy.best(&state).is_none(),
                "best must be None before the first tell"
            );

            for _generation in 0..4 {
                let (population, asked) = strategy.ask(&params, &state, &mut rng, &device);
                // Invariant 1: `ask` yields exactly `[λ, D]` offspring.
                prop_assert_eq!(population.dims(), [lambda, d], "ask output shape");

                let fitness = Tensor::<Flex, 1>::from_data(
                    TensorData::new(fitness_vals.clone(), [lambda]),
                    &device,
                );
                let (told, _metrics) =
                    strategy.tell(&params, population, fitness, asked, &mut rng);

                let cov: &[f32] = told.cov();
                // Invariant 2: covariance is symmetric. The single-seed test
                // asserts *bit-exact* symmetry, but that holds only for inputs
                // where the rank-μ accumulation happens to round identically:
                // `params.weights[rank] * yi[i] * yi[j]` parses as
                // `(w · yi[i]) · yi[j]`, whose transpose `(w · yi[j]) · yi[i]`
                // is equal under commutativity but *not* associativity, so the
                // two triangle entries can diverge by a few ULPs. Across the
                // sampled space we therefore assert tight relative symmetry.
                for i in 0..d {
                    for j in 0..d {
                        prop_assert!(
                            approx::relative_eq!(
                                cov[i * d + j],
                                cov[j * d + i],
                                epsilon = 1e-6,
                                max_relative = 1e-4
                            ),
                            "asymmetry at ({}, {}): {} vs {}",
                            i,
                            j,
                            cov[i * d + j],
                            cov[j * d + i]
                        );
                    }
                }
                // Invariant 3: positive-definite — every eigenvalue and every
                // diagonal variance is strictly positive.
                let eig: SymEigen = jacobi_eigen(cov, d);
                prop_assert!(
                    eig.values.iter().all(|&l| l > 0.0),
                    "covariance not positive-definite: eigenvalues {:?}",
                    eig.values
                );
                for i in 0..d {
                    prop_assert!(cov[i * d + i] > 0.0, "non-positive variance at {}", i);
                }

                // Invariant 4: the search distribution stays finite.
                prop_assert!(told.mean().iter().all(|v| v.is_finite()), "mean finite");
                prop_assert!(told.cov().iter().all(|v| v.is_finite()), "cov finite");
                prop_assert!(told.sigma().is_finite(), "sigma finite");

                // Invariant 5: `best()` is `Some` with a `[1, D]` genome after a
                // `tell`.
                let best = strategy.best(&told);
                prop_assert!(best.is_some(), "best must be Some after a tell");
                let (genome, fit): (Tensor<Flex, 2>, f32) =
                    best.expect("best is Some after a tell");
                prop_assert!(fit.is_finite(), "best fitness finite");
                prop_assert_eq!(genome.dims(), [1, d], "best genome shape");

                state = told;
            }
        }
    }
}
