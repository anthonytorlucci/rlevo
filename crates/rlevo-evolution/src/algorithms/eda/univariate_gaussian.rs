//! Univariate Gaussian model (UMDA — Univariate Marginal Distribution
//! Algorithm) for continuous search spaces.
//!
//! Each dimension is modelled by an independent Gaussian. [`fit`] performs an
//! unweighted maximum-likelihood estimate over the `k` selected rows: the
//! per-column mean and variance are computed via `÷k` (not `÷(k-1)`), and the
//! variance is floored at [`UnivariateGaussianParams::min_variance`] to prevent
//! collapse to a point mass. The `fitness` tensor is accepted by the
//! [`ProbabilityModel`] interface but ignored; the fit is unweighted.
//! [`sample`] draws each gene from its dimension's fitted Gaussian using the
//! supplied host RNG (never `Tensor::random` / `B::seed`).
//!
//! Cross-dimension dependencies are not captured — for those, see
//! [`super::dependency_chain`].
//!
//! # References
//!
//! - Mühlenbein & Paaß (1996), *From recombination of genes to the
//!   estimation of distributions I. Binary parameters*.
//!
//! [`fit`]: crate::ProbabilityModel::fit
//! [`sample`]: crate::ProbabilityModel::sample

use burn::tensor::{Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand_distr::{Distribution as _, Normal};
use rlevo_core::config::{self, ConfigError, ConstraintKind};

use crate::probability_model::ProbabilityModel;

/// Per-run configuration for the [`UnivariateGaussian`] model.
///
/// Held inside [`EdaParams::model`](crate::algorithms::eda::EdaParams::model)
/// for the lifetime of a run. All fields are `pub` for struct-literal
/// construction; use [`UnivariateGaussianParams::default_for`] for typical
/// continuous-optimisation defaults.
#[derive(Debug, Clone)]
pub struct UnivariateGaussianParams {
    /// Number of genes per genome; determines the length of
    /// [`UnivariateGaussianState::mean`] and [`UnivariateGaussianState::variance`].
    pub genome_dim: usize,
    /// Prior mean for every dimension, used when `prev = None`.
    pub init_mean: f32,
    /// Prior standard deviation for every dimension, used when `prev = None`.
    /// The prior variance is `init_std²`.
    pub init_std: f32,
    /// Minimum variance for any dimension; prevents the model from collapsing
    /// to a point mass. The MLE estimate is floored at this value after each
    /// [`ProbabilityModel::fit`] call.
    pub min_variance: f32,
}

impl UnivariateGaussianParams {
    /// Sensible defaults for a `genome_dim`-dimensional continuous problem.
    #[must_use]
    pub fn default_for(genome_dim: usize) -> Self {
        Self {
            genome_dim,
            init_mean: 0.0,
            init_std: 2.0,
            min_variance: 1e-6,
        }
    }
}

/// Fitted statistics for the [`UnivariateGaussian`] model after one call to
/// [`ProbabilityModel::fit`].
///
/// Both vectors have length `genome_dim`. On the prior path (`prev = None`)
/// they are initialised from [`UnivariateGaussianParams::init_mean`] /
/// `init_std²`; on subsequent calls they hold the unweighted MLE estimates
/// computed from the truncation-selected population.
///
/// Fields are private so a mismatched `mean` / `variance` length (or a
/// negative / non-finite variance) is unrepresentable from outside this
/// module; use [`try_new`](UnivariateGaussianState::try_new) to build one and
/// the [`mean`](UnivariateGaussianState::mean) /
/// [`variance`](UnivariateGaussianState::variance) accessors to read it.
#[derive(Debug, Clone)]
pub struct UnivariateGaussianState {
    /// Per-dimension MLE mean.
    mean: Vec<f32>,
    /// Per-dimension MLE variance, floored at
    /// [`UnivariateGaussianParams::min_variance`].
    variance: Vec<f32>,
}

impl UnivariateGaussianState {
    /// Builds a fitted Gaussian state from per-dimension `mean` and `variance`.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `mean` is empty, if `mean` and `variance`
    /// differ in length, or if any variance is negative or non-finite. A
    /// variance is a squared deviation, so it must be `>= 0` and finite.
    pub fn try_new(mean: Vec<f32>, variance: Vec<f32>) -> Result<Self, ConfigError> {
        config::nonzero("UnivariateGaussianState", "mean", mean.len())?;
        if mean.len() != variance.len() {
            return Err(ConfigError {
                config: "UnivariateGaussianState",
                field: "variance",
                kind: ConstraintKind::Custom("mean and variance must have equal length"),
            });
        }
        if variance.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(ConfigError {
                config: "UnivariateGaussianState",
                field: "variance",
                kind: ConstraintKind::Custom("every variance must be finite and non-negative"),
            });
        }
        Ok(Self { mean, variance })
    }

    /// Per-dimension MLE means, length `genome_dim`.
    #[must_use]
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    /// Per-dimension MLE variances, length `genome_dim`.
    #[must_use]
    pub fn variance(&self) -> &[f32] {
        &self.variance
    }
}

/// Univariate Marginal Distribution Algorithm for continuous spaces (UMDA).
///
/// Implements [`ProbabilityModel`] with an unweighted per-dimension Gaussian
/// fit (`÷k` MLE variance, [`UnivariateGaussianParams::min_variance`] floor)
/// and independent per-dimension Gaussian sampling via the host RNG.
/// The fitness tensor passed to [`ProbabilityModel::fit`] is accepted but
/// ignored; the estimate is always unweighted.
///
/// See the [module docs](self) for the algorithm description and references.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnivariateGaussian;

impl<B: Backend> ProbabilityModel<B> for UnivariateGaussian {
    type Params = UnivariateGaussianParams;
    type State = UnivariateGaussianState;

    /// Fit per-dimension Gaussian statistics to the selected population.
    ///
    /// When `prev = None` returns the prior (length-`genome_dim` vectors of
    /// `init_mean` and `init_std²`); `population` and `fitness` are ignored
    /// on that path. Otherwise computes the unweighted `÷k` MLE mean and
    /// variance for every column and floors each variance at `min_variance`.
    /// The `fitness` argument is accepted but always ignored.
    ///
    /// # Panics
    ///
    /// Does not panic. The `expect` inside `sample` (not `fit`) is guarded
    /// by the variance floor, which guarantees a positive, finite standard
    /// deviation for every dimension.
    fn fit(
        &self,
        params: &Self::Params,
        prev: Option<&Self::State>,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::State {
        let _ = device;
        // Fitness is accepted but ignored: the MLE fit is unweighted.
        let _ = fitness;
        let Some(_prev) = prev else {
            // Prior path: build purely from params, ignore population/fitness.
            let d = params.genome_dim;
            return UnivariateGaussianState {
                mean: vec![params.init_mean; d],
                variance: vec![params.init_std * params.init_std; d],
            };
        };
        // `prev`'s contents are unused on this path — UMDA is a full refit.

        let [k, d] = population.dims();
        let rows = population
            .into_data()
            .into_vec::<f32>()
            .expect("population tensor must be readable as f32");
        // k is a selected-population count, far below f32's 2^24 exact-integer
        // limit; the cast is lossless in practice.
        #[allow(clippy::cast_precision_loss)]
        let kf = k as f32;

        let mut mean = vec![0.0_f32; d];
        for i in 0..k {
            for j in 0..d {
                mean[j] += rows[i * d + j];
            }
        }
        for m in &mut mean {
            let mu = *m / kf;
            // `Normal::new` accepts any mean, so a single non-finite genome value
            // (common from divergent DRL rollouts) would propagate into every
            // sample for this dimension and silently poison the search. Fall back
            // to the prior mean. (The `tell` chokepoint now also sanitizes the
            // population as a coarse backstop; this is the precise per-dimension
            // guard for direct `fit` callers.)
            *m = if mu.is_finite() { mu } else { params.init_mean };
        }

        let mut variance = vec![0.0_f32; d];
        for i in 0..k {
            for j in 0..d {
                let diff = rows[i * d + j] - mean[j];
                variance[j] += diff * diff;
            }
        }
        for v in &mut variance {
            // Floor the lower bound AND reject non-finite estimates. `f32::max`
            // suppresses `NaN` (returns `min_variance`), but an overflowed `inf`
            // MLE variance would flow through as `inf` → `inf` std →
            // `Normal::new` rejects non-finite σ → `sample` panics. Making the
            // stored variance always finite and `≥ min_variance` keeps `sample`
            // panic-free.
            let mle = *v / kf;
            *v = if mle.is_finite() && mle > params.min_variance {
                mle
            } else {
                params.min_variance
            };
        }

        UnivariateGaussianState { mean, variance }
    }

    /// Draw `n` genomes from the fitted per-dimension Gaussians.
    ///
    /// All randomness is drawn from `rng` (host RNG only; never
    /// `Tensor::random` / `B::seed`). The returned tensor has shape `(n, D)`.
    ///
    /// # Panics
    ///
    /// Does not panic under normal operation. The internal `Normal::new`
    /// constructor is guarded by `min_variance`, ensuring the standard
    /// deviation is always strictly positive and finite.
    fn sample(
        &self,
        state: &Self::State,
        n: usize,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 2> {
        let d = state.mean.len();
        // One Normal per dimension; the floored std is positive and finite.
        let normals: Vec<Normal<f32>> = (0..d)
            .map(|j| {
                Normal::new(state.mean[j], state.variance[j].sqrt())
                    .expect("floored std is positive and finite")
            })
            .collect();
        // Row-major fill: outer loop individuals, inner loop dimensions.
        let mut rows = Vec::with_capacity(n * d);
        for _ in 0..n {
            for normal in &normals {
                rows.push(normal.sample(rng));
            }
        }
        Tensor::<B, 2>::from_data(TensorData::new(rows, [n, d]), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type TestBackend = Flex;

    fn pop(rows: Vec<f32>, n: usize, d: usize) -> Tensor<TestBackend, 2> {
        let device = Default::default();
        Tensor::<TestBackend, 2>::from_data(TensorData::new(rows, [n, d]), &device)
    }

    fn fitness(values: Vec<f32>) -> Tensor<TestBackend, 1> {
        let device = Default::default();
        let n = values.len();
        Tensor::<TestBackend, 1>::from_data(TensorData::new(values, [n]), &device)
    }

    #[test]
    fn prior_from_params() {
        let device = Default::default();
        let model = UnivariateGaussian;
        let p = UnivariateGaussianParams::default_for(3);
        let state = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &model,
            &p,
            None,
            pop(vec![], 0, 0),
            fitness(vec![]),
            &device,
        );
        assert_eq!(state.mean, vec![0.0, 0.0, 0.0]);
        for v in &state.variance {
            approx::assert_relative_eq!(*v, 4.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn mle_matches_hand_computed() {
        let device = Default::default();
        let model = UnivariateGaussian;
        let p = UnivariateGaussianParams::default_for(2);
        // Column 0: [0, 2, 4] → mean 2, var = (4+0+4)/3 = 8/3.
        // Column 1: [1, 1, 4] → mean 2, var = (1+1+4)/3 = 2.
        let population = pop(vec![0.0, 1.0, 2.0, 1.0, 4.0, 4.0], 3, 2);
        let prior = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &model,
            &p,
            None,
            pop(vec![], 0, 0),
            fitness(vec![]),
            &device,
        );
        let state = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &model,
            &p,
            Some(&prior),
            population,
            fitness(vec![0.0, 1.0, 2.0]),
            &device,
        );
        approx::assert_relative_eq!(state.mean[0], 2.0, epsilon = 1e-5);
        approx::assert_relative_eq!(state.mean[1], 2.0, epsilon = 1e-5);
        approx::assert_relative_eq!(state.variance[0], 8.0 / 3.0, epsilon = 1e-5);
        approx::assert_relative_eq!(state.variance[1], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn variance_floor_engages_on_constant_column() {
        let device = Default::default();
        let model = UnivariateGaussian;
        let p = UnivariateGaussianParams::default_for(1);
        let prior = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &model,
            &p,
            None,
            pop(vec![], 0, 0),
            fitness(vec![]),
            &device,
        );
        // Constant column → raw variance 0, floored to min_variance.
        let state = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &model,
            &p,
            Some(&prior),
            pop(vec![3.0, 3.0, 3.0], 3, 1),
            fitness(vec![0.0, 0.0, 0.0]),
            &device,
        );
        approx::assert_relative_eq!(state.variance[0], p.min_variance, epsilon = 1e-9);
    }

    #[test]
    fn fitness_is_ignored() {
        let device = Default::default();
        let model = UnivariateGaussian;
        let p = UnivariateGaussianParams::default_for(2);
        let prior = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &model,
            &p,
            None,
            pop(vec![], 0, 0),
            fitness(vec![]),
            &device,
        );
        let rows = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let a = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &model,
            &p,
            Some(&prior),
            pop(rows.clone(), 3, 2),
            fitness(vec![0.0, 1.0, 2.0]),
            &device,
        );
        let b = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &model,
            &p,
            Some(&prior),
            pop(rows, 3, 2),
            fitness(vec![100.0, -7.0, 42.0]),
            &device,
        );
        assert_eq!(a.mean, b.mean);
        assert_eq!(a.variance, b.variance);
    }

    #[test]
    fn try_new_accepts_valid_and_round_trips() {
        let state = UnivariateGaussianState::try_new(vec![1.0, -2.0], vec![0.5, 4.0]).unwrap();
        assert_eq!(state.mean(), &[1.0, -2.0]);
        assert_eq!(state.variance(), &[0.5, 4.0]);
    }

    #[test]
    fn try_new_rejects_length_mismatch_and_bad_variance() {
        assert!(UnivariateGaussianState::try_new(vec![0.0, 0.0], vec![1.0]).is_err());
        assert!(UnivariateGaussianState::try_new(vec![], vec![]).is_err());
        assert!(UnivariateGaussianState::try_new(vec![0.0], vec![-1.0]).is_err());
        assert!(UnivariateGaussianState::try_new(vec![0.0], vec![f32::NAN]).is_err());
    }

    #[test]
    fn seeded_sampling_mean_matches_state() {
        let device = Default::default();
        let model = UnivariateGaussian;
        let state = UnivariateGaussianState {
            mean: vec![3.0, -1.0],
            variance: vec![1.0, 0.25],
        };
        let mut rng = StdRng::seed_from_u64(123);
        let samples = <UnivariateGaussian as ProbabilityModel<TestBackend>>::sample(
            &model, &state, 10_000, &mut rng, &device,
        );
        let dims = samples.dims();
        assert_eq!(dims, [10_000, 2]);
        let data = samples
            .into_data()
            .into_vec::<f32>()
            .expect("samples host-read of a tensor this test just built");
        let mut sum0 = 0.0_f32;
        let mut sum1 = 0.0_f32;
        for i in 0..10_000 {
            sum0 += data[i * 2];
            sum1 += data[i * 2 + 1];
        }
        approx::assert_relative_eq!(sum0 / 10_000.0, 3.0, epsilon = 0.1);
        approx::assert_relative_eq!(sum1 / 10_000.0, -1.0, epsilon = 0.1);
    }

    #[test]
    fn inf_variance_floored_to_min() {
        // Squared deviations overflow f32 → inf MLE variance. The floor (#129)
        // must reject the non-finite estimate instead of passing inf through to
        // sample() (where Normal::new would then panic on a non-finite σ).
        let device = Default::default();
        let p = UnivariateGaussianParams::default_for(1);
        let prior = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &UnivariateGaussian,
            &p,
            None,
            pop(vec![], 0, 0),
            fitness(vec![]),
            &device,
        );
        let state = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &UnivariateGaussian,
            &p,
            Some(&prior),
            pop(vec![1e38, -1e38], 2, 1),
            fitness(vec![0.0, 1.0]),
            &device,
        );
        let v = state.variance[0];
        assert!(v.is_finite(), "variance must be finite, got {v}");
        approx::assert_relative_eq!(v, p.min_variance, epsilon = 1e-12);
    }

    #[test]
    fn nonfinite_gene_mean_falls_back_to_init_mean() {
        // A NaN gene makes the column mean NaN; the guard (#129) falls back to
        // init_mean so the fitted mean stays finite (and sampling stays sane).
        let device = Default::default();
        let mut p = UnivariateGaussianParams::default_for(1);
        p.init_mean = 3.0;
        let prior = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &UnivariateGaussian,
            &p,
            None,
            pop(vec![], 0, 0),
            fitness(vec![]),
            &device,
        );
        let state = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &UnivariateGaussian,
            &p,
            Some(&prior),
            pop(vec![f32::NAN, 0.0], 2, 1),
            fitness(vec![0.0, 1.0]),
            &device,
        );
        assert!(state.mean[0].is_finite(), "mean must be finite");
        approx::assert_relative_eq!(state.mean[0], p.init_mean, epsilon = 1e-12);
    }
}
