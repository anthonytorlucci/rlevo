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
    /// Panics if the `population` tensor cannot be read back as `f32`
    /// (`.expect("population tensor must be readable as f32")`), or with an
    /// out-of-bounds index if the host buffer is shorter than `k * d`. Callers
    /// must therefore pass an `f32`, `(k, d)`-shaped population tensor.
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
    /// The internal `Normal::new` constructor is guarded by `min_variance`,
    /// ensuring the standard deviation is always strictly positive and finite,
    /// so `.expect("floored std is positive and finite")` does not fire. Any
    /// non-finite genome values are already contained in [`fit`](Self::fit):
    /// a non-finite mean falls back to `init_mean`, and the MLE variance is
    /// floored to a finite, positive value at least `min_variance`. Given a
    /// state produced by `fit`, this method does not panic under normal
    /// operation.
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

    #[test]
    fn single_row_variance_floored() {
        // §7.1: k == 1. Each column's only deviation is from its own value, so
        // the MLE variance is exactly 0 and must floor to min_variance, while the
        // mean equals the single row.
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
        let state = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &model,
            &p,
            Some(&prior),
            pop(vec![5.0, -3.0], 1, 2),
            fitness(vec![0.0]),
            &device,
        );
        approx::assert_relative_eq!(state.mean[0], 5.0, epsilon = 1e-6);
        approx::assert_relative_eq!(state.mean[1], -3.0, epsilon = 1e-6);
        for v in &state.variance {
            approx::assert_relative_eq!(*v, p.min_variance, epsilon = 1e-9);
        }
    }

    #[test]
    fn seeded_sampling_variance_matches_state() {
        // §7.1: the empirical variance of a large seeded sample must match the
        // per-dimension variance stored in the state (the mean case is covered by
        // `seeded_sampling_mean_matches_state`).
        let device = Default::default();
        let model = UnivariateGaussian;
        let state = UnivariateGaussianState {
            mean: vec![3.0, -1.0],
            variance: vec![1.0, 0.25],
        };
        let mut rng = StdRng::seed_from_u64(321);
        let n = 20_000_usize;
        let samples = <UnivariateGaussian as ProbabilityModel<TestBackend>>::sample(
            &model, &state, n, &mut rng, &device,
        );
        let data = samples
            .into_data()
            .into_vec::<f32>()
            .expect("samples host-read of a tensor this test just built");
        // n = 20_000 fits f64 exactly; the cast is lossless.
        #[allow(clippy::cast_precision_loss)]
        let nf = n as f64;
        for j in 0..2 {
            let mut sum = 0.0_f64;
            for i in 0..n {
                sum += f64::from(data[i * 2 + j]);
            }
            let mean = sum / nf;
            let mut var = 0.0_f64;
            for i in 0..n {
                let diff = f64::from(data[i * 2 + j]) - mean;
                var += diff * diff;
            }
            var /= nf;
            // Empirical variance narrowed to f32 for comparison against the state.
            #[allow(clippy::cast_possible_truncation)]
            let var_f32 = var as f32;
            approx::assert_abs_diff_eq!(var_f32, state.variance()[j], epsilon = 0.05);
        }
    }

    #[test]
    fn refit_overwrites_prev_state() {
        // §7.1: UMDA is a full refit — `fit` recomputes the MLE from the current
        // population and never blends with the prior state. A wildly different
        // `prev` must not leak into the result.
        let device = Default::default();
        let model = UnivariateGaussian;
        let p = UnivariateGaussianParams::default_for(1);
        let prev = UnivariateGaussianState {
            mean: vec![100.0],
            variance: vec![50.0],
        };
        // Column [0, 2, 4] → mean 2, var (4+0+4)/3 = 8/3.
        let state = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
            &model,
            &p,
            Some(&prev),
            pop(vec![0.0, 2.0, 4.0], 3, 1),
            fitness(vec![0.0, 1.0, 2.0]),
            &device,
        );
        approx::assert_relative_eq!(state.mean[0], 2.0, epsilon = 1e-5);
        approx::assert_relative_eq!(state.variance[0], 8.0 / 3.0, epsilon = 1e-5);
        // No blend with the prior's mean 100 / variance 50.
        assert!(
            (state.mean[0] - 100.0).abs() > 50.0,
            "refit must overwrite, not blend with prev mean"
        );
    }

    use proptest::prelude::*;

    proptest! {
        // §7.2: `fit` output-shape and variance-floor contract over arbitrary
        // populations. Host-side tensor ops only (no backend train) → 64 cases.
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        /// For ANY selected population, the fitted state has both vectors of
        /// length `genome_dim`, every mean entry finite, and every variance
        /// entry finite and floored at `min_variance` (never below, never
        /// non-finite — the §7.1 floor / #129 guards hold universally).
        ///
        /// RNG boundary (ADR 0029): proptest generates host config only
        /// (`data`, `d`); `fit` is a deterministic MLE update and takes no rng.
        /// The MLE path is only reached with `Some(&prior)` — `None` returns the
        /// prior and ignores the population, so a prior is fitted first and then
        /// the generated data is fitted against it.
        #[test]
        fn fit_produces_finite_floored_state(
            data in prop::collection::vec(-1e6f32..1e6f32, 2usize..200),
            d in 2usize..=8,
        ) {
            let device = Default::default();
            let model = UnivariateGaussian;
            let params = UnivariateGaussianParams::default_for(d);

            // Take `k = data.len() / d` full rows; reject the rare case where the
            // generated data is shorter than one row.
            let k = data.len() / d;
            prop_assume!(k >= 1);
            let rows: Vec<f32> = data[..k * d].to_vec();
            let population = pop(rows, k, d);

            let prior = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
                &model,
                &params,
                None,
                pop(vec![], 0, 0),
                fitness(vec![]),
                &device,
            );
            let state = <UnivariateGaussian as ProbabilityModel<TestBackend>>::fit(
                &model,
                &params,
                Some(&prior),
                population,
                fitness(vec![0.0_f32; k]),
                &device,
            );

            prop_assert_eq!(state.mean().len(), d);
            prop_assert_eq!(state.variance().len(), d);
            for &m in state.mean() {
                prop_assert!(m.is_finite(), "mean entry not finite: {}", m);
            }
            for &v in state.variance() {
                prop_assert!(v.is_finite(), "variance entry not finite: {}", v);
                prop_assert!(
                    v >= params.min_variance,
                    "variance {} below floor {}",
                    v,
                    params.min_variance
                );
            }
        }
    }

    proptest! {
        // §7.2: `sample` unbiasedness. Backend-heavy (up to 20k draws per case) →
        // keep case count low and shrinking bounded.
        #![proptest_config(ProptestConfig {
            cases: 16,
            max_shrink_iters: 64,
            ..ProptestConfig::default()
        })]

        /// A large seeded sample from a one-dimensional fitted Gaussian has a
        /// sample mean close to the state's mean `mu`.
        ///
        /// RNG boundary (ADR 0029): proptest generates host config only
        /// (`mu`, `sigma2`, `n`, `seed`); the sampler is seeded via
        /// `StdRng::seed_from_u64(seed)` (module idiom), never `B::seed` /
        /// `Tensor::random`.
        ///
        /// Flakiness margin: the standard error of the mean is `sigma / sqrt(n)`.
        /// With `n >= 5000`, `SE <= sigma / sqrt(5000) ≈ 0.01414 * sigma`, so the
        /// `0.1 * sigma` bound is ≈ `7.07 * SE` at the worst case (and looser for
        /// larger `n`). At ~7 sigma the per-case failure probability is ≈ 1e-12,
        /// so 16 cases across arbitrary seeds do not flake.
        #[test]
        fn sample_mean_is_unbiased(
            mu in -10f32..10f32,
            sigma2 in 1e-3f32..10f32,
            n in 5_000usize..=20_000,
            seed in any::<u64>(),
        ) {
            let device = Default::default();
            let model = UnivariateGaussian;
            let state = UnivariateGaussianState {
                mean: vec![mu],
                variance: vec![sigma2],
            };
            let mut rng = StdRng::seed_from_u64(seed);
            let samples = <UnivariateGaussian as ProbabilityModel<TestBackend>>::sample(
                &model, &state, n, &mut rng, &device,
            );
            prop_assert_eq!(samples.dims(), [n, 1]);

            let data = samples
                .into_data()
                .into_vec::<f32>()
                .expect("samples host-read of a tensor this test just built");
            let sum: f64 = data.iter().map(|&x| f64::from(x)).sum();
            // `n <= 20_000` is exactly representable in f64; the cast is lossless.
            #[allow(clippy::cast_precision_loss)]
            let sample_mean = sum / n as f64;

            let sigma = f64::from(sigma2).sqrt();
            let bound = 0.1 * sigma;
            let diff = (sample_mean - f64::from(mu)).abs();
            prop_assert!(
                diff < bound,
                "sample mean {} strayed from mu {} by {} (bound {})",
                sample_mean,
                mu,
                diff,
                bound
            );
        }
    }
}
