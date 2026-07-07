//! Univariate Bernoulli model (PBIL — Population-Based Incremental Learning)
//! for binary search spaces.
//!
//! Each gene is an independent Bernoulli variable with a learned probability
//! of being `1`. [`fit`] nudges the probability vector toward the **best**
//! individual of the selected subset (positive update at rate
//! [`UnivariateBernoulliParams::learning_rate`]) and away from the **worst**
//! on genes where best and worst disagree (negative update at rate
//! [`UnivariateBernoulliParams::negative_learning_rate`]). The classic PBIL
//! probability-mutation step is **not** implemented. [`sample`] emits a fresh
//! tensor of raw `{0, 1}` `f32` genes; [`EdaParams::bounds`](crate::algorithms::eda::EdaParams::bounds)
//! clamps are therefore no-ops.
//!
//! The best and worst individuals are the argmax and argmin of the fitness
//! vector (canonical maximise: higher is better) over the truncation-selected
//! subset, not over the full population.
//! This departs slightly from the original Baluja formulation, which drew a
//! fresh sample for comparison, but is consistent with the upstream selection
//! already performed by [`EdaStrategy`](crate::algorithms::eda::EdaStrategy).
//!
//! # References
//!
//! - Baluja (1994), *Population-based incremental learning: a method for
//!   integrating genetic search based function optimization and competitive
//!   learning*.
//!
//! [`fit`]: crate::ProbabilityModel::fit
//! [`sample`]: crate::ProbabilityModel::sample

use burn::tensor::{Tensor, TensorData, backend::Backend};
use rand::{Rng, RngExt};
use rlevo_core::config::{self, ConfigError};

use crate::probability_model::ProbabilityModel;

/// Per-run configuration for the [`UnivariateBernoulli`] model.
///
/// Held inside [`EdaParams::model`](crate::algorithms::eda::EdaParams::model)
/// for the lifetime of a run. Use
/// [`UnivariateBernoulliParams::default_for`] for typical PBIL defaults.
#[derive(Debug, Clone)]
pub struct UnivariateBernoulliParams {
    /// Number of bits per genome; determines the length of
    /// [`UnivariateBernoulliState::prob`].
    pub genome_dim: usize,
    /// Interpolation rate toward the best individual's gene value per
    /// generation (`0 < learning_rate < 1`; original Baluja uses 0.1).
    pub learning_rate: f32,
    /// Additional interpolation rate applied on genes where the best and
    /// worst individuals disagree. The extra step interpolates toward the
    /// *best* individual's gene — identical, for binary `{0, 1}` genes, to
    /// moving away from the worst's value, since the two differ
    /// (`0 ≤ negative_learning_rate < 1`; original uses 0.075).
    pub negative_learning_rate: f32,
}

impl UnivariateBernoulliParams {
    /// Sensible PBIL defaults for a `genome_dim`-bit problem.
    #[must_use]
    pub fn default_for(genome_dim: usize) -> Self {
        Self {
            genome_dim,
            learning_rate: 0.1,
            negative_learning_rate: 0.075,
        }
    }
}

/// Fitted state for the [`UnivariateBernoulli`] model after one call to
/// [`ProbabilityModel::fit`].
///
/// The vector has length `genome_dim`. On the prior path (`prev = None`) it
/// is uniformly `0.5`; on subsequent calls the entries are nudged by the PBIL
/// update rule (see [module docs](self)).
///
/// The field is private so an out-of-range probability is unrepresentable
/// from outside this module; build one with
/// [`try_new`](UnivariateBernoulliState::try_new) and read it via
/// [`prob`](UnivariateBernoulliState::prob).
#[derive(Debug, Clone)]
pub struct UnivariateBernoulliState {
    /// Per-gene probability of sampling a `1.0` (always in `[0, 1]`).
    prob: Vec<f32>,
}

impl UnivariateBernoulliState {
    /// Builds a PBIL state from a per-gene probability vector.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `prob` is empty or if any entry is outside
    /// the closed interval `[0, 1]` (or is non-finite).
    pub fn try_new(prob: Vec<f32>) -> Result<Self, ConfigError> {
        config::nonzero("UnivariateBernoulliState", "prob", prob.len())?;
        for &p in &prob {
            config::in_range("UnivariateBernoulliState", "prob", 0.0, 1.0, f64::from(p))?;
        }
        Ok(Self { prob })
    }

    /// Per-gene probabilities of sampling a `1.0`, each in `[0, 1]`.
    #[must_use]
    pub fn prob(&self) -> &[f32] {
        &self.prob
    }
}

/// Population-Based Incremental Learning model for binary spaces (PBIL).
///
/// Implements [`ProbabilityModel`] with a per-gene Bernoulli probability vector
/// updated by nudging toward the best (and away from the worst) of the
/// truncation-selected subset. The classic probability-mutation step is omitted.
/// Samples are raw `{0, 1}` `f32` values;
/// [`EdaParams::bounds`](crate::algorithms::eda::EdaParams::bounds) clamps
/// are no-ops for this model.
///
/// See the [module docs](self) for the update rule and references.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnivariateBernoulli;

impl<B: Backend> ProbabilityModel<B> for UnivariateBernoulli {
    type Params = UnivariateBernoulliParams;
    type State = UnivariateBernoulliState;

    /// Update the per-gene probability vector from the selected population.
    ///
    /// When `prev = None` returns the uniform-`0.5` prior; `population` and
    /// `fitness` are ignored on that path. Otherwise locates the argmax
    /// (best) and argmin (worst) individuals by fitness, applies the positive
    /// PBIL interpolation toward the best's genes, and applies the negative
    /// update on genes where the best and worst disagree. Does **not** apply
    /// the classic Baluja probability-mutation step.
    fn fit(
        &self,
        params: &Self::Params,
        prev: Option<&Self::State>,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::State {
        let _ = device;
        let Some(prev) = prev else {
            // Prior path: uniform 0.5 per gene; population/fitness ignored.
            let _ = (population, fitness);
            return UnivariateBernoulliState {
                prob: vec![0.5; params.genome_dim],
            };
        };

        let [k, d] = population.dims();
        if k == 0 {
            // Empty selected population: the argmax/argmin below would leave
            // `best_idx`/`worst_idx` at `0` and then index `rows[0 * d + j]` on
            // an empty `rows`, panicking out of bounds. Return the previous
            // probabilities unchanged. `EdaStrategy::tell` clamps `k ≥ 2`, but
            // `fit` is a public trait method reachable directly.
            return UnivariateBernoulliState {
                prob: prev.prob.clone(),
            };
        }
        let rows = population
            .into_data()
            .into_vec::<f32>()
            .expect("population tensor must be readable as f32");
        let fit_host = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness tensor must be readable as f32");

        // Argmax (best) and argmin (worst), ties → lowest index.
        // Canonical maximise: higher is better.
        let mut best_idx = 0_usize;
        let mut worst_idx = 0_usize;
        let mut best_f = f32::NEG_INFINITY;
        let mut worst_f = f32::INFINITY;
        for i in 0..k {
            // Sanitize `NaN → −inf` at the seam, mirroring `compact_genetic` so
            // the two binary EDAs stay symmetric. `tell` sanitizes upstream, but
            // a direct `fit` caller passing a `NaN` fitness would otherwise have
            // it sort as the largest value under `total_cmp` and be picked as the
            // best individual.
            let f = crate::fitness::sanitize_fitness(
                fit_host.get(i).copied().unwrap_or(f32::NEG_INFINITY),
            );
            if f.total_cmp(&best_f) == std::cmp::Ordering::Greater {
                best_f = f;
                best_idx = i;
            }
            if f.total_cmp(&worst_f) == std::cmp::Ordering::Less {
                worst_f = f;
                worst_idx = i;
            }
        }

        let lr = params.learning_rate;
        let neg_lr = params.negative_learning_rate;
        let mut prob = prev.prob.clone();
        for j in 0..d {
            let best_gene = rows[best_idx * d + j];
            let worst_gene = rows[worst_idx * d + j];
            // Positive update: interpolate toward the best individual's gene.
            let mut updated = prob[j] * (1.0 - lr) + lr * best_gene;
            // Negative update only where best and worst disagree.
            if (best_gene - worst_gene).abs() > 0.5 {
                updated = updated * (1.0 - neg_lr) + neg_lr * best_gene;
            }
            prob[j] = updated;
        }

        UnivariateBernoulliState { prob }
    }

    /// Draw `n` binary genomes from the per-gene Bernoulli probabilities.
    ///
    /// Each gene is sampled independently as `1.0` with probability `p[j]`,
    /// `0.0` otherwise, using the supplied host RNG (never `Tensor::random` /
    /// `B::seed`). The returned tensor has shape `(n, D)` and contains only
    /// `0.0` and `1.0` values.
    fn sample(
        &self,
        state: &Self::State,
        n: usize,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 2> {
        let d = state.prob.len();
        let mut rows = Vec::with_capacity(n * d);
        // Row-major: outer individuals, inner dimensions.
        for _ in 0..n {
            for &p in &state.prob {
                let gene = if rng.random::<f32>() < p { 1.0 } else { 0.0 };
                rows.push(gene);
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

    fn fit_prior(p: &UnivariateBernoulliParams) -> UnivariateBernoulliState {
        let device = Default::default();
        <UnivariateBernoulli as ProbabilityModel<TestBackend>>::fit(
            &UnivariateBernoulli,
            p,
            None,
            pop(vec![], 0, 0),
            fitness(vec![]),
            &device,
        )
    }

    #[test]
    fn prior_is_half() {
        let p = UnivariateBernoulliParams::default_for(4);
        let state = fit_prior(&p);
        assert_eq!(state.prob, vec![0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn try_new_accepts_valid_and_rejects_out_of_range() {
        let state = UnivariateBernoulliState::try_new(vec![0.0, 0.5, 1.0]).unwrap();
        assert_eq!(state.prob(), &[0.0, 0.5, 1.0]);
        assert!(UnivariateBernoulliState::try_new(vec![]).is_err());
        assert!(UnivariateBernoulliState::try_new(vec![1.2]).is_err());
        assert!(UnivariateBernoulliState::try_new(vec![-0.5]).is_err());
        assert!(UnivariateBernoulliState::try_new(vec![f32::NAN]).is_err());
    }

    #[test]
    fn interpolation_not_overwrite() {
        let device = Default::default();
        let p = UnivariateBernoulliParams {
            genome_dim: 1,
            learning_rate: 0.1,
            negative_learning_rate: 0.0,
        };
        let prior = fit_prior(&p);
        // best = row 0 (gene 1), worst = row 1 (gene 0). neg_lr = 0 so only the
        // positive update fires: p = 0.5*0.9 + 0.1*1 = 0.55, strictly in (0.5, 1).
        let state = <UnivariateBernoulli as ProbabilityModel<TestBackend>>::fit(
            &UnivariateBernoulli,
            &p,
            Some(&prior),
            pop(vec![1.0, 0.0], 2, 1),
            fitness(vec![1.0, 0.0]),
            &device,
        );
        assert!(state.prob[0] > 0.5 && state.prob[0] < 1.0);
        approx::assert_relative_eq!(state.prob[0], 0.55, epsilon = 1e-6);
    }

    #[test]
    fn neg_lr_applies_only_to_differing_genes() {
        let device = Default::default();
        let p = UnivariateBernoulliParams {
            genome_dim: 2,
            learning_rate: 0.1,
            negative_learning_rate: 0.2,
        };
        let prior = fit_prior(&p);
        // Canonical maximise: row 0 (fitness 1.0) is best, row 1 (0.0) worst.
        // gene 0: best=1, worst=1 (same) → only positive update.
        // gene 1: best=1, worst=0 (differ) → positive then negative update.
        let state = <UnivariateBernoulli as ProbabilityModel<TestBackend>>::fit(
            &UnivariateBernoulli,
            &p,
            Some(&prior),
            pop(vec![1.0, 1.0, 1.0, 0.0], 2, 2),
            fitness(vec![1.0, 0.0]),
            &device,
        );
        // gene 0: 0.5*0.9 + 0.1 = 0.55.
        approx::assert_relative_eq!(state.prob[0], 0.55, epsilon = 1e-6);
        // gene 1: 0.55 then 0.55*0.8 + 0.2 = 0.64.
        approx::assert_relative_eq!(state.prob[1], 0.64, epsilon = 1e-6);
    }

    #[test]
    fn convergence_direction_toward_zeros() {
        let device = Default::default();
        let p = UnivariateBernoulliParams::default_for(1);
        let mut state = fit_prior(&p);
        // Best individual (highest fitness) is all-zeros; repeated fits must
        // drive p down.
        for _ in 0..50 {
            state = <UnivariateBernoulli as ProbabilityModel<TestBackend>>::fit(
                &UnivariateBernoulli,
                &p,
                Some(&state),
                pop(vec![0.0, 1.0], 2, 1),
                fitness(vec![1.0, 0.0]),
                &device,
            );
        }
        assert!(
            state.prob[0] < 0.1,
            "p did not converge toward 0, got {}",
            state.prob[0]
        );
    }

    #[test]
    fn samples_are_binary() {
        let device = Default::default();
        let state = UnivariateBernoulliState {
            prob: vec![0.3, 0.7, 0.5],
        };
        let mut rng = StdRng::seed_from_u64(5);
        let samples = <UnivariateBernoulli as ProbabilityModel<TestBackend>>::sample(
            &UnivariateBernoulli,
            &state,
            500,
            &mut rng,
            &device,
        );
        let data = samples
            .into_data()
            .into_vec::<f32>()
            .expect("samples host-read of a tensor this test just built");
        for v in data {
            // Exact float compare is correct here: sample() writes literal 0.0
            // or 1.0, never a computed value.
            #[allow(clippy::float_cmp)]
            let is_binary = v == 0.0 || v == 1.0;
            assert!(is_binary, "non-binary gene {v}");
        }
    }

    #[test]
    fn fit_empty_population_returns_prior() {
        // k == 0 would index an empty `rows` and panic; the guard (#129) returns
        // the previous probabilities unchanged.
        let device = Default::default();
        let p = UnivariateBernoulliParams::default_for(3);
        let prior = fit_prior(&p);
        let state = <UnivariateBernoulli as ProbabilityModel<TestBackend>>::fit(
            &UnivariateBernoulli,
            &p,
            Some(&prior),
            pop(vec![], 0, 3),
            fitness(vec![]),
            &device,
        );
        assert_eq!(
            state.prob, prior.prob,
            "empty population must return prior unchanged"
        );
    }

    #[test]
    fn nan_fitness_not_selected_as_best() {
        // Row 0 all-ones + NaN fitness; row 1 all-zeros + finite fitness. The
        // sanitized seam (#129) must pick row 1 as best and push prob toward 0.
        let device = Default::default();
        let p = UnivariateBernoulliParams::default_for(2);
        let prior = fit_prior(&p);
        let state = <UnivariateBernoulli as ProbabilityModel<TestBackend>>::fit(
            &UnivariateBernoulli,
            &p,
            Some(&prior),
            pop(vec![1.0, 1.0, 0.0, 0.0], 2, 2),
            fitness(vec![f32::NAN, 5.0]),
            &device,
        );
        for &pj in &state.prob {
            assert!(
                pj < 0.5,
                "best should be the finite-fitness zero row, got {pj}"
            );
        }
    }
}
