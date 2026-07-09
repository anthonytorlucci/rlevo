//! The [`ProbabilityModel`] trait shared by estimation-of-distribution
//! algorithms (EDAs).
//!
//! An EDA replaces an explicit recombination operator with an explicit
//! probabilistic model of the promising region of search space. Each
//! generation fits a model to the selected individuals and then samples a
//! fresh population from it. This module defines the seam between the
//! generic [`crate::algorithms::eda::EdaStrategy`] driver and the concrete
//! model implementations (univariate Gaussian, Bernoulli, compact-GA, and a
//! dependency-chain MIMIC model).

use std::fmt::Debug;

use burn::tensor::{Tensor, backend::Backend};
use rand::Rng;

/// A probabilistic model of a promising region of search space.
///
/// EDAs run a `fit` → `sample` loop: each generation, [`fit`](Self::fit)
/// estimates model parameters from the (already truncation-selected)
/// population, then [`sample`](Self::sample) draws the next candidate
/// population from the fitted model. The model fully replaces the crossover
/// and mutation operators of a classical GA.
///
/// The two associated types separate *static* per-model configuration
/// ([`Params`](Self::Params), e.g. learning rates, initial means) from the
/// *evolving* fitted statistics ([`State`](Self::State), e.g. per-dimension
/// means and variances). [`State`](Self::State) is deliberately `Sync` (a
/// stronger bound than the `Send`-only [`crate::Strategy::State`]) so a future
/// covariance-carrying CMA-ES state can be shared across threads; the same
/// `fit`/`sample` shape is intended to host that model unchanged.
///
/// The `fitness` tensor is passed to [`fit`](Self::fit) so models that weight
/// or rank the selected individuals (rank-μ updates, weighted MLE) can use it.
/// The univariate models shipped here perform an unweighted maximum-likelihood
/// fit and ignore it.
///
/// # Invariants
///
/// - **Prior path.** When `prev = None`, the model builds its prior *purely*
///   from [`params`](Self::Params). On this path the `population` and
///   `fitness` tensors are ignored; [`EdaStrategy`](crate::algorithms::eda::EdaStrategy)'s
///   `init` passes them as a `0 × 0` population and a length-`0` fitness tensor,
///   so a model must never read their contents when `prev` is `None`.
/// - **Host RNG only.** All randomness in [`sample`](Self::sample) must come
///   from the supplied `rng`. Implementations must never call `Tensor::random`
///   or `B::seed`: Burn's GPU PRNG kernels are seeded through process-global
///   state, which interleaves across parallel strategy calls and breaks the
///   per-stream determinism the crate guarantees. Sample on the host and upload
///   with `Tensor::from_data`.
/// - **Selection order.** The population rows handed to [`fit`](Self::fit) by
///   [`EdaStrategy`](crate::algorithms::eda::EdaStrategy)'s `tell` arrive in
///   *ascending-fitness*
///   order (best first), deterministically. Models that need the best or worst
///   row (e.g. PBIL, cGA) must still compute argmin/argmax themselves rather
///   than assume a fixed index — the ascending order is a convenience, not a
///   contract a model may hard-code against.
/// - **`Sync` state.** [`State`](Self::State) requires `Sync`, deliberately
///   exceeding [`crate::Strategy::State`]'s `Send`-only bound, to leave room for
///   a future thread-shared CMA-ES state.
/// - **Sanitized input assumed.** [`fit`](Self::fit) and
///   [`sample`](Self::sample) are infallible by design. The
///   [`EdaStrategy`](crate::algorithms::eda::EdaStrategy) driver sanitizes
///   fitness (`NaN` → `-inf`) and clamps the selected-row count to `≥ 2` before
///   calling `fit`, so the supported call path never supplies empty populations
///   or non-finite fitness. Callers that invoke `fit`/`sample` directly
///   (bypassing the driver) must uphold the same preconditions; otherwise
///   behaviour is unspecified — implementations may emit `NaN` statistics or
///   panic in `sample`.
///
/// # Examples
///
/// ```
/// use burn::backend::Flex;
/// use burn::tensor::{Tensor, TensorData};
/// use rand::{rngs::StdRng, SeedableRng};
/// use rlevo_evolution::ProbabilityModel;
/// use rlevo_evolution::algorithms::eda::{UnivariateGaussian, UnivariateGaussianParams};
///
/// let device = Default::default();
/// let model = UnivariateGaussian;
/// let params = UnivariateGaussianParams::default_for(2);
///
/// // Fit to a tiny two-row, two-column selected population.
/// let pop = Tensor::<Flex, 2>::from_data(
///     TensorData::new(vec![0.0_f32, 0.0, 2.0, 2.0], [2, 2]),
///     &device,
/// );
/// let fitness = Tensor::<Flex, 1>::from_data(TensorData::new(vec![0.0_f32, 8.0], [2]), &device);
/// let state =
///     ProbabilityModel::<Flex>::fit(&model, &params, None, pop.clone(), fitness.clone(), &device);
/// let next = ProbabilityModel::<Flex>::fit(&model, &params, Some(&state), pop, fitness, &device);
///
/// // Sample a fresh population from the fitted model.
/// let mut rng = StdRng::seed_from_u64(0);
/// let drawn: Tensor<Flex, 2> = model.sample(&next, 4, &mut rng, &device);
/// assert_eq!(drawn.dims(), [4, 2]);
/// ```
pub trait ProbabilityModel<B: Backend>: Send + Sync {
    /// Static, per-run model configuration (learning rates, priors, …).
    type Params: Clone + Debug + Send + Sync;

    /// Evolving fitted statistics carried generation to generation.
    type State: Clone + Debug + Send + Sync;

    /// Fit the model to the selected population.
    ///
    /// When `prev = None` the returned state is a prior derived purely from
    /// `params` (see the trait [`Invariants`](Self#invariants)); `population`
    /// and `fitness` are ignored on that path. Otherwise the model is refit to
    /// the supplied selected rows (ascending fitness order). `fitness` is
    /// available for weighted or rank-based updates; unweighted models ignore
    /// it.
    fn fit(
        &self,
        params: &Self::Params,
        prev: Option<&Self::State>,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::State;

    /// Draw `n` candidate genomes from the fitted model.
    ///
    /// All randomness must be drawn from `rng` (host RNG only — never
    /// `Tensor::random`/`B::seed`). The returned tensor has shape `(n, D)`
    /// where `D` is the model's genome dimensionality.
    ///
    /// # Panics
    ///
    /// Implementations may panic if `state` was not produced by a prior
    /// [`fit`](Self::fit) call — e.g. a user-constructed state with non-finite
    /// mean or variance (concrete impls call `Normal::new(...).expect(...)`,
    /// which panics on non-finite parameters).
    fn sample(
        &self,
        state: &Self::State,
        n: usize,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 2>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::eda::{UnivariateGaussian, UnivariateGaussianParams};
    use burn::backend::Flex;

    type TestBackend = Flex;

    #[test]
    fn prior_ignores_empty_population_and_fitness() {
        // Pin the `prev = None` invariant: a 0×0 population and 0-length
        // fitness tensor (exactly what `EdaStrategy::init` passes) must be
        // ignored, with the prior built purely from `params`.
        let device = Default::default();
        let model = UnivariateGaussian;
        let params = UnivariateGaussianParams::default_for(3);

        let empty_pop = Tensor::<TestBackend, 2>::empty([0, 0], &device);
        let empty_fitness = Tensor::<TestBackend, 1>::empty([0], &device);

        let state = model.fit(&params, None, empty_pop, empty_fitness, &device);

        assert_eq!(state.mean().len(), 3);
        assert_eq!(state.variance().len(), 3);
        for &m in state.mean() {
            approx::assert_relative_eq!(m, params.init_mean, epsilon = 1e-6);
        }
        let expected_var = params.init_std * params.init_std;
        for &v in state.variance() {
            approx::assert_relative_eq!(v, expected_var, epsilon = 1e-6);
        }
    }
}
