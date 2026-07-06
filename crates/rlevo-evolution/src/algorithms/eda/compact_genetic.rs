//! Compact Genetic Algorithm (cGA) model for binary search spaces.
//!
//! cGA represents a virtual binary population as a per-gene probability vector
//! and emulates a steady-state GA of size `virtual_pop_size` without storing
//! the population explicitly. [`fit`] competes the **winner** (best fitness)
//! against the **loser** (worst fitness) of the truncation-selected subset: on
//! every gene where they disagree the probability is nudged by
//! `±1 / virtual_pop_size` toward the winner's bit. [`sample`] emits raw
//! `{0, 1}` `f32` genes; [`EdaParams::bounds`](crate::algorithms::eda::EdaParams::bounds) clamps are therefore no-ops.
//!
//! # Deviation from classic cGA
//!
//! The textbook cGA (Harik et al., 1999) draws *two* individuals uniformly at
//! random from the whole population and competes them. Here the winner and
//! loser are the best and worst of the **truncation-selected subset** handed to
//! [`fit`](ProbabilityModel::fit) by
//! [`EdaStrategy`](crate::algorithms::eda::EdaStrategy), so the update is
//! biased by the selection pressure already applied upstream.
//!
//! # References
//!
//! - Harik, Lobo & Goldberg (1999), *The compact genetic algorithm*.
//!
//! [`fit`]: crate::ProbabilityModel::fit
//! [`sample`]: crate::ProbabilityModel::sample

use burn::tensor::{Tensor, TensorData, backend::Backend};
use rand::{Rng, RngExt};
use rlevo_core::config::{self, ConfigError};

use crate::probability_model::ProbabilityModel;

/// Per-run configuration for the [`CompactGenetic`] model.
///
/// Held inside [`EdaParams::model`](crate::algorithms::eda::EdaParams::model)
/// for the lifetime of a run. Use [`CompactGeneticParams::default_for`] for
/// typical cGA defaults.
#[derive(Debug, Clone)]
pub struct CompactGeneticParams {
    /// Number of bits per genome; determines the length of
    /// [`CompactGeneticState::prob`].
    pub genome_dim: usize,
    /// Size of the virtual population being emulated; the per-generation
    /// probability step is `1 / virtual_pop_size`. Larger values slow
    /// convergence and preserve diversity; the original paper uses values in
    /// the range `[50, 200]`.
    pub virtual_pop_size: usize,
}

impl CompactGeneticParams {
    /// Sensible cGA defaults for a `genome_dim`-bit problem.
    #[must_use]
    pub fn default_for(genome_dim: usize) -> Self {
        Self {
            genome_dim,
            virtual_pop_size: 50,
        }
    }
}

/// Fitted state for the [`CompactGenetic`] model after one call to
/// [`ProbabilityModel::fit`].
///
/// The vector has length `genome_dim`. On the prior path (`prev = None`) it
/// is uniformly `0.5`; on subsequent calls entries are nudged by
/// `±1 / virtual_pop_size` and clamped to `[0, 1]`.
///
/// The field is private so an out-of-range probability is unrepresentable
/// from outside this module; build one with
/// [`try_new`](CompactGeneticState::try_new) and read it via
/// [`prob`](CompactGeneticState::prob).
#[derive(Debug, Clone)]
pub struct CompactGeneticState {
    /// Per-gene probability of sampling a `1.0` (always in `[0, 1]`).
    prob: Vec<f32>,
}

impl CompactGeneticState {
    /// Builds a cGA state from a per-gene probability vector.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `prob` is empty or if any entry is outside
    /// the closed interval `[0, 1]` (or is non-finite).
    pub fn try_new(prob: Vec<f32>) -> Result<Self, ConfigError> {
        config::nonzero("CompactGeneticState", "prob", prob.len())?;
        for &p in &prob {
            config::in_range("CompactGeneticState", "prob", 0.0, 1.0, f64::from(p))?;
        }
        Ok(Self { prob })
    }

    /// Per-gene probabilities of sampling a `1.0`, each in `[0, 1]`.
    #[must_use]
    pub fn prob(&self) -> &[f32] {
        &self.prob
    }
}

/// Compact Genetic Algorithm for binary spaces (cGA).
///
/// Implements [`ProbabilityModel`] with a per-gene probability vector updated
/// by winner/loser competition from the truncation-selected subset. Samples
/// are raw `{0, 1}` `f32` values; [`EdaParams::bounds`](crate::algorithms::eda::EdaParams::bounds)
/// clamps are no-ops for this model.
///
/// See the [module docs](self) for the update rule, the deviation from classic
/// cGA, and the reference.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompactGenetic;

impl<B: Backend> ProbabilityModel<B> for CompactGenetic {
    type Params = CompactGeneticParams;
    type State = CompactGeneticState;

    /// Update the per-gene probability vector by winner/loser competition.
    ///
    /// When `prev = None` returns the uniform-`0.5` prior; `population` and
    /// `fitness` are ignored on that path. Otherwise finds the argmax (winner)
    /// and argmin (loser) of the fitness vector (canonical maximise: higher is
    /// better), then nudges each gene where winner and loser disagree by
    /// `±1 / virtual_pop_size` (toward the winner), clamped to `[0, 1]`.
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
            return CompactGeneticState {
                prob: vec![0.5; params.genome_dim],
            };
        };

        let [k, d] = population.dims();
        let rows = population.into_data().into_vec::<f32>().unwrap_or_default();
        let fit_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();

        // Winner = argmax (best fitness), loser = argmin (worst); ties →
        // lowest index. Canonical maximise: higher is better.
        let mut winner_idx = 0_usize;
        let mut loser_idx = 0_usize;
        let mut best_f = f32::NEG_INFINITY;
        let mut worst_f = f32::INFINITY;
        for i in 0..k {
            // Sanitize `NaN → −inf` at the seam. `EdaStrategy::tell` already does
            // this upstream, but `fit` is a public trait method reachable
            // directly; without this a `NaN` would sort as the largest value
            // under `total_cmp` and be selected as the winner, nudging the model
            // toward a meaningless genome.
            let f = crate::fitness::sanitize_fitness(
                fit_host.get(i).copied().unwrap_or(f32::NEG_INFINITY),
            );
            if f.total_cmp(&best_f) == std::cmp::Ordering::Greater {
                best_f = f;
                winner_idx = i;
            }
            if f.total_cmp(&worst_f) == std::cmp::Ordering::Less {
                worst_f = f;
                loser_idx = i;
            }
        }

        // virtual_pop_size is a small tunable count, far below f32's 2^24
        // exact-integer limit; the cast is lossless in practice.
        #[allow(clippy::cast_precision_loss)]
        let step = 1.0 / params.virtual_pop_size as f32;
        let mut prob = prev.prob.clone();
        for j in 0..d {
            let winner = rows[winner_idx * d + j];
            let loser = rows[loser_idx * d + j];
            // Only nudge genes where winner and loser disagree.
            if (winner - loser).abs() > 0.5 {
                if winner > 0.5 {
                    prob[j] += step;
                } else {
                    prob[j] -= step;
                }
                prob[j] = prob[j].clamp(0.0, 1.0);
            }
        }

        CompactGeneticState { prob }
    }

    /// Draw `n` binary genomes from the per-gene Bernoulli probabilities.
    ///
    /// Each gene is sampled independently as `1.0` with probability `prob[j]`,
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

    fn fit_prior(p: &CompactGeneticParams) -> CompactGeneticState {
        let device = Default::default();
        <CompactGenetic as ProbabilityModel<TestBackend>>::fit(
            &CompactGenetic,
            p,
            None,
            pop(vec![], 0, 0),
            fitness(vec![]),
            &device,
        )
    }

    #[test]
    fn prior_is_half() {
        let p = CompactGeneticParams::default_for(3);
        assert_eq!(fit_prior(&p).prob, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn try_new_accepts_valid_and_rejects_out_of_range() {
        let state = CompactGeneticState::try_new(vec![0.0, 0.5, 1.0]).unwrap();
        assert_eq!(state.prob(), &[0.0, 0.5, 1.0]);
        assert!(CompactGeneticState::try_new(vec![]).is_err());
        assert!(CompactGeneticState::try_new(vec![0.5, 1.5]).is_err());
        assert!(CompactGeneticState::try_new(vec![-0.1]).is_err());
        assert!(CompactGeneticState::try_new(vec![f32::NAN]).is_err());
    }

    #[test]
    fn nudge_is_exactly_one_over_vps() {
        let device = Default::default();
        let p = CompactGeneticParams {
            genome_dim: 2,
            virtual_pop_size: 10,
        };
        let prior = fit_prior(&p);
        // winner = row 0 (fitness 0): genes [1, 0]. loser = row 1: genes [0, 1].
        // Both genes differ: gene 0 winner=1 → +0.1; gene 1 winner=0 → -0.1.
        let state = <CompactGenetic as ProbabilityModel<TestBackend>>::fit(
            &CompactGenetic,
            &p,
            Some(&prior),
            pop(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
            fitness(vec![1.0, 0.0]),
            &device,
        );
        approx::assert_relative_eq!(state.prob[0], 0.6, epsilon = 1e-6);
        approx::assert_relative_eq!(state.prob[1], 0.4, epsilon = 1e-6);
    }

    #[test]
    fn clamp_at_zero_and_one() {
        let device = Default::default();
        let p = CompactGeneticParams {
            genome_dim: 2,
            virtual_pop_size: 2,
        };
        let mut state = CompactGeneticState {
            prob: vec![0.9, 0.1],
        };
        // gene 0: winner=1 → +0.5 → 1.4 clamped to 1.0.
        // gene 1: winner=0 → -0.5 → -0.4 clamped to 0.0.
        for _ in 0..3 {
            state = <CompactGenetic as ProbabilityModel<TestBackend>>::fit(
                &CompactGenetic,
                &p,
                Some(&state),
                pop(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                fitness(vec![1.0, 0.0]),
                &device,
            );
        }
        approx::assert_relative_eq!(state.prob[0], 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(state.prob[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn genes_where_winner_equals_loser_untouched() {
        let device = Default::default();
        let p = CompactGeneticParams {
            genome_dim: 2,
            virtual_pop_size: 10,
        };
        let prior = fit_prior(&p);
        // gene 0: winner=1, loser=1 (same) → untouched at 0.5.
        // gene 1: winner=1, loser=0 (differ) → +0.1 → 0.6.
        let state = <CompactGenetic as ProbabilityModel<TestBackend>>::fit(
            &CompactGenetic,
            &p,
            Some(&prior),
            pop(vec![1.0, 1.0, 1.0, 0.0], 2, 2),
            fitness(vec![1.0, 0.0]),
            &device,
        );
        approx::assert_relative_eq!(state.prob[0], 0.5, epsilon = 1e-6);
        approx::assert_relative_eq!(state.prob[1], 0.6, epsilon = 1e-6);
    }

    #[test]
    fn not_the_column_mean() {
        let device = Default::default();
        let p = CompactGeneticParams {
            genome_dim: 1,
            virtual_pop_size: 10,
        };
        let prior = fit_prior(&p);
        // Column mean of [1, 0, 0] is 1/3 ≈ 0.333. cGA instead nudges 0.5 by
        // +0.1 to 0.6 (winner=row 0 with gene 1, loser=row 2 with gene 0).
        // Canonical maximise: row 0 has the highest fitness (winner), row 2 the
        // lowest (loser).
        let state = <CompactGenetic as ProbabilityModel<TestBackend>>::fit(
            &CompactGenetic,
            &p,
            Some(&prior),
            pop(vec![1.0, 0.0, 0.0], 3, 1),
            fitness(vec![2.0, 1.0, 0.0]),
            &device,
        );
        approx::assert_relative_eq!(state.prob[0], 0.6, epsilon = 1e-6);
        assert!((state.prob[0] - 1.0 / 3.0).abs() > 0.2);
    }

    #[test]
    fn samples_are_binary() {
        let device = Default::default();
        let state = CompactGeneticState {
            prob: vec![0.2, 0.8],
        };
        let mut rng = StdRng::seed_from_u64(11);
        let samples = <CompactGenetic as ProbabilityModel<TestBackend>>::sample(
            &CompactGenetic,
            &state,
            300,
            &mut rng,
            &device,
        );
        for v in samples.into_data().into_vec::<f32>().unwrap() {
            // Exact float compare is correct: sample() writes literal 0.0/1.0.
            #[allow(clippy::float_cmp)]
            let is_binary = v == 0.0 || v == 1.0;
            assert!(is_binary);
        }
    }

    #[test]
    fn nan_fitness_not_selected_as_winner() {
        // Row 0 all-ones with NaN fitness, row 1 all-zeros with finite fitness.
        // Under total_cmp, an unsanitized NaN would sort as the largest value
        // and be picked as the winner; with the seam guard (#129) the finite
        // row is the winner and probabilities move toward 0.
        let device = Default::default();
        let p = CompactGeneticParams::default_for(2);
        let prior = fit_prior(&p);
        let state = <CompactGenetic as ProbabilityModel<TestBackend>>::fit(
            &CompactGenetic,
            &p,
            Some(&prior),
            pop(vec![1.0, 1.0, 0.0, 0.0], 2, 2),
            fitness(vec![f32::NAN, 5.0]),
            &device,
        );
        for &pj in &state.prob {
            assert!(pj.is_finite() && (0.0..=1.0).contains(&pj), "prob out of range: {pj}");
            assert!(pj < 0.5, "winner should be the finite-fitness zero row, got {pj}");
        }
    }
}
