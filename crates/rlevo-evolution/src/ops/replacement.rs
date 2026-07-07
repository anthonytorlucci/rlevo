//! Survivor-selection / replacement operators.
//!
//! Each function takes the current generation's population and fitness
//! plus the offspring population and fitness, and returns the
//! `(population, fitness)` pair that becomes the next generation's
//! starting state.
//!
//! All selection logic operates on host-side `&[f32]` fitness slices
//! (canonical: higher is better), and winners are lifted back to the
//! device via a single [`Tensor::select`] gather. None of these functions
//! draw random numbers; they are deterministic given their inputs.
//!
//! # Fitness convention
//!
//! Fitness is **canonical (maximise)**: larger values are better. This
//! matches the convention used throughout [`crate::ops::selection`].
//!
//! # Choosing a replacement strategy
//!
//! | Strategy | Parent survival | Use when |
//! |---|---|---|
//! | [`generational`] | none | offspring quality is trusted; GA / CMA-ES |
//! | [`elitist`] | top-k | preserving known-good solutions matters |
//! | [`mu_plus_lambda`] | best of μ+λ pool | ES / DE with strong elitism |
//! | [`mu_comma_lambda`] | none (offspring only) | ES with deliberate age-based forgetting |

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};

use crate::ops::selection::truncation_indices_host;

/// Replaces the entire current generation with the offspring (no elitism).
///
/// Discards `_current_pop` and `_current_fitness` entirely; the returned
/// pair is `(offspring_pop, offspring_fitness)` unchanged. This is the
/// standard generational replacement used in classic GAs and CMA-ES: the
/// offspring generation fully succeeds the parent generation with no
/// carry-over of parent individuals.
#[must_use]
pub fn generational<B: Backend>(
    _current_pop: Tensor<B, 2>,
    _current_fitness: &[f32],
    offspring_pop: Tensor<B, 2>,
    offspring_fitness: Vec<f32>,
) -> (Tensor<B, 2>, Vec<f32>) {
    (offspring_pop, offspring_fitness)
}

/// Elitist replacement: keeps the `k` best parents and the best remaining offspring.
///
/// Selects the `k` highest-fitness members of the current generation
/// (elites) and the `pop_size − k` highest-fitness offspring, then
/// concatenates them to form the next generation of size `pop_size`.
/// Both selections use [`truncation_indices_host`] and are therefore
/// deterministic.
///
/// The returned fitness vector has the elites' fitnesses first,
/// followed by the kept offspring fitnesses, in the same order as the
/// corresponding rows of the returned tensor.
///
/// # Panics
///
/// Panics if `k > current_fitness.len()`, or if
/// `pop_size − k > offspring_fitness.len()` (not enough offspring to
/// backfill).
#[must_use]
pub fn elitist<B: Backend>(
    current_pop: Tensor<B, 2>,
    current_fitness: &[f32],
    offspring_pop: Tensor<B, 2>,
    offspring_fitness: &[f32],
    k: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> (Tensor<B, 2>, Vec<f32>) {
    let pop_size = current_fitness.len();
    assert!(k <= pop_size, "elite count must be <= population size");
    let elite_idx = truncation_indices_host(current_fitness, k);
    let elites = current_pop
        .select(0, Tensor::<B, 1, Int>::from_data(TensorData::new(elite_idx.clone(), [k]), device));

    let n_offspring_to_keep = pop_size - k;
    let offspring_keep_idx = truncation_indices_host(offspring_fitness, n_offspring_to_keep);
    let kept_offspring = offspring_pop.select(
        0,
        Tensor::<B, 1, Int>::from_data(
            TensorData::new(offspring_keep_idx.clone(), [n_offspring_to_keep]),
            device,
        ),
    );

    let combined = Tensor::cat(vec![elites, kept_offspring], 0);

    let mut combined_fitness = Vec::with_capacity(pop_size);
    for i in elite_idx {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        combined_fitness.push(current_fitness[i as usize]);
    }
    for i in offspring_keep_idx {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        combined_fitness.push(offspring_fitness[i as usize]);
    }

    (combined, combined_fitness)
}

/// (μ + λ) replacement: keeps the μ best individuals from the merged parent and offspring pool.
///
/// Concatenates the μ parent rows with the λ offspring rows into a
/// combined pool of size μ + λ, then retains the `mu` members with
/// the highest fitness. Parents and offspring compete on equal footing,
/// so a highly-fit parent can survive indefinitely.
///
/// The returned tensor has shape `(mu, genome_dim)` and the returned
/// fitness vector has length `mu`, both ordered by selection rank
/// (best first).
///
/// # Panics
///
/// Panics if `mu > parent_fitness.len() + offspring_fitness.len()`
/// (the underlying truncation cannot select more winners than the
/// combined pool contains).
#[must_use]
pub fn mu_plus_lambda<B: Backend>(
    parents: Tensor<B, 2>,
    parent_fitness: &[f32],
    offspring: Tensor<B, 2>,
    offspring_fitness: &[f32],
    mu: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> (Tensor<B, 2>, Vec<f32>) {
    let combined = Tensor::cat(vec![parents, offspring], 0);
    let combined_fitness: Vec<f32> = parent_fitness
        .iter()
        .chain(offspring_fitness.iter())
        .copied()
        .collect();
    let winners = truncation_indices_host(&combined_fitness, mu);
    let next_fitness: Vec<f32> = winners
        .iter()
        .map(|&i| {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            combined_fitness[i as usize]
        })
        .collect();
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(winners, [mu]), device);
    (combined.select(0, indices), next_fitness)
}

/// (μ, λ) replacement: discards parents and keeps the μ best offspring.
///
/// Parents are not passed to this function; only the λ offspring
/// compete for the μ survivor slots. This strategy deliberately
/// discards parent solutions each generation, which can help the
/// population escape local optima and tracks moving optima better than
/// (μ + λ). Requires `lambda >= mu` (`offspring_fitness.len() >= mu`).
///
/// The returned tensor has shape `(mu, genome_dim)` and the returned
/// fitness vector has length `mu`, both ordered by selection rank
/// (best first).
///
/// # Panics
///
/// Panics if `mu > offspring_fitness.len()` (i.e. `lambda < mu`).
#[must_use]
pub fn mu_comma_lambda<B: Backend>(
    offspring: Tensor<B, 2>,
    offspring_fitness: &[f32],
    mu: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> (Tensor<B, 2>, Vec<f32>) {
    assert!(
        mu <= offspring_fitness.len(),
        "(μ, λ): lambda must be >= mu",
    );
    let winners = truncation_indices_host(offspring_fitness, mu);
    let next_fitness: Vec<f32> = winners
        .iter()
        .map(|&i| {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            offspring_fitness[i as usize]
        })
        .collect();
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(winners, [mu]), device);
    (offspring.select(0, indices), next_fitness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    type TestBackend = Flex;

    #[test]
    fn generational_discards_current() {
        let device = Default::default();
        let current = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![0.0_f32; 4], [2, 2]),
            &device,
        );
        let offspring = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0_f32; 4], [2, 2]),
            &device,
        );
        let (next, f) = generational::<TestBackend>(
            current,
            &[0.0, 0.0],
            offspring,
            vec![1.0, 1.0],
        );
        let values = next.into_data().into_vec::<f32>().expect("population host-read of a tensor this test just built");
        for v in values {
            approx::assert_relative_eq!(v, 1.0, epsilon = 1e-6);
        }
        assert_eq!(f, vec![1.0, 1.0]);
    }

    #[test]
    fn mu_plus_lambda_keeps_best_overall() {
        let device = Default::default();
        let parents = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![10.0_f32, 10.0, 10.0, 10.0], [2, 2]),
            &device,
        );
        let offspring = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0_f32, 1.0, 5.0, 5.0], [2, 2]),
            &device,
        );
        let (next, f) = mu_plus_lambda::<TestBackend>(
            parents,
            &[0.5, 100.0],
            offspring,
            &[0.1, 50.0],
            2,
            &device,
        );
        let rows = next.into_data().into_vec::<f32>().expect("population host-read of a tensor this test just built");
        // best two (highest fitness): parent row 1 (100.0) and offspring row 1 (50.0)
        assert_eq!(rows.len(), 4);
        // fitness should be {50.0, 100.0}
        let mut f_sorted = f;
        f_sorted.sort_by(f32::total_cmp);
        approx::assert_relative_eq!(f_sorted[0], 50.0, epsilon = 1e-6);
        approx::assert_relative_eq!(f_sorted[1], 100.0, epsilon = 1e-6);
    }

    #[test]
    fn mu_comma_lambda_keeps_best_of_offspring() {
        let device = Default::default();
        let offspring = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0_f32, 1.0, 2.0, 2.0, 3.0, 3.0], [3, 2]),
            &device,
        );
        let (next, f) = mu_comma_lambda::<TestBackend>(
            offspring,
            &[5.0, 1.0, 3.0],
            2,
            &device,
        );
        assert_eq!(next.dims(), [2, 2]);
        // best two of offspring (highest fitness): 5.0 and 3.0
        let mut fs = f;
        fs.sort_by(f32::total_cmp);
        approx::assert_relative_eq!(fs[0], 3.0, epsilon = 1e-6);
        approx::assert_relative_eq!(fs[1], 5.0, epsilon = 1e-6);
    }
}
