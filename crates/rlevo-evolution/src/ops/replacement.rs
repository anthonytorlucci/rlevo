//! Survivor-selection / replacement operators.
//!
//! Each function takes the current generation's population and fitness
//! plus the offspring population and fitness, and returns the
//! (population, fitness) pair for the next generation.
//!
//! The helpers work on host-side fitness slices for selection logic and
//! lift winners back to the device via [`Tensor::select`].

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};

use crate::ops::selection::truncation_indices_host;

/// Replace the entire current generation with offspring (no elitism).
#[must_use]
pub fn generational<B: Backend>(
    _current_pop: Tensor<B, 2>,
    _current_fitness: &[f32],
    offspring_pop: Tensor<B, 2>,
    offspring_fitness: Vec<f32>,
) -> (Tensor<B, 2>, Vec<f32>) {
    (offspring_pop, offspring_fitness)
}

/// Elitist replacement: keep the `k` best members of the current
/// generation and fill the rest from offspring, preserving the best
/// offspring first.
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
    offspring_fitness: Vec<f32>,
    k: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Vec<f32>) {
    let pop_size = current_fitness.len();
    assert!(k <= pop_size, "elite count must be <= population size");
    let elite_idx = truncation_indices_host(current_fitness, k);
    let elites = current_pop
        .clone()
        .select(0, Tensor::<B, 1, Int>::from_data(TensorData::new(elite_idx.clone(), [k]), device));

    let n_offspring_to_keep = pop_size - k;
    let offspring_keep_idx = truncation_indices_host(&offspring_fitness, n_offspring_to_keep);
    let kept_offspring = offspring_pop.clone().select(
        0,
        Tensor::<B, 1, Int>::from_data(
            TensorData::new(offspring_keep_idx.clone(), [n_offspring_to_keep]),
            device,
        ),
    );

    let combined = Tensor::cat(vec![elites, kept_offspring], 0);

    let mut combined_fitness = Vec::with_capacity(pop_size);
    for i in elite_idx {
        #[allow(clippy::cast_sign_loss)]
        combined_fitness.push(current_fitness[i as usize]);
    }
    for i in offspring_keep_idx {
        #[allow(clippy::cast_sign_loss)]
        combined_fitness.push(offspring_fitness[i as usize]);
    }

    (combined, combined_fitness)
}

/// (μ + λ) replacement: merge the μ parents with the λ offspring and
/// keep the μ lowest-fitness members overall.
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
    device: &B::Device,
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
            #[allow(clippy::cast_sign_loss)]
            combined_fitness[i as usize]
        })
        .collect();
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(winners, [mu]), device);
    (combined.select(0, indices), next_fitness)
}

/// (μ, λ) replacement: discard parents; keep the μ best of the λ
/// offspring. Requires `lambda >= mu`.
///
/// # Panics
///
/// Panics if `mu > offspring_fitness.len()` (i.e. `lambda < mu`).
#[must_use]
pub fn mu_comma_lambda<B: Backend>(
    offspring: Tensor<B, 2>,
    offspring_fitness: &[f32],
    mu: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Vec<f32>) {
    assert!(
        mu <= offspring_fitness.len(),
        "(μ, λ): lambda must be >= mu",
    );
    let winners = truncation_indices_host(offspring_fitness, mu);
    let next_fitness: Vec<f32> = winners
        .iter()
        .map(|&i| {
            #[allow(clippy::cast_sign_loss)]
            offspring_fitness[i as usize]
        })
        .collect();
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(winners, [mu]), device);
    (offspring.select(0, indices), next_fitness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type TestBackend = NdArray;

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
        let values = next.into_data().into_vec::<f32>().unwrap();
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
        let rows = next.into_data().into_vec::<f32>().unwrap();
        // best two: offspring row 0 (0.1) and parent row 0 (0.5)
        assert_eq!(rows.len(), 4);
        // fitness should be {0.1, 0.5}
        let mut f_sorted = f;
        f_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        approx::assert_relative_eq!(f_sorted[0], 0.1, epsilon = 1e-6);
        approx::assert_relative_eq!(f_sorted[1], 0.5, epsilon = 1e-6);
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
        assert_eq!(next.shape().dims, vec![2, 2]);
        let mut fs = f;
        fs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        approx::assert_relative_eq!(fs[0], 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(fs[1], 3.0, epsilon = 1e-6);
    }
}
