//! Parent-selection operators.
//!
//! Selection operators turn a fitness tensor into winner indices that the
//! caller uses (via `Tensor::select`) to gather parents out of the
//! population. The baseline is host-side sampling + a single device
//! gather; a fused [kernel](super::kernels) variant is tracked as
//! follow-up work.
//!
//! # Fitness convention
//!
//! Fitness values are interpreted as "lower is better" (cost).
//! Tournament selection picks the smaller of the two sampled fitnesses.

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use rand::{Rng, RngExt};

/// Binary tournament selection over a fitness slice.
///
/// Samples two indices uniformly at random and keeps the one with the
/// smaller fitness. Repeats `n_winners` times, returning one winner
/// index per draw.
///
/// # Panics
///
/// Panics if `fitness.is_empty()`.
#[must_use]
pub fn tournament_indices_host(
    fitness: &[f32],
    tournament_size: usize,
    n_winners: usize,
    rng: &mut dyn Rng,
) -> Vec<i64> {
    assert!(!fitness.is_empty(), "fitness must be non-empty");
    assert!(tournament_size >= 2, "tournament size must be >= 2");
    let pop_size = fitness.len();
    let mut winners = Vec::with_capacity(n_winners);
    for _ in 0..n_winners {
        let mut best_idx = rng.random_range(0..pop_size);
        let mut best_f = fitness[best_idx];
        for _ in 1..tournament_size {
            let idx = rng.random_range(0..pop_size);
            if fitness[idx] < best_f {
                best_f = fitness[idx];
                best_idx = idx;
            }
        }
        #[allow(clippy::cast_possible_wrap)]
        winners.push(best_idx as i64);
    }
    winners
}

/// Gather `n_winners` rows out of the population by tournament.
///
/// Convenience wrapper that lifts [`tournament_indices_host`] into a
/// device gather on the population tensor.
#[must_use]
pub fn tournament_select<B: Backend>(
    population: &Tensor<B, 2>,
    fitness: &[f32],
    tournament_size: usize,
    n_winners: usize,
    rng: &mut dyn Rng,
    device: &B::Device,
) -> Tensor<B, 2> {
    let winners = tournament_indices_host(fitness, tournament_size, n_winners, rng);
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(winners, [n_winners]), device);
    population.clone().select(0, indices)
}

/// Truncation selection: returns the indices of the `top_k` lowest-
/// fitness members.
///
/// # Panics
///
/// Panics if `top_k > fitness.len()` or `fitness.is_empty()`.
#[must_use]
pub fn truncation_indices_host(fitness: &[f32], top_k: usize) -> Vec<i64> {
    assert!(!fitness.is_empty(), "fitness must be non-empty");
    assert!(top_k <= fitness.len(), "top_k must be <= population size");
    let mut indexed: Vec<(usize, f32)> = fitness.iter().copied().enumerate().collect();
    indexed
        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    #[allow(clippy::cast_possible_wrap)]
    indexed
        .into_iter()
        .take(top_k)
        .map(|(i, _)| i as i64)
        .collect()
}

/// Gather the `top_k` lowest-fitness rows out of the population.
#[must_use]
pub fn truncation_select<B: Backend>(
    population: &Tensor<B, 2>,
    fitness: &[f32],
    top_k: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let winners = truncation_indices_host(fitness, top_k);
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(winners, [top_k]), device);
    population.clone().select(0, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type TestBackend = NdArray;

    #[test]
    fn tournament_prefers_better_fitness_in_expectation() {
        let mut rng = StdRng::seed_from_u64(1);
        let fitness = [100.0f32, 0.0, 100.0, 100.0];
        let winners = tournament_indices_host(&fitness, 2, 1000, &mut rng);
        let wins_for_best = winners.iter().filter(|&&w| w == 1).count();
        // For pop_size=4 and tournament_size=2, P(best wins) =
        // 1 − (3/4)² = 7/16 ≈ 0.4375 → ~437 wins per 1000 trials.
        // Use a generous band to stay stable across RNG versions.
        assert!(
            (350..=550).contains(&wins_for_best),
            "wins_for_best={wins_for_best} (expected ~437)",
        );
    }

    #[test]
    fn truncation_returns_smallest_fitness_first() {
        let fitness = [5.0f32, 1.0, 3.0, 2.0, 4.0];
        let idx = truncation_indices_host(&fitness, 3);
        // The three smallest fitnesses live at indices 1 (=1.0), 3 (=2.0), 2 (=3.0).
        assert_eq!(idx.len(), 3);
        assert!(idx.contains(&1));
        assert!(idx.contains(&3));
        assert!(idx.contains(&2));
    }

    #[test]
    fn tournament_select_returns_shaped_tensor() {
        let device = Default::default();
        let data = TensorData::new(vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], [3, 2]);
        let pop = Tensor::<TestBackend, 2>::from_data(data, &device);
        let fitness = [10.0_f32, 0.0, 10.0];
        let mut rng = StdRng::seed_from_u64(2);
        let parents = tournament_select(&pop, &fitness, 2, 4, &mut rng, &device);
        assert_eq!(parents.shape().dims, vec![4, 2]);
    }
}
