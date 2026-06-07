//! Parent-selection operators.
//!
//! Selection operators turn a fitness slice into winner indices that the
//! caller uses (via `Tensor::select`) to gather parents out of the
//! population. The baseline is host-side sampling + a single device
//! gather; a fused [kernel](super::kernels) variant is tracked as
//! follow-up work.
//!
//! All operators that draw random numbers accept an explicit `&mut dyn Rng`
//! (the host-RNG convention). Operators never touch thread-local or
//! process-wide backend RNG state (`B::seed` / `Tensor::random`), which
//! would race with sibling tests under the parallel test runner.
//!
//! # Fitness convention
//!
//! Fitness values are interpreted as "lower is better" (cost).
//! Tournament selection retains the *smallest* fitness seen across all
//! candidates in a single draw; truncation selection returns the `top_k`
//! entries with the smallest fitnesses.

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use rand::{Rng, RngExt};

/// K-ary tournament selection over a fitness slice.
///
/// Draws `tournament_size` candidate indices uniformly at random (with
/// replacement) and retains the one with the smallest fitness. Repeats
/// the draw `n_winners` times and returns the winning index for each
/// draw. Randomness is drawn entirely from the caller-supplied `rng`;
/// no global or backend RNG state is touched.
///
/// `fitness` is a flat slice where index `i` is the cost of population
/// member `i` (lower is better). `tournament_size` controls selection
/// pressure: larger values yield stronger pressure toward lower-cost
/// members.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand::rngs::StdRng;
/// use rlevo_evolution::ops::selection::tournament_indices_host;
///
/// let fitness = [10.0_f32, 1.0, 10.0, 10.0];
/// let mut rng = StdRng::seed_from_u64(0);
/// let winners = tournament_indices_host(&fitness, 2, 100, &mut rng);
/// // The unique best member (index 1) is selected far more often than any
/// // single higher-cost member.
/// let best_wins = winners.iter().filter(|&&w| w == 1).count();
/// let high_cost_wins = winners.iter().filter(|&&w| w == 0).count();
/// assert!(best_wins > high_cost_wins);
/// ```
///
/// # Panics
///
/// Panics if `fitness.is_empty()` or if `tournament_size < 2`.
#[must_use]
pub fn tournament_indices_host(
    fitness: &[f32],
    tournament_size: usize,
    n_winners: usize,
    rng: &mut dyn Rng,
) -> Vec<i32> {
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
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        winners.push(best_idx as i32);
    }
    winners
}

/// Gathers `n_winners` rows out of a population tensor by tournament.
///
/// Convenience wrapper that runs [`tournament_indices_host`] on the
/// host and then performs a single `Tensor::select` gather on the
/// device. The returned tensor has shape `(n_winners, genome_dim)`.
///
/// `population` must have shape `(N, D)` where `N` matches
/// `fitness.len()`. `tournament_size` and `n_winners` are forwarded
/// directly to [`tournament_indices_host`].
///
/// # Panics
///
/// Inherits the panic conditions of [`tournament_indices_host`]:
/// `fitness.is_empty()` or `tournament_size < 2`.
#[must_use]
pub fn tournament_select<B: Backend>(
    population: &Tensor<B, 2>,
    fitness: &[f32],
    tournament_size: usize,
    n_winners: usize,
    rng: &mut dyn Rng,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    let winners = tournament_indices_host(fitness, tournament_size, n_winners, rng);
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(winners, [n_winners]), device);
    population.clone().select(0, indices)
}

/// Returns the indices of the `top_k` lowest-fitness members.
///
/// Sorts the population by fitness (ascending, i.e. lowest cost first)
/// and returns the first `top_k` indices. The returned `Vec` is ordered
/// from best to worst among the selected members. Ties are broken by
/// `f32::partial_cmp`, with `NaN` values sorted last.
///
/// This is the host-side building block; call [`truncation_select`] when
/// you need the corresponding population rows as a tensor.
///
/// # Examples
///
/// ```
/// use rlevo_evolution::ops::selection::truncation_indices_host;
///
/// let fitness = [5.0_f32, 1.0, 3.0, 2.0, 4.0];
/// let idx = truncation_indices_host(&fitness, 3);
/// assert_eq!(idx.len(), 3);
/// // The three cheapest members are at original indices 1 (1.0), 3 (2.0), 2 (3.0).
/// assert!(idx.contains(&1));
/// assert!(idx.contains(&3));
/// assert!(idx.contains(&2));
/// ```
///
/// # Panics
///
/// Panics if `top_k > fitness.len()` or `fitness.is_empty()`.
#[must_use]
pub fn truncation_indices_host(fitness: &[f32], top_k: usize) -> Vec<i32> {
    assert!(!fitness.is_empty(), "fitness must be non-empty");
    assert!(top_k <= fitness.len(), "top_k must be <= population size");
    let mut indexed: Vec<(usize, f32)> = fitness.iter().copied().enumerate().collect();
    indexed
        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    indexed
        .into_iter()
        .take(top_k)
        .map(|(i, _)| i as i32)
        .collect()
}

/// Gathers the `top_k` lowest-fitness rows out of a population tensor.
///
/// Convenience wrapper that runs [`truncation_indices_host`] on the
/// host and then performs a single `Tensor::select` gather on the
/// device. The returned tensor has shape `(top_k, genome_dim)` with
/// rows ordered from best to worst fitness.
///
/// `population` must have shape `(N, D)` where `N` matches
/// `fitness.len()`.
///
/// # Panics
///
/// Inherits the panic conditions of [`truncation_indices_host`]:
/// `fitness.is_empty()` or `top_k > fitness.len()`.
#[must_use]
pub fn truncation_select<B: Backend>(
    population: &Tensor<B, 2>,
    fitness: &[f32],
    top_k: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    let winners = truncation_indices_host(fitness, top_k);
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(winners, [top_k]), device);
    population.clone().select(0, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type TestBackend = Flex;

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
        assert_eq!(parents.dims(), [4, 2]);
    }
}
