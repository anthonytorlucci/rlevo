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
//! Fitness values are **canonical (maximise)**: *higher is better*.
//! Tournament selection retains the *largest* fitness seen across all
//! candidates in a single draw; truncation selection returns the `top_k`
//! entries with the largest fitnesses; [`argmax_host`] reduces a fitness
//! slice to the index of its single largest entry.

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::{Rng, RngExt};

/// Returns the index of the largest fitness value.
///
/// `fitness` is canonical (maximise): *higher is better*. The scan keeps
/// the running best under a strict `>` seeded from `NEG_INFINITY`, so:
///
/// - ties resolve to the lowest index,
/// - `NaN` and `−inf` entries never displace the running best (every
///   comparison against them or the seed is strict), and
/// - a slice with no entry above `−inf` (e.g. all-`NaN`) falls back to
///   index `0`.
///
/// # Examples
///
/// ```
/// use rlevo_evolution::ops::selection::argmax_host;
///
/// let fitness = [1.0_f32, 5.0, 3.0];
/// assert_eq!(argmax_host(&fitness), 1);
/// ```
///
/// # Panics
///
/// Panics if `fitness.is_empty()`.
#[must_use]
pub fn argmax_host(fitness: &[f32]) -> usize {
    assert!(!fitness.is_empty(), "fitness must be non-empty");
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in fitness.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    best_idx
}

/// K-ary tournament selection over a fitness slice.
///
/// Draws `tournament_size` candidate indices uniformly at random (with
/// replacement) and retains the one with the largest fitness. Repeats
/// the draw `n_winners` times and returns the winning index for each
/// draw. Randomness is drawn entirely from the caller-supplied `rng`;
/// no global or backend RNG state is touched.
///
/// `fitness` is a flat slice where index `i` is the canonical fitness of
/// population member `i` (higher is better). `tournament_size` controls
/// selection pressure: larger values yield stronger pressure toward
/// higher-fitness members. A `tournament_size` of 1 degenerates to
/// pressure-free uniform-random selection.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand::rngs::StdRng;
/// use rlevo_evolution::ops::selection::tournament_indices_host;
///
/// let fitness = [1.0_f32, 10.0, 1.0, 1.0];
/// let mut rng = StdRng::seed_from_u64(0);
/// let winners = tournament_indices_host(&fitness, 2, 100, &mut rng);
/// // The unique best member (index 1) is selected far more often than any
/// // single lower-fitness member.
/// let best_wins = winners.iter().filter(|&&w| w == 1).count();
/// let low_fitness_wins = winners.iter().filter(|&&w| w == 0).count();
/// assert!(best_wins > low_fitness_wins);
/// ```
///
/// # Panics
///
/// Panics if `fitness.is_empty()` or if `tournament_size == 0`.
#[must_use]
pub fn tournament_indices_host(
    fitness: &[f32],
    tournament_size: usize,
    n_winners: usize,
    rng: &mut dyn Rng,
) -> Vec<i32> {
    assert!(!fitness.is_empty(), "fitness must be non-empty");
    assert!(tournament_size >= 1, "tournament size must be >= 1");
    let pop_size = fitness.len();
    let mut winners = Vec::with_capacity(n_winners);
    for _ in 0..n_winners {
        let mut best_idx = rng.random_range(0..pop_size);
        let mut best_f = fitness[best_idx];
        for _ in 1..tournament_size {
            let idx = rng.random_range(0..pop_size);
            if fitness[idx] > best_f {
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
/// `fitness.is_empty()` or `tournament_size == 0`.
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

/// Returns the indices of the `top_k` highest-fitness members.
///
/// Sorts the population by fitness (descending, i.e. highest first) and
/// returns the first `top_k` indices. The returned `Vec` is ordered from
/// best to worst among the selected members. `NaN` fitnesses are sanitised to
/// `−inf` (worst, per the maximise convention) and ordered with `f32::total_cmp`,
/// so a `NaN`-fitness member always sorts last and can never be selected as best.
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
/// // The three fittest members are at original indices 0 (5.0), 4 (4.0), 2 (3.0).
/// assert!(idx.contains(&0));
/// assert!(idx.contains(&4));
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
    // Sanitize NaN → −inf (worst under maximise) so a NaN-fitness member can
    // never rank as best; `total_cmp` then gives a deterministic total order.
    let mut indexed: Vec<(usize, f32)> = fitness
        .iter()
        .map(|&f| crate::fitness::sanitize_fitness(f))
        .enumerate()
        .collect();
    // Descending: highest fitness first; sanitized NaN sorts last.
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    indexed
        .into_iter()
        .take(top_k)
        .map(|(i, _)| i as i32)
        .collect()
}

/// Gathers the `top_k` highest-fitness rows out of a population tensor.
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

    // ANCHOR: tournament_expectation_test
    #[test]
    fn tournament_prefers_better_fitness_in_expectation() {
        let mut rng = StdRng::seed_from_u64(1);
        let fitness = [0.0f32, 100.0, 0.0, 0.0];
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
    // ANCHOR_END: tournament_expectation_test

    // ANCHOR: truncation_ordering_test
    #[test]
    fn truncation_returns_largest_fitness_first() {
        let fitness = [5.0f32, 1.0, 3.0, 2.0, 4.0];
        let idx = truncation_indices_host(&fitness, 3);
        // The three largest fitnesses live at indices 0 (=5.0), 4 (=4.0), 2 (=3.0).
        assert_eq!(idx.len(), 3);
        assert!(idx.contains(&0));
        assert!(idx.contains(&4));
        assert!(idx.contains(&2));
    }
    // ANCHOR_END: truncation_ordering_test

    #[test]
    fn argmax_returns_index_of_largest() {
        assert_eq!(argmax_host(&[1.0, 5.0, 3.0]), 1);
        assert_eq!(argmax_host(&[-3.0, -1.0, -2.0]), 1);
        assert_eq!(argmax_host(&[7.0]), 0);
    }

    #[test]
    fn argmax_tie_resolves_to_lowest_index() {
        assert_eq!(argmax_host(&[2.0, 5.0, 5.0, 5.0]), 1);
    }

    #[test]
    fn argmax_nan_never_wins() {
        assert_eq!(argmax_host(&[1.0, f32::NAN, 3.0]), 2);
        assert_eq!(argmax_host(&[f32::NAN, 1.0]), 1);
    }

    #[test]
    fn argmax_degenerate_slice_falls_back_to_zero() {
        // No entry strictly exceeds the NEG_INFINITY seed, so the scan
        // keeps the initial index 0.
        assert_eq!(argmax_host(&[f32::NAN, f32::NAN]), 0);
        assert_eq!(argmax_host(&[f32::NAN, f32::NEG_INFINITY]), 0);
        assert_eq!(argmax_host(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
    }

    #[test]
    #[should_panic(expected = "fitness must be non-empty")]
    fn argmax_empty_panics() {
        let _ = argmax_host(&[]);
    }

    #[test]
    fn tournament_size_one_is_uniform_random() {
        // A 1-ary "tournament" is a single uniform draw: the dominant
        // member wins ~1/4 of the time, not most of the time.
        let mut rng = StdRng::seed_from_u64(3);
        let fitness = [0.0f32, 100.0, 0.0, 0.0];
        let winners = tournament_indices_host(&fitness, 1, 1000, &mut rng);
        let wins_for_best = winners.iter().filter(|&&w| w == 1).count();
        assert!(
            (150..=350).contains(&wins_for_best),
            "wins_for_best={wins_for_best} (expected ~250)",
        );
    }

    #[test]
    fn truncation_sorts_nan_fitness_last() {
        // `total_cmp` must place NaN-fitness members last (never panic), so a
        // full-population truncation ranks the finite members ahead of the NaN.
        let fitness = [3.0f32, f32::NAN, 5.0, 1.0];
        let idx = truncation_indices_host(&fitness, fitness.len());
        assert_eq!(idx.len(), 4);
        // Best-first among finite fitnesses: 5.0 (idx 2), 3.0 (idx 0), 1.0 (idx 3).
        assert_eq!(&idx[..3], &[2, 0, 3]);
        // The NaN-fitness member (idx 1) sorts last.
        assert_eq!(idx[3], 1);
    }

    // ANCHOR: tournament_select_shape_test
    #[test]
    fn tournament_select_returns_shaped_tensor() {
        let device = Default::default();
        let data = TensorData::new(vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], [3, 2]);
        let pop = Tensor::<TestBackend, 2>::from_data(data, &device);
        let fitness = [0.0_f32, 10.0, 0.0];
        let mut rng = StdRng::seed_from_u64(2);
        let parents = tournament_select(&pop, &fitness, 2, 4, &mut rng, &device);
        assert_eq!(parents.dims(), [4, 2]);
    }
    // ANCHOR_END: tournament_select_shape_test
}
