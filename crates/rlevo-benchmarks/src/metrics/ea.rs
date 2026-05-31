//! EA-specific metrics: diversity, convergence, fitness variance.
//!
//! Provides metric name constants and helpers for evolutionary algorithm
//! evaluation, plus [`ea_metrics`] which assembles an optional-field metric set
//! from per-generation EA data.  All inputs are `Option` so callers emit only
//! the metrics they actually compute — absent data produces no corresponding
//! [`Metric`] entry.
//!
//! EA evaluation is noisy, so [`fitness_variance`] is tracked across repeated
//! evaluations of the same individual, not only across episodes.
//!
//! # Examples
//!
//! ```no_run
//! use rlevo_benchmarks::metrics::ea::{ea_metrics, BEST_FITNESS};
//! use rlevo_benchmarks::metrics::Metric;
//!
//! let population = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.5, 0.5]];
//! let metrics = ea_metrics(Some(42.0), Some(&population), None);
//! assert!(metrics.iter().any(|m| matches!(
//!     m, Metric::Scalar { name, .. } if name == BEST_FITNESS
//! )));
//! ```

use super::{Metric, core};

/// Metric key for mean pairwise Euclidean distance across the current population — emitted by [`ea_metrics`].
pub const POPULATION_DIVERSITY: &str = "ea/population_diversity";
/// Metric key for the best fitness score in the current generation — emitted by [`ea_metrics`].
pub const BEST_FITNESS: &str = "ea/best_fitness";
/// Metric key for generations elapsed until convergence — **not** emitted by [`ea_metrics`];
/// computed post-run by the caller after the generation loop completes.
pub const GENERATIONS_TO_CONVERGE: &str = "ea/generations_to_converge";
/// Metric key for sample variance of repeated fitness evaluations — emitted by [`ea_metrics`].
pub const FITNESS_VARIANCE: &str = "ea/fitness_variance";

/// Computes the mean pairwise Euclidean distance across all individuals.
///
/// Iterates over all unique pairs (i, j) where i < j, computes the L2 distance
/// between genome vectors, and returns the mean. Returns `0.0` for populations
/// with fewer than two individuals.
///
/// # Examples
///
/// ```
/// # use rlevo_benchmarks::metrics::ea::population_diversity;
/// use approx::assert_relative_eq;
/// assert_relative_eq!(population_diversity(&[vec![0.0, 0.0], vec![3.0, 4.0]]), 5.0);
/// assert_eq!(population_diversity(&[vec![1.0, 2.0]]), 0.0);  // single individual
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn population_diversity(population: &[Vec<f64>]) -> f64 {
    let n = population.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut pairs = 0_usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = population[i]
                .iter()
                .zip(&population[j])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            sum += d;
            pairs += 1;
        }
    }
    sum / pairs as f64
}

/// Computes the sample variance of repeated fitness evaluations of one individual.
///
/// Variance (rather than standard deviation) is tracked because it combines additively
/// across independent noise sources, making it easier to separate environmental
/// stochasticity from genome-inherent variability. Delegates to [`core::std_dev`] and
/// squares the result, so Bessel's correction is inherited. Returns `0.0` for
/// fewer than two evaluations.
///
/// # Examples
///
/// ```
/// # use rlevo_benchmarks::metrics::ea::fitness_variance;
/// use approx::assert_relative_eq;
/// assert_relative_eq!(fitness_variance(&[1.0, 2.0, 3.0, 4.0, 5.0]), 2.5, epsilon = 1e-12);
/// assert_eq!(fitness_variance(&[42.0]), 0.0);  // single evaluation — undefined, returns 0
/// ```
#[must_use]
pub fn fitness_variance(evaluations: &[f64]) -> f64 {
    let s = core::std_dev(evaluations);
    s * s
}

/// Assembles the EA-specific metric set from optional per-generation data.
///
/// All inputs are `Option` — pass `None` for any quantity the caller does not
/// compute or track. Only `Some` inputs produce a [`Metric`] entry; missing data
/// is silently omitted rather than filled with a sentinel, keeping the report
/// compact and unambiguous.
///
/// The evolution loop typically calls this once per generation alongside
/// [`core_metrics`][super::core::core_metrics] and merges both into
/// [`crate::report::TrialReport::absorb_metrics`]:
///
/// ```text
/// // inside evolution::run_generation
/// report.absorb_metrics(core_metrics(&returns, &lengths, wall, cfg.success_threshold));
/// report.absorb_metrics(ea_metrics(Some(best), Some(&population), Some(&evals)));
/// ```
///
/// ## Inputs and their origin
///
/// - `best_fitness` — highest fitness score seen this generation.
/// - `population` — genome vectors for all living individuals; passed to
///   [`population_diversity`] (mean pairwise Euclidean distance).
/// - `repeated_evaluations` — multiple fitness scores for one individual from
///   repeated rollouts; passed to [`fitness_variance`].
///
/// ## Output shape
///
/// Returns between 0 and 3 [`Metric::Scalar`] values depending on which inputs
/// are `Some`. [`GENERATIONS_TO_CONVERGE`] is **not** emitted here — it is a
/// post-run statistic derived by the caller after the generation loop ends.
///
/// # Examples
///
/// ```
/// use rlevo_benchmarks::metrics::ea::{ea_metrics, BEST_FITNESS, POPULATION_DIVERSITY};
/// use rlevo_benchmarks::metrics::Metric;
/// use approx::assert_relative_eq;
///
/// let pop = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
/// let metrics = ea_metrics(Some(99.0), Some(&pop), None);
/// assert_eq!(metrics.len(), 2);  // best_fitness + population_diversity, no fitness_variance
///
/// let find = |key: &str| -> f64 {
///     metrics.iter().find_map(|m| match m {
///         Metric::Scalar { name, value } if name == key => Some(*value),
///         _ => None,
///     }).unwrap()
/// };
/// assert_eq!(find(BEST_FITNESS), 99.0);
/// assert_relative_eq!(find(POPULATION_DIVERSITY), 5.0);
/// ```
#[must_use]
pub fn ea_metrics(
    best_fitness: Option<f64>,
    population: Option<&[Vec<f64>]>,
    repeated_evaluations: Option<&[f64]>,
) -> Vec<Metric> {
    let mut out = Vec::new();
    if let Some(bf) = best_fitness {
        out.push(scalar(BEST_FITNESS, bf));
    }
    if let Some(pop) = population {
        out.push(scalar(POPULATION_DIVERSITY, population_diversity(pop)));
    }
    if let Some(evals) = repeated_evaluations {
        out.push(scalar(FITNESS_VARIANCE, fitness_variance(evals)));
    }
    out
}

fn scalar(name: &str, value: f64) -> Metric {
    Metric::Scalar {
        name: name.to_string(),
        value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn diversity_zero_for_identical_population() {
        let pop = vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        assert_relative_eq!(population_diversity(&pop), 0.0);
    }

    #[test]
    fn diversity_nonzero_for_spread_population() {
        let pop = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        assert_relative_eq!(population_diversity(&pop), 5.0);
    }

    #[test]
    fn fitness_variance_matches_sample_variance() {
        assert_relative_eq!(
            fitness_variance(&[1.0, 2.0, 3.0, 4.0, 5.0]),
            2.5,
            epsilon = 1e-12
        );
    }
}
