//! EA-specific metrics: diversity, convergence, fitness variance.
//!
//! EA evaluation is noisy, so `fitness_variance` is tracked across
//! repeated evaluations of the same individual, not only across episodes.

use super::{Metric, core};

pub const POPULATION_DIVERSITY: &str = "ea/population_diversity";
pub const BEST_FITNESS: &str = "ea/best_fitness";
pub const GENERATIONS_TO_CONVERGE: &str = "ea/generations_to_converge";
pub const FITNESS_VARIANCE: &str = "ea/fitness_variance";

/// Mean pairwise Euclidean distance between individuals.
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

/// Sample variance of repeated fitness evaluations of the same individual.
#[must_use]
pub fn fitness_variance(evaluations: &[f64]) -> f64 {
    let s = core::std_dev(evaluations);
    s * s
}

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
