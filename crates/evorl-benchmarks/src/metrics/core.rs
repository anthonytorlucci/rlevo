//! Tier-1 core metrics: return stats, episode length, throughput, success rate.
//!
//! All metric names are flat slash-separated keys per spec §12 Q4.

use super::Metric;

pub const RETURN_MEAN: &str = "return/mean";
pub const RETURN_MEDIAN: &str = "return/median";
pub const RETURN_STD: &str = "return/std";
pub const RETURN_MIN: &str = "return/min";
pub const RETURN_MAX: &str = "return/max";
pub const EPISODE_LENGTH_MEAN: &str = "episode_length/mean";
pub const SUCCESS_RATE: &str = "success_rate";
pub const STEPS_PER_SEC: &str = "throughput/steps_per_sec";
pub const WALL_CLOCK_SECONDS: &str = "wall_clock_seconds";

#[must_use]
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

#[must_use]
pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

#[must_use]
pub fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    var.sqrt()
}

#[must_use]
pub fn min(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::INFINITY, f64::min)
}

#[must_use]
pub fn max(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Compute the tier-1 core metric set from per-episode returns and lengths.
#[must_use]
pub fn core_metrics(
    returns: &[f64],
    lengths: &[usize],
    wall_clock_seconds: f64,
    success_threshold: Option<f64>,
) -> Vec<Metric> {
    let mut out = Vec::with_capacity(9);
    out.push(scalar(RETURN_MEAN, mean(returns)));
    out.push(scalar(RETURN_MEDIAN, median(returns)));
    out.push(scalar(RETURN_STD, std_dev(returns)));
    out.push(scalar(RETURN_MIN, min(returns)));
    out.push(scalar(RETURN_MAX, max(returns)));

    let lengths_f: Vec<f64> = lengths.iter().map(|&l| l as f64).collect();
    out.push(scalar(EPISODE_LENGTH_MEAN, mean(&lengths_f)));

    let total_steps: usize = lengths.iter().sum();
    let sps = if wall_clock_seconds > 0.0 {
        total_steps as f64 / wall_clock_seconds
    } else {
        0.0
    };
    out.push(scalar(STEPS_PER_SEC, sps));
    out.push(scalar(WALL_CLOCK_SECONDS, wall_clock_seconds));

    if let Some(threshold) = success_threshold {
        let rate = if returns.is_empty() {
            0.0
        } else {
            let hits = returns.iter().filter(|&&r| r >= threshold).count();
            hits as f64 / returns.len() as f64
        };
        out.push(scalar(SUCCESS_RATE, rate));
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
    fn stats_basics() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(mean(&xs), 3.0);
        assert_relative_eq!(median(&xs), 3.0);
        assert_relative_eq!(min(&xs), 1.0);
        assert_relative_eq!(max(&xs), 5.0);
        assert_relative_eq!(std_dev(&xs), 2.5_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn median_even_count() {
        assert_relative_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn empty_inputs_are_safe() {
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(median(&[]), 0.0);
        assert_eq!(std_dev(&[]), 0.0);
    }

    #[test]
    fn core_metrics_shape_and_success_rate() {
        let returns = [10.0, 20.0, 30.0, 40.0];
        let lengths = [5_usize, 5, 5, 5];
        let metrics = core_metrics(&returns, &lengths, 2.0, Some(25.0));
        assert_eq!(metrics.len(), 9);

        let find = |key: &str| -> f64 {
            metrics
                .iter()
                .find_map(|m| match m {
                    Metric::Scalar { name, value } if name == key => Some(*value),
                    _ => None,
                })
                .unwrap()
        };
        assert_relative_eq!(find(SUCCESS_RATE), 0.5);
        assert_relative_eq!(find(STEPS_PER_SEC), 10.0);
        assert_relative_eq!(find(RETURN_MEAN), 25.0);
    }

    #[test]
    fn core_metrics_no_threshold_omits_success_rate() {
        let metrics = core_metrics(&[1.0, 2.0], &[1, 1], 1.0, None);
        assert!(!metrics.iter().any(|m| matches!(
            m,
            Metric::Scalar { name, .. } if name == SUCCESS_RATE
        )));
    }
}
