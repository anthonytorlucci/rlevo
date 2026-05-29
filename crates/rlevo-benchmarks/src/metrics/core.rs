//! Tier-1 core metrics: return statistics, episode length, throughput, and success rate.
//!
//! Provides metric name constants, statistical primitives ([`mean`], [`median`],
//! [`std_dev`], [`min`], [`max`]), and the [`core_metrics`] convenience function that
//! assembles a standard [`Metric`] set from per-episode return and length data.
//!
//! All metric names are flat slash-separated keys compatible with the [`Metric`] wire format.
//!
//! # Examples
//!
//! ```no_run
//! use rlevo_benchmarks::metrics::core::{core_metrics, RETURN_MEAN};
//! use rlevo_benchmarks::metrics::Metric;
//!
//! let returns = [10.0_f64, 20.0, 30.0];
//! let lengths  = [5_usize, 5, 5];
//! let metrics  = core_metrics(&returns, &lengths, 1.5, Some(15.0));
//! assert!(metrics.iter().any(|m| matches!(
//!     m, Metric::Scalar { name, .. } if name == RETURN_MEAN
//! )));
//! ```

use super::Metric;

/// Metric key for mean episode return — emitted by [`core_metrics`].
pub const RETURN_MEAN: &str = "return/mean";
/// Metric key for median episode return — emitted by [`core_metrics`].
pub const RETURN_MEDIAN: &str = "return/median";
/// Metric key for sample standard deviation of episode returns — emitted by [`core_metrics`].
pub const RETURN_STD: &str = "return/std";
/// Metric key for minimum episode return — emitted by [`core_metrics`].
pub const RETURN_MIN: &str = "return/min";
/// Metric key for maximum episode return — emitted by [`core_metrics`].
pub const RETURN_MAX: &str = "return/max";
/// Metric key for mean episode length in steps — emitted by [`core_metrics`].
pub const EPISODE_LENGTH_MEAN: &str = "episode_length/mean";
/// Metric key for the fraction of episodes that met the success threshold — emitted by
/// [`core_metrics`] only when `success_threshold` is `Some`.
pub const SUCCESS_RATE: &str = "success_rate";
/// Metric key for environment throughput in steps per wall-clock second — emitted by
/// [`core_metrics`].
pub const STEPS_PER_SEC: &str = "throughput/steps_per_sec";
/// Metric key for total wall-clock time of a trial in seconds — emitted by [`core_metrics`].
pub const WALL_CLOCK_SECONDS: &str = "wall_clock_seconds";

/// Computes the arithmetic mean of a slice.
///
/// Returns `0.0` for an empty slice, matching the safe-default convention used
/// throughout this module so callers never receive `NaN` or need to guard against
/// division by zero.
///
/// # Examples
///
/// ```
/// # use rlevo_benchmarks::metrics::core::mean;
/// assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
/// assert_eq!(mean(&[]), 0.0);
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Computes the median of a slice.
///
/// Allocates and sorts a copy of the input, leaving the original order unchanged.
/// For even-length slices the two middle values are averaged. Returns `0.0` for
/// an empty slice.
///
/// # Examples
///
/// ```
/// # use rlevo_benchmarks::metrics::core::median;
/// assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);       // odd length
/// assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);  // even length — average of midpoints
/// assert_eq!(median(&[]), 0.0);
/// ```
#[must_use]
pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    } else {
        sorted[n / 2]
    }
}

/// Computes the sample standard deviation (Bessel-corrected, *n* − 1 denominator).
///
/// Bessel's correction keeps the estimator unbiased when episode counts are small,
/// which is common during early training. Returns `0.0` for slices with fewer than
/// two elements.
///
/// # Examples
///
/// ```
/// # use rlevo_benchmarks::metrics::core::std_dev;
/// use approx::assert_relative_eq;
/// assert_relative_eq!(std_dev(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 2.0);
/// assert_eq!(std_dev(&[42.0]), 0.0);  // single element — undefined, returns 0
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    var.sqrt()
}

/// Returns the minimum value in a slice.
///
/// Returns [`f64::INFINITY`] for an empty slice, consistent with the fold identity
/// for minimum. Callers that pass episode return data from a completed trial will
/// always have at least one element.
///
/// # Examples
///
/// ```
/// # use rlevo_benchmarks::metrics::core::min;
/// assert_eq!(min(&[3.0, 1.0, 4.0, 1.0, 5.0]), 1.0);
/// assert_eq!(min(&[]), f64::INFINITY);
/// ```
#[must_use]
pub fn min(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Returns the maximum value in a slice.
///
/// Returns [`f64::NEG_INFINITY`] for an empty slice, consistent with the fold identity
/// for maximum. Callers that pass episode return data from a completed trial will
/// always have at least one element.
///
/// # Examples
///
/// ```
/// # use rlevo_benchmarks::metrics::core::max;
/// assert_eq!(max(&[3.0, 1.0, 4.0, 1.0, 5.0]), 5.0);
/// assert_eq!(max(&[]), f64::NEG_INFINITY);
/// ```
#[must_use]
pub fn max(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Assembles the tier-1 core metric set from per-episode returns and lengths.
///
/// This is the single source of truth for the standard benchmark metric set.
/// The evaluator calls it exactly once per trial — after the episode loop
/// finishes — and passes the result straight into [`BenchmarkReport::absorb_metrics`]:
///
/// ```text
/// // inside evaluator::run_trial
/// report.absorb_metrics(core_metrics(&returns, &lengths, wall, cfg.success_threshold));
/// report.absorb_metrics(agent.emit_metrics());   // agent-side metrics follow
/// ```
///
/// Keeping this logic here rather than in the evaluator means every caller
/// gets identical statistics and naming with no duplication.
///
/// ## Inputs and their origin
///
/// - `returns` — per-episode cumulative reward, accumulated inside the step
///   loop via `total_reward += step.reward`.
/// - `lengths` — per-episode step count, incremented once per `env.step` call.
/// - `wall_clock_seconds` — elapsed wall time for the **entire trial** (all
///   episodes), so [`STEPS_PER_SEC`] reflects real throughput including
///   environment overhead and any inter-episode setup cost.
/// - `success_threshold` — sourced from [`EvaluatorConfig::success_threshold`];
///   when `Some`, a [`SUCCESS_RATE`] metric is appended. Omitting it keeps the
///   output compact for environments where "success" is not meaningful.
///
/// ## Output shape
///
/// Returns 8 [`Metric::Scalar`] values when `success_threshold` is `None`, or
/// 9 when it is `Some`. The consumer (`absorb_metrics`) is additive, so agent
/// metrics emitted afterward overlay this baseline without conflict.
///
/// # Examples
///
/// ```
/// use rlevo_benchmarks::metrics::core::{core_metrics, RETURN_MEAN, SUCCESS_RATE};
/// use rlevo_benchmarks::metrics::Metric;
///
/// let returns = [8.0_f64, 12.0, 20.0];
/// let lengths  = [10_usize, 8, 15];
/// let metrics  = core_metrics(&returns, &lengths, 3.0, Some(10.0));
///
/// // success_rate == 2/3 ≈ 0.667 (returns ≥ 10.0)
/// let rate = metrics.iter().find_map(|m| match m {
///     Metric::Scalar { name, value } if name == SUCCESS_RATE => Some(*value),
///     _ => None,
/// }).unwrap();
/// assert!((rate - 2.0 / 3.0).abs() < 1e-10);
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
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
    #[allow(clippy::float_cmp)]
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
