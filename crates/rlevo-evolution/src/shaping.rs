//! Fitness shaping transforms.
//!
//! Strategies occasionally benefit from monotone transforms of raw
//! fitness values: centered-rank mapping flattens outliers, z-scoring
//! normalizes scale across generations, and weight decay penalizes
//! large-norm genomes (common in neuroevolution). These helpers work
//! directly on device tensors and are pure functions (RNG-free).

use burn::prelude::ElementConversion;
use burn::tensor::{backend::Backend, Tensor};

/// Error returned by fallible fitness-shaping transforms.
#[derive(Debug, thiserror::Error)]
pub enum ShapingError {
    /// The input tensor's element data could not be read as `f32` — e.g. an
    /// integer-typed backend tensor was passed to a transform that ranks host
    /// values. This is a backend/dtype mismatch, surfaced as a recoverable error
    /// rather than a panic.
    #[error("shaping transform requires f32 tensor data")]
    NonFloatData,
}

/// Returns `fitness - fitness.mean()` divided by the (population) std-dev,
/// clamped to avoid divide-by-zero when all fitnesses are equal.
///
/// The std-dev floor is `1e-8`; degenerate populations (all-equal fitness)
/// therefore map to a vector of zeros rather than producing NaNs.
///
/// # Examples
///
/// ```
/// use burn::backend::Flex;
/// use burn::tensor::Tensor;
/// use rlevo_evolution::shaping::z_score;
///
/// let device = Default::default();
/// // Five fitness values: mean 3.0, all distinct.
/// let t = Tensor::<Flex, 1>::from_floats([1.0f32, 2.0, 3.0, 4.0, 5.0], &device);
/// let z = z_score(t);
/// let values = z.into_data().into_vec::<f32>().unwrap();
/// // After z-scoring the mean of the output is 0 (within floating-point tolerance).
/// let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
/// assert!(mean.abs() < 1e-5);
/// ```
#[must_use]
pub fn z_score<B: Backend>(fitness: Tensor<B, 1>) -> Tensor<B, 1> {
    let mean = fitness.clone().mean().into_scalar().elem::<f32>();
    let n = fitness.dims()[0];
    #[allow(clippy::cast_precision_loss)]
    let n_f = n.max(1) as f32;
    let centered = fitness - mean;
    let var = centered.clone().powf_scalar(2.0).sum().into_scalar().elem::<f32>() / n_f;
    let std = var.sqrt().max(1e-8);
    centered / std
}

/// Returns centered ranks: the largest fitness maps to `+0.5`, the
/// smallest to `-0.5`, with linear spacing in between.
///
/// Under the crate's maximise convention (canonical: higher is better)
/// this assigns the **best** (highest-fitness) individual the highest
/// utility `+0.5` and the worst the lowest `-0.5`, which is the sign a
/// gradient-style ES update expects — no negation at the call site.
///
/// Centered ranks are standard in modern ES (e.g. OpenAI-ES) because they
/// remove outlier fitness magnitudes and keep the signal scale-free across
/// generations. Implemented host-side because the argsort pathway is
/// easier to reason about; swap in a tensor-native implementation if this
/// ever shows up on a profile.
///
/// An empty input returns an empty tensor.
///
/// # Examples
///
/// ```
/// use burn::backend::Flex;
/// use burn::tensor::Tensor;
/// use rlevo_evolution::shaping::centered_rank;
///
/// let device = Default::default();
/// let t = Tensor::<Flex, 1>::from_floats([10.0f32, 20.0, 30.0, 40.0], &device);
/// let r = centered_rank(t, &device).unwrap();
/// let values = r.into_data().into_vec::<f32>().unwrap();
/// // Smallest value maps to -0.5, largest to +0.5.
/// assert!((values[0] - (-0.5)).abs() < 1e-6);
/// assert!((values[3] - 0.5).abs() < 1e-6);
/// ```
///
/// # Errors
///
/// Returns [`ShapingError::NonFloatData`] if the tensor's element data cannot be
/// read as `f32` — for example, when using a backend that stores integer-typed
/// tensors and `into_vec::<f32>()` fails.
pub fn centered_rank<B: Backend>(fitness: Tensor<B, 1>, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Result<Tensor<B, 1>, ShapingError> {
    let raw = fitness
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| ShapingError::NonFloatData)?;
    let n = raw.len();
    if n == 0 {
        return Ok(Tensor::<B, 1>::from_floats([0.0f32; 0], device));
    }
    // Sanitize NaN → −inf (worst under maximise) so a NaN fitness ranks lowest
    // rather than corrupting the ascending order.
    let data: Vec<f32> = raw
        .iter()
        .map(|&f| crate::fitness::sanitize_fitness(f))
        .collect();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| data[i].total_cmp(&data[j]));

    #[allow(clippy::cast_precision_loss)]
    let n_f = (n - 1).max(1) as f32;
    let mut ranks = vec![0.0_f32; n];
    for (rank, &idx) in indices.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let r = rank as f32 / n_f - 0.5;
        ranks[idx] = r;
    }
    Ok(Tensor::<B, 1>::from_floats(ranks.as_slice(), device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    type TestBackend = Flex;

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn z_score_zero_mean_unit_std() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 1>::from_floats([1.0f32, 2.0, 3.0, 4.0, 5.0], &device);
        let z = z_score(t);
        let values = z.into_data().into_vec::<f32>().unwrap();
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        approx::assert_relative_eq!(mean, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn centered_rank_spans_half_interval() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 1>::from_floats([10.0f32, 20.0, 30.0, 40.0], &device);
        let r = centered_rank(t, &device).unwrap();
        let values = r.into_data().into_vec::<f32>().unwrap();
        // smallest → -0.5, largest → +0.5
        approx::assert_relative_eq!(values[0], -0.5, epsilon = 1e-6);
        approx::assert_relative_eq!(values[3], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn centered_rank_preserves_order() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 1>::from_floats([3.0f32, 1.0, 2.0], &device);
        let r = centered_rank(t, &device).unwrap();
        let values = r.into_data().into_vec::<f32>().unwrap();
        // original: 3, 1, 2 → ranks sorted ascending: [1, 2, 3] at indices [1, 2, 0]
        // rank-positions centered: index 1 → -0.5, index 2 → 0.0, index 0 → 0.5
        approx::assert_relative_eq!(values[1], -0.5, epsilon = 1e-6);
        approx::assert_relative_eq!(values[2], 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(values[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn centered_rank_empty_is_ok() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 1>::from_floats([0.0f32; 0], &device);
        let r = centered_rank(t, &device).expect("empty input is not an error");
        assert_eq!(r.dims()[0], 0);
    }
}
