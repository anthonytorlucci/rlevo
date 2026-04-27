//! Recombination / crossover operators for real-valued genomes.
//!
//! Operators in this module consume two parent tensors of shape
//! `(N, D)` and produce an offspring tensor of the same shape. Seeding
//! the per-generation RNG via `B::seed(device, ...)` is the caller's
//! responsibility — see `crate::strategy` for the canonical pattern.
//!
//! # BLX-α
//!
//! For each gene, child ∈ `U(min(a,b) − α·|a−b|, max(a,b) + α·|a−b|)`.
//! A common default is α = 0.5.
//!
//! # Uniform
//!
//! For each gene, child takes parent A's value with probability `p` and
//! parent B's otherwise. Pure swap crossover — no blending — so the
//! distribution is exactly preserved. A binary-genome variant
//! ([`binary_uniform_crossover`]) operates on `Tensor<B, 2, Int>` with
//! values in `{0, 1}`.

use burn::tensor::{backend::Backend, Distribution, Int, Tensor};

/// BLX-α crossover between two parent populations.
///
/// # Panics
///
/// Panics if the parents do not have matching shapes.
#[must_use]
pub fn blx_alpha<B: Backend>(
    parent_a: Tensor<B, 2>,
    parent_b: Tensor<B, 2>,
    alpha: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    assert_eq!(
        parent_a.shape().dims,
        parent_b.shape().dims,
        "BLX-α: parents must have identical shapes"
    );
    let shape = parent_a.shape();

    let min = parent_a.clone().min_pair(parent_b.clone());
    let max = parent_a.max_pair(parent_b);
    let diff = max.clone() - min.clone();
    let lo = min - diff.clone().mul_scalar(alpha);
    let hi = max + diff.mul_scalar(alpha);

    let u = Tensor::<B, 2>::random(shape, Distribution::Uniform(0.0, 1.0), device);
    lo.clone() + u * (hi - lo)
}

/// Uniform crossover: per-gene Bernoulli swap between parents.
///
/// `p` is the probability of keeping parent A's gene.
///
/// # Panics
///
/// Panics if the parents do not have matching shapes.
#[must_use]
pub fn uniform_crossover<B: Backend>(
    parent_a: Tensor<B, 2>,
    parent_b: Tensor<B, 2>,
    p: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    assert_eq!(
        parent_a.shape().dims,
        parent_b.shape().dims,
        "uniform crossover: parents must have identical shapes"
    );
    let shape = parent_a.shape();
    let u = Tensor::<B, 2>::random(shape, Distribution::Uniform(0.0, 1.0), device);
    let keep_a = u.lower_elem(p);
    parent_a.mask_where(keep_a.bool_not(), parent_b)
}

/// Binary uniform crossover on `Tensor<B, 2, Int>` populations.
///
/// For each gene, keep parent A with probability `p`, else parent B.
/// `parent_a` and `parent_b` must hold values in `{0, 1}`.
///
/// # Panics
///
/// Panics if the parents do not have matching shapes.
#[must_use]
pub fn binary_uniform_crossover<B: Backend>(
    parent_a: Tensor<B, 2, Int>,
    parent_b: Tensor<B, 2, Int>,
    p: f32,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    assert_eq!(
        parent_a.shape().dims,
        parent_b.shape().dims,
        "binary uniform crossover: parents must have identical shapes"
    );
    let shape = parent_a.shape();
    let u = Tensor::<B, 2>::random(shape, Distribution::Uniform(0.0, 1.0), device);
    let keep_a = u.lower_elem(p);
    parent_a.mask_where(keep_a.bool_not(), parent_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    use burn::tensor::TensorData;
    #[allow(unused_imports)]
    use burn::tensor::backend::Backend as _;
    type TestBackend = NdArray;

    #[test]
    fn blx_alpha_lies_between_bounds() {
        let device: NdArrayDevice = Default::default();
        TestBackend::seed(&device, 13);
        let a = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![0.0_f32, 0.0, 0.0, 0.0], [2, 2]),
            &device,
        );
        let b = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0_f32, 1.0, 1.0, 1.0], [2, 2]),
            &device,
        );
        let c = blx_alpha(a, b, 0.0, &device);
        let values = c.into_data().into_vec::<f32>().unwrap();
        // α = 0: children lie strictly in [0, 1].
        for v in values {
            assert!((0.0..=1.0).contains(&v), "value out of bounds: {v}");
        }
    }

    #[test]
    fn uniform_all_from_a_when_p_is_one() {
        let device: NdArrayDevice = Default::default();
        TestBackend::seed(&device, 5);
        let a = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![7.0_f32, 7.0, 7.0, 7.0], [2, 2]),
            &device,
        );
        let b = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![-7.0_f32, -7.0, -7.0, -7.0], [2, 2]),
            &device,
        );
        let c = uniform_crossover(a, b, 1.0, &device);
        let values = c.into_data().into_vec::<f32>().unwrap();
        for v in values {
            approx::assert_relative_eq!(v, 7.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn uniform_all_from_b_when_p_is_zero() {
        let device: NdArrayDevice = Default::default();
        TestBackend::seed(&device, 5);
        let a = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![7.0_f32, 7.0, 7.0, 7.0], [2, 2]),
            &device,
        );
        let b = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![-7.0_f32, -7.0, -7.0, -7.0], [2, 2]),
            &device,
        );
        let c = uniform_crossover(a, b, 0.0, &device);
        let values = c.into_data().into_vec::<f32>().unwrap();
        for v in values {
            approx::assert_relative_eq!(v, -7.0, epsilon = 1e-6);
        }
    }
}
