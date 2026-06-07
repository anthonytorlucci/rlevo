//! Recombination / crossover operators for real-valued genomes.
//!
//! Operators in this module consume two parent tensors of shape
//! `(N, D)` and produce an offspring tensor of the same shape. Each
//! operator draws its randomness from a caller-supplied host `rng` and
//! materialises the draws via `Tensor::from_data`, rather than seeding
//! the process-wide backend RNG (`B::seed` + `Tensor::random`). Host
//! sampling keeps results reproducible across thread schedules: the
//! global Flex RNG mutex would otherwise interleave draws with sibling
//! tests under the parallel runner.
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

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use rand::{Rng, RngExt};

/// Builds an `(n·d,)` host vector of `U[0, 1)` draws sized for a
/// `(n, d)` genome tensor.
fn unit_uniform_rows(n: usize, d: usize, rng: &mut dyn Rng) -> Vec<f32> {
    let mut rows = Vec::with_capacity(n * d);
    for _ in 0..n * d {
        rows.push(rng.random::<f32>());
    }
    rows
}

/// BLX-α (Blend Crossover α) between two parent populations.
///
/// For each gene position `i`, the child's value is drawn uniformly from the
/// extended interval
///
/// ```text
/// U(min(a_i, b_i) − α·|a_i − b_i|,  max(a_i, b_i) + α·|a_i − b_i|)
/// ```
///
/// When `α = 0` the child lies strictly within the parents' bounding box.
/// `α = 0.5` is the conventional default and allows mild extrapolation beyond
/// either parent. All `n·d` draws are taken from the caller-supplied host `rng`
/// and loaded onto the device via [`Tensor::from_data`]; no backend-global RNG
/// state is touched.
///
/// Both parent tensors must have shape `(N, D)` where `N` is the population
/// size and `D` is the genome length; the returned offspring tensor has the
/// same shape.
///
/// # Panics
///
/// Panics if `parent_a` and `parent_b` do not have identical shapes.
#[must_use]
pub fn blx_alpha<B: Backend>(
    parent_a: Tensor<B, 2>,
    parent_b: Tensor<B, 2>,
    alpha: f32,
    rng: &mut dyn Rng,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    assert_eq!(
        parent_a.dims(),
        parent_b.dims(),
        "BLX-α: parents must have identical shapes"
    );
    let [n, d] = parent_a.dims();

    let min = parent_a.clone().min_pair(parent_b.clone());
    let max = parent_a.max_pair(parent_b);
    let diff = max.clone() - min.clone();
    let lo = min - diff.clone().mul_scalar(alpha);
    let hi = max + diff.mul_scalar(alpha);

    let u = Tensor::<B, 2>::from_data(TensorData::new(unit_uniform_rows(n, d, rng), [n, d]), device);
    lo.clone() + u * (hi - lo)
}

/// Uniform crossover: per-gene Bernoulli swap between two parents.
///
/// For each gene position, the child inherits the value from `parent_a` with
/// probability `p` and from `parent_b` with probability `1 − p`. No blending
/// occurs; the child's gene values are drawn exclusively from the two parents'
/// existing alleles, so the distribution over individual gene values is
/// exactly preserved.
///
/// `p = 0.5` gives an unbiased mix; `p = 1.0` returns a clone of `parent_a`;
/// `p = 0.0` returns a clone of `parent_b`. All `n·d` Bernoulli draws are
/// taken from the caller-supplied host `rng` and loaded onto the device via
/// [`Tensor::from_data`]; no backend-global RNG state is touched.
///
/// Both parent tensors must have shape `(N, D)` where `N` is the population
/// size and `D` is the genome length; the returned offspring tensor has the
/// same shape.
///
/// For binary genomes (`Tensor<B, 2, Int>` with values in `{0, 1}`) see
/// [`binary_uniform_crossover`].
///
/// # Panics
///
/// Panics if `parent_a` and `parent_b` do not have identical shapes.
#[must_use]
pub fn uniform_crossover<B: Backend>(
    parent_a: Tensor<B, 2>,
    parent_b: Tensor<B, 2>,
    p: f32,
    rng: &mut dyn Rng,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    assert_eq!(
        parent_a.dims(),
        parent_b.dims(),
        "uniform crossover: parents must have identical shapes"
    );
    let [n, d] = parent_a.dims();
    let u = Tensor::<B, 2>::from_data(TensorData::new(unit_uniform_rows(n, d, rng), [n, d]), device);
    let keep_a = u.lower_elem(p);
    parent_a.mask_where(keep_a.bool_not(), parent_b)
}

/// Binary uniform crossover on `Tensor<B, 2, Int>` populations.
///
/// The Int-tensor counterpart of [`uniform_crossover`], intended for binary
/// genomes. For each gene, the child inherits `parent_a`'s allele with
/// probability `p` and `parent_b`'s allele with probability `1 − p`. No
/// blending is performed; the operation is a pure bitwise swap.
///
/// Both parents must hold values in `{0, 1}`. The returned tensor has the
/// same shape and element type as the inputs. All `n·d` Bernoulli draws are
/// taken from the caller-supplied host `rng` and loaded onto the device via
/// [`Tensor::from_data`]; no backend-global RNG state is touched.
///
/// # Panics
///
/// Panics if `parent_a` and `parent_b` do not have identical shapes.
#[must_use]
pub fn binary_uniform_crossover<B: Backend>(
    parent_a: Tensor<B, 2, Int>,
    parent_b: Tensor<B, 2, Int>,
    p: f32,
    rng: &mut dyn Rng,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2, Int> {
    assert_eq!(
        parent_a.dims(),
        parent_b.dims(),
        "binary uniform crossover: parents must have identical shapes"
    );
    let [n, d] = parent_a.dims();
    let u = Tensor::<B, 2>::from_data(TensorData::new(unit_uniform_rows(n, d, rng), [n, d]), device);
    let keep_a = u.lower_elem(p);
    parent_a.mask_where(keep_a.bool_not(), parent_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{flex::FlexDevice, Flex};
    #[allow(unused_imports)]
    use burn::tensor::backend::Backend as _;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    type TestBackend = Flex;

    #[test]
    fn blx_alpha_lies_between_bounds() {
        let device: FlexDevice = Default::default();
        let mut rng = StdRng::seed_from_u64(13);
        let a = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![0.0_f32, 0.0, 0.0, 0.0], [2, 2]),
            &device,
        );
        let b = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0_f32, 1.0, 1.0, 1.0], [2, 2]),
            &device,
        );
        let c = blx_alpha(a, b, 0.0, &mut rng, &device);
        let values = c.into_data().into_vec::<f32>().unwrap();
        // α = 0: children lie strictly in [0, 1].
        for v in values {
            assert!((0.0..=1.0).contains(&v), "value out of bounds: {v}");
        }
    }

    #[test]
    fn uniform_all_from_a_when_p_is_one() {
        let device: FlexDevice = Default::default();
        let mut rng = StdRng::seed_from_u64(5);
        let a = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![7.0_f32, 7.0, 7.0, 7.0], [2, 2]),
            &device,
        );
        let b = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![-7.0_f32, -7.0, -7.0, -7.0], [2, 2]),
            &device,
        );
        let c = uniform_crossover(a, b, 1.0, &mut rng, &device);
        let values = c.into_data().into_vec::<f32>().unwrap();
        for v in values {
            approx::assert_relative_eq!(v, 7.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn uniform_all_from_b_when_p_is_zero() {
        let device: FlexDevice = Default::default();
        let mut rng = StdRng::seed_from_u64(5);
        let a = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![7.0_f32, 7.0, 7.0, 7.0], [2, 2]),
            &device,
        );
        let b = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![-7.0_f32, -7.0, -7.0, -7.0], [2, 2]),
            &device,
        );
        let c = uniform_crossover(a, b, 0.0, &mut rng, &device);
        let values = c.into_data().into_vec::<f32>().unwrap();
        for v in values {
            approx::assert_relative_eq!(v, -7.0, epsilon = 1e-6);
        }
    }
}
