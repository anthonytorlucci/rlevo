//! Mutation operators for real-valued genomes.
//!
//! Every mutation is an in-place-ish tensor transform: it consumes a
//! population tensor and returns a new one with noise added per gene.
//! The noise is sampled from a caller-supplied host `rng` and
//! materialised via `Tensor::from_data`, rather than seeding the
//! process-wide backend RNG (`B::seed` + `Tensor::random`). Host
//! sampling keeps results reproducible across thread schedules: the
//! global Flex RNG mutex would otherwise interleave draws with sibling
//! tests under the parallel runner.

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use rand::{Rng, RngExt};
use rand_distr::{Distribution as _, Normal};

/// Builds an `(n·d,)` host vector of `N(0, 1)` draws sized for a
/// `(n, d)` tensor.
fn standard_normal_rows(n: usize, d: usize, rng: &mut dyn Rng) -> Vec<f32> {
    let normal = Normal::new(0.0f32, 1.0).expect("unit normal is well-defined");
    let mut rows = Vec::with_capacity(n * d);
    for _ in 0..n * d {
        rows.push(normal.sample(rng));
    }
    rows
}

/// Isotropic Gaussian mutation with scalar σ.
///
/// Each gene is perturbed by `σ · N(0, 1)` noise, independently.
#[must_use]
pub fn gaussian_mutation<B: Backend>(
    population: Tensor<B, 2>,
    sigma: f32,
    rng: &mut dyn Rng,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    let [n, d] = population.dims();
    let noise =
        Tensor::<B, 2>::from_data(TensorData::new(standard_normal_rows(n, d, rng), [n, d]), device);
    population + noise.mul_scalar(sigma)
}

/// Per-row anisotropic Gaussian mutation.
///
/// `sigmas` is a `(N,)` tensor holding the σ for each individual; the
/// noise tensor is multiplied row-wise before being added to the
/// population.
///
/// # Panics
///
/// Panics if `sigmas`'s length does not match the population's first
/// dimension (the `reshape([n, 1])` step requires exactly `n` σ values).
#[must_use]
pub fn gaussian_mutation_per_row<B: Backend>(
    population: Tensor<B, 2>,
    sigmas: Tensor<B, 1>,
    rng: &mut dyn Rng,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    let [n, d] = population.dims();
    let noise =
        Tensor::<B, 2>::from_data(TensorData::new(standard_normal_rows(n, d, rng), [n, d]), device);
    let sigmas_2d = sigmas.reshape([n, 1]).expand([n, d]);
    population + noise * sigmas_2d
}

/// Uniform-reset mutation with per-gene probability `p`.
///
/// Each gene is replaced by a draw from `U(lo, hi)` with probability
/// `p`; otherwise it is left unchanged.
#[must_use]
pub fn uniform_reset<B: Backend>(
    population: Tensor<B, 2>,
    lo: f32,
    hi: f32,
    p: f32,
    rng: &mut dyn Rng,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    let [n, d] = population.dims();
    let mut noise_rows = Vec::with_capacity(n * d);
    let mut coin_rows = Vec::with_capacity(n * d);
    for _ in 0..n * d {
        noise_rows.push(lo + (hi - lo) * rng.random::<f32>());
        coin_rows.push(rng.random::<f32>());
    }
    let noise = Tensor::<B, 2>::from_data(TensorData::new(noise_rows, [n, d]), device);
    let coin = Tensor::<B, 2>::from_data(TensorData::new(coin_rows, [n, d]), device);
    let reset = coin.lower_elem(p);
    population.mask_where(reset, noise)
}

/// Bit-flip mutation on a binary `Tensor<B, 2, Int>` population.
///
/// Each gene is flipped independently with probability `p`. The input
/// must hold values in `{0, 1}`; the flip is computed arithmetically
/// as `1 − x` and will produce out-of-range values if that contract is
/// violated.
#[must_use]
pub fn bit_flip_mutation<B: Backend>(
    population: Tensor<B, 2, Int>,
    p: f32,
    rng: &mut dyn Rng,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2, Int> {
    let shape = population.shape();
    let [n, d] = population.dims();
    let mut coin_rows = Vec::with_capacity(n * d);
    for _ in 0..n * d {
        coin_rows.push(rng.random::<f32>());
    }
    let coin = Tensor::<B, 2>::from_data(TensorData::new(coin_rows, [n, d]), device);
    let flip = coin.lower_elem(p);
    // XOR via arithmetic: new = (1 - old) where flip, else old.
    let ones = Tensor::<B, 2, Int>::ones(shape, device);
    let flipped = ones - population.clone();
    population.mask_where(flip, flipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::backend::flex::FlexDevice;
    #[allow(unused_imports)]
    use burn::tensor::backend::Backend as _;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type TestBackend = Flex;

    #[test]
    fn gaussian_with_zero_sigma_is_identity() {
        let device: FlexDevice = Default::default();
        let mut rng = StdRng::seed_from_u64(3);
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]),
            &device,
        );
        let out = gaussian_mutation(input.clone(), 0.0, &mut rng, &device);
        let before = input.into_data().into_vec::<f32>().unwrap();
        let after = out.into_data().into_vec::<f32>().unwrap();
        for (a, b) in before.iter().zip(after.iter()) {
            approx::assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn gaussian_preserves_shape() {
        let device: FlexDevice = Default::default();
        let mut rng = StdRng::seed_from_u64(3);
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![0.0_f32; 12], [3, 4]),
            &device,
        );
        let out = gaussian_mutation(input, 1.0, &mut rng, &device);
        assert_eq!(out.dims(), [3, 4]);
    }

    #[test]
    fn per_row_applies_distinct_sigmas() {
        let device: FlexDevice = Default::default();
        let mut rng = StdRng::seed_from_u64(4);
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![0.0_f32; 4], [2, 2]),
            &device,
        );
        let sigmas = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.0_f32, 0.0], [2]),
            &device,
        );
        let out = gaussian_mutation_per_row(input, sigmas, &mut rng, &device);
        let values = out.into_data().into_vec::<f32>().unwrap();
        for v in values {
            approx::assert_relative_eq!(v, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn uniform_reset_with_p_zero_is_identity() {
        let device: FlexDevice = Default::default();
        let mut rng = StdRng::seed_from_u64(9);
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![3.0_f32, 4.0, 5.0, 6.0], [2, 2]),
            &device,
        );
        let out = uniform_reset(input.clone(), -10.0, 10.0, 0.0, &mut rng, &device);
        let before = input.into_data().into_vec::<f32>().unwrap();
        let after = out.into_data().into_vec::<f32>().unwrap();
        for (a, b) in before.iter().zip(after.iter()) {
            approx::assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }
}
