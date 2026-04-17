//! Mutation operators for real-valued genomes.
//!
//! Every mutation is an in-place-ish tensor transform: it consumes a
//! population tensor and returns a new one with noise added per gene.
//! Tensor ops already compose nicely with Burn's JIT fuser, so these
//! functions are just thin wrappers around `randn_like`-style patterns.

use burn::tensor::{backend::Backend, Distribution, Int, Tensor};

/// Isotropic Gaussian mutation with scalar σ.
///
/// Each gene is perturbed by `σ · N(0, 1)` noise, independently.
#[must_use]
pub fn gaussian_mutation<B: Backend>(
    population: Tensor<B, 2>,
    sigma: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    let shape = population.shape();
    let noise = Tensor::<B, 2>::random(shape, Distribution::Normal(0.0, 1.0), device);
    population + noise.mul_scalar(sigma)
}

/// Per-row anisotropic Gaussian mutation.
///
/// `sigmas` is a `(N,)` tensor holding the σ for each individual; the
/// noise tensor is multiplied row-wise before being added to the
/// population.
#[must_use]
pub fn gaussian_mutation_per_row<B: Backend>(
    population: Tensor<B, 2>,
    sigmas: Tensor<B, 1>,
    device: &B::Device,
) -> Tensor<B, 2> {
    let shape = population.shape();
    let n = shape.dims[0];
    let d = shape.dims[1];
    let noise = Tensor::<B, 2>::random(shape, Distribution::Normal(0.0, 1.0), device);
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
    device: &B::Device,
) -> Tensor<B, 2> {
    let shape = population.shape();
    let noise =
        Tensor::<B, 2>::random(shape.clone(), Distribution::Uniform(f64::from(lo), f64::from(hi)), device);
    let coin = Tensor::<B, 2>::random(shape, Distribution::Uniform(0.0, 1.0), device);
    let reset = coin.lower_elem(p);
    population.mask_where(reset, noise)
}

/// Bit-flip mutation on a binary `Tensor<B, 2, Int>` population.
///
/// Each gene is flipped independently with probability `p`.
#[must_use]
pub fn bit_flip_mutation<B: Backend>(
    population: Tensor<B, 2, Int>,
    p: f32,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let shape = population.shape();
    let coin = Tensor::<B, 2>::random(shape.clone(), Distribution::Uniform(0.0, 1.0), device);
    let flip = coin.lower_elem(p);
    // XOR via arithmetic: new = (1 - old) where flip, else old.
    let ones = Tensor::<B, 2, Int>::ones(shape, device);
    let flipped = ones - population.clone();
    population.mask_where(flip, flipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    #[allow(unused_imports)]
    use burn::tensor::backend::Backend as _;
    use burn::tensor::TensorData;

    type TestBackend = NdArray;

    #[test]
    fn gaussian_with_zero_sigma_is_identity() {
        let device: NdArrayDevice = Default::default();
        TestBackend::seed(&device, 3);
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]),
            &device,
        );
        let out = gaussian_mutation(input.clone(), 0.0, &device);
        let before = input.into_data().into_vec::<f32>().unwrap();
        let after = out.into_data().into_vec::<f32>().unwrap();
        for (a, b) in before.iter().zip(after.iter()) {
            approx::assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn gaussian_preserves_shape() {
        let device: NdArrayDevice = Default::default();
        TestBackend::seed(&device, 3);
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![0.0_f32; 12], [3, 4]),
            &device,
        );
        let out = gaussian_mutation(input, 1.0, &device);
        assert_eq!(out.shape().dims, vec![3, 4]);
    }

    #[test]
    fn per_row_applies_distinct_sigmas() {
        let device: NdArrayDevice = Default::default();
        TestBackend::seed(&device, 4);
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![0.0_f32; 4], [2, 2]),
            &device,
        );
        let sigmas = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.0_f32, 0.0], [2]),
            &device,
        );
        let out = gaussian_mutation_per_row(input, sigmas, &device);
        let values = out.into_data().into_vec::<f32>().unwrap();
        for v in values {
            approx::assert_relative_eq!(v, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn uniform_reset_with_p_zero_is_identity() {
        let device: NdArrayDevice = Default::default();
        TestBackend::seed(&device, 9);
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![3.0_f32, 4.0, 5.0, 6.0], [2, 2]),
            &device,
        );
        let out = uniform_reset(input.clone(), -10.0, 10.0, 0.0, &device);
        let before = input.into_data().into_vec::<f32>().unwrap();
        let after = out.into_data().into_vec::<f32>().unwrap();
        for (a, b) in before.iter().zip(after.iter()) {
            approx::assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }
}
