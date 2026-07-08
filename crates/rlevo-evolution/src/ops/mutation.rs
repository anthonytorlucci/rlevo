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

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::{Rng, RngExt};
use rlevo_core::probability::Probability;
use rlevo_core::rate::NonNegativeRate;

/// Builds an `(n·d,)` host vector of `N(0, 1)` draws sized for a
/// `(n, d)` tensor.
fn standard_normal_rows(n: usize, d: usize, rng: &mut dyn Rng) -> Vec<f32> {
    let mut rows = Vec::with_capacity(n * d);
    for _ in 0..n * d {
        rows.push(crate::sampling::standard_normal(rng));
    }
    rows
}

/// Isotropic Gaussian mutation with a scalar step-size σ.
///
/// Each gene in the population is independently perturbed by `σ · N(0, 1)`
/// noise. The same σ is applied to every individual and every gene dimension.
/// When `σ = 0` the function is an identity and returns a tensor numerically
/// equal to the input.
///
/// The `n·d` standard-normal draws are taken from the caller-supplied host
/// `rng` via [`crate::sampling::standard_normal`] and loaded onto the device
/// using [`Tensor::from_data`]; no backend-global RNG state is touched.
///
/// The input tensor must have shape `(N, D)` where `N` is the population size
/// and `D` is the genome length; the returned tensor has the same shape.
///
/// For per-individual step-sizes see [`gaussian_mutation_per_row`].
#[must_use]
pub fn gaussian_mutation<B: Backend>(
    population: Tensor<B, 2>,
    sigma: NonNegativeRate,
    rng: &mut dyn Rng,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    let [n, d] = population.dims();
    let noise = Tensor::<B, 2>::from_data(
        TensorData::new(standard_normal_rows(n, d, rng), [n, d]),
        device,
    );
    population + noise.mul_scalar(sigma.get())
}

/// Per-individual anisotropic Gaussian mutation.
///
/// A generalisation of [`gaussian_mutation`] that applies a distinct step-size
/// σ to each individual. `sigmas` is a `(N,)` tensor whose `i`-th entry is the
/// σ for the `i`-th genome row. All `D` genes within a row share the same σ,
/// so mutation is isotropic within a row but can vary across the population.
/// This matches the self-adaptive σ convention used by (1+1)-ES and CMA-ES
/// warm-starts.
///
/// When all entries of `sigmas` are zero the function is an identity and
/// returns a tensor numerically equal to the input. The `n·d` standard-normal
/// draws are taken from the caller-supplied host `rng` and loaded onto the
/// device via [`Tensor::from_data`]; no backend-global RNG state is touched.
///
/// # Panics
///
/// Panics if the length of `sigmas` does not equal the population's first
/// dimension `N` (the internal `reshape([n, 1])` step requires exactly `n`
/// σ values).
#[must_use]
pub fn gaussian_mutation_per_row<B: Backend>(
    population: Tensor<B, 2>,
    sigmas: Tensor<B, 1>,
    rng: &mut dyn Rng,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    let [n, d] = population.dims();
    let noise = Tensor::<B, 2>::from_data(
        TensorData::new(standard_normal_rows(n, d, rng), [n, d]),
        device,
    );
    let sigmas_2d = sigmas.reshape([n, 1]).expand([n, d]);
    population + noise * sigmas_2d
}

/// Uniform-reset mutation: replace each gene with a fresh sample from
/// `U(lo, hi)` with probability `p`.
///
/// For each gene position, an independent Bernoulli trial with success
/// probability `p` determines whether the gene is replaced by a draw from the
/// uniform distribution `U(lo, hi)` or left at its current value.
///
/// `p = 0.0` is an identity (no genes mutated); `p = 1.0` reinitialises the
/// entire population uniformly. This operator is suitable for integer-coded or
/// bounded real-valued genomes where reinitialisation is preferable to additive
/// noise.
///
/// All `n·d` uniform draws and `n·d` Bernoulli coin flips are taken from the
/// caller-supplied host `rng` and loaded onto the device via
/// [`Tensor::from_data`]; no backend-global RNG state is touched.
///
/// The input tensor must have shape `(N, D)`; the returned tensor has the same
/// shape.
#[must_use]
pub fn uniform_reset<B: Backend>(
    population: Tensor<B, 2>,
    lo: f32,
    hi: f32,
    p: Probability,
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
    let reset = coin.lower_elem(p.get());
    population.mask_where(reset, noise)
}

/// Bit-flip mutation on a binary `Tensor<B, 2, Int>` population.
///
/// Each gene is independently flipped with probability `p`. The flip is
/// computed arithmetically as `1 − x`, so the input tensor must hold values
/// exclusively in `{0, 1}`; any value outside that set will produce
/// out-of-range results silently.
///
/// `p = 0.0` is an identity; `p = 1.0` inverts every gene. The conventional
/// mutation rate for bit-string GAs is `1 / D` (one expected flip per
/// individual), but this function does not enforce any particular rate.
///
/// All `n·d` Bernoulli coin flips are taken from the caller-supplied host
/// `rng` and loaded onto the device via [`Tensor::from_data`]; no
/// backend-global RNG state is touched.
///
/// The input tensor must have shape `(N, D)`; the returned tensor has the same
/// shape and element type.
#[must_use]
pub fn bit_flip_mutation<B: Backend>(
    population: Tensor<B, 2, Int>,
    p: Probability,
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
    let flip = coin.lower_elem(p.get());
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
        let out = gaussian_mutation(input.clone(), NonNegativeRate::new(0.0), &mut rng, &device);
        let before = input
            .into_data()
            .into_vec::<f32>()
            .expect("genome host-read of a tensor this test just built");
        let after = out
            .into_data()
            .into_vec::<f32>()
            .expect("genome host-read of a tensor this test just built");
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
        let out = gaussian_mutation(input, NonNegativeRate::new(1.0), &mut rng, &device);
        assert_eq!(out.dims(), [3, 4]);
    }

    #[test]
    fn per_row_applies_distinct_sigmas() {
        let device: FlexDevice = Default::default();
        let mut rng = StdRng::seed_from_u64(4);
        let input =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(vec![0.0_f32; 4], [2, 2]), &device);
        let sigmas =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(vec![0.0_f32, 0.0], [2]), &device);
        let out = gaussian_mutation_per_row(input, sigmas, &mut rng, &device);
        let values = out
            .into_data()
            .into_vec::<f32>()
            .expect("genome host-read of a tensor this test just built");
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
        let out = uniform_reset(
            input.clone(),
            -10.0,
            10.0,
            Probability::new(0.0),
            &mut rng,
            &device,
        );
        let before = input
            .into_data()
            .into_vec::<f32>()
            .expect("genome host-read of a tensor this test just built");
        let after = out
            .into_data()
            .into_vec::<f32>()
            .expect("genome host-read of a tensor this test just built");
        for (a, b) in before.iter().zip(after.iter()) {
            approx::assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }
}
