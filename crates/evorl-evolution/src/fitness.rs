//! Fitness evaluation traits and adapters.
//!
//! Two traits model the two evaluation shapes strategies expect:
//!
//! - [`FitnessFn`] — evaluates a single member. Callers hand the fitness
//!   function a host-side genome row (typically `Vec<f32>`) and receive a
//!   scalar. Useful for simple benchmarks and for unit-testing operators.
//! - [`BatchFitnessFn`] — evaluates an entire population in one call and
//!   returns a device-resident `Tensor<B, 1>` of shape `(pop_size,)`. This
//!   is the hot path — strategies call it once per generation.
//!
//! The [`FromFitnessEvaluable`] adapter bridges
//! `evorl-benchmarks::FitnessEvaluable<Individual = Vec<f64>, Landscape = L>`
//! into a [`BatchFitnessFn`] over `Tensor<B, 2>`. It pulls each row to
//! host, evaluates on the CPU, then stacks the results back onto the
//! device. That is the straightforward (and only) path for fitness
//! functions defined in terms of host-side scalar code. Purpose-built
//! batched-on-device landscapes should implement [`BatchFitnessFn`]
//! directly to avoid the round-trip.

use burn::tensor::{backend::Backend, Tensor, TensorData};

use evorl_benchmarks::agent::FitnessEvaluable;

/// Single-member fitness evaluation.
///
/// Implementors may hold mutable state (e.g. a counter for number of
/// evaluations) and are therefore `&mut self`.
pub trait FitnessFn<G>: Send {
    /// Evaluates one genome and returns its scalar fitness.
    fn evaluate_one(&mut self, member: &G) -> f32;
}

/// Batched fitness evaluation over a population genome container `G`.
///
/// The returned tensor has shape `(pop_size,)` on the supplied device.
/// Implementors must preserve row order — `fitness[i]` refers to the
/// individual at row `i` of `population`.
pub trait BatchFitnessFn<B: Backend, G>: Send {
    /// Evaluates every member of `population` and stacks fitnesses.
    fn evaluate_batch(&mut self, population: &G, device: &B::Device) -> Tensor<B, 1>;
}

/// Adapter from `FitnessEvaluable` to [`BatchFitnessFn<B, Tensor<B, 2>>`].
///
/// Each row of the population is pulled to host, converted to `Vec<f64>`,
/// and passed to the underlying evaluator with the configured landscape.
/// Fitness is computed on the host and then re-uploaded as a single
/// `Tensor<B, 1>`.
///
/// # Precision
///
/// Populations are read as `f32` and widened to `f64` for the evaluator
/// call; the returned `f64` fitness is narrowed back to `f32` before it
/// is uploaded as a `Tensor<B, 1>`. Fitness values that exceed `f32`
/// range (or rely on sub-ulp precision) will lose information at the
/// narrowing step. Purpose-built batched-on-device landscapes should
/// implement [`BatchFitnessFn`] directly to avoid the round-trip.
///
/// # Type Parameters
///
/// - `FE`: Concrete [`FitnessEvaluable`] implementation.
/// - `L`: Landscape type; must match `FE::Landscape`.
#[derive(Debug)]
pub struct FromFitnessEvaluable<FE, L> {
    evaluator: FE,
    landscape: L,
}

impl<FE, L> FromFitnessEvaluable<FE, L> {
    /// Builds the adapter from an evaluator and a landscape.
    pub fn new(evaluator: FE, landscape: L) -> Self {
        Self {
            evaluator,
            landscape,
        }
    }

    /// Returns a reference to the wrapped landscape.
    pub fn landscape(&self) -> &L {
        &self.landscape
    }
}

impl<FE, L, B> BatchFitnessFn<B, Tensor<B, 2>> for FromFitnessEvaluable<FE, L>
where
    B: Backend,
    FE: FitnessEvaluable<Individual = Vec<f64>, Landscape = L> + Send,
    L: Send + Sync,
{
    fn evaluate_batch(
        &mut self,
        population: &Tensor<B, 2>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let dims = population.shape().dims;
        assert_eq!(dims.len(), 2, "population tensor must be rank 2");
        let pop_size = dims[0];
        let genome_dim = dims[1];

        let flat = population
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("tensor data must be readable as f32");
        debug_assert_eq!(flat.len(), pop_size * genome_dim);

        let mut fitness = Vec::with_capacity(pop_size);
        let mut individual = Vec::with_capacity(genome_dim);
        for row in 0..pop_size {
            individual.clear();
            let start = row * genome_dim;
            individual.extend(flat[start..start + genome_dim].iter().map(|&v| f64::from(v)));
            let f = self.evaluator.evaluate(&individual, &self.landscape);
            #[allow(clippy::cast_possible_truncation)]
            fitness.push(f as f32);
        }

        let data = TensorData::new(fitness, [pop_size]);
        Tensor::<B, 1>::from_data(data, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type TestBackend = NdArray;

    #[derive(Debug, Clone, Copy)]
    struct Sphere;

    struct SphereFit;
    impl FitnessEvaluable for SphereFit {
        type Individual = Vec<f64>;
        type Landscape = Sphere;
        fn evaluate(&self, x: &Self::Individual, _: &Self::Landscape) -> f64 {
            x.iter().map(|v| v * v).sum()
        }
    }

    #[test]
    fn from_fitness_evaluable_preserves_row_order() {
        let device = Default::default();
        let data = TensorData::new(
            vec![1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
            [3, 3],
        );
        let pop = Tensor::<TestBackend, 2>::from_data(data, &device);

        let mut adapter = FromFitnessEvaluable::new(SphereFit, Sphere);
        let fitness = adapter.evaluate_batch(&pop, &device);

        let values = fitness.into_data().into_vec::<f32>().unwrap();
        assert_eq!(values.len(), 3);
        approx::assert_relative_eq!(values[0], 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(values[1], 4.0, epsilon = 1e-6);
        approx::assert_relative_eq!(values[2], 9.0, epsilon = 1e-6);
    }
}
