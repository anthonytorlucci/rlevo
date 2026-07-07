//! [`ModuleEvalFn`] — fitness adapter that scores a population of flat
//! parameter rows by reconstructing one Burn module per row.
//!
//! This is the *only* place flatten/unflatten happens in the weight-only
//! pipeline. The [`WeightOnly`](crate::algorithms::neuroevolution::WeightOnly)
//! strategy keeps the genome as a flat `Tensor<B, 2>` end-to-end and never
//! reshapes; reshaping is confined to this evaluation boundary so the inner
//! strategy's selection/crossover/mutation operate on plain tensors.
//!
//! Evaluation is loop-over-N: Burn 0.21 has no `vmap`/batched-forward
//! primitive, so each population row is unflattened into a module and scored
//! individually. A batched forward path is a future addition.

use std::marker::PhantomData;

use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::fitness::BatchFitnessFn;
use crate::param_reshaper::ParamReshaper;
use rlevo_core::objective::ObjectiveSense;

/// Bridges a flat population tensor to per-member module scoring.
///
/// Implements [`BatchFitnessFn<B, Tensor<B, 2>>`]: each row of the
/// `(pop_size, num_params)` population is unflattened into a module via the
/// [`ParamReshaper`], passed to a host-side `scorer`, and the resulting scalar
/// is collected into the fitness tensor in the scorer's **natural** value
/// space. Direction is declared once via [`ObjectiveSense`] (default
/// [`ObjectiveSense::Maximize`]; pass [`with_sense`](ModuleEvalFn::with_sense)
/// for a cost scorer like MSE) and reconciled by the harness — no hand-negation
/// in the scorer.
///
/// # Gradient isolation
///
/// `B: Backend`, not `AutodiffBackend`. The reconstructed modules carry no
/// gradient tracking — `scorer` performs forward-only work (loss, rollout
/// return, …).
///
/// # Type Parameters
///
/// - `R`: a [`ParamReshaper`] producing `R::Module`.
/// - `F`: a host-side `Fn(&R::Module) -> f32` scorer (MSE, accuracy, negative
///   episode return, …).
///
/// # Device convention
///
/// Population rows are sliced on the supplied `device`; the reshaper splats
/// them into modules on that same device. Callers must construct the
/// population and the reshaper's template on one device.
pub struct ModuleEvalFn<B: Backend, R: ParamReshaper<B>, F> {
    reshaper: R,
    scorer: F,
    sense: ObjectiveSense,
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend, R: ParamReshaper<B>, F> std::fmt::Debug for ModuleEvalFn<B, R, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleEvalFn").finish_non_exhaustive()
    }
}

impl<B, R, F> ModuleEvalFn<B, R, F>
where
    B: Backend,
    R: ParamReshaper<B>,
    F: Fn(&R::Module) -> f32 + Send,
{
    /// Build an evaluator from a reshaper and a per-module scorer, defaulting
    /// the objective sense to [`ObjectiveSense::Maximize`] (accuracy, reward,
    /// episode return). Use [`with_sense`](Self::with_sense) for a cost scorer
    /// such as MSE.
    pub fn new(reshaper: R, scorer: F) -> Self {
        Self::with_sense(reshaper, scorer, ObjectiveSense::Maximize)
    }

    /// Build an evaluator with an explicit [`ObjectiveSense`].
    pub fn with_sense(reshaper: R, scorer: F, sense: ObjectiveSense) -> Self {
        Self {
            reshaper,
            scorer,
            sense,
            _backend: PhantomData,
        }
    }

    /// Borrow the underlying reshaper.
    #[must_use]
    pub fn reshaper(&self) -> &R {
        &self.reshaper
    }
}

impl<B, R, F> BatchFitnessFn<B, Tensor<B, 2>> for ModuleEvalFn<B, R, F>
where
    B: Backend,
    R: ParamReshaper<B>,
    F: Fn(&R::Module) -> f32 + Send,
{
    /// Score every row of `population` by reconstructing one module per row.
    ///
    /// # Panics
    ///
    /// Panics at batch entry if the population width (`population.dims()[1]`)
    /// differs from the reshaper's [`num_params`](ParamReshaper::num_params),
    /// failing fast before any per-row work rather than mid-loop inside
    /// [`ParamReshaper::unflatten`].
    fn evaluate_batch(&mut self, population: &Tensor<B, 2>, device: &B::Device) -> Tensor<B, 1> {
        let [pop_size, num_params] = population.dims();
        assert_eq!(
            num_params,
            self.reshaper.num_params(),
            "population genome width must equal reshaper num_params"
        );
        let mut fitness: Vec<f32> = Vec::with_capacity(pop_size);
        for row in 0..pop_size {
            #[allow(clippy::single_range_in_vec_init)]
            let genome: Tensor<B, 1> = population
                .clone()
                .slice([row..row + 1])
                .reshape([num_params]);
            let module = self.reshaper.unflatten(genome);
            fitness.push((self.scorer)(&module));
        }
        Tensor::<B, 1>::from_data(TensorData::new(fitness, [pop_size]), device)
    }

    fn sense(&self) -> ObjectiveSense {
        self.sense
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param_reshaper::ModuleReshaper;
    use burn::backend::Flex;
    use burn::module::Module;
    use burn::nn::{Linear, LinearConfig};

    type TestBackend = Flex;

    /// Single linear layer `2 -> 1`: 2 weights + 1 bias = 3 float leaves.
    #[derive(Module, Debug)]
    struct TestTiny<B: Backend> {
        l: Linear<B>,
    }

    impl<B: Backend> TestTiny<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                l: LinearConfig::new(2, 1).init(device),
            }
        }

        fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
            self.l.forward(x)
        }
    }

    #[test]
    fn test_module_eval_fn_evaluate_batch_preserves_shape_and_order() {
        let device = Default::default();
        let reshaper = ModuleReshaper::new(TestTiny::<TestBackend>::new(&device));
        let num_params = reshaper.num_params();
        assert_eq!(num_params, 3);

        // Scorer: forward a fixed input [1, 1] through the reconstructed net and
        // return the scalar output — deterministic given the genome.
        let dev = device;
        let mut eval = ModuleEvalFn::new(reshaper, move |m: &TestTiny<TestBackend>| {
            let x = Tensor::<TestBackend, 2>::from_data(TensorData::new(vec![1.0f32, 1.0], [1, 2]), &dev);
            let y = m.forward(x);
            y.into_data().into_vec::<f32>().expect("output host-read of a tensor this test just built")[0]
        });

        // Row 0: weights [1, 0], bias [0] -> output = 1*1 + 0*1 + 0 = 1.
        // Row 1: weights [0, 0], bias [5] -> output = 5.
        let pop = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 5.0], [2, 3]),
            &device,
        );
        let fitness = eval.evaluate_batch(&pop, &device);
        let values = fitness.into_data().into_vec::<f32>().expect("fitness host-read of a tensor this test just built");
        assert_eq!(values.len(), 2);
        approx::assert_relative_eq!(values[0], 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(values[1], 5.0, epsilon = 1e-6);
    }
}
