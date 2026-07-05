//! Integration test: weight-only neuroevolution of a small MLP against a
//! supervised (MSE) fitness on a noisy sine wave.
//!
//! Exercises the full weight-only loop end-to-end:
//! `WeightOnly<B, GA, Mlp>` (strategy) → `ModuleEvalFn` (fitness adapter,
//! unflattening each genome row into an MLP and scoring MSE) →
//! `EvolutionaryHarness`.
//!
//! The assertion is deliberately **directional**: the best MSE after 50
//! generations must be strictly better than the best MSE in generation 0. This
//! proves the optimization loop actually descends without pinning a brittle
//! absolute convergence threshold that a fixed seed could satisfy by chance.

use burn::backend::Flex;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor, TensorData, activation, backend::Backend};

use rlevo_evolution::algorithms::ga::{GaConfig, GeneticAlgorithm};
use rlevo_evolution::module_eval_fn::ModuleEvalFn;
use rlevo_evolution::param_reshaper::ModuleReshaper;
use rlevo_evolution::strategy::EvolutionaryHarness;
use rlevo_evolution::WeightOnly;

type TestBackend = Flex;

/// `1 -> 16 -> 1` MLP with a tanh hidden activation — enough capacity to bend
/// toward a sine curve.
///
/// The backend generic must be named `B`: Burn's `#[derive(Module)]` emits
/// code that references `B::Device` by that literal name.
#[derive(Module, Debug)]
struct SineMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

impl<B: Backend> SineMlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(1, 16).init(device),
            l2: LinearConfig::new(16, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::tanh(self.l1.forward(x));
        self.l2.forward(h)
    }
}

/// Builds a fixed `(inputs, targets)` dataset: `y = sin(x)` plus a small,
/// deterministic pseudo-noise over `n` points in `[-pi, pi]`.
fn dataset(
    device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
    n: usize,
) -> (Tensor<TestBackend, 2>, Tensor<TestBackend, 2>) {
    #![allow(clippy::trivially_copy_pass_by_ref)]
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    #[allow(clippy::cast_precision_loss)]
    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        let x = -std::f32::consts::PI + t * 2.0 * std::f32::consts::PI;
        // Deterministic, seed-free "noise": a small high-frequency wiggle.
        let noise = 0.05 * (12.9898 * x).sin();
        xs.push(x);
        ys.push(x.sin() + noise);
    }
    let inputs = Tensor::<TestBackend, 2>::from_data(TensorData::new(xs, [n, 1]), device);
    let targets = Tensor::<TestBackend, 2>::from_data(TensorData::new(ys, [n, 1]), device);
    (inputs, targets)
}

#[test]
fn weight_only_ga_fits_noisy_sine_directional() {
    let device = Default::default();
    let n = 32;
    let (inputs, targets) = dataset(&device, n);

    let template = SineMlp::<TestBackend>::new(&device);
    let num_params = ModuleReshaper::new(template.clone()).num_params();
    // 1*16 + 16 + 16*1 + 1 = 49
    assert_eq!(num_params, 49);

    // Fitness: mean squared error of the reconstructed net over the dataset.
    let mse_inputs = inputs;
    let mse_targets = targets;
    let scorer = move |m: &SineMlp<TestBackend>| -> f32 {
        let preds = m.forward(mse_inputs.clone());
        let diff = preds - mse_targets.clone();
        let mse = diff.clone().mul(diff).mean();
        mse.into_data().into_vec::<f32>().unwrap()[0]
    };

    // MSE is a cost — declare Minimize so the harness reconciles direction.
    let eval = ModuleEvalFn::with_sense(
        ModuleReshaper::new(template.clone()),
        scorer,
        rlevo_core::objective::ObjectiveSense::Minimize,
    );

    let mut params = GaConfig::default_for(64, num_params);
    params.mutation_sigma = 0.3;
    let strategy = WeightOnly::new(GeneticAlgorithm::<TestBackend>::new(), template);

    let mut harness =
        EvolutionaryHarness::<TestBackend, _, _>::new(strategy, params, eval, 42, device, 50).expect("valid params");
    harness.reset();

    // Generation 0.
    harness.step(());
    let initial_best = harness.latest_metrics().unwrap().best_fitness_ever;

    // Generations 1..50.
    loop {
        if harness.step(()).done {
            break;
        }
    }
    let final_best = harness.latest_metrics().unwrap().best_fitness_ever;

    assert!(
        final_best < initial_best,
        "expected directional improvement: final best MSE {final_best} \
         should be < generation-0 best MSE {initial_best}"
    );
    // Sanity: the loop ran the full budget.
    assert_eq!(harness.generation(), 50);
}
