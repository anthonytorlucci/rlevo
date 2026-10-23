//! Integration test: weight-only neuroevolution of a small MLP against a
//! supervised (MSE) fitness on a noisy sine wave.
//!
//! Exercises the full weight-only loop end-to-end: a GA over the MLP's
//! flattened weights (strategy) → `ModuleEvalFn` (fitness adapter,
//! unflattening each genome row into an MLP and scoring MSE) →
//! `EvolutionaryLoop`.
//!
//! The assertion is deliberately **directional**: the best MSE after 50
//! generations must be strictly better than the best MSE in generation 0. This
//! proves the optimization loop actually descends without pinning a brittle
//! absolute convergence threshold that a fixed seed could satisfy by chance.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, TensorData, activation};

use rlevo_core::rate::NonNegativeRate;
use rlevo_evolution::algorithms::ga::{GaConfig, GeneticAlgorithm};
use rlevo_evolution::module_eval_fn::ModuleEvalFn;
use rlevo_evolution::param_reshaper::ModuleReshaper;
use rlevo_evolution::strategy::EvolutionaryLoop;

/// `1 -> 16 -> 1` MLP with a tanh hidden activation — enough capacity to bend
/// toward a sine curve.
#[derive(Module, Debug)]
struct SineMlp {
    l1: Linear,
    l2: Linear,
}

impl SineMlp {
    fn new(device: &Device) -> Self {
        Self {
            l1: LinearConfig::new(1, 16).init(device),
            l2: LinearConfig::new(16, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let h = activation::tanh(self.l1.forward(x));
        self.l2.forward(h)
    }
}

/// Builds a fixed `(inputs, targets)` dataset: `y = sin(x)` plus a small,
/// deterministic pseudo-noise over `n` points in `[-pi, pi]`.
fn dataset(device: &Device, n: usize) -> (Tensor<2>, Tensor<2>) {
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
    let inputs = Tensor::<2>::from_data(TensorData::new(xs, [n, 1]), device);
    let targets = Tensor::<2>::from_data(TensorData::new(ys, [n, 1]), device);
    (inputs, targets)
}

#[test]
fn weight_only_ga_fits_noisy_sine_directional() {
    let device = Device::flex();
    let n = 32;
    let (inputs, targets) = dataset(&device, n);

    let template = SineMlp::new(&device);
    // Single-source width: build ONE reshaper, read its width, then move it
    // into the fitness adapter so both agree on `num_params` by construction.
    let reshaper = ModuleReshaper::new(template);
    let num_params = reshaper.num_params();
    // 1*16 + 16 + 16*1 + 1 = 49
    assert_eq!(num_params, 49);

    // Fitness: mean squared error of the reconstructed net over the dataset.
    let mse_inputs = inputs;
    let mse_targets = targets;
    let scorer = move |m: &SineMlp| -> f32 {
        let preds = m.forward(mse_inputs.clone());
        let diff = preds - mse_targets.clone();
        let mse = diff.clone().mul(diff).mean();
        mse.into_data().try_into_vec::<f32>().unwrap()[0]
    };

    // MSE is a cost — declare Minimize so the loop reconciles direction.
    let eval = ModuleEvalFn::with_sense(
        reshaper,
        scorer,
        rlevo_core::objective::ObjectiveSense::Minimize,
    );

    let mut params = GaConfig::default_for(64, num_params);
    params.mutation_sigma = NonNegativeRate::new(0.3);
    let strategy = GeneticAlgorithm::new();

    let mut evo_loop =
        EvolutionaryLoop::new(strategy, params, eval, 42, device, 50).expect("valid params");
    evo_loop.reset();

    // Generation 0.
    evo_loop.step();
    let initial_best = evo_loop.latest_metrics().unwrap().best_fitness_ever();

    // Generations 1..50.
    while !evo_loop.is_done() {
        evo_loop.step();
    }
    let final_best = evo_loop.latest_metrics().unwrap().best_fitness_ever();

    assert!(
        final_best < initial_best,
        "expected directional improvement: final best MSE {final_best} \
         should be < generation-0 best MSE {initial_best}"
    );
    // Sanity: the loop ran the full budget.
    assert_eq!(evo_loop.generation(), 50);
}
