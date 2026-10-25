//! Integration test for bounded architecture NAS: a fixed menu of network
//! topology variants (as opposed to open-ended topology search) with
//! evolution selecting among them and co-evolving weights per variant.
//!
//! Mirrors the structure of `neuroevolution_supervised.rs`: [`ArchNasStrategy`]
//! runs in an `EvolutionaryLoop`, scored by the `ArchNasFitnessFn` its builder
//! emits. The toy task is XOR: three MLP variants of differing depth compete
//! to fit the four XOR points while their weights are co-evolved.
//!
//! Everything runs on a plain `Flex` device, forward only: no module here is
//! switched to training, so no autodiff graph is built.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, TensorData, activation};

use rlevo_core::objective::ObjectiveSense;
use rlevo_evolution::{ArchNasBuilder, ArchNasStrategy, EvolutionaryLoop, NasBuilderConfig};

/// Shallow variant: `2 -> 8 -> 1` (33 params).
#[derive(Module, Debug)]
struct ShallowMlp {
    l1: Linear,
    l2: Linear,
}

impl ShallowMlp {
    fn new(device: &Device) -> Self {
        Self {
            l1: LinearConfig::new(2, 8).init(device),
            l2: LinearConfig::new(8, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let h = activation::tanh(self.l1.forward(x));
        self.l2.forward(h)
    }
}

/// Medium variant: `2 -> 16 -> 8 -> 1` (193 params).
#[derive(Module, Debug)]
struct MediumMlp {
    l1: Linear,
    l2: Linear,
    l3: Linear,
}

impl MediumMlp {
    fn new(device: &Device) -> Self {
        Self {
            l1: LinearConfig::new(2, 16).init(device),
            l2: LinearConfig::new(16, 8).init(device),
            l3: LinearConfig::new(8, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let h = activation::tanh(self.l1.forward(x));
        let h = activation::tanh(self.l2.forward(h));
        self.l3.forward(h)
    }
}

/// Deep variant: `2 -> 32 -> 16 -> 8 -> 1` (769 params).
#[derive(Module, Debug)]
struct DeepMlp {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    l4: Linear,
}

impl DeepMlp {
    fn new(device: &Device) -> Self {
        Self {
            l1: LinearConfig::new(2, 32).init(device),
            l2: LinearConfig::new(32, 16).init(device),
            l3: LinearConfig::new(16, 8).init(device),
            l4: LinearConfig::new(8, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let h = activation::tanh(self.l1.forward(x));
        let h = activation::tanh(self.l2.forward(h));
        let h = activation::tanh(self.l3.forward(h));
        self.l4.forward(h)
    }
}

/// The four XOR points as a `[4, 2]` input / `[4, 1]` target pair.
fn xor_dataset(device: &Device) -> (Tensor<2>, Tensor<2>) {
    let inputs = Tensor::<2>::from_data(
        TensorData::new(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2]),
        device,
    );
    let targets =
        Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 1.0, 1.0, 0.0], [4, 1]), device);
    (inputs, targets)
}

/// Mean-squared error of a `[4, 1]` prediction against the XOR targets.
fn mse(preds: Tensor<2>, targets: &Tensor<2>) -> f32 {
    let diff = preds - targets.clone();
    let m = diff.clone().mul(diff).mean();
    m.into_data().try_into_vec::<f32>().unwrap()[0]
}

#[test]
fn arch_nas_selects_architecture_and_improves_on_xor() {
    let device: Device = Device::flex();
    let (inputs, targets) = xor_dataset(&device);

    // Per-variant scorers: forward the four XOR points and return the MSE, a cost. The builder
    // declares it `Minimize`, and the loop negates it before the strategy sees it.
    let (in_s, tg_s) = (inputs.clone(), targets.clone());
    let shallow_scorer = move |m: &ShallowMlp| mse(m.forward(in_s.clone()), &tg_s);
    let (in_m, tg_m) = (inputs.clone(), targets.clone());
    let medium_scorer = move |m: &MediumMlp| mse(m.forward(in_m.clone()), &tg_m);
    let (in_d, tg_d) = (inputs.clone(), targets.clone());
    let deep_scorer = move |m: &DeepMlp| mse(m.forward(in_d.clone()), &tg_d);

    // Single registration point → arch_id == registration index for all three.
    let mut builder = ArchNasBuilder::new();
    builder
        .add_variant(ShallowMlp::new(&device), shallow_scorer) // arch 0
        .add_variant(MediumMlp::new(&device), medium_scorer) // arch 1
        .add_variant(DeepMlp::new(&device), deep_scorer); // arch 2
    let (params, fitness) = builder.build(NasBuilderConfig {
        pop_size: 36,
        arch_mutation_rate: 0.1,
        weight_mutation_std: 0.1,
        weight_init_std: 0.7,
        tournament_size: 3,
        elite_count: 2,
        sense: ObjectiveSense::Minimize,
    });

    assert_eq!(
        params.num_variants(),
        3,
        "exactly three architecture variants"
    );
    assert_eq!(
        params.per_variant_params(),
        vec![33, 193, 769],
        "param counts in registration order",
    );
    assert_eq!(params.max_param_count(), 769);

    // Generation 0 evaluates the seed population; 15 more follow.
    let generations = 15;
    let mut evo = EvolutionaryLoop::new(
        ArchNasStrategy::new(),
        params,
        fitness,
        2026,
        device,
        generations + 1,
    )
    .expect("the builder's parameters are valid");
    evo.reset();

    // AC: after init, all three architectures are represented (pop 36 over 3
    // variants makes a missing variant astronomically unlikely).
    let state = evo.state().expect("the loop was reset");
    for arch_id in 0..3 {
        assert!(
            state.population().arch_ids().contains(&arch_id),
            "architecture {arch_id} must be represented after init",
        );
    }

    evo.step();
    let gen0_best = evo.state().expect("the loop was reset").best_fitness();

    // AC: no panic across many generations.
    while !evo.is_done() {
        evo.step();
    }

    let state = evo.state().expect("the loop was reset");
    let final_best = state.best_fitness();
    let (champion, best_cost) = evo.best().expect("best exists after at least one tell");
    let best_arch = champion.arch_ids()[0];

    // AC: directional improvement — the final best `$-\text{MSE}$` (the strategy's canonical,
    // maximise-space value) is strictly greater (closer to 0) than the generation-0 best.
    assert!(
        final_best > gen0_best,
        "expected directional improvement: final best −MSE {final_best} \
         should be > generation-0 best −MSE {gen0_best}",
    );
    // The winning architecture is one of the three registered variants.
    assert!(best_arch < 3, "winning arch_id {best_arch} in range");
    // The loop reports the champion's MSE in the scorers' own sense.
    approx::assert_relative_eq!(best_cost, -final_best, epsilon = 1e-6);
    // Ran the full budget.
    assert_eq!(state.generation(), generations + 1);
}
