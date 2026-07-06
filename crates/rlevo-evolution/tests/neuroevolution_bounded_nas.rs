//! Integration test for bounded architecture NAS (issue #42).
//!
//! Mirrors the structure of `neuroevolution_supervised.rs` but drives the
//! custom [`ArchNasStrategy`] harness directly (it is **not** a `Strategy<B>`,
//! so the `EvolutionaryHarness` adapter does not apply). The toy task is XOR:
//! three MLP variants of differing depth compete to fit the four XOR points
//! while their weights are co-evolved.
//!
//! Backend is `Flex` and every type is `B: Backend` only — no `AutodiffBackend`
//! anywhere (gradient isolation at the type level).

use burn::backend::Flex;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor, TensorData, activation, backend::Backend};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_evolution::{ArchNasBuilder, ArchNasStrategy, NasBuilderConfig};

type TestBackend = Flex;
type Device = <TestBackend as burn::tensor::backend::BackendTypes>::Device;

/// Shallow variant: `2 -> 8 -> 1` (33 params).
#[derive(Module, Debug)]
struct ShallowMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

impl<B: Backend> ShallowMlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(2, 8).init(device),
            l2: LinearConfig::new(8, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::tanh(self.l1.forward(x));
        self.l2.forward(h)
    }
}

/// Medium variant: `2 -> 16 -> 8 -> 1` (193 params).
#[derive(Module, Debug)]
struct MediumMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> MediumMlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(2, 16).init(device),
            l2: LinearConfig::new(16, 8).init(device),
            l3: LinearConfig::new(8, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::tanh(self.l1.forward(x));
        let h = activation::tanh(self.l2.forward(h));
        self.l3.forward(h)
    }
}

/// Deep variant: `2 -> 32 -> 16 -> 8 -> 1` (769 params).
#[derive(Module, Debug)]
struct DeepMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    l4: Linear<B>,
}

impl<B: Backend> DeepMlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(2, 32).init(device),
            l2: LinearConfig::new(32, 16).init(device),
            l3: LinearConfig::new(16, 8).init(device),
            l4: LinearConfig::new(8, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::tanh(self.l1.forward(x));
        let h = activation::tanh(self.l2.forward(h));
        let h = activation::tanh(self.l3.forward(h));
        self.l4.forward(h)
    }
}

/// The four XOR points as a `[4, 2]` input / `[4, 1]` target pair.
fn xor_dataset(device: Device) -> (Tensor<TestBackend, 2>, Tensor<TestBackend, 2>) {
    let inputs = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2]),
        &device,
    );
    let targets = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![0.0f32, 1.0, 1.0, 0.0], [4, 1]),
        &device,
    );
    (inputs, targets)
}

/// Mean-squared error of a `[4, 1]` prediction against the XOR targets.
fn mse(preds: Tensor<TestBackend, 2>, targets: &Tensor<TestBackend, 2>) -> f32 {
    let diff = preds - targets.clone();
    let m = diff.clone().mul(diff).mean();
    m.into_data().into_vec::<f32>().unwrap()[0]
}

#[test]
fn arch_nas_selects_architecture_and_improves_on_xor() {
    let device: Device = Default::default();
    let (inputs, targets) = xor_dataset(device);

    // Per-variant scorers: forward the four XOR points, return −MSE — a
    // canonical maximise fitness (higher is better; arch_nas is maximise-native
    // and this test drives the strategy directly, so no harness chokepoint
    // reconciles direction).
    let (in_s, tg_s) = (inputs.clone(), targets.clone());
    let shallow_scorer = move |m: &ShallowMlp<TestBackend>| -mse(m.forward(in_s.clone()), &tg_s);
    let (in_m, tg_m) = (inputs.clone(), targets.clone());
    let medium_scorer = move |m: &MediumMlp<TestBackend>| -mse(m.forward(in_m.clone()), &tg_m);
    let (in_d, tg_d) = (inputs.clone(), targets.clone());
    let deep_scorer = move |m: &DeepMlp<TestBackend>| -mse(m.forward(in_d.clone()), &tg_d);

    // Single registration point → arch_id == registration index for all three.
    let mut builder = ArchNasBuilder::<TestBackend>::new();
    builder
        .add_variant(ShallowMlp::<TestBackend>::new(&device), shallow_scorer) // arch 0
        .add_variant(MediumMlp::<TestBackend>::new(&device), medium_scorer) // arch 1
        .add_variant(DeepMlp::<TestBackend>::new(&device), deep_scorer); // arch 2
    let (params, fitness) = builder.build(NasBuilderConfig {
        pop_size: 36,
        arch_mutation_rate: 0.1,
        weight_mutation_std: 0.1,
        weight_init_std: 0.7,
        tournament_size: 3,
        elite_count: 2,
    });

    assert_eq!(params.num_variants(), 3, "exactly three architecture variants");
    assert_eq!(
        params.per_variant_params(),
        vec![33, 193, 769],
        "param counts in registration order",
    );
    assert_eq!(params.max_param_count(), 769);

    let strat = ArchNasStrategy::<TestBackend>::new();
    let mut rng = StdRng::seed_from_u64(2026);
    let mut state = strat.init(&params, &mut rng, &device);

    // AC: after init, all three architectures are represented (pop 36 over 3
    // variants makes a missing variant astronomically unlikely).
    for arch_id in 0..3 {
        assert!(
            state.population().arch_ids().contains(&arch_id),
            "architecture {arch_id} must be represented after init",
        );
    }

    // Generation 0: evaluate the seed population.
    let (genome, next) = strat.ask(&params, &state, &mut rng, &device);
    let fit = fitness.evaluate(&genome, &device);
    state = strat.tell(&params, genome, fit, next, &mut rng);
    let gen0_best = state.best_fitness();

    // AC: no panic across many generations.
    let generations = 15;
    for _ in 0..generations {
        let (genome, next) = strat.ask(&params, &state, &mut rng, &device);
        let fit = fitness.evaluate(&genome, &device);
        state = strat.tell(&params, genome, fit, next, &mut rng);
    }

    let final_best = state.best_fitness();
    let (best_arch, _best_weights, best_cost) = strat
        .best(&state)
        .expect("best exists after at least one tell");

    // AC: directional improvement (canonical maximise) — final best −MSE is
    // strictly greater (closer to 0) than the generation-0 best.
    assert!(
        final_best > gen0_best,
        "expected directional improvement: final best −MSE {final_best} \
         should be > generation-0 best −MSE {gen0_best}",
    );
    // The winning architecture is one of the three registered variants.
    assert!(best_arch < 3, "winning arch_id {best_arch} in range");
    // best() and best_fitness() agree.
    approx::assert_relative_eq!(best_cost, final_best, epsilon = 1e-6);
    // Ran the full budget.
    assert_eq!(state.generation(), generations + 1);
}
