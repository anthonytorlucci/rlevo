//! NEAT and bounded architecture search run through `Evaluator::run_trials`.
//!
//! Both are `Strategy`s, so each trial is an `EvolutionaryLoop` in a
//! `GenerationTrial`, with no code of their own on the harness side (ADR
//! 0098). Each suite runs twice and must repeat itself exactly, and the
//! architecture search is scored as a cost, which the loop negates and the
//! report gives back in the scorers' own sense.

use std::collections::BTreeMap;

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, TensorData, activation};
use rlevo_core::objective::ObjectiveSense;
use rlevo_evolution::fitness::BatchFitnessFn;
use rlevo_evolution::neuroevolution::phenotype::InterpretedPhenotype;
use rlevo_evolution::neuroevolution::topology::TopologyGenome;
use rlevo_evolution::strategy::EvolutionaryLoop;
use rlevo_evolution::{
    ArchNasBuilder, ArchNasFitnessFn, ArchNasStrategy, NasBuilderConfig, NeatParams, NeatStrategy,
};
use rlevo_harness::evaluator::{Evaluator, EvaluatorConfig, GenerationTrial};
use rlevo_harness::metrics::core::WALL_CLOCK_SECONDS;
use rlevo_harness::report::BenchmarkReport;
use rlevo_harness::reporter::logging::LoggingReporter;

const GENS: usize = 12;
const XOR_INPUTS: [f32; 8] = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
const XOR_TARGETS: [f32; 4] = [0.0, 1.0, 1.0, 0.0];

fn cfg() -> EvaluatorConfig {
    EvaluatorConfig {
        num_episodes: 1,
        train_steps: 0,
        num_trials_per_env: 2,
        max_steps: GENS,
        base_seed: 2026,
        num_threads: Some(1),
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: None,
    }
}

fn xor_inputs(device: &Device) -> Tensor<2> {
    Tensor::<2>::from_data(TensorData::new(XOR_INPUTS.to_vec(), [4, 2]), device)
}

/// XOR as `4 - SSE`, a score.
struct XorScore {
    inputs: Tensor<2>,
}

impl BatchFitnessFn<Vec<TopologyGenome>> for XorScore {
    fn evaluate_batch(&mut self, population: &Vec<TopologyGenome>, device: &Device) -> Tensor<1> {
        let scores: Vec<f32> = population
            .iter()
            .map(|genome| {
                let out = InterpretedPhenotype::new(genome).forward(self.inputs.clone());
                let out = out.into_data().try_into_vec::<f32>().unwrap();
                let sse: f32 = out
                    .iter()
                    .zip(XOR_TARGETS.iter())
                    .map(|(o, t)| (o - t) * (o - t))
                    .sum();
                4.0 - sse
            })
            .collect();
        let n = scores.len();
        Tensor::<1>::from_data(TensorData::new(scores, [n]), device)
    }

    fn sense(&self) -> ObjectiveSense {
        ObjectiveSense::Maximize
    }
}

fn neat_loop(seed: u64) -> EvolutionaryLoop<NeatStrategy, XorScore> {
    let device = Device::flex();
    EvolutionaryLoop::new(
        NeatStrategy::new(),
        NeatParams::default_for(32, 2, 1),
        XorScore {
            inputs: xor_inputs(&device),
        },
        seed,
        device,
        GENS,
    )
    .expect("the NEAT defaults are valid")
}

#[derive(Module, Debug)]
struct Shallow {
    l1: Linear,
    l2: Linear,
}

#[derive(Module, Debug)]
struct Deep {
    l1: Linear,
    l2: Linear,
    l3: Linear,
}

fn mse(predictions: Tensor<2>, targets: &Tensor<2>) -> f32 {
    let diff = predictions - targets.clone();
    diff.clone()
        .mul(diff)
        .mean()
        .into_data()
        .try_into_vec::<f32>()
        .unwrap()[0]
}

fn nas_loop(seed: u64) -> EvolutionaryLoop<ArchNasStrategy, ArchNasFitnessFn> {
    let device = Device::flex();
    let targets = Tensor::<2>::from_data(TensorData::new(XOR_TARGETS.to_vec(), [4, 1]), &device);
    let (inputs_s, targets_s) = (xor_inputs(&device), targets.clone());
    let (inputs_d, targets_d) = (xor_inputs(&device), targets);
    let mut builder = ArchNasBuilder::new();
    builder
        .add_variant(
            Shallow {
                l1: LinearConfig::new(2, 4).init(&device),
                l2: LinearConfig::new(4, 1).init(&device),
            },
            move |m: &Shallow| {
                let h = activation::tanh(m.l1.forward(inputs_s.clone()));
                mse(m.l2.forward(h), &targets_s)
            },
        )
        .add_variant(
            Deep {
                l1: LinearConfig::new(2, 8).init(&device),
                l2: LinearConfig::new(8, 4).init(&device),
                l3: LinearConfig::new(4, 1).init(&device),
            },
            move |m: &Deep| {
                let h = activation::tanh(m.l1.forward(inputs_d.clone()));
                let h = activation::tanh(m.l2.forward(h));
                mse(m.l3.forward(h), &targets_d)
            },
        );
    let (params, fitness) = builder.build(NasBuilderConfig {
        pop_size: 12,
        arch_mutation_rate: 0.1,
        weight_mutation_std: 0.1,
        weight_init_std: 0.7,
        tournament_size: 3,
        elite_count: 1,
        sense: ObjectiveSense::Minimize,
    });
    EvolutionaryLoop::new(ArchNasStrategy::new(), params, fitness, seed, device, GENS)
        .expect("the builder's parameters are valid")
}

/// Each trial's scalars, the wall clock aside, for a bit-for-bit comparison.
fn scalars(report: &BenchmarkReport) -> Vec<BTreeMap<String, u64>> {
    report
        .trials
        .iter()
        .map(|t| {
            assert!(!t.errored, "trial errored: {:?}", t.error_message);
            assert!(
                t.episodes.is_empty(),
                "a generation trial reports no episodes"
            );
            #[allow(clippy::cast_precision_loss)]
            let generations = GENS as f64;
            assert_eq!(t.scalars.get("generations").copied(), Some(generations));
            t.scalars
                .iter()
                .filter(|(name, _)| name.as_str() != WALL_CLOCK_SECONDS)
                .map(|(name, value)| (name.clone(), value.to_bits()))
                .collect()
        })
        .collect()
}

#[test]
fn neat_runs_through_run_trials_and_repeats() {
    let suite = || {
        Evaluator::new(cfg()).run_trials(
            "neat-xor",
            &["xor".to_string()],
            |_key, seed, _agent_seed| GenerationTrial {
                probe: neat_loop(seed),
            },
            &mut LoggingReporter::new(),
        )
    };
    let first = suite();
    assert_eq!(first.trials.len(), 2);
    let best = first.trials[0].scalars["ea/best_fitness_ever"];
    assert!(
        (0.0..=4.0).contains(&best),
        "a 4 - SSE score lies in [0, 4], got {best}"
    );
    assert_eq!(
        scalars(&first),
        scalars(&suite()),
        "a second suite repeats the first"
    );
}

#[test]
fn arch_nas_runs_through_run_trials_as_a_cost_and_repeats() {
    let suite = || {
        Evaluator::new(cfg()).run_trials(
            "nas-xor",
            &["xor".to_string()],
            |_key, seed, _agent_seed| GenerationTrial {
                probe: nas_loop(seed),
            },
            &mut LoggingReporter::new(),
        )
    };
    let first = suite();
    assert_eq!(first.trials.len(), 2);
    for trial in &first.trials {
        let best = trial.scalars["ea/best_fitness_ever"];
        let mean = trial.scalars["ea/mean_fitness"];
        assert!(
            best >= 0.0,
            "the report gives the MSE back as a cost, got {best}"
        );
        assert!(
            best <= mean,
            "for a cost the best is the smallest: {best} > {mean}"
        );
    }
    assert_eq!(
        scalars(&first),
        scalars(&suite()),
        "a second suite repeats the first"
    );
}
