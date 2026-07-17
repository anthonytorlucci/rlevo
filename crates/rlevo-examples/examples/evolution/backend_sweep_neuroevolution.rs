//! Backend sweep — the *same* neuroevolution run on CPU and GPU.
//!
//! This example exists to make two points that the PPO-on-CartPole examples
//! cannot:
//!
//! 1. **Burn's backend genericity is real leverage.** The entire run below is
//!    written once over `B: Backend`. Nothing in [`run`], in the fitness
//!    function, or in `rlevo`'s evolution stack names a concrete backend. Main
//!    instantiates that one generic function on [`Flex`] (a portable CPU
//!    backend) and on [`Wgpu`] (Apple-Silicon / Vulkan / DX12 GPU) by changing
//!    a *type argument* — not a line of logic.
//!
//! 2. **GPUs win on the workload they were built for: large batched matmuls.**
//!    A single-environment PPO rollout is the opposite — tiny, sequential,
//!    sync-bound ops where a CPU backend is ~70× faster (see the sibling
//!    `ppo_cartpole` scaffolding, which is pinned to `Flex` for exactly that
//!    reason). Neuroevolution flips every one of those properties: a whole
//!    *population* of candidate networks is one `(pop, …)` tensor, and scoring
//!    it is a batched forward pass — `pop × samples × in × hidden` fused
//!    multiply-adds with **no autodiff tape and no per-step host sync**. That is
//!    GPU territory, and the GPU's margin *widens* as you scale `pop_size`.
//!
//! The fitness here is a self-contained regression task: interpret each genome
//! row as the weights of a small `in → hidden → 1` MLP, run every candidate in
//! the population over a fixed input batch in **one batched matmul**, and score
//! it by mean-squared error against a fixed nonlinear target. The optimisation
//! outcome is beside the point; the point is the *shape* of the computation and
//! where it runs fastest.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p rlevo-examples --example backend_sweep_neuroevolution
//! ```
//!
//! `--release` matters: the CPU backend leans on optimised `gemm`, and a debug
//! build distorts the comparison. A representative run (Apple M2, `in = 16`,
//! `hidden = 128`, `samples = 512`, `gens = 20`):
//!
//! ```text
//! pop_size     Flex (CPU)      Wgpu (GPU)     speedup
//!       64        0.970 s         0.187 s        5.18x
//!      512        8.209 s         0.479 s       17.14x
//!     4096       64.829 s         3.000 s       21.61x
//! ```
//!
//! Note there is no CPU-favourable crossover *inside* this range: once the work
//! is a batched matmul, the GPU wins even at `pop_size = 64`, and its margin
//! **widens** with the population. The regime where the CPU wins is the *other*
//! extreme — the tiny, sequential, single-environment PPO rollout in the
//! sibling `ppo_cartpole` scaffolding, where `Flex` is ~70× faster. The two
//! examples bracket the trade-off: sequential-and-tiny favours CPU,
//! batched-and-large favours GPU. The absolute numbers are machine-specific;
//! the *shape* of the result is the lesson.
//!
//! This example deliberately installs **no** `tracing` subscriber. The verbose
//! adapter-selection / autotune log lines you may have seen from the GPU path
//! elsewhere are `cubecl`/`wgpu` records bridged into `tracing` only when a
//! subscriber with a `LogTracer` is installed; with none installed they go
//! nowhere and the GPU run is as quiet as the CPU run.

// A perf-illustration binary: the sweep constants and the CPU/GPU pair in
// `main` are the product; the `#[cfg(test)]` smoke test drives `run` too.
#![allow(dead_code)]

use std::time::{Duration, Instant};

use burn::backend::{Flex, Wgpu};
use burn::tensor::activation::tanh;
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::{Tensor, TensorData};

use rlevo_core::bounds::Bounds;
use rlevo_core::objective::ObjectiveSense;
use rlevo_core::rate::NonNegativeRate;
use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_evolution::fitness::BatchFitnessFn;
use rlevo_evolution::strategy::EvolutionaryHarness;

/// Shared seed so both backends drive the same GA trajectory.
const SEED: u64 = 42;
/// Generations per run — enough matmul volume to time honestly, small enough
/// that the largest CPU sweep stays interactive.
const GENS: usize = 20;

/// Population sizes swept, smallest first. The GPU already wins at 64 (the
/// per-eval work is a 512-sample batched matmul); the point of the sweep is
/// that its margin *widens* as the population grows and the matmul gets larger.
const POP_SIZES: [usize; 3] = [64, 512, 4096];

/// One knob-set for a single [`run`]. Pulled into a struct so the smoke test
/// can drive a tiny, fast configuration through the identical code path.
#[derive(Debug, Clone, Copy)]
struct SweepConfig {
    /// Number of candidate networks evaluated together as one batched tensor.
    pop_size: usize,
    /// GA generations to run.
    gens: usize,
    /// Fixed regression-batch size (rows fed through every candidate at once).
    samples: usize,
    /// MLP input width.
    in_dim: usize,
    /// MLP hidden width — the dominant matmul dimension.
    hidden: usize,
}

impl SweepConfig {
    /// Flat genome length for an `in → hidden → 1` MLP:
    /// `W1 (in·hidden) + b1 (hidden) + W2 (hidden·1) + b2 (1)`.
    fn genome_dim(&self) -> usize {
        self.in_dim * self.hidden + self.hidden + self.hidden + 1
    }
}

/// Batched-on-device fitness: score a whole population of MLPs in one forward
/// pass and return per-member MSE.
///
/// This is the deliberate contrast with the bundled [`FromLandscape`] /
/// [`FromFitnessEvaluable`] adapters, which pull every genome row to the host
/// and evaluate on the CPU — correct, but backend-agnostic *because the compute
/// never touches the device*. Here the entire evaluation stays on `B`'s device
/// as batched tensor ops, so switching `B` from `Flex` to `Wgpu` genuinely
/// moves the arithmetic from CPU to GPU. That is what makes the sweep a fair
/// backend comparison rather than a measurement of host-side glue.
///
/// [`FromLandscape`]: rlevo_evolution::fitness::FromLandscape
/// [`FromFitnessEvaluable`]: rlevo_evolution::fitness::FromFitnessEvaluable
#[derive(Debug)]
struct MlpRegressionFitness<B: Backend> {
    /// Fixed input batch, shape `(samples, in_dim)`, resident on the device.
    inputs: Tensor<B, 2>,
    /// Fixed regression target, shape `(1, samples)`, broadcast over the pop.
    target: Tensor<B, 2>,
    in_dim: usize,
    hidden: usize,
    samples: usize,
}

impl<B: Backend> MlpRegressionFitness<B> {
    /// Builds the fixed dataset once, on the device. The pattern is arbitrary
    /// but deterministic — a smooth nonlinear surface the MLP can chase — so the
    /// two backends see byte-identical inputs and the comparison is apples to
    /// apples.
    fn new(cfg: &SweepConfig, device: &<B as BackendTypes>::Device) -> Self {
        let (n, i) = (cfg.samples, cfg.in_dim);

        let mut xs = Vec::with_capacity(n * i);
        let mut ys = Vec::with_capacity(n);
        for row in 0..n {
            let mut sum = 0.0_f32;
            for col in 0..i {
                #[allow(clippy::cast_precision_loss)]
                let v = (row as f32 * 0.13 + col as f32 * 0.29).sin();
                xs.push(v);
                sum += v;
            }
            ys.push((sum).sin());
        }

        let inputs = Tensor::<B, 2>::from_data(TensorData::new(xs, [n, i]), device);
        let target = Tensor::<B, 2>::from_data(TensorData::new(ys, [1, n]), device);
        Self {
            inputs,
            target,
            in_dim: i,
            hidden: cfg.hidden,
            samples: n,
        }
    }
}

impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2>> for MlpRegressionFitness<B> {
    fn evaluate_batch(
        &mut self,
        population: &Tensor<B, 2>,
        _device: &<B as BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        let [pop, _genome_dim] = population.dims();
        let (in_dim, hidden_dim, samples) = (self.in_dim, self.hidden, self.samples);

        // Column layout of one genome row: [W1 | b1 | W2 | b2]. Slice the whole
        // population at once and reshape into batched weight tensors.
        let w1_end = in_dim * hidden_dim;
        let b1_end = w1_end + hidden_dim;
        let w2_end = b1_end + hidden_dim;
        let b2_end = w2_end + 1;

        let w1 = population
            .clone()
            .slice([0..pop, 0..w1_end])
            .reshape([pop, in_dim, hidden_dim]);
        let b1 = population
            .clone()
            .slice([0..pop, w1_end..b1_end])
            .reshape([pop, 1, hidden_dim]);
        let w2 = population
            .clone()
            .slice([0..pop, b1_end..w2_end])
            .reshape([pop, hidden_dim, 1]);
        let b2 = population
            .clone()
            .slice([0..pop, w2_end..b2_end])
            .reshape([pop, 1, 1]);

        // Broadcast the shared input batch across the population: (pop, samples, in).
        let inputs = self.inputs.clone().unsqueeze_dim::<3>(0).repeat_dim(0, pop);

        // ANCHOR: forward
        // Batched forward pass — the heavy, GPU-favourable work.
        // (pop, samples, in)·(pop, in, hidden) -> (pop, samples, hidden), then ·(pop, hidden, 1).
        let activated = tanh(inputs.matmul(w1) + b1); // (pop, samples, hidden)
        let out = (activated.matmul(w2) + b2).reshape([pop, samples]); // (pop, samples)

        // MSE per member vs the broadcast target: (pop, samples) - (1, samples) -> (pop,).
        let err = out - self.target.clone();
        err.powf_scalar(2.0).mean_dim(1).squeeze_dim::<1>(1)
        // ANCHOR_END: forward
    }

    fn sense(&self) -> ObjectiveSense {
        // Regression error: smaller is better.
        ObjectiveSense::Minimize
    }
}

/// Runs the full GA for `cfg` on backend `B` and returns
/// `(best_fitness, wall_clock)`.
///
/// This is the whole point of the example: it is written *once*, generic over
/// `B: Backend`, and called below on both a CPU and a GPU backend with no other
/// change. A one-off warm-up evaluation precedes the timed loop so the GPU's
/// one-time shader-compilation / autotune cost is not charged to the measured
/// generations (the CPU backend has no such warm-up, so this only makes the
/// comparison *fairer* to the GPU).
fn run<B: Backend>(cfg: &SweepConfig) -> (f32, Duration) {
    let device = <B as BackendTypes>::Device::default();

    let mut fitness = MlpRegressionFitness::<B>::new(cfg, &device);

    // Warm up the device: compile kernels / run autotune before the clock
    // starts. `into_data()` forces a host sync so the work truly completes.
    let warm = Tensor::<B, 2>::zeros([cfg.pop_size, cfg.genome_dim()], &device);
    let _ = fitness.evaluate_batch(&warm, &device).into_data();

    let strategy = GeneticAlgorithm::<B>::new();
    let config = GaConfig {
        pop_size: cfg.pop_size,
        genome_dim: cfg.genome_dim(),
        bounds: Bounds::new(-3.0, 3.0),
        mutation_sigma: NonNegativeRate::new(0.1),
        selection: GaSelection::Tournament { size: 3 },
        crossover: GaCrossover::BlxAlpha {
            alpha: NonNegativeRate::new(0.5),
        },
        replacement: GaReplacement::Elitist { elitism_k: 2 },
    };

    let mut harness =
        EvolutionaryHarness::<B, _, _>::new(strategy, config, fitness, SEED, device, cfg.gens)
            .expect("valid GA params");
    harness.reset();

    let start = Instant::now();
    loop {
        // Each step scores the population on-device and reads fitness back to
        // host for selection, so device work is synced every generation and the
        // timer reflects real completed compute.
        if harness.step(()).done {
            break;
        }
    }
    let elapsed = start.elapsed();

    let best = harness
        .latest_metrics()
        .expect("at least one generation ran")
        .best_fitness_ever();
    (best, elapsed)
}

fn main() {
    println!(
        "Backend sweep: identical generic `run::<B>()` on Flex (CPU) vs Wgpu (GPU)\n\
         gens = {GENS}, in = {IN}, hidden = {HID}, samples = {N}\n",
        IN = 16,
        HID = 128,
        N = 512,
    );
    println!(
        "{:>9}   {:>13}   {:>13}   {:>9}",
        "pop_size", "Flex (CPU)", "Wgpu (GPU)", "speedup"
    );

    for &pop_size in &POP_SIZES {
        let cfg = SweepConfig {
            pop_size,
            gens: GENS,
            samples: 512,
            in_dim: 16,
            hidden: 128,
        };

        // ANCHOR: sweep
        // The same generic `run` — CPU and GPU differ only by the type argument.
        let (cpu_best, cpu_t) = run::<Flex>(&cfg);
        let (gpu_best, gpu_t) = run::<Wgpu>(&cfg);
        // ANCHOR_END: sweep
        let speedup = cpu_t.as_secs_f64() / gpu_t.as_secs_f64().max(f64::MIN_POSITIVE);

        println!(
            "{:>9}   {:>11.3} s   {:>11.3} s   {:>8.2}x",
            pop_size,
            cpu_t.as_secs_f64(),
            gpu_t.as_secs_f64(),
            speedup,
        );
        // Sanity: both backends should land in the same ballpark on the same
        // seed. Printed to stderr so it doesn't clutter the timing table.
        eprintln!("    (best MSE: Flex {cpu_best:.4}, Wgpu {gpu_best:.4} at pop_size {pop_size})");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: a tiny sweep on the CPU backend completes and yields a finite
    /// best fitness. Keeps `run` and the batched fitness compiled and exercised
    /// in CI without depending on GPU availability.
    #[test]
    fn cpu_run_completes_with_finite_fitness() {
        let cfg = SweepConfig {
            pop_size: 16,
            gens: 3,
            samples: 32,
            in_dim: 4,
            hidden: 8,
        };
        let (best, elapsed) = run::<Flex>(&cfg);
        assert!(best.is_finite(), "best fitness must be finite, got {best}");
        assert!(
            elapsed.as_nanos() > 0,
            "timed loop should take nonzero time"
        );
    }
}
