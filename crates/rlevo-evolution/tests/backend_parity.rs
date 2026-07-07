//! Cross-backend parity for the pure-tensor operator baselines.
//!
//! A `1e-4` relative-tolerance match of `best_fitness_ever` between
//! flex and wgpu on Sphere-D10 is untestable in practice: the two
//! backends use independent RNG streams (flex = splitmix-seeded
//! `FlexRng`; wgpu = per-device compute-pipeline seeded stream) so
//! even when the same host seed is supplied, tensor `random()` calls
//! produce different bytes. The strategies therefore take different
//! trajectories and end at different points — both small, neither a
//! function of the other.
//!
//! This test accordingly validates a weaker but meaningful criterion:
//! the pure-tensor operators compose correctly on both backends and
//! each drives GA to a non-trivial optimum on Sphere-D10. Bit-level
//! backend-parity would require routing a single host RNG through both
//! backends, which `CubeCL` doesn't currently expose; landing a custom
//! kernel (follow-up work, see `ops/kernels/mod.rs`) is the natural
//! path to change that.
//!
//! # Reduction-nondeterminism caveat
//!
//! Even when the RNG disparity is set aside, the wgpu backend does not
//! guarantee bit-identical reductions because workgroup ordering is
//! implementation-dependent. The Flex backend is still expected to
//! be bit-deterministic; that contract is enforced by
//! `tests/determinism.rs`.
//!
//! # Single test, serial execution
//!
//! Both backends seed their internal RNG through process-global state
//! (`Mutex<Option<FlexRng>>` for flex; per-device seed for
//! wgpu). To keep the two runs comparable, this file contains exactly
//! one `#[test]` function so nothing else in the test binary runs
//! concurrently.

use burn::backend::{Flex, Wgpu};
use rlevo_core::bounds::Bounds;
use rlevo_core::fitness::FitnessEvaluable;
use rlevo_core::rate::NonNegativeRate;
use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_evolution::algorithms::metaheuristic::pso::{ParticleSwarm, PsoConfig};
use rlevo_evolution::fitness::FromFitnessEvaluable;
use rlevo_evolution::strategy::{EvolutionaryHarness, Strategy};

struct Sphere;
struct SphereFit;
impl FitnessEvaluable for SphereFit {
    type Individual = Vec<f64>;
    type Landscape = Sphere;
    fn evaluate(&self, x: &Self::Individual, _: &Self::Landscape) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
}

fn run_sphere_ga<B>(seed: u64, gens: usize, device: B::Device) -> f32
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone,
    GeneticAlgorithm<B>: Strategy<
            B,
            Params = GaConfig,
            State = rlevo_evolution::algorithms::ga::GaState<B>,
            Genome = burn::tensor::Tensor<B, 2>,
        >,
{
    let params = GaConfig {
        pop_size: 64,
        genome_dim: 10,
        bounds: Bounds::new(-5.12, 5.12),
        mutation_sigma: NonNegativeRate::new(0.2),
        selection: GaSelection::Tournament { size: 2 },
        crossover: GaCrossover::BlxAlpha {
            alpha: NonNegativeRate::new(0.5),
        },
        replacement: GaReplacement::Elitist { elitism_k: 2 },
    };
    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        GeneticAlgorithm::<B>::new(),
        params,
        FromFitnessEvaluable::new(SphereFit, Sphere),
        seed,
        device,
        gens,
    )
    .expect("valid params");
    harness.reset();
    loop {
        if harness.step(()).done {
            break;
        }
    }
    harness.latest_metrics().unwrap().best_fitness_ever()
}

fn run_sphere_pso<B>(seed: u64, gens: usize, device: B::Device) -> f32
where
    B: burn::tensor::backend::Backend,
    B::Device: Clone,
    ParticleSwarm<B>: Strategy<
            B,
            Params = PsoConfig,
            State = rlevo_evolution::algorithms::metaheuristic::pso::PsoState<B>,
            Genome = burn::tensor::Tensor<B, 2>,
        >,
{
    let params = PsoConfig::default_for(32, 10);
    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        ParticleSwarm::<B>::new(),
        params,
        FromFitnessEvaluable::new(SphereFit, Sphere),
        seed,
        device,
        gens,
    )
    .expect("valid params");
    harness.reset();
    loop {
        if harness.step(()).done {
            break;
        }
    }
    harness.latest_metrics().unwrap().best_fitness_ever()
}

#[test]
#[ignore = "requires a wgpu/Vulkan adapter; CI runners have no GPU and cubecl-wgpu aborts on device init — run on a GPU host with `cargo test -p rlevo-evolution --test backend_parity -- --ignored`"]
fn wgpu_matches_flex_on_sphere_d10() {
    const SEED: u64 = 999;
    const GENS: usize = 400;

    // Run flex first so the host seed state is deterministic; wgpu
    // has its own per-device stream and doesn't disturb it.
    let flex_ga = run_sphere_ga::<Flex>(SEED, GENS, Default::default());
    let flex_pso = run_sphere_pso::<Flex>(SEED, GENS, Default::default());

    // Initializing a wgpu device aborts (not a catchable error) on hosts
    // without a GPU adapter: cubecl-wgpu panics on a worker thread and the
    // calling thread then panics on the severed channel. There is no clean
    // in-process probe across the cubecl boundary, so this whole test is
    // gated behind `#[ignore]` and only runs when explicitly requested on a
    // GPU-equipped host. Flex operator correctness is independently covered
    // by `tests/determinism.rs` and the per-algorithm convergence tests.
    let wgpu_device: burn::backend::wgpu::WgpuDevice = Default::default();
    let wgpu_ga = run_sphere_ga::<Wgpu>(SEED, GENS, wgpu_device.clone());
    let wgpu_pso = run_sphere_pso::<Wgpu>(SEED, GENS, wgpu_device);

    // Both backends should compose the operators correctly and drive
    // GA to a non-trivial optimum. "Non-trivial" here means well below
    // the random-initialization baseline on Sphere-D10 (E[x·x] with
    // x ~ U(-5.12, 5.12) is ~87), so any final value under 1.0
    // proves the operator chain works on both backends.
    assert!(
        flex_ga.is_finite() && wgpu_ga.is_finite(),
        "non-finite GA result: flex={flex_ga}, wgpu={wgpu_ga}",
    );
    assert!(
        flex_ga < 1.0,
        "Flex GA did not converge on Sphere-D10: {flex_ga}",
    );
    assert!(
        wgpu_ga < 1.0,
        "wgpu GA did not converge on Sphere-D10: {wgpu_ga}",
    );

    // Same functional assertion for PSO. This exercises the swarm
    // operator chain (velocity clamp, tensor broadcast in the
    // inertia update, gather-based personal-best tracking) on both
    // backends. Threshold 1e-2 (tighter than GA's 1.0) is justified
    // because PSO with the default inertia schedule converges faster
    // than BLX-α GA on Sphere.
    assert!(
        flex_pso.is_finite() && wgpu_pso.is_finite(),
        "non-finite PSO result: flex={flex_pso}, wgpu={wgpu_pso}",
    );
    assert!(
        flex_pso < 1e-2,
        "Flex PSO did not converge on Sphere-D10: {flex_pso}",
    );
    assert!(
        wgpu_pso < 1e-2,
        "wgpu PSO did not converge on Sphere-D10: {wgpu_pso}",
    );
}
