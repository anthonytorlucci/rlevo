# evorl-evolution

Tensor-native classical evolutionary algorithms for `burn-evorl`, built on
the [Burn](https://burn.dev/) framework.

## Status

**Alpha.** v1 of the `classical-evolutionary-algorithms` spec. The
trait surface and five algorithm families below are shipping; custom
CubeCL kernels on hot paths are preserved as design docs
(`src/ops/kernels/mod.rs`) for a follow-up.

## Algorithm families

| Family | Entry point | Genome | Convergence on Sphere-D10 (ndarray) |
|---|---|---|---|
| Genetic Algorithm (real) | `algorithms::ga::GeneticAlgorithm` | `Tensor<B, 2>` | ~1e-1 (fixed σ) |
| Genetic Algorithm (binary) | `algorithms::ga_binary::BinaryGeneticAlgorithm` | `Tensor<B, 2, Int>` | OneMax: exact |
| Evolution Strategy | `algorithms::es_classical::EvolutionStrategy` | `Tensor<B, 2>` | < 1e-30 (μ+λ) |
| Evolutionary Programming | `algorithms::ep::EvolutionaryProgramming` | `Tensor<B, 2>` | ~3e-17 |
| Differential Evolution | `algorithms::de::DifferentialEvolution` | `Tensor<B, 2>` | < 1e-22 (rand/1/bin) |
| Cartesian Genetic Programming | `algorithms::gp_cgp::CartesianGeneticProgramming` | `Tensor<B, 2, Int>` | see symbolic regression test |

See `examples/sphere_showcase.rs` for full numbers.

## Quick start

```rust,no_run
use burn::backend::NdArray;
use evorl_benchmarks::agent::FitnessEvaluable;
use evorl_benchmarks::env::BenchEnv;
use evorl_evolution::algorithms::ga::{GaConfig, GeneticAlgorithm};
use evorl_evolution::fitness::FromFitnessEvaluable;
use evorl_evolution::strategy::EvolutionaryHarness;

struct Sphere;
struct SphereFit;
impl FitnessEvaluable for SphereFit {
    type Individual = Vec<f64>;
    type Landscape = Sphere;
    fn evaluate(&self, x: &Self::Individual, _: &Self::Landscape) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
}

fn main() {
    let device = Default::default();
    let mut harness = EvolutionaryHarness::<NdArray, _, _>::new(
        GeneticAlgorithm::<NdArray>::new(),
        GaConfig::default_for(64, 10),
        FromFitnessEvaluable::new(SphereFit, Sphere),
        42,          // base seed
        device,
        500,         // max generations
    );
    harness.reset();
    while !harness.step(()).done {}
    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("final best = {best:.3e}");
}
```

Run the showcase across every shipping family:

```bash
cargo run --release -p evorl-evolution --example sphere_showcase
```

## Design

The central abstraction is the pure [`Strategy<B>`](src/strategy.rs)
trait: `init`, `ask`, `tell`, and `best` all take `&self` and an
explicit RNG, returning a new `State` rather than mutating. This lets
many strategy instances run concurrently without interior mutability
and keeps checkpointing trivial (just `Clone` the state).

Populations are tensors on any Burn backend — `Tensor<B, 2>` for
real-valued families, `Tensor<B, 2, Int>` for binary / integer / CGP.
The fitness function is a separate trait
([`BatchFitnessFn`](src/fitness.rs)), so users plug in any
device-resident evaluator; the
[`FromFitnessEvaluable`](src/fitness.rs) adapter lifts any
`evorl-benchmarks::FitnessEvaluable` (host-side `Vec<f64>` in,
`f64` out) onto a device tensor.

The [`EvolutionaryHarness<B, S, F>`](src/strategy.rs) wraps a strategy
into `evorl_benchmarks::env::BenchEnv`, so the benchmark evaluator
drives it identically to an RL environment — one generation per
`step`, reward = `-best_fitness_ever`.

## Reproducibility

Same `base_seed` → bit-identical generation-by-generation trajectory on
the ndarray backend (enforced by `tests/determinism.rs`). The
`wgpu` backend uses an independent RNG stream per device, so
trajectories diverge; both backends are expected to reach similar
optima, not identical intermediate points.

Burn backends seed their tensor RNG through process-global state. When
running multiple harness instances concurrently (e.g. with rayon),
pass `EvaluatorConfig::num_threads = Some(1)` to keep the `seed →
random` call pair atomic.

## Cargo features

- `custom-kernels` (default) — compile-time gate for future CubeCL
  kernels on hot paths (tournament selection, DE trial-vector
  construction, roulette). Design sketches in
  `src/ops/kernels/mod.rs`. No kernels ship in v1; the feature
  currently only affects the compile matrix.

## Caveats

- **DE/Best1Bin and DE/CurrentToBest1Bin** converge prematurely on
  unimodal landscapes — documented on `DeVariant`.
- **Classical ES `(1+1)` and `(1+λ)` use fixed σ** (no log-normal
  adaptation) and therefore converge more slowly than `(μ,λ)` / `(μ+λ)`
  which do adapt σ.
- **CGP phenotype evaluation runs on the host** (topological-sweep
  dispatch is not a good fit for dense tensor ops). Genotype storage
  stays on-device.

## Related work

- [evosax](https://github.com/RobertTLange/evosax) — JAX-based evolution
  strategies, the closest analogue in the Python ecosystem.
- [EvoJAX](https://github.com/google/evojax) — hardware-accelerated
  neuroevolution on JAX.

## License

MIT.
