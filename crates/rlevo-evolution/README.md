# rlevo-evolution

Tensor-native classical evolutionary algorithms for `rlevo`, built on
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

## Quick start

```rust,no_run
use burn::backend::NdArray;
use rlevo_benchmarks::agent::FitnessEvaluable;
use rlevo_benchmarks::env::BenchEnv;
use rlevo_evolution::algorithms::ga::{GaConfig, GeneticAlgorithm};
use rlevo_evolution::fitness::FromFitnessEvaluable;
use rlevo_evolution::strategy::EvolutionaryHarness;

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

The quick start uses `rlevo_benchmarks` (a sibling workspace crate) to supply the
`FitnessEvaluable` trait and `BenchEnv` driver; add it to your `[dev-dependencies]`
or swap in your own fitness function.

Run the showcase across every shipping family:

```bash
cargo run --release -p rlevo-evolution --example sphere_showcase
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
into `rlevo_benchmarks::env::BenchEnv`, so the benchmark evaluator
drives it identically to an RL environment — one generation per
`step`, reward = `-best_fitness_ever`.

## Reproducibility

Same `base_seed` → bit-identical trajectory on the `ndarray` backend
(enforced by `tests/determinism.rs`). The `wgpu` backend is
non-deterministic across runs; both backends converge to similar optima.

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

## References

- H.-G. Beyer and H.-P. Schwefel, "Evolution strategies – A comprehensive introduction," Natural Computing, vol. 1, pp. 3–52, Mar. 2002, doi: 10.1023/A:1015059928466. [Springer Nature Link](https://link.springer.com/article/10.1023/A:1015059928466)
- R. Storn and K. Price, "Differential evolution – A simple and efficient heuristic for global optimization over continuous spaces," Journal of Global Optimization, vol. 11, pp. 341–359, Dec. 1997, doi: 10.1023/A:1008202821328. [springer](https://link.springer.com/article/10.1023/a:1008202821328)
- D. B. Fogel, "An introduction to simulated evolutionary optimization," IEEE Trans. Neural Networks, vol. 5, no. 1, pp. 3–14, Jan. 1994, doi: 10.1109/72.265956. [IEEE](https://ieeexplore.ieee.org/abstract/document/265956)
- D. E. Goldberg, Genetic Algorithms in Search, Optimization and Machine Learning. Reading, MA: Addison-Wesley, 1989. [ACM](https://dl.acm.org/doi/10.5555/534133)
- K. Deb and R. B. Agrawal, "Simulated binary crossover for continuous search space," Complex Syst., vol. 9, no. 2, pp. 115–148, 1995.
- J. F. Miller, Ed., Cartesian Genetic Programming. Berlin, Heidelberg: Springer, 2011, doi: 10.1007/978-3-642-17310-3. [Springer](https://link.springer.com/book/10.1007/978-3-642-17310-3)
- J. Kennedy and R. Eberhart, "Particle swarm optimization," in Proceedings of the 1995 International Conference on Neural Networks, Perth, Australia, November 1995, pp. 1942–1948. [IEEE](https://ieeexplore.ieee.org/document/488968)
- Y. Shi and R. Eberhart, "A modified particle swarm optimizer," in Proc. IEEE Int. Conf. Evol. Comput., Anchorage, AK, USA, May 1998, pp. 69–73. [Semantic Scholar](https://www.semanticscholar.org/paper/A-modified-particle-swarm-optimizer-Shi-Eberhart/506172b0e0dd4269bdcfe96dda9ea9d8602bbfb6)
- M. Clerc and J. Kennedy, "The particle swarm - explosion, stability, and convergence in a multidimensional complex space," in IEEE Trans. Evol. Comput., vol. 6, no. 1, pp. 58–73, Feb. 2002, doi: 10.1109/4235.985692.[ACM](https://dl.acm.org/doi/abs/10.1109/4235.985692)
- K. Socha and M. Dorigo, "Ant colony optimization for continuous domains," Eur. J. Oper. Res., vol. 185, no. 3, pp. 1155–1173, Mar. 2008, doi: 10.1016/j.ejor.2006.06.046.
- D. Karaboga, "An idea based on honey bee swarm for numerical optimization," Erciyes University, Technical Report TR06, Oct. 2005.
- S. Mirjalili, S. Mirjalili, and A. Lewis, "Grey wolf optimizer," Advances in Engineering Software, vol. 69, pp. 46–61, 2014.
- C. L. Camacho Villalón, T. Stützle, and M. Dorigo, "Grey wolf, firefly and bat algorithms: Three widespread algorithms that do not contain any novelty," in Swarm Intelligence. ANTS 2020, Lecture Notes in Computer Science, vol. 12421. Springer, Cham, 2020, pp. 121–133, doi: 10.1007/978-3-030-60376-2_10. [Springer](https://link.springer.com/chapter/10.1007/978-3-030-60376-2_10)
- S. Mirjalili and A. Lewis, "The whale optimization algorithm," Advances in Engineering Software, vol. 95, pp. 51–67, 2016.
- X.-S. Yang and S. Deb, "Cuckoo search via Lévy flights," in Proc. World Congr. Nature Biologically Inspired Comput. (NaBIC 2009), Dec. 2009, pp. 210–214. [arXiv:1003.1594](https://arxiv.org/abs/1003.1594)
- R. N. Mantegna, "Fast, accurate algorithm for numerical simulation of Lévy stable stochastic processes," Physical Review E, vol. 49, no. 5, pp. 4677–4683, May 1994, doi: 10.1103/PhysRevE.49.4677.
- X. S. Yang, "Nature-inspired metaheuristic algorithms," Luniver Press, 2008.
- [#] X.-S. Yang, "A new metaheuristic bat-inspired algorithm," in Nature Inspired Cooperative Strategies for Optimization (NICSO 2010), J. R. González, D. A. Pelta, C. Cruz, G. Terrazas, and N. Krasnogor, Eds. Berlin, Heidelberg: Springer, 2010, vol. 284, pp. 65–74. doi: 10.1007/978-3-642-12538-6_6. [Springer](https://link.springer.com/chapter/10.1007/978-3-642-12538-6_6) [arXiv:1004.4170](https://arxiv.org/abs/1004.4170)
- S. Mirjalili, A. H. Gandomi, S. Z. Mirjalili, S. Saremi, H. Faris, and S. M. Mirjalili, "Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems," Advances in Engineering Software, vol. 114, pp. 163-191, Dec. 2017.
- K. Sörensen, "Metaheuristics—the metaphor exposed," Int. Trans. Oper. Res., vol. 22, no. 1, pp. 3–18, Jan. 2015, doi: 10.1111/itor.12001.
- A. P. Piotrowski, J. J. Napiorkowski, and P. M. Rowinski, "How novel is the 'novel' black hole optimization approach?" Information Sci., vol. 267, pp. 191–200, 2014.


## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
