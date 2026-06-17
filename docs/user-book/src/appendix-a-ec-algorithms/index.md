# Appendix A — Evolutionary Computation Algorithms

This appendix gives derivations, pseudocode, and implementation notes for the
evolutionary computation algorithms in `rlevo::evo`. It is the place to come
when the conceptual summaries in [Part I](../part-1-foundations/20-evolutionary-computation.md)
are not enough.

> Algorithm pages are being added incrementally. Entries without links are
> implemented in the codebase but do not yet have a documentation page.

## Classical families

- [Binary Encoded Genetic Algorithm](binary-encoded-genetic-algorithm.md) — bit-string genomes, uniform crossover, bit-flip mutation, elitist replacement
- [Real-Valued Genetic Algorithm](real-valued-genetic-algorithm.md) — continuous genomes, BLX-α / uniform crossover, Gaussian mutation, generational and elitist replacement
- [Evolution Strategies](evolution-strategies.md) — `(1+1)`, `(1+λ)`, `(μ,λ)`, `(μ+λ)` variants; 1/5th rule and log-normal σ adaptation
- [Differential Evolution](differential-evolution.md) — `Rand1Bin`, `Rand1Exp`, `Rand2Bin`, `Best1Bin`, `CurrentToBest1Bin` variants; greedy per-slot replacement
- [Evolutionary Programming](evolutionary-programming.md) — Fogel-style; per-individual log-normal σ adaptation, q-tournament survivor selection over `(μ + μ)` pool
- CMA-ES and CMSA-ES — covariance matrix adaptation, path-length control, step-size adaptation *(tracked in [issue #59](https://github.com/anthonytorlucci/rlevo/issues/59))*

## Estimation-of-distribution algorithms (EDAs)

All EDAs share a common `EdaStrategy` driver with a `fit → sample` loop; only the `ProbabilityModel` changes.

- UMDA (`UnivariateGaussian`) — per-dimension Gaussian; unweighted MLE, minimum-variance floor
- PBIL (`UnivariateBernoulli`) — per-bit probability vector; best/worst individual update
- cGA (`CompactGenetic`) — virtual-population probability vector; winner/loser from truncation-selected subset
- MIMIC (`DependencyChain`) — continuous Gaussian chain capturing pairwise dependencies
- BOA (`BayesianNetwork`) — BIC-scored Bayesian network DAG over binary genes; ancestral sampling

## Symbolic and program evolution

- Cartesian Genetic Programming (`gp_cgp`) — fixed-grid CGP; function-set configurable
- Gene Expression Programming (`gep`) — linear head/tail chromosome decoded to expression tree; symbolic regression

## Hybrid and composite strategies

- Memetic wrapper (`memetic`) — wraps any real-valued strategy with per-individual local-search refinement; Lamarckian, Baldwinian, and partial writeback policies
- Neuroevolution: weight-only (`WeightOnly`) — evolves the flattened weights of any Burn `Module` using any real-valued strategy
- Neuroevolution: architecture NAS (`ArchNasStrategy`) — co-evolves which fixed-topology module variant and its weights
- Neuroevolution: NEAT (`NeatStrategy`) — grows network topology and weights from a minimal seed via speciation and innovation-aligned crossover

## Swarm and nature-inspired metaheuristics

> The `metaheuristic` module docs note that GWO, WOA, Bat, and SSA are
> "legacy comparators" per Camacho-Villalón et al. (2023) and Sörensen (2015).
> Start with PSO for most continuous problems.

- Particle Swarm Optimization (PSO)
- Ant Colony Optimization — continuous (`ACO_R`)
- Ant Colony Optimization — permutation (`aco_perm`) *(stub — deferred to a future release)*
- [Artificial Bee Colony (ABC)](artificial-bee-colony.md) — employed/onlooker/scout phases; single-coordinate difference perturbation
- Cuckoo Search — Lévy flights and random walk
- Firefly Algorithm
- Grey Wolf Optimizer (GWO) — legacy comparator
- Whale Optimization Algorithm (WOA) — legacy comparator
- Bat Algorithm — legacy comparator
- Salp Swarm Algorithm (SSA) — legacy comparator

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
