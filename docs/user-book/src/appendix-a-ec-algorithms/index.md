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
- [CMA-ES and CMSA-ES](cma-es.md) — full covariance matrix adaptation; CSA path-length control and rank-1/rank-μ updates (CMA-ES) vs. per-individual log-normal σ and rank-μ ML blend (CMSA-ES)

## Estimation-of-distribution algorithms (EDAs)

All EDAs share a common `EdaStrategy` driver with a `fit → sample` loop; only the `ProbabilityModel` changes — see [Estimation-of-Distribution Algorithms](estimation-of-distribution.md) for the driver and all five models.

- [UMDA (`UnivariateGaussian`)](estimation-of-distribution.md#umda--univariategaussian) — per-dimension Gaussian; unweighted MLE, minimum-variance floor
- [PBIL (`UnivariateBernoulli`)](estimation-of-distribution.md#pbil--univariatebernoulli) — per-bit probability vector; best/worst individual update
- [cGA (`CompactGenetic`)](estimation-of-distribution.md#cga--compactgenetic) — virtual-population probability vector; winner/loser from truncation-selected subset
- [MIMIC (`DependencyChain`)](estimation-of-distribution.md#mimic--dependencychain) — continuous Gaussian chain capturing pairwise dependencies
- [BOA (`BayesianNetwork`)](estimation-of-distribution.md#boa--bayesiannetwork) — BIC-scored Bayesian network DAG over binary genes; ancestral sampling

## Symbolic and program evolution

- [Cartesian Genetic Programming](cartesian-genetic-programming.md) (`gp_cgp`) — fixed-grid integer genome decoded to a DAG; `(1+λ)` engine, point mutation, neutral drift
- [Gene Expression Programming](gene-expression-programming.md) (`gep`) — fixed-length head/tail chromosome decoded to an expression tree; repair-free operators with crossover; symbolic regression

## Hybrid and composite strategies

- Memetic wrapper (`memetic`) — wraps any real-valued strategy with per-individual local-search refinement; Lamarckian, Baldwinian, and partial writeback policies
- Neuroevolution: weight-only (`WeightOnly`) — evolves the flattened weights of any Burn `Module` using any real-valued strategy
- Neuroevolution: architecture NAS (`ArchNasStrategy`) — co-evolves which fixed-topology module variant and its weights
- Neuroevolution: NEAT (`NeatStrategy`) — grows network topology and weights from a minimal seed via speciation and innovation-aligned crossover

## Swarm and nature-inspired metaheuristics

> The `metaheuristic` module classifies Firefly, GWO, WOA, and Bat as
> "legacy comparators": Camacho-Villalón et al. (2020, 2023) analyse each
> component-by-component and show it reduces to PSO-style mechanisms under a new
> metaphor, echoing Sörensen's (2015) critique. Salp Swarm carries a *separate*
> caveat — Castelli et al. (2022) show its leader update is shift-variant rather
> than PSO-equivalent. Start with PSO for most continuous problems.

- [Particle Swarm Optimization (PSO)](particle-swarm-optimization.md) — inertia and constriction variants; cognitive/social velocity update
- [Ant Colony Optimization — continuous (`ACO_R`)](ant-colony-continuous.md) — solution archive as pheromone; rank-weighted Gaussian kernels
- Ant Colony Optimization — permutation (`aco_perm`) *(stub — deferred to a future release)*
- [Artificial Bee Colony (ABC)](artificial-bee-colony.md) — employed/onlooker/scout phases; single-coordinate difference perturbation
- [Cuckoo Search](cuckoo-search.md) (`cuckoo`) — Mantegna Lévy flights; greedy per-nest acceptance and worst-nest abandonment
- [Firefly Algorithm](firefly-algorithm.md) (`firefly`) — multi-attractor swarm; each firefly drawn to every brighter one with `O(N²)` distance-decayed attraction; legacy comparator (capped at 128 fireflies)
- [Grey Wolf Optimizer (GWO)](grey-wolf-optimizer.md) (`gwo`) — α/β/δ leaders; equal-weight three-attractor update with a linearly annealed step coefficient; legacy comparator
- [Whale Optimization Algorithm (WOA)](whale-optimization-algorithm.md) (`woa`) — per-whale choice of shrink-encircle, random search, or logarithmic-spiral move toward the best; legacy comparator with an origin bias
- Bat Algorithm — legacy comparator
- Salp Swarm Algorithm (SSA) — legacy comparator

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
