# Evolutionary Computation

Evolutionary computation (EC) is a family of population-based optimization
methods inspired by Darwinian natural selection. The common skeleton is:

1. Maintain a **population** of candidate solutions.
2. **Evaluate** each candidate with the fitness function.
3. **Select** candidates that performed better.
4. **Vary** them (mutation, recombination) to produce the next population.
5. Repeat until a budget is exhausted or a solution is good enough.

The key insight is that a *population* of candidates shares information across
the search space simultaneously, while variation operators inject diversity that
prevents premature convergence. Selection then amplifies what works.

**In `rlevo`.** Every algorithm in `rlevo::evo` implements the `Strategy<B>`
trait, which maps the five-step skeleton onto four methods over three associated
types (`Params`, the static run config; `State`, carried generation to
generation; `Genome`, the population container):

- `init` — sample the first population and build the initial state;
- `ask` — propose the next population to evaluate (steps 3–4);
- `tell` — consume that population with its fitness and produce the next state
  (steps 2–3);
- `best` — report the best solution found so far.

The RNG is passed explicitly so the harness owns all stochasticity — strategies
carry no internal PRNG state. The
[Strategies](evolutionary-computation/24-strategy.md) chapter walks the full
trait signature and traces a GA from `init` to convergence.

Each genome is a *row* of an on-device rank-2 population tensor, and its *kind* —
real, binary, or integer — is fixed at the type level by a zero-sized marker
implementing the `GenomeKind` trait:

```rust
pub struct Real;    // genes are f32 — used by GA, ES, EP, DE
pub struct Binary;  // genes are 0/1 i32 — used by binary GA and EDAs
pub struct Integer; // genes are bounded non-negative integers — CGP, discrete search
```

The marker is a compile-time label, not stored data: it lets operators specialize
without runtime dispatch (Gaussian mutation only compiles for `Real`; bit-flip
only for `Binary`). Strategies carry the bare population tensor as their `Genome`
associated type; the `Population<B, K>` wrapper pairs that tensor with its shape
metadata and kind tag for use at API boundaries. The [Genome
Representation](evolutionary-computation/22-genome.md) chapter is the full tour of
the representation layer — the kind markers, the `Population` invariant, and the
`ParamReshaper` bridge that turns a network's weights into an evolvable genome.

Fitness evaluation is injected through `BatchFitnessFn<B, G>`, which receives
the population and returns a `Tensor<B, 1>` of shape `(pop_size,)`. Strategies
themselves never call the objective function — the harness does — so the same
strategy implementation works against any landscape. One convention runs through
all of it: the engine treats fitness as a **cost to minimise** (lower is better).
This is an internal contract of the evolution engine rather than a property of EAs
in general — the [Fitness Evaluation](evolutionary-computation/23-fitness.md#where-its-enforced--and-where-its-your-responsibility)
chapter shows where it is enforced and where pointing the objective the right way
is your responsibility.

<!-- additional context [Simon, 2013, p 2-3]
Some authors use the term *evolutionary computation* to refer to EAs. This 
emphasizes the point tht EAs are implemented in computers. However, 
evolutionary computing could refer to algorithms that are not used for 
optimization; for example, the first genetic algorithms (GAs) were not used for 
optimization per se, but were intended to study the process of natural selection ...

Others use the term *population-based optimization* to refer to EAs. This 
emphasizes the point that EAs generally consist of a population of candidate 
solutions to some problem, and as time passes, the population evolves to a 
better solution to the problem.However, many EAs can consist of only a single 
candidate solution at each iteration (for example hill climbing and evolutionary 
strategies)...

Other authors use terms like *nature-inspired computing* or *bio-inspired 
computing* to refer to EAs. However, some EAs, like differential evolution and 
estimation of distribution algorithms, might not be motivated by nature. Other 
EAs, like evolution strategies and opposition-based learning, have a very weak 
connection with natural processes. EAs are more general than nature-inspired 
algorithms becuase EAs include non-biologically motivated algorithms.

... Heursitic algorithms are methods that use rules of thumb or common sense 
approaches to solve a problem. Heuristic algorithms usually are not expected to 
find the best answer to a problem, but are only expected to find solutions that 
are "close enough" to the best. The term *metaheuristic* is used to describe a 
family of heuristic algorithms... 
-->

## The Genetic Algorithm

The canonical GA, as implemented in `rlevo::evo`, operates on real-valued
or binary genomes with three operators:

**Selection** - picks which individuals reproduce. Common schemes:
- *Tournament*: draw \\(k\\) candidates at random; the best wins. Selective
  pressure scales with \\(k\\).
- *Fitness-proportionate* (roulette wheel): probability of selection is
  proportional to fitness. Sensitive to fitness scaling.
- *Rank-based*: selection probability based on rank, not raw fitness; more
  robust to outliers.

**Crossover** - combines two parents into offspring:
- *Single-point*: split each parent at a random locus; swap tails.
- *BLX-α*: for real-valued genomes, sample offspring genes uniformly from
  \\([\min(p_1, p_2) - \alpha \cdot d,\ \max(p_1, p_2) + \alpha \cdot d]\\)
  where \\(d = |p_1 - p_2|\\). Larger \\(\alpha\\) → more exploration.
- *Simulated Binary Crossover (SBX)*: mimics single-point on binary
  representations but works in continuous space [[Deb and Agrawal, 1995]](#bibliography).

**Mutation** - perturbs an individual:
- *Gaussian*: add \\(\mathcal{N}(0, \sigma^2)\\) noise to each gene.
- *Uniform*: replace a gene with a random draw from its bounds.

**In `rlevo`.** The operators above are a textbook menu; `rlevo::evo::ops`
implements a focused subset of them, organised by the role each plays in a
generation:
- `ops::selection` — tournament, truncation
- `ops::crossover` — BLX-α, uniform (real); uniform (binary)
- `ops::mutation`  — Gaussian (scalar and per-row), uniform-reset (real); bit-flip (binary)
- `ops::replacement` — generational, elitist, (μ + λ), (μ, λ) survivor selection

Each operator is a free function that takes a population `Tensor<B, _>` and
returns a new one, leaving the input unchanged. Kind-specialization is enforced
at the type level: real-valued operators take `Tensor<B, 2>` and binary
operators take `Tensor<B, 2, Int>`, so passing a binary genome to Gaussian
mutation is a compile error.

The [Evolutionary Operators](evolutionary-computation/21-ops.md) chapter is a
full tour of the catalogue and the conventions behind it; [Appendix
A](../appendix-a-ec-algorithms/index.md) gives the GA and ES pseudocode that
assembles these operators end-to-end.

<!-- [Simon, 2013, p. 188]
This section discusses elitism, which is a way of making sure that the best 
individuals in an EA are retained in the population from one generation to the 
next... 

... How can we retain the nice results that arise from recombination, while 
avoiding the loss of the best individual in the population?

The answer to this question is to keep the best individuals in the EA from one 
generation to the next. This idea, first proposed by [De Jong, 1975], called 
*elitism* and usually improves the performance of an EA...
-->

## Evolution Strategies

While GAs were designed with binary strings in mind, Evolution Strategies (ES)
were built from the outset for continuous domains. Their defining idea is
**self-adaptation**: the mutation distribution is not fixed but evolves alongside
the solution, so the search rescales its own step size as it homes in on an
optimum.

`rlevo` implements the **classical ES** family — four canonical variants, all
parameterised by a single `EsConfig`:

- `(1+1)` — one parent, one offspring; step size \\(\sigma\\) adapted by
  Rechenberg's **1/5th success rule**.
- `(1+λ)` — one parent, λ offspring; the best offspring replaces the parent only
  on improvement.
- `(μ,λ)` — μ parents produce λ offspring; the parents are discarded each
  generation.
- `(μ+λ)` — survivors are the μ best of the combined parent-plus-offspring pool.

The multi-parent variants adapt \\(\sigma\\) by **log-normal self-adaptation**,
and it is worth pausing on what *self-adaptation* means, because it is the idea
that distinguishes ES from a method like the \\((1+1)\\) 1/5th rule. The 1/5th
rule is an external controller: it observes the success rate and rescales
\\(\sigma\\) by an explicit formula. Self-adaptation has no such controller.
Instead, each individual stores its own step size \\(\sigma\\) *inside the
genome*, alongside the object variables \\(\mathbf{x}\\), and \\(\sigma\\) is
subject to the same selection pressure as \\(\mathbf{x}\\). Selection only ever
scores the fitness of the mutated genes — yet individuals that happen to carry a
well-scaled \\(\sigma\\) tend to produce fitter offspring, survive, and carry
that \\(\sigma\\) forward. Good step sizes are thus selected *indirectly*, as a
side effect of the offspring they generate surviving.

The mechanism is a two-step mutation. First the step size is perturbed
multiplicatively by a log-normal factor, then the *new* step size drives the
gene mutation:

```math
\sigma' = \sigma \cdot \exp(\tau \, \mathcal{N}(0, 1)),
\qquad
\mathbf{x}' = \mathbf{x} + \sigma' \, \mathcal{N}(0, \mathbf{I}).
```

The order is the crux: mutating \\(\sigma\\) *before* \\(\mathbf{x}\\) ties each
step size to the step it actually produced, so selection can judge it. Perturb
\\(\sigma\\) afterwards and the surviving \\(\sigma'\\) would never have been
tested by the move it took. The log-normal form is what keeps the scheme
well-behaved — it holds \\(\sigma\\) strictly positive and is unbiased in
log-space, so absent selection \\(\sigma\\) drifts neither up nor down. The full
rationale, the per-coordinate generalisation, and pseudocode are in
[Appendix A](../appendix-a-ec-algorithms/evolution-strategies.md#log-normal-sigma-adaptation).

### Covariance Matrix Adaptation

The classical variants adapt a single scalar \\(\sigma\\) — they can rescale the
search, but not *reshape* it. **CMA-ES** (Covariance Matrix Adaptation ES; Hansen
and Ostermeier, 2001) [[Hansen and Ostermeier, 2001]](#bibliography) and its
self-adaptive cousin **CMSA-ES** (Beyer and Sendhoff, 2008)
[[Beyer and Sendhoff, 2008]](#bibliography) go further: they maintain a full
covariance matrix \\(\mathbf{C}\\) over the search space and sample each
generation from \\(\mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})\\). The contour
ellipses of \\(\mathbf{C}\\) rotate and stretch to match the local landscape, so
a narrow diagonal valley or a rotated, ill-conditioned bowl is searched along its
*natural* axes rather than the coordinate axes. This shape-learning is what makes
CMA-ES the de facto standard for continuous black-box optimisation at
low-to-medium dimensionality. Both ship in `rlevo::evo` today.

Two ideas do the work, and CMA-ES keeps them deliberately separate:

- **Covariance adaptation** learns the *shape* of good steps. Successful
  offspring directions are accumulated into \\(\mathbf{C}\\) — a rank-1 term from
  an *evolution path* plus a rank-μ term from the current generation — so the
  distribution elongates along directions that have been paying off.
- **Step-size control** learns the *scale* independently. A separate conjugate
  path tracks whether consecutive steps reinforce or cancel: aligned steps grow
  \\(\sigma\\) (speed up), cancelling steps shrink it (refine). Decoupling scale
  from shape lets the search take long strides early and fine steps near the
  optimum.

**CMSA-ES** keeps the covariance idea but drops the evolution paths, replacing
step-size control with the same per-individual log-normal σ-self-adaptation the
classical multi-parent variants use — a simpler, path-free alternative at a small
cost in convergence speed (the two strategies share the self-adaptation
*mechanism* but use different learning-rate constants). The full update
equations, default parameters, and a when-to-use table are in
[Appendix A: CMA-ES and CMSA-ES](../appendix-a-ec-algorithms/cma-es.md).

<!-- [Simon, 2013, p 135] 
... The goal of CMA-ES, ..., is to fit (as well as possible) the distribution 
of the ES mutations to the contour of the objective funciton... The drawbacks 
of CMA-ES are its complicated adaptation strategy, and its complicated tuning 
parameter settings...
-->

## Estimation of Distribution Algorithms

Instead of maintaining individuals and applying variation operators, **Estimation
of Distribution Algorithms** (EDAs) maintain an explicit probabilistic model of
the search space and sample new candidates from it [[Larrañaga and Lozano, 2002]](#bibliography).

The core loop:

1. Sample a population from the model.
2. Evaluate and select the top individuals.
3. **Fit** the model to the selected individuals.
4. Go to 1.

Simple EDAs (UMDA, PBIL, cGA) assume gene independence — each gene's marginal
distribution is updated separately. More powerful EDAs (MIMIC, BOA) capture
pairwise or higher-order dependencies between genes, which is critical for
*deceptive* problems where the optimal solution requires specific combinations of
genes that no marginal model would find.

`rlevo` implements UMDA, PBIL, cGA, MIMIC, and BOA. Their derivations and the
deceptive benchmark (Concatenated Trap) used to compare them are in
[Appendix A](../appendix-a-ec-algorithms/index.md).

**In `rlevo`.** The fit → sample loop is captured by the `ProbabilityModel`
trait, which is separate from `Strategy`. `EdaStrategy<B, M>` is a generic
driver that implements `Strategy<B>` for any `M: ProbabilityModel<B>`:

```rust
pub trait ProbabilityModel<B: Backend> {
    type Params;
    type State: Clone + Debug + Send + Sync;

    /// Fit the model to the selected (top-μ) population.
    /// `prev = None` on the first generation; the model builds its prior from `params`.
    fn fit(&self, params: &Self::Params, population: &Tensor<B, 2>,
           fitness: Tensor<B, 1>, prev: Option<&Self::State>) -> Self::State;

    /// Sample n new candidates from the fitted model using the host RNG.
    fn sample(&self, params: &Self::Params, state: &Self::State,
              n: usize, rng: &mut dyn Rng, device: &B::Device) -> Tensor<B, 2>;
}
```

All randomness in `sample` comes from the host `rng`; implementations must
never call `Tensor::random` or `B::seed` (Burn's GPU PRNG kernels share
process-global state and would interleave across parallel strategy instances).
Swapping one EDA for another is a one-line type change: `EdaStrategy<B, UnivariateGaussian>`
→ `EdaStrategy<B, BayesianNetwork>`.

<!-- see [Simon, 2013] 
- Estimation of Distribution Algorithms (p. 313)
- UDMA (p. 316)
- cGA (p. 318)
- PBIL (p. 320)
- MIMIC (p. 324)
-->

### Where CMA-ES sits: the ES ↔ EDA spectrum

CMA-ES blurs the line between the two families above. A full-covariance Gaussian
EDA fits its distribution by **maximum likelihood** on the selected elite — mean
and covariance are *overwritten* each generation from the current crop. Strip
CMA-ES of its evolution paths and force its learning rates to 1 and it collapses
to exactly that: a continuous EDA (EMNA). What keeps CMA-ES on the *ES* side of
the line is **memory** — it does not overwrite \\(\mathbf{C}\\) and \\(\sigma\\)
but blends each generation's estimate into accumulated evolution paths, a fading
momentum that records which directions have been paying off.

So the spectrum runs from memoryless density estimation (EDA) to path-driven,
momentum-carrying adaptation (CMA-ES), with **CMSA-ES** sitting between them: it
learns covariance like CMA-ES but adapts its step size by self-adaptation rather
than path-tracking. `rlevo` keeps the families architecturally distinct for this
reason — EDAs implement the `fit → sample` `ProbabilityModel` trait, while
CMA-ES and CMSA-ES are self-contained strategies whose path machinery does not
fit that mould (ADR 0021).

## Other strategy families in `rlevo`

The GA, ES, and EDA sections above cover three distinct *ideas* — recombination,
self-adaptation, and explicit distribution learning — but the same `Strategy`
contract backs a wider menagerie. Each ships today and is documented with full
pseudocode in [Appendix A](../appendix-a-ec-algorithms/index.md); in brief:

- **Differential Evolution (DE)** mutates by adding *scaled difference vectors*
  between population members — a self-scaling scheme that needs no externally
  tuned step size. `rlevo` ships the `Rand1Bin`, `Rand1Exp`, `Rand2Bin`,
  `Best1Bin`, and `CurrentToBest1Bin` variants with greedy per-slot replacement.
- **Evolutionary Programming (EP)** is Fogel-style, mutation-only evolution with
  per-individual log-normal \\(\sigma\\) adaptation and q-tournament survivor
  selection over a \\((\mu + \mu)\\) pool.
- **Genetic programming** evolves programs rather than fixed-length vectors:
  *Cartesian GP* (`gp_cgp`, an integer genome decoded to a computation graph) and
  *Gene Expression Programming* (`gep`, a linear chromosome decoded to an
  expression tree).
- **Neuroevolution** evolves networks directly: `WeightOnly` evolves the
  flattened weights of any Burn `Module` through the `ParamReshaper` bridge from
  the [genome chapter](evolutionary-computation/22-genome.md); `ArchNas`
  co-evolves which fixed-topology variant *and* its weights; `Neat` grows
  topology from a minimal seed via speciation and innovation-aligned crossover.
- **The memetic wrapper** wraps any real-valued strategy with per-individual
  local search (hill-climbing, Nelder–Mead, simulated annealing) under
  Lamarckian, Baldwinian, or partial-writeback policies.
- **Swarm metaheuristics** include PSO, ABC, ACO (continuous `ACO_R`; a
  permutation variant is stubbed), cuckoo search, and firefly. Four more — GWO,
  WOA, Bat, and SSA — ship as *legacy comparators*: included for baselining, but
  the module docs steer you to PSO or CMA-ES (or LSHADE once it lands) for real
  work, following [[Camacho-Villalón et al., 2023]](#bibliography) and
  [[Sörensen, 2015]](#bibliography).

## Multi-Objective Optimisation

Most real problems have more than one objective — faster *and* cheaper, higher
reward *and* lower energy. When objectives conflict there is no single optimum
but a *set* of incomparable trade-offs: a solution **dominates** another when it
is no worse on every objective and strictly better on at least one. The
**Pareto front** is the set of non-dominated solutions — those for which
improving one objective necessarily worsens another. The notion of optimality it
formalises traces to the economist Vilfredo Pareto [[Pareto, 1896]](#bibliography);
an evolutionary algorithm is a natural fit for it, because a population can
approximate the *whole* front in one run rather than scalarising the objectives
into a single weighted sum and solving repeatedly.

NSGA-II (Deb et al., 2002) [[Deb et al., 2002]](#bibliography) is the canonical
multi-objective EA — its fast non-dominated sorting and crowding-distance
diversity operator remain the baseline every new algorithm is compared against;
SPEA2 [[Zitzler et al., 2001]](#bibliography) is the other classic reference
point. `rlevo` does not yet implement multi-objective optimisation; it is on
the research roadmap (see [Part III](../part-3-open-problems/02-research-directions.md)).

> **Further reading.** Deb, *Multi-Objective Optimization Using Evolutionary
> Algorithms* (Wiley, 2001) [[Deb, 2001]](#bibliography) is the standard
> book-length treatment from the EA side; Coello Coello, Lamont and Van
> Veldhuizen, *Evolutionary Algorithms for Solving Multi-Objective Problems*
> (2nd ed., Springer, 2007) [[Coello Coello et al., 2007]](#bibliography) is the
> broadest algorithm survey. For the classical (non-evolutionary) optimisation
> theory underpinning Pareto optimality and scalarisation, see Miettinen,
> *Nonlinear Multiobjective Optimization* (Kluwer, 1999)
> [[Miettinen, 1999]](#bibliography).

> **Deeper reading.** Eiben and Smith, *Introduction to Evolutionary Computing*
> (Springer, 2015) is the most accessible modern textbook. Back, *Evolutionary
> Algorithms in Theory and Practice* (Oxford, 1996) is more rigorous. The CMA-ES
> tutorial by Hansen (2023, arXiv:1604.00772) is freely available and
> authoritative. For EDAs, see Larrañaga and Lozano (eds.), *Estimation of
> Distribution Algorithms* (Kluwer, 2002).

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
