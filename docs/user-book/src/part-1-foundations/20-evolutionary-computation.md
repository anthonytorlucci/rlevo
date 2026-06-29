# Evolutionary Computation

The previous chapter ended on a question: *do you even have a gradient?* When the
answer is no — the objective is a game score, a simulation outcome, a discrete
structure, or just too rugged to trust a local slope — you need a search that
asks only that you can *score* a candidate, never that you can differentiate the
scoring. That is the bet evolutionary computation makes, and it is the
evolutionary half of `rlevo`'s thesis.

**Evolutionary computation** (EC) is a family of population-based, derivative-free
optimisation methods inspired by Darwinian natural selection. Strip away the
biology and every one of them runs the same five-step loop:

1. Maintain a **population** of candidate solutions.
2. **Evaluate** each candidate with the fitness function.
3. **Select** the candidates that performed better.
4. **Vary** them — mutate, recombine — to produce the next population.
5. Repeat until the budget runs out or a solution is good enough.

The insight that makes this work is that a *population* probes the search space
in many places at once, sharing information across all of them, while variation
operators keep injecting the diversity that stops the search collapsing onto the
first decent valley it finds. Selection then amplifies whatever is paying off.

Here is the single thread to hold onto for the rest of this chapter. Every family
below runs that exact loop; they differ in just one decision — **where the next
generation of guesses comes from.** A genetic algorithm recombines building
blocks. Evolution strategies sample from a Gaussian they reshape as they go. EDAs
fit an explicit probability model and sample it. Differential evolution reads the
geometry straight off the population's own spread. Once you see each algorithm as
an answer to that one question, the menagerie stops being a list to memorise and
becomes a small set of design choices you can reason about.

**In `rlevo`.** Because the loop is shared, we factored it into one contract.
Every algorithm in `rlevo::evo` implements the
[`Strategy<B>`](https://docs.rs/rlevo-core) trait, which maps the five steps onto
four methods over three associated types (`Params`, the static run config;
`State`, carried generation to generation; `Genome`, the population container):

- `init` — sample the first population and build the initial state;
- `ask` — propose the next population to evaluate (steps 3–4);
- `tell` — consume that population with its fitness and produce the next state
  (steps 2–3);
- `best` — report the best solution found so far.

We pass the RNG in explicitly so the *harness* owns all stochasticity — a
strategy carries no internal PRNG state, which is what lets many of them run
side by side in tests without their random streams colliding. The
[Strategies](evolutionary-computation/24-strategy.md) chapter walks the full
trait signature and traces a GA from `init` to convergence.

What does a single candidate look like? Each genome is a *row* of an on-device
rank-2 population tensor, and its *kind* — real, binary, or integer — is fixed at
the type level by a zero-sized marker implementing the `GenomeKind` trait:

```rust
pub struct Real;    // genes are f32 — used by GA, ES, EP, DE
pub struct Binary;  // genes are 0/1 i32 — used by binary GA and EDAs
pub struct Integer; // genes are bounded non-negative integers — CGP, discrete search
```

The marker is a compile-time label, not stored data: it lets operators
specialise without runtime dispatch (Gaussian mutation only compiles for `Real`;
bit-flip only for `Binary`), and it turns a whole class of "wrong genome handed
to the wrong operator" mistakes into compile errors rather than silent garbage.
Strategies carry the bare population tensor as their `Genome` associated type; the
`Population<B, K>` wrapper pairs that tensor with its shape metadata and kind tag
for use at API boundaries. The [Genome
Representation](evolutionary-computation/22-genome.md) chapter is the full tour of
the representation layer — the kind markers, the `Population` invariant, and the
`ParamReshaper` bridge that turns a network's weights into an evolvable genome.

The last piece of the contract is how a strategy learns how good its guesses
were. Fitness comes in through `BatchFitnessFn<B, G>`, which receives the whole
population and returns a `Tensor<B, 1>` of shape `(pop_size,)` — one score per
row. A strategy never calls the objective itself; the harness does, then hands
back the scores. That separation is what lets the *same* strategy run against any
landscape you can score. One convention runs through all of it: the engine
**maximises** a canonical fitness (higher is better), and a cost objective
declares its direction with `ObjectiveSense::Minimize` rather than asking you to
negate by hand. This is an internal contract of the evolution engine, not a law
of EAs in general — the [Fitness
Evaluation](evolutionary-computation/23-fitness.md#the-engine-maximises--and-you-declare-your-objectives-sense)
chapter shows how the harness reconciles a cost at one chokepoint, so a forgotten
sign can never invert a whole run.

With the contract fixed, every family below is just a different `ask`/`tell`. We
take them roughly in order of *where they get their search geometry* — from
crude recombination, through learned Gaussians, to explicit models and the
population's own spread.

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

Start with the family that gives the field its name. The **genetic algorithm**
(GA) is the most literal reading of the metaphor: where does the next guess come
from? From *recombining* two parents that already scored well, in the hope that
each carried a useful fragment — a *building block* — and that the child inherits
both. As implemented in `rlevo::evo`, the canonical GA operates on real-valued or
binary genomes through three operators, and it is worth seeing the textbook menu
before we say which slice `rlevo` actually ships.

**Selection** decides *who* reproduces:
- *Tournament*: draw \\(k\\) candidates at random; the best wins. Selective
  pressure scales with \\(k\\) — turn \\(k\\) up to exploit harder, down to
  preserve diversity.
- *Fitness-proportionate* (roulette wheel): probability of selection is
  proportional to fitness. Simple, but sensitive to how fitness is scaled.
- *Rank-based*: select on rank rather than raw fitness; more robust to outliers.

**Crossover** decides how two parents *combine*:
- *Single-point*: split each parent at a random locus; swap tails.
- *BLX-α*: for real genomes, sample each offspring gene uniformly from
  \\([\min(p_1, p_2) - \alpha \cdot d,\ \max(p_1, p_2) + \alpha \cdot d]\\) with
  \\(d = |p_1 - p_2|\\). Larger \\(\alpha\\) reaches further outside the parents
  — more exploration.
- *Simulated Binary Crossover (SBX)*: mimics single-point crossover but in
  continuous space [[Deb and Agrawal, 1995]](#bibliography).

**Mutation** decides how a single individual is *perturbed*:
- *Gaussian*: add \\(\mathcal{N}(0, \sigma^2)\\) noise to each gene.
- *Uniform*: replace a gene with a fresh draw from its bounds.

**In `rlevo`.** That menu is the literature's; we ship a focused subset of it,
filed by the role each operator plays in a single generation:
- `ops::selection` — tournament, truncation
- `ops::crossover` — BLX-α, uniform (real); uniform (binary)
- `ops::mutation`  — Gaussian (scalar and per-row), uniform-reset (real); bit-flip (binary)
- `ops::replacement` — generational, elitist, (μ + λ), (μ, λ) survivor selection

Each operator is a free function that takes a population `Tensor<B, _>` and
returns a new one, leaving its input untouched — so a generation reads as a
pipeline of pure transforms rather than in-place mutation you have to track in
your head. Kind-specialisation is enforced at the type level: real operators take
`Tensor<B, 2>` and binary operators take `Tensor<B, 2, Int>`, so handing a binary
genome to Gaussian mutation does not compile. The [Evolutionary
Operators](evolutionary-computation/21-ops.md) chapter tours the full catalogue
and the conventions behind it; [Appendix
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

The GA gets its geometry from *parents*. **Evolution Strategies** (ES) get theirs
from a *distribution* — and, unlike the GA, they were built for continuous
domains from the outset. The next guess is a sample from a Gaussian centred on
what is working; the defining idea, **self-adaptation**, is that the *width* of
that Gaussian is not a fixed hyperparameter you tune once but a quantity the
search evolves alongside the solution, rescaling its own step as it homes in.

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
and it is worth slowing down here, because self-adaptation is the idea that sets
ES apart from a scheme like the \\((1+1)\\) 1/5th rule. The 1/5th rule is an
*external controller*: it watches the success rate and rescales \\(\sigma\\) by an
explicit formula. Self-adaptation has no controller at all. Instead, each
individual carries its own step size \\(\sigma\\) *inside the genome*, right
beside the object variables \\(\mathbf{x}\\), and \\(\sigma\\) is subject to the
very same selection pressure as \\(\mathbf{x}\\). Selection only ever scores the
fitness of the mutated genes — yet individuals that happen to carry a well-scaled
\\(\sigma\\) tend to produce fitter offspring, survive, and pass that \\(\sigma\\)
on. Good step sizes are thus selected *indirectly*, as a side effect of the
offspring they produce surviving. That indirectness is the whole trick: nobody
tells the search what step size to use; the right one simply outbreeds the wrong
ones.

Mechanically it is a two-step mutation. First the step size is perturbed
multiplicatively by a log-normal factor; then the *new* step size drives the gene
mutation:

```math
\sigma' = \sigma \cdot \exp(\tau \, \mathcal{N}(0, 1)),
\qquad
\mathbf{x}' = \mathbf{x} + \sigma' \, \mathcal{N}(0, \mathbf{I}).
```

The order is the crux. Mutating \\(\sigma\\) *before* \\(\mathbf{x}\\) ties each
step size to the step it actually produced, so selection can judge it on its
results. Perturb \\(\sigma\\) afterwards and the surviving \\(\sigma'\\) would
never have been tested by the move it took — you would be selecting a step size
on the strength of a step it never made. The log-normal form keeps the scheme
honest: it holds \\(\sigma\\) strictly positive and is unbiased in log-space, so
absent selection \\(\sigma\\) drifts neither up nor down. The full rationale, the
per-coordinate generalisation, and pseudocode are in [Appendix
A](../appendix-a-ec-algorithms/evolution-strategies.md#log-normal-sigma-adaptation).

### Covariance Matrix Adaptation

A single scalar \\(\sigma\\) lets the classical variants *rescale* the search but
not *reshape* it — every direction is stretched the same amount. That is fine for
a round bowl and poor for a long diagonal valley, where the productive direction
is not aligned with any coordinate axis. **CMA-ES** (Covariance Matrix Adaptation
ES; Hansen and Ostermeier, 2001) [[Hansen and Ostermeier, 2001]](#bibliography)
and its self-adaptive cousin **CMSA-ES** (Beyer and Sendhoff, 2008)
[[Beyer and Sendhoff, 2008]](#bibliography) lift exactly that limitation: they
maintain a full covariance matrix \\(\mathbf{C}\\) and sample each generation from
\\(\mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})\\). The contour ellipses of
\\(\mathbf{C}\\) rotate and stretch to match the local landscape, so a narrow
diagonal valley or a rotated, ill-conditioned bowl gets searched along its
*natural* axes rather than the coordinate ones. This shape-learning is what makes
CMA-ES the de facto standard for continuous black-box optimisation at
low-to-medium dimensionality. Both ship in `rlevo::evo` today.

Two ideas do the work, and CMA-ES keeps them deliberately separate — a
separation worth understanding, because it is *why* CMA-ES is so robust:

- **Covariance adaptation** learns the *shape* of good steps. Successful
  offspring directions accumulate into \\(\mathbf{C}\\) — a rank-1 term from an
  *evolution path* plus a rank-μ term from the current generation — so the
  distribution elongates along directions that keep paying off.
- **Step-size control** learns the *scale* independently. A separate conjugate
  path tracks whether consecutive steps reinforce or cancel: aligned steps grow
  \\(\sigma\\) (speed up), cancelling steps shrink it (refine). Decoupling scale
  from shape lets the search stride boldly early and take fine steps near the
  optimum, instead of being forced to compromise on one global rate.

**CMSA-ES** keeps the covariance idea but drops the evolution paths, replacing
step-size control with the same per-individual log-normal σ-self-adaptation the
classical multi-parent variants use — a simpler, path-free alternative at a small
cost in convergence speed (the two share the self-adaptation *mechanism* but use
different learning-rate constants). The full update equations, default parameters,
and a when-to-use table are in [Appendix A: CMA-ES and
CMSA-ES](../appendix-a-ec-algorithms/cma-es.md).

<!-- [Simon, 2013, p 135] 
... The goal of CMA-ES, ..., is to fit (as well as possible) the distribution 
of the ES mutations to the contour of the objective funciton... The drawbacks 
of CMA-ES are its complicated adaptation strategy, and its complicated tuning 
parameter settings...
-->

## Estimation of Distribution Algorithms

ES reshapes a Gaussian implicitly, through which offspring survive. **Estimation
of Distribution Algorithms** (EDAs) take the obvious next step: build the
probability model *explicitly*, then sample the next generation straight from it
[[Larrañaga and Lozano, 2002]](#bibliography). There are no variation operators
at all — no crossover, no mutation — only a model that is refit each generation
to the candidates that survived.

The loop is just four lines:

1. Sample a population from the model.
2. Evaluate, and select the top individuals.
3. **Fit** the model to those selected individuals.
4. Go to 1.

What separates a weak EDA from a strong one is *what the model can represent*.
Simple EDAs (UMDA, PBIL, cGA) assume the genes are independent — each gene's
marginal distribution is updated on its own. That is fast and often enough, but it
is blind to *interactions*. More powerful EDAs (MIMIC, BOA) capture pairwise or
higher-order dependencies between genes, which is decisive on *deceptive* problems
where the optimum demands a specific *combination* of genes that no
independent-marginal model would ever propose.

`rlevo` implements UMDA, PBIL, cGA, MIMIC, and BOA. Their derivations and the
deceptive benchmark (Concatenated Trap) used to compare them are in [Appendix
A](../appendix-a-ec-algorithms/index.md).

**In `rlevo`.** Because the EDA loop is *fit → sample* rather than *select →
vary*, it does not fit the operator-pipeline shape of the GA, so we gave it its
own seam. The `ProbabilityModel` trait captures the model, separately from
`Strategy`; `EdaStrategy<B, M>` is a generic driver that implements `Strategy<B>`
for any `M: ProbabilityModel<B>`:

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

Note where the randomness comes from: all of it in `sample` flows from the host
`rng`. Implementations must never call `Tensor::random` or `B::seed`, because
Burn's GPU PRNG kernels share process-global state and would interleave across
strategy instances running in parallel — the kind of bug that only shows up under
load and never reproduces in a single-threaded test. The payoff of routing the
model through one trait is that swapping one EDA for another is a one-line type
change: `EdaStrategy<B, UnivariateGaussian>` → `EdaStrategy<B, BayesianNetwork>`.

<!-- see [Simon, 2013] 
- Estimation of Distribution Algorithms (p. 313)
- UDMA (p. 316)
- cGA (p. 318)
- PBIL (p. 320)
- MIMIC (p. 324)
-->

### Where CMA-ES sits: the ES ↔ EDA spectrum

If ES learns a Gaussian and an EDA *is* a learned distribution, you may already
suspect the two families are closer than their separate sections imply — and you
would be right. CMA-ES blurs the line. A full-covariance Gaussian EDA fits its
distribution by **maximum likelihood** on the selected elite — mean and
covariance are *overwritten* each generation from the current crop. Strip CMA-ES
of its evolution paths and force its learning rates to 1, and it collapses to
exactly that: a continuous EDA (EMNA). What keeps CMA-ES on the *ES* side of the
line is **memory** — it does not overwrite \\(\mathbf{C}\\) and \\(\sigma\\) but
blends each generation's estimate into accumulated evolution paths, a fading
momentum that records which directions have been paying off.

So picture a spectrum: memoryless density estimation (EDA) at one end,
path-driven, momentum-carrying adaptation (CMA-ES) at the other, with
**CMSA-ES** sitting between them — it learns covariance like CMA-ES but adapts
its step size by self-adaptation rather than path-tracking. `rlevo` keeps the
families architecturally distinct for precisely this reason: EDAs implement the
`fit → sample` `ProbabilityModel` trait, while CMA-ES and CMSA-ES are
self-contained strategies whose path machinery does not fit that mould
(ADR 0021). The cleanest abstraction tracks the real structure, not the surface
resemblance.

## Differential Evolution

GA, ES, and EDA all build a *model* of where to look next — a recombination, a
Gaussian, a fitted distribution. **Differential Evolution** (DE; Storn and Price,
1997) [[Storn and Price, 1997]](#bibliography) throws the model away and reads the
search geometry straight off the population itself. Its mutation adds a *scaled
difference* between randomly chosen members,
\\(\mathbf{v} = \mathbf{x}_{r_1} + F\,(\mathbf{x}_{r_2} - \mathbf{x}_{r_3})\\),
which means the perturbation is automatically calibrated to the population's
current spread: large while the population is dispersed and exploring, shrinking
to fine steps as it converges. There is no separate step size to tune — the
difference vectors *are* the step-size controller. That self-scaling, on two knobs
only (the scale factor \\(F\\) and the crossover rate \\(CR\\)), is why DE is
often competitive with the real-valued GA and classical ES on multi-modal
landscapes while asking far less of you up front.

**In `rlevo`.** `DifferentialEvolution` chooses its mutation/crossover scheme
through the `DeVariant` enum — `Rand1Bin` (the recommended default), `Rand1Exp`,
`Rand2Bin`, `Best1Bin`, and `CurrentToBest1Bin` — and all five share the same
**greedy per-slot replacement** in `tell`: each trial vector competes only
against the parent that produced it, and replaces it only on improvement. The
`Best1`/`CurrentToBest1` variants pull toward the incumbent best for faster
exploitation, at the cost of higher premature-convergence risk — the explore /
exploit dial from the last chapter, exposed as a one-line variant choice. The
mutation formulae, the binomial-versus-exponential crossover masks, and the
per-variant guidance are in [Appendix A: Differential
Evolution](../appendix-a-ec-algorithms/differential-evolution.md).

## Genetic Programming

Every family so far searches a *fixed-length* genome — a real, binary, or integer
vector of known width. **Genetic programming** (GP; Koza, 1992)
[[Koza, 1992]](#bibliography) raises the ambition: it evolves *programs*. The
phenotype is a computation graph or expression tree of variable shape, and the
canonical job is symbolic regression — recovering a closed-form expression that
explains a dataset, rather than a point in \\(\mathbb{R}^n\\). The hard part is
the **genotype–phenotype map**: variation has to act on a representation that
always decodes to a *valid* program, or every offspring needs a repair pass and
the search drowns in bookkeeping.

`rlevo` sidesteps repair by keeping both GP variants on a fixed-length *integer*
genome — the same on-device `Tensor<B, 2, Int>` storage as every other strategy —
and decoding it to a program:

- **Cartesian GP** (`gp_cgp`) decodes an integer genome to a directed acyclic
  computation graph on a `rows × cols` grid; each node names a function from the
  function set and the wires feeding it, and evolution rewires the graph and swaps
  operators.
- **Gene Expression Programming** (`gep`) decodes a linear, fixed-length
  chromosome to a variable-shape expression tree. A `head`/`tail` split sizes the
  chromosome so that *every* genome decodes to a complete tree — **repair-free
  validity** — while keeping GA-style array-edit operators, including real
  crossover.

Both decode strategies, their function sets, and worked symbolic-regression
examples are in [Appendix A: Cartesian GP](../appendix-a-ec-algorithms/cartesian-genetic-programming.md)
and [Gene Expression Programming](../appendix-a-ec-algorithms/gene-expression-programming.md).

## Neuroevolution

The families above optimise abstract vectors. **Neuroevolution** points the very
same machinery at the object this library ultimately cares about — a neural
network — and evolves it directly, with no gradients and no backpropagation. That
makes it the load-bearing half of `rlevo`'s thesis: the same `Strategy` loop that
minimised a benchmark function can train a policy. `rlevo` ships three approaches,
graded by how much structure you fix before the search begins: `WeightOnly` (fix
the architecture, evolve the flattened weights of any Burn `Module`), `ArchNas`
(co-evolve *which* architecture from a closed menu alongside its weights), and
`Neat` (grow the topology itself from a minimal seed). The dedicated
[Neuroevolution](40-neuroevolution.md) chapter is the full treatment; how
neuroevolution and gradient-based RL *combine* rather than compete is the subject
of [Why Combine Them?](50-why-combine.md).

## Other strategy families in `rlevo`

The same `Strategy` contract backs a wider menagerie — variations on the ideas
above, a meta-wrapper, and a set of swarm metaheuristics. Each ships today and is
documented with full pseudocode in [Appendix
A](../appendix-a-ec-algorithms/index.md); in brief:

- **Evolutionary Programming (EP)** is Fogel-style, mutation-only evolution with
  per-individual log-normal \\(\sigma\\) adaptation and q-tournament survivor
  selection over a \\((\mu + \mu)\\) pool — close kin to the self-adaptive ES
  variants above, minus recombination.
- **The memetic wrapper** wraps any real-valued strategy with per-individual
  local search (hill-climbing, Nelder–Mead, simulated annealing) under
  Lamarckian, Baldwinian, or partial-writeback policies — evolution for the
  global map, local search for the last mile.
- **Swarm metaheuristics** include PSO, ABC, ACO (continuous `ACO_R`; a
  permutation variant is stubbed), cuckoo search, and firefly. Four more — GWO,
  WOA, Bat, and SSA — ship as *legacy comparators*: included for baselining, but
  the module docs steer you to PSO or CMA-ES (or LSHADE once it lands) for real
  work, following [[Camacho-Villalón et al., 2023]](#bibliography) and
  [[Sörensen, 2015]](#bibliography). We keep them, but we are honest about them.

## Multi-Objective Optimisation

Every family so far has hunted for a single best *point*, varying only *how* it
searches while holding the objective fixed at one scalar (the engine maximises it;
a cost declares `ObjectiveSense::Minimize`). But many real problems refuse to
collapse to one number — you want higher reward *and* lower energy, and the two
fight. When objectives conflict there is no single optimum, only a *set* of
incomparable trade-offs: a solution **dominates** another when it is no worse on
every objective and strictly better on at least one. The **Pareto front** is the
set of non-dominated solutions — those where improving one objective necessarily
worsens another. The notion traces to the economist Vilfredo Pareto
[[Pareto, 1896]](#bibliography), and a *population* is a natural fit for it,
because it can approximate the *whole* front in one run rather than scalarising
the objectives into a weighted sum and re-solving for every weighting.

NSGA-II (Deb et al., 2002) [[Deb et al., 2002]](#bibliography) is the canonical
multi-objective EA — its fast non-dominated sorting and crowding-distance
diversity operator remain the baseline every newer algorithm is measured against;
SPEA2 [[Zitzler et al., 2001]](#bibliography) is the other classic reference
point. `rlevo` does not implement multi-objective optimisation yet; it is on the
research roadmap (see [Part
IV](../part-4-open-problems/02-research-directions.md)). The single-objective
`ObjectiveSense` you met earlier is deliberately the \\(K = 1\\) atom of that
future: a multi-objective problem carries one sense *per* objective, and dominance
canonicalises each to maximise space (negating every `Minimize` component) before
applying "no worse on all, strictly better on at least one" — the same
one-chokepoint reconciliation, scaled to a vector. Today's scalar contract is
exactly its \\(K = 1\\) restriction, so landing NSGA-II adds a path beside it
rather than reworking it.

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

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
