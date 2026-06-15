# Evolutionary Computation

Evolutionary computation (EC) is a family of population-based optimization
methods inspired by Darwinian natural selection. The common skeleton is:

1. Maintain a **population** of candidate solutions.
2. **Evaluate** each candidate with the objective function.
3. **Select** candidates that performed better.
4. **Vary** them (mutation, recombination) to produce the next population.
5. Repeat until a budget is exhausted or a solution is good enough.

The key insight is that a *population* of candidates shares information across
the search space simultaneously, while variation operators inject diversity that
prevents premature convergence. Selection then amplifies what works.

> **In `rlevo`.** Every algorithm in `rlevo::evo` implements the
> `Strategy<B>` trait, which maps the five-step skeleton above to three
> methods:
>
> ```rust
> pub trait Strategy<B: Backend>: Send + Sync {
>     type Params: Clone + Debug + Send + Sync;  // static run config
>     type State:  Clone + Debug + Send;          // generation-to-generation state
>     type Genome: Clone + Send;                  // genome container produced by ask
>
>     fn init(&self, params: &Self::Params, rng: &mut dyn Rng, device: &B::Device) -> Self::State;
>     fn ask (&self, params: &Self::Params, state: &Self::State, rng: &mut dyn Rng, device: &B::Device) -> (Self::Genome, Self::State);
>     fn tell(&self, params: &Self::Params, population: Self::Genome, fitness: Tensor<B, 1>, state: Self::State, rng: &mut dyn Rng) -> (Self::State, StrategyMetrics);
>     fn best(&self, state: &Self::State) -> Option<(Self::Genome, f32)>;
> }
> ```
>
> `ask` proposes the next population (steps 3–4); `tell` consumes it together
> with its fitness tensor and produces the next state (steps 2–3). The RNG is
> passed explicitly so the harness owns all stochasticity — strategies carry no
> internal PRNG state.
>
> Genomes are stored on-device in a `Population<B, K>` wrapper, where the
> const type parameter `K` is a zero-sized *genome kind* marker:
>
> ```rust
> pub struct Real;    // genes are f32 — used by GA, ES, DE, CMA-ES
> pub struct Binary;  // genes are 0/1 i32 — used by binary GA and EDAs
> pub struct Integer; // genes are bounded non-negative integers
> ```
>
> The separation between the genome kind and the tensor type means operators
> can specialize at compile time (e.g. Gaussian mutation only compiles for
> `Real`; bit-flip mutation only compiles for `Binary`) without runtime dispatch.
>
> Fitness evaluation is injected through `BatchFitnessFn<B, G>`, which receives
> the population and returns a `Tensor<B, 1>` of shape `(pop_size,)`. Strategies
> themselves never call the objective function — the harness does — so the same
> strategy implementation works against any landscape.

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

> **In `rlevo`.** All three operator families live in `rlevo::evo::ops`:
> - `ops::selection` — tournament, rank-based, fitness-proportionate selectors
> - `ops::crossover` — single-point, BLX-α, SBX (real); single-point (binary)
> - `ops::mutation`  — Gaussian, uniform (real); bit-flip (binary)
>
> Each operator function takes a `Population<B, K>` and returns a new one,
> leaving the input unchanged. Operator kind-specialization is enforced at the
> type level: passing a `Population<B, Binary>` to a Gaussian mutation function
> is a compile error.
>
> [Appendix A](../appendix-a-ec-algorithms/index.md) gives the full GA
> pseudocode as implemented, including elitism and boundary clamping.

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

## Evolution Strategies and CMA-ES

While GAs were designed with binary strings in mind, Evolution Strategies were
designed from the outset for continuous domains. The key idea is that the
mutation distribution should adapt to the landscape.

**CMA-ES** (Covariance Matrix Adaptation Evolution Strategy), introduced by
Hansen and Ostermeier (2001) [[Hansen and Ostermeier, 2001]](#bibliography), maintains a full
covariance matrix \\(\mathbf{C}\\) over the search space and updates it based on
the direction of successful steps. Each generation samples a population from
\\(\mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})\\), selects the top \\(\mu\\)
individuals, and updates \\(\mathbf{m}\\), \\(\sigma\\), and \\(\mathbf{C}\\):

```math
\mathbf{m}^{(g+1)} = \sum_{i=1}^{\mu} w_i \mathbf{x}_{i:\lambda}^{(g)}
```

where \\(\mathbf{x}_{i:\lambda}\\) denotes the \\(i\\)-th ranked individual
out of \\(\lambda\\) samples, and \\(w_i\\) are recombination weights. CMA-ES
is the de facto standard for continuous black-box optimisation and is considered
the most powerful general-purpose single-objective EA for moderate dimensions
(\\(n \lesssim 10^3\\)).

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

> **In `rlevo`.** The fit → sample loop is captured by the `ProbabilityModel`
> trait, which is separate from `Strategy`. `EdaStrategy<B, M>` is a generic
> driver that implements `Strategy<B>` for any `M: ProbabilityModel<B>`:
>
> ```rust
> pub trait ProbabilityModel<B: Backend> {
>     type Params;
>     type State: Clone + Debug + Send + Sync;
>
>     /// Fit the model to the selected (top-μ) population.
>     /// `prev = None` on the first generation; the model builds its prior from `params`.
>     fn fit(&self, params: &Self::Params, population: &Tensor<B, 2>,
>            fitness: Tensor<B, 1>, prev: Option<&Self::State>) -> Self::State;
>
>     /// Sample n new candidates from the fitted model using the host RNG.
>     fn sample(&self, params: &Self::Params, state: &Self::State,
>               n: usize, rng: &mut dyn Rng, device: &B::Device) -> Tensor<B, 2>;
> }
> ```
>
> All randomness in `sample` comes from the host `rng`; implementations must
> never call `Tensor::random` or `B::seed` (Burn's GPU PRNG kernels share
> process-global state and would interleave across parallel strategy instances).
> Swapping one EDA for another is a one-line type change: `EdaStrategy<B, UnivariateGaussian>`
> → `EdaStrategy<B, BayesianNetwork>`.

<!-- see [Simon, 2013] 
- Estimation of Distribution Algorithms (p. 313)
- UDMA (p. 316)
- cGA (p. 318)
- PBIL (p. 320)
- MIMIC (p. 324)
-->

## Multi-Objective Optimisation

Most real problems have more than one objective — faster *and* cheaper, higher
reward *and* lower energy. The **Pareto front** is the set of solutions where
improving one objective necessarily worsens another.

NSGA-II (Deb et al., 2002) [[Deb et al., 2002]](#bibliography) is the canonical
multi-objective EA and remains the baseline every new algorithm is compared
against. `rlevo` does not yet implement multi-objective optimisation; it is on
the research roadmap (see [Part III](../part-3-open-problems/02-research-directions.md)).

> **Deeper reading.** Eiben and Smith, *Introduction to Evolutionary Computing*
> (Springer, 2015) is the most accessible modern textbook. Back, *Evolutionary
> Algorithms in Theory and Practice* (Oxford, 1996) is more rigorous. The CMA-ES
> tutorial by Hansen (2023, arXiv:1604.00772) is freely available and
> authoritative. For EDAs, see Larrañaga and Lozano (eds.), *Estimation of
> Distribution Algorithms* (Kluwer, 2002).

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*

---

*Co-Authored-By: Anthropic Claude Opus 4.8*
*Reviewed-By: (Human) Anthony Torlucci*
