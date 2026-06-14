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

## A Brief History

John Holland formalised the first rigorous framework for genetic algorithms in
*Adaptation in Natural and Artificial Systems* (1975) [[Holland75]](#bibliography),
proving via the **Schema Theorem** that short, high-fitness patterns ("building
blocks") propagate exponentially under selection. Holland's GAs worked on binary
strings; the building-block hypothesis — that good solutions are composed of
reusable sub-structures — remains influential, though also contested.

Simultaneously and independently, Ingo Rechenberg and Hans-Paul Schwefel in
Germany developed **Evolution Strategies** (ES) for continuous optimisation of
engineering designs [[Rechenberg73]](#bibliography). ES works directly in
\\(\mathbb{R}^n\\), using Gaussian mutations with *self-adaptive* step sizes — the
algorithm learns not just the solution but also how aggressively to mutate it.

Lawrence Fogel introduced **Evolutionary Programming** (EP) around the same time,
focusing on evolving finite-state machines for prediction tasks.

John Koza's **Genetic Programming** (1992) [[Koza92]](#bibliography) extended GAs
to tree-structured programs, opening the door to automatic program synthesis.

Thomas Back's *Evolutionary Algorithms in Theory and Practice* (1996)
[[Back96]](#bibliography) unified these threads and remains the standard
reference for the theoretical foundations of the field.

## The Genetic Algorithm

The canonical GA, as implemented in `rlevo-evolution`, operates on real-valued
or binary genomes with three operators:

**Selection** — picks which individuals reproduce. Common schemes:
- *Tournament*: draw \\(k\\) candidates at random; the best wins. Selective
  pressure scales with \\(k\\).
- *Fitness-proportionate* (roulette wheel): probability of selection is
  proportional to fitness. Sensitive to fitness scaling.
- *Rank-based*: selection probability based on rank, not raw fitness; more
  robust to outliers.

**Crossover** — combines two parents into offspring:
- *Single-point*: split each parent at a random locus; swap tails.
- *BLX-α*: for real-valued genomes, sample offspring genes uniformly from
  \\([\min(p_1, p_2) - \alpha \cdot d,\ \max(p_1, p_2) + \alpha \cdot d]\\)
  where \\(d = |p_1 - p_2|\\). Larger \\(\alpha\\) → more exploration.
- *Simulated Binary Crossover (SBX)*: mimics single-point on binary
  representations but works in continuous space [[Deb95]](#bibliography).

**Mutation** — perturbs an individual:
- *Gaussian*: add \\(\mathcal{N}(0, \sigma^2)\\) noise to each gene.
- *Uniform*: replace a gene with a random draw from its bounds.

> **In the appendix.** [Appendix A](../appendix-a-ec-algorithms/index.md)
> gives the full pseudocode for the GA as implemented in `rlevo`, including
> elitism, boundary clamping, and the seeded mutation path.

## Evolution Strategies and CMA-ES

While GAs were designed with binary strings in mind, Evolution Strategies were
designed from the outset for continuous domains. The key idea is that the
mutation distribution should adapt to the landscape.

**CMA-ES** (Covariance Matrix Adaptation Evolution Strategy), introduced by
Hansen and Ostermeier (2001) [[HO01]](#bibliography), maintains a full
covariance matrix \\(\mathbf{C}\\) over the search space and updates it based on
the direction of successful steps. Each generation samples a population from
\\(\mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})\\), selects the top \\(\mu\\)
individuals, and updates \\(\mathbf{m}\\), \\(\sigma\\), and \\(\mathbf{C}\\):

\\[
\mathbf{m}^{(g+1)} = \sum_{i=1}^{\mu} w_i \mathbf{x}_{i:\lambda}^{(g)}
\\]

where \\(\mathbf{x}_{i:\lambda}\\) denotes the \\(i\\)-th ranked individual
out of \\(\lambda\\) samples, and \\(w_i\\) are recombination weights. CMA-ES
is the de facto standard for continuous black-box optimisation and is considered
the most powerful general-purpose single-objective EA for moderate dimensions
(\\(n \lesssim 10^3\\)).

## Estimation of Distribution Algorithms

Instead of maintaining individuals and applying variation operators, **Estimation
of Distribution Algorithms** (EDAs) maintain an explicit probabilistic model of
the search space and sample new candidates from it [[Larranaga02]](#bibliography).

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

## Multi-Objective Optimisation

Most real problems have more than one objective — faster *and* cheaper, higher
reward *and* lower energy. The **Pareto front** is the set of solutions where
improving one objective necessarily worsens another.

NSGA-II (Deb et al., 2002) [[Deb02]](#bibliography) is the canonical
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
