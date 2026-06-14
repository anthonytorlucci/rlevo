# What Is Optimization?

At its core, optimization is the task of finding the input to a function that
produces the best output. "Best" means lowest cost, highest reward, smallest
error, or whatever the problem says it means. Formally, we want

\\[
\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})
\\]

where \\(f : \mathcal{X} \to \mathbb{R}\\) is the **objective function** (also
called a **fitness function**, a **cost function**, or a **loss**) and
\\(\mathcal{X}\\) is the **search space**.

Everything else in optimization is a strategy for searching \\(\mathcal{X}\\)
efficiently.

## Fitness Landscapes

A useful mental model is the **fitness landscape** — imagine \\(f\\) as a
physical terrain where elevation represents cost: valleys are good solutions,
peaks are bad ones. A perfect optimizer would teleport to the lowest valley
without looking at the terrain. Every real algorithm is a compromise: it gathers
local information and uses it to decide where to look next.

The landscape metaphor was introduced in evolutionary biology by Sewall Wright
(1932) to describe how populations move through genotype space under selection.
Stuart Kauffman later formalised the idea of *ruggedness* — how many local minima
a landscape has — in the NK model [[Kauffman93]](#bibliography). Both concepts
transferred directly into evolutionary computation.

Three landscape properties matter most for algorithm choice:

| Property | What it means | Implication |
| -------- | ------------- | ----------- |
| **Unimodal** | One global minimum, no local minima | Gradient descent works well |
| **Multimodal** | Many local minima | Local search gets trapped; populations help |
| **Deceptive** | Gradient points away from the global optimum | EAs can still escape; gradient methods cannot |

The sphere function minimised in Part II is unimodal and convex — an easy
landscape chosen to show the mechanics. The Rastrigin and Rosenbrock functions
(available in `rlevo-environments::landscapes`) are multimodal and deceptive
respectively, and they are where evolutionary methods start to earn their keep.

## Exploitation vs Exploration

Every search algorithm must balance two competing pressures:

- **Exploitation** — concentrate effort near solutions that are already known to
  be good.
- **Exploration** — probe unknown regions that might be better.

A pure exploiter converges fast but gets trapped in local minima. A pure explorer
never converges at all. The optimal balance depends on the landscape, the budget
(number of evaluations), and the acceptable risk of missing the global optimum.

This tension appears under different names across the field: the
**exploration–exploitation dilemma** in reinforcement learning, the **diversity–
selection pressure** trade-off in evolutionary computation, and the
**bias–variance trade-off** in statistics. They are all the same underlying
problem.

## Why Gradient Descent Is Not Always the Answer

Gradient-based methods — stochastic gradient descent, Adam, L-BFGS — are
dominant in modern machine learning because neural network loss surfaces are high-
dimensional but empirically well-behaved for training. They are not the right
tool when:

1. **The objective is not differentiable.** Reward in a game, a simulation result,
   or a user preference score have no gradient.
2. **The landscape is highly multimodal.** Gradients point downhill locally but
   give no information about whether a better valley exists elsewhere.
3. **The search space is discrete or combinatorial.** Gene sequences, network
   architectures, and routing tables are not real-valued.
4. **Evaluation is noisy.** A single observation of \\(f(\mathbf{x})\\) is
   misleading; you need strategies that are robust to stochastic evaluations.

Evolutionary computation was designed for exactly these regimes.

## The No Free Lunch Theorem

Wolpert and Macready (1997) proved that **no search algorithm outperforms every
other algorithm averaged uniformly over all possible objective functions**
[[WM97]](#bibliography). If an algorithm does well on one class of problems, there
must be another class on which it does equivalently worse.

The practical implication: choosing an algorithm means making an assumption about
the structure of your problem. A genetic algorithm assumes that good solutions
share building blocks. CMA-ES assumes the landscape is locally well-approximated
by a multivariate Gaussian. DQN assumes the value function can be approximated
by a deep network. Part II makes these assumptions explicit for each algorithm it
uses.

> **Deeper reading.** For a rigorous treatment of optimization landscapes and
> algorithm analysis, see Nocedal and Wright, *Numerical Optimization* (2nd ed.,
> Springer, 2006) for the gradient side and Eiben and Smith,
> *Introduction to Evolutionary Computing* (Springer, 2015) for the population
> side. The NK model is developed in Kauffman, *The Origins of Order* (Oxford,
> 1993). The No Free Lunch theorem is in Wolpert and Macready (1997), IEEE
> Transactions on Evolutionary Computation.

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*
