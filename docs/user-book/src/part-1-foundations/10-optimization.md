# What Is Optimization?

At its core, optimization is the task of finding the input to a function that
produces the best output. "Best" means lowest cost, highest reward, smallest
error, or whatever the problem says it means. Formally, we want

```math
\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})
```

where \\(f : \mathcal{X} \to \mathbb{R}\\) is the **objective function** (also
called a **fitness function**, a **cost function**, or a **loss**) and
\\(\mathcal{X}\\) is the **search space**.

Everything else in optimization is a strategy for searching \\(\mathcal{X}\\)
efficiently.

### A Note on Terminology (or: Why Everyone Calls It Something Different)

The four terms above all refer to the same fundamental idea - a scalar-valued 
function you're trying to maximize or minimize - this distinction is more 
important than the terminology. *If you've ever opimized in the wrong 
direction, you know what I mean.*

- **Loss** / **cost function** is deep learning's dialect. You're minimizing 
  it, it's differentiable (or you're pretending it is), and your optimizer is 
  almost certainly some flavour of SGD. Mean squared error, cross-entropy, 
  Huber loss — all losses. The distinction between "loss" and "cost" is mostly 
  vibes: loss tends to refer to a per-sample quantity, cost to its aggregate 
  over a batch or dataset, though nobody enforces this at the border.

- **Objective function** is the neutral, field-agnostic term. Operations 
  research, convex optimization, and anyone who wants to sound rigorous without 
  committing to a community uses this one. It can be minimized *or* maximized, 
  which makes it the honest choice when the direction isn't obvious from context.

- **Fitness function** is evolutionary computing's term, and the direction 
  flips — you're *maximizing* it, because organisms with higher fitness 
  survive. If you hand an evolutionary algorithm a loss and forget to negate 
  it, you will breed a population of spectacularly wrong solutions and spend an 
  afternoon wondering why your GA is converging so confidently in the wrong 
  direction.

- **Reward** is reinforcement learning's contribution to the pile. Technically 
  the reward signal \\(r_t\\) and the objective (expected cumulative return 
  \\(\mathbb{E}[\sum_t \gamma^t r_t]\\)) are distinct things, but colloquially 
  people call the whole thing "the reward function." Like fitness, it's 
  maximized — RL agents are optimists by design.

The upshot: when you see \\(\arg\min\\) in the literature, someone is probably 
doing ML or OR. When you see \\(\arg\max\\), someone is doing RL or EC.

## Fitness Landscapes

A useful mental model is the **fitness landscape** — imagine \\(f\\) as a
physical terrain where elevation represents cost: valleys are good solutions,
peaks are bad ones. A perfect optimizer would teleport to the lowest valley
without looking at the terrain. Every real algorithm is a compromise: it gathers
local information and uses it to decide where to look next.

The landscape metaphor was introduced in evolutionary biology by Sewall Wright
(1932) to describe how populations move through genotype space under selection.
Stuart Kauffman later formalised the idea of *ruggedness* — how many local minima
a landscape has — in the NK model [[Kauffman, 1993]](#bibliography). Both concepts
transferred directly into evolutionary computation.

Three landscape properties matter most for algorithm choice:

| Property | What it means | Implication |
| -------- | ------------- | ----------- |
| **Unimodal** | One global minimum, no local minima | Gradient descent works well |
| **Multimodal** | Many local minima | Local search gets trapped; populations help |
| **Deceptive** | Gradient points away from the global optimum | EAs can still escape; gradient methods cannot |

The sphere function minimised in Part II is unimodal and convex — an easy
landscape chosen to show the mechanics. The Rastrigin and Rosenbrock functions
(available in `rlevo::envs::landscapes`) are multimodal and non-separable
respectively, and they are where evolutionary methods start to earn their keep.
For a more in-depth tour of landscape geometry and the benchmark suite `rlevo` ships,
see [Fitness Landscapes](../appendix-d-suppl/fitness-landscape.md) in the Appendices.

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

<!-- todo! additional context provided below; consider creating a supplementary article that dives deeper
[Simon, 2013, p.28-29]
*Exploration* is the search for new ideas or new strategies. *Exploitation* is 
the use of existing ideas and strategies that have proven successful in the 
past. Exploration is high-risk; a lot of new ideas waste time and lead to dead 
ends. However, exploration can also be high return; a lot of new ideas pay off
in ways that we could not have imagined. Exploitation is closely related to the 
feedback strategies ... The proper balance of exploration and exploitation 
depends on how regular our environment is. If our environment is rapidly 
changing then our knowledge quickly becomes obsolete and we cannot rely as much 
on exploitation. However, if our environment is highly consistent, then our 
knowledge is dependable and it may not make sense to try very many new ideas.

Our EA designs will need a proper balance of exploration and expoitation to be 
successful. Too much exploration is similar to too much randomness... and will 
probably not give good optimization results. But too much exploitation is 
related to too little randomness. The proper balance of exploration and exploitation in EAs was called "the optimal allocation of trials" by John Holland, one of the pioneers of genetic algorithms [Holland, 1975].

---
[Sutton & Barto, 2018, p.3]
One of the challenges that arise in reinforcement learning, and not in other kinds
of learning, is the trade-o↵ between exploration and exploitation. To obtain a lot of
reward, a reinforcement learning agent must prefer actions that it has tried in the past
and found to be e↵ective in producing reward. But to discover such actions, it has to
try actions that it has not selected before. The agent has to exploit what it has already
experienced in order to obtain reward, but it also has to explore in order to make better
action selections in the future. The dilemma is that neither exploration nor exploitation
can be pursued exclusively without failing at the task. The agent must try a variety of
actions and progressively favor those that appear to be best. On a stochastic task, each
action must be tried many times to gain a reliable estimate of its expected reward. The
exploration–exploitation dilemma has been intensively studied by mathematicians for
many decades, yet remains unresolved. For now, we simply note that the entire issue of
balancing exploration and exploitation does not even arise in supervised and unsupervised
learning, at least in the purest forms of these paradigms.
-->

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
[[Wolpert and Macready, 1997]](#bibliography). If an algorithm does well on one 
class of problems, there must be another class on which it does equivalently 
worse.

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
> Transactions on Evolutionary Computation. Simon (2013) further discusses the 
> No Free Lunch theorem and provides a few examples in Appendix B.1, page 614.


<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
